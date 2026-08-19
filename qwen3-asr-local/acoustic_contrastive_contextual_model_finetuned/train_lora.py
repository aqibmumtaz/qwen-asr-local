#!/usr/bin/env python3
"""
Step 3 -- LoRA fine-tune Qwen3-ASR-1.7B's decoder attention layers.

*** NOT RUN OR VALIDATED ON A GPU -- no forward/backward pass has actually
been executed. What IS grounded: the qwen_asr package (installed locally,
version site-packages/qwen_asr) was directly inspected this session --
Qwen3ASRModel is a thin wrapper holding `self.model` (a real HF
PreTrainedModel, registered via AutoModel.register(Qwen3ASRConfig,
Qwen3ASRForConditionalGeneration)) and `self.processor` (AutoProcessor,
multimodal text+audio). The message format, chat-template prompt
construction, and processor call signature below are copied directly from
qwen_asr/inference/qwen3_asr.py's own inference path
(_build_messages/_build_text_prompt/_infer_asr_transformers), not guessed.
What's still unverified: that this exact prompt+labels construction
produces a working loss/backward pass -- only actually running it on the
GPU machine confirms that. Loading the full model was attempted locally
this session and failed with an MPS buffer-size error (RuntimeError:
Invalid buffer size: 6.43 GB) -- expect to debug device/dtype issues on
first real GPU run regardless of the logic below being correct. ***

Three sequential phases, each phase's adapter is the starting checkpoint
for the next:

  python train_lora.py --phase 1 --data data/train_augmented.jsonl --run-name run1
  python train_lora.py --phase 2 --data data/train_augmented.jsonl --resume-from adapters/run1/phase1 --run-name run1
  python train_lora.py --phase 3 --data data/adversarial.jsonl     --resume-from adapters/run1/phase2 --run-name run1

Phase semantics (see plan for full rationale):
  1  domain vocabulary  -- positive+negative examples, teach correct Hindi
     spellings for names the model has actually heard
  2  context-conditioned -- same data, vary context= between correct-name
     and empty across passes so the model learns to actually USE it
  3  adversarial robustness -- adversarial.jsonl, decoy context must NOT
     appear in output; lower LR (this phase is corrective, not additive)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"   # same constant as acoustic_contextual_biasing/asr.py

LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]  # decoder attention ONLY
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

HYPERPARAMS = {
    1: {"lr": 1e-4, "epochs": 15},
    2: {"lr": 1e-4, "epochs": 10},
    3: {"lr": 5e-5, "epochs": 10},   # lower LR -- corrective phase, not additive
}
BATCH_SIZE = 4
GRAD_ACCUM = 8
WARMUP_FRACTION = 0.10
MAX_AUDIO_SECONDS = 25   # matches chunk size, no truncation needed


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def build_model(qlora: bool = False):
    """Load base Qwen3-ASR-1.7B via the qwen_asr package (same as LocalASR
    elsewhere in this repo). Returns (wrapper, device) -- wrapper.model is
    the actual HF PreTrainedModel that LoRA attaches to; wrapper.processor
    builds the multimodal text+audio inputs. Keep the wrapper around, not
    just wrapper.model -- .processor is needed for every training step."""
    import torch
    from qwen_asr import Qwen3ASRModel

    kwargs = {}
    if qlora:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    wrapper = Qwen3ASRModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=device, **kwargs)
    return wrapper, device


def attach_fresh_lora(wrapper):
    """LoRA attaches to wrapper.model (the real PreTrainedModel), NOT the
    Qwen3ASRModel wrapper itself -- the wrapper has no forward()/generate()
    of its own, those live on .model (confirmed by inspecting
    qwen_asr/inference/qwen3_asr.py: self.model.generate(**inputs, ...))."""
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                         target_modules=LORA_TARGET_MODULES,
                         lora_dropout=LORA_DROPOUT, task_type="CAUSAL_LM")
    wrapper.model = get_peft_model(wrapper.model, config)
    return wrapper


def resume_lora(wrapper, adapter_path: str):
    from peft import PeftModel
    wrapper.model = PeftModel.from_pretrained(wrapper.model, adapter_path, is_trainable=True)
    return wrapper


def build_prompt(wrapper, context: str) -> str:
    """Copied from qwen_asr's own _build_messages/_build_text_prompt
    (qwen3_asr.py lines ~448-460) -- the exact chat-template + forced-
    language scaffold used at inference, reused here so training sees the
    identical prompt format the model will actually be served with."""
    msgs = [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    base = wrapper.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return base + "language Hindi<asr_text>"   # force_language="Hindi", matches
                                                # every ASR call elsewhere in this repo


class ASRTripletDataset:
    """One example = one (audio, context, target_hindi) triplet. __getitem__
    returns raw fields; tokenization + label-masking happens in the collator
    (needs the processor, which is loaded once, not per-example)."""

    def __init__(self, examples: list[dict], audio_root: Path):
        self.examples = examples
        self.audio_root = audio_root

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        return {"audio_path": self.audio_root / ex["audio_path"] if not
                Path(ex["audio_path"]).is_absolute() else Path(ex["audio_path"]),
                "context": ex["context"], "target": ex["target_hindi"]}


def make_collate_fn(wrapper):
    """Builds (prompt_only, prompt+target) pairs per example, batches them
    through wrapper.processor with audio, then masks every label position
    that falls within the prompt (loss computed on target tokens only --
    standard causal-LM fine-tuning pattern). Prompt length per example is
    found by tokenizing prompt-alone and prompt+target separately and
    diffing input_id counts, since padding is applied per-batch."""
    import torch
    import soundfile as sf

    def collate(batch):
        prompts = [build_prompt(wrapper, ex["context"]) for ex in batch]
        full_texts = [p + ex["target"] + wrapper.processor.tokenizer.eos_token
                      for p, ex in zip(prompts, batch)]
        wavs = []
        for ex in batch:
            audio, sr = sf.read(str(ex["audio_path"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            wavs.append(audio)

        inputs = wrapper.processor(text=full_texts, audio=wavs, return_tensors="pt", padding=True)
        labels = inputs["input_ids"].clone()

        # mask the prompt portion of each row -- tokenize prompt-alone
        # (text only, no audio placeholder resolved yet at this level is
        # fine since we only need the TOKEN COUNT of the prompt segment)
        for i, p in enumerate(prompts):
            prompt_ids = wrapper.processor.tokenizer(p, return_tensors="pt")["input_ids"]
            prompt_len = prompt_ids.shape[1]
            labels[i, :prompt_len] = -100
        labels[inputs["attention_mask"] == 0] = -100   # mask padding too

        inputs["labels"] = labels
        return inputs

    return collate


def build_training_batch(examples: list[dict], phase: int):
    """Phase-specific input construction.
    Phase 1: (audio, context=found_names, target=target_hindi) as-is.
    Phase 2: EACH example seen twice per epoch -- once with its real
      context, once with context="" and the SAME target -- so the model
      sees both conditions on identical audio and learns context is a
      hint, not a requirement.
    Phase 3: adversarial.jsonl already has decoy context + unchanged
      target baked in by prepare_data.py -- used as-is.
    """
    if phase == 2:
        doubled = []
        for ex in examples:
            doubled.append(ex)
            doubled.append({**ex, "context": ""})
        return doubled
    return examples


def train_one_phase(args):
    from transformers import Trainer, TrainingArguments

    examples = load_jsonl(Path(args.data))
    examples = build_training_batch(examples, args.phase)
    print(f"[phase {args.phase}] {len(examples)} training examples (after phase-specific expansion)", flush=True)

    wrapper, device = build_model(qlora=args.qlora)
    if args.resume_from:
        print(f"[phase {args.phase}] resuming adapter from {args.resume_from}", flush=True)
        wrapper = resume_lora(wrapper, args.resume_from)
    else:
        print(f"[phase {args.phase}] attaching fresh LoRA adapter", flush=True)
        wrapper = attach_fresh_lora(wrapper)
    wrapper.model.print_trainable_parameters()

    hp = HYPERPARAMS[args.phase]
    out_dir = HERE / "adapters" / args.run_name / f"phase{args.phase}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ASRTripletDataset(examples, ROOT)
    collate_fn = make_collate_fn(wrapper)

    training_args = TrainingArguments(
        output_dir=str(out_dir), learning_rate=hp["lr"], num_train_epochs=hp["epochs"],
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_FRACTION, bf16=True, save_strategy="epoch",
        logging_steps=10, report_to=[], remove_unused_columns=False)

    trainer = Trainer(model=wrapper.model, args=training_args,
                       train_dataset=dataset, data_collator=collate_fn)
    trainer.train()
    wrapper.model.save_pretrained(str(out_dir))   # saves ONLY the adapter
                                                    # weights (peft default),
                                                    # base model untouched
    print(f"[phase {args.phase}] adapter saved to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--data", required=True)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--qlora", action="store_true", help="4-bit quantized base model, ~12GB VRAM")
    args = ap.parse_args()
    train_one_phase(args)


if __name__ == "__main__":
    main()
