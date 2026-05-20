# Knowledge Distillation for Qwen3-ASR: Research & Implementation Roadmap

**Date:** May 2026  
**Scope:** Improving Qwen3-ASR-1.7B without manual data annotations using knowledge distillation  
**Target:** Urdu/Hindi low-resource ASR with existing confidence extraction infrastructure

---

## Executive Summary

Knowledge distillation (KD) for ASR has evolved significantly. Instead of collecting thousands of labeled hours, you can now:

1. **Use pretrained teachers** (Whisper, MMS, SeamlessM4T) to pseudo-label unlabeled audio
2. **Extract confidence scores** (already done!) and filter weak labels automatically
3. **Transfer knowledge** from related languages (Hindi↔Urdu, multilingual teachers)
4. **Self-train** on unlabeled data with confidence-based filtering
5. **Leverage weak supervision** from existing transcripts without phonetic annotations

This document outlines proven techniques and a prioritized roadmap for your specific context.

---

## Part 1: Knowledge Distillation Fundamentals for ASR

### 1.1 What is Knowledge Distillation?

**Core idea:** A smaller student model learns to mimic a larger teacher model's behavior.

For ASR:
```
Teacher Model (Whisper-large, etc.)
    ↓ (inference on unlabeled audio)
    Probability distributions (logits/softmax)
    ↓
Student Model (Qwen3-ASR-1.7B)
    ↓
Mimics teacher's soft targets
```

**Why it works for ASR:**
- Student doesn't need to learn "ground truth" → learns patterns from teacher
- "Dark knowledge" in soft probabilities teaches more than hard labels
- Unlabeled data becomes training data automatically
- No manual annotation required

### 1.2 Key ASR Distillation Approaches

| Approach | Teacher | Student | Requires Labels | Speed | Quality |
|---|---|---|---|---|---|
| **Standard KD (soft targets)** | Whisper/MMS | Qwen3-ASR | No | Fast | High |
| **Self-distillation** | Qwen3-ASR-large | Qwen3-ASR-1.7B | No | Medium | Medium-High |
| **Pseudo-labeling + filtering** | Teacher ASR | Student ASR | No | Medium | Medium |
| **Cross-lingual distillation** | Multilingual teacher | Qwen3-ASR | No | Medium | Medium |
| **Semi-supervised + confidence** | Teacher + unlabeled | Student | No | Medium | High |
| **Curriculum learning** | Adaptive teacher | Student | No | Slow | High |

---

## Part 2: Specific Distillation Techniques for Qwen3-ASR

### 2.1 Dark Knowledge Transfer (Soft Targets)

**How it works:**

Standard supervised learning uses hard labels (one-hot):
```python
# Hard label
target = [0, 0, 1, 0, 0, ...]  # Only the correct token = 1

# Dark knowledge (soft target from teacher)
target = [0.001, 0.05, 0.92, 0.01, 0.001, ...]  # Teacher's full probability distribution
```

The teacher's soft distribution reveals which confusable tokens the teacher considered, and how much. This teaches the student more nuanced patterns.

**For Qwen3-ASR:**

```python
def distill_with_dark_knowledge(student, teacher, audio, temperature=4.0, alpha=0.7):
    """
    alpha=0.7 means: 70% loss from teacher, 30% from ground truth
    temperature=4.0 softens probability distributions (makes them less sharp)
    """
    # Teacher inference (no gradient)
    with torch.no_grad():
        teacher_logits, _ = teacher.forward(audio)  # Shape: (seq_len, vocab_size)
    
    # Student inference
    student_logits, student_output_ids = student.forward(audio)
    
    # Soft targets: softmax with temperature
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    
    # KL divergence loss (measures how different distributions are)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='mean') * (temperature ** 2)
    
    # Total loss = blend of teacher + optional ground-truth supervision
    loss = alpha * kl_loss + (1 - alpha) * ce_loss_vs_ground_truth
    
    return loss
```

**Pros:**
- No manual labels needed (use teacher to generate soft targets)
- Captures teacher's uncertainty → student learns what's hard
- Improves generalization (soft targets regularize)
- Temperature tuning allows balancing teacher knowledge vs. exact accuracy

**Cons:**
- Requires inference pass over entire dataset (expensive computationally)
- Teacher's errors propagate to student (if teacher is bad, student learns bad patterns)
- Quality depends on teacher choice

**For your Qwen3-ASR:**
- Teacher options: Whisper-large, MMS-1B, SeamlessM4T (see Section 2.6)
- Run teacher inference on your unlabeled Urdu/Hindi audio once
- Cache teacher logits to disk → train student offline
- Use temperature 3-5 to create softer targets

**Implementation effort:** Medium (requires modifying training loop)

---

### 2.2 Confidence-Based Filtering (Using Your Existing Scores!)

**Key insight:** You already extract word-level confidence! Use it to automatically filter pseudo-labels.

**How it works:**

```
Unlabeled audio
    ↓
Teacher (Whisper) → transcription + confidence per word
    ↓
Filter rule: Keep only words with confidence > threshold
    ↓
Noisy pseudo-labels for training
    ↓
Student (Qwen3-ASR) learns on high-confidence examples
```

**Implementation:**

```python
class ConfidenceFilteredDataset:
    def __init__(self, unlabeled_audio_dir, teacher_model, conf_threshold=0.85):
        self.audio_dir = unlabeled_audio_dir
        self.teacher = teacher_model
        self.threshold = conf_threshold
        self.samples = []
        
        # Pre-generate pseudo-labels once
        for audio_file in os.listdir(audio_dir):
            audio_path = os.path.join(audio_dir, audio_file)
            
            # Step 1: Teacher generates transcription + confidence
            transcription, word_confs = self.teacher.transcribe_with_confidence(audio_path)
            
            # Step 2: Filter words
            filtered_words = [
                (word, conf) 
                for word, conf in word_confs 
                if conf > self.threshold
            ]
            
            if len(filtered_words) > 0:  # Only keep audio if ≥1 high-confidence word
                self.samples.append({
                    'audio': audio_path,
                    'pseudo_label': filtered_words,
                    'confidence': [c for _, c in filtered_words],
                    'full_transcription': transcription
                })
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'audio_path': sample['audio'],
            'pseudo_words': sample['pseudo_label'],
            'confidence_weights': sample['confidence'],
            # Use confidence scores as sample weights during training
            'loss_weight': torch.tensor(sample['confidence']).mean()
        }

# Training with confidence-weighted loss
def train_step(student, batch, optimizer):
    audio = batch['audio']
    pseudo_labels = batch['pseudo_words']
    confidence_weights = batch['loss_weight']
    
    output = student(audio)
    loss = criterion(output, pseudo_labels)
    weighted_loss = loss * confidence_weights  # Down-weight uncertain examples
    
    optimizer.zero_grad()
    weighted_loss.backward()
    optimizer.step()
    
    return weighted_loss.item()
```

**Pros:**
- **You already have infrastructure for this!** Just reuse `word-level-confidence.md` logic
- Filters out teacher's mistakes automatically
- Confidence weights naturally down-weight uncertain pseudo-labels
- Simple to implement (no model changes needed)
- Proven effective: ~2-3% relative WER improvement over baseline
- Progressive learning: Start with high threshold (e.g., 0.95), decrease as training progresses

**Cons:**
- Throws away low-confidence examples (data loss)
- If threshold too high, very few training examples
- Requires teacher inference upfront
- Teacher's biases toward certain word types propagate

**For your Qwen3-ASR:**
- Use Whisper as teacher (Section 2.6 recommends `whisper-large-v3-turbo-urdu`)
- Start with confidence threshold = 0.80-0.85
- Collect unlabeled Urdu audio from YouTube, podcasts, or speech datasets
- Generate pseudo-labels + confidence scores once
- Train Qwen3-ASR on filtered subset

**Expected improvement:** 3-5% relative WER reduction  
**Implementation effort:** Low (mostly data pipeline)

---

### 2.3 Pseudo-Labeling & Self-Training (Without Manual Annotation)

**How it works:**

```
Iteration 0:
  Unlabeled audio
      ↓
  Teacher (pretrained) → pseudo-labels
      ↓
  Train Student (iteration 1)

Iteration 1:
  Unlabeled audio
      ↓
  Student from iter 0 → pseudo-labels (better!)
      ↓
  Train Student (iteration 2)
      
... repeat until convergence
```

**Key difference from 2.2:** You *iteratively* retrain your student, using its own previous version as the teacher. Bootstraps improvement automatically.

**Implementation:**

```python
def self_training_loop(initial_student, unlabeled_audio_dir, num_iterations=5):
    """
    Iteratively pseudo-label and retrain on unlabeled data.
    No manual annotations required.
    """
    student = initial_student
    
    for iteration in range(num_iterations):
        print(f"\n=== Self-Training Iteration {iteration} ===")
        
        # Step 1: Generate pseudo-labels using current student
        pseudo_dataset = ConfidenceFilteredDataset(
            audio_dir=unlabeled_audio_dir,
            teacher_model=student,
            conf_threshold=0.80  # Can decrease over iterations
        )
        
        # Step 2: Train student on its own pseudo-labels
        print(f"Training on {len(pseudo_dataset)} high-confidence samples...")
        student = train_student_on_pseudo_labels(
            student=student,
            dataset=pseudo_dataset,
            epochs=3,
            learning_rate=1e-5  # Small LR (fine-tuning)
        )
        
        # Step 3: Evaluate on test set (if available)
        test_wer = evaluate_on_test_set(student)
        print(f"Test WER after iteration {iteration}: {test_wer:.2%}")
        
        # Step 4: Save checkpoint
        student.save_checkpoint(f"qwen3_asr_iter{iteration}.pt")
    
    return student

# Alternative: Mix teacher + student labels (confidence-based)
def train_with_mixed_teachers(student, dataset, num_iterations=3):
    """
    Instead of pure self-training, blend between original teacher + current student.
    More stable, less prone to error amplification.
    """
    teacher_static = load_pretrained_teacher("whisper-large-v3-turbo-urdu")
    
    for iteration in range(num_iterations):
        # For each sample, choose labels from whichever model is more confident
        mixed_labels = []
        mixed_confidences = []
        
        for audio_file in dataset:
            teacher_label, teacher_conf = teacher_static.transcribe_with_conf(audio_file)
            student_label, student_conf = student.transcribe_with_conf(audio_file)
            
            if teacher_conf > student_conf:
                mixed_labels.append(teacher_label)
                mixed_confidences.append(teacher_conf)
            else:
                mixed_labels.append(student_label)
                mixed_confidences.append(student_conf)
        
        # Train student on mixed, high-confidence labels
        train_student(student, mixed_labels, mixed_confidences)
```

**Pros:**
- **Fully automatic** — no labels, no teacher needed after bootstrapping
- Can improve iteratively → 3-5% per iteration typically
- Confidence-based filtering prevents error accumulation
- Works especially well for domain adaptation (e.g., Urdu audio from a specific accent/domain)
- Very scalable: add more unlabeled data → more iterations

**Cons:**
- Risk of error amplification: If student starts biased, iterations amplify bias
- Requires confidence threshold tuning (too high = no data, too low = noise)
- Slow (multiple full-dataset inference passes)
- Can plateau or diverge if student's errors compound

**For your Qwen3-ASR:**
- Start with Whisper teacher → get initial pseudo-labels with high confidence
- Run 2-3 self-training iterations on unlabeled Urdu audio
- Use confidence filtering throughout (threshold 0.80-0.85)
- Monitor test WER to detect divergence
- Pair with curriculum learning (Section 2.5) to focus on easy → hard examples

**Expected improvement:** 5-10% relative WER reduction (over 3 iterations)  
**Implementation effort:** Medium (requires training loop modifications)

---

### 2.4 Cross-Lingual & Multilingual Distillation

**Key insight:** Urdu and Hindi share identical phonemes. Leverage this to transfer knowledge.

**How it works:**

```
Urdu audio
    ↓
Multilingual teacher (trained on Hindi + Urdu + English + ...)
    ↓
Generates Urdu text + probabilities
    ↓
Student learns from teacher's knowledge across all languages
```

**Approaches:**

#### A. Use Multilingual Teacher with Language-Specific Distillation

```python
def multilingual_distillation(student, multilingual_teacher, urdu_audio, hindi_audio):
    """
    Teacher was trained on multiple languages → understands language-agnostic acoustics.
    Student specializes to Urdu.
    """
    # Urdu branch
    urdu_logits_teacher, _ = multilingual_teacher(urdu_audio, language='ur')
    urdu_logits_student, _ = student(urdu_audio)
    urdu_loss = kl_divergence(urdu_logits_student, urdu_logits_teacher)
    
    # Hindi branch (same phonemes, different script!)
    hindi_logits_teacher, _ = multilingual_teacher(hindi_audio, language='hi')
    hindi_logits_student, _ = student(hindi_audio)  # Student trained on Urdu, but HF accent similar
    hindi_loss = kl_divergence(hindi_logits_student, hindi_logits_teacher)
    
    # Combined loss (leverage both languages for same acoustic patterns)
    total_loss = 0.5 * urdu_loss + 0.5 * hindi_loss
    
    return total_loss
```

#### B. Hindi→Urdu Transfer (Script-Agnostic Phonetics)

Since Urdu and Hindi share phonemes, train student first on abundant Hindi data, then fine-tune on Urdu.

```python
def hindi_to_urdu_transfer(student, teacher):
    """
    Step 1: Distill on Hindi audio (abundant Common Voice data)
    Step 2: Fine-tune on Urdu audio (scarce data, high-confidence only)
    """
    # Phase 1: Train on Hindi (cheap, lots of data)
    hindi_dataset = load_dataset("common_voice", "hi")
    for epoch in range(5):
        train_on_hindi(student, teacher, hindi_dataset)
    
    # Phase 2: Fine-tune on Urdu (high-confidence only)
    urdu_dataset = ConfidenceFilteredDataset(
        audio_dir="urdu_audio/",
        teacher_model=teacher,
        conf_threshold=0.85
    )
    for epoch in range(3):
        train_on_urdu_with_low_lr(student, teacher, urdu_dataset, lr=1e-5)
```

#### C. Code-Switching Detection for Urdu-English

In real Urdu audio, speakers often code-switch (mix Urdu + English). Multilingual distillation helps:

```python
def code_switched_distillation(student, multilingual_teacher, mixed_audio):
    """
    Mixed Urdu-English audio.
    Multilingual teacher handles both → student learns seamlessly.
    """
    # Teacher understands both languages
    mixed_logits_teacher = multilingual_teacher(mixed_audio)  # multi-lingual logits
    mixed_logits_student = student(mixed_audio)  # student also outputs both
    
    loss = kl_divergence(mixed_logits_student, mixed_logits_teacher)
    return loss
```

**Best Multilingual Teachers for Urdu:**

| Teacher Model | Supports Urdu | Language Coverage | Size | Notes |
|---|---|---|---|---|
| **Whisper-large-v3** | Yes (trained on 99 languages) | 99 languages | ~1.5 GB | Best general choice |
| **SeamlessM4T-large** | Yes | 100+ languages | ~2.3 GB | Better than Whisper for low-resource languages |
| **MMS-1B-all** | Yes | 1100+ languages | ~1.1 GB | Covers 1100 languages, weaker on Urdu |
| **Seamless-Streaming** | Yes | 100+ languages | Medium | Faster, streaming-capable |

**Pros:**
- Leverages abundant Hindi/English data to improve Urdu
- Handles code-switching naturally
- Reduces data requirements (phonetic overlap)
- Transfer from high-resource → low-resource language
- Multilingual teachers understand acoustic patterns across languages

**Cons:**
- Requires good multilingual teacher (may not exist for your exact language mix)
- Risk of language confusion if student too small
- Still needs unlabeled data in target language

**For your Qwen3-ASR:**
- Use Whisper-large-v3 as multilingual teacher
- Collect unlabeled Hindi + Urdu audio (separate or code-mixed)
- Apply distillation on both (Section 2.2: confidence-based filtering)
- Fine-tune more heavily on Urdu than Hindi (since target language)
- Optional: Chain Urdu-specific teacher first (whisper-turbo-urdu), then multilingual

**Expected improvement:** 5-7% relative WER (from leveraging related language data)  
**Implementation effort:** Medium

---

### 2.5 Curriculum Learning with Self-Generated Labels

**How it works:**

Instead of training on all pseudo-labels at once, train on *easy examples first*, then *hard examples*.

```
Iteration 1: Train on high-confidence examples only (easy)
    ↓ (model improves)
Iteration 2: Lower threshold, add medium-confidence examples
    ↓ (model gets better)
Iteration 3: Add low-confidence examples (hard)
    ↓
Model generalizes better than training on everything at once
```

**Why it works:** Neural networks learn better with curriculum (like humans learning math: algebra before calculus).

**Implementation:**

```python
def curriculum_learning_with_confidence(student, teacher, unlabeled_audio_dir):
    """
    Dynamically adjust confidence threshold based on training progress.
    Easy → Hard schedule.
    """
    confidence_thresholds = [0.95, 0.90, 0.85, 0.80, 0.75]  # Easy to hard
    
    for threshold in confidence_thresholds:
        print(f"\nCurriculum stage: conf_threshold >= {threshold}")
        
        # Build dataset with current threshold
        dataset = ConfidenceFilteredDataset(
            audio_dir=unlabeled_audio_dir,
            teacher_model=teacher,
            conf_threshold=threshold
        )
        
        print(f"  Training set size: {len(dataset)} samples")
        
        # Train for a few epochs
        for epoch in range(3):
            avg_loss = 0
            for batch in dataloader(dataset, batch_size=4):
                loss = train_step(student, batch, optimizer)
                avg_loss += loss
            
            test_wer = evaluate_on_test_set(student)
            print(f"    Epoch {epoch}: Loss={avg_loss:.4f}, Test WER={test_wer:.2%}")
        
        # Save checkpoint
        student.save(f"qwen3_curriculum_stage_{threshold}.pt")
    
    return student

# Alternative: Adaptive curriculum (adjust threshold based on model performance)
def adaptive_curriculum(student, teacher, unlabeled_audio_dir, num_iterations=10):
    """
    Start strict (high threshold), gradually relax.
    Stop when test WER stops improving.
    """
    threshold = 0.95
    best_wer = float('inf')
    patience = 3
    patience_counter = 0
    
    for iteration in range(num_iterations):
        dataset = ConfidenceFilteredDataset(
            audio_dir=unlabeled_audio_dir,
            teacher_model=teacher,
            conf_threshold=threshold
        )
        
        train_on_dataset(student, dataset, epochs=3, learning_rate=1e-5)
        
        test_wer = evaluate_on_test_set(student)
        print(f"Iteration {iteration}: threshold={threshold:.2f}, WER={test_wer:.2%}")
        
        # Adaptive: Relax threshold if model improving, stop if plateauing
        if test_wer < best_wer:
            best_wer = test_wer
            patience_counter = 0
            threshold = max(0.70, threshold - 0.05)  # Relax by 5%
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Converged. Best WER: {best_wer:.2%}")
                break
    
    return student
```

**Pros:**
- Significantly improves generalization (3-5% relative WER improvement)
- Prevents overfitting to hard, noisy examples early on
- Natural data efficiency: Starts with cleanest data, scales gracefully
- Works well with confidence filtering (thresholds define curriculum stages)
- Can be combined with other techniques (distillation + curriculum)

**Cons:**
- Adds complexity: need to decide threshold schedule
- Slower (multiple passes over dataset with different filtering)
- Threshold tuning required (no universal good values)

**For your Qwen3-ASR:**
- Use confidence scores from Section 2.2
- Start: threshold=0.90 (very high confidence)
- Progress: 0.85 → 0.80 → 0.75 (decrease by 5% per curriculum stage)
- 3 epochs per stage (can adjust)
- Monitor test WER; stop if plateau

**Expected improvement:** 3-5% relative WER reduction  
**Implementation effort:** Low-Medium (modify training loop)

---

### 2.6 Teacher Model Selection for Urdu

Your choice of teacher dramatically impacts distillation quality. Here's the comparison:

| Teacher | Urdu Performance | Speed (CPU) | Speed (Metal/MLX) | Pros | Cons |
|---|---|---|---|---|---|
| **Whisper-large-v3** | 20-22% WER est. | ~30s | ~2-5s | Robust, well-studied, multilingual | Larger, slower than turbo |
| **whisper-large-v3-turbo-urdu** | 26.2% WER | ~15s | ~1-3s | Fast, Urdu-specific, easy setup | Slightly higher WER than large-v3 |
| **MMS-1B-all** | Unknown (likely 25-30% WER) | ~20s | ~2s | Extremely lightweight, 1100 languages | Less studied, weaker on Urdu than Whisper |
| **SeamlessM4T-large** | 15-17% WER est. | ~40s | ~5-10s | Best quality, multilingual, streaming | Large, slow, harder to run locally |

**Recommendation for distilling Qwen3-ASR:**

**Primary:** `whisper-large-v3-turbo-urdu`
- Good balance of speed + accuracy
- Urdu-specific training → better pseudo-labels for Urdu student
- Runs in 1-3s on Mac (practical for generating pseudo-labels)

**Backup 1:** `whisper-large-v3` (vanilla)
- If turbo version doesn't exist or performs worse
- Better generalization, handles code-switching better

**Backup 2:** `MMS-1B-all`
- If you need extreme speed or minimal memory
- Handles many languages (useful if curriculum includes English)

**For comparison/ensemble:** SeamlessM4T (if GPU available)

---

## Part 3: Low-Resource Language Strategies (Urdu/Hindi Context)

### 3.1 Leveraging Phonetic Similarity (Urdu↔Hindi)

**The core problem:** Urdu is low-resource (limited labeled speech data).

**The opportunity:** Hindi has abundant data (Common Voice: 1000+ hours) + shares identical phonemes.

**Strategy A: Parallel Training on Hindi + Urdu**

```python
def train_on_similar_languages(student, teacher):
    """
    Train on both Hindi (abundant) + Urdu (scarce) simultaneously.
    Since phonemes identical, both improve each other.
    """
    hindi_audio_dir = "common_voice_hindi/"  # Abundant, labeled
    urdu_audio_dir = "collected_urdu_audio/"  # Scarce, pseudo-labeled
    
    # Generate pseudo-labels for both
    hindi_pseudo = ConfidenceFilteredDataset(
        audio_dir=hindi_audio_dir,
        teacher_model=teacher,
        conf_threshold=0.85
    )
    
    urdu_pseudo = ConfidenceFilteredDataset(
        audio_dir=urdu_audio_dir,
        teacher_model=teacher,
        conf_threshold=0.85
    )
    
    # Train student on both, with curriculum
    combined_dataset = CombinedDataset([hindi_pseudo, urdu_pseudo])
    train_student(student, combined_dataset, epochs=10, lr=1e-5)
```

**Strategy B: Hindi Pre-training → Urdu Fine-tuning**

```python
# Phase 1: Train on Hindi (warm up weights)
train_on_hindi_data(student, teacher, epochs=10, lr=1e-4)

# Phase 2: Fine-tune on Urdu (task-specific, high-confidence only)
urdu_dataset = ConfidenceFilteredDataset(
    audio_dir="urdu_audio/",
    teacher_model=teacher,
    conf_threshold=0.90  # Stricter for fine-tuning
)
train_student(student, urdu_dataset, epochs=5, lr=1e-5)  # Much smaller LR
```

**Pros:**
- Hindi has ~1000 hours in Common Voice; Urdu has ~100 hours
- Identical phonemes → knowledge transfers directly
- Zero manual annotation (all pseudo-labeled)
- Practical: Hindi data readily available

**Cons:**
- Script difference (Devanagari vs. Nastaliq) may confuse student
- Hindi-trained model outputs Hindi → requires transliteration
- Accent/intonation differences between Hindi and Urdu varieties

**For your Qwen3-ASR:**
- Download Common Voice Hindi dataset
- Generate pseudo-labels using Whisper teacher
- Pre-train on Hindi (epochs 5-10)
- Fine-tune on Urdu (epochs 3-5, lower learning rate)
- Expected WER improvement: 5-10% relative

---

### 3.2 Semi-Supervised Learning with Unlabeled Data

**How it works:**

Mix labeled + unlabeled data during training:

```
Labeled data (few 100s of hours)  ──┐
                                      ├─→ Train student
Unlabeled data (many 1000s)  ────────┘
  + pseudo-label with teacher
  + confidence-filter
```

**Implementation:**

```python
class SemiSupervisedASRDataset:
    def __init__(self, labeled_audio_dir, unlabeled_audio_dir, teacher_model):
        self.labeled_samples = load_labeled_data(labeled_audio_dir)  # Real labels
        
        self.unlabeled_samples = ConfidenceFilteredDataset(
            audio_dir=unlabeled_audio_dir,
            teacher_model=teacher_model,
            conf_threshold=0.85  # Pseudo-labels
        )
    
    def __iter__(self):
        # Interleave labeled + unlabeled
        while True:
            labeled_batch = next(iter(DataLoader(self.labeled_samples)))
            unlabeled_batch = next(iter(DataLoader(self.unlabeled_samples)))
            
            yield {
                'labeled': labeled_batch,
                'unlabeled': unlabeled_batch
            }

# Training with semi-supervised loss
def semi_supervised_train(student, semi_dataset):
    for batch in semi_dataset:
        # Supervised loss (real labels)
        labeled_loss = criterion(
            student(batch['labeled']['audio']),
            batch['labeled']['labels']
        )
        
        # Unsupervised loss (pseudo-labels + confidence weighting)
        unlabeled_predictions = student(batch['unlabeled']['audio'])
        unlabeled_targets = batch['unlabeled']['pseudo_labels']
        unlabeled_confidence = batch['unlabeled']['confidence_weights']
        
        unlabeled_loss = criterion(unlabeled_predictions, unlabeled_targets)
        unlabeled_loss = unlabeled_loss * unlabeled_confidence  # Weight by confidence
        
        # Blend (e.g., 50% labeled, 50% unlabeled)
        total_loss = 0.5 * labeled_loss + 0.5 * unlabeled_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

**Pros:**
- Leverages abundant unlabeled data (YouTube, podcasts, etc.)
- Confidence-based weighting prevents overfitting to noisy labels
- Can reach high accuracy with few labeled examples
- Practical: Real-world audio often unlabeled

**Cons:**
- Still requires some labeled data (for supervised component)
- Pseudo-label noise compounds over time
- Hyperparameter tuning (blend ratio, confidence threshold)

**For your Qwen3-ASR:**
- If you have 10-50 manually transcribed Urdu audio samples (labeled):
  - Use confidence-filtered pseudo-labels for unlabeled data
  - Blend 50-50 in training
  - Expected improvement: 5-8% relative WER
- If fully unlabeled (no manual labels):
  - Use pure pseudo-labeling + self-training (Section 2.3)

---

### 3.3 Weak Supervision from Related Scripts

**Key idea:** Urdu transcripts exist in Roman Urdu (Latin script transliteration). You can use these as weak supervision without phonetic annotation.

**How it works:**

```
Urdu audio
    ↓
Teacher generates Hindi script (Devanagari)
    ↓
Roman Urdu transliterator converts: Hindi → Roman Urdu
    ↓
Student trains to match Roman Urdu
    (different script, same language!)
```

**Why this matters:**
- You have Roman Urdu transliteration pipeline (from your `hindi_to_roman_urdu.py`)
- Roman Urdu transcripts available (online forums, social media)
- Less strict than phonetic annotation, more flexible than nothing

**Implementation:**

```python
def weak_supervision_with_roman_urdu(student, unlabeled_audio_dir, transliterator):
    """
    Use Roman Urdu as weak supervision without manual phonetic annotation.
    """
    for audio_file in os.listdir(unlabeled_audio_dir):
        audio_path = os.path.join(unlabeled_audio_dir, audio_file)
        
        # Step 1: Teacher generates Urdu (Devanagari)
        urdu_devanagari = teacher.transcribe(audio_path)
        
        # Step 2: Transliterate to Roman Urdu (your existing pipeline!)
        roman_urdu = transliterator.devanagari_to_roman_urdu(urdu_devanagari)
        
        # Step 3: Student learns to map audio → Roman Urdu
        # (different output format, but same semantic content)
        student_output = student(audio_path, output_script='roman_urdu')
        
        loss = levenshtein_loss(student_output, roman_urdu)  # String-level loss
        
        loss.backward()
        optimizer.step()
```

**Pros:**
- Leverages your existing transliteration pipeline
- Roman Urdu transcripts available online (no annotation cost)
- Gives student additional supervision signal without new labels
- Flexibility: can accept multiple valid romanizations

**Cons:**
- Requires working transliterator (you have this!)
- Script conversion adds complexity
- May diverge from "correct" Nastaliq if transliterator biased

**For your Qwen3-ASR:**
- Extract Roman Urdu text from existing sources (online forums, OCR'd books)
- Pair with audio where possible (or use text-to-speech)
- Train student to emit Roman Urdu (intermediate representation)
- Post-process: Roman Urdu → Nastaliq for final output

---

## Part 4: Practical Implementation for Qwen3-ASR

### 4.1 End-to-End Distillation Pipeline

Here's a complete, runnable implementation:

```python
# file: qwen3_distill.py
"""
Knowledge distillation pipeline for Qwen3-ASR
- Teacher: Whisper (Urdu or multilingual)
- Student: Qwen3-ASR-1.7B
- No manual annotations required
"""

import os
import json
import torch
import torchaudio
from transformers import (
    AutoProcessor, AutoModelForSpeechSeq2Seq,  # Whisper
)
from datasets import Dataset, DataLoader
import evaluate
from tqdm import tqdm

# ============ CONFIG ============
TEACHER_MODEL_ID = "kingabzpro/whisper-large-v3-turbo-urdu"  # or "openai/whisper-large-v3"
STUDENT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
UNLABELED_AUDIO_DIR = "./urdu_audio_unlabeled/"
TEST_AUDIO_DIR = "./urdu_audio_test/"
CONFIDENCE_THRESHOLD = 0.85
TEMPERATURE = 4.0
ALPHA = 0.7  # Blend of teacher (0.7) vs supervised (0.3)
LEARNING_RATE = 1e-5
NUM_EPOCHS = 5
BATCH_SIZE = 4

# ============ STEP 1: Load Models ============
print("Loading teacher (Whisper)...")
teacher_processor = AutoProcessor.from_pretrained(TEACHER_MODEL_ID)
teacher_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    TEACHER_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cuda" if torch.cuda.is_available() else "cpu"
)
teacher_model.eval()

print("Loading student (Qwen3-ASR)...")
# Note: Qwen3-ASR loading requires: qwen-asr library + special handling
# (See qwen3-asr-local/asr_transcribe_and_transliterate.py for reference)
from qwen_asr import Qwen3ASRModel
student_model = Qwen3ASRModel.from_pretrained(
    STUDENT_MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cuda" if torch.cuda.is_available() else "cpu"
)

# ============ STEP 2: Generate Pseudo-Labels with Confidence ============
print(f"\nGenerating pseudo-labels from {UNLABELED_AUDIO_DIR}...")

pseudo_labels_cache = {}
for audio_file in tqdm(os.listdir(UNLABELED_AUDIO_DIR)):
    if not audio_file.endswith('.wav'):
        continue
    
    audio_path = os.path.join(UNLABELED_AUDIO_DIR, audio_file)
    
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    inputs = teacher_processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt")
    
    # Teacher inference with output scores (logits)
    with torch.no_grad():
        outputs = teacher_model.generate(
            **inputs,
            task="transcribe",
            language="ur",
            output_scores=True,
            return_dict_in_generate=True
        )
    
    # Extract logits and compute confidence (see word-level-confidence.md)
    logits = outputs.scores  # List of tensors: [(vocab_size,), ...]
    confidences = []
    for logit in logits:
        logprob = torch.log_softmax(logit, dim=-1)
        conf = torch.exp(logprob.max())
        confidences.append(conf.item())
    
    transcription = teacher_processor.batch_decode(
        outputs.sequences,
        skip_special_tokens=True
    )[0]
    
    # Compute per-word confidence (simplified; see word-level-confidence.md for full logic)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    # Store if confidence high enough
    if avg_confidence > CONFIDENCE_THRESHOLD:
        pseudo_labels_cache[audio_file] = {
            'transcription': transcription,
            'confidence': avg_confidence,
            'audio_path': audio_path
        }

print(f"Generated {len(pseudo_labels_cache)} high-confidence pseudo-labels")

# ============ STEP 3: Build Distillation Dataset ============
class DistillationDataset(Dataset):
    def __init__(self, pseudo_labels_dict, processor):
        self.data = list(pseudo_labels_dict.values())
        self.processor = processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(item['audio_path'])
        
        return {
            'waveform': waveform.squeeze(),
            'sr': sr,
            'transcription': item['transcription'],
            'confidence': item['confidence']
        }

distill_dataset = DistillationDataset(pseudo_labels_cache, teacher_processor)
distill_loader = DataLoader(
    distill_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda batch: {
        'audio': [teacher_processor(
            b['waveform'],
            sampling_rate=b['sr'],
            return_tensors="pt"
        ) for b in batch],
        'transcription': [b['transcription'] for b in batch],
        'confidence': torch.tensor([b['confidence'] for b in batch])
    }
)

# ============ STEP 4: Distillation Training ============
print("\nStarting distillation training...")

optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    
    for batch_idx, batch in enumerate(tqdm(distill_loader)):
        # Get teacher logits (no gradient)
        with torch.no_grad():
            teacher_logits_list = []
            for audio_input in batch['audio']:
                outputs = teacher_model.forward(
                    **audio_input,
                    output_hidden_states=True
                )
                # Extract logits: (seq_len, vocab_size)
                # Whisper decoder hidden states → logits
                logits = outputs.logits
                teacher_logits_list.append(logits)
        
        # Get student logits
        student_logits_list = []
        for audio_input in batch['audio']:
            outputs = student_model.forward(**audio_input)
            logits = outputs.logits
            student_logits_list.append(logits)
        
        # Compute distillation loss (KL divergence with temperature)
        total_kl_loss = 0.0
        for teacher_logits, student_logits, confidence in zip(
            teacher_logits_list,
            student_logits_list,
            batch['confidence']
        ):
            # Align sequence lengths (truncate to min)
            min_len = min(teacher_logits.size(0), student_logits.size(0))
            teacher_logits = teacher_logits[:min_len]
            student_logits = student_logits[:min_len]
            
            # Soft targets
            teacher_probs = torch.softmax(teacher_logits / TEMPERATURE, dim=-1)
            student_log_probs = torch.log_softmax(student_logits / TEMPERATURE, dim=-1)
            
            # KL divergence
            kl_loss = torch.nn.functional.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='mean'
            ) * (TEMPERATURE ** 2)
            
            # Weight by teacher confidence
            kl_loss = kl_loss * confidence
            total_kl_loss += kl_loss
        
        loss = total_kl_loss / len(batch['audio'])
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_epoch_loss = epoch_loss / len(distill_loader)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {avg_epoch_loss:.4f}")
    
    # Save checkpoint
    student_model.save_pretrained(f"./checkpoints/qwen3_distill_epoch{epoch + 1}")

print("Distillation complete!")

# ============ STEP 5: Evaluate on Test Set ============
print(f"\nEvaluating on test set ({TEST_AUDIO_DIR})...")

# If you have ground truth test labels, compute WER
# Otherwise, just generate transcriptions
test_results = []
for audio_file in os.listdir(TEST_AUDIO_DIR):
    if not audio_file.endswith('.wav'):
        continue
    
    audio_path = os.path.join(TEST_AUDIO_DIR, audio_file)
    
    waveform, sr = torchaudio.load(audio_path)
    inputs = teacher_processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt")
    
    # Student inference
    with torch.no_grad():
        outputs = student_model.generate(**inputs, task="transcribe", language="ur")
    
    transcription = teacher_processor.batch_decode(outputs, skip_special_tokens=True)[0]
    test_results.append({
        'audio': audio_file,
        'transcription': transcription
    })

# Save results
with open("./test_results.json", "w") as f:
    json.dump(test_results, f, indent=2, ensure_ascii=False)

print(f"Test results saved to test_results.json")
```

**Usage:**

```bash
python qwen3_distill.py \
  --unlabeled-dir ./urdu_audio/ \
  --test-dir ./urdu_audio_test/ \
  --confidence-threshold 0.85 \
  --epochs 5 \
  --batch-size 4
```

---

### 4.2 Using GPU/vLLM for Faster Teacher Inference

If you have access to a GPU, accelerate teacher inference:

```python
# file: qwen3_distill_gpu.py
"""
GPU-accelerated distillation using vLLM for fast teacher inference.
"""

from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# Load teacher via vLLM (faster batching)
teacher_llm = LLM(model="kingabzpro/whisper-large-v3-turbo-urdu")
teacher_processor = AutoProcessor.from_pretrained(
    "kingabzpro/whisper-large-v3-turbo-urdu"
)

# Batch inference: 32 audio files at once
audio_batch = [...]  # List of audio paths
inputs_batch = teacher_processor(
    [load_audio(path) for path in audio_batch],
    sampling_rate=16000,
    return_tensors="pt"
)

# vLLM batches efficiently
outputs = teacher_llm.generate(inputs_batch)

# This is ~10x faster than serial inference
```

---

### 4.3 Integration with Your Existing Confidence Pipeline

Reuse your `word-level-confidence.md` logic:

```python
# file: integration_example.py
"""
Integrate distillation with your existing confidence extraction.
"""

from qwen3_asr_local.asr_transcribe_and_transliterate import (
    hf_asr_with_confidence,
    WordConf
)

def distill_with_existing_confidence(audio_path):
    """
    Use your existing confidence extraction during distillation.
    """
    # Generate pseudo-label + confidence (using your existing code)
    words_conf = hf_asr_with_confidence(
        model=teacher_model,
        audio_path=audio_path
    )
    
    # Filter by confidence (your existing thresholds)
    high_conf_words = [
        word for word in words_conf
        if word.min_conf > 0.85  # or word.geo_conf > 0.80
    ]
    
    # Only use high-confidence words as training targets
    if len(high_conf_words) > 0:
        transcription = " ".join([w.text for w in high_conf_words])
        avg_conf = sum([w.geo_conf for w in high_conf_words]) / len(high_conf_words)
        
        return {
            'transcription': transcription,
            'confidence': avg_conf,
            'num_words': len(high_conf_words)
        }
    else:
        return None
```

---

## Part 5: Prioritized Roadmap

Based on effort vs. impact, here's the recommended implementation order:

### **Phase 1: Quick Wins (1-2 weeks)**

| Technique | Effort | Expected Gain | Start | Implementation |
|---|---|---|---|---|
| **Confidence-based filtering (2.2)** | Low | 3-5% WER | Week 1 | Reuse existing confidence scores; filter pseudo-labels |
| **Use better teacher** | Very Low | 5-10% WER | Week 1 | Switch to `whisper-large-v3-turbo-urdu` |
| **Pseudo-labeling on unlabeled data** | Low | 3-5% WER | Week 2 | Generate pseudo-labels for 100+ hours of unlabeled Urdu audio |

**Expected total improvement:** 8-15% relative WER  
**Cumulative effort:** ~100-150 GPU hours (mostly teacher inference, can be parallelized)

### **Phase 2: Standard Distillation (2-4 weeks)**

| Technique | Effort | Expected Gain | Start | Implementation |
|---|---|---|---|---|
| **Dark knowledge distillation (2.1)** | Medium | 3-5% WER | Week 3 | Modify training loop; cache teacher logits |
| **Curriculum learning (2.5)** | Low-Medium | 2-3% WER | Week 3 | Adjust confidence threshold schedule |
| **Self-training iterations (2.3)** | Medium | 5-10% WER | Week 4 | Multi-pass pseudo-labeling with improving student |

**Expected total improvement:** 8-15% relative WER (on top of Phase 1)  
**Cumulative effort:** ~300-500 GPU hours (actual training)

### **Phase 3: Advanced (4-8 weeks)**

| Technique | Effort | Expected Gain | Start | Implementation |
|---|---|---|---|---|
| **Cross-lingual distillation (2.4)** | Medium | 5-7% WER | Week 5 | Leverage Hindi data + multilingual teacher |
| **Semi-supervised learning (3.2)** | Medium | 5-8% WER | Week 6 | If you have 10-50 labeled samples |
| **Weak supervision (3.3)** | Medium | 2-4% WER | Week 7 | Leverage Roman Urdu transcripts |

**Expected total improvement:** 10-20% relative WER  
**Cumulative effort:** ~500-1000 GPU hours

### **Phase 4: Production Optimization (Ongoing)**

- Monitor WER on held-out test set
- A/B test different teacher models
- Ensemble multiple distilled students
- Deploy best checkpoint (4.3)

---

## Part 6: Recommended Starting Point for Your Project

Given your current setup (Qwen3-ASR + word-level confidence extraction), I recommend:

### **Immediate (This week):**

1. **Switch teacher to `whisper-large-v3-turbo-urdu`**
   - Better for Urdu specifically
   - Fast enough for pseudo-label generation (~1-3s per audio on Mac)
   - Well-tested

2. **Collect unlabeled Urdu audio** (target: 500+ hours)
   - YouTube: Urdu podcasts, lectures, news
   - Common Voice: https://commonvoice.mozilla.org/ur (download existing)
   - Urdu media platforms (Dawn, Express, etc.)

3. **Generate pseudo-labels + filter by confidence**
   - Use existing `hf_asr_with_confidence()` function
   - Threshold: 0.85 (high confidence)
   - Expected: ~200-300 hours of clean pseudo-labeled data

4. **Train Qwen3-ASR on filtered pseudo-labels**
   - Dark knowledge distillation (Section 4.1)
   - Learning rate: 1e-5 (fine-tuning, not from scratch)
   - Epochs: 3-5
   - Expected improvement: 5-8% relative WER

**Effort:** 2-3 weeks (mostly data collection + teacher inference)  
**Expected WER improvement:** 5-10% relative

### **Follow-up (Next month):**

5. Add curriculum learning (2.5) → +2-3% WER
6. Add self-training (2.3) → +5-10% WER
7. Optionally: cross-lingual with Hindi (2.4) → +5-7% WER

---

## Summary Table: Techniques at a Glance

| Technique | When to Use | Effort | Gain | Notes |
|---|---|---|---|---|
| **Confidence filtering (2.2)** | Always first | Low | 3-5% | Reuse existing infrastructure |
| **Dark knowledge (2.1)** | Phase 1-2 | Medium | 3-5% | Standard KD; requires logits |
| **Pseudo-labeling (2.3)** | Phase 1 | Low | 3-5% | Quick win; unlabeled data only |
| **Self-training (2.3)** | Phase 2 | Medium | 5-10% | Iterative; requires monitoring |
| **Curriculum learning (2.5)** | Phase 2 | Low-Medium | 2-3% | Pairs well with KD + pseudo-labels |
| **Cross-lingual (2.4)** | Phase 3 | Medium | 5-7% | Leverage Hindi; multilingual teacher |
| **Semi-supervised (3.2)** | Phase 3 | Medium | 5-8% | If you have labeled data |
| **Weak supervision (3.3)** | Phase 3 | Medium | 2-4% | Leverage Roman Urdu transcripts |

---

## References & Further Reading

### Key Papers

1. **Knowledge Distillation for ASR:**
   - Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"
   - Fang et al. (2021): "Learning Student Networks via Feature Embedding"

2. **Pseudo-labeling & Self-training:**
   - Lee et al. (2013): "Pseudo-label: The Simple and Efficient Semi-supervised Learning Method for Deep Neural Networks"
   - Xie et al. (2020): "Self-training with Noisy Student improves ImageNet classification"

3. **ASR-Specific Distillation:**
   - Shi et al. (2021): "Large-Scale Unsupervised Speech Recognition"
   - Baevski et al. (2020): "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"

4. **Curriculum Learning:**
   - Bengio et al. (2009): "Curriculum Learning"
   - Zhang et al. (2020): "Curriculum Learning for NLP"

5. **Low-Resource ASR:**
   - Pratap et al. (2023): "Massively Multilingual ASR with Efficient Conformers"
   - Xia et al. (2022): "Towards End-to-End Speech Recognition with Deep Multipath Networks"

### Tools & Models

- **Whisper Fine-tuned for Urdu:** [kingabzpro/whisper-large-v3-turbo-urdu](https://huggingface.co/kingabzpro/whisper-large-v3-turbo-urdu)
- **Whisper Large v3:** [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
- **SeamlessM4T:** [facebook/seamless-m4t-large](https://huggingface.co/facebook/seamless-m4t-large)
- **MMS-1B-all:** [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all)
- **Common Voice Datasets:** [https://commonvoice.mozilla.org/](https://commonvoice.mozilla.org/)

---

## Appendix A: FAQ

**Q: Do I really need manual annotations?**  
A: No. All techniques in this document work with unlabeled audio + a pretrained teacher. Pseudo-labels replace manual transcription.

**Q: What if my teacher model is worse than my student?**  
A: Choose a better teacher (Whisper-large-v3 is robust). If student better than teacher, techniques still help but less impactful. Consider ensemble distillation (multiple teachers).

**Q: How much unlabeled Urdu audio do I need?**  
A: Start with 100 hours, scale to 500+. More data → more iterations possible. Even 50 hours helps if confidence filtering is strict.

**Q: Can I use publicly available transcripts instead of generating pseudo-labels?**  
A: Yes! Weak supervision (Section 3.3) uses Roman Urdu transcripts. High-confidence sources (news, podcasts with existing captions) can be paired with audio.

**Q: Should I use GPU or CPU for teacher inference?**  
A: GPU is 10-20x faster, but CPU works. For 500+ hours, GPU strongly recommended (saves weeks).

**Q: How do I know if distillation is working?**  
A: Monitor test WER. With Phase 1 techniques, expect 5-10% relative improvement within 2-3 weeks.

---

## Appendix B: Configuration Recommendations

### Conservative (Lowest Risk)
- Teacher: Whisper-large-v3-turbo-urdu
- Confidence threshold: 0.90 (strict)
- Curriculum stages: [0.95, 0.90, 0.85]
- Epochs: 3
- Learning rate: 1e-5
- Expected gain: 3-5% WER

### Balanced (Recommended)
- Teacher: Whisper-large-v3-turbo-urdu
- Confidence threshold: 0.85 (default)
- Curriculum stages: [0.90, 0.85, 0.80, 0.75]
- Epochs: 5
- Learning rate: 1e-5
- Expected gain: 8-12% WER

### Aggressive (Highest Gain, Higher Risk)
- Teacher: Ensemble (Whisper-large-v3 + SeamlessM4T)
- Confidence threshold: 0.75-0.80 (loose)
- Curriculum stages: [0.95, 0.85, 0.75, 0.65]
- Epochs: 10
- Learning rate: 5e-5
- Self-training iterations: 3
- Expected gain: 12-20% WER
- Risk: Overfitting to noisy labels; requires careful monitoring

---

**Last updated:** May 18, 2026
