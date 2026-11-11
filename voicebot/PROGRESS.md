# Implementation progress (v1)

Last updated 2026-10-22. Plan: PLAN.md. Environment: conda env `voicebot` (environment.yml); the old `.venv` was removed.

## Status: v1 built and tested without the real STT / TTS / LLM endpoints

| Phase | Status | Evidence |
|---|---|---|
| 0 Scaffold + endpoint check | done | `scripts/ping_endpoints.py`; STT/TTS Spaces answer 404/timeout without tokens (private) |
| 1 Workflow A, data, chat UI | done | 41 scenarios + 70+ unit tests pass |
| 1b Workflow B + switching | done | `booking_status`; switch/resume/decline/non-interruptible scenarios pass |
| 2 Noise robustness | done | `scripts/run_noise_eval.py`: 100% complete at 10-20% word noise, 83% at 30%, 0% wrong data submitted |
| 2b Router bake-off | done (Qwen not measurable yet) | Laya zero-shot unusable; rules 99.3% on the (self-tuned) set; default router set to qwen + rules |
| 3 Voice | done with fake STT/TTS; real endpoints pending tokens | `scripts/e2e_voice_fake.py`: LiveKit server (Docker) + agent + simulated caller with real call audio; full booking, switch+resume and barge-in pass |
| 4 Telephony | files only | `deploy/` compose + configs; needs a public-IP host and SIP provider, not exercised |
| 5 Pilot hardening | not started | needs real endpoints, API, shadow-mode calls |
| UI | done | `scripts/ui_smoke.py` drives headless Chrome: new call, chat turn, panels, guided scenario, Run all, voice toggle, 0 JS errors |

## Sample data (replace with the real lists later)
- `data/tests.csv`: 129 sample tests (pathology, serology, tumor markers, imaging, cardiac, packages). Aliases include how the
  speech recogniser really writes names in your recorded calls (e.g. `seebeesee` = CBC, `elaephti` = LFT, `echbi a one si` = HbA1c).
  `home_collection=no` marks tests that need a lab visit (MRI, CT, ultrasound, ECG, biopsy...): the home-sampling workflow
  explains this instead of booking them. Prices are placeholders.
- `data/areas.csv` (66 areas, Lahore-focused plus Karachi, Islamabad, Rawalpindi, Faisalabad, Multan, Sialkot, Gujranwala),
  `data/branches.csv` (12), `data/pricing.csv` (placeholder values and contact details, NOT real).
- `config/faq.yaml`: 33 sample FAQs on topics callers actually ask (online/WhatsApp reports, lost slip, report delay, home sampling
  cities/hours/charge, MRI/CT appointments and discounts, doctor transfer, payment, fasting, prescriptions). Wording and facts are
  placeholders (`sample: true`); entries with `handoff: true` answer and then transfer to a person.
- `scripts/check_data.py` validates all lists and the FAQ file and warns while sample content is in use.

## Blockers (need you)
- `QWEN_VLLM_TOKEN`, `OMNI_TTS_TOKEN` (STT/TTS Spaces are private) and `LLM_BASE_URL` for Qwen3-8B in `.env`.
- Real test list, areas, prices, FAQ list (seed data in use, marked `source=seed`).
- Confirm workflow B (`booking_status`), telephony provider and pilot host.

## Decisions made while building (deviations from PLAN.md, with reasons)
1. **Bot replies are templated Roman Urdu, not LLM-phrased.** The LLM (Qwen3-8B) is used for name extraction and routing only. Reasons: determinism, testability, latency, no hallucination on a medical-adjacent flow. LLM-phrased replies can be added later behind a flag.
2. **STT uses the HTTP batch endpoint by default** (`STT_MODE=http`, variant `chughtai`), websocket optional. The repo's own notes (qwen3-asr-local/acoustic_contextual_biasing/asr.py) report the realtime websocket scoring ~32% vs ~68% for HTTP on the same audio.
3. **Default router is Qwen + rules, not Laya.** Laya zero-shot on Roman Urdu: 16-22% accuracy on utterances needing a model, never above 0.8 confidence, 1.3-5.8 s per call on this CPU. Adapter kept; revisit after fine-tuning.
4. Laya needs `torch==2.2.2`, `numpy<2`, `transformers==4.49` on this Intel Mac and a `torch.compile` no-op shim (Python 3.12); pinned in environment.yml.

## Known limitations / not verified
- Real STT/TTS/LLM never called (no tokens). TTS websocket message format is a best guess; run `scripts/ping_endpoints.py` once tokens exist and adjust `TTS_FORMAT` / `TTS_TEXT_FIELD`.
- The router eval set (149 utterances) was written alongside the rules, so 99.3% is optimistic; needs held-out real utterances.
- Noise eval uses a blind simulated caller that confirms whatever the bot reads back; it measures robustness of understanding, not caller attentiveness.
- Voice mode in the browser (livekit-client) is implemented but was not exercised with a microphone; the Python-side voice loop is tested end to end.
- Latency numbers with real STT/TTS/LLM are unknown; fake-endpoint runs only prove the plumbing.
- Phone line (SIP), TLS/TURN, multi-call load, Docker Compose pilot stack: not tested.
- Seeded mock booking date uses the real calendar date, not the scenario date (cosmetic).

## Commands
`pytest` | `python scripts/run_scenarios.py` | `python scripts/run_noise_eval.py` | `python scripts/run_router_bakeoff.py -v` |
`python scripts/e2e_voice_fake.py [--scenario ID | --barge-in]` | `python scripts/ui_smoke.py` | `python scripts/check_workflow.py <name>`
