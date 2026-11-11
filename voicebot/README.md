# Chughtai Lab voice bot (v1)

Roman Urdu phone/voice assistant: **home sampling booking** and **booking status / reschedule / cancel**, with FAQs,
test price lookups, and switching between tasks mid-call. Plan: [PLAN.md](PLAN.md). Progress: [PROGRESS.md](PROGRESS.md).

## Run it

```bash
conda env create -f environment.yml      # separate env `voicebot` (not the ASR env)
conda activate voicebot
cp .env.example .env                     # fill in tokens/URLs (never commit .env)

python -m app.server                     # test console: http://localhost:7860  (chat works with no endpoints)
python scripts/run_scenarios.py          # 40+ scripted conversations with expectations
python scripts/run_router_bakeoff.py -v  # routing accuracy: rules vs Laya (vs Qwen when LLM_BASE_URL is set)
python scripts/check_data.py             # validate data lists
python scripts/ping_endpoints.py         # STT / TTS / LLM reachability, latency, message formats
python scripts/run_noise_eval.py         # robustness to speech-recognition errors
python scripts/check_workflow.py <name>  # validate a workflow + run every scenario and the routing eval
python scripts/new_workflow.py <name> "<title>"   # scaffold a new workflow folder
pytest                                   # unit + conversation + API + voice-plugin tests

# voice loop with FAKE STT/TTS (no tokens): needs Docker for the LiveKit server
docker compose -f deploy/docker-compose.yml up -d livekit
python scripts/e2e_voice_fake.py                      # also: --scenario hs_happy_path | --barge-in
python scripts/ui_smoke.py                            # drives the console in headless Chrome
python -m app.agent start                             # real voice worker (needs STT/TTS tokens in .env)
```

## Status
v1 is built and tested without the real STT/TTS/LLM endpoints (they are private Spaces and need tokens). See
[PROGRESS.md](PROGRESS.md) for what was verified, what was not, and decisions that differ from the plan.

## What works without any endpoint
Everything in chat mode: the conversation engine, router (rules; Laya if installed), FAQ, test lookups, mock booking
backend, scenario picker. The LLM (Qwen3-8B, `LLM_BASE_URL`) improves name extraction and routing when set.
STT/TTS (voice mode) need the Space tokens: `QWEN_VLLM_TOKEN`, `OMNI_TTS_TOKEN`.

## Layout
- `app/bot.py` turn loop (route -> dispatch), `app/flow/` workflow engine and slot parsers, `app/router/` router chain
- `app/datakit/` searchable data lists (`data/*.csv` + `config/entities/*.yaml`), `app/faq.py` + `config/faq.yaml`
- `workflows/<name>/` one folder per workflow: `workflow.yaml`, `manifest.yaml`, `hooks.py`
- `eval/scenarios/*.yaml` conversation scenarios (shared by the UI picker and the tests), `eval/routing/` routing eval set
- `ui/` test console, `app/server.py` FastAPI backend, `app/services/` STT/TTS clients, `app/agent.py` LiveKit worker

## Data is seed data
`data/*.csv` and `config/faq.yaml` contain **placeholder** tests, prices, areas and answers (column `source=seed`).
Replace them with the real lists before any real use; run `scripts/check_data.py` afterwards.

## Add a workflow
Create `workflows/<name>/` with `workflow.yaml` (slots + steps), `manifest.yaml` (description, example phrases,
`strong_keywords`) and optional `hooks.py`; add scenarios under `eval/scenarios/`; run
`scripts/run_scenarios.py` and `scripts/run_router_bakeoff.py`. See PLAN.md section 3c.
