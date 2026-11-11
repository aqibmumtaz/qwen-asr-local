# Chughtai Lab Voice Bot: Home Sampling (v1 plan)

**Framework decision: LiveKit Agents (Python), self-hosted** (no LiveKit Cloud), chosen over Pipecat (see §3a).

Status: v1 implemented 2026-10-22 and tested without the real STT/TTS/LLM endpoints; see PROGRESS.md for results and for deviations from this plan. Plan date: 2026-10-21.

Target environment: a **call center**. The call-concurrency plan is in v2 (§14.3); everything else is v1.

Scope of v1: **two real workflows with switching between them**, plus FAQs and lookups.
1. **Workflow A, `home_sampling`:** books a home sampling request by talking in Urdu.
2. **Workflow B (proposed: `booking_status`):** check, reschedule or cancel an existing home-sampling booking. Chosen because it overlaps closely with A (same vocabulary, same entities), which makes it the hardest and most useful test of routing and switching. It reveals and changes one caller's booking, so it needs light verification: the caller must give the booking reference and the phone number on the booking (matched exactly, read back digit by digit) before details are spoken or a cancel/reschedule is done. **To be confirmed by you**; any other workflow can replace it (see §11).
3. **Switching:** the caller can move between A and B at any point (suspend, run the other, resume), with confirmation and safety rules (§3c, §3e).
4. **FAQs** answered word for word from `faq.yaml`, and **test price / test info** (`lookup`: resolve the test name from the test list and read the fact, then return to what was pending).

Everything else (report status, complaints, results interpretation) goes to a human. "Working perfectly" is defined by measurable targets in §10, not by a promise: real calls over noisy phone audio will still produce errors, so the design makes every error recoverable (confirm, retry, human) and measures how often they happen.

---

## 1. Verdict on feasibility

**Feasible, with one real risk: STT accuracy.**

| Area | Feasibility | Why |
|---|---|---|
| Pipeline (VAD → STT → LLM → TTS) in LiveKit Agents | High | Standard shape. STT and TTS are WebSocket endpoints and the LLM is OpenAI-compatible HTTP, all already available, so the work is writing thin STT/TTS plugins (WebSocket clients). LiveKit adds a server to run (see §3a). |
| Conversation design for home sampling | High | Narrow, form-like task: ~8 slots to fill, then 1 API call. A deterministic flow with LLM help is far more reliable than a free-form agent. |
| Qwen3-8B as the brain | Medium-High | Good enough for slot extraction and short replies in Roman Urdu *if* it is kept on a tight leash (small tool set, flow state in the prompt, thinking mode off). Not good enough to trust with free-form tool use or free-form confirmation logic. |
| STT quality on real calls | **Medium-Low, the main risk** | The `/chughtai` benchmark in this repo (`benchmark_acoustic_contrastive_contextual.py`) reports ~65.87% for the HTTP + dynamic-chunk setup. Names, addresses, phone digits and test names are exactly the content the bot must get right. So the design assumes STT is wrong a third of the time and defends against it (see §6). |
| TTS | Unknown | Depends on your TTS endpoint (input script, latency, streaming). See open questions. |
| Latency | Medium | If the STT websocket streams partial results, latency is lower; if it only returns after the clip ends, it adds a full round trip after the caller stops talking (to be measured in Phase 0). Target ≤ 2.0 s turn gap; realistic first target ≤ 3 s. See §7. |

**Bottom line:** a working voice demo with two workflows and switching in ~2 weeks (about 9 working days) is realistic. A bot that books correctly *unattended* on noisy phone calls needs the confirmation and human-fallback design below, and its success rate will be set mostly by STT. Recommend launching as "bot collects the request, human verifies by callback" first, then removing the callback once measured accuracy allows.

---

## 2. What the bot does (v1 call flow)

Slots to collect:

| Slot | Notes |
|---|---|
| `patient_name` | Often wrong from STT. Confirm. |
| `phone` | Prefer caller ID from telephony. Otherwise ask digit by digit and read back. |
| `age`, `gender` | Needed for test/phlebotomist choice. |
| `tests` | Free text ("sugar, CBC, vitamin D") or "doctor ki parchi mein likha hai". Matched against a test catalog. |
| `address` | Area/society, street/house number, city, landmark. Matched against the service-area list. |
| `preferred_date`, `preferred_slot` | "kal subah", "parson dopahar". Resolved in code to a date plus a slot id. |
| `fasting_ack` | If any selected test needs fasting, tell the caller and get a yes. |
| `notes` | Optional ("do bar bell na bajayen", "lift nahi hai"). |

Happy-path states (a flow, not free chat):

```
GREETING → ASK_TESTS → ASK_PATIENT (name/age/gender) → ASK_ADDRESS
 → CHECK_COVERAGE → ASK_DATETIME → CONFIRM_PHONE → READ_BACK_SUMMARY
 → (haan) SUBMIT → CLOSING
```

Exits from any state: `TRANSFER_TO_HUMAN` (caller asks, 2 failed attempts on one slot, out-of-scope request, caller angry) and `END_CALL`.

Sample (Roman Urdu, as the LLM sees it):

```
Bot : Assalam o alaikum, Chughtai Lab mein khush aamdeed. Main aapki home sampling booking mein madad kar sakti hoon. Kaun se tests karwane hain?
User: sugar aur cbc karwana hai ghar pe
Bot : Ji, sugar aur CBC. Mareez ka naam kya hai?
User: muhammad ali, umar painteesh saal
Bot : Theek hai, Muhammad Ali, painteesh saal. Sample kis address par lena hai?
User: gulshan e iqbal block 13 d ghar number 45
Bot : Gulshan-e-Iqbal block 13 D, ghar number paintaalees. Kis din aur kis waqt aana theek rahega?
User: kal subah
Bot : Kal subah saat se gyarah baje tak ka slot hai. Theek hai?
...
Bot : Confirm kar leti hoon: Muhammad Ali, painteesh saal, sugar aur CBC, Gulshan-e-Iqbal block 13 D ghar 45, kal subah, rabta number 0300 ... Sab theek hai?
User: haan
Bot : Aapki booking ho gayi. Reference number ... Shukriya.
```

---

## 3. Architecture

```
Phone (SIP) / browser test UI
      │ audio (8 kHz telephony, or 16 kHz WebRTC for dev)
      ▼
[LiveKit room: WebRTC / SIP] ─► [Silero VAD + StreamAdapter] ─► [Custom STT plugin] ──WebSocket──► QWEN_VLLM_BASE_URL (Roman Urdu out)
                                              │
                                   [Text normalizer / ASR corrector]   (reuse resolver.py / corrector.py ideas)
                                              ▼
                                  [Flow engine: state + slots]
                                              │
                          ┌───────────────────┴──────────────────┐
                          ▼                                      ▼
                  [Qwen3-8B via OpenAI-compatible API]    [Deterministic tools in code]
                  (extract slots, phrase reply)            (catalog match, coverage, slots, submit)
                                              │ reply text (Roman Urdu)
                                              ▼
                              [Script adapter: Roman→Urdu if TTS needs it]
                                              ▼
                              [Custom TTS plugin] ──WebSocket──► OMNI_TTS
                                              ▼
                                      [LiveKit room audio out]
```

Key decisions:

1. **LiveKit Agents** (`AgentSession`) with custom STT and TTS plugins. STT is a WebSocket realtime endpoint: if it streams usable partials, the plugin feeds audio continuously (Silero VAD still decides end of turn); if it only works per clip, Silero VAD plus LiveKit's `StreamAdapter` cuts utterances and sends each whole clip. Phase 0 decides which. The LLM uses LiveKit's OpenAI plugin pointed at the Qwen endpoint via `base_url`. Check class names and signatures against the installed version, because LiveKit's own plugin guide shows older APIs.
2. **Flow-driven, LLM-assisted.** The state machine decides what to ask next and when to call APIs. The LLM is used for (a) extracting slot values from noisy Roman Urdu, (b) phrasing natural replies, (c) handling digressions ("ye test kitne ka hai?" → polite redirect or transfer). Implemented as a hand-written state machine in plain Python (`flow/home_sampling.py`) driving one LiveKit Agent whose instructions and tools change per state (agent/task handoffs can be used if they fit). Keeping the flow framework-free keeps it testable without audio.
3. **Tools are few, English-only, and mostly deterministic**: see §5. The LLM never decides *whether* to submit; code does, after the confirmed flag is set.
4. **Language handling.** Internal text is Roman Urdu end to end. Tool args are English/canonical. A single `script_adapter` module owns Roman→Urdu-script conversion if the TTS needs it, so the rest of the system never cares.
5. **Mock backend first.** `backend/` exposes the same interface as the real Chughtai booking API (`check_coverage`, `get_slots`, `create_request`) backed by JSON files. Swap in the real API later without touching the flow.

---

### Build scope: what is in v1 and what is deliberately deferred

| In v1 | Deferred until needed |
|---|---|
| Two workflows (`home_sampling`, `booking_status`) as plain-Python state machines behind one workflow interface | Adopting a dialogue framework (Rasa, LiveKit tasks). Revisit at 3-4 workflows if the state machines get messy. |
| Router `route()`: **Laya primary + Qwen3-8B fallback** (env-configurable), cheap "is this an answer?" pass, all decisions: `continue`, `faq`, `lookup`, `switch`, `resume`, `unclear`, `human`, `end` | Fine-tuned small router / custom model training (only if the bake-off in Phase 2b misses the targets) |
| Router bake-off (Laya vs Qwen3-8B vs baseline) on a Roman Urdu routing eval set, and the calibrated thresholds | Embedding-based shortlist (add when workflows exceed ~5) |
| Session manager: workflow stack, suspend/resume, shared facts (name, phone), non-interruptible states, max 2 suspended | Verification sub-workflow (added with the first workflow that reveals patient data) |
| FAQ layer (`faq.yaml`), `lookup` answers (test price/info via datakit), human transfer | Data tab, alias-learning pipeline from logs |
| `datakit` (CSV loader + resolver, 3 functions) and the `resolve_entity` spoken sub-flow | datakit service mode, other search backends, embeddings, index versioning, cross-list links (§3h) |
| Logging of every routing decision (data for later training) | |

This design follows the common hybrid pattern for regulated call bots (fixed flows for side-effecting tasks, LLM for language understanding, code-side validation, human handoff). It is a hypothesis until Phases 0-1 are built and tested with real conversations.

## 3a. Why LiveKit Agents (and what it costs)

Chosen by you over Pipecat. What it gives us here:
- **Telephony is first-class:** LiveKit SIP bridges phone calls into a room; an inbound trunk plus a dispatch rule sends each call to the agent. Phase 4 gets simpler than with a generic transport.
- **Ready browser client and SDKs** for the test console (§8a).
- **Plugin model** for custom STT/TTS; OpenAI plugin with custom `base_url` for Qwen.

What it adds:
- **A LiveKit server must run** (LiveKit Cloud, or self-hosted `livekit-server`). Self-hosted SIP also needs the LiveKit SIP service, Redis, and a public IP reachable by the SIP provider. For development, run `livekit-server --dev` locally (verify the current flag in the docs).
- The SIP docs I found list SIP as **Developer Preview**; check the current status before promising a production date.
- If the STT turns out to be per-clip, it goes through the VAD + `StreamAdapter` path, so end-of-speech wait is part of latency; a streaming STT route would avoid this.
- **Decision: self-hosted.** Patient data and audio stay on our infrastructure. Consequences are listed in §3b.

Mitigation for lock-in: flow, normalizers, tools, prompts and the mock backend stay framework-free; only `agent.py` and `services/` touch LiveKit.

Sources checked: [LiveKit plugins guide](https://docs.livekit.io/agents/plugins), [models overview](https://docs.livekit.io/agents/models.md), [SIP overview](https://docs.livekit.io/sip), [self-hosting](https://docs.livekit.io/transport/self-hosting/).

## 3b. Self-hosted LiveKit: what we run

| Component | Purpose | Dev | Production pilot |
|---|---|---|---|
| `livekit-server` | WebRTC media/signalling, rooms | `livekit-server --dev` on the Mac (dev keys) | Own host, real API key/secret, TLS (domain + certificate), TURN/UDP ports open |
| LiveKit SIP service | Bridges phone calls (SIP) into rooms | Not needed (use browser) | Needs public IP, SIP and RTP port ranges open to the provider |
| Redis | Required by SIP service and multi-node setups | Not needed in dev | Redis next to the server |
| Agent worker (ours) | Runs the bot; registers with the server and is dispatched per call | Local process | Systemd/Docker service on the same network as the STT/TTS/LLM endpoints |
| Test UI server (ours) | Issues room tokens, serves the console | Local | Internal network only |

Self-hosted notes:
- The agent talks to the server over its WebSocket URL with an API key/secret; keep them in `.env`, never in the repo.
- Phase 4 needs a host with a **public IP and open ports** reachable from the SIP provider. Not possible from a laptop behind NAT, so the pilot host must be decided (see §11).
- Behind NAT the browser may need a TURN server; on the local network, `--dev` works without one. Check the LiveKit docs for the current config keys before deploying.
- We own upgrades, monitoring and restarts. Add basic health checks and log rotation in Phase 5.
- Docker Compose file for server + Redis + SIP + agent is a Phase 4 deliverable (`deploy/`).

## 3c. Many workflows, caller can talk about anything at any time

v1 builds two workflows with switching between them (plus FAQ and lookup answers). The pieces below (manifests, workflow stack, switch/resume) are all in v1; embedding-based shortlist retrieval is added when workflows exceed ~5. Callers do not follow menus, so the design assumes **any utterance can belong to any workflow, to none, or to the current one.**

**Structure**
- **Workflow registry:** each workflow is a folder with `manifest.yaml` (name, one-line description, 10-20 example utterances in Roman Urdu, required verification level, can-interrupt rules), plus its slots, states, prompts, tools and evals. Adding a workflow = adding a folder; nothing in the core changes.
- **Session manager:** holds a stack of active/suspended workflows, shared facts (verified identity, phone, name), and conversation history.
- **Router (runs every turn, before the active workflow):**
  1. *Cheap pass:* is this clearly an answer to the bot's last question? If so, continue without calling the router LLM (fast, and the safest default).
  2. *Candidate retrieval:* match the utterance against all manifests' descriptions and examples (keyword/lexicon + embedding or fuzzy match) to shortlist the top 3-5 workflows. This keeps the router prompt small however many workflows exist.
  3. *LLM decision:* Qwen picks from the shortlist or returns one decision from the table in §3e. Output is constrained JSON with a confidence value.
  4. *Policy in code:* low confidence → one clarifying question; a switch away from an active flow with data collected → one-line confirm; sensitive targets → verification first.
- **General layer (no workflow):** greetings, small talk, "lab kab tak khulti hai", simple FAQ. Answered from a curated FAQ file (retrieval, no free invention) and then steered back. Anything not in the FAQ and not a workflow → human, never guessed.

**Rules**
- Bias to `continue`; switches need clear intent. STT is ~66% accurate, so a misheard word must not abandon a booking.
- Suspend, never discard; offer resume after the new flow ends. Max 2 suspended flows, then offer human.
- A workflow can mark states as non-interruptible (final read-back, submit).
- Verification is a reusable sub-workflow, triggered by any workflow that needs it.
- The caller can always say "insaan se baat karao" → human, from any state.

**Honest limits**
- Qwen3-8B will misroute more as workflows multiply and overlap (e.g. "report" vs "result" vs "complaint about report"). Retrieval shortlisting and per-workflow example utterances help; **routing accuracy is a tracked metric with its own eval set**, and growth in workflows is gated on it staying above target (proposed ≥ 90% on noisy text).
- If the 8B router is the weak link, only the router can be moved to a larger model; workflows are unaffected.
- Open-ended talk is where hallucination risk lives. The rule: the bot answers only from workflows, tools or the FAQ file; otherwise it says it can't help and offers a human.
- Each workflow still needs its own backend API, catalog and 30+ scenarios; the framework removes plumbing, not data work.

**Adding a workflow (checklist and exit criteria)**
1. Write the goal, slots, final action and whether it needs verification.
2. Create `workflows/<name>/` with `manifest.yaml` (description, 10-20 example utterances, can-interrupt rules), states, slots, per-state prompts, tools, scenarios.
3. Add or reuse data lists (`data/*.csv` + `config/entities/*.yaml`), then run `scripts/check_data.py`.
4. Connect the backend API (mock first).
5. Add 30+ scenarios, including switching to and from every existing workflow.
6. Add its example utterances to the routing eval set and re-run it. **Exit:** routing accuracy ≥ 90% and false-switch ≤ 1% still hold for all workflows, switching scenarios pass, TTS pronunciation reviewed. If routing drops: add examples, merge overlapping workflows, or retrain the router.

**Rollout order:** home sampling first, then the second workflow together with the stack, switch and resume, then the router bake-off. Later workflows are added one at a time, re-running the routing and switching evals each time.

## 3d. Router model: Laya (primary) with fallbacks

The router (§3c) sits behind one interface so the model can be swapped without touching workflows:

```python
route(utterance, context) -> {decision: continue|faq|lookup|switch|resume|unclear|human|end,
                              workflow: str|None, confidence: float}
```

**Decision: Laya is the primary router, Qwen3-8B is the runtime fallback. Both are chosen in `.env` (`ROUTER_PRIMARY`, `ROUTER_FALLBACK`).**

**Primary: Laya multilingual** (open-source, Apache-2.0, non-autoregressive "System 1" decision encoder; `laya-multilingual`, mmBERT-base, ~322M params, locally hosted, runs on CPU or GPU).
- Outputs a choice over labels with a confidence in one forward pass, with no text generation. About 30 ms on a GPU and about 120-450 ms on CPU/Apple GPU per the README (self-reported, unverified).
- Fine-tuned on our own routing data, then **temperature-calibrated** (the multilingual checkpoint ships without fitted temperatures, so its confidence is not trusted until we do this).
- Runs as a small local FastAPI service next to the agent (its own server/playground can be reused); no patient data leaves our infrastructure.

**Unknown to verify first: Roman Urdu.** Laya's language list most likely covers Urdu script, not informal Latin-script Roman Urdu. Zero-shot quality is untested; assume it is weaker and rely on fine-tuning. Repo numbers (stars, benchmarks) came from page summaries and are unverified.

**Bake-off (Phase 2b, ~1 day; runs in v1 once both workflows work in chat)**
- Eval set: ~100-200 labelled routing utterances in Roman Urdu from the 80 real calls plus synthetic noisy variants and hard cases (overlapping intents, short answers like "haan", misheard words, code-switching).
- Candidates, same data and metrics:
  1. Laya multilingual, zero-shot
  2. Laya multilingual, fine-tuned and calibrated
  3. Qwen3-8B router (JSON constrained, thinking off)
  4. Embedding shortlist only (baseline)
- Metrics: routing accuracy, **false-switch rate** (must be near zero), calibration (confidence vs accuracy), p50/p95 latency, memory.
- Decision rule: pick the model with ≥ 90% accuracy on noisy text and false-switch ≤ 1% at the chosen confidence threshold; ties go to the lower latency.

**Runtime fallback (every turn):** Laya answers first. The Qwen3-8B router takes over for that turn when Laya
- errors or exceeds `ROUTER_TIMEOUT_MS` (`error`, `timeout`), or
- returns confidence below `LAYA_CONFIDENCE_MIN` (`low_confidence`).

`ROUTER_FALLBACK_ON` lists which of these triggers are active. If both disagree on a *switch*, the bot does not switch; it asks the caller (one clarifying question). Because untuned Laya may be weak on Roman Urdu and uncalibrated, the starting threshold is deliberately high, so early on many turns fall through to Qwen; the fallback rate is tracked and the threshold re-tuned after calibration and fine-tuning.

**Configuration (all in `.env`, see `.env.example`):** `ROUTER_PRIMARY`, `ROUTER_FALLBACK`, `ROUTER_FALLBACK_ON`, `ROUTER_TIMEOUT_MS`, `LAYA_MODEL` (stock checkpoint or path to our fine-tuned one), `LAYA_DEVICE`, `LAYA_CONFIDENCE_MIN`, `LAYA_TEMPERATURE`, `ROUTER_QWEN_CONFIDENCE_MIN`, `ROUTER_SHORTLIST_SIZE`, `ROUTER_SWITCH_CONFIDENCE_MIN`, `ROUTER_LOG_DECISIONS`. Setting `ROUTER_PRIMARY=qwen` and `ROUTER_FALLBACK=none` reverts to Qwen-only with no code change.

**Bake-off result (run 2026-10-22 on this repo's routing set, 149 labelled Roman Urdu utterances, CPU only):**
- The cheap rule pass decides 91 of 149 utterances (replies to pending questions, human, goodbye, clear FAQ). The other 58 need a model.
- **Laya zero-shot (multilingual checkpoint) is not usable:** 16% accuracy with the first criteria text, 22% with Roman Urdu example phrases in the criteria; 0 calls reach the 0.8 confidence bar; 1.3 s to 5.8 s per call on this Intel CPU (budget 0.4 s). It needs fine-tuning and calibration on routing data (v2).
- Rule baseline: 99.3% on the set. **Caveat:** the rules were tuned while building this set, so the number is optimistic; a held-out set of real utterances is needed.
- Qwen3-8B router: not yet evaluated (no LLM endpoint configured).
- **Decision (plan ladder step B): default `ROUTER_PRIMARY=qwen`, `ROUTER_FALLBACK=none`, rule baseline always last.** The Laya adapter stays in the code and can be re-enabled by config after fine-tuning.

**Fallback order if Laya does not work out (offline decisions after the bake-off)**
| Step | Approach | When |
|---|---|---|
| A | Laya zero-shot fails but fine-tuned works | Ship the fine-tuned one; keep collecting logged routing decisions to retrain. |
| B | Fine-tuned Laya still below target | Use **Qwen3-8B router** (already running) with embedding shortlist; add Laya as a confidence pre-filter later if useful. |
| C | Qwen3-8B too slow or inaccurate on overlapping intents | Fine-tune a **small Qwen (0.6-1.7B) or another multilingual encoder** (mBERT/XLM-R class) as a classifier on the same labelled data, same `route()` interface. |
| D | Everything below target for a workflow pair | Merge or restructure those workflows (e.g. one parent workflow with sub-intents) or confirm with the caller before switching. |
| E | OpenJev (serving layer, Jev-compatible API) | Optional later wrapper if we want one standard decision API; its default 26B backbone is too heavy for the router, so only its small encoders are candidates. Verdict (Heman10x) is English-only with weak reported results, so it is not a candidate. |

Hosted Jev (TypeSafe) is excluded for patient traffic because it is not self-hosted; usable only for tests on non-patient utterances.

**Data loop (starts in Phase 1, used by the bake-off and retraining):** every routing decision is logged with the utterance (redacted), context, chosen route, confidence and any human correction. Corrections become labelled training data.

## 3e. Response types, router decisions and the full flow map

**The bot answers in two ways:**
| | Workflow | FAQ answer |
|---|---|---|
| Purpose | Get a task done | Answer a question |
| Example | Book a home sample | "Lab kitne baje khulti hai?" |
| Length | Many turns, collects slots | One turn |
| Data | Slots, tools, backend API | Curated `config/faq.yaml` |
| Ends with | Submitted request + reference number | The answer, then back to the pending question |
| Third path | Human handoff for sensitive topics (interpreting results, complaints, refunds). The bot never improvises; it speaks only from workflows, tools or the FAQ file. | |

**Router decisions (output of `route()` on every turn):**
| Decision | Meaning | Bot action |
|---|---|---|
| `continue` | Caller answers the bot's last question | Pass to active workflow (cheap pass, no router LLM call) |
| `faq` | A stored question | Answer from `faq.yaml`, return to the pending question |
| `lookup` | A question answered from a data list (e.g. "vitamin d ki price kya hai?") | Resolve the item with datakit (asking the caller to choose if ambiguous), speak the fact (price etc.), return to the pending question. No stack needed. |
| `switch` | Clear request for a different task | Confirm if a half-filled workflow would be left, suspend it, start the new one |
| `resume` | Caller wants the suspended workflow back | Restore its slots and continue |
| `unclear` | Cannot tell | One clarifying question; after two unclear turns offer a human |
| `human` | Caller asks for a person, sensitive topic, or bot cannot help | Transfer or book a callback |
| `end` | Caller is done ("bas, shukriya") | Close politely and hang up |

**Flow map:**
```
caller speaks → STT → corrector → ROUTER (Laya first, Qwen fallback)
   ├─ continue ─► active workflow ─► slot extraction ─► reply ─► (done) submit ─► closing
   ├─ faq ──────► faq.yaml answer ─► back to pending question
   ├─ lookup ───► datakit.resolve ─► (ask which one?) ─► datakit.facts ─► speak ─► back to pending question
   ├─ switch ───► confirm? ─► suspend current ─► new workflow ─► done ─► offer resume
   ├─ resume ───► restore suspended workflow
   ├─ unclear ──► one clarifying question (2 strikes → human)
   ├─ human ────► transfer / callback
   └─ end ──────► closing
```

**Exits not decided by the router (workflow or code):** two failed attempts on one slot → human; caller rejects the final read-back → re-ask the wrong slot; STT/TTS/LLM endpoint failure → "system problem" message + callback offer; silence → reprompt, then hang up.

**Safety limits:** lean toward `continue` (STT is ~66% accurate); leaving a half-filled workflow needs `ROUTER_SWITCH_CONFIDENCE_MIN`; final read-back and submit are non-interruptible; at most 2 suspended workflows.

## 3f. FAQs and workflow discovery

The FAQ list is used three ways:
1. **Classify each FAQ** into one of four types:
   | Type | Test | Example | Goes to |
   |---|---|---|---|
   | Static answer | Same answer for every caller | opening hours, home-sampling charge | `faq.yaml`, no workflow |
   | Lookup | Needs caller or day data | report ETA, test price | Small workflow or tool-backed answer (1-2 slots + API) |
   | Action | Caller wants something done | resend report, cancel booking | Full workflow |
   | Sensitive / judgment | Medical interpretation, complaints, refunds | "meri report mein kya abnormal hai?" | Human handoff |
2. **Static FAQs** become `config/faq.yaml`: spoken-length Roman Urdu answers (≤ 2 sentences), several spelling variants of each question, `volatile: true` for prices/timings that must come from a data source instead of a fixed answer.
3. **All FAQ questions** (with rephrasings) become router training and eval examples. Lookup and Action FAQs ranked by call frequency give the order in which workflows are built. Overlapping FAQs are flagged as likely router confusions.

**FAQ entry format (`config/faq.yaml`)**
```yaml
- id: faq_home_sampling_charge
  category: home_sampling
  questions:                       # phrasings and spelling variants, incl. real STT errors
    - "home sampling ka charge kitna hai"
    - "ghar se sample dene ki fees"
  answer: "Home sampling ka charge {home_sampling_fee} rupay hai."   # spoken length, max 2 sentences
  data: {home_sampling_fee: pricing.home_sampling}   # filled from a data list, never typed in
  owner: ops
  status: approved                 # only approved entries are spoken
```
**Matching:** narrow by category, then hybrid keyword/lexicon + fuzzy match with two thresholds: high score and a clear gap to the second match → answer; middle → confirm ("Kya aap home sampling ke charge ke baare mein pooch rahe hain?"); low or tie → "mujhe is ke baare mein pakka maloom nahi", offer a human. The bot reads the stored answer word for word.
**Load-time checks (`check_data.py`):** duplicate or near-duplicate questions across FAQs, answers longer than two spoken sentences, unapproved entries, missing data placeholders.
**Eval:** every FAQ's phrasings plus STT-style variants become test cases; wrong-answer rate must stay near zero. Unmatched and wrong questions are logged and added by hand as new phrasings or new FAQs.

Input needed from you: the FAQ list (questions and answers if available) and, if possible, which questions callers ask most (e.g. from the 80 call transcripts).

## 3g. Entity resolution: searching and confirming against data lists

Callers refer to things that live in our data: tests, test packages, areas, branches, doctors, slots. STT mangles the names, and one spoken name can match several entries. This is **retrieval over structured lists with a spoken disambiguation dialogue**, not generative RAG. Prices and other facts always come from the data or API, never from the LLM.

### One generic resolver for every list
Adding a new list = adding a config file and a data source. No new code per list.

```yaml
# config/entities/test.yaml
name: test
source: {type: csv, path: data/tests.csv, refresh: daily}   # csv | json | api | db
id_field: test_id
display_field: name_roman          # what the bot says
alias_fields: [aliases_roman, aliases_urdu, abbreviations]
attributes: [sample_type, department, fasting, price]
disambiguate_by: [sample_type, department]   # attributes used to tell candidates apart
search: {lexicon: true, phonetic: true, fuzzy: true, embeddings: false}
thresholds: {accept: 0.90, gap: 0.15, confirm: 0.70, max_choices: 3, max_rounds: 2}
confirm_before_use: false          # true for costly / side-effecting use
spell_fallback: true
```

Entity types planned: `test`, `package`, `area`, `branch`, `doctor`. **Identifier lists** (MR number, booking reference) are a different kind: exact match plus checksum/format validation, no fuzzy matching, read back digit by digit.

API: `resolve(entity_type, text, context) -> {status: resolved|confirm|ambiguous|none, candidates:[{id, name, score, differs_by}]}`.

### Outcomes and the spoken dialogue (shared `resolve_entity` sub-flow)
| Result | Bot behaviour |
|---|---|
| `resolved` (score ≥ accept and gap to #2 ≥ gap) | Use it; confirm only if `confirm_before_use` |
| `confirm` (one candidate, score between confirm and accept) | "Kya aap CBC test ke baare mein pooch rahe hain?" |
| `ambiguous` (2-4 close candidates) | Name the options (max `max_choices`), stating the difference that matters, caller picks: "Vitamin D ke do tests hain: Vitamin D total aur Vitamin D 25 hydroxy. Kaunsa chahiye?" |
| still ambiguous | Ask one distinguishing attribute ("khoon ka test ya peshab ka?"), then ask to **spell the name** (letters are normalised and re-searched) |
| `none` | "Test ka naam dobara bata dein, ya spelling bata dein." After `max_rounds` failed rounds → human |

Choice answers are understood as "pehla/doosra wala", the option's name, a distinguishing word ("total wala"), or "dono" (both, if the parent flow allows).

### How it connects to the router and session
- The sub-flow stores a `pending_question = choose_one_of [A, B]` in the session.
- The router's cheap pass treats an answer to a pending question as `continue` and the resolver maps it to the candidate. No router model call is needed.
- If the caller says something unrelated instead ("chhodein, booking karni hai"), the router returns `switch` and the pending question is dropped or suspended with its workflow.
- Any workflow can call `resolve_entity`. Example flows: `lookup` for test price/info (resolve test → read price from data), `home_sampling` (resolve each test and the area).

### Scaling the data
*v1 uses the small `datakit` in §3h (CSV + in-memory alias/fuzzy/phonetic matching, a data check script, per-list eval). The items below are the growth path; each is added only when needed (see the deferred table in §3h).*
- **Sources are pluggable:** CSV/JSON files for v1, then API or DB. A loader normalises each source into the same in-memory record format.
- **Index built offline** (later; not in v1): lexicon and alias tables, phonetic keys, n-gram index, optional embeddings. Versioned (`data/index/<entity>/<version>`), hot-reloaded on refresh, previous version kept for rollback.
- **Size:** in-memory fuzzy + phonetic is fine for a few thousand entries. Beyond about 10-20k entries or for fast embedding search, move to SQLite FTS5 / a search engine / pgvector behind the same `search` interface.
- **Data quality report on every load:** duplicate or near-identical names, entries without Roman Urdu aliases, entries with identical aliases pointing to different ids. Bad data blocks the index build.
- **Alias learning:** unresolved or corrected utterances are logged (PII redacted). A reviewer approves new aliases, which go into the data and are re-indexed. This is the main way accuracy grows over time without code changes.
- **Per-list eval sets** (`eval/entities/<entity>.yaml`): spoken variants, STT-style errors, ambiguous cases. Metrics: correct-resolution rate, **wrong-resolution rate (must stay near zero)**, correct-refusal rate, average rounds needed. Re-run when data or thresholds change.
- **Voice constraints:** max three options per question, option names must be pronounceable by the TTS (pronunciation lexicon applies).
- **Config over code:** thresholds, attributes used to disambiguate, and confirmation rules are per entity in YAML and can be overridden by env (`ENTITY_*`).

## 3h. The data module (`datakit/`): small in v1, same interface later

`datakit/` is a small, self-contained folder that knows nothing about LiveKit, the router or workflows. Workflows call its three functions only. It is deliberately **not** a platform in v1.

```
app/datakit/
  loader.py      # read CSV/Excel into records
  resolve.py     # alias + fuzzy + phonetic match, thresholds, ambiguity result
config/entities/*.yaml  # one config per list: columns for name, aliases, facts, disambiguation
data/*.csv       # the lists
scripts/check_data.py   # duplicates, missing aliases, bad ids
```

**Interface (kept stable so the inside can change later):**
| Call | Returns |
|---|---|
| `resolve(entity, text, context)` | status (`resolved\|confirm\|ambiguous\|none`) + ranked candidates with scores and `differs_by` |
| `get(entity, id)` | the full record |
| `facts(entity, id, fields)` | only the requested fields, ready to speak or inject into a prompt |

**Adding a new list (no code change):**
1. Put `data/<list>.csv` in place (needs an id and a name; aliases and facts are optional but improve matching).
2. Add `config/entities/<list>.yaml` (source, id/display/alias columns, `facts`, `disambiguate_by`, thresholds).
3. Run `scripts/check_data.py <list>` (and optionally the entity eval), then restart.

**Adding a new workflow** is separate: a new folder under `workflows/`. A workflow that needs a new list does both; nothing else links them.

### Deliberately deferred (add only when it hurts)
| Deferred | Add when |
|---|---|
| Service mode (`datakit` behind HTTP) | The agent and the data need to scale or deploy separately |
| Other search backends (SQLite FTS, search engine, pgvector), embeddings | A list passes ~10k rows, or fuzzy/alias matching measurably fails |
| Index versioning, rollback, hot reload | Rebuild time or live editing becomes a problem |
| Alias-learning pipeline from logs | There are enough real calls to learn from (until then, add aliases by hand) |
| Cross-list links and `related()` | The data actually has references (package → tests, area → branch) |
| Data tab in the console | Lists are edited often (a script and a spreadsheet are enough first) |

Cheap things done now so nothing needs rewriting: the three-function interface, thresholds in config, data in files (never in prompts), and a per-list eval set.

## 4. Qwen3-8B specifics

- LiveKit's OpenAI plugin takes a custom `base_url`; confirm it uses the chat-completions mode your server implements (vLLM and llama.cpp differ on newer APIs).
- Serve with vLLM (or llama.cpp server, which is already in this machine's workspace) with an OpenAI-compatible endpoint. If using tool calls, vLLM needs `--enable-auto-tool-choice --tool-call-parser hermes`.
- **Disable thinking mode** for latency: pass `chat_template_kwargs: {"enable_thinking": false}` (or `/no_think`). Thinking tokens would blow the latency budget.
- Prefer **structured extraction over free tool calling**: ask the model for a small JSON object per turn (`{"slots": {...}, "intent": "...", "reply": "..."}`) with guided/JSON-schema decoding. This is more robust on an 8B model than multi-tool function calling, and easier to eval. Native tool calls are kept only for the final `create_home_sampling_request` and `transfer_to_human`.
- Temperature 0.2-0.3. Max reply 40 tokens (spoken, ≤ 2 sentences).
- Prompt layout (English rules, Roman Urdu examples), per-state: only the instructions and examples for the *current* state plus the slot table, not one giant prompt. Smaller context = faster and less drift.
- Few-shot examples should come from your 80 real call transcripts (the benchmark workbook), with real STT errors included.

---

## 5. Tools / backend interface (all English)

| Tool | Type | Purpose |
|---|---|---|
| `resolve_entity(type, text)` | code (see §3g) | One resolver for every list (`test`, `area`, `branch`, `doctor`...). Maps "sugar", "shugar", "cbc", "vitamin di" → catalog entries with scores; ambiguous → the bot asks. `match_tests` and `check_coverage` are thin wrappers over it. |
| `datakit.facts(entity, id, fields)` | code | Price/fasting/etc. for a resolved item (used by `lookup` and by the home-sampling fasting reminder). |
| `get_slots(date)` | backend | Available sampling windows. |
| `create_home_sampling_request(payload)` | backend | Final submit. Only reachable after `confirmed=true` in flow state. Returns reference id. |
| `transfer_to_human(reason)` | telephony | Warm transfer or callback request. |
| `end_call()` | transport | Hang up after closing. |

Code-side normalizers (never trust the LLM for these): `normalize_phone` (digit words → digits, "double", "triple", Urdu/Roman numerals), `normalize_number` (age, house no.: "painteesh" → 35), `resolve_date` ("kal", "parson", "agle somwar" → ISO, Asia/Karachi), `normalize_name` (title-case, keep as heard), `canonicalize_test_name`.

**TTS pronunciation:** keep a `config/pronunciation.yaml` lexicon (test names like CBC/HbA1c, area names, abbreviations) applied in `script_adapter` before text goes to TTS, and review spoken output for the top 50 terms in Phase 3.

Domain knowledge to reuse from this repo (`qwen3-asr-local/`): the Chughtai lexicon, the `corrector.py` / `resolver.py` phonetic correction, and the glossary. These already target lab-specific vocabulary (test names, areas). Run them on STT output *before* the LLM sees it, then check they actually help on test names.

---

## 6. Handling bad STT (the core robustness design)

1. **Read-back confirmation for every high-risk slot**, in a way that is easy to say "no" to: name, address, phone, tests, date/time. One consolidated read-back at the end plus a per-slot confirm for phone and address.
2. **Phone number:** use caller ID when present ("Kya aap isi number par rabta chahte hain?"). Otherwise collect in two chunks and read back digit by digit.
3. **Catalog snapping:** tests and areas are matched to closed lists, so mistakes become "pick the closest option" questions ("CBC, yaani complete blood count, theek hai?").
4. **Two-strikes rule:** a slot that fails or is rejected twice → rephrase once, then offer human transfer or callback. No infinite loops.
5. **ASR confidence/empty handling:** empty or very short transcripts → "Maaf kijiye, awaaz saaf nahi aayi, dobara bata dein."
6. **Names are never "corrected"**: stored as heard, flagged `needs_human_verification` unless the caller confirmed. Human callback verifies in v1.
7. **DTMF fallback for phone numbers (telephony):** after two failed spoken attempts, ask the caller to type the number on the keypad. Needs DTMF support from the SIP path (check in Phase 4).
8. **Log everything** (audio, raw STT, corrected text, LLM output, slots per turn) so failures can be replayed offline.

---

## 7. Latency budget (target: caller stops → bot starts speaking)

| Stage | Budget | Notes |
|---|---|---|
| VAD end-of-speech wait | 400-600 ms | Tune `stop_secs`; too low cuts people off mid-sentence. |
| STT (websocket; streaming or per-utterance) | 300-1200 ms after end of speech | Depends on the endpoint and clip length. **Measure first.** |
| Correction/normalizing | < 50 ms | Local code. |
| LLM first token (Qwen3-8B, no think) | 200-500 ms | Depends on GPU; streaming to TTS. |
| TTS first audio | 300-800 ms | Sentence-level streaming if supported. |
| **Total** | **≈ 1.5-3 s** | Mitigate with a spoken filler during slow steps and by streaming LLM → TTS sentence by sentence. |

Barge-in: enable LiveKit's interruption handling; the TTS output must be cancellable. Verify that your TTS endpoint can be cut off cleanly.

---

## 8. Project layout (to be created in `voicebot/`)

```
voicebot/
  PLAN.md
  README.md
  pyproject.toml            # uv or pip; Python 3.11+
  .env.example              # STT_URL, TTS_URL, LLM_URL, LLM_MODEL, LIVEKIT_URL/API_KEY/API_SECRET, TZ
  config/
    settings.yaml           # VAD params, timeouts, retries, thresholds
    prompts/                # one file per flow state (English rules + RU examples)
    entities/*.yaml         # one config per searchable list (test, area, branch, doctor...), see §3g
  app/
    agent.py                # LiveKit AgentSession + worker entrypoint
    services/stt_ws.py      # LiveKit STT plugin (WebSocket) -> QWEN_VLLM_BASE_URL
    services/tts_ws.py      # LiveKit TTS plugin (WebSocket) -> OMNI_TTS
    services/script_adapter.py
    flow/home_sampling.py   # states, transitions, slots
    flow/extractor.py       # LLM slot extraction (JSON schema)
    nlp/normalizers.py      # phone, numbers, dates, names
    nlp/asr_corrector.py    # wraps lexicon / phonetic correction
    tools/                  # tools used by workflows (wrappers over resolve_entity, backend calls)
    datakit/                # small data + resolver module (see §3h): loader, resolve
    flow/resolve_entity.py  # the spoken disambiguation sub-flow; calls datakit
    flow/lookup.py          # lookup answers (test price/info) and faq answers
    backend/mock_api.py     # swap for the real booking API later
  ui/                       # test console (see §8a)
    index.html              # single page, no build step
    app.js
    style.css
  router/
    interface.py            # route() contract
    laya_router.py          # Laya multilingual (primary)
    llm_router.py           # Qwen3-8B JSON router (runtime fallback)
    chain.py                # primary -> fallback policy driven by .env
    shortlist.py            # (later, >5 workflows) manifest/embedding candidate retrieval
    train/                  # fine-tune + calibrate notebooks/scripts
  workflows/                # one folder per workflow: home_sampling/, booking_status/ (each with manifest, states, prompts, scenarios)
  app/session.py            # session manager: workflow stack, suspend/resume, shared facts
  data/
    tests.csv, areas.csv ... # source data (or API/DB in production)
  config/faq.yaml           # curated FAQ answers (spoken length, placeholders from data)
  eval/
    entities/*.yaml         # per-list resolution eval sets
    scenarios/*.yaml        # scripted caller scenarios (also listed in the UI picker, see §8a)
    audio/                  # optional caller clips for audio auto-play
    run_text_eval.py        # text-only (no audio), fast loop
    run_entity_eval.py      # resolver accuracy per list
    run_router_bakeoff.py   # Laya vs Qwen3-8B vs baselines on the routing set
    run_audio_eval.py       # audio in -> STT -> flow, WER + slot accuracy
    noise/                  # real ASR outputs used as noisy inputs
  scripts/
    check_data.py           # validate data lists (duplicates, missing aliases, bad ids)
    ping_endpoints.py       # latency + sanity check for STT/TTS/LLM
    new_workflow.py         # scaffold a workflow folder (end of v1)
    check_workflow.py       # validate + run scenarios + routing/switching checks (end of v1)
    run_dev.sh              # starts livekit-server (dev), the agent worker, and the UI server (http://localhost:7860)
  deploy/                   # docker-compose + livekit.yaml, sip.yaml (Phase 4)
  logs/                     # per-call JSONL (gitignored)
```

---

## 8a. Test UI (web console for live calls and conversations)

A single web page served by the bot server at `http://localhost:7860`, so any call can be tried and watched without a phone. Plain HTML/JS (no build step) to start.

**Two input modes, same pipeline:**
- **Type mode:** type Roman Urdu as if it were the STT output. Works from Phase 1, needs no STT/TTS. Also a "paste noisy STT line" box to replay real ASR errors.
- **Voice mode:** Start Call / End Call button, the page joins a LiveKit room with `livekit-client` (JS SDK) using a token from our server, and the agent joins the same room. Bot audio plays back in the page, with mute and a barge-in test.

**What the page shows while the call is going on (live, via a WebSocket event stream from the pipeline):**
| Panel | Content |
|---|---|
| Conversation | Chat-style turns: caller (raw STT → corrected text) and bot (Roman Urdu and, if used, the Urdu-script text sent to TTS). Interim/partial text while speaking. |
| Flow state | Current state (e.g. `ASK_ADDRESS`), retry counts per slot, transition history. |
| Slots | Table of the booking being built: value, confidence, confirmed yes/no. Highlights changes each turn. |
| Tool calls | Each call with args, result and duration (`resolve_entity`, `create_home_sampling_request`). |
| Timings | Per-turn bar: VAD wait, STT, corrector, LLM first token, TTS first audio, total. Running median/p95. |
| Raw I/O (collapsible) | Exact prompt sent to Qwen and its raw JSON reply, STT endpoint response. |
| Controls | Pick scenario/persona, reset session, pause bot, force transfer-to-human, change settings live (VAD stop time, temperature), switch mock backend responses (e.g. "area not covered"). |
| Call history | List of past sessions (from `logs/`), click to replay the transcript with timings and download the audio and JSONL. |

**Scenario picker (sample conversations you can pick and run, by chat or voice):**

The same scenario files used by the automated evals (`eval/scenarios/*.yaml`) are listed in a **Scenarios** side panel, grouped by tag, so what you test by hand is what the test suite checks.

| Mode | What happens |
|---|---|
| **Guided (voice or chat)** | A script card shows the caller's next line, what the bot should do (expected state/slot/tool), and a live check mark per step. You say it into the mic or click "send" / edit and send it as chat. Good for trying real accents, noise and your own wording around a scenario. |
| **Auto-play (chat)** | The scripted caller lines are fed to the bot automatically, one per turn, at a speed you set. Shows pass/fail at the end. |
| **Auto-play (audio)** | Each caller line is spoken into the call as audio, either from a recorded wav/real call clip if the scenario has one, or generated with the TTS endpoint as a synthetic caller. Exercises VAD, STT and barge-in without a person. |
| **Free talk** | No script; the scenario list is ignored. |

Per scenario the page shows: expected vs actual final booking payload, which steps passed, the router decision each turn (and which router answered), and a replay link. A **"Run all"** button runs a tag (e.g. `smoke`, `noisy`, `switching`) in auto-play chat mode and shows a results table, so a prompt or threshold change can be re-checked in one click.

Scenario file shape (shared with the evals):
```yaml
id: hs_happy_path
title: Home sampling, everything clear
tags: [smoke, home_sampling]
persona: {name: Muhammad Ali, phone: "0300..."}
steps:
  - say: "sugar aur cbc karwana hai ghar pe"
    expect: {router: continue, state: ASK_PATIENT, slots: {tests: [glucose, cbc]}}
  - say: "muhammad ali, umar painteesh saal"
    expect: {slots: {name: Muhammad Ali, age: 35}}
  - say: "haan"
    expect: {tool: create_home_sampling_request}
audio: {mode: tts}          # tts | clips | none
final: {status: submitted}
```

Starter scenarios to ship (all selectable in the picker):
| Tag | Scenarios |
|---|---|
| `smoke` | Home sampling happy path; caller gives everything in one sentence; caller corrects a detail ("nahi, kal nahi parson") |
| `entities` | Ambiguous test name (bot lists options, caller answers "pehla wala"); test not found then spelled out; area not covered; two similar areas |
| `lookup` | "vitamin d ki price kya hai?" (ambiguous → choose → price); price question in the middle of a booking and return to the pending question; unknown test; FAQ in the middle of a booking |
| `routing` | FAQ in the middle of a booking and return; unclear request; "insaan se baat karao"; caller says "bas shukriya" |
| `switching` | A → B → resume A ("pehle meri booking ka status batao" in the middle of a new booking); B → A; switch declined ("nahi, booking hi karni hai"); false switch from a misheard word (must stay in A); switch attempted during the non-interruptible read-back (must wait); two switches (stack depth 2) and a third (offer human); shared name/phone not asked twice; slots of the suspended workflow still intact after resume; caller abandons the suspended workflow |
| `noisy` | Same flows with real STT errors from the benchmark transcripts; silence; very short answers; code-switching |
| `safety` | Refuses to confirm read-back; two failed attempts then human; abusive caller; medical-advice question (must go to human) |
| `failures` | STT, TTS, or LLM endpoint down (simulated switch in the UI): bot should apologise and offer callback |

Scenarios for new workflows are added to the same folder by the person adding the workflow and appear in the picker automatically.

**Implementation notes:**
- The agent emits structured events (`transcript`, `slots`, `state`, `tool`, `timing`, `llm_raw`) into one `EventBus`, using `AgentSession` events plus our own hooks; the UI and the JSONL logger both subscribe, so what you see equals what gets logged. Events reach the page over a LiveKit data channel (or a plain WebSocket in type mode).
- Server: FastAPI (`/api/token` for LiveKit room tokens, `/ws/events`, `/api/session`). One room per browser tab for now.
- LiveKit ships an Agents Playground and starter web frontends for quick checks; we use our own page because it must show slots, state and timings. Verify what the current LiveKit starters provide before building from scratch.
- Dev-only: bind to localhost, no auth in v1, and redact PII toggle for shared screens.

## 9. Build phases

**Phase 0: Endpoint reality check (0.5 day)**
- Scaffold the repo (§8): `pyproject.toml`, dependencies pinned, `README.md`, `scripts/`, empty package layout. Pin the LiveKit Agents version and confirm its STT/TTS plugin interfaces and the `livekit-server --dev` flags against the installed docs/source.
- Endpoints and env vars are in `.env.example` (STT and TTS are WebSockets; LLM URL still to be provided).
- `ping_endpoints.py`: send known audio to STT, a sentence to TTS, a prompt to the LLM. Record latency, formats (sample rate, streaming or not, input script for TTS), error behavior.
- Decide the TTS script question (see §11) with real samples: Roman Urdu input vs Urdu script, listen to both.

**Phase 1: Workflow A, data and test UI in type mode (3 days)**
- Include the scenario picker in the console (Guided and Auto-play chat) with the starter scenarios from §8a.
- Wire the router chain (Laya primary, Qwen fallback) behind `route()` and show which router answered each turn in the UI. Decisions needed by A: `continue`, `faq`, `lookup`, `unclear`, `human`, `end`. First check how stock Laya does on a small Roman Urdu sample; if poor, run with `ROUTER_PRIMARY=qwen`.
- Build `datakit` (loader + resolve), the `resolve_entity` sub-flow, and `lookup`/`faq` answers.
- Build the test console first in type mode (§8a): conversation, state, slots, tool calls, raw LLM I/O panels. Everything after this is debugged through it.
- Data lists (`data/tests.csv`, `data/areas.csv`, pricing columns) with `config/entities/*.yaml`, normalizers (with unit tests), mock backend, `config/faq.yaml`.
- Flow engine + LLM extractor. Drive it from the terminal with typed Roman Urdu.
- Define the **workflow folder contract** now (`workflow.yaml`/states, `prompts/`, `manifest.yaml`, `tools`, `scenarios/`, optional `hooks.py`) and build workflow A to it, so workflow B and later ones use the same format (§14.1).
- Scenario eval harness (text) with ~30 scenarios: happy path, spelling variants, corrections ("nahi, kal nahi parson"), unknown test, uncovered area, caller refuses, out-of-scope, abusive, silence.
- **Exit:** ≥ 90% scenario pass on clean text; every high-risk slot has a confirmation step.

**Phase 1b: Workflow B and switching (2.5 days)**
- Build workflow B (`booking_status`, or the workflow you pick in §11) against the mock backend, using the same workflow interface, datakit and `resolve_entity`.
- Session manager: stack, suspend/resume, shared facts (name/phone carried over), non-interruptible states, depth limit, `switch`/`resume` decisions with the confirmation rules from §3c/§3e.
- Light verification for B: booking reference + phone on the booking, matched exactly and read back digit by digit, before details are spoken or a cancel/reschedule is done.
- UI: show the workflow stack and the suspended workflow's slots; add the `switching` scenarios to the picker.
- Note what in B's folder was config and what needed code (`hooks.py`); this decides how far the v2 automation (§14.1) can go.
- **Exit:** all `switching` and `routing` scenarios pass on clean text; resumed workflows keep their slots; no switch occurs during a non-interruptible state.

**Phase 2: Add noise (1 day, both workflows and switching)**
- Build noisy inputs from the real STT outputs of your 80 calls (benchmark workbook) and inject ASR-style errors into the scenarios, including misheard switch phrases. Re-run.
- Tune the corrector, fuzzy thresholds and prompts. **Exit:** ≥ 80% scenario pass with noisy text, and no wrongly submitted booking in the unconfirmed cases.

**Phase 2b: Router bake-off (1 day), in v1 after Phase 2**
- Build the routing eval set (labelled Roman Urdu utterances covering both workflows, FAQ, lookup, switch/resume, unclear, human, end; clean and noisy), run the four candidates from §3d, pick the router, calibrate `LAYA_CONFIDENCE_MIN` and `ROUTER_SWITCH_CONFIDENCE_MIN`. **Exit:** chosen router meets the decision rule in §3d.
- End of v1 dev: add the two cheap automation items from §14.1: `scripts/new_workflow.py` (scaffold) and `scripts/check_workflow.py` (validate, scenarios, routing and switching checks).

**Phase 3: Voice in the test UI (1.5 days)**
- Add Start/End Call and mic/speaker to the existing console; timings panel shows per-stage latency.
- Enable Guided voice and Auto-play audio modes in the scenario picker (synthetic caller via the TTS endpoint).
- LiveKit `AgentSession` with the STT/TTS plugins, local `livekit-server --dev`, VAD tuning, barge-in, filler audio.
- Measure turn latency; log per-stage timings.
- **Exit:** a full booking by voice from the UI with ≤ 3 s median turn gap.

**Phase 4: Telephony (2 days, depends on provider)**
- LiveKit SIP: inbound trunk + dispatch rule to the agent; connect the telephony provider. Stand up the self-hosted stack from §3b (`deploy/`) on a host with a public IP. Handle 8 kHz mu-law or A-law → 16 kHz resample, caller ID, hangup, transfer/callback.
- **Exit:** a real phone call completes a mock booking.

**Phase 5: Pilot hardening (ongoing)**
- Monitoring and alerts: failed submits, human-transfer rate, repeated-slot-failure rate, latency spikes, endpoint errors (STT/TTS/LLM down → graceful "abhi system mein masla hai" + callback).
- Real booking API, human-verification callback queue, call dashboard (log viewer), alerting on failed submits, cost/latency monitoring.
- Run on a small set of real inbound calls in shadow mode (bot listens and proposes the booking, human still handles) to measure accuracy before going live.

Rough total to a voice demo: **~9 working days** (Phases 0, 1, 1b, 2, 2b, 3). To a telephony pilot: **~12-14 days** (self-hosting adds server, SIP and network setup), mostly limited by provider setup and endpoint behavior.

---

## 10. Evaluation and acceptance criteria

Measure separately; do not collapse into one number.

| Metric | How | v1 target |
|---|---|---|
| Slot accuracy per slot (name, phone, tests, address, date) | scenario evals, text then audio | tests ≥ 90%, phone ≥ 95% *after confirmation* |
| Booking completion rate | scripted calls | ≥ 80% without human transfer |
| False submit rate (wrong data submitted) | confirmed-flag audits | ~0 for phone/address with human callback verification |
| Median / p95 turn latency | pipeline timings | ≤ 2.5 s / ≤ 4 s |
| STT WER on Pakistani phone audio | your 80-call set + new calls | track only; drives corrector work |
| Routing | routing eval set (both workflows, FAQ, lookup) | accuracy ≥ 90% on noisy text, false-switch ≤ 1% |
| Switching | `switching` scenarios, clean then noisy | 100% pass on clean text; ≥ 90% noisy; 0 switches during non-interruptible states; 100% of resumed workflows keep their slots |
| Each workflow's completion rate | scripted calls | ≥ 80% without human transfer |
| Safe fallback rate | forced-failure scenarios | 100% reach human/callback instead of looping |

---

## 11. Open questions (answer before Phase 0 ends)

1. **TTS input script:** does the endpoint expect Urdu script or does it handle Roman Urdu? Streaming support? Output sample rate? Voice gender? (Decides whether we need the Roman→Urdu adapter and the persona's grammatical gender.)
2. **STT endpoint:** given as `wss://…/en/v1/realtime` (`QWEN_VLLM_BASE_URL` in `.env`). The benchmark in this repo uses the domain-tuned `/chughtai/v1/realtime`. **Which path should the bot use?** The `/en/` path may not return Roman Urdu. Also: message format, partial results, max clip length, auth header for the token.
3. **LLM hosting:** where does Qwen3-8B run (GPU type), and is it vLLM, llama.cpp or something else? Is the quantization known? This sets latency and tool-call support.
4. **Telephony and host:** which provider/number for inbound calls (Pakistani number, SIP trunk, or web client only for now)? Which machine hosts the LiveKit stack for the pilot (public IP, open ports, domain/TLS)? Is it also where the GPU for Qwen runs?
5. **Real booking API:** does one exist? What fields does the home-sampling request need (and which are mandatory)? Without it we ship the mock and a callback queue.
6. **Data lists:** a sample (~50 rows) of the real test list (name, aliases, sample type, fasting, price), plus areas/cities and slot windows. Does the price vary by branch or by home sampling vs in-lab (then pricing is a separate table keyed by test id)? Do the same test names appear in several rows (then which column tells them apart)? Otherwise we start with a ~40-test seed from the call transcripts.
5b. **Second workflow:** is `booking_status` (check/reschedule/cancel an existing home-sampling booking) the right second workflow, or which one should replace it? It needs a status/cancel API (or a mock), and identifier fields (booking reference, phone).
6b. **FAQ list:** questions and answers, and which are asked most (feeds `faq.yaml` and router examples).
7. **Compliance:** call recording/consent line, PII storage and retention for patient names, phones and addresses.
8. **Handoff:** where do human transfers or callbacks go (phone number, ticket queue)?

---

## 12. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| STT ~66% on real calls | Wrong name/address/test | Catalog snapping, read-back, human verification callback in v1, corrector reuse |
| Qwen3-8B drifts or hallucinates | Wrong replies, wrong slots | Per-state prompts, JSON-schema output, code-side validation, low temperature, scenario evals on every prompt change |
| High latency from per-clip STT or non-streaming TTS | Awkward pauses, talk-over | Measure early, streaming where possible, filler lines, LLM→TTS sentence streaming |
| Roman Urdu spelling variance | Missed intents | Code normalizers + few-shot from real transcripts; no reliance on exact words |
| Barge-in/echo on phone lines | Bot hears itself | LiveKit interruption + noise/echo settings, test on real calls in Phase 4 |
| Router model (Laya) weak on Roman Urdu | Misroutes, false switches | Bake-off before commit; fine-tune + calibrate; fallback ladder in §3d; bias to "continue" |
| Self-hosted LiveKit infra (server, SIP service, Redis, ports, TLS; SIP preview status) | Extra ops, schedule risk, calls fail if ports/NAT are wrong | Dev mode locally first; confirm host with public IP and SIP status before Phase 4; Docker Compose; test SIP connectivity before debugging the agent |
| Over-scoping | Delay | v1 = home sampling only; everything else → human |
| Privacy (patient data in logs) | Compliance | Redact logs by default, retention policy, access control |

---

## 13. Start conditions, next steps and definition of done

**Needed from you (Phase 0 can start with the first three):**
1. STT path (`/en/v1/realtime` vs `/chughtai/v1/realtime`) and the auth/token format for STT and TTS.
2. Qwen3-8B URL (and whether vLLM or llama.cpp).
3. Go-ahead to scaffold `voicebot/` and commit as I go.
4. Before Phase 1: a sample (~50 rows) of the real test list incl. price columns, the areas list, and the FAQ list.
5. Before Phase 1b: confirm workflow B (`booking_status` or another) and its API or mock fields.
6. Before Phase 4: telephony provider and the pilot host (§11.4).

**Order of work:** Phase 0 (endpoints, scaffold) → Phase 1 (workflow A + data + chat UI) → Phase 1b (workflow B + switching) → Phase 2 (noise) → Phase 2b (router bake-off) → Phase 3 (voice) → Phase 4 (phone) → Phase 5 (pilot). Review typed conversations in the UI before adding voice.

**v1 is done when:**
- Both workflows, FAQ, lookup and switching work in the console by chat and by voice.
- The §10 targets are met on clean and noisy scenarios (routing, switching, slot accuracy, no wrong submits, safe fallback).
- The chosen router and its thresholds come from the bake-off, and are set in `.env`.
- A new data list and a new workflow can be added following §3h and §3c, and `check_workflow` reports on them.
- A phone call can complete a mock booking (Phase 4), or the plan records why not.

---

## 14. v2 plan (after v1 is working and measured)

v1 is two workflows written as folders (config files plus small code where needed) with switching. v2 reduces the effort and risk of adding the third, fourth and later workflows. Nothing here blocks v1; each item is started only when v1 numbers or real use show the need.

### 14.1 Workflow automation

**Design goal:** a workflow is a folder of config; code lives only in an optional `hooks.py`. Form-like workflows (collect fields, confirm, call one API) need no Python. v1's two workflows test whether this format holds; v2 formalizes it.

| Step | Item | Notes |
|---|---|---|
| 1 | **Scaffold command:** `new_workflow <name>` | Creates `workflows/<name>/` with template `workflow.yaml`, `prompts/`, `manifest.yaml`, `tools.yaml`, `scenarios/`. Cheap, can be done late in v1. |
| 2 | **One check command:** `check_workflow <name>` | Validates files, runs `check_data.py`, the workflow's scenarios, then the routing eval and the switching tests against every existing workflow. Prints pass/fail against the §10 targets. Cheap, can be done late in v1. |
| 3 | **Checks in CI** | Run `check_workflow` on every change to a workflow folder. A failing workflow cannot be merged or enabled. |
| 4 | **Declarative engine** | Core reads `workflow.yaml` (slots, states, transitions, confirmations, non-interruptible states) and `tools.yaml` (URL, method, request/response mapping) and runs the workflow. `hooks.py` only for custom validation or logic. |
| 5 | **LLM-drafted workflow** | Input: a plain description, optional OpenAPI spec, optional call transcripts. Output: draft `workflow.yaml`, prompts with Roman Urdu examples, router example phrases, `tools.yaml`. Uses a larger model, offline (not Qwen3-8B, not during calls). Drafts are never auto-enabled. |
| 6 | **Generated scenarios** | From a workflow definition: happy path, correction, retry, refusal, switching to/from every other workflow, and noisy variants built from real STT error patterns. |
| 7 | **Workflow builder page in the console** | Describe or upload a spec, review the draft, edit, run scenarios live, then Enable. |

**Stays manual (by design):** a person reviews every generated flow and prompt (a wrong confirmation rule on a booking or cancel is a real risk); real API credentials and connections are set up by a person; verification rules for patient data are approved by a person; enabling a workflow in production is a deliberate step, never automatic.

**Exit criteria:** a new form-like workflow can go from description to passing `check_workflow` in under one working day, with no Python written; routing and switching targets from §10 still hold with it enabled.

### 14.2 Other v2 candidates (each gated on evidence)

| Item | Start when |
|---|---|
| Fine-tune and calibrate a router model (Laya multilingual or a small Qwen) on logged routing decisions | Phase 2b misses targets, or the fallback rate to Qwen stays high |
| Embedding shortlist for the router | More than ~5 workflows |
| Verification sub-workflow (OTP, MR number + date of birth) | The first workflow that reveals patient data (e.g. report status) |
| Data tab in the console; alias learning from logs with reviewer approval | Lists are edited often, or enough real calls exist |
| datakit upgrades: other search backends, embeddings, index versioning, service mode, cross-list links | A list passes ~10k rows, fuzzy matching measurably fails, or lists reference each other |
| Adopt a dialogue framework (Rasa, LiveKit tasks) | The hand-written engine becomes hard to maintain at 3-4+ workflows |
| Streaming improvements (STT partials, TTS sentence streaming), latency tuning | Phase 3 numbers miss the turn-gap target |
| Real-call quality loop: sample and review calls, add failures to scenarios | After the pilot starts |

### 14.3 Call concurrency and call-center scale (v2)

Scope: everything about handling many simultaneous calls. v1 does not include it (v1 is built and tested one call at a time; a call-center deployment is the v2 goal).

**Inputs needed from you:** calls per hour, busiest-hour volume and peak simultaneous calls; number of human agents and how the PBX/agent desk takes transfers; whether bot capacity may overflow to humans. This sets the v2 scaling work (§14.3) and shows whether the HF Space endpoints are enough (they probably are not).

**The framework supports it; the endpoints are the limit.**
- **LiveKit:** one room per call. Agent workers register with the server and each call is dispatched as a job, so more calls means more worker processes or machines (horizontal scaling). Each call needs its own session state (workflow stack, slots), kept per job and never shared.
- **Per-call cost on the agent worker:** VAD, audio handling, orchestration. This is CPU and memory per call and is measured, not guessed.
- **Shared services, where the real limits are:**
  | Service | Concurrency question |
  |---|---|
  | STT websocket (HF Space) | Does one endpoint handle N simultaneous streams, and what is the latency at N? The URLs we have are "async-test" Spaces and may not be built for load. |
  | TTS websocket (HF Space) | Same question. |
  | Qwen3-8B (vLLM) | One GPU batches requests; latency rises with N. vLLM handles concurrency well, llama.cpp less so. |
  | Laya router | Small model; cheap on CPU/GPU, but measure at N. |
  | Booking/status APIs | Their own limits. |
- **Capacity measurement:** script N scripted callers (audio auto-play, §8a) at once, N = 1, 2, 5, 10, and record per-stage latency, turn gap, error rate and CPU/GPU use. The result is the measured maximum and the first bottleneck, and it sizes the scaling work.
- **Behaviour at capacity:** when the bot is full or an endpoint is saturated it must not degrade silently: play a polite "all lines are busy" message and offer a callback or a human number. Queueing and overflow routing are in the requirements below.
- **Design rules to apply when this is built:** no global mutable state, per-session logging with call ids, and endpoint clients that pool connections.

**Call-center requirements:**

| Requirement | Plan |
|---|---|
| Dedicated, scalable STT/TTS/LLM | Move off test Spaces to dedicated instances behind a load balancer; GPU capacity sized from the load test; autoscaling or pre-provisioned for peak hour |
| LiveKit and agent workers | Multiple `livekit-server` nodes with Redis, multiple SIP service nodes, N agent workers per machine, health checks and automatic restart |
| SIP capacity | Enough trunk channels for peak, a number-routing plan, failover to a second trunk or provider |
| Overflow and queueing | When bot capacity is full: callback offer, queue, or direct transfer to a human agent |
| Human handoff with context | Transfer to the agent desk with the transcript, collected slots, detected intent and reason (screen-pop); warm transfer where the PBX supports it |
| Availability | No single point of failure for the call path; rolling updates without dropping live calls (drain workers) |
| Operations view | Live dashboard: active calls, queue, bot-vs-human split, completion rate, transfer reasons, latency, error rate; alerts to on-call |
| Recording and compliance | Call recording with consent line, retention policy, access control, PII redaction in logs |
| Reporting | Per-workflow completion, containment (calls fully handled by the bot), average handle time, STT/LLM failure rate; weekly review of failed calls |
| Cost | Cost per call (GPU, STT/TTS, telephony) tracked against human-handled cost |

**Phase 7 (v2): Call-center scale-up**
- Provision dedicated STT/TTS/LLM capacity and the multi-node LiveKit/SIP deployment above; run a load test at the target peak plus margin.
- Implement overflow/queueing, human handoff with context, the live operations dashboard, recording and reporting.
- **Exit:** at the agreed peak of simultaneous calls, turn-gap and error targets from §10 still hold; capacity behaviour and failover were tested by killing a worker, an endpoint and a SIP trunk; handoff to the agent desk works end to end.
