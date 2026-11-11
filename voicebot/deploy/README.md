# Deploy notes (self-hosted LiveKit)

Dev: `docker compose -f deploy/docker-compose.yml up -d livekit`, then `python -m app.agent start` and `python -m app.server`.

Pilot / phone line (Phase 4, not exercised yet - needs a host with a public IP and a SIP provider):
1. Edit `livekit.yaml` and `sip.yaml` (keys, external IP). Keep real keys out of git.
2. `docker compose -f deploy/docker-compose.yml --profile sip up -d`.
3. Create the inbound trunk and dispatch rule with the LiveKit CLI/API so each call joins a new room; the agent worker
   (started without an explicit agent name) is dispatched automatically.
4. Check SIP connectivity from the provider before debugging the agent. Verify DTMF delivery for the phone-number fallback.
LiveKit SIP is documented as Developer Preview; confirm its status before promising a production date.
