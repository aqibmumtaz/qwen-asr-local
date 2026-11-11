#!/usr/bin/env bash
# Start the test console: http://localhost:${UI_PORT:-7860}
# Uses the `voicebot` conda env (conda activate voicebot) or PYTHON=/path/to/python.
# Voice mode also needs: docker compose -f deploy/docker-compose.yml up -d livekit   and   python -m app.agent start
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python}"
exec "$PY" -m app.server
