#!/bin/bash
MODE="${1:-dev}"

if [ "$MODE" = "prod" ]; then
    uv run uvicorn server:app --host 0.0.0.0 --port 8082
else
    uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
fi
