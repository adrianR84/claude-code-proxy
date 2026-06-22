run:
	uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload

run-prod:
	uv run uvicorn server:app --host 0.0.0.0 --port 8082
