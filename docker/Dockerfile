# Sandbox container for ALEF trial execution (Section 7.5: "The sandboxed
# agentic harness wraps ... instances within Docker containers isolated
# from external network access"). By default this image runs fully offline
# against the mock provider; mount API keys as env vars / secrets only if
# you intend to run real provider adapters, and note that doing so means the
# container is no longer network-isolated.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY alef/ ./alef/
COPY scripts/ ./scripts/
COPY examples/ ./examples/
COPY tests/ ./tests/
RUN pip install --no-cache-dir -e .

# Offline smoke test by default.
CMD ["python", "scripts/run_experiment.py", "--provider", "mock", \
     "--payloads", "alef/data/sample_payloads.jsonl", "--out", "runs/mock_run.jsonl"]
