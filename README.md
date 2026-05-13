# vox-agent

Vendor-agnostic agentic framework for multilingual VoiceBot (outbound marketing calls) and ChatBot (RAG-powered text assistant) systems.

See `PRD.md` for the full specification.

## Status

Phase 1 (Foundation) + Phase 2 (Critical-path adapters) implemented:

- Interfaces: STT, LLM, TTS, Telephony, VectorStore
- Adapters: Sarvam STT, Groq LLM, Sarvam TTS, Twilio, FAISS
- FastAPI app with `/health`, async PostgreSQL via SQLAlchemy 2, Redis session store

Phases 3–7 (voice pipeline, agents, RAG, campaigns, benchmarks) are not yet implemented.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env

docker compose up -d redis postgres
alembic upgrade head

uvicorn src.main:app --reload --port 8000
curl http://localhost:8000/health
```

## Tests

```bash
pytest tests/unit/ -v
```

All adapter tests are fully mocked — no live API access required.
