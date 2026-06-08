FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY README.md ./
COPY src ./src
COPY config ./config
COPY static ./static
COPY alembic.ini ./
COPY alembic ./alembic

# Install with the 'voice' extra so SileroVAD (silero-vad + onnxruntime) is
# available — otherwise the dev console silently falls back to the rougher
# EnergyVAD and turn-end detection regresses.
RUN pip install --upgrade pip && pip install -e ".[voice]"

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
