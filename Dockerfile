FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VOICE_LOOP_DB_PATH=/app/data/knowledge_base.sqlite3 \
    VOICE_LOOP_OUTPUT_DIR=/app/output \
    VOICE_LOOP_QA_SEED_AUTO_SYNC=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        libasound2 \
        libasound2-plugins \
        portaudio19-dev \
        alsa-utils \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY voice_loop/ ./voice_loop/
COPY data/knowledge_base.sqlite3 ./data/knowledge_base.sqlite3
COPY .env.example .
COPY README.md DEPLOYMENT.md DEPLOYMENT_CHECKLIST.md DOCKER_HANDOFF.md TRANSCRIPT_CHEATS.md implementation.md ./

RUN mkdir -p /app/output /app/data/live_audio \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

CMD ["python", "main.py", "--language", "adaptive"]
