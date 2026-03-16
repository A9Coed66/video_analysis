# =============================================================================
# Stage 1: Builder — compile dependencies in a virtualenv
# =============================================================================
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    curl \
    build-essential \
    libsndfile1-dev \
    ffmpeg \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv and install dependencies into it
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY web/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# Stage 2: Runtime — lean production image
# =============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 runtime and system libs needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire virtualenv from builder
COPY --from=builder /opt/venv /opt/venv

# Activate the virtualenv for all subsequent commands
ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

# Create non-root user for security
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Create persistent data directories
RUN mkdir -p /data/uploads /data/results \
    && chown -R appuser:appuser /data

# Copy application source code
WORKDIR /app
COPY web/ /app/web/
COPY pipeline/ /app/pipeline/
COPY voice_separator/ /app/voice_separator/
COPY worker/ /app/worker/

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

USER appuser

# Expose API port
EXPOSE 8000

# Default: run API server (workers override this in docker-compose)
CMD ["python", "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
