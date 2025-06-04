# Dockerfile
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    libpq-dev \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    python3-dev \
    git \
    curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY data/ data/
COPY logs/ logs/

COPY src/models/DockerVersion src/models/V1
COPY src/models/sbert src/models/sbert
COPY src/models/__init__.py src/models/model_training.py src/models/

COPY src/preprocessing/ src/preprocessing/
COPY src/queue/ src/queue/
COPY src/realtime/ src/realtime/

COPY src/__init__.py src/celery_app.py src/config.py src/logger.py src/main.py src/

EXPOSE 8000

CMD ["python", "-m", "src.main"]
