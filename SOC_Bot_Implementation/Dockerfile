# Dockerfile
FROM python:3.10.12-slim

ENV DEBIAN_FRONTEND=noninteractive

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

COPY requirementsV10.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirementsV10.txt

COPY data/ data/
COPY logs/ logs/

COPY src/models/V11/ src/models/V11/
COPY src/models/__init__.py src/models/model_training.py src/models/rl_agent.zip src/models/

COPY src/preprocessing/ src/preprocessing/
COPY src/queue/ src/queue/
COPY src/realtime/ src/realtime/

COPY src/__init__.py src/celery_app.py src/config.py src/logger.py src/main.py src/

EXPOSE 8000

CMD ["python", "-m", "src.main"]
