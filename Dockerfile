# Stage 1: builder 
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt && \
    pip install --prefix=/install dvc[gdrive] mlflow

# Stage 2: runtime 
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY --from=builder /install /usr/local

COPY src ./src
COPY config ./config
COPY data ./data
COPY models ./models
COPY dvc.yaml .
COPY dvc.lock .
COPY .dvc ./.dvc
COPY requirements.txt .

RUN mkdir -p /app/mlruns

CMD ["python", "src/train.py", "data/processed/processed_data.pickle", "models", "--max_rows", "5000"]