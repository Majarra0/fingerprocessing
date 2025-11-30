# Multi-purpose image for FastAPI API and Celery worker
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    REDIS_URL=redis://redis:6379/0

WORKDIR /app

# System deps for Pillow/OpenCV/TensorFlow wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libglib2.0-0 libgl1 libsm6 libxext6 \
       libjpeg-dev zlib1g-dev libpng-dev \
    && pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

ARG INSTALL_TRAINING=false
COPY requirements.txt requirements-train.txt ./
RUN if [ "$INSTALL_TRAINING" = "true" ]; then \
      pip install -r requirements-train.txt; \
    else \
      pip install -r requirements.txt; \
    fi

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
