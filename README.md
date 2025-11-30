# Fingerprint Processor

FastAPI + Celery service for enhancing fingerprint images and handling batch jobs. Includes utilities for generating distorted fingerprints and training a classifier.

## Features
- REST API for uploading, processing, listing, and downloading processed fingerprints
- Asynchronous background processing via Celery + Redis
- Image enhancement pipeline (grayscale, CLAHE, sharpening, contrast/brightness, noise reduction)
- Batch processing and cleanup task for old outputs
- Utilities for dataset distortion generation and classifier training (TensorFlow)

## Stack
- Python 3.10, FastAPI, Uvicorn
- Celery with Redis broker/backend
- Pillow + OpenCV for image processing
- TensorFlow/Keras, Albumentations, scikit-learn for training utilities

## Quickstart (Docker)
1) Build and start the stack:
   ```bash
   docker-compose up --build
   ```
   Services: `api` on port 8000, `worker` for Celery, `redis`.

2) Open docs: http://localhost:8000/docs

3) Upload images:
   ```bash
   curl -X POST "http://localhost:8000/upload/" \
     -F "files=@/path/to/img1.png" -F "files=@/path/to/img2.jpg"
   ```

4) Check job status:
   ```bash
   curl http://localhost:8000/status/<job_id>
   ```

5) Download files:
   ```bash
   curl -O http://localhost:8000/file/processed_images/<filename>
   ```

### Environment variables
- `REDIS_URL` (default `redis://redis:6379/0` in Docker, `redis://localhost:6379/0` otherwise)
- `OUTPUT_FOLDER` is optional per request (via API params)

### Volumes
`processed_images` is mounted to persist outputs between runs (see `docker-compose.yml`). The build context ignores large data folders (e.g., `cleanfp`, processed outputs); mount them explicitly if needed.

## Local Run (without Docker)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export REDIS_URL=redis://localhost:6379/0  # ensure Redis is running
# terminal 1
uvicorn main:app --reload --host 0.0.0.0 --port 8000
# terminal 2
celery -A tasks.celery_app worker --loglevel=info
```

## Project Layout
- `main.py` – FastAPI endpoints for upload/status/download/cleanup
- `tasks.py` – Celery tasks + image enhancement pipeline
- `fingerprint_distortion_generator.py` – synthesize distorted fingerprints for training
- `fingerprint_distortion_classifier.py` – transfer-learning classifier trainer (TensorFlow)
- `cleanfp/` – sample clean fingerprints

## Testing the API quickly
```bash
curl -X POST "http://localhost:8000/upload/" \
  -F "files=@cleanfp/012_3_1.tif" \
  -F "files=@cleanfp/012_3_2.tif"
curl http://localhost:8000/status/<job_id>
```

## Notes on the classifier & data tools
- Training script expects dataset at `fpds/` with class subfolders.
- Distortion generator can synthesize multiple distortion types to augment data.
- TensorFlow is included; CPU wheels are used by default (no GPU in Docker image).

## Next steps
- Add model inference endpoint if you want classification served alongside enhancement.
- Consider persisting job metadata/results in a database or Redis instead of in-memory.
