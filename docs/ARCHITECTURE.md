# Architecture & Operations

## Services
- **API (`main.py`)** – FastAPI app exposing upload/status/download/list/cleanup endpoints.
- **Worker (`tasks.py`)** – Celery worker consuming tasks from Redis; handles image enhancement and cleanup.
- **Redis** – Broker + result backend for Celery (configurable via `REDIS_URL`).

## Data flow
1. Client uploads one or more images to `/upload/`.
2. API validates extensions and enqueues `process_fingerprint` (single) or `batch_process_fingerprints` (batch) with Celery.
3. Worker enhances images and saves results under the requested output folder (default `processed_images`).
4. Client polls `/status/{job_id}`. When complete, `/download/{job_id}` lists files and `/file/{folder}/{filename}` streams them.
5. `/cleanup/` can remove files older than `days_old` via Celery.

## Image enhancement pipeline (worker)
- Convert to grayscale if needed.
- Apply CLAHE to boost ridge contrast while limiting noise amplification.
- Sharpen + contrast (+50%) + slight brightness (+10%).
- Unsharp mask and mild Gaussian blur + sharpen to reduce noise and reinforce edges.
- Save both enhanced and original BMP copies with timestamp + UUID.

## Configuration
- `REDIS_URL` – broker/backend URL (default `redis://localhost:6379/0`, overridden to `redis://redis:6379/0` in Docker).
- `output_folder` & `filename_prefix` – per-request query/form params on upload.

## Docker topology
- `docker-compose.yml` defines three services (`api`, `worker`, `redis`). Both app containers share `./processed_images` to persist outputs.
- Single `Dockerfile` builds the app image for both API and worker commands.

## Development commands
- Run API locally: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
- Run worker: `celery -A tasks.celery_app worker --loglevel=info`
- Lint (optional): `ruff .` (not included by default)

## Training & data utilities
- `fingerprint_distortion_generator.py` – generate distorted samples for augmentation; see script arguments within the file.
- `fingerprint_distortion_classifier.py` – trains a transfer-learning classifier on dataset at `fpds/` (class subfolders). Saves model (`fingerprint_classifier_model*`) and training curves.

## Operational notes / TODOs
- Current job metadata is stored in-memory; migrate to Redis/DB for multi-instance or restarts.
- Add authentication and size limits before exposing publicly.
- Consider separating requirements for API-only vs. training (TensorFlow makes images heavy).
