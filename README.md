# Fingerprint Processor

FastAPI + Celery service for enhancing fingerprint images, handling batch jobs, and ultimately comparing a new fingerprint against a stored database to identify who it belongs to. The database is a set of per-fingerprint JSON descriptor files derived from processed images. Includes utilities for generating distorted fingerprints and training a classifier.

## Features
- REST API for uploading, processing, listing, and downloading processed fingerprints (processed images feed the matching database)
- Asynchronous background processing via Celery + Redis, plus a synchronous combined endpoint
- Image enhancement pipeline (grayscale, CLAHE, sharpening, contrast/brightness, noise reduction)
- Batch processing and cleanup task for old outputs
- Fingerprint matching endpoint that compares an uploaded print to per-fingerprint JSON descriptors generated from processed images
- Utilities for dataset distortion generation and classifier training (TensorFlow)

## Stack
- Python 3.10, FastAPI, Uvicorn
- Celery with Redis broker/backend
- Pillow + OpenCV for image processing
- TensorFlow/Keras, Albumentations, scikit-learn for training utilities

## Quickstart (Docker)
1) Build and start the stack (Redis starts inside the compose stack; no separate Redis start needed):
   ```bash
   docker-compose up --build
   ```
   Services: `api` on port 8000, `worker` for Celery, `redis` (with a healthcheck so workers wait until Redis is ready).
   (The default image installs only the runtime deps; training extras are opt-in. To bake them in, run `docker compose build --build-arg INSTALL_TRAINING=true`.)

2) Open docs: http://localhost:8000/docs

3) Process and build per-fingerprint JSON descriptors in one call (synchronous):
   ```bash
   curl -X POST "http://localhost:8000/process-and-index/" \
     -F "files=@/path/to/img1.png" \
     -F "files=@/path/to/img2.jpg" \
     -F "output_folder=processed_images" \
     -F "filename_prefix=enhanced" \
     -F "index_path=fingerprint_index"
   ```

4) (Optional async flow) Upload images via Celery:
   ```bash
   curl -X POST "http://localhost:8000/upload/" \
     -F "files=@/path/to/img1.png" -F "files=@/path/to/img2.jpg"
   ```

5) Check job status (async flow):
   ```bash
   curl http://localhost:8000/status/<job_id>
   ```

6) Download files:
   ```bash
   curl -O http://localhost:8000/file/processed_images/<filename>
   ```

7) Match a fingerprint against the enrolled database (expects per-fingerprint JSON descriptors under `fingerprint_index/`):
   ```bash
   curl -X POST "http://localhost:8000/match/" \
     -F "file=@/path/to/query_fingerprint.png"
   ```

### Environment variables
- `REDIS_URL` (default `redis://redis:6379/0` in Docker, `redis://localhost:6379/0` otherwise)
- `OUTPUT_FOLDER` is optional per request (via API params)

### Volumes
`processed_images` is mounted to persist outputs between runs (see `docker-compose.yml`). The build context ignores large data folders (e.g., `cleanfp`, processed outputs); mount them explicitly if needed.
- `fingerprint_index/` contains per-fingerprint JSON descriptor files generated from `processed_images/` (ignored by git). The matching endpoint reads these files; rebuild by re-running `POST /process-and-index/` after adding new processed images.

## Local Run (without Docker)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# For training/distortion utilities add: pip install -r requirements-train.txt
export REDIS_URL=redis://localhost:6379/0  # ensure Redis is running
# terminal 1
uvicorn main:app --reload --host 0.0.0.0 --port 8000
# terminal 2
celery -A tasks.celery_app worker --loglevel=info
```

## Project Layout
- `main.py` – FastAPI endpoints for upload/status/download/cleanup, combined process+index endpoint, and matching
- `tasks.py` – Celery tasks + image enhancement pipeline
- `fingerprint_distortion_generator.py` – synthesize distorted fingerprints for training
- `fingerprint_distortion_classifier.py` – transfer-learning classifier trainer (TensorFlow)
- `cleanfp/` – sample clean fingerprints for testing; for real deployments use your own source with clear naming per person to distinguish identities

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
- Training deps (TensorFlow, scikit-learn, matplotlib, etc.) live in `requirements-train.txt` and are not installed in the default Docker image to avoid slow/fragile builds on Arch/ARM. Install them locally or rebuild with `--build-arg INSTALL_TRAINING=true` if you need the training utilities in a container.

## Next steps
- Add model inference endpoint if you want classification served alongside enhancement.
- Consider persisting job metadata/results in a database or Redis instead of in-memory.
