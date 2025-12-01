from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from tasks import process_fingerprint, batch_process_fingerprints, cleanup_old_images, _enhance_fingerprint
from typing import List, Optional
import os
from datetime import datetime
import uuid
import cv2
import numpy as np
import json

app = FastAPI(title="Fingerprint Processing API", version="1.0.0")

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store task results temporarily (in production, use Redis or database)
task_results = {}

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
DEFAULT_DB_FOLDER = os.getenv("FP_DB_FOLDER", "processed_images")
# Folder to store per-fingerprint descriptor JSON files
DEFAULT_DB_INDEX = os.getenv("FP_DB_INDEX", "fingerprint_index")


def _is_allowed_file(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)


def _bytes_to_gray_image(file_bytes: bytes):
    """Decode uploaded bytes into a grayscale OpenCV image."""
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    return img


def _compute_orb_descriptors(image):
    """Return ORB descriptors for a grayscale image."""
    orb = cv2.ORB_create(nfeatures=800)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def _match_descriptors(query_desc, candidate_desc):
    """Score similarity between two ORB descriptor sets (higher is better)."""
    if query_desc is None or candidate_desc is None:
        return 0.0
    if isinstance(candidate_desc, list):
        candidate_desc = np.array(candidate_desc, dtype=np.uint8)
    if isinstance(query_desc, list):
        query_desc = np.array(query_desc, dtype=np.uint8)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(query_desc, candidate_desc)
    if not matches:
        return 0.0
    # Keep reasonably good matches; lower distance is better.
    good = [m for m in matches if m.distance < 50]
    denominator = max(1, min(len(query_desc), len(candidate_desc)))
    return len(good) / denominator


def _collect_images(db_folder: str):
    """Return list of image paths under db_folder."""
    images = []
    if not os.path.isdir(db_folder):
        return images
    for fname in os.listdir(db_folder):
        if _is_allowed_file(fname):
            images.append(os.path.join(db_folder, fname))
    return images


def _build_index(db_folder: str, index_path: str):
    """
    Build per-fingerprint JSON descriptor files from processed images.
    Returns an aggregated in-memory index.
    """
    images = _collect_images(db_folder)
    entries = []
    os.makedirs(index_path, exist_ok=True)
    for path in images:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, desc = _compute_orb_descriptors(img)
        if desc is None:
            continue
        base = os.path.splitext(os.path.basename(path))[0]
        json_path = os.path.join(index_path, f"{base}.json")
        entry = {
            "id": base,
            "file": path,
            "descriptors": desc.tolist()
        }
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(entry, f)
        except Exception:
            pass
        entries.append(entry)
    return {
        "db_folder": db_folder,
        "count": len(entries),
        "entries": entries,
        "built_at": datetime.utcnow().isoformat() + "Z",
        "index_dir": index_path,
    }


def _load_index(index_path: str):
    if not os.path.isdir(index_path):
        return None
    entries = []
    for fname in os.listdir(index_path):
        if not fname.lower().endswith(".json"):
            continue
        json_path = os.path.join(index_path, fname)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("descriptors"):
                    entries.append(data)
        except Exception:
            continue
    return {
        "db_folder": None,
        "count": len(entries),
        "entries": entries,
        "built_at": None,
        "index_dir": index_path,
    }


def _ensure_index(db_folder: str = DEFAULT_DB_FOLDER, index_path: str = DEFAULT_DB_INDEX, rebuild: bool = False):
    """Load per-fingerprint JSON descriptors; rebuild if requested or missing."""
    if rebuild or not os.path.isdir(index_path):
        return _build_index(db_folder, index_path)
    return _load_index(index_path)


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Fingerprint Processing API. See /docs for usage."}


@app.post("/process-and-index/")
async def process_and_index(
    files: List[UploadFile] = File(...),
    output_folder: str = DEFAULT_DB_FOLDER,
    filename_prefix: str = "enhanced",
    index_path: str = DEFAULT_DB_INDEX
):
    """
    Process fingerprints synchronously and immediately rebuild the per-fingerprint JSON index from the output folder.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    for f in files:
        if not _is_allowed_file(f.filename):
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {f.filename}")

    processed = []
    for f in files:
        content = await f.read()
        result = _enhance_fingerprint(content, output_folder=output_folder, filename_prefix=filename_prefix)
        processed.append({
            "uploaded_filename": f.filename,
            "enhanced_image_path": result.get("enhanced_image_path"),
            "original_image_path": result.get("original_image_path"),
            "status": result.get("status"),
            "message": result.get("message")
        })

    index = _build_index(output_folder, index_path)
    return {
        "message": "Processed files and rebuilt index",
        "output_folder": output_folder,
        "index_path": index_path,
        "processed": processed,
        "index_count": index.get("count", 0),
        "index_built_at": index.get("built_at"),
        "index_dir": index.get("index_dir")
    }


@app.post("/upload/")
async def upload_fingerprints(
    files: List[UploadFile] = File(...), 
    output_folder: Optional[str] = "processed_images",
    filename_prefix: Optional[str] = "enhanced"
):
    """
    Upload and process fingerprint images
    
    Args:
        files: List of image files to process
        output_folder: Custom output folder name (optional)
        filename_prefix: Custom filename prefix (optional)
    
    Returns:
        Processing job information with task IDs
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Validate file types
    for file in files:
        if not _is_allowed_file(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"File {file.filename} has unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
    
    job_id = str(uuid.uuid4())
    task_ids = []
    
    try:
        if len(files) == 1:
            # Single file processing
            content = await files[0].read()
            task = process_fingerprint.delay(
                content.decode('latin1'), 
                output_folder, 
                filename_prefix
            )
            task_ids.append(task.id)
            
        else:
            # Batch processing for multiple files
            file_contents = []
            for file in files:
                content = await file.read()
                file_contents.append(content.decode('latin1'))
            
            task = batch_process_fingerprints.delay(
                file_contents, 
                output_folder, 
                filename_prefix
            )
            task_ids.append(task.id)
        
        # Store job information
        task_results[job_id] = {
            "task_ids": task_ids,
            "status": "processing",
            "created_at": datetime.now().isoformat(),
            "file_count": len(files),
            "output_folder": output_folder
        }
        
        return {
            "message": f"{len(files)} file(s) sent for processing",
            "job_id": job_id,
            "task_ids": task_ids,
            "status": "processing",
            "file_count": len(files),
            "output_folder": output_folder,
            "check_status_url": f"/status/{job_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a processing job
    
    Args:
        job_id: The job ID returned from upload endpoint
    
    Returns:
        Job status and results
    """
    if job_id not in task_results:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = task_results[job_id]
    task_ids = job_info["task_ids"]
    
    # Check task status
    completed_tasks = []
    pending_tasks = []
    failed_tasks = []
    task_states = []
    
    for task_id in task_ids:
        task = process_fingerprint.AsyncResult(task_id) if len(task_ids) == 1 else batch_process_fingerprints.AsyncResult(task_id)
        state_info = {"task_id": task_id, "state": task.state}
        if task.info:
            info = task.info if isinstance(task.info, dict) else {"detail": str(task.info)}
            state_info["info"] = info
        task_states.append(state_info)

        if task.ready():
            if task.successful():
                result = task.result
                if isinstance(result, dict) and result.get("status") == "error":
                    failed_tasks.append({
                        "task_id": task_id,
                        "error": result.get("message", "Unknown error")
                    })
                    continue
                completed_tasks.append({
                    "task_id": task_id,
                    "result": result
                })
            else:
                failed_tasks.append({
                    "task_id": task_id,
                    "error": str(task.info)
                })
        else:
            pending_tasks.append(task_id)
    
    overall_status = "completed" if not pending_tasks and not failed_tasks else "failed" if failed_tasks else "processing"
    
    return {
        "job_id": job_id,
        "status": overall_status,
        "created_at": job_info["created_at"],
        "file_count": job_info["file_count"],
        "output_folder": job_info["output_folder"],
        "completed_tasks": len(completed_tasks),
        "pending_tasks": len(pending_tasks),
        "failed_tasks": len(failed_tasks),
        "results": completed_tasks,
        "errors": failed_tasks,
        "task_states": task_states
    }

@app.get("/download/{job_id}")
async def download_results(job_id: str):
    """
    Download processed images for a job (returns info about available files)
    
    Args:
        job_id: The job ID
    
    Returns:
        Information about available files for download
    """
    if job_id not in task_results:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = task_results[job_id]
    output_folder = job_info["output_folder"]
    
    if not os.path.exists(output_folder):
        raise HTTPException(status_code=404, detail="Output folder not found")
    
    # List available files
    files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))]
    
    return {
        "job_id": job_id,
        "output_folder": output_folder,
        "available_files": files,
        "download_urls": [f"/file/{output_folder}/{file}" for file in files]
    }


@app.post("/match/")
async def match_fingerprint(
    file: UploadFile = File(...),
    db_folder: str = DEFAULT_DB_FOLDER,
    index_path: str = DEFAULT_DB_INDEX,
    top_k: int = 3,
    rebuild_index: bool = False
):
    """
    Compare an uploaded fingerprint against a JSON index of processed images.
    The index is built from images in `db_folder` (defaults to processed outputs).
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if not _is_allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    if not os.path.isdir(db_folder):
        raise HTTPException(status_code=404, detail=f"Database folder '{db_folder}' not found")
    
    file_bytes = await file.read()
    query_img = _bytes_to_gray_image(file_bytes)
    if query_img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    _, query_desc = _compute_orb_descriptors(query_img)
    if query_desc is None:
        raise HTTPException(status_code=400, detail="No fingerprint features detected in upload")

    index = _ensure_index(db_folder=db_folder, index_path=index_path, rebuild=rebuild_index)
    if not index or not index.get("entries"):
        raise HTTPException(status_code=404, detail="Fingerprint index empty. Process images with /process-and-index/ or ensure processed_images contains files.")

    results = []
    for entry in index["entries"]:
        cand_desc = entry.get("descriptors")
        score = _match_descriptors(query_desc, cand_desc)
        results.append({
            "id": entry.get("id"),
            "file": entry.get("file"),
            "score": round(float(score), 4)
        })

    results = [r for r in results if r["score"] > 0]
    if not results:
        raise HTTPException(status_code=404, detail="No matches found in index")

    results = sorted(results, key=lambda r: r["score"], reverse=True)
    top_results = results[:max(1, top_k)]
    return {
        "query_filename": file.filename,
        "db_folder": db_folder,
        "index_path": index_path,
        "best_match": top_results[0],
        "top_matches": top_results,
        "total_candidates": len(results),
        "index_built_at": index.get("built_at"),
        "index_dir": index.get("index_dir"),
        "note": "ORB feature matcher on processed images; higher score means closer match"
    }


@app.get("/file/{folder_path:path}/{filename}")
async def download_file(folder_path: str, filename: str):
    """
    Download a specific processed file
    
    Args:
        folder_path: The folder containing the file
        filename: The filename to download
    
    Returns:
        File download response
    """
    file_path = os.path.join(folder_path, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

@app.get("/list-outputs/")
async def list_output_folders():
    """
    List all available output folders
    
    Returns:
        List of output folders and their contents
    """
    current_dir = "."
    output_folders = []
    
    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        if os.path.isdir(item_path) and (item.startswith("processed") or item.startswith("batch") or item.startswith("enhanced")):
            files = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]
            output_folders.append({
                "folder_name": item,
                "file_count": len(files),
                "files": files
            })
    
    return {"output_folders": output_folders}

@app.post("/cleanup/")
async def cleanup_files(
    background_tasks: BackgroundTasks,
    folder_path: str = "processed_images", 
    days_old: int = 7
):
    """
    Clean up old processed files
    
    Args:
        folder_path: Path to folder to clean
        days_old: Delete files older than this many days
    
    Returns:
        Cleanup job information
    """
    task = cleanup_old_images.delay(folder_path, days_old)
    
    return {
        "message": f"Cleanup job started for folder: {folder_path}",
        "task_id": task.id,
        "folder_path": folder_path,
        "days_old": days_old
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Fingerprint Processing API"
    }

@app.on_event("startup")
async def startup_event():
    """
    Application startup tasks
    """
    # Create default output directory
    os.makedirs("processed_images", exist_ok=True)
    print("Fingerprint Processing API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown tasks
    """
    print("Fingerprint Processing API shutting down")
