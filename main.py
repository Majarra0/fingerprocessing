from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tasks import process_fingerprint, batch_process_fingerprints, cleanup_old_images
from typing import List, Optional
import os
import asyncio
from datetime import datetime
import uuid

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

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the bundled web UI."""
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index_path)


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
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    for file in files:
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(
                status_code=400, 
                detail=f"File {file.filename} has unsupported format. Allowed: {', '.join(allowed_extensions)}"
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
