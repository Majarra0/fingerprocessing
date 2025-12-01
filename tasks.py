from celery import Celery
from PIL import Image, ImageFilter, ImageEnhance
import io
import os
import uuid
import numpy as np
import cv2
from datetime import datetime

# Configure Celery (allows overriding via REDIS_URL env var)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)

def _progress_callback(progress_fn, step, detail=None, state="PROGRESS"):
    if not progress_fn:
        return
    meta = {"step": step}
    if detail:
        meta.update(detail)
    try:
        progress_fn(state=state, meta=meta)
    except Exception:
        pass


def _enhance_fingerprint(file_content, output_folder="processed_images", filename_prefix="enhanced", progress_fn=None):
    """
    Process and enhance fingerprint images with multiple enhancement techniques.
    Allows optional progress_fn for Celery state updates.
    """
    try:
        os.makedirs(output_folder, exist_ok=True)

        _progress_callback(progress_fn, "decode")
        file_bytes = file_content if isinstance(file_content, (bytes, bytearray)) else file_content.encode('latin1')
        img = Image.open(io.BytesIO(file_bytes))

        if img.mode != 'L':
            img = img.convert('L')

        _progress_callback(progress_fn, "clahe")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_array = np.array(img)
        img_array = clahe.apply(img_array)
        img = Image.fromarray(img_array)

        _progress_callback(progress_fn, "enhance")
        enhanced_img = img.copy()
        enhanced_img = enhanced_img.filter(ImageFilter.SHARPEN)

        contrast_enhancer = ImageEnhance.Contrast(enhanced_img)
        enhanced_img = contrast_enhancer.enhance(1.5)

        brightness_enhancer = ImageEnhance.Brightness(enhanced_img)
        enhanced_img = brightness_enhancer.enhance(1.1)

        enhanced_img = enhanced_img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        enhanced_img = enhanced_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        enhanced_img = enhanced_img.filter(ImageFilter.SHARPEN)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{filename_prefix}_{timestamp}_{unique_id}.bmp"
        output_path = os.path.join(output_folder, filename)

        _progress_callback(progress_fn, "save", {"enhanced_image_path": output_path})
        enhanced_img.save(output_path, quality=95)

        original_filename = f"original_{timestamp}_{unique_id}.bmp"
        original_path = os.path.join(output_folder, original_filename)
        img.save(original_path)

        return {
            "status": "success",
            "message": "Image processed successfully",
            "enhanced_image_path": output_path,
            "original_image_path": original_path,
            "output_folder": output_folder,
            "enhancements_applied": [
                "Converted to grayscale",
                "Applied CLAHE contrast boost",
                "Sharpened",
                "Enhanced contrast (+50%)",
                "Enhanced brightness (+10%)",
                "Applied unsharp mask",
                "Noise reduction"
            ],
            "processing_timestamp": timestamp
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error processing image: {str(e)}",
            "enhanced_image_path": None,
            "original_image_path": None
        }


def _make_celery_progress(task):
    def _progress(state="PROGRESS", meta=None):
        try:
            task.update_state(state=state, meta=meta or {})
        except Exception:
            pass
    return _progress


@celery_app.task(bind=True)
def process_fingerprint(self, file_content, output_folder="processed_images", filename_prefix="enhanced"):
    """
    Celery task wrapper around the enhancement function with progress reporting.
    """
    progress_fn = _make_celery_progress(self)
    _progress_callback(progress_fn, "start")
    result = _enhance_fingerprint(file_content, output_folder, filename_prefix, progress_fn=progress_fn)
    if isinstance(result, dict) and result.get("status") == "error":
        _progress_callback(progress_fn, "error", {"message": result.get("message")}, state="FAILURE")
    return result

@celery_app.task(bind=True)
def batch_process_fingerprints(self, file_contents_list, output_folder="batch_processed", filename_prefix="batch_enhanced"):
    """
    Process multiple fingerprint images in batch with progress reporting.
    
    Args:
        file_contents_list: List of image file contents
        output_folder: Directory to save processed images
        filename_prefix: Prefix for output filenames
    
    Returns:
        dict: Batch processing results
    """
    progress_fn = _make_celery_progress(self)
    _progress_callback(progress_fn, "batch_start", {"total": len(file_contents_list)})

    results = []
    failed_count = 0
    success_count = 0

    os.makedirs(output_folder, exist_ok=True)

    for i, file_content in enumerate(file_contents_list):
        _progress_callback(progress_fn, "item_start", {"index": i + 1, "total": len(file_contents_list)})
        result = _enhance_fingerprint(
            file_content,
            output_folder,
            f"{filename_prefix}_{i+1}",
            progress_fn=lambda state="PROGRESS", meta=None: _progress_callback(
                progress_fn,
                (meta or {}).get("step", "item_progress"),
                {"index": i + 1, "total": len(file_contents_list), **(meta or {})},
                state=state
            ),
        )
        results.append(result)

        if result.get("status") == "success":
            success_count += 1
        else:
            failed_count += 1

    _progress_callback(progress_fn, "batch_complete", {"successful": success_count, "failed": failed_count})

    return {
        "batch_status": "completed",
        "total_images": len(file_contents_list),
        "successful": success_count,
        "failed": failed_count,
        "results": results,
        "output_folder": output_folder
    }

# Optional: Add a cleanup task to remove old processed images
@celery_app.task
def cleanup_old_images(folder_path, days_old=7):
    """
    Clean up processed images older than specified days
    
    Args:
        folder_path: Path to the folder containing processed images
        days_old: Number of days after which files should be deleted
    
    Returns:
        dict: Cleanup results
    """
    if not os.path.exists(folder_path):
        return {"status": "error", "message": "Folder does not exist"}
    
    import time
    current_time = time.time()
    deleted_count = 0
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > (days_old * 24 * 60 * 60):  # Convert days to seconds
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except OSError:
                    pass
    
    return {
        "status": "success",
        "deleted_files": deleted_count,
        "folder": folder_path
    }
