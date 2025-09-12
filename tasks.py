from celery import Celery
from PIL import Image, ImageFilter, ImageEnhance
import io
import os
import uuid
from datetime import datetime

# Configure Celery
celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",  # Redis URL
    backend="redis://localhost:6379/0"
)

@celery_app.task
def process_fingerprint(file_content, output_folder="processed_images", filename_prefix="enhanced"):
    """
    Process and enhance fingerprint images with multiple enhancement techniques
    
    Args:
        file_content: Image file content as string (encoded with latin1)
        output_folder: Directory to save processed images (default: "processed_images")
        filename_prefix: Prefix for output filename (default: "enhanced")
    
    Returns:
        dict: Processing results including file path and applied enhancements
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Convert string back to bytes
        file_bytes = file_content.encode('latin1')
        img = Image.open(io.BytesIO(file_bytes))
        
        # Convert to grayscale if not already (common for fingerprints)
        if img.mode != 'L':
            img = img.convert('L')
        
        # Apply multiple enhancement techniques
        enhanced_img = img.copy()
        
        # 1. Sharpen the image
        enhanced_img = enhanced_img.filter(ImageFilter.SHARPEN)
        
        # 2. Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(enhanced_img)
        enhanced_img = contrast_enhancer.enhance(1.5)  # Increase contrast by 50%
        
        # 3. Enhance brightness slightly
        brightness_enhancer = ImageEnhance.Brightness(enhanced_img)
        enhanced_img = brightness_enhancer.enhance(1.1)  # Increase brightness by 10%
        
        # 4. Apply unsharp mask for better edge definition
        enhanced_img = enhanced_img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        
        # 5. Reduce noise with a mild blur then sharpen again
        enhanced_img = enhanced_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        enhanced_img = enhanced_img.filter(ImageFilter.SHARPEN)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{filename_prefix}_{timestamp}_{unique_id}.bmp"
        output_path = os.path.join(output_folder, filename)
        
        # Save enhanced image
        enhanced_img.save(output_path, quality=95)
        
        # Also save original for comparison (optional)
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

@celery_app.task
def batch_process_fingerprints(file_contents_list, output_folder="batch_processed", filename_prefix="batch_enhanced"):
    """
    Process multiple fingerprint images in batch
    
    Args:
        file_contents_list: List of image file contents
        output_folder: Directory to save processed images
        filename_prefix: Prefix for output filenames
    
    Returns:
        dict: Batch processing results
    """
    results = []
    failed_count = 0
    success_count = 0
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    for i, file_content in enumerate(file_contents_list):
        result = process_fingerprint(file_content, output_folder, f"{filename_prefix}_{i+1}")
        results.append(result)
        
        if result["status"] == "success":
            success_count += 1
        else:
            failed_count += 1
    
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