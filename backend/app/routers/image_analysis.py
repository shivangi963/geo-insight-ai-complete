"""
Image Analysis Router
Extracted from main.py - Handles computer vision tasks
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from typing import Dict, Any
import logging
from datetime import datetime
import tempfile
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analysis", tags=["image-analysis"])


@router.post("/image", status_code=202)
async def analyze_image(
    file: UploadFile = File(...),
    analysis_type: str = Query("object_detection", regex="^(object_detection|segmentation)$")
):
    """
    Analyze uploaded image using YOLO models
    
    Args:
        file: Image file (jpg, jpeg, png)
        analysis_type: Type of analysis
            - "object_detection": Detect cars, people, etc.
            - "segmentation": Segment green spaces
    
    Returns:
        Analysis results with detections/segments
    """
    try:    
        logger.info(f"📸 Image analysis request: type={analysis_type}")
        
        # Validate file exists
        if file is None:
            logger.error("No file uploaded")
            raise HTTPException(
                status_code=400,
                detail="No image file provided"
            )
        
        # Validate content type
        if not hasattr(file, 'content_type') or file.content_type is None:
            logger.error("File content_type is missing")
            raise HTTPException(
                status_code=400,
                detail="Invalid file upload - missing content type"
            )
        
        # Check if it's an image
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Please upload an image."
            )
        
        # Read file content
        image_data = await file.read()
        
        # Validate image size
        max_size = 10 * 1024 * 1024  # 10MB
        if len(image_data) > max_size:
            raise HTTPException(
                status_code=400,
                detail="Image file too large. Maximum size is 10MB."
            )
        
        logger.info(f"📊 Image size: {len(image_data)} bytes")
        
        # Convert to numpy array for YOLO
        import numpy as np
        from PIL import Image
        import io
        
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        logger.info(f"🖼️ Image dimensions: {image_np.shape}")
        
        # Perform analysis based on type
        if analysis_type == "object_detection":
            return await perform_object_detection(image_np, file.filename)
        elif analysis_type == "segmentation":
            return await perform_segmentation(image_np, file.filename)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {str(e)}"
        )


async def perform_object_detection(image_np, filename: str) -> Dict:
    """Perform YOLO object detection"""
    try:
        from ultralytics import YOLO
        
        # Load model
        model = YOLO('yolov8n.pt')
        
        # Run detection
        results = model(image_np)
        
        # Extract detections
        detections = []
        class_counts = {}
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    "class": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                }
                detections.append(detection)
                
                # Count classes
                class_name = detection["class"]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        logger.info(f"✅ Detected {len(detections)} objects")
        
        return {
            "status": "success",
            "analysis_type": "object_detection",
            "detections": detections,
            "class_counts": class_counts,
            "total_objects": len(detections),
            "image_size": {
                "width": image_np.shape[1],
                "height": image_np.shape[0]
            }
        }
    
    except Exception as e:
        logger.error(f"Object detection failed: {e}")
        raise


async def perform_segmentation(image_np, filename: str) -> Dict:
    """Perform YOLO segmentation for green space"""
    try:
        from ultralytics import YOLO
        
        # Load segmentation model
        model = YOLO('yolov8n-seg.pt')
        results = model(image_np)
        
        segments = []
        total_green_area = 0
        
        for result in results:
            if result.masks is not None:
                masks = result.masks
                boxes = result.boxes
                
                for mask, box in zip(masks, boxes):
                    segment = {
                        "class": result.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "mask_area": float(mask.data.sum())
                    }
                    segments.append(segment)
                    
                    # Check if it's green (tree, grass, plant)
                    if segment["class"] in ["tree", "grass", "plant"]:
                        total_green_area += segment["mask_area"]
        
        logger.info(f"✅ Found {len(segments)} segments")
        
        total_pixels = image_np.shape[0] * image_np.shape[1]
        green_percentage = (total_green_area / total_pixels) * 100 if total_pixels > 0 else 0
        
        return {
            "status": "success",
            "analysis_type": "segmentation",
            "segments": segments,
            "total_segments": len(segments),
            "green_space_percentage": round(green_percentage, 2),
            "total_pixels": total_pixels
        }
    
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise