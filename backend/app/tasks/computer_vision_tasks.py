import os
from celery import shared_task
from ultralytics import YOLO
from PIL import Image
import numpy as np
from typing import Dict, List, Optional
import json
from datetime import datetime

_yolo_model = None

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO('yolov8n.pt')
    return _yolo_model

_seg_model = None

def get_seg_model():
    global _seg_model
    if _seg_model is None:
        _seg_model = YOLO('yolov8n-seg.pt')
    return _seg_model


@shared_task(bind=True, name="analyze_street_image")
def analyze_street_image_task(self, image_path: str) -> Dict:

    try:
        self.update_state(state='PROGRESS', meta={'status': 'Loading model...'})
        
        model = get_yolo_model()
  
        
        self.update_state(state='PROGRESS', meta={'status': 'Processing image...'})
        
        results = model(image_path)
        
        self.update_state(state='PROGRESS', meta={'status': 'Extracting results...'})
        
        result = results[0]
        detections = []
        
        if result.boxes is not None:
            for box in result.boxes:
                detection = {
                    'class': result.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()  
                }
                detections.append(detection)
  
        annotated_img = Image.fromarray(result.plot()[:, :, ::-1])
        os.makedirs("results", exist_ok=True)  
        output_path = f"results/annotated_{os.path.basename(image_path)}"
        annotated_img.save(output_path)
        
        class_counts = {}
        for detection in detections:
            class_name = detection['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'task_id': self.request.id,
            'status': 'SUCCESS',
            'detections': detections,
            'class_counts': class_counts,
            'total_detections': len(detections),
            'annotated_image_path': output_path,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'task_id': self.request.id,
            'status': 'FAILED',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

@shared_task(bind=True, name="calculate_green_space")
def calculate_green_space_task(self, image_path: str) -> Dict:
  
    try:
        self.update_state(state='PROGRESS', meta={'status': 'Loading segmentation model...'})
        
        model = get_seg_model()
        
        self.update_state(state='PROGRESS', meta={'status': 'Processing image...'})
        
        results = model(image_path)
        
        self.update_state(state='PROGRESS', meta={'status': 'Calculating green space...'})
        
       
        result = results[0]
        
        
        green_classes = ['tree', 'grass', 'plant']
        green_class_ids = [
            k for k, v in result.names.items()
            if v in green_classes
        ]

        total_pixels = result.orig_shape[0] * result.orig_shape[1]
        green_pixels = 0
        
        if result.masks is not None:
            for mask, cls in zip(result.masks.data, result.boxes.cls):
                if int(cls) in green_class_ids:
                    green_pixels += (mask > 0.5).sum().item()
        
        green_percentage = (green_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        annotated_img = Image.fromarray(result.plot()[:, :, ::-1])
        os.makedirs("results", exist_ok=True)  
        output_path = f"results/green_space_{os.path.basename(image_path)}"
        annotated_img.save(output_path)
        
        return {
            'task_id': self.request.id,
            'status': 'SUCCESS',
            'green_space_percentage': round(green_percentage, 2),
            'total_pixels': total_pixels,
            'green_pixels': int(green_pixels),
            'visualization_path': output_path,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'task_id': self.request.id,
            'status': 'FAILED',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }