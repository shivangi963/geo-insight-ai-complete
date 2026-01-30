import os
from datetime import datetime
from typing import Dict

from celery import shared_task

# ===================== GLOBAL MODELS =====================
_yolo_model = None
_seg_model = None
CV2_AVAILABLE = False


# ===================== MODEL LOADERS =====================
def get_yolo_model():
    global _yolo_model, CV2_AVAILABLE
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO("yolov8n.pt")
            CV2_AVAILABLE = True
        except Exception:
            CV2_AVAILABLE = False
            return None
    return _yolo_model


def get_seg_model():
    global _seg_model, CV2_AVAILABLE
    if _seg_model is None:
        try:
            from ultralytics import YOLO
            _seg_model = YOLO("yolov8n-seg.pt")
            CV2_AVAILABLE = True
        except Exception:
            CV2_AVAILABLE = False
            return None
    return _seg_model


# ===================== YOLO DETECTION TASK =====================
@shared_task(bind=True, name="analyze_street_image")
def analyze_street_image_task(self, image_path: str) -> Dict:

    try:
        self.update_state(state="PROGRESS", meta={"status": "Loading detection model..."})

        model = get_yolo_model()
        if model is None:
            return {"status": "FAILED", "error": "YOLO model not available"}

        from PIL import Image  # ✅ FIXED: lazy import

        self.update_state(state="PROGRESS", meta={"status": "Running inference..."})
        results = model(image_path)
        result = results[0]

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                detections.append({
                    "class": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()
                })

        # Save annotated image
        os.makedirs("results", exist_ok=True)
        annotated_img = Image.fromarray(result.plot()[:, :, ::-1])
        output_path = f"results/annotated_{os.path.basename(image_path)}"
        annotated_img.save(output_path)

        # Count classes
        class_counts = {}
        for d in detections:
            class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1

        return {
            "task_id": self.request.id,
            "status": "SUCCESS",
            "detections": detections,
            "class_counts": class_counts,
            "total_detections": len(detections),
            "annotated_image_path": output_path,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        return {
            "task_id": self.request.id,
            "status": "FAILED",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# ===================== GREEN SPACE SEGMENTATION =====================
@shared_task(bind=True, name="calculate_green_space")
def calculate_green_space_task(self, image_path: str) -> Dict:

    try:
        self.update_state(state="PROGRESS", meta={"status": "Loading segmentation model..."})

        model = get_seg_model()
        if model is None:
            return {"status": "FAILED", "error": "Segmentation model not available"}

        from PIL import Image  # ✅ FIXED
        import numpy as np

        self.update_state(state="PROGRESS", meta={"status": "Running segmentation..."})
        results = model(image_path)
        result = results[0]

        green_classes = {"tree", "grass", "plant"}
        green_class_ids = {
            k for k, v in result.names.items() if v in green_classes
        }

        total_pixels = result.orig_shape[0] * result.orig_shape[1]
        green_pixels = 0

        if result.masks is not None and result.boxes is not None:
            for mask, cls in zip(result.masks.data, result.boxes.cls):
                if int(cls) in green_class_ids:
                    green_pixels += int((mask > 0.5).sum().item())

        green_percentage = (
            (green_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        )

        os.makedirs("results", exist_ok=True)
        annotated_img = Image.fromarray(result.plot()[:, :, ::-1])
        output_path = f"results/green_space_{os.path.basename(image_path)}"
        annotated_img.save(output_path)

        return {
            "task_id": self.request.id,
            "status": "SUCCESS",
            "green_space_percentage": round(green_percentage, 2),
            "green_pixels": green_pixels,
            "total_pixels": total_pixels,
            "visualization_path": output_path,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        return {
            "task_id": self.request.id,
            "status": "FAILED",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }