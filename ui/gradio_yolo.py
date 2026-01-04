import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os


MODEL_PATH = "../yolov8n.pt"  
model = YOLO(MODEL_PATH)

def detect_objects(image):

    image_np = np.array(image)
    
    results = model(image_np)
    
    annotated_image = results[0].plot()  
    
  
    if len(annotated_image.shape) == 3:
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return annotated_image

def create_gradio_interface():


    iface = gr.Interface(
        fn=detect_objects,
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs=gr.Image(label="Detected Objects"),
        title=" GeoInsight AI - Object Detection",
        description="Upload an image to detect objects using YOLOv8. Perfect for analyzing street scenes!",
        examples=[
            ["../results/annotated_street.jpg"] if os.path.exists("../results/annotated_street.jpg") else None
        ]
    )
    
    return iface

if __name__ == "__main__":

    iface = create_gradio_interface()
    iface.launch(
        share=False,  
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft()
    )