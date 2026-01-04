import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

def street_scene_analysis(image_path, output_path="results/annotated_street.jpg"):
    """
    Analyze a street scene using YOLO object detection and draw bounding boxes
    around cars and people.
    """
  
    model = YOLO('yolov8n.pt')  
    
    results = model(image_path)
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
   
    car_class_id = 2 
    person_class_id = 0  

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                 
                if (class_id == car_class_id or class_id == person_class_id) and confidence > 0.5:
                 
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    if class_id == car_class_id:
                        color = (255, 0, 0) 
                        label = f"Car: {confidence:.2f}"
                    else:
                        color = (0, 255, 0)  
                        label = f"Person: {confidence:.2f}"
                                    
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    
                   
                    cv2.putText(image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Street Scene Analysis - Cars (Red) and People (Green)')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()
    
    print(f"Annotated image saved as: {output_path}")
    return image


if __name__ == "__main__":
    
    image_path = "data/street.jpg"
    annotated_image = street_scene_analysis(image_path)