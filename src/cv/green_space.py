import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

def green_space_calculator(satellite_image_path, output_path="results/green_space_analysis.jpg"):
    """
    Calculate the percentage of green space in a satellite image using YOLO segmentation.
    """
    
    model = YOLO('yolov8n-seg.pt')  
    
    results = model(satellite_image_path)
    
    image = cv2.imread(satellite_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    
    total_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    green_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for result in results:
        if result.masks is not None:
            masks = result.masks.data
            boxes = result.boxes
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                class_id = int(box.cls[0].cpu().numpy())
                confidence = box.conf[0].cpu().numpy()
                              
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))                
                
                total_mask = np.logical_or(total_mask, mask_resized > 0.5)
                          
                if confidence > 0.5:
                    
                    obj_mask = (mask_resized > 0.5).astype(np.uint8)
            
                    masked_region = cv2.bitwise_and(image, image, mask=obj_mask)
                
                    hsv = cv2.cvtColor(masked_region, cv2.COLOR_RGB2HSV)
                                   
                    lower_green = np.array([35, 40, 40])
                    upper_green = np.array([85, 255, 255])
                                   
                    green_color_mask = cv2.inRange(hsv, lower_green, upper_green)                 
                
                    green_pixels = np.sum(green_color_mask > 0)
                    total_pixels = np.sum(obj_mask > 0)
                    
                    if total_pixels > 0 and (green_pixels / total_pixels) > 0.3:
                        green_mask = np.logical_or(green_mask, obj_mask > 0)
    
    def detect_green_by_color(img):
        """Detect green spaces using color analysis"""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        lower_green1 = np.array([35, 40, 40])
        upper_green1 = np.array([85, 255, 255])
        
        lower_green2 = np.array([25, 40, 40])
        upper_green2 = np.array([35, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
        mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
        
        green_mask_color = cv2.bitwise_or(mask1, mask2)   
     
        kernel = np.ones((5, 5), np.uint8)
        green_mask_color = cv2.morphologyEx(green_mask_color, cv2.MORPH_CLOSE, kernel)
        green_mask_color = cv2.morphologyEx(green_mask_color, cv2.MORPH_OPEN, kernel)
        
        return green_mask_color
     
    color_green_mask = detect_green_by_color(image)
    
    total_pixels = image.shape[0] * image.shape[1]
    green_pixels_color = np.sum(color_green_mask > 0)
    green_percentage_color = (green_pixels_color / total_pixels) * 100
    
    overlay = original_image.copy()
    
    green_overlay = np.zeros_like(original_image)
    green_overlay[color_green_mask > 0] = [0, 255, 0] 
    
    alpha = 0.6  
    overlay = cv2.addWeighted(original_image, 1 - alpha, green_overlay, alpha, 0)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Green Space: {green_percentage_color:.2f}%"
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    
    
    text_bg = (0, 0, 0, 128)  
    cv2.rectangle(overlay, (10, 10), (20 + text_size[0], 50), (0, 0, 0), -1)
    
    cv2.putText(overlay, text, (20, 40), font, 1, (255, 255, 255), 2)
       
    plt.figure(figsize=(12, 8))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title(f'Green Space Analysis - {green_percentage_color:.2f}% Green Space')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, pad_inches=0)
    plt.show()
    
    print(f"Green space analysis saved as: {output_path}")
    print(f"Green space percentage: {green_percentage_color:.2f}%")
    print(f"Total green area: {green_pixels_color} pixels")
    print(f"Total image area: {total_pixels} pixels")
    
    return green_percentage_color, overlay


if __name__ == "__main__":
    satellite_image_path = "data/satellite.jpg"
    green_percentage, overlay_image = green_space_calculator(satellite_image_path)


