"""
Computer Vision Module for Green Space Analysis
Analyzes satellite imagery and OSM maps to calculate green space percentage
"""
import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
import os
import logging
from PIL import Image

logger = logging.getLogger(__name__)


def analyze_osm_green_spaces(image_path: str, analysis_id: str) -> Optional[Dict]:
    
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return None
        
        # Convert to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        logger.info(f"🖼️ Analyzing OSM image: {image.shape}")
        
        # Detect different green areas
        green_masks = detect_osm_green_areas(image_rgb)
        
        # Combine all green masks
        combined_mask = combine_green_masks(green_masks)
        
        # Calculate metrics
        total_pixels = image.shape[0] * image.shape[1]
        green_pixels = np.sum(combined_mask > 0)
        green_percentage = (green_pixels / total_pixels) * 100
        
        logger.info(f"📊 Green space analysis:")
        logger.info(f"   Total pixels: {total_pixels:,}")
        logger.info(f"   Green pixels: {green_pixels:,}")
        logger.info(f"   Green percentage: {green_percentage:.2f}%")
        
        # Log breakdown by type
        breakdown = {}
        for green_type, mask in green_masks.items():
            type_pixels = np.sum(mask > 0)
            type_pct = (type_pixels / total_pixels) * 100
            breakdown[green_type] = round(type_pct, 2)
            logger.info(f"   {green_type}: {type_pct:.2f}%")
        
        # Create visualization
        visualization_path = create_osm_green_visualization(
            image_rgb, combined_mask, green_masks, green_percentage, analysis_id
        )
        
        return {
            'green_space_percentage': round(green_percentage, 2),
            'green_pixels': int(green_pixels),
            'total_pixels': int(total_pixels),
            'visualization_path': visualization_path,
            'breakdown': breakdown
        }
        
    except Exception as e:
        logger.error(f"Error analyzing OSM green spaces: {e}", exc_info=True)
        return None


def detect_osm_green_areas(image: np.ndarray) -> Dict[str, np.ndarray]:
   
    green_masks = {}
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rgb = image.copy()
    
    # Define color ranges for different OSM green areas
    green_ranges = {
        'parks_grass': {
            # Light green used for parks (#c8facc, #aed1a0, #b5e3b5)
            'rgb_ranges': [
                ((160, 200, 160), (220, 255, 220)),  # Light green
                ((170, 209, 160), (180, 225, 180)),  # Parks
            ],
            'hsv_ranges': [
                ((35, 20, 180), (85, 100, 255)),  # Light green in HSV
            ]
        },
        'forests_woods': {
            # Dark green used for forests (#8dc56c, #6fc18e)
            'rgb_ranges': [
                ((100, 180, 100), (150, 210, 150)),  # Forest green
                ((111, 193, 142), (141, 213, 172)),  # Woods
            ],
            'hsv_ranges': [
                ((35, 40, 140), (85, 180, 220)),  # Forest green in HSV
            ]
        },
        'recreation': {
            # Medium green for recreation areas (#add19e)
            'rgb_ranges': [
                ((160, 200, 150), (180, 220, 170)),  # Recreation grounds
            ],
            'hsv_ranges': [
                ((35, 30, 160), (80, 120, 230)),  # Medium green
            ]
        },
        'natural_areas': {
            # Olive/natural reserve greens
            'rgb_ranges': [
                ((120, 140, 80), (160, 180, 120)),  # Olive green
            ],
            'hsv_ranges': [
                ((25, 40, 120), (45, 140, 190)),  # Olive in HSV
            ]
        }
    }
    
    # Detect each type
    for green_type, ranges in green_ranges.items():
        type_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Apply RGB ranges
        for lower, upper in ranges.get('rgb_ranges', []):
            lower_rgb = np.array(lower, dtype=np.uint8)
            upper_rgb = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(rgb, lower_rgb, upper_rgb)
            type_mask = cv2.bitwise_or(type_mask, mask)
        
        # Apply HSV ranges
        for lower, upper in ranges.get('hsv_ranges', []):
            lower_hsv = np.array(lower, dtype=np.uint8)
            upper_hsv = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
            type_mask = cv2.bitwise_or(type_mask, mask)
        
        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        type_mask = cv2.morphologyEx(type_mask, cv2.MORPH_CLOSE, kernel)
        type_mask = cv2.morphologyEx(type_mask, cv2.MORPH_OPEN, kernel)
        
        green_masks[green_type] = type_mask
    
    return green_masks


def combine_green_masks(green_masks: Dict[str, np.ndarray]) -> np.ndarray:
    """Combine all green masks into a single mask"""
    combined = None
    
    for mask in green_masks.values():
        if combined is None:
            combined = mask.copy()
        else:
            combined = cv2.bitwise_or(combined, mask)
    
    return combined if combined is not None else np.zeros((100, 100), dtype=np.uint8)


def create_osm_green_visualization(
    image: np.ndarray,
    combined_mask: np.ndarray,
    green_masks: Dict[str, np.ndarray],
    green_percentage: float,
    analysis_id: str
) -> str:
   
    try:
        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create colored overlay for different green types
        overlay = image.copy()
        colored_overlay = np.zeros_like(image)
        
        # Different colors for each green type
        colors = {
            'parks_grass': [144, 238, 144],      # Light green
            'forests_woods': [34, 139, 34],      # Forest green
            'recreation': [60, 179, 113],        # Medium sea green
            'natural_areas': [107, 142, 35]      # Olive drab
        }
        
        # Apply colored overlays
        for green_type, mask in green_masks.items():
            color = colors.get(green_type, [0, 255, 0])
            colored_overlay[mask > 0] = color
        
        # Blend overlay with original
        alpha = 0.5
        blended = cv2.addWeighted(overlay, 1 - alpha, colored_overlay, alpha, 0)
        
        # Add border around detected areas
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 255, 0), 2)
        
        # Add text with statistics
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Main percentage
        main_text = f"Green Space: {green_percentage:.2f}%"
        text_size = cv2.getTextSize(main_text, font, 1, 2)[0]
        
        # Background for text
        cv2.rectangle(blended, (10, 10), (30 + text_size[0], 60), (0, 0, 0), -1)
        cv2.putText(blended, main_text, (20, 40), font, 1, (255, 255, 255), 2)
        
        # Add legend
        y_offset = 80
        legend_items = [
            ("Parks/Grass", colors['parks_grass']),
            ("Forests/Woods", colors['forests_woods']),
            ("Recreation", colors['recreation']),
            ("Natural Areas", colors['natural_areas'])
        ]
        
        for label, color in legend_items:
            # Color box
            cv2.rectangle(blended, (10, y_offset), (40, y_offset + 20), color, -1)
            cv2.rectangle(blended, (10, y_offset), (40, y_offset + 20), (0, 0, 0), 1)
            
            # Label with background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(blended, (45, y_offset), (55 + label_size[0], y_offset + 25), (0, 0, 0), -1)
            cv2.putText(blended, label, (50, y_offset + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 30
        
        # Save visualization
        output_filename = f"osm_green_space_{analysis_id}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # Convert back to BGR for saving
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, blended_bgr)
        
        logger.info(f"💾 Visualization saved: {output_path}")
        
        # Return relative path
        return f"results/{output_filename}"
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return None




def create_green_space_visualization(
    image: np.ndarray,
    green_mask: np.ndarray,
    green_percentage: float,
    analysis_id: str
) -> str:
    """
    Create visualization overlay showing detected green spaces
    
    Args:
        image: Original RGB image
        green_mask: Binary mask of green areas
        green_percentage: Green space percentage
        analysis_id: Analysis ID for filename
    
    Returns:
        Path to saved visualization image
    """
    try:
        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create green overlay
        overlay = image.copy()
        green_overlay = np.zeros_like(image)
        green_overlay[green_mask > 0] = [0, 255, 0]  # Bright green
        
        # Blend overlay with original
        alpha = 0.5
        blended = cv2.addWeighted(overlay, 1 - alpha, green_overlay, alpha, 0)
        
        # Add text with green percentage
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Green Space: {green_percentage:.2f}%"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        
        # Add black background for text
        cv2.rectangle(blended, (10, 10), (20 + text_size[0], 50), (0, 0, 0), -1)
        
        # Add white text
        cv2.putText(blended, text, (20, 40), font, 1, (255, 255, 255), 2)
        
        # Save visualization
        output_filename = f"green_space_{analysis_id}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        # Convert back to BGR for saving
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, blended_bgr)
        
        logger.info(f"Visualization saved: {output_path}")
        
        # Return relative path
        return f"results/{output_filename}"
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return None