"""
Computer Vision Module for Green Space Analysis - UPDATED VERSION
Uses dark matte green colors and reduced image size
"""
import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
import os
import logging
from PIL import Image

logger = logging.getLogger(__name__)


def analyze_osm_green_spaces(image_path: str, analysis_id: str) -> Optional[Dict]:
    """
    Analyze OSM map image for green spaces - FIXED COLOR DETECTION
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return None
        
        # Convert to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        logger.info(f"🖼️ Analyzing OSM image: {image.shape}")
        logger.info(f"   Image stats - Min: {image_rgb.min()}, Max: {image_rgb.max()}, Mean: {image_rgb.mean():.1f}")
        
        # Detect different green areas with CORRECT OSM colors
        green_masks = detect_osm_green_areas_fixed(image_rgb)
        
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
            logger.info(f"   {green_type}: {type_pct:.2f}% ({type_pixels:,} pixels)")
        
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


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def detect_osm_green_areas_fixed(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    FIXED: Detect OSM green areas using CORRECT OpenStreetMap colors
    
    OSM Land Use Colors:
    - Parks/leisure: #c8facc (200, 250, 204)
    - Forests: #add19e (173, 209, 158), #8dc56c (141, 197, 108)
    - Grass/meadow: #cfeca8 (207, 236, 168)
    - Recreation: #add19e (173, 209, 158)
    - Natural: #ddf1d6 (221, 241, 214)
    """
    green_masks = {}
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rgb = image.copy()
    
    logger.info("🎨 Detecting green areas with OSM color standards...")
    
    # FIXED: Correct OSM color ranges
    green_ranges = {
        'parks_grass': {
            # OSM parks (#c8facc), grass (#cfeca8), leisure areas
            'rgb_ranges': [
                # Light green parks - EXPANDED RANGE
                ((180, 230, 180), (230, 255, 230)),  # Very light greens
                ((190, 240, 190), (210, 255, 210)),  # Parks
                ((195, 225, 155), (215, 245, 175)),  # Grass/meadow
            ],
            'hsv_ranges': [
                # Green hue (35-85), medium-high saturation, high value
                ((30, 15, 150), (90, 100, 255)),  # EXPANDED - light greens
                ((35, 20, 180), (75, 80, 255)),   # Medium light greens
            ]
        },
        'forests_woods': {
            # OSM forests (#add19e, #8dc56c) - darker greens
            'rgb_ranges': [
                # Forest greens - EXPANDED RANGE
                ((120, 180, 100), (180, 220, 180)),  # Medium forest green
                ((130, 190, 150), (180, 215, 170)),  # Woods
                ((100, 170, 90), (160, 210, 140)),   # Dense forest
            ],
            'hsv_ranges': [
                ((30, 25, 120), (90, 150, 230)),  # EXPANDED - forest greens
                ((35, 30, 100), (80, 130, 220)),  # Dark forest
            ]
        },
        'recreation': {
            # OSM recreation/sport (#add19e, similar to forests)
            'rgb_ranges': [
                ((165, 200, 150), (185, 220, 170)),  # Recreation grounds
                ((150, 190, 140), (180, 215, 165)),  # Sports fields
            ],
            'hsv_ranges': [
                ((32, 20, 150), (75, 90, 230)),  # EXPANDED
            ]
        },
        'natural_areas': {
            # OSM natural features (#ddf1d6) - very light green
            'rgb_ranges': [
                ((210, 235, 200), (230, 250, 225)),  # Natural/wetlands
                ((200, 230, 190), (225, 245, 220)),  # Scrubland
            ],
            'hsv_ranges': [
                ((30, 10, 180), (85, 60, 255)),  # EXPANDED - very light greens
            ]
        }
    }
    
    # Additional: Detect ANY green-ish color as fallback
    green_ranges['any_green'] = {
        'hsv_ranges': [
            # Very broad green detection
            ((25, 10, 100), (95, 200, 255)),  # Almost any green
        ]
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
        
        # Clean up mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        type_mask = cv2.morphologyEx(type_mask, cv2.MORPH_CLOSE, kernel)
        type_mask = cv2.morphologyEx(type_mask, cv2.MORPH_OPEN, kernel)
        
        # Store mask
        pixels_found = np.sum(type_mask > 0)
        logger.info(f"   {green_type}: {pixels_found:,} pixels detected")
        green_masks[green_type] = type_mask
    
    # Remove 'any_green' from final results (it's just for debugging)
    if 'any_green' in green_masks:
        any_green_pixels = np.sum(green_masks['any_green'] > 0)
        logger.info(f"   [Debug] Total green-ish pixels: {any_green_pixels:,}")
        # Don't include in final breakdown
        del green_masks['any_green']
    
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
    """Create visualization with detected green areas highlighted - DARK MATTE COLORS & 40% SIZE REDUCTION"""
    try:
        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create colored overlay for different green types
        overlay = image.copy()
        colored_overlay = np.zeros_like(image)
        
        # ✅ UPDATED: Dark matte green colors (instead of bright greens)
        colors = {
            'parks_grass': [60, 120, 60],        # Dark matte green
            'forests_woods': [30, 80, 30],       # Darker forest green
            'recreation': [50, 100, 70],         # Dark matte olive
            'natural_areas': [70, 130, 70]       # Medium dark green
        }
        
        # Apply colored overlays
        for green_type, mask in green_masks.items():
            color = colors.get(green_type, [0, 100, 0])  # Default dark green
            colored_overlay[mask > 0] = color
        
        # Blend overlay with original
        alpha = 0.5
        blended = cv2.addWeighted(overlay, 1 - alpha, colored_overlay, alpha, 0)
        
        # Add border around detected areas
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 180, 0), 2)  # Darker green border
        
        # Add text with statistics
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Main percentage
        main_text = f"Green Space: {green_percentage:.2f}%"
        text_size = cv2.getTextSize(main_text, font, 1, 2)[0]
        
        # Background for text
        cv2.rectangle(blended, (10, 10), (30 + text_size[0], 60), (0, 0, 0), -1)
        cv2.putText(blended, main_text, (20, 40), font, 1, (255, 255, 255), 2)
        
        # Add legend with dark colors
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
        
        # ✅ RESIZE IMAGE BY 60% (40% reduction)
        scale_factor = 0.6
        new_width = int(blended.shape[1] * scale_factor)
        new_height = int(blended.shape[0] * scale_factor)
        blended = cv2.resize(blended, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        logger.info(f"📏 Resized visualization: {new_width}x{new_height} (60% of original)")
        
        # Save visualization
        output_filename = f"osm_green_space_{analysis_id}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # Convert back to BGR for saving
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, blended_bgr)
        
        logger.info(f"✅ Visualization saved: {output_path}")
        
        # Return relative path
        return f"results/{output_filename}"
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return None