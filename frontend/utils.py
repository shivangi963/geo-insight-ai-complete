"""
Utility Functions for Frontend
Helper functions used across the application
"""
import streamlit as st
import time
from datetime import datetime
from typing import Optional, Dict, Any
from api_client import api
from config import TASK_POLL_INTERVAL, TASK_MAX_WAIT, TASK_PROGRESS_BAR_ENABLED

def format_currency(amount: float, decimals: int = 0) -> str:
    """Format number as currency"""
    return f"${amount:,.{decimals}f}"

def format_number(num: float, decimals: int = 0) -> str:
    """Format number with commas"""
    return f"{num:,.{decimals}f}"

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format as percentage"""
    return f"{value:.{decimals}f}%"

def format_date(date_str: str) -> str:
    """Format datetime string"""
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return date_str

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def calculate_price_per_sqft(price: float, square_feet: int) -> float:
    """Calculate price per square foot"""
    if square_feet > 0:
        return price / square_feet
    return 0

def get_walkability_label(score: float) -> tuple[str, str]:
    """Get walkability description and color"""
    if score >= 90:
        return ("Walker's Paradise", "walk-score-excellent")
    elif score >= 70:
        return ("Very Walkable", "walk-score-excellent")
    elif score >= 50:
        return ("Somewhat Walkable", "walk-score-good")
    elif score >= 25:
        return ("Car-Dependent", "walk-score-moderate")
    else:
        return ("Very Car-Dependent", "walk-score-poor")

def get_roi_label(roi: float) -> tuple[str, str]:
    """Get ROI quality label and emoji"""
    if roi > 12:
        return ("Excellent", "⭐⭐⭐")
    elif roi > 8:
        return ("Good", "⭐⭐")
    elif roi > 5:
        return ("Fair", "⭐")
    else:
        return ("Poor", "")

def validate_file_size(file, max_size_mb: int = 10) -> bool:
    """Validate uploaded file size"""
    if file.size > max_size_mb * 1024 * 1024:
        st.error(f"❌ File too large. Max size: {max_size_mb}MB")
        return False
    return True

def show_loading_spinner(message: str = "Loading..."):
    """Show loading spinner with message"""
    return st.spinner(message)

def poll_task_status(task_id: str, 
                    max_wait: int = TASK_MAX_WAIT,
                    show_progress: bool = TASK_PROGRESS_BAR_ENABLED) -> Optional[Dict]:
    """
    Poll task status until complete or timeout
    Returns result dict or None if failed/timeout
    """
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        data = api.get_task_status(task_id)
        
        if not data:
            if show_progress:
                progress_bar.empty()
                status_text.empty()
            return None
        
        status = data.get('status', 'unknown')
        progress = data.get('progress', 0)
        message = data.get('message', '')
        
        if show_progress:
            progress_bar.progress(min(progress / 100, 1.0))
            
            if status == 'pending':
                status_text.info(f"⏳ Queued: {message}")
            elif status == 'processing':
                status_text.info(f"⚙️ {message} ({progress}%)")
            elif status == 'completed':
                progress_bar.progress(1.0)
                status_text.success("✅ Complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                return data.get('result', {})
            elif status == 'failed':
                progress_bar.empty()
                error = data.get('error', 'Unknown error')
                status_text.error(f"❌ Failed: {error}")
                return None
        else:
            if status == 'completed':
                return data.get('result', {})
            elif status == 'failed':
                st.error(f"Task failed: {data.get('error')}")
                return None
        
        time.sleep(TASK_POLL_INTERVAL)
    
    if show_progress:
        progress_bar.empty()
        status_text.warning(f"⏱️ Timeout after {max_wait}s. Task: {task_id}")
    
    return None

def display_metric_card(label: str, value: str, delta: str = None, help_text: str = None):
    """Display a styled metric card"""
    if delta:
        st.metric(label, value, delta=delta, help=help_text)
    else:
        st.metric(label, value, help=help_text)

def create_download_button(data: str, filename: str, label: str = "Download"):
    """Create download button for data"""
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime="text/plain"
    )

def show_success_message(message: str, icon: str = "✅"):
    """Show success message with icon"""
    st.success(f"{icon} {message}")

def show_error_message(message: str, icon: str = "❌"):
    """Show error message with icon"""
    st.error(f"{icon} {message}")

def show_info_message(message: str, icon: str = "ℹ️"):
    """Show info message with icon"""
    st.info(f"{icon} {message}")

def show_warning_message(message: str, icon: str = "⚠️"):
    """Show warning message with icon"""
    st.warning(f"{icon} {message}")

def init_session_state(key: str, default_value: Any):
    """Initialize session state variable if not exists"""
    if key not in st.session_state:
        st.session_state[key] = default_value

def get_session_state(key: str, default: Any = None) -> Any:
    """Get session state value with default"""
    return st.session_state.get(key, default)

def set_session_state(key: str, value: Any):
    """Set session state value"""
    st.session_state[key] = value

def clear_session_state(*keys: str):
    """Clear specific session state keys"""
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]

def format_analysis_summary(analysis: Dict) -> str:
    """Format analysis data for display"""
    lines = []
    lines.append(f"**Address:** {analysis.get('address', 'N/A')}")
    lines.append(f"**Walk Score:** {analysis.get('walk_score', 'N/A')}/100")
    lines.append(f"**Total Amenities:** {analysis.get('total_amenities', 0)}")
    lines.append(f"**Status:** {analysis.get('status', 'unknown').title()}")
    
    created = analysis.get('created_at')
    if created:
        lines.append(f"**Created:** {format_date(created)}")
    
    return "\n".join(lines)

def create_amenity_emoji_map() -> Dict[str, str]:
    """Map amenity types to emojis"""
    return {
        'restaurant': '🍽️',
        'cafe': '☕',
        'school': '🏫',
        'hospital': '🏥',
        'park': '🌳',
        'supermarket': '🛒',
        'bank': '🏦',
        'pharmacy': '💊',
        'gym': '💪',
        'library': '📚',
        'transit_station': '🚇'
    }

def get_amenity_display_name(amenity_type: str) -> str:
    """Get display name with emoji for amenity"""
    emoji_map = create_amenity_emoji_map()
    emoji = emoji_map.get(amenity_type, '📍')
    name = amenity_type.replace('_', ' ').title()
    return f"{emoji} {name}"