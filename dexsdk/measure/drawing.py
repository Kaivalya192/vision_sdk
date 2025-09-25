from typing import Dict, Any, Tuple
import numpy as np
import cv2

def draw_caliper_debug(img, roi, result: Dict[str, Any]):
    """Draw caliper visualization including strip box and detected edges."""
    out = img.copy()
    
    # Draw strip box
    if "strip_box" in result:
        p0, p1, p2 = result["strip_box"]
        cv2.line(out, tuple(map(int, p0)), tuple(map(int, p1)), (0,255,0), 1)  # length
        cv2.line(out, tuple(map(int, p0)), tuple(map(int, p2)), (0,255,255), 1) # thickness
    
    # Draw detected edges
    for x, y, strength in result.get("edges_img", []):
        cv2.circle(out, (int(x), int(y)), 3, (0,0,255), -1)
        
    return out

def draw_caliper_roi(img, roi):
    """Draw interactive caliper ROI with handles."""
    out = img.copy()
    
    # Center point
    cx, cy = int(roi.cx), int(roi.cy)
    cv2.circle(out, (cx, cy), 4, (255,0,0), -1)
    
    # Direction vector
    a = np.radians(roi.angle_deg)
    dx, dy = roi.length * 0.5 * np.array([np.cos(a), np.sin(a)])
    p1 = (int(cx + dx), int(cy + dy))
    p2 = (int(cx - dx), int(cy - dy))
    cv2.line(out, p1, p2, (0,255,0), 1)
    
    # Thickness indicators
    nx = roi.thickness * 0.5 * np.array([-np.sin(a), np.cos(a)])
    t1 = (int(p1[0] + nx[0]), int(p1[1] + nx[1]))
    t2 = (int(p1[0] - nx[0]), int(p1[1] - nx[1]))
    t3 = (int(p2[0] + nx[0]), int(p2[1] + nx[1]))
    t4 = (int(p2[0] - nx[0]), int(p2[1] - nx[1]))
    
    cv2.line(out, t1, t2, (0,255,255), 1)
    cv2.line(out, t3, t4, (0,255,255), 1)
    cv2.line(out, t1, t3, (0,255,255), 1, cv2.LINE_DASH)
    cv2.line(out, t2, t4, (0,255,255), 1, cv2.LINE_DASH)
    
    return out