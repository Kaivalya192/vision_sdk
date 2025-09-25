import cv2
import numpy as np

def draw_caliper_debug(img_bgr: np.ndarray, roi, result, color=(0,255,255)):
    img = img_bgr
    # Axis line
    a = np.deg2rad(roi.angle_deg)
    ux = np.array([np.cos(a), np.sin(a)], np.float32)
    p0 = (roi.cx - 0.5*roi.length*ux[0], roi.cy - 0.5*roi.length*ux[1])
    p1 = (roi.cx + 0.5*roi.length*ux[0], roi.cy + 0.5*roi.length*ux[1])
    cv2.line(img, (int(p0[0]),int(p0[1])), (int(p1[0]),int(p1[1])), color, 2, cv2.LINE_AA)

    # Strip rectangle (3 corners from result["strip_box"])
    o, xe, ye = result["strip_box"]
    pts = np.array([o, xe, xe+ (ye-o), ye], np.int32)
    cv2.polylines(img, [pts], True, (60, 200, 60), 1, cv2.LINE_AA)

    # Edge markers (triangles)
    for (x,y,s) in result["edges_img"]:
        tri = np.array([[x,y-6],[x-6,y+6],[x+6,y+6]], np.int32)
        cv2.fillConvexPoly(img, tri, (0,255,255))

    # Status
    txt = "GO" if result["ok"] else "NO-GO"
    cv2.putText(img, f"{txt}: {result['msg']}", (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if result["ok"] else (0,0,255), 2, cv2.LINE_AA)
    return img
