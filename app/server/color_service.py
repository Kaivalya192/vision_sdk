from __future__ import annotations
from typing import Dict, List, Tuple
import cv2, numpy as np

def _circ(cnt) -> float:
    p = cv2.arcLength(cnt, True)
    a = cv2.contourArea(cnt)
    return (4.0*np.pi*a/(p*p)) if p>0 else 0.0

class ColorService:
    """
    Lightweight color rule processor for server-side runs.
    Compatible with your laptop POC's 'classes' schema.
    """
    def __init__(self):
        self.classes: List[Dict] = []     # [{"name", h_min.., circularity_min, ...}, ...]
        self.kernel_size: int = 5
        self.min_area_global: int = 100
        self.open_iter: int = 1
        self.close_iter: int = 1

    # ---- configuration ----
    def set_rules(self, classes: List[Dict], *, kernel_size: int|None=None,
                  min_area_global: int|None=None, open_iter: int|None=None,
                  close_iter: int|None=None) -> Dict:
        self.classes = list(classes or [])
        if kernel_size is not None: self.kernel_size = int(kernel_size)
        if min_area_global is not None: self.min_area_global = int(min_area_global)
        if open_iter is not None: self.open_iter = int(open_iter)
        if close_iter is not None: self.close_iter = int(close_iter)
        return {
            "n_classes": len(self.classes),
            "kernel_size": self.kernel_size,
            "min_area_global": self.min_area_global,
            "open_iter": self.open_iter,
            "close_iter": self.close_iter,
        }

    # ---- inference ----
    def _class_color(self, i: int) -> Tuple[int,int,int]:
        r = (37*i) % 256; g = (97*i) % 256; b = (173*i) % 256
        return (int(b), int(g), int(r))  # BGR

    def _morph(self, mask: np.ndarray) -> np.ndarray:
        k = max(1, int(self.kernel_size))
        if k % 2 == 0: k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        if self.open_iter  > 0: mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=self.open_iter)
        if self.close_iter > 0: mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.close_iter)
        return mask
    
    def run(self, frame_bgr: np.ndarray):
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr, [], np.zeros((1,1), np.uint8)

        vis = frame_bgr.copy()
        H, W = frame_bgr.shape[:2]

        # Prepare both HSV variants (BGR->HSV and RGB->HSV) to be robust
        hsv_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hsv_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2HSV)  # if the source was RGB

        def count_hits(hsv_img) -> int:
            total = 0
            for cls in self.classes:
                hmin = int(cls["h_min"]); hmax = int(cls["h_max"])
                smin = int(cls["s_min"]); smax = int(cls["s_max"])
                vmin = int(cls["v_min"]); vmax = int(cls["v_max"])

                if hmin <= hmax:
                    lower = np.array([hmin, smin, vmin], dtype=np.uint8)
                    upper = np.array([hmax, smax, vmax], dtype=np.uint8)
                    mask = cv2.inRange(hsv_img, lower, upper)
                else:
                    lower1 = np.array([hmin, smin, vmin], dtype=np.uint8)
                    upper1 = np.array([179,  smax, vmax], dtype=np.uint8)
                    lower2 = np.array([0,    smin, vmin], dtype=np.uint8)
                    upper2 = np.array([hmax, smax, vmax], dtype=np.uint8)
                    mask = cv2.bitwise_or(cv2.inRange(hsv_img, lower1, upper1),
                                        cv2.inRange(hsv_img, lower2, upper2))
                total += int(cv2.countNonZero(mask))
            return total

        hits_bgr = count_hits(hsv_bgr)
        hits_rgb = count_hits(hsv_rgb)

        hsv = hsv_rgb if hits_rgb > hits_bgr else hsv_bgr

        combined = np.zeros((H, W), np.uint8)
        out: List[Dict] = []

        for idx, cls in enumerate(self.classes, start=1):
            hmin = int(cls["h_min"]); hmax = int(cls["h_max"])
            smin = int(cls["s_min"]); smax = int(cls["s_max"])
            vmin = int(cls["v_min"]); vmax = int(cls["v_max"])

            if hmin <= hmax:
                lower = np.array([hmin, smin, vmin], dtype=np.uint8)
                upper = np.array([hmax, smax, vmax], dtype=np.uint8)
                mask = cv2.inRange(hsv, lower, upper)
            else:
                lower1 = np.array([hmin, smin, vmin], dtype=np.uint8)
                upper1 = np.array([179,  smax, vmax], dtype=np.uint8)
                lower2 = np.array([0,    smin, vmin], dtype=np.uint8)
                upper2 = np.array([hmax, smax, vmax], dtype=np.uint8)
                mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                                    cv2.inRange(hsv, lower2, upper2))

            mask = self._morph(mask)

            cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = int(cv2.contourArea(c))
                if area < max(self.min_area_global, int(cls.get("min_area", 0))):  continue
                if area > int(cls.get("max_area", 10**9)):                        continue
                x,y,w,h = cv2.boundingRect(c)
                aspect = (w/float(h)) if h>0 else 0.0
                if not (float(cls.get("aspect_min",0.0)) <= aspect <= float(cls.get("aspect_max",100.0))):
                    continue
                circ = _circ(c)
                if circ < float(cls.get("circularity_min", 0.0)):                  continue

                color = self._class_color(idx)
                cv2.drawContours(vis, [c], -1, color, 2)
                cv2.rectangle(vis, (x,y), (x+w,y+h), color, 1)
                cv2.putText(vis, f"{cls['name']} {area}", (x, max(0,y-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.drawContours(combined, [c], -1, int(idx), thickness=cv2.FILLED)

                M = cv2.moments(c)
                cx = int(M["m10"]/M["m00"]) if M["m00"] else x+w//2
                cy = int(M["m01"]/M["m00"]) if M["m00"] else y+h//2
                out.append({
                    "class_name": cls["name"], "bbox": [int(x),int(y),int(w),int(h)],
                    "area": area, "centroid": [cx,cy], "circularity": float(circ)
                })

        return vis, out, combined
