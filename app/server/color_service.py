from __future__ import annotations
from typing import Dict, List, Tuple
import time
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
    def __init__(self, debug: bool = True):
        self.classes: List[Dict] = []     # [{"name", h_min.., circularity_min, ...}, ...]
        self.kernel_size: int = 5
        self.min_area_global: int = 100
        self.open_iter: int = 1
        self.close_iter: int = 1
        self.debug: bool = bool(debug)
        self._frame_idx: int = 0

    def set_debug(self, on: bool):
        self.debug = bool(on)

    def _log(self, msg: str):
        if self.debug:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] [ColorService] {msg}", flush=True)

    def set_rules(self, classes: List[Dict], *, kernel_size: int|None=None,
                min_area_global: int|None=None, open_iter: int|None=None,
                close_iter: int|None=None) -> Dict:
        self.classes = list(classes or [])
        if kernel_size is not None: self.kernel_size = int(kernel_size)
        if min_area_global is not None: self.min_area_global = int(min_area_global)
        if open_iter is not None: self.open_iter = int(open_iter)
        if close_iter is not None: self.close_iter = int(close_iter)

        self._log(f"set_rules: n_classes={len(self.classes)}, "
                f"kernel={self.kernel_size}, min_area_global={self.min_area_global}, "
                f"open_iter={self.open_iter}, close_iter={self.close_iter}")

        for i, c in enumerate(self.classes):
            self._log(
                f"  [{i}] '{c.get('name','(unnamed)')}' "
                f"H[{c.get('h_min')}..{c.get('h_max')}] "
                f"S[{c.get('s_min')}..{c.get('s_max')}] "
                f"V[{c.get('v_min')}..{c.get('v_max')}] "
                f"min_area={c.get('min_area',0)} max_area={c.get('max_area','-')} "
                f"aspect=[{c.get('aspect_min',0.0)}..{c.get('aspect_max',100.0)}] "
                f"circ>={c.get('circularity_min',0.0)}"
            )

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
            self._log("run: empty frame")
            return frame_bgr, [], np.zeros((1,1), np.uint8)

        self._frame_idx += 1
        H, W = frame_bgr.shape[:2]
        vis = frame_bgr.copy()

        # Build HSV in both orders — helps catch RGB-vs-BGR issues
        hsv_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hsv_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2HSV)

        def _hits_total(hsv_img) -> int:
            total = 0
            for cls in self.classes:
                hmin = int(cls["h_min"]); hmax = int(cls["h_max"])
                smin = int(cls["s_min"]); smax = int(cls["s_max"])
                vmin = int(cls["v_min"]); vmax = int(cls["v_max"])
                if hmin <= hmax:
                    mask = cv2.inRange(hsv_img,
                                    np.array([hmin,smin,vmin], np.uint8),
                                    np.array([hmax,smax,vmax], np.uint8))
                else:
                    m1 = cv2.inRange(hsv_img,
                                    np.array([hmin,smin,vmin], np.uint8),
                                    np.array([179, smax,vmax], np.uint8))
                    m2 = cv2.inRange(hsv_img,
                                    np.array([0,   smin,vmin], np.uint8),
                                    np.array([hmax,smax,vmax], np.uint8))
                    mask = cv2.bitwise_or(m1, m2)
                total += int(cv2.countNonZero(mask))
            return total

        hits_bgr = _hits_total(hsv_bgr)
        hits_rgb = _hits_total(hsv_rgb)
        hsv = hsv_rgb if hits_rgb > hits_bgr else hsv_bgr
        chosen = "RGB->HSV" if hsv is hsv_rgb else "BGR->HSV"

        if (self._frame_idx % 15) == 1:
            self._log(f"run: frame#{self._frame_idx} size={W}x{H} "
                    f"hits(BGR)={hits_bgr} hits(RGB)={hits_rgb} using={chosen}")

        # Morph kernel prepared once
        k = max(1, int(self.kernel_size))
        if k % 2 == 0: k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        combined = np.zeros((H, W), np.uint8)
        out: List[Dict] = []
        total_kept = 0

        for idx, cls in enumerate(self.classes, start=1):
            name = cls.get("name","(unnamed)")
            hmin = int(cls["h_min"]); hmax = int(cls["h_max"])
            smin = int(cls["s_min"]); smax = int(cls["s_max"])
            vmin = int(cls["v_min"]); vmax = int(cls["v_max"])

            # Raw mask + wrap handling
            if hmin <= hmax:
                mask = cv2.inRange(hsv,
                                np.array([hmin,smin,vmin], np.uint8),
                                np.array([hmax,smax,vmax], np.uint8))
            else:
                m1 = cv2.inRange(hsv,
                                np.array([hmin,smin,vmin], np.uint8),
                                np.array([179, smax,vmax], np.uint8))
                m2 = cv2.inRange(hsv,
                                np.array([0,   smin,vmin], np.uint8),
                                np.array([hmax,smax,vmax], np.uint8))
                mask = cv2.bitwise_or(m1, m2)

            nz_raw = int(cv2.countNonZero(mask))

            # Morphology
            if self.open_iter  > 0: mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=self.open_iter)
            if self.close_iter > 0: mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.close_iter)
            nz_morph = int(cv2.countNonZero(mask))
            
            cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            n_cnt = len(cnts)
            kept_this = 0

            # NEW: reject counters
            rej_area = rej_maxarea = rej_aspect = rej_circ = 0

            for c in cnts:
                area = int(cv2.contourArea(c))
                if area < max(self.min_area_global, int(cls.get("min_area", 0))):
                    rej_area += 1
                    continue
                if area > int(cls.get("max_area", 10**9)):
                    rej_maxarea += 1
                    continue
                x,y,w,h = cv2.boundingRect(c)
                aspect = (w/float(h)) if h>0 else 0.0
                if not (float(cls.get("aspect_min",0.0)) <= aspect <= float(cls.get("aspect_max",100.0))):
                    rej_aspect += 1
                    continue
                circ = _circ(c)
                if circ < float(cls.get("circularity_min", 0.0)):
                    rej_circ += 1
                    continue

                # draw + record
                color = ( (37*idx)%256, (97*idx)%256, (173*idx)%256 )
                cv2.drawContours(vis, [c], -1, color, 2)
                cv2.rectangle(vis, (x,y), (x+w,y+h), color, 1)
                cv2.putText(vis, f"{name} {area}", (x, max(0,y-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.drawContours(combined, [c], -1, int(idx), thickness=cv2.FILLED)

                M = cv2.moments(c)
                cx = int(M["m10"]/M["m00"]) if M["m00"] else x+w//2
                cy = int(M["m01"]/M["m00"]) if M["m00"] else y+h//2
                out.append({
                    "class_name": name, "bbox": [int(x),int(y),int(w),int(h)],
                    "area": area, "centroid": [cx,cy], "circularity": float(circ)
                })
                kept_this += 1

            # Log per-class with rejects
            self._log(
                f"frame#{self._frame_idx} class='{name}' "
                f"nz_raw={nz_raw} nz_morph={nz_morph} cnt={n_cnt} kept={kept_this} "
                f"rej(area={rej_area},max={rej_maxarea},asp={rej_aspect},circ={rej_circ}) "
                f"min_area_g={self.min_area_global} min_area_c={int(cls.get('min_area',0))}"
            )


        if (self._frame_idx % 15) == 1 or total_kept:
            self._log(f"frame#{self._frame_idx} total objects kept={total_kept}")

        return vis, out, combined
