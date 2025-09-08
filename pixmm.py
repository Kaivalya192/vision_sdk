#!/usr/bin/env python3
# Pixel->mm mapper for A4 using PiCam3 + Tkinter (with dropdown for mode selection).
# Shows mm/px ratio after 4 corner clicks. Does not save calibration file.

import time
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk

from dexsdk.camera.picam3 import PiCam3

A4_LONG_MM, A4_SHORT_MM = 297.0, 210.0
WIDTH_MODES = [(320,240),(480,360),(640,480),(800,600),(960,720),(1280,960)]

# --- helpers ---
def order_corners(pts: np.ndarray):
    s, diff = pts.sum(axis=1), np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
    return np.array([tl,tr,br,bl], dtype=np.float32)

def edge_lengths(tl,tr,br,bl):
    d = lambda a,b: float(np.hypot(*(a-b)))
    return 0.5*(d(br,bl)+d(tr,tl)), 0.5*(d(tr,br)+d(tl,bl))

def compute_calibration(img_pts: np.ndarray):
    pts = order_corners(img_pts.astype(np.float32))
    tl,tr,br,bl = pts
    width_px, height_px = edge_lengths(tl,tr,br,bl)
    if width_px >= height_px:   # width is long side
        mmx, mmy, long_axis = A4_LONG_MM/width_px, A4_SHORT_MM/height_px, "x"
    else:
        mmx, mmy, long_axis = A4_SHORT_MM/width_px, A4_LONG_MM/height_px, "y"
    return {
        "width_px": float(width_px),
        "height_px": float(height_px),
        "long_axis": long_axis,
        "mm_per_px_x": float(mmx),
        "mm_per_px_y": float(mmy)
    }

# --- App ---
class CalibApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("A4 Pixel→mm Calibration")

        # Info text
        self.info = tk.StringVar()
        tk.Label(self.root, textvariable=self.info).pack(anchor="w")

        # Dropdown for resolution
        self.selected_mode = tk.StringVar(value="640x480")
        dropdown = tk.OptionMenu(self.root, self.selected_mode, *[f"{w}x{h}" for w,h in WIDTH_MODES], command=self.change_mode)
        dropdown.pack(anchor="w")

        # Canvas
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(fill="both", expand=True)

        # Bind events
        self.root.bind("<Key>", self.on_key)
        self.canvas.bind("<Button-1>", self.on_click)

        # State
        self.clicks = []
        self.cam = None
        self.img_item = None

        # Start default
        self.change_mode("640x480")

    def change_mode(self, selection):
        if self.cam:
            try: self.cam.stop()
            except: pass
        w, h = map(int, selection.split("x"))
        self.info.set(f"{w}x{h} | click 4 corners (keys: r=reset, space=recapture, q=quit)")
        self.cam = PiCam3(width=w, height=h, fps=30, preview=None)
        time.sleep(0.2)
        self._capture_and_show()
        self.clicks.clear()

    def _capture_and_show(self):
        frame = self.cam.get_frame()  # RGB
        img = Image.fromarray(frame)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.config(width=frame.shape[1], height=frame.shape[0])
        if self.img_item is None:
            self.img_item = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        else:
            self.canvas.itemconfig(self.img_item, image=self.photo)
        self._redraw_overlay()

    def on_click(self, event):
        if len(self.clicks) < 4:
            self.clicks.append((event.x, event.y))
            self._redraw_overlay()
            if len(self.clicks) == 4:
                self._finalize()

    def on_key(self, event):
        k = event.keysym.lower()
        if k == "q":
            if self.cam: self.cam.stop()
            self.root.destroy()
        elif k == "r":
            self.clicks.clear()
            self._redraw_overlay()
        elif k == "space":
            self._capture_and_show()

    def _redraw_overlay(self):
        self.canvas.delete("overlay")
        for i,(x,y) in enumerate(self.clicks):
            self.canvas.create_oval(x-4,y-4,x+4,y+4,fill="yellow",tags="overlay")
            self.canvas.create_text(x+6,y-6,text=str(i+1),fill="yellow",tags="overlay")
        if len(self.clicks) == 4:
            pts = order_corners(np.array(self.clicks, np.float32))
            tl,tr,br,bl = [tuple(map(int,p)) for p in pts]
            self.canvas.create_line(tl,tr,br,bl,tl,fill="lime",width=2,tags="overlay")

    def _finalize(self):
        calib = compute_calibration(np.array(self.clicks, np.float32))
        msg = (f"mm/px X={calib['mm_per_px_x']:.5f}, Y={calib['mm_per_px_y']:.5f} "
               f"(long={calib['long_axis']}, W_px={calib['width_px']:.1f}, H_px={calib['height_px']:.1f})")
        self.info.set(msg)

    def run(self):
        self.root.mainloop()

def main():
    CalibApp().run()

if __name__ == "__main__":
    main()
