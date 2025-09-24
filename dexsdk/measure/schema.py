# d exsdk/measure/schema.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Literal, Dict, Any, Optional

ToolName = Literal[
    "line_caliper",        # fit a line with a 1D caliper band
    "edge_pair_width",     # find two edges within a band → width
    "angle_between",       # two calipers → angle
    "circle_diameter",     # ring search → diameter
]

@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int

@dataclass
class CaliperGuide:
    # ROI-relative guide endpoints (floats okay)
    p0: Tuple[float, float]
    p1: Tuple[float, float]
    band_px: int = 24
    n_scans: int = 32
    samples_per_scan: int = 64
    polarity: Literal["any", "rising", "falling"] = "any"
    min_contrast: float = 8.0

@dataclass
class AngleBetween:
    g1: CaliperGuide
    g2: CaliperGuide

@dataclass
class CircleParams:
    # ring search, ROI-relative center and radius range
    cx: float
    cy: float
    r_min: float
    r_max: float
    n_rays: int = 64
    samples_per_ray: int = 64
    polarity: Literal["any", "rising", "falling"] = "any"
    min_contrast: float = 8.0

@dataclass
class Job:
    tool: ToolName
    roi: ROI
    params: Dict[str, Any]

@dataclass
class Measurement:
    id: str
    kind: str
    value: float
    sigma: float
    passed: Optional[bool] = None

@dataclass
class Packet:
    units: str  # "px" or "mm"
    measures: List[Measurement]
    overlay: Optional[Any] = None   # cv2 BGR image for server rendering if needed
