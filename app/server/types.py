from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict


class WSMessage(TypedDict, total=False):
    type: str
    req_id: str
    # generic payload
    mode: str
    width: int
    n: int
    params: Dict[str, Any]
    flip_h: bool
    flip_v: bool
    rot_quadrant: int
    slot: int
    name: str
    rect: List[int]
    points: List[List[float]]


class PoseDict(TypedDict, total=False):
    x: float
    y: float
    theta_deg: float
    x_scale: float
    y_scale: float


class DetectionInstance(TypedDict, total=False):
    instance_id: int
    score: float
    inliers: int
    pose: PoseDict
    center: Optional[List[float]]
    quad: Optional[List[List[float]]]
    color: Dict[str, float]


class ObjectReport(TypedDict, total=False):
    object_id: int
    name: str
    template_size: Optional[List[int]]
    detections: List[DetectionInstance]


@dataclass
class ViewConfig:
    flip_h: bool = False
    flip_v: bool = False
    rot_quadrant: int = 0
    proc_width: int = 640


CaptureMode = Literal["training", "trigger"]

