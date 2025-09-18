from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict
from typing import Literal


MeasurementKind = Literal["edge", "distance", "angle", "circle_radius"]


class MeasureJob(TypedDict, total=False):
    tool: MeasurementKind  # required
    roiA_poly: List[List[float]]  # required for all tools
    roiB_poly: List[List[float]]  # required for distance/angle
    mm_per_px: float | List[float]
    anchor: bool
    anchor_object: str
    id: str


class MeasuresPacket(TypedDict, total=False):
    units: Literal["mm", "px"]
    measures: List[Dict[str, Any]]
    overlay_jpeg_b64: str


def _ensure_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {} if meta is None else dict(meta)


def make_measure(
    *,
    id: str,
    kind: str,
    value: float,
    sigma: Optional[float] = None,
    passed: Optional[bool] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a measure dict matching the schema.

    Returns a dictionary with keys:
      - id: unique identifier for the measurement
      - kind: type, e.g., "distance_p2p"
      - value: numeric value (typically in millimeters)
      - sigma: optional standard deviation
      - pass: optional boolean pass/fail
      - meta: optional dictionary with additional context
    """

    return {
        "id": id,
        "kind": kind,
        "value": float(value),
        "sigma": None if sigma is None else float(sigma),
        "pass": passed,
        "meta": _ensure_meta(meta),
    }


def make_result(
    *,
    units: str = "mm",
    measures: Optional[List[Dict[str, Any]]] = None,
    primitives: Optional[Dict[str, Any]] = None,
    version: str = "1.0",
) -> Dict[str, Any]:
    """Build a result dictionary with the standard shape.

    Shape:
    {
      "version": "1.0",
      "units": "mm",
      "measures": [ ... ],
      "primitives": { ... }
    }
    """

    if units not in {"mm", "px"}:
        raise ValueError("units must be either 'mm' or 'px'")

    return {
        "version": str(version),
        "units": units,
        "measures": [] if measures is None else list(measures),
        "primitives": {} if primitives is None else dict(primitives),
    }


__all__ = [
    "MeasurementKind",
    "MeasureJob",
    "MeasuresPacket",
    "make_measure",
    "make_result",
]
