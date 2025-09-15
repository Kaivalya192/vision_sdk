from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type
from abc import ABC, abstractmethod


@dataclass
class MeasureContext:
    """Holds calibration and conversion parameters for measurement tools.

    Attributes
    - scale: (sx, sy) conversion from pixels to millimeters (px → mm).
    - homography: Optional 3x3 homography mapping pixels to a plane (row-major).
    - K: Optional 3x3 intrinsic matrix.
    - dist: Optional distortion coefficients.
    - units: Target reporting units, either "mm" or "px".
    """

    scale: Tuple[float, float] = (1.0, 1.0)
    homography: Optional[Sequence[Sequence[float]]] = None
    K: Optional[Sequence[Sequence[float]]] = None
    dist: Optional[Sequence[float]] = None
    units: str = "mm"

    def __post_init__(self) -> None:
        if self.units not in {"mm", "px"}:
            raise ValueError("units must be either 'mm' or 'px'")

    @property
    def scale_x(self) -> float:
        return float(self.scale[0])

    @property
    def scale_y(self) -> float:
        return float(self.scale[1])

    def px_len_to_mm(self, length_px: float, axis: str = "avg") -> float:
        """Convert a pixel length to millimeters using configured scale.

        axis: "x" | "y" | "avg" (default)
        """
        sx, sy = self.scale
        if axis == "x":
            return float(length_px) * float(sx)
        if axis == "y":
            return float(length_px) * float(sy)
        if axis == "avg":
            return float(length_px) * float((sx + sy) * 0.5)
        raise ValueError("axis must be 'x', 'y', or 'avg'")

    def px_vec_to_mm(self, dx_px: float, dy_px: float) -> Tuple[float, float]:
        """Convert a pixel vector to millimeters component-wise."""
        sx, sy = self.scale
        return float(dx_px) * float(sx), float(dy_px) * float(sy)


@dataclass
class ResultPacket:
    """Container for measurement results before schema formatting.

    This structure is intended to be easy to unit test and transform into a
    schema-compliant dict using helpers in `dexsdk.measure.schema`.
    """

    timestamp_ms: int
    frame_id: Optional[str] = None
    camera_info: Optional[Dict[str, Any]] = None
    measures: List[Dict[str, Any]] = field(default_factory=list)
    primitives: Dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """Abstract measurement tool.

    Implementations should override `run` and return a `ResultPacket` or a
    dictionary that can be adapted into the standard schema.
    """

    @abstractmethod
    def run(
        self,
        ctx: MeasureContext,
        image_bgr: Any,
        roi: Optional[Tuple[int, int, int, int]] = None,
        **kw: Any,
    ) -> ResultPacket | Dict[str, Any]:
        raise NotImplementedError


# --- Simple registry -------------------------------------------------------

_REGISTRY: Dict[str, Type[Tool]] = {}


def register(name: str, tool_cls: Optional[Type[Tool]] = None):
    """Register a Tool class under a name.

    Can be used as a function or decorator:

        register("ruler", RulerTool)

    or

        @register("ruler")
        class RulerTool(Tool):
            ...
    """

    lname = name.strip().lower()

    def _do_register(cls: Type[Tool]):
        if not issubclass(cls, Tool):
            raise TypeError("Registered class must subclass Tool")
        _REGISTRY[lname] = cls
        return cls

    if tool_cls is None:
        return _do_register
    return _do_register(tool_cls)


def get(name: str) -> Type[Tool]:
    """Retrieve a registered Tool class by name."""
    lname = name.strip().lower()
    if lname not in _REGISTRY:
        raise KeyError(f"Tool '{name}' not found. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[lname]


def registry_names() -> List[str]:
    """List registered tool names."""
    return sorted(_REGISTRY.keys())


if __name__ == "__main__":
    # Tiny sanity demo: build a dummy packet and print JSON in schema shape
    from .schema import make_result, make_measure

    ctx = MeasureContext(scale=(0.10, 0.10), units="mm")
    ts_ms = int(time.time() * 1000)

    # Example: a point-to-point distance measurement
    m = make_measure(
        id="demo_distance",
        kind="distance_p2p",
        value=12.345,
        sigma=0.06,
        passed=True,
        meta={
            "p1_px": [100, 120],
            "p2_px": [220, 150],
            "scale_mm_per_px": [ctx.scale_x, ctx.scale_y],
        },
    )

    result = make_result(
        units=ctx.units,
        measures=[m],
        primitives={
            "points": [[100, 120], [220, 150]],
            "lines": [[[100, 120], [220, 150]]],
        },
    )

    print(json.dumps(result, indent=2))

