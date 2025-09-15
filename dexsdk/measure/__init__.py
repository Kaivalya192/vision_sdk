"""Measurement core API and schema helpers.

This package provides:
- Core types: MeasureContext, ResultPacket, Tool
- A minimal registry: register(name, cls), get(name), registry_names()
- Schema helpers to build standard result dictionaries
"""

from .core import (
    MeasureContext,
    ResultPacket,
    Tool,
    register,
    get,
    registry_names,
)
from .schema import make_result, make_measure
from .geometry import PointPick, LineFit, DistanceP2P  # noqa: F401

# Bind default tools into registry
register("point_pick", PointPick)
register("line_fit", LineFit)
register("distance_p2p", DistanceP2P)

__all__ = [
    "MeasureContext",
    "ResultPacket",
    "Tool",
    "register",
    "get",
    "registry_names",
    "make_result",
    "make_measure",
]
