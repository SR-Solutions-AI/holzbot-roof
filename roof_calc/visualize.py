"""
Loader shim.

Restores original implementation from `roof_calc/_orig_pyc/visualize.<cache_tag>.pyc`.
"""

from __future__ import annotations

import sys
from importlib.machinery import SourcelessFileLoader
from pathlib import Path

_base = Path(__file__).with_name("_orig_pyc")
_stem = Path(__file__).stem
_pyc = _base / f"{_stem}.{sys.implementation.cache_tag}.pyc"
if not _pyc.exists():
    matches = sorted(_base.glob(f"{_stem}.*.pyc"))
    if not matches:
        raise FileNotFoundError(f"Missing cached bytecode for {_stem} in {_base}")
    _pyc = matches[0]

SourcelessFileLoader(__name__, str(_pyc)).exec_module(sys.modules[__name__])

# ---------------------------------------------------------------------------
# Hotfix: multi-floor alignment
#
# We observed misaligned floor polygons (different coordinate systems per floor),
# which breaks 3D rendering (roofs/walls appear far apart). We override
# `_ordered_floor_polygons` to align all floors to the SAME reference center
# (derived from the roof floor's largest rectangle bbox when available).
# ---------------------------------------------------------------------------

from typing import Any, Dict, List, Optional, Tuple  # noqa: E402


def _ordered_floor_polygons(  # type: ignore[override]
    all_floor_paths: List[str],
    roof_floor_path: Optional[str],
    floor_roof_results: Optional[List[Dict[str, Any]]],
) -> List[Tuple[str, Any]]:
    import cv2  # local import
    import numpy as np
    from shapely import affinity as shapely_affinity

    from roof_calc.flood_fill import flood_fill_interior, get_house_shape_mask
    from roof_calc.geometry import extract_polygon

    # 1) Build polygons for each floor
    floors: List[Tuple[str, Any]] = []
    for p in all_floor_paths or []:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        filled = flood_fill_interior(img)
        house_mask = get_house_shape_mask(filled)
        poly = extract_polygon(house_mask)
        if poly is None or poly.is_empty:
            continue
        floors.append((p, poly))

    if not floors:
        return []

    # 2) Sort bottom->top by footprint area (largest first)
    floors.sort(key=lambda t: float(getattr(t[1], "area", 0.0) or 0.0), reverse=True)

    # Map path -> roof_result (because we will sort floors)
    res_by_path: Dict[str, Dict[str, Any]] = {}
    if floor_roof_results and len(floor_roof_results) == len(all_floor_paths or []):
        for p, rr in zip(all_floor_paths, floor_roof_results):
            if isinstance(rr, dict):
                res_by_path[p] = rr

    # 3) Pick roof floor index (reference) to compute common center
    roof_idx: Optional[int] = None
    if roof_floor_path:
        for i, (p, _poly) in enumerate(floors):
            if p == roof_floor_path:
                roof_idx = i
                break
    if roof_idx is None:
        # Heuristic: the smallest footprint is typically the top/roof floor
        roof_idx = int(np.argmin([float(getattr(poly, "area", 0.0) or 0.0) for _p, poly in floors]))

    cx_ref: Optional[float] = None
    cy_ref: Optional[float] = None

    # 4) Try to derive center from roof floor's largest rectangle bbox (best match to roof section coords)
    bbox = None
    try:
        roof_path_ref = floors[roof_idx][0]
        rr = res_by_path.get(roof_path_ref)
        if rr is not None:
            bbox = _get_largest_rect_bbox(rr)  # type: ignore[name-defined]
    except Exception:
        bbox = None
    if bbox:
        cx_ref = (bbox[0] + bbox[2]) / 2.0
        cy_ref = (bbox[1] + bbox[3]) / 2.0

    # 5) Fallback: use centroid of roof floor polygon
    if cx_ref is None or cy_ref is None:
        poly_ref = floors[roof_idx][1]
        c = poly_ref.centroid
        cx_ref = float(c.x)
        cy_ref = float(c.y)

    # 6) Translate ALL floor polygons by the same (-cx_ref, -cy_ref)
    aligned: List[Tuple[str, Any]] = []
    for p, poly in floors:
        aligned.append((p, shapely_affinity.translate(poly, xoff=-cx_ref, yoff=-cy_ref)))

    return aligned


