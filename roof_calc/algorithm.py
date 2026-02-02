"""
Algorithm selection and main entry point: `calculate_roof_from_walls`.

Updated to use stronger rectangle detection (grid+coverage) when available,
inspired by the provided end-to-end pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

AlgorithmType = Literal["auto", "simple", "decomposition", "skeleton"]


def select_algorithm(shape_type: str) -> str:
    if shape_type == "rectangle":
        return "simple_bbox"
    if shape_type in ("L_shape", "T_shape"):
        return "rectangle_decomposition"
    if shape_type == "U_shape":
        return "rectangle_decomposition_with_courtyard"
    return "medial_axis"


def calculate_roof_from_walls(
    wall_mask: Union[str, np.ndarray],
    roof_angle: float = 35.0,
    overhang_px: float = 10.0,
    algorithm: AlgorithmType = "auto",
    adjacent_floor_rects: Optional[List[Polygon]] = None,
    main_roof_orientation: Optional[str] = None,
) -> Dict[str, Any]:
    """
    From wall mask compute roof sections, connections, total area.
    Toate lungimile și ariile în pixeli.
    """
    try:
        import cv2

        if isinstance(wall_mask, str):
            img = cv2.imread(wall_mask, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return _empty_result(f"cannot read image: {wall_mask}")
            wall_arr = img
        else:
            wall_arr = np.asarray(wall_mask, dtype=np.uint8)
            if wall_arr.ndim == 3:
                wall_arr = cv2.cvtColor(wall_arr, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        logger.error("Failed to load wall_mask: %s", e)
        return _empty_result(str(e))

    from roof_calc.flood_fill import EXTERIOR_FILL, flood_fill_interior, get_house_shape_mask
    from roof_calc.geometry import classify_shape, detect_components, extract_polygon
    from roof_calc.decomposition import _oriented_bounding_rect, partition_into_rectangles
    from roof_calc.roof_sections import (
        COMPLEXITY_COMPLEX,
        COMPLEXITY_L_T,
        COMPLEXITY_SIMPLE,
        apply_complexity_factor,
        apply_waste_factor,
        calculate_ridge_lines_with_hierarchy,
        calculate_roof_for_rectangle,
        calculate_total_area,
        calculate_valley_lengths,
        find_section_intersections,
    )
    from roof_calc.validation import validate_inputs, validate_roof_output

    ok, err = validate_inputs(wall_arr, roof_angle, overhang_px)
    if not ok:
        return _empty_result(err or "validation failed")

    filled = flood_fill_interior(wall_arr)
    components = detect_components(filled, use_house_shape=True, exterior_fill_value=EXTERIOR_FILL)
    if not components:
        return _empty_result("no building components")

    all_sections: List[Dict[str, Any]] = []
    shape_type_used = "complex"
    section_id = 0
    total_footprint = 0.0

    for comp_mask, _label, _bbox in components:
        poly = extract_polygon(comp_mask)
        if poly is None or poly.is_empty:
            continue
        total_footprint += float(poly.area)

        shape_type_used = classify_shape(poly)
        _algo = select_algorithm(shape_type_used) if algorithm == "auto" else (
            "simple_bbox" if algorithm == "simple" else
            "rectangle_decomposition" if algorithm == "decomposition" else "medial_axis"
        )

        # Use stronger decomposition using the filled component mask when available
        rects = partition_into_rectangles(poly, filled_mask=comp_mask)
        if not rects:
            # fallback to oriented bbox
            rect, _w, _h, angle = _oriented_bounding_rect(poly)
            rects = [(rect, angle)] if rect is not None and not rect.is_empty else []

        rectangles_only = [r for r, _a in rects if r is not None and not r.is_empty]
        ridge_h = calculate_ridge_lines_with_hierarchy(
            rectangles_only,
            adjacent_floor_rects=adjacent_floor_rects,
            main_roof_orientation=main_roof_orientation,
        )

        # Attach ridge info into sections (roof_sections expects it)
        for (rect, orientation_deg), ridge in zip(rects, ridge_h or []):
            if rect is None or rect.is_empty:
                continue
            sec = calculate_roof_for_rectangle(
                rect,
                float(orientation_deg),
                roof_angle=roof_angle,
                overhang_px=overhang_px,
                section_id=section_id,
            )
            if sec:
                # ensure main/secondary flags match hierarchy
                if isinstance(ridge, dict):
                    if "is_main" in ridge:
                        sec["is_main"] = bool(ridge["is_main"])
                    if "orientation" in ridge:
                        sec["ridge_orientation"] = str(ridge["orientation"])
                    if "start" in ridge and "end" in ridge:
                        sec["ridge_line"] = [list(ridge["start"]), list(ridge["end"])]
                all_sections.append(sec)
                section_id += 1

    if not all_sections:
        return _empty_result("no roof sections")

    all_connections = find_section_intersections(all_sections)
    total_area = calculate_total_area(all_sections, all_connections)
    total_valley = calculate_valley_lengths(all_sections, all_connections)
    area_with_complexity = apply_complexity_factor(total_area, shape_type_used)
    num_valleys = len([c for c in all_connections if c.get("type") == "valley"])
    final_area, waste_factor = apply_waste_factor(area_with_complexity, num_valleys)

    result: Dict[str, Any] = {
        "sections": all_sections,
        "connections": all_connections,
        "total_roof_area": round(float(final_area), 4),
        "total_valley_length": round(float(total_valley), 4),
        "complexity_factor": float(
            COMPLEXITY_SIMPLE if shape_type_used == "rectangle" else (
                COMPLEXITY_L_T if shape_type_used in ("L_shape", "T_shape") else COMPLEXITY_COMPLEX
            )
        ),
        "waste_factor": round(float(waste_factor), 4),
    }

    try:
        validate_roof_output(result, float(total_footprint))
    except Exception:
        pass

    return result


def _empty_result(message: str) -> Dict[str, Any]:
    return {
        "sections": [],
        "connections": [],
        "total_roof_area": 0.0,
        "total_valley_length": 0.0,
        "complexity_factor": 1.0,
        "waste_factor": 0.0,
        "error": message,
    }


