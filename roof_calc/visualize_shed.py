"""
Vizualizare 2D pentru acoperiș într-o apă (shed): o singură pantă, creasta pe latura
cea mai apropiată de etajul superior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _translate_union(geom: Any, dx: float, dy: float) -> Optional[Any]:
    """Translatează geometria Shapely cu (dx, dy)."""
    if geom is None or getattr(geom, "is_empty", True):
        return geom
    try:
        from shapely import affinity as shapely_affinity
        return shapely_affinity.translate(geom, xoff=dx, yoff=dy)
    except Exception:
        return geom


def _union_from_sections(sections: List[Dict[str, Any]]) -> Optional[Any]:
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.ops import unary_union
    except Exception:
        return None
    polys = []
    for s in sections or []:
        br = s.get("bounding_rect") or []
        if len(br) < 3:
            continue
        try:
            polys.append(ShapelyPolygon([(float(p[0]), float(p[1])) for p in br]))
        except Exception:
            continue
    if not polys:
        return None
    try:
        return unary_union(polys)
    except Exception:
        return polys[0] if polys else None


def _union_from_footprint(footprint: Optional[List[List[Tuple[float, float]]]]) -> Optional[Any]:
    if not footprint:
        return None
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.ops import unary_union
    except Exception:
        return None
    polys = []
    for coords in footprint:
        if not coords or len(coords) < 3:
            continue
        try:
            polys.append(ShapelyPolygon([(float(c[0]), float(c[1])) for c in coords]))
        except Exception:
            continue
    if not polys:
        return None
    try:
        return unary_union(polys)
    except Exception:
        return polys[0] if polys else None


def visualize_shed_lines(
    wall_mask_path: str,
    roof_result: Dict[str, Any],
    *,
    output_path: str,
    roof_angle_deg: float = 30.0,
    wall_height: float = 300.0,
    upper_floor_roof_sections: Optional[List[Dict[str, Any]]] = None,
    upper_floor_footprint: Optional[List[List[Tuple[float, float]]]] = None,
    overhang_px: float = 0.0,
    mask_offset: Optional[Tuple[float, float]] = None,
) -> bool:
    """
    Desenează liniile acoperișului într-o apă (o singură pantă).
    Creasta (partea înaltă) e pe latura cea mai apropiată de etajul superior.
    Perimetrul (streșină) e deplasat cu overhang_px pe laturile libere.
    mask_offset: (ox, oy) – originea măștii în overlay; transformă union_upper în coords măștii.
    """
    import cv2

    img = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    from roof_calc.overhang import (
        high_side_for_shed_from_upper_floor,
        apply_overhang_to_sections,
        compute_overhang_sides_from_union_boundary,
    )

    sections = roof_result.get("sections") or []
    if not sections:
        return False

    union_upper = _union_from_sections(upper_floor_roof_sections or [])
    if union_upper is None:
        union_upper = _union_from_footprint(upper_floor_footprint)
    # Pentru etaj cu remaining mask: secțiunile sunt în coords măștii; union_upper în overlay
    if union_upper is not None and mask_offset is not None:
        ox, oy = mask_offset
        union_upper = _translate_union(union_upper, -ox, -oy)

    # free_sides: laturi libere pentru overhang; exclude latura înaltă (ridge) care atinge etajul superior
    free_sides = compute_overhang_sides_from_union_boundary(sections)
    # high_sides din secțiuni base (înainte de overhang) pentru a exclude ridge-ul de la overhang
    high_sides_base = high_side_for_shed_from_upper_floor(sections, union_upper)
    for idx, hs in enumerate(high_sides_base):
        if idx < len(free_sides) and hs in free_sides[idx]:
            free_sides[idx] = {**free_sides[idx], hs: False}
    secs_oh = (
        apply_overhang_to_sections(sections, overhang_px=float(overhang_px), free_sides=free_sides)
        if overhang_px > 0
        else sections
    )
    high_sides = high_side_for_shed_from_upper_floor(secs_oh, union_upper)

    h, w = img.shape[:2]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f"Trasarea liniilor pentru acoperiș într-o apă (shed, {roof_angle_deg:.0f}°)")
    ax.set_aspect("equal")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)

    for idx, sec in enumerate(secs_oh):
        br = sec.get("bounding_rect") or []
        if len(br) < 3:
            continue
        minx = min(p[0] for p in br)
        maxx = max(p[0] for p in br)
        miny = min(p[1] for p in br)
        maxy = max(p[1] for p in br)
        hs = high_sides[idx] if idx < len(high_sides) else "top"

        # Ridge (creastă) = latura înaltă
        if hs == "top":
            ridge = [[minx, miny], [maxx, miny]]
        elif hs == "bottom":
            ridge = [[minx, maxy], [maxx, maxy]]
        elif hs == "left":
            ridge = [[minx, miny], [minx, maxy]]
        else:
            ridge = [[maxx, miny], [maxx, maxy]]

        # Perimetru (streșină)
        perim = [
            [minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]
        ]
        lbl_ridge = "Ridge (creastă)" if idx == 0 else ""
        lbl_eave = "Streașină" if idx == 0 else ""
        ax.plot([p[0] for p in ridge], [p[1] for p in ridge], color="darkred", linewidth=2, label=lbl_ridge)
        ax.plot([p[0] for p in perim], [p[1] for p in perim], color="green", linewidth=1, linestyle="--", label=lbl_eave)

    # Legenda (o singură dată)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="darkred", lw=2, label="Ridge (creastă)"),
        Line2D([0], [0], color="green", lw=1, ls="--", label="Streașină"),
    ]
    ax.legend(handles=handles, loc="upper right")

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        plt.close()
        return False
