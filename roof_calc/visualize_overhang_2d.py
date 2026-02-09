from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def render_overhang_2d(
    wall_mask_path: str,
    roof_result: Dict[str, Any],
    *,
    output_path: str,
    title: str,
    overhang_px: float,
    upper_floor_footprint: Optional[List[List[tuple[float, float]]]] = None,
    show_drip_edge: bool = False,
) -> bool:
    """
    Saves a 2D visualization with roof rectangles expanded by `overhang_px`
    only on sides that are not attached to another roof rectangle.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    from roof_calc.overhang import apply_overhang_to_sections, compute_overhang_sides_from_union_boundary

    img = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    h, w = img.shape[:2]

    secs = roof_result.get("sections") or []
    if not secs:
        return False

    free = compute_overhang_sides_from_union_boundary(secs)
    secs_oh = apply_overhang_to_sections(secs, overhang_px=float(overhang_px), free_sides=free)

    # Clip: for lower floors, remove parts covered by upper-floor footprint.
    def _union_of_rects(sections: List[Dict[str, Any]]) -> Optional[Any]:
        polys = []
        for s in sections:
            br = s.get("bounding_rect") or []
            if len(br) < 3:
                continue
            try:
                polys.append(ShapelyPolygon(br))
            except Exception:
                continue
        if not polys:
            return None
        try:
            return unary_union(polys)
        except Exception:
            return polys[0]

    base_union = _union_of_rects(secs)
    oh_union = _union_of_rects(secs_oh)
    if upper_floor_footprint:
        upp = []
        for coords in upper_floor_footprint:
            if not coords or len(coords) < 3:
                continue
            try:
                upp.append(ShapelyPolygon(coords))
            except Exception:
                continue
        if upp:
            upper_union = unary_union(upp)
            try:
                if base_union is not None:
                    base_union = base_union.difference(upper_union)
                if oh_union is not None:
                    oh_union = oh_union.difference(upper_union)
            except Exception:
                pass

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # image coords
    ax.axis("off")

    # Background (walls)
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)

    def _draw_geom_outline(g: Any, *, color: str, lw: float, ls: str, alpha: float) -> None:
        if g is None or getattr(g, "is_empty", True):
            return
        geoms = getattr(g, "geoms", [g])
        for gg in geoms:
            ext = getattr(gg, "exterior", None)
            if ext is None:
                continue
            xs = [float(x) for x, y in list(ext.coords)]
            ys = [float(y) for x, y in list(ext.coords)]
            ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, alpha=alpha)

    def _draw_geom_filled(g: Any, *, color: str, alpha: float) -> None:
        if g is None or getattr(g, "is_empty", True):
            return
        geoms = getattr(g, "geoms", [g])
        for gg in geoms:
            ext = getattr(gg, "exterior", None)
            if ext is None:
                continue
            xs = [float(x) for x, y in list(ext.coords)]
            ys = [float(y) for x, y in list(ext.coords)]
            ax.fill(xs, ys, color=color, alpha=alpha)

    # Draw baseline outline (dashed)
    _draw_geom_outline(base_union, color="black", lw=1.0, ls="--", alpha=0.6)

    # Draw overhang outline/fill
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
    _draw_geom_filled(oh_union, color=colors[0], alpha=0.35)
    _draw_geom_outline(oh_union, color="black", lw=1.2, ls="-", alpha=0.9)

    # Drip edge a_frame: la marginea overhang-ului, 30% în jos (în 2D = contur distinct)
    if show_drip_edge and overhang_px > 0:
        _draw_geom_outline(oh_union, color="#8B4513", lw=2.0, ls="-", alpha=0.9)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.0)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def render_drip_debug_2d(
    wall_mask_path: str,
    roof_result: Dict[str, Any],
    *,
    output_path: str,
    overhang_px: float,
    upper_floor_footprint: Optional[List[List[tuple[float, float]]]] = None,
    exclude_low_side_shed: Optional[str] = None,
    title: Optional[str] = None,
) -> bool:
    """
    Vizualizare 2D a modului de generare a drip-ului: segmente drip + capete marcate.
    exclude_low_side_shed: pentru acoperiș shed - exclude latura cea mai de jos ("top"|"bottom"|"left"|"right").
    title: titlul figurii (implicit: "Drip edge – segmente și capete").
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    from roof_calc.overhang import (
        apply_overhang_to_sections,
        compute_overhang_sides_from_union_boundary,
        compute_overhang_sides_from_footprint,
        get_drip_edge_segments_2d,
    )

    img = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    h, w = img.shape[:2]

    secs = roof_result.get("sections") or []
    if not secs:
        return False

    free = compute_overhang_sides_from_union_boundary(secs)
    secs_oh = apply_overhang_to_sections(secs, overhang_px=float(overhang_px), free_sides=free)

    if upper_floor_footprint:
        try:
            fp_polys = [
                ShapelyPolygon([(float(c[0]), float(c[1])) for c in coords])
                for coords in upper_floor_footprint if coords and len(coords) >= 3
            ]
            if fp_polys:
                fp_union = unary_union(fp_polys)
                free = compute_overhang_sides_from_footprint(secs_oh, fp_union)
        except Exception:
            pass

    drip_segments = get_drip_edge_segments_2d(
        secs_oh,
        free_sides=free,
        exclude_eaves=exclude_low_side_shed is None,
        exclude_low_side=exclude_low_side_shed,
    )

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title or "Drip edge – segmente și capete")
    ax.set_aspect("equal")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)

    def _draw_union_outline(sections: List[Dict[str, Any]], color: str, ls: str) -> None:
        polys = []
        for s in sections:
            br = s.get("bounding_rect") or []
            if len(br) < 3:
                continue
            try:
                polys.append(ShapelyPolygon([(float(p[0]), float(p[1])) for p in br]))
            except Exception:
                continue
        if not polys:
            return
        try:
            u = unary_union(polys)
        except Exception:
            return
        if u is None or getattr(u, "is_empty", True):
            return
        geoms = getattr(u, "geoms", [u])
        for g in geoms:
            ext = getattr(g, "exterior", None)
            if ext is None:
                continue
            xs = [float(x) for x, y in list(ext.coords)]
            ys = [float(y) for x, y in list(ext.coords)]
            ax.plot(xs, ys, color=color, linewidth=1.0, linestyle=ls, alpha=0.7)

    _draw_union_outline(secs_oh, "black", "--")

    for idx, ((x1, y1), (x2, y2)) in enumerate(drip_segments):
        ax.plot([x1, x2], [y1, y2], color="#8B4513", linewidth=2.5, alpha=0.9)
        ax.scatter([x1, x2], [y1, y2], c="red", s=40, zorder=5, edgecolors="yellow", linewidths=1.5)
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.annotate(str(idx), (mid_x, mid_y), fontsize=8, color="white", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.2", facecolor="#8B4513", alpha=0.8))

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color="#8B4513", lw=2, label="Segment drip"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markeredgecolor="yellow",
               markersize=10, label="Capete segment"),
    ]
    ax.legend(handles=handles, loc="upper right")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.0)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True

