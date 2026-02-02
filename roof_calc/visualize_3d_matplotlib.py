"""
Matplotlib 3D renderers (standard + pyramid) using the **rectangles_floor** pipeline.

Key points:
- Builds per-floor roofs from `floor_roof_results` (sections = rectangles_floor truth)
- Aligns floors using the same offsets used by `floors_overlay.png` (largest rectangle center)
- Skips roof sections fully covered by upper floors (simple area threshold)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _ensure_matplotlib_cache_dirs() -> None:
    """Avoid Matplotlib/fontconfig cache permission issues."""
    root = Path(__file__).resolve().parents[1]
    mpl = root / ".mplconfig"
    cache = root / ".cache"
    mpl.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))


def _largest_section_bbox(rr: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    best = None
    best_area = -1
    for sec in rr.get("sections") or []:
        rect = sec.get("bounding_rect", [])
        if len(rect) < 3:
            continue
        xs = [p[0] for p in rect]
        ys = [p[1] for p in rect]
        minx, maxx = int(min(xs)), int(max(xs))
        miny, maxy = int(min(ys)), int(max(ys))
        area = (maxx - minx) * (maxy - miny)
        if area > best_area:
            best_area = area
            best = (minx, miny, maxx, maxy)
    return best


def _bbox_center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _compute_offsets(paths: List[str], results: List[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
    bboxes: List[Tuple[int, int, int, int]] = []
    for rr in results:
        bb = _largest_section_bbox(rr) or (0, 0, 0, 0)
        bboxes.append(bb)
    widths = [b[2] - b[0] for b in bboxes]
    heights = [b[3] - b[1] for b in bboxes]
    max_w = max(widths) if widths else 0
    max_h = max(heights) if heights else 0
    padding = 50
    ref_cx = (max_w + 2 * padding) / 2.0
    ref_cy = (max_h + 2 * padding) / 2.0
    out: Dict[str, Tuple[int, int]] = {}
    for p, bb in zip(paths, bboxes):
        cx, cy = _bbox_center(bb)
        ox = int(round(ref_cx - cx))
        oy = int(round(ref_cy - cy))
        out[p] = (ox, oy)
    return out


def _polygon_from_path(path: str) -> Optional[Any]:
    import cv2
    from roof_calc.flood_fill import flood_fill_interior, get_house_shape_mask
    from roof_calc.geometry import extract_polygon

    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    filled0 = flood_fill_interior(im)
    house0 = get_house_shape_mask(filled0)
    poly0 = extract_polygon(house0)
    if poly0 is None or poly0.is_empty:
        return None
    return poly0


def _translate_sections(secs: List[Dict[str, Any]], dx: float, dy: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sec in secs:
        rect = sec.get("bounding_rect", [])
        ridge = sec.get("ridge_line", [])
        out.append(
            {
                **sec,
                "bounding_rect": [(float(p[0]) + dx, float(p[1]) + dy) for p in rect] if rect else [],
                "ridge_line": [(float(p[0]) + dx, float(p[1]) + dy) for p in ridge] if ridge and len(ridge) >= 2 else [],
            }
        )
    return out


def _filter_connections_by_sections(conns: List[Dict[str, Any]], keep_ids: set[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in conns or []:
        ids = c.get("section_ids") or c.get("section_id") or []
        try:
            if isinstance(ids, (list, tuple)):
                if all(int(x) in keep_ids for x in ids):
                    out.append(c)
            else:
                if int(ids) in keep_ids:
                    out.append(c)
        except Exception:
            continue
    return out


def _covered_by_upper(sec_poly: Any, union_above: Any, area_thresh: float = 500.0) -> bool:
    if union_above is None or getattr(union_above, "is_empty", True):
        return False
    try:
        rem = sec_poly.difference(union_above)
        return float(getattr(rem, "area", 0.0) or 0.0) < area_thresh
    except Exception:
        return False


def visualize_3d_standard_matplotlib(
    output_path: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    all_floor_paths: Optional[List[str]] = None,
    floor_roof_results: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """
    Render `house_3d.png` (gable/standard) using matplotlib, based on rectangles_floor.
    """
    _ensure_matplotlib_cache_dirs()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union
    from shapely import affinity as shapely_affinity

    import roof_calc.roof_faces_3d as rf

    config = config or {}
    roof_angle_deg = float(config.get("roof_angle", 30.0))
    wall_height = float(config.get("wall_height", 300.0))
    colors = config.get("colors", ["#FF6B6B", "#4ECDC4", "#45B7D1"])
    views = config.get("views", [{"elev": 30, "azim": 45, "title": "Vedere Sud-Est"}, {"elev": 20, "azim": 225, "title": "Vedere Nord-Vest"}])
    # Rendering controls (reduce "cracks" and unnecessary diagonals/edges)
    show_edges = bool(config.get("show_edges", False))
    wall_show_edges = bool(config.get("wall_show_edges", show_edges))
    roof_show_edges = bool(config.get("roof_show_edges", show_edges))
    edge_color = str(config.get("edge_color", "black"))
    wall_edge_color = str(config.get("wall_edge_color", edge_color))
    roof_edge_color = str(config.get("roof_edge_color", edge_color))
    edge_width = float(config.get("edge_width", 0.4))

    if not all_floor_paths or not floor_roof_results or len(all_floor_paths) != len(floor_roof_results):
        return False

    offsets = _compute_offsets(all_floor_paths, floor_roof_results)
    floors_payload: List[Tuple[str, Any, Dict[str, Any], Tuple[int, int]]] = []
    for p, rr in zip(all_floor_paths, floor_roof_results):
        poly0 = _polygon_from_path(p)
        if poly0 is None:
            continue
        ox, oy = offsets.get(p, (0, 0))
        poly_t = shapely_affinity.translate(poly0, xoff=ox, yoff=oy)
        floors_payload.append((p, poly_t, rr, (ox, oy)))

    if not floors_payload:
        return False

    floors_payload.sort(key=lambda t: float(getattr(t[1], "area", 0.0) or 0.0), reverse=True)
    num_floors = len(floors_payload)

    fig = plt.figure(figsize=(16, 7))
    axs = []
    for i in range(2):
        axs.append(fig.add_subplot(1, 2, i + 1, projection="3d"))

    for ax, v in zip(axs, views[:2]):
        ax.set_title(v.get("title", "View"))
        ax.view_init(elev=float(v.get("elev", 30)), azim=float(v.get("azim", 45)))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_facecolor("white")

    # draw geometry once, on both axes
    for floor_idx, (_p, poly, rr, (ox, oy)) in enumerate(floors_payload):
        z0 = floor_idx * wall_height
        z1 = (floor_idx + 1) * wall_height

        coords = list(poly.exterior.coords) if hasattr(poly, "exterior") else []
        wall_faces = []
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            wall_faces.append([[p1[0], p1[1], z0], [p2[0], p2[1], z0], [p2[0], p2[1], z1], [p1[0], p1[1], z1]])

        union_above = None
        if floor_idx + 1 < len(floors_payload):
            union_above = unary_union([fp[1] for fp in floors_payload[floor_idx + 1 :]])

        # filter sections covered by upper floors
        secs = rr.get("sections") or []
        conns = rr.get("connections") or []
        kept: List[Dict[str, Any]] = []
        keep_ids: set[int] = set()
        for sec in secs:
            br = sec.get("bounding_rect", [])
            if len(br) < 3:
                continue
            sp = ShapelyPolygon(br)
            sp = shapely_affinity.translate(sp, xoff=ox, yoff=oy)
            if _covered_by_upper(sp, union_above, area_thresh=500.0):
                continue
            sec_t = _translate_sections([sec], float(ox), float(oy))[0]
            kept.append(sec_t)
            try:
                keep_ids.add(int(sec_t.get("section_id")))
            except Exception:
                pass

        conns_kept = _filter_connections_by_sections(conns, keep_ids)

        # roof faces (standard)
        roof_faces = rf.get_faces_3d_standard(kept, conns_kept, roof_angle_deg=roof_angle_deg, wall_height=z1)
        roof_polys = [f["vertices_3d"] for f in roof_faces if f.get("vertices_3d")]

        for ax in axs:
            if wall_faces:
                wall_coll = Poly3DCollection(
                    wall_faces,
                    facecolors="#B0B0B0",
                    edgecolors=(wall_edge_color if wall_show_edges else "none"),
                    linewidths=(edge_width if wall_show_edges else 0.0),
                    alpha=1.0,
                    zsort="average",
                )
                # Avoid tiny white cracks between adjacent faces
                wall_coll.set_antialiased(False)
                ax.add_collection3d(wall_coll)
            for idx, face in enumerate(roof_polys):
                col = colors[idx % len(colors)]
                roof_coll = Poly3DCollection(
                    [face],
                    facecolors=col,
                    edgecolors=(roof_edge_color if roof_show_edges else "none"),
                    linewidths=(edge_width if roof_show_edges else 0.0),
                    alpha=1.0,
                    zsort="average",
                )
                roof_coll.set_antialiased(False)
                ax.add_collection3d(roof_coll)

    # bounds
    xs = []
    ys = []
    for _p, poly, rr, (ox, oy) in floors_payload:
        coords = list(poly.exterior.coords) if hasattr(poly, "exterior") else []
        xs.extend([c[0] for c in coords])
        ys.extend([c[1] for c in coords])
        for sec in rr.get("sections") or []:
            br = sec.get("bounding_rect", [])
            xs.extend([float(p[0]) + ox for p in br] if br else [])
            ys.extend([float(p[1]) + oy for p in br] if br else [])
    if xs and ys:
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        for ax in axs:
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            zmax = num_floors * wall_height + 500
            ax.set_zlim(0, zmax)
            # keep aspect ratio visually consistent (prevents "stretched" houses)
            try:
                dx = maxx - minx
                dy = maxy - miny
                dz = zmax
                ax.set_box_aspect((dx if dx > 0 else 1, dy if dy > 0 else 1, dz if dz > 0 else 1))
            except Exception:
                pass

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def visualize_3d_pyramid_matplotlib(
    output_path: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    all_floor_paths: Optional[List[str]] = None,
    floor_roof_results: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """
    Render `house_3d_pyramid.png` using the same Matplotlib style as standard, but with pyramid ends.
    """
    _ensure_matplotlib_cache_dirs()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union
    from shapely import affinity as shapely_affinity

    from roof_calc.visualize import _free_roof_ends
    from roof_calc.visualize_3d_pyvista import _roof_section_faces_pyramid

    config = config or {}
    roof_angle_deg = float(config.get("roof_angle", 30.0))
    wall_height = float(config.get("wall_height", 300.0))
    colors = config.get("colors", ["#FF6B6B", "#4ECDC4", "#45B7D1"])
    views = config.get("views", [{"elev": 30, "azim": 45, "title": "Vedere Sud-Est"}, {"elev": 20, "azim": 225, "title": "Vedere Nord-Vest"}])
    roof_angle_rad = float(np.radians(roof_angle_deg))
    show_edges = bool(config.get("show_edges", False))
    wall_show_edges = bool(config.get("wall_show_edges", show_edges))
    roof_show_edges = bool(config.get("roof_show_edges", show_edges))
    edge_color = str(config.get("edge_color", "black"))
    wall_edge_color = str(config.get("wall_edge_color", edge_color))
    roof_edge_color = str(config.get("roof_edge_color", edge_color))
    edge_width = float(config.get("edge_width", 0.4))

    if not all_floor_paths or not floor_roof_results or len(all_floor_paths) != len(floor_roof_results):
        return False

    offsets = _compute_offsets(all_floor_paths, floor_roof_results)
    floors_payload: List[Tuple[str, Any, Dict[str, Any], Tuple[int, int]]] = []
    for p, rr in zip(all_floor_paths, floor_roof_results):
        poly0 = _polygon_from_path(p)
        if poly0 is None:
            continue
        ox, oy = offsets.get(p, (0, 0))
        poly_t = shapely_affinity.translate(poly0, xoff=ox, yoff=oy)
        floors_payload.append((p, poly_t, rr, (ox, oy)))
    if not floors_payload:
        return False

    floors_payload.sort(key=lambda t: float(getattr(t[1], "area", 0.0) or 0.0), reverse=True)
    num_floors = len(floors_payload)

    fig = plt.figure(figsize=(16, 7))
    axs = []
    for i in range(2):
        axs.append(fig.add_subplot(1, 2, i + 1, projection="3d"))
    for ax, v in zip(axs, views[:2]):
        ax.set_title(v.get("title", "View"))
        ax.view_init(elev=float(v.get("elev", 30)), azim=float(v.get("azim", 45)))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_facecolor("white")

    for floor_idx, (_p, poly, rr, (ox, oy)) in enumerate(floors_payload):
        z0 = floor_idx * wall_height
        z1 = (floor_idx + 1) * wall_height

        coords = list(poly.exterior.coords) if hasattr(poly, "exterior") else []
        wall_faces = []
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            wall_faces.append([[p1[0], p1[1], z0], [p2[0], p2[1], z0], [p2[0], p2[1], z1], [p1[0], p1[1], z1]])

        union_above = None
        if floor_idx + 1 < len(floors_payload):
            union_above = unary_union([fp[1] for fp in floors_payload[floor_idx + 1 :]])

        # upper sections in this floor coords (for end suppression)
        upper_secs_all: List[Dict[str, Any]] = []
        for _pp, _poly_u, rr_u, (ox_u, oy_u) in floors_payload[floor_idx + 1 :]:
            upper_secs_all.extend(_translate_sections(rr_u.get("sections") or [], float(ox_u - ox), float(oy_u - oy)))

        secs = _translate_sections(rr.get("sections") or [], float(ox), float(oy))
        conns = rr.get("connections") or []
        free_ends = _free_roof_ends(secs, conns)

        roof_faces: List[List[List[float]]] = []
        for s_idx, sec in enumerate(secs):
            br = sec.get("bounding_rect", [])
            if len(br) < 3:
                continue
            sp = ShapelyPolygon(br)
            if _covered_by_upper(sp, union_above, area_thresh=500.0):
                continue
            fe = free_ends[s_idx] if s_idx < len(free_ends) else {}
            roof_faces.extend(_roof_section_faces_pyramid(sec, z1, roof_angle_rad, fe, upper_secs_all))

        for ax in axs:
            if wall_faces:
                wall_coll = Poly3DCollection(
                    wall_faces,
                    facecolors="#B0B0B0",
                    edgecolors=(wall_edge_color if wall_show_edges else "none"),
                    linewidths=(edge_width if wall_show_edges else 0.0),
                    alpha=1.0,
                    zsort="average",
                )
                wall_coll.set_antialiased(False)
                ax.add_collection3d(wall_coll)
            for idx, face in enumerate(roof_faces):
                col = colors[idx % len(colors)]
                roof_coll = Poly3DCollection(
                    [face],
                    facecolors=col,
                    edgecolors=(roof_edge_color if roof_show_edges else "none"),
                    linewidths=(edge_width if roof_show_edges else 0.0),
                    alpha=1.0,
                    zsort="average",
                )
                roof_coll.set_antialiased(False)
                ax.add_collection3d(roof_coll)

    xs = []
    ys = []
    for _p, poly, rr, (ox, oy) in floors_payload:
        coords = list(poly.exterior.coords) if hasattr(poly, "exterior") else []
        xs.extend([c[0] for c in coords])
        ys.extend([c[1] for c in coords])
        for sec in rr.get("sections") or []:
            br = sec.get("bounding_rect", [])
            xs.extend([float(p[0]) + ox for p in br] if br else [])
            ys.extend([float(p[1]) + oy for p in br] if br else [])
    if xs and ys:
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        for ax in axs:
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            zmax = num_floors * wall_height + 500
            ax.set_zlim(0, zmax)
            try:
                dx = maxx - minx
                dy = maxy - miny
                dz = zmax
                ax.set_box_aspect((dx if dx > 0 else 1, dy if dy > 0 else 1, dz if dz > 0 else 1))
            except Exception:
                pass

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return True

