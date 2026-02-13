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


def _z_roof_at(
    roof_faces: List[Dict[str, Any]],
    x: float,
    y: float,
    default_z: float,
    tol: float = 15.0,
) -> float:
    """Înălțimea acoperișului la (x,y). Buffer pentru margini; fallback la fața cea mai apropiată."""
    best: Optional[float] = None
    best_dist: Optional[float] = None
    try:
        from shapely.geometry import Point
        from shapely.geometry import Polygon as ShapelyPolygon
    except Exception:
        return default_z
    pt = Point(x, y)
    for f in roof_faces:
        vs = f.get("vertices_3d") or []
        if len(vs) < 3:
            continue
        xy = [(float(v[0]), float(v[1])) for v in vs]
        xs = [p[0] for p in xy]
        ys = [p[1] for p in xy]
        if x < min(xs) - tol or x > max(xs) + tol or y < min(ys) - tol or y > max(ys) + tol:
            continue
        try:
            poly = ShapelyPolygon(xy)
            if poly.is_empty:
                continue
            inside = poly.contains(pt) or poly.buffer(tol).contains(pt)
            dist = float(poly.distance(pt)) if not inside else 0.0
            if not inside and dist > tol:
                continue
        except Exception:
            continue
        v0, v1, v2 = vs[0], vs[1], vs[2]
        x0, y0, z0 = float(v0[0]), float(v0[1]), float(v0[2])
        x1, y1, z1 = float(v1[0]), float(v1[1]), float(v1[2])
        x2, y2, z2 = float(v2[0]), float(v2[1]), float(v2[2])
        denom = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(denom) < 1e-12:
            z = max(z0, z1, z2)
        else:
            w1 = ((x - x0) * (y2 - y0) - (x2 - x0) * (y - y0)) / denom
            w2 = ((x1 - x0) * (y - y0) - (x - x0) * (y1 - y0)) / denom
            z = (1.0 - w1 - w2) * z0 + w1 * z1 + w2 * z2
        if best is None or (dist <= (best_dist or 1e9) and z > (best or -1e9)):
            best = z
            best_dist = dist
    return float(best) if best is not None else default_z


def _compute_offsets(paths: List[str], results: List[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
    """Etaje cu aceeași mărime și formă se pun unul peste altul: aliniem centrele (centroid poligon)."""
    polys = [_polygon_from_path(p) for p in paths]
    best_idx = 0
    best_area = -1.0
    for i, poly in enumerate(polys):
        if poly is not None and not getattr(poly, "is_empty", True):
            a = float(getattr(poly, "area", 0) or 0)
            if a > best_area:
                best_area = a
                best_idx = i
    ref_poly = polys[best_idx] if best_idx < len(polys) else None
    if ref_poly is None or getattr(ref_poly, "is_empty", True):
        cx_ref, cy_ref = 0.0, 0.0
    else:
        c = ref_poly.centroid
        cx_ref, cy_ref = float(c.x), float(c.y)
    out: Dict[str, Tuple[int, int]] = {}
    for p, poly in zip(paths, polys):
        if poly is None or getattr(poly, "is_empty", True):
            out[p] = (int(round(-cx_ref)), int(round(-cy_ref)))
        else:
            c = poly.centroid
            ox = int(round(cx_ref - c.x))
            oy = int(round(cy_ref - c.y))
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


def _triangulate_face(face_pts: List[List[float]]) -> List[List[List[float]]]:
    if len(face_pts) == 3:
        return [face_pts]
    if len(face_pts) == 4:
        a, b, c, d = face_pts
        return [[a, b, c], [a, c, d]]
    return []


def _triangulate_fan(pts: List[List[float]]) -> List[List[List[float]]]:
    """Triangulare fan pentru poligoane cu 5+ vârfuri."""
    if len(pts) < 3:
        return []
    if len(pts) == 3:
        return [pts]
    return [[pts[0], pts[i], pts[i + 1]] for i in range(1, len(pts) - 1)]


def _z_at_point_pyramid(
    x: float, y: float,
    sec: Dict[str, Any],
    base_z: float,
    roof_angle_rad: float,
) -> float:
    br = sec.get("bounding_rect", [])
    ridge = sec.get("ridge_line", [])
    orient = str(sec.get("ridge_orientation", "horizontal"))
    if len(br) < 3:
        return base_z
    minx = min(p[0] for p in br)
    maxx = max(p[0] for p in br)
    miny = min(p[1] for p in br)
    maxy = max(p[1] for p in br)
    if len(ridge) >= 2:
        ridge_mid_x = (ridge[0][0] + ridge[1][0]) / 2
        ridge_mid_y = (ridge[0][1] + ridge[1][1]) / 2
    else:
        ridge_mid_x = (minx + maxx) / 2
        ridge_mid_y = (miny + maxy) / 2
    span = (maxy - miny) / 2 if orient == "horizontal" else (maxx - minx) / 2
    ridge_height = base_z + span * float(np.tan(roof_angle_rad))
    tanv = float(np.tan(roof_angle_rad))
    d = abs(y - ridge_mid_y) if orient == "horizontal" else abs(x - ridge_mid_x)
    return float(ridge_height - d * tanv)


def _clip_pyramid_faces_with_polygon(
    faces: List[List[List[float]]],
    sec: Dict[str, Any],
    base_z: float,
    roof_angle_rad: float,
    clip_poly: Any,
) -> List[List[List[float]]]:
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
    except Exception:
        return faces
    if clip_poly is None or getattr(clip_poly, "is_empty", True):
        return faces
    result: List[List[List[float]]] = []
    for face in faces:
        tris = _triangulate_face(face)
        for tri in tris:
            if len(tri) != 3:
                continue
            xy = [(float(p[0]), float(p[1])) for p in tri]
            try:
                tri_poly = ShapelyPolygon(xy)
            except Exception:
                continue
            try:
                inter = tri_poly.intersection(clip_poly)
            except Exception:
                continue
            if inter is None or inter.is_empty:
                continue
            geoms = getattr(inter, "geoms", [inter])
            for g in geoms:
                if getattr(g, "is_empty", True):
                    continue
                ext = getattr(g, "exterior", None)
                if ext is None:
                    continue
                coords = list(ext.coords)[:-1]
                if len(coords) < 3:
                    continue
                c0 = coords[0]
                for i in range(1, len(coords) - 1):
                    c1, c2 = coords[i], coords[i + 1]
                    x0, y0 = float(c0[0]), float(c0[1])
                    x1, y1 = float(c1[0]), float(c1[1])
                    x2, y2 = float(c2[0]), float(c2[1])
                    z0 = _z_at_point_pyramid(x0, y0, sec, base_z, roof_angle_rad)
                    z1 = _z_at_point_pyramid(x1, y1, sec, base_z, roof_angle_rad)
                    z2 = _z_at_point_pyramid(x2, y2, sec, base_z, roof_angle_rad)
                    result.append([[x0, y0, z0], [x1, y1, z1], [x2, y2, z2]])
    return result if result else faces


def visualize_3d_standard_matplotlib(
    output_path: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    all_floor_paths: Optional[List[str]] = None,
    floor_roof_results: Optional[List[Dict[str, Any]]] = None,
    roof_levels: Optional[List[Tuple[float, Dict[str, Any], float, float, int]]] = None,
    lower_floor_roof_mode: str = "standard",
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
    overhang_px = float(config.get("overhang_px", 0.0))
    overhang_keep_height = bool(config.get("overhang_keep_height", True))
    overhang_drop_down = bool(config.get("overhang_drop_down", True))
    overhang_shift_whole_roof_down = bool(config.get("overhang_shift_whole_roof_down", False))
    overhang_shift_whole_roof_down = bool(config.get("overhang_shift_whole_roof_down", False))

    from roof_calc.overhang import (
        apply_overhang_to_sections,
        clip_roof_faces_to_polygon,
        compute_overhang_sides_from_free_ends,
        compute_overhang_sides_from_footprint,
        compute_overhang_sides_from_union_boundary,
        extend_sections_to_connect,
        extend_secondary_sections_to_main_ridge,
        get_faces_3d_aframe_with_magenta,
        ridge_intersection_corner_lines,
        get_downspout_faces_for_floors,
        get_downspout_faces_pyramid,
        get_drip_edge_faces_3d,
        get_gutter_end_closures_3d,
        get_gutter_endpoints_3d,
        get_gutter_faces_3d,
    )

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

    roof_faces_by_floor: Dict[int, List[Dict[str, Any]]] = {}
    roof_faces_base_by_floor: Dict[int, List[Dict[str, Any]]] = {}
    use_shed_lower = lower_floor_roof_mode == "shed"
    if roof_levels:
        try:
            if use_shed_lower:
                from roof_calc.visualize_3d_pyvista import _roof_section_faces_shed  # type: ignore
                from roof_calc.overhang import high_side_for_shed_from_upper_floor
            for z_base, roof_data, dx, dy, _fl in roof_levels:
                fl = int(_fl) if _fl is not None else 0
                secs0 = roof_data.get("sections") or []
                conns0 = roof_data.get("connections") or []
                secs_t = _translate_sections(secs0, float(dx), float(dy))
                if secs_t and conns0:
                    secs_t = extend_sections_to_connect(secs_t, conns0)
                if secs_t and not use_shed_lower:
                    secs_t = extend_secondary_sections_to_main_ridge(secs_t)
                # Base (fără overhang) pentru cliparea pereților
                if use_shed_lower:
                    union_upper = unary_union([floors_payload[i][1] for i in range(fl + 1, len(floors_payload))]) if 0 <= fl < len(floors_payload) else None
                    high_sides = high_side_for_shed_from_upper_floor(secs_t, union_upper)
                    faces_base_fl = []
                    roof_angle_rad = np.radians(roof_angle_deg)
                    for s_idx, sec in enumerate(secs_t):
                        hs = high_sides[s_idx] if s_idx < len(high_sides) else "top"
                        for face in _roof_section_faces_shed(sec, float(z_base), roof_angle_rad, hs):
                            faces_base_fl.append({"vertices_3d": face})
                else:
                    cl_base = ridge_intersection_corner_lines(secs_t)
                    faces_base_fl = get_faces_3d_aframe_with_magenta(
                        secs_t, conns0, roof_angle_deg=roof_angle_deg, wall_height=float(z_base),
                        corner_lines=cl_base,
                    )
                footprint_fl = floors_payload[fl][1] if 0 <= fl < len(floors_payload) else None
                if footprint_fl is not None and faces_base_fl:
                    faces_base_fl = clip_roof_faces_to_polygon(faces_base_fl, footprint_fl)
                roof_faces_base_by_floor[fl] = faces_base_fl or []
                if overhang_px > 0 and secs_t:
                    footprint = floors_payload[fl][1] if 0 <= fl < len(floors_payload) else None
                    free = (
                        compute_overhang_sides_from_footprint(secs_t, footprint)
                        if footprint
                        else compute_overhang_sides_from_union_boundary(secs_t)
                    )
                    secs_t = apply_overhang_to_sections(secs_t, overhang_px=overhang_px, free_sides=free)
                if use_shed_lower:
                    union_upper = unary_union([floors_payload[i][1] for i in range(fl + 1, len(floors_payload))]) if 0 <= fl < len(floors_payload) else None
                    high_sides = high_side_for_shed_from_upper_floor(secs_t, union_upper)
                    faces = []
                    roof_angle_rad = np.radians(roof_angle_deg)
                    for s_idx, sec in enumerate(secs_t):
                        hs = high_sides[s_idx] if s_idx < len(high_sides) else "top"
                        for face in _roof_section_faces_shed(sec, float(z_base), roof_angle_rad, hs):
                            faces.append({"vertices_3d": face})
                else:
                    cl_use = ridge_intersection_corner_lines(secs_t)
                    faces = get_faces_3d_aframe_with_magenta(
                        secs_t, conns0, roof_angle_deg=roof_angle_deg, wall_height=float(z_base),
                        corner_lines=cl_use,
                    )
                if overhang_px > 0 and overhang_shift_whole_roof_down and faces:
                    import math
                    dz = float(math.tan(math.radians(roof_angle_deg)) * overhang_px)
                    for f in faces:
                        vs = f.get("vertices_3d") or []
                        f["vertices_3d"] = [[float(x), float(y), float(z) - dz] for x, y, z in vs]
                if footprint_fl is not None and faces:
                    faces = clip_roof_faces_to_polygon(faces, footprint_fl)
                roof_faces_by_floor[fl] = faces
        except Exception:
            pass

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
    # also collect bounds from FINAL geometry (after overhang), then add padding to avoid clipping
    xs_all: List[float] = []
    ys_all: List[float] = []
    gutter_endpoints: List[Tuple[float, float, float]] = []
    gutter_segment_sections: List[List[Dict[str, Any]]] = []
    gutter_closure_calls: List[Tuple[tuple, dict]] = []
    for floor_idx, (_p, poly, rr, (ox, oy)) in enumerate(floors_payload):
        z0 = floor_idx * wall_height
        z1 = (floor_idx + 1) * wall_height

        coords = list(poly.exterior.coords) if hasattr(poly, "exterior") else []
        xs_all.extend([float(c[0]) for c in coords])
        ys_all.extend([float(c[1]) for c in coords])

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

        # roof faces (standard): top-floor roof only (lower floors use `roof_levels`)
        draw_roof = floor_idx == (num_floors - 1)
        kept_extended = extend_sections_to_connect(kept, conns_kept) if (draw_roof and kept and conns_kept) else kept
        if draw_roof and kept_extended:
            kept_extended = extend_secondary_sections_to_main_ridge(kept_extended)

        # baseline (no overhang) height – folosim linii magenta pentru secțiunile cu intersecții
        corner_lines_base = ridge_intersection_corner_lines(kept_extended) if draw_roof and kept_extended else []
        roof_faces_base = (
            get_faces_3d_aframe_with_magenta(
                kept_extended, conns_kept, roof_angle_deg=roof_angle_deg, wall_height=z1,
                corner_lines=corner_lines_base, floor_polygon=floor_poly,
            )
            if draw_roof and kept_extended
            else []
        )

        kept_use = kept_extended
        if draw_roof and overhang_px > 0 and kept_extended:
            free = (
                compute_overhang_sides_from_footprint(kept_extended, poly)
                if poly is not None and not getattr(poly, "is_empty", True)
                else compute_overhang_sides_from_union_boundary(kept_extended)
            )
            kept_use = apply_overhang_to_sections(kept_extended, overhang_px=overhang_px, free_sides=free)

        corner_lines_use = ridge_intersection_corner_lines(kept_use) if draw_roof and kept_use else []
        roof_faces = (
            get_faces_3d_aframe_with_magenta(
                kept_use, conns_kept, roof_angle_deg=roof_angle_deg, wall_height=z1,
                corner_lines=corner_lines_use, floor_polygon=floor_poly,
            )
            if draw_roof and kept_use
            else []
        )
        if roof_faces and poly is not None and not getattr(poly, "is_empty", True):
            roof_faces = clip_roof_faces_to_polygon(roof_faces, poly)
        if roof_faces_base and poly is not None and not getattr(poly, "is_empty", True):
            roof_faces_base = clip_roof_faces_to_polygon(roof_faces_base, poly)

        # Keep the same ridge height as baseline (do NOT raise roof due to overhang)
        if overhang_px > 0 and overhang_keep_height and roof_faces_base and roof_faces:
            try:
                base_z = float(z1)
                maxz_base = max(float(v[2]) for f in roof_faces_base for v in (f.get("vertices_3d") or []))
                maxz_new = max(float(v[2]) for f in roof_faces for v in (f.get("vertices_3d") or []))
                d_base = maxz_base - base_z
                d_new = maxz_new - base_z
                if d_base > 1e-6 and d_new > 1e-6:
                    s = d_base / d_new
                    for f in roof_faces:
                        vs = f.get("vertices_3d") or []
                        f["vertices_3d"] = [[float(x), float(y), base_z + (float(z) - base_z) * s] for x, y, z in vs]
            except Exception:
                pass

        # User rule: after generating roof with overhang and keeping peak height,
        # shift the whole roof down by Δz = tan(angle) * overhang_px.
        if overhang_px > 0 and overhang_shift_whole_roof_down and roof_faces:
            try:
                import math

                dz = float(math.tan(math.radians(roof_angle_deg)) * float(overhang_px))
                for f in roof_faces:
                    vs = f.get("vertices_3d") or []
                    if not vs:
                        continue
                    f["vertices_3d"] = [[float(x), float(y), float(z) - dz] for x, y, z in vs]
            except Exception:
                pass

        # Make overhang drop BELOW the wall top (base_z), while keeping the original roof unchanged.
        # We only adjust vertices that are further from the ridge than the original (no-overhang) run.
        if overhang_px > 0 and overhang_drop_down and roof_faces and kept:
            try:
                import math, re

                tanv = float(math.tan(math.radians(roof_angle_deg)))
                # Peak Z per section index, based on current (possibly scaled) faces
                peaks: Dict[int, float] = {}
                for f in roof_faces:
                    lbl = str(f.get("label") or "")
                    m = re.search(r"sec(\\d+)_", lbl)
                    if not m:
                        continue
                    si = int(m.group(1))
                    vs = f.get("vertices_3d") or []
                    if not vs:
                        continue
                    mz = max(float(v[2]) for v in vs)
                    peaks[si] = max(peaks.get(si, float("-inf")), mz)

                for f in roof_faces:
                    lbl = str(f.get("label") or "")
                    m = re.search(r"sec(\\d+)_", lbl)
                    if not m:
                        continue
                    si = int(m.group(1))
                    if si < 0 or si >= len(kept):
                        continue
                    peak_z = float(peaks.get(si, z1))
                    base_sec = kept[si]
                    br0 = base_sec.get("bounding_rect") or []
                    if len(br0) < 3:
                        continue
                    xs0 = [float(p[0]) for p in br0]
                    ys0 = [float(p[1]) for p in br0]
                    minx0, maxx0 = min(xs0), max(xs0)
                    miny0, maxy0 = min(ys0), max(ys0)
                    orient = str(base_sec.get("ridge_orientation", "horizontal"))
                    ridge = base_sec.get("ridge_line") or []
                    if len(ridge) >= 2:
                        rx0, ry0 = float(ridge[0][0]), float(ridge[0][1])
                        rx1, ry1 = float(ridge[1][0]), float(ridge[1][1])
                        ridge_x = (rx0 + rx1) / 2.0
                        ridge_y = (ry0 + ry1) / 2.0
                    else:
                        ridge_x = (minx0 + maxx0) / 2.0
                        ridge_y = (miny0 + maxy0) / 2.0

                    if orient == "vertical":
                        # Use footprint band intersection so the “kink” is at the wall line,
                        # not at an inset rectangle boundary (prevents “roof floating”).
                        try:
                            from shapely.geometry import box as _box

                            big = 1e6
                            band = _box(-big, miny0, big, maxy0)
                            fp = poly  # footprint for this floor (overlay coords)
                            inter = fp.intersection(band) if fp is not None else None
                            if inter is not None and (not getattr(inter, "is_empty", True)):
                                minx_fp, _miny_fp, maxx_fp, _maxy_fp = inter.bounds
                                minx0, maxx0 = float(minx_fp), float(maxx_fp)
                        except Exception:
                            pass
                        d0 = max(abs(minx0 - ridge_x), abs(maxx0 - ridge_x))
                        def d_perp(x: float, y: float) -> float:  # noqa: ARG001
                            return abs(x - ridge_x)
                    else:
                        try:
                            from shapely.geometry import box as _box

                            big = 1e6
                            band = _box(minx0, -big, maxx0, big)
                            fp = poly
                            inter = fp.intersection(band) if fp is not None else None
                            if inter is not None and (not getattr(inter, "is_empty", True)):
                                _minx_fp, miny_fp, _maxx_fp, maxy_fp = inter.bounds
                                miny0, maxy0 = float(miny_fp), float(maxy_fp)
                        except Exception:
                            pass
                        d0 = max(abs(miny0 - ridge_y), abs(maxy0 - ridge_y))
                        def d_perp(x: float, y: float) -> float:  # noqa: ARG001
                            return abs(y - ridge_y)

                    new_vs = []
                    for x, y, z in (f.get("vertices_3d") or []):
                        x = float(x); y = float(y); z = float(z)
                        d1 = float(d_perp(x, y))
                        # Only change the overhang-only area (outside original run)
                        if d1 > d0 + 1e-6:
                            z = peak_z - tanv * d1
                        new_vs.append([x, y, z])
                    f["vertices_3d"] = new_vs
            except Exception:
                pass

        # Pereți prelungiți până la acoperiș (formă geometrică: segment perete + contur acoperiș)
        faces_for_wall_z = roof_faces_base if (draw_roof and roof_faces_base) else roof_faces_base_by_floor.get(floor_idx, [])
        if not faces_for_wall_z and (draw_roof and roof_faces):
            faces_for_wall_z = roof_faces
        if not faces_for_wall_z:
            faces_for_wall_z = roof_faces_by_floor.get(floor_idx, [])
        wall_faces = []
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])
            if faces_for_wall_z:
                # Pereți: min(z1, zt) – tăiem orice bucată care iese peste outline-ul acoperișului (fără overhang)
                n_pts = 13
                pts_top = []
                for k in range(n_pts):
                    t = k / (n_pts - 1) if n_pts > 1 else 1.0
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    zt = _z_roof_at(faces_for_wall_z, x, y, z1, tol=40.0)
                    pts_top.append([x, y, min(z1, zt)])
                face_pts = [[x1, y1, z0], [x2, y2, z0]] + [list(p) for p in reversed(pts_top)]
                for j in range(1, len(face_pts) - 1):
                    wall_faces.append([face_pts[0], face_pts[j], face_pts[j + 1]])
            else:
                wall_faces.append(
                    [[x1, y1, z0], [x2, y2, z0], [x2, y2, z1], [x1, y1, z1]]
                )

        roof_polys = [f["vertices_3d"] for f in roof_faces if f.get("vertices_3d")]
        drip_polys: List[List[List[float]]] = []
        gutter_polys: List[List[List[float]]] = []
        if draw_roof and overhang_px > 0 and kept_use:
            import math as _m
            roof_shift_dz = float(_m.tan(_m.radians(roof_angle_deg)) * overhang_px) if overhang_shift_whole_roof_down else 0.0
            free_drip = compute_overhang_sides_from_union_boundary(kept_use)
            for df in get_drip_edge_faces_3d(
                kept_use, float(z1), overhang_px, roof_angle_deg,
                roof_shift_dz=roof_shift_dz, roof_faces=roof_faces, free_sides=free_drip
            ):
                v = df.get("vertices_3d")
                if v:
                    drip_polys.append(v)
                    for vx, vy, _ in v:
                        xs_all.append(float(vx))
                        ys_all.append(float(vy))
            for gf in get_gutter_faces_3d(
                kept_use, float(z1), overhang_px, roof_angle_deg,
                roof_shift_dz=roof_shift_dz, roof_faces=roof_faces,
                include_eaves_only=True,
                eaves_z_lift=overhang_px * 0.60,
            ):
                v = gf.get("vertices_3d")
                if v:
                    gutter_polys.append(v)
                    for vx, vy, _ in v:
                        xs_all.append(float(vx))
                        ys_all.append(float(vy))
            gutter_closure_calls.append((
                (kept_use, float(z1), overhang_px, roof_angle_deg),
                {"roof_shift_dz": roof_shift_dz, "roof_faces": roof_faces, "include_eaves_only": True, "eaves_z_lift": overhang_px * 0.60},
            ))
            ep = get_gutter_endpoints_3d(
                kept_use, float(z1), overhang_px, roof_angle_deg,
                roof_shift_dz=roof_shift_dz, roof_faces=roof_faces,
                include_eaves_only=True,
                eaves_z_lift=overhang_px * 0.60,
            )
            gutter_endpoints.extend(ep)
            for _ in range(len(ep) // 2):
                gutter_segment_sections.append(kept_use)

        for face in roof_polys:
            for vx, vy, _vz in face:
                xs_all.append(float(vx))
                ys_all.append(float(vy))

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
            for dp in drip_polys:
                drip_coll = Poly3DCollection(
                    [dp],
                    facecolors="#8B4513",
                    edgecolors=(roof_edge_color if roof_show_edges else "none"),
                    linewidths=(edge_width if roof_show_edges else 0.0),
                    alpha=1.0,
                    zsort="average",
                )
                drip_coll.set_antialiased(False)
                ax.add_collection3d(drip_coll)
            for gp in gutter_polys:
                gutter_coll = Poly3DCollection(
                    [gp],
                    facecolors="#6B7280",
                    edgecolors=(roof_edge_color if roof_show_edges else "none"),
                    linewidths=(edge_width if roof_show_edges else 0.0),
                    alpha=1.0,
                    zsort="average",
                )
                gutter_coll.set_antialiased(False)
                ax.add_collection3d(gutter_coll)

    # Burlani (cilindri verticali + legături la streașină) la fiecare colț exterior
    gutter_radius = max(2.0, overhang_px * 0.24) * 0.70 if overhang_px > 0 else None
    downspout_result = get_downspout_faces_for_floors(
        floors_payload,
        wall_height,
        cylinder_radius=gutter_radius,
        gutter_endpoints=gutter_endpoints,
        gutter_segment_sections=gutter_segment_sections if gutter_segment_sections else None,
        return_used_endpoints=True,
    )
    if isinstance(downspout_result, tuple):
        downspout_faces, used_endpoints = downspout_result
    else:
        downspout_faces = downspout_result
        used_endpoints = []
    for ax in axs:
        for df in downspout_faces:
            v = df.get("vertices_3d")
            if v:
                for vx, vy, _ in v:
                    xs_all.append(float(vx))
                    ys_all.append(float(vy))
                dp_coll = Poly3DCollection(
                    [v],
                    facecolors=df.get("color", "#6B7280"),
                    edgecolors="none",
                    alpha=1.0,
                    zsort="average",
                )
                dp_coll.set_antialiased(False)
                ax.add_collection3d(dp_coll)
        for args, kwargs in gutter_closure_calls:
            kwargs = dict(kwargs, downspout_endpoints=used_endpoints)
            for gf in get_gutter_end_closures_3d(*args, **kwargs):
                v = gf.get("vertices_3d")
                if v:
                    for vx, vy, _ in v:
                        xs_all.append(float(vx))
                        ys_all.append(float(vy))
                    gutter_coll = Poly3DCollection(
                        [v],
                        facecolors="#6B7280",
                        edgecolors=(roof_edge_color if roof_show_edges else "none"),
                        linewidths=(edge_width if roof_show_edges else 0.0),
                        alpha=1.0,
                        zsort="average",
                    )
                    gutter_coll.set_antialiased(False)
                    ax.add_collection3d(gutter_coll)

    # Add lower-floor roofs from `roof_levels` (remaining areas only)
    use_shed_lower = lower_floor_roof_mode == "shed"
    if use_shed_lower:
        from roof_calc.visualize_3d_pyvista import _roof_section_faces_shed
        from roof_calc.overhang import high_side_for_shed_from_upper_floor

    if roof_levels:
        try:
            for z_base, roof_data, dx, dy, _fl in roof_levels:
                secs0 = roof_data.get("sections") or []
                conns0 = roof_data.get("connections") or []
                base_secs_t = _translate_sections(secs0, float(dx), float(dy))

                if use_shed_lower:
                    union_upper = None
                    if _fl is not None and 0 <= int(_fl) < len(floors_payload):
                        try:
                            polys_above = [floors_payload[i][1] for i in range(int(_fl) + 1, len(floors_payload))]
                            if polys_above:
                                union_upper = unary_union(polys_above)
                        except Exception:
                            pass
                    high_sides_base = high_side_for_shed_from_upper_floor(base_secs_t, union_upper)
                    roof_angle_rad = float(np.radians(roof_angle_deg))
                    faces_base = []
                    for s_idx, sec in enumerate(base_secs_t):
                        hs = high_sides_base[s_idx] if s_idx < len(high_sides_base) else "top"
                        for face in _roof_section_faces_shed(sec, float(z_base), roof_angle_rad, hs):
                            faces_base.append({"vertices_3d": face})
                else:
                    faces_base = rf.get_faces_3d_standard(
                        base_secs_t, conns0, roof_angle_deg=roof_angle_deg, wall_height=float(z_base)
                    )

                secs_t = base_secs_t
                if overhang_px > 0 and secs_t:
                    footprint = None
                    try:
                        if _fl is not None and 0 <= int(_fl) < len(floors_payload):
                            footprint = floors_payload[int(_fl)][1]
                    except Exception:
                        footprint = None
                    free = (
                        compute_overhang_sides_from_footprint(secs_t, footprint)
                        if footprint is not None
                        else compute_overhang_sides_from_union_boundary(secs_t)
                    )
                    secs_t = apply_overhang_to_sections(secs_t, overhang_px=overhang_px, free_sides=free)

                if use_shed_lower:
                    high_sides = high_side_for_shed_from_upper_floor(secs_t, union_upper)
                    faces = []
                    for s_idx, sec in enumerate(secs_t):
                        hs = high_sides[s_idx] if s_idx < len(high_sides) else "top"
                        for face in _roof_section_faces_shed(sec, float(z_base), roof_angle_rad, hs):
                            faces.append({"vertices_3d": face})
                else:
                    faces = rf.get_faces_3d_standard(
                        secs_t, conns0, roof_angle_deg=roof_angle_deg, wall_height=float(z_base)
                    )

                # Keep peak height same as baseline (do NOT raise roof due to overhang)
                if overhang_px > 0 and overhang_keep_height and faces_base and faces:
                    try:
                        base_z = float(z_base)
                        maxz_base = max(float(v[2]) for f in faces_base for v in (f.get("vertices_3d") or []))
                        maxz_new = max(float(v[2]) for f in faces for v in (f.get("vertices_3d") or []))
                        d_base = maxz_base - base_z
                        d_new = maxz_new - base_z
                        if d_base > 1e-6 and d_new > 1e-6:
                            s = d_base / d_new
                            for f in faces:
                                vs = f.get("vertices_3d") or []
                                f["vertices_3d"] = [[float(x), float(y), base_z + (float(z) - base_z) * s] for x, y, z in vs]
                    except Exception:
                        pass

                # Shift whole roof down after keep-height scaling
                if overhang_px > 0 and overhang_shift_whole_roof_down and faces:
                    try:
                        import math

                        dz = float(math.tan(math.radians(roof_angle_deg)) * float(overhang_px))
                        for f in faces:
                            vs = f.get("vertices_3d") or []
                            if not vs:
                                continue
                            f["vertices_3d"] = [[float(x), float(y), float(z) - dz] for x, y, z in vs]
                    except Exception:
                        pass

                # Drop ONLY the overhang part below this level's wall-top (z_base) (gable; skip for shed)
                if not use_shed_lower and overhang_px > 0 and overhang_drop_down and faces and secs0:
                    try:
                        import math, re

                        tanv = float(math.tan(math.radians(roof_angle_deg)))
                        base_secs_local = _translate_sections(secs0, float(dx), float(dy))

                        peaks: Dict[int, float] = {}
                        for f in faces:
                            lbl = str(f.get("label") or "")
                            m = re.search(r"sec(\\d+)_", lbl)
                            if not m:
                                continue
                            si = int(m.group(1))
                            vs = f.get("vertices_3d") or []
                            if not vs:
                                continue
                            mz = max(float(v[2]) for v in vs)
                            peaks[si] = max(peaks.get(si, float("-inf")), mz)

                        for f in faces:
                            lbl = str(f.get("label") or "")
                            m = re.search(r"sec(\\d+)_", lbl)
                            if not m:
                                continue
                            si = int(m.group(1))
                            if si < 0 or si >= len(base_secs_local):
                                continue
                            peak_z = float(peaks.get(si, float(z_base)))
                            base_sec = base_secs_local[si]
                            br0 = base_sec.get("bounding_rect") or []
                            if len(br0) < 3:
                                continue
                            xs0 = [float(p[0]) for p in br0]
                            ys0 = [float(p[1]) for p in br0]
                            minx0, maxx0 = min(xs0), max(xs0)
                            miny0, maxy0 = min(ys0), max(ys0)
                            orient = str(base_sec.get("ridge_orientation", "horizontal"))
                            ridge = base_sec.get("ridge_line") or []
                            if len(ridge) >= 2:
                                rx0, ry0 = float(ridge[0][0]), float(ridge[0][1])
                                rx1, ry1 = float(ridge[1][0]), float(ridge[1][1])
                                ridge_x = (rx0 + rx1) / 2.0
                                ridge_y = (ry0 + ry1) / 2.0
                            else:
                                ridge_x = (minx0 + maxx0) / 2.0
                                ridge_y = (miny0 + maxy0) / 2.0

                            if orient == "vertical":
                                # Use outer floor footprint for the “no-overhang” extent
                                try:
                                    from shapely.geometry import box as _box

                                    fp = footprint
                                    if fp is not None:
                                        big = 1e6
                                        band = _box(-big, miny0, big, maxy0)
                                        inter = fp.intersection(band)
                                        if not getattr(inter, "is_empty", True):
                                            minx_fp, _miny_fp, maxx_fp, _maxy_fp = inter.bounds
                                            minx0, maxx0 = float(minx_fp), float(maxx_fp)
                                except Exception:
                                    pass
                                d0 = max(abs(minx0 - ridge_x), abs(maxx0 - ridge_x))

                                def d_perp(x: float, y: float) -> float:  # noqa: ARG001
                                    return abs(x - ridge_x)

                            else:
                                try:
                                    from shapely.geometry import box as _box

                                    fp = footprint
                                    if fp is not None:
                                        big = 1e6
                                        band = _box(minx0, -big, maxx0, big)
                                        inter = fp.intersection(band)
                                        if not getattr(inter, "is_empty", True):
                                            _minx_fp, miny_fp, _maxx_fp, maxy_fp = inter.bounds
                                            miny0, maxy0 = float(miny_fp), float(maxy_fp)
                                except Exception:
                                    pass
                                d0 = max(abs(miny0 - ridge_y), abs(maxy0 - ridge_y))

                                def d_perp(x: float, y: float) -> float:  # noqa: ARG001
                                    return abs(y - ridge_y)

                            new_vs = []
                            for x, y, z in (f.get("vertices_3d") or []):
                                x = float(x)
                                y = float(y)
                                z = float(z)
                                d1 = float(d_perp(x, y))
                                if d1 > d0 + 1e-6:
                                    z = peak_z - tanv * d1
                                new_vs.append([x, y, z])
                            f["vertices_3d"] = new_vs
                    except Exception:
                        pass

                polys = [f["vertices_3d"] for f in faces if f.get("vertices_3d")]
                drip_polys_lower: List[List[List[float]]] = []
                gutter_polys_lower: List[List[List[float]]] = []
                if overhang_px > 0 and secs_t:
                    import math as _m
                    roof_shift_dz = float(_m.tan(_m.radians(roof_angle_deg)) * overhang_px) if overhang_shift_whole_roof_down else 0.0
                    free_drip = free if overhang_px > 0 else compute_overhang_sides_from_union_boundary(secs_t)
                    if use_shed_lower:
                        _low = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
                        exclude_low_sides = [
                            _low.get(high_sides[i] if i < len(high_sides) else "top", "bottom")
                            for i in range(len(secs_t))
                        ]
                        for df in get_drip_edge_faces_3d(
                            secs_t, float(z_base), overhang_px, roof_angle_deg,
                            roof_shift_dz=roof_shift_dz, roof_faces=faces, free_sides=free_drip,
                            exclude_low_sides=exclude_low_sides,
                        ):
                            v = df.get("vertices_3d")
                            if v:
                                drip_polys_lower.append(v)
                                for vx, vy, _ in v:
                                    xs_all.append(float(vx))
                                    ys_all.append(float(vy))
                        for gf in get_gutter_faces_3d(
                            secs_t, float(z_base), overhang_px, roof_angle_deg,
                            roof_shift_dz=roof_shift_dz, roof_faces=faces,
                            exclude_low_sides=exclude_low_sides,
                        ):
                            v = gf.get("vertices_3d")
                            if v:
                                gutter_polys_lower.append(v)
                                for vx, vy, _ in v:
                                    xs_all.append(float(vx))
                                    ys_all.append(float(vy))
                        for gf in get_gutter_end_closures_3d(
                            secs_t, float(z_base), overhang_px, roof_angle_deg,
                            roof_shift_dz=roof_shift_dz, roof_faces=faces,
                            exclude_low_sides=exclude_low_sides,
                            eaves_z_lift=overhang_px * 0.60,
                            downspout_endpoints=used_endpoints,
                        ):
                            v = gf.get("vertices_3d")
                            if v:
                                gutter_polys_lower.append(v)
                                for vx, vy, _ in v:
                                    xs_all.append(float(vx))
                                    ys_all.append(float(vy))
                        ep = get_gutter_endpoints_3d(
                            secs_t, float(z_base), overhang_px, roof_angle_deg,
                            roof_shift_dz=roof_shift_dz, roof_faces=faces,
                            exclude_low_sides=exclude_low_sides,
                        )
                        gutter_endpoints.extend(ep)
                        for _ in range(len(ep) // 2):
                            gutter_segment_sections.append(secs_t)
                    else:
                        for df in get_drip_edge_faces_3d(
                            secs_t, float(z_base), overhang_px, roof_angle_deg,
                            roof_shift_dz=roof_shift_dz, roof_faces=faces, free_sides=free_drip
                        ):
                            v = df.get("vertices_3d")
                            if v:
                                drip_polys_lower.append(v)
                                for vx, vy, _ in v:
                                    xs_all.append(float(vx))
                                    ys_all.append(float(vy))
                for face in polys:
                    for vx, vy, _vz in face:
                        xs_all.append(float(vx))
                        ys_all.append(float(vy))
                for ax in axs:
                    for idx, face in enumerate(polys):
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
                    for dp in drip_polys_lower:
                        drip_coll = Poly3DCollection(
                            [dp],
                            facecolors="#8B4513",
                            edgecolors=(roof_edge_color if roof_show_edges else "none"),
                            linewidths=(edge_width if roof_show_edges else 0.0),
                            alpha=1.0,
                            zsort="average",
                        )
                        drip_coll.set_antialiased(False)
                        ax.add_collection3d(drip_coll)
                    for gp in gutter_polys_lower:
                        gutter_coll = Poly3DCollection(
                            [gp],
                            facecolors="#6B7280",
                            edgecolors=(roof_edge_color if roof_show_edges else "none"),
                            linewidths=(edge_width if roof_show_edges else 0.0),
                            alpha=1.0,
                            zsort="average",
                        )
                        gutter_coll.set_antialiased(False)
                        ax.add_collection3d(gutter_coll)
        except Exception:
            pass

    # bounds
    if xs_all and ys_all:
        minx, maxx = float(min(xs_all)), float(max(xs_all))
        miny, maxy = float(min(ys_all)), float(max(ys_all))
        # add padding so overhang/roof edges never get clipped by axis limits
        pad_ratio = float(config.get("bounds_pad_ratio", 0.08))
        pad_min = float(config.get("bounds_pad_px", 60.0))
        span = max(maxx - minx, maxy - miny, 1.0)
        pad = max(pad_min, pad_ratio * span)
        minx, maxx = (minx - pad, maxx + pad)
        miny, maxy = (miny - pad, maxy + pad)
        for ax in axs:
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            zmax = num_floors * wall_height + 500 + float(config.get("bounds_pad_z", 0.0))
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


def visualize_3d_shed_matplotlib(
    output_path: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    all_floor_paths: Optional[List[str]] = None,
    floor_roof_results: Optional[List[Dict[str, Any]]] = None,
    roof_levels: Optional[List[Tuple[float, Dict[str, Any], float, float, int]]] = None,
) -> bool:
    """
    Randare 3D acoperiș într-o apă (shed): ultimul etaj = a_frame, restul = o singură pantă.
    Creasta pe latura cea mai apropiată de etajul superior.
    """
    return visualize_3d_standard_matplotlib(
        output_path,
        config=config,
        all_floor_paths=all_floor_paths,
        floor_roof_results=floor_roof_results,
        roof_levels=roof_levels,
        lower_floor_roof_mode="shed",
    )


def visualize_3d_pyramid_matplotlib(
    output_path: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    all_floor_paths: Optional[List[str]] = None,
    floor_roof_results: Optional[List[Dict[str, Any]]] = None,
    roof_levels: Optional[List[Tuple[float, Dict[str, Any], float, float, int]]] = None,
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

    from roof_calc.visualize import _ends_adjacent_to_upper_floor, _free_roof_ends
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
    overhang_px = float(config.get("overhang_px", 0.0))
    overhang_keep_height = bool(config.get("overhang_keep_height", True))
    overhang_drop_down = bool(config.get("overhang_drop_down", True))
    overhang_shift_whole_roof_down = bool(config.get("overhang_shift_whole_roof_down", False))

    from roof_calc.overhang import (
        apply_overhang_to_sections,
        compute_overhang_sides_from_footprint,
        compute_overhang_sides_from_free_ends,
        get_downspout_faces_for_floors,
        get_gutter_end_closures_3d,
        get_gutter_endpoints_3d,
        get_gutter_faces_3d,
        get_pyramid_corner_hemispheres_3d,
    )

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

    # Collect bounds from FINAL geometry (roof faces included), then pad to avoid clipping.
    xs_all: List[float] = []
    ys_all: List[float] = []
    gutter_endpoints_pyr: List[Tuple[float, float, float]] = []
    gutter_segment_sections_pyr: List[List[Dict[str, Any]]] = []
    pyramid_gutter_closure_calls: List[tuple] = []
    for floor_idx, (_p, poly, rr, (ox, oy)) in enumerate(floors_payload):
        z0 = floor_idx * wall_height
        z1 = (floor_idx + 1) * wall_height

        coords = list(poly.exterior.coords) if hasattr(poly, "exterior") else []
        xs_all.extend([float(c[0]) for c in coords])
        ys_all.extend([float(c[1]) for c in coords])

        union_above = None
        if floor_idx + 1 < len(floors_payload):
            union_above = unary_union([fp[1] for fp in floors_payload[floor_idx + 1 :]])

        # upper sections in this floor coords (for end suppression)
        upper_secs_all: List[Dict[str, Any]] = []
        for _pp, _poly_u, rr_u, (ox_u, oy_u) in floors_payload[floor_idx + 1 :]:
            upper_secs_all.extend(_translate_sections(rr_u.get("sections") or [], float(ox_u - ox), float(oy_u - oy)))

        # Top-floor roof: draw directly. Lower floors: via roof_levels (same as standard)
        draw_roof = floor_idx == (num_floors - 1)

        secs_base = _translate_sections(rr.get("sections") or [], float(ox), float(oy))
        conns = rr.get("connections") or []
        free_ends = _free_roof_ends(secs_base, conns)

        secs = secs_base
        if draw_roof and overhang_px > 0 and secs_base:
            free = compute_overhang_sides_from_free_ends(secs_base, free_ends)
            secs = apply_overhang_to_sections(secs_base, overhang_px=overhang_px, free_sides=free)

        roof_faces: List[List[List[float]]] = []
        roof_faces_base: List[List[List[float]]] = []
        sections_to_draw: List[Tuple[int, Any]] = [(i, sec) for i, sec in enumerate(secs)] if draw_roof else []

        for s_idx, sec in sections_to_draw:
            br = sec.get("bounding_rect", [])
            if len(br) < 3:
                continue
            sp = ShapelyPolygon(br)
            if draw_roof and _covered_by_upper(sp, union_above, area_thresh=500.0):
                continue
            fe = free_ends[s_idx] if s_idx < len(free_ends) else {}
            base_sec_ridge = secs_base[s_idx] if overhang_px > 0 and s_idx < len(secs_base) else None
            faces_sec = _roof_section_faces_pyramid(
                sec, z1, roof_angle_rad, fe, upper_secs_all, base_sec_for_ridge=base_sec_ridge
            )
            roof_faces.extend(faces_sec)

        # baseline (no overhang) faces for height reference
        if draw_roof and overhang_px > 0 and overhang_keep_height and secs_base:
            for s_idx, sec0 in enumerate(secs_base):
                br0 = sec0.get("bounding_rect", [])
                if len(br0) < 3:
                    continue
                sp0 = ShapelyPolygon(br0)
                if _covered_by_upper(sp0, union_above, area_thresh=500.0):
                    continue
                fe0 = free_ends[s_idx] if s_idx < len(free_ends) else {}
                roof_faces_base.extend(_roof_section_faces_pyramid(sec0, z1, roof_angle_rad, fe0, upper_secs_all))

        # Keep same peak height as baseline
        if overhang_px > 0 and overhang_keep_height and roof_faces_base and roof_faces:
            try:
                base_z = float(z1)
                maxz_base = max(float(p[2]) for face in roof_faces_base for p in face)
                maxz_new = max(float(p[2]) for face in roof_faces for p in face)
                d_base = maxz_base - base_z
                d_new = maxz_new - base_z
                if d_base > 1e-6 and d_new > 1e-6:
                    s = d_base / d_new
                    roof_faces = [[[float(x), float(y), base_z + (float(z) - base_z) * s] for x, y, z in face] for face in roof_faces]
            except Exception:
                pass

        # Drop ONLY the overhang part below wall-top (z1).
        # Pentru piramidă: folosim distanța 2D de la vârf (centru), nu creasta – toate 4 laturile sunt streașini.
        if overhang_px > 0 and overhang_drop_down and draw_roof and secs_base and roof_faces:
            try:
                import math

                tanv = float(math.tan(roof_angle_rad))
                peak_z = max(float(p[2]) for face in roof_faces for p in face)
                tol = 1e-6
                # Footprint combinat din toate secțiunile
                all_x, all_y = [], []
                ridge_x_sum, ridge_y_sum, ridge_n = 0.0, 0.0, 0
                for sec0 in secs_base:
                    br0 = sec0.get("bounding_rect") or []
                    if len(br0) < 3:
                        continue
                    for p in br0:
                        all_x.append(float(p[0]))
                        all_y.append(float(p[1]))
                    ridge = sec0.get("ridge_line") or []
                    if len(ridge) >= 2:
                        ridge_x_sum += (float(ridge[0][0]) + float(ridge[1][0])) / 2.0
                        ridge_y_sum += (float(ridge[0][1]) + float(ridge[1][1])) / 2.0
                        ridge_n += 1
                if not all_x or not all_y:
                    raise ValueError("no bounds")
                minx0, maxx0 = min(all_x), max(all_x)
                miny0, maxy0 = min(all_y), max(all_y)
                ridge_x = ridge_x_sum / ridge_n if ridge_n else (minx0 + maxx0) / 2.0
                ridge_y = ridge_y_sum / ridge_n if ridge_n else (miny0 + maxy0) / 2.0

                for face in roof_faces:
                    for p in face:
                        x, y = float(p[0]), float(p[1])
                        in_overhang = x < minx0 - tol or x > maxx0 + tol or y < miny0 - tol or y > maxy0 + tol
                        if in_overhang:
                            d_peak = math.sqrt((x - ridge_x) ** 2 + (y - ridge_y) ** 2)
                            p[2] = peak_z - tanv * d_peak
            except Exception:
                pass

        # Shift: NU la etaj (top floor) - îl coboară prea mult. Parter (roof_levels) îl aplică.
        if False and overhang_px > 0 and overhang_shift_whole_roof_down and roof_faces:
            try:
                import math

                dz = float(math.tan(math.radians(roof_angle_deg)) * float(overhang_px))
                roof_faces = [[[float(x), float(y), float(z) - dz] for x, y, z in face] for face in roof_faces]
            except Exception:
                pass

        # Pereți prelungiți până la acoperiș (piramidă) – eșantionăm pe lungime ca să tăiem unde ies prin acoperiș
        faces_for_wall_z = [{"vertices_3d": f} for f in roof_faces] if (draw_roof and roof_faces) else []
        wall_faces = []
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])
            if faces_for_wall_z:
                n_pts = 13
                pts_top = []
                for k in range(n_pts):
                    t = k / (n_pts - 1) if n_pts > 1 else 1.0
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    zt = _z_roof_at(faces_for_wall_z, x, y, z1, tol=40.0)
                    pts_top.append([x, y, min(z1, zt)])
                face_pts = [[x1, y1, z0], [x2, y2, z0]] + [list(p) for p in reversed(pts_top)]
                for j in range(1, len(face_pts) - 1):
                    wall_faces.append([face_pts[0], face_pts[j], face_pts[j + 1]])
            else:
                wall_faces.append(
                    [[x1, y1, z0], [x2, y2, z0], [x2, y2, z1], [x1, y1, z1]]
                )

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
                for vx, vy, _vz in face:
                    xs_all.append(float(vx))
                    ys_all.append(float(vy))
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
            # Streașină colectoare pe eaves – pyramid main roof
            if draw_roof and overhang_px > 0 and secs and roof_faces:
                parter_secs_for_main: List[Dict[str, Any]] = []
                if roof_levels:
                    for _zb, roof_data, _dx, _dy, _fl in roof_levels:
                        parter_secs_for_main.extend(
                            _translate_sections(roof_data.get("sections") or [], float(_dx), float(_dy))
                        )
                import math as _m_pyr
                roof_shift_dz = float(_m_pyr.tan(_m_pyr.radians(roof_angle_deg)) * float(overhang_px)) if overhang_shift_whole_roof_down else 0.0
                all_faces = [{"vertices_3d": f} for f in roof_faces]
                for gf in get_gutter_faces_3d(
                    secs, float(z1), overhang_px, roof_angle_deg,
                    roof_shift_dz=roof_shift_dz, roof_faces=all_faces,
                    pyramid_all_sides=True,
                    pyramid_extend=True,
                    eaves_z_lift=overhang_px * 0.60,
                    interior_reference_sections=parter_secs_for_main if parter_secs_for_main else None,
                ):
                    v = gf.get("vertices_3d")
                    if v:
                        for vx, vy, _ in v:
                            xs_all.append(float(vx))
                            ys_all.append(float(vy))
                        gutter_coll = Poly3DCollection(
                            [v],
                            facecolors="#6B7280",
                            edgecolors="none",
                            alpha=1.0,
                            zsort="average",
                        )
                        gutter_coll.set_antialiased(False)
                        ax.add_collection3d(gutter_coll)
                ep = get_gutter_endpoints_3d(
                    secs, float(z1), overhang_px, roof_angle_deg,
                    roof_shift_dz=roof_shift_dz, roof_faces=all_faces,
                    pyramid_all_sides=True,
                    eaves_z_lift=overhang_px * 0.60,
                )
                gutter_endpoints_pyr.extend(ep)
                for _ in range(len(ep) // 2):
                    gutter_segment_sections_pyr.append(secs)
                pyramid_gutter_closure_calls.append((secs, float(z1), roof_shift_dz, all_faces))

    interior_refs_pyr: List[Dict[str, Any]] = []
    # Add lower-floor pyramid roofs from roof_levels (same as standard - parter visible)
    if roof_levels:
        try:
            for z_base, roof_data, dx, dy, _fl in roof_levels:
                secs0 = roof_data.get("sections") or []
                conns0 = roof_data.get("connections") or []
                secs_t = _translate_sections(secs0, float(dx), float(dy))
                free_ends0 = _free_roof_ends(secs_t, conns0)
                if overhang_px > 0 and secs_t:
                    # For lower floors, only extend on the OUTER boundary of that floor footprint
                    footprint = None
                    try:
                        if _fl is not None and 0 <= int(_fl) < len(floors_payload):
                            footprint = floors_payload[int(_fl)][1]
                    except Exception:
                        footprint = None
                    free = (
                        compute_overhang_sides_from_footprint(secs_t, footprint)
                        if footprint is not None
                        else compute_overhang_sides_from_free_ends(secs_t, free_ends0)
                    )
                    secs_t = apply_overhang_to_sections(secs_t, overhang_px=overhang_px, free_sides=free)
                # recompute free_ends after overhang for pyramid logic
                free_ends1 = _free_roof_ends(secs_t, conns0)
                # Secțiuni etaj superior în overlay (ca pyramid_lines) - fără diagonală pe laturile lipite
                upper_secs_parter: List[Dict[str, Any]] = []
                if _fl is not None and 0 <= int(_fl) < len(floors_payload):
                    for _pp, _poly_u, rr_u, (ox_u, oy_u) in floors_payload[int(_fl) + 1 :]:
                        upper_secs_parter.extend(
                            _translate_sections(rr_u.get("sections") or [], float(ox_u), float(oy_u))
                        )
                if upper_secs_parter:
                    interior_refs_pyr.extend(upper_secs_parter)
                faces: List[List[List[float]]] = []
                for s_idx, sec in enumerate(secs_t):
                    fe = free_ends1[s_idx] if s_idx < len(free_ends1) else {}
                    faces.extend(
                        _roof_section_faces_pyramid(
                            sec, float(z_base), roof_angle_rad, fe, upper_secs_parter if upper_secs_parter else None
                        )
                    )

                # Parter: aplicăm shift ca overhang-ul să fie sub nivelul peretelui
                if overhang_px > 0 and overhang_shift_whole_roof_down and faces:
                    try:
                        import math

                        dz = float(math.tan(math.radians(roof_angle_deg)) * float(overhang_px))
                        faces = [[[float(x), float(y), float(z) - dz] for x, y, z in face] for face in faces]
                    except Exception:
                        pass

                # Keep same peak height as baseline (computed without overhang for these roofs)
                if overhang_px > 0 and overhang_keep_height and secs0:
                    try:
                        secs_base_local = _translate_sections(secs0, float(dx), float(dy))
                        free_ends_base = _free_roof_ends(secs_base_local, conns0)
                        faces_base: List[List[List[float]]] = []
                        for s_idx, sec0 in enumerate(secs_base_local):
                            fe0 = free_ends_base[s_idx] if s_idx < len(free_ends_base) else {}
                            faces_base.extend(
                                _roof_section_faces_pyramid(
                                    sec0, float(z_base), roof_angle_rad, fe0,
                                    upper_secs_parter if upper_secs_parter else None
                                )
                            )
                        base_z = float(z_base)
                        maxz_base = max(float(p[2]) for face in faces_base for p in face) if faces_base else None
                        maxz_new = max(float(p[2]) for face in faces for p in face) if faces else None
                        if maxz_base is not None and maxz_new is not None:
                            d_base = float(maxz_base) - base_z
                            d_new = float(maxz_new) - base_z
                            if d_base > 1e-6 and d_new > 1e-6:
                                s = d_base / d_new
                                faces = [
                                    [[float(x), float(y), base_z + (float(z) - base_z) * s] for x, y, z in face]
                                    for face in faces
                                ]
                    except Exception:
                        pass

                # Drop ONLY the overhang part below wall-top (z_base)
                if overhang_px > 0 and overhang_drop_down and secs0 and faces:
                    try:
                        import math

                        tanv = float(math.tan(roof_angle_rad))
                        # base sections (no overhang) in overlay coords
                        secs_base_local = _translate_sections(secs0, float(dx), float(dy))
                        for s_idx, sec0 in enumerate(secs_base_local):
                            br0 = sec0.get("bounding_rect") or []
                            if len(br0) < 3:
                                continue
                            xs0 = [float(p[0]) for p in br0]
                            ys0 = [float(p[1]) for p in br0]
                            minx0, maxx0 = min(xs0), max(xs0)
                            miny0, maxy0 = min(ys0), max(ys0)
                            orient = str(sec0.get("ridge_orientation", "horizontal"))
                            ridge = sec0.get("ridge_line") or []
                            if len(ridge) >= 2:
                                rx0, ry0 = float(ridge[0][0]), float(ridge[0][1])
                                rx1, ry1 = float(ridge[1][0]), float(ridge[1][1])
                                ridge_x = (rx0 + rx1) / 2.0
                                ridge_y = (ry0 + ry1) / 2.0
                            else:
                                ridge_x = (minx0 + maxx0) / 2.0
                                ridge_y = (miny0 + maxy0) / 2.0

                            if orient == "vertical":
                                d0 = max(abs(minx0 - ridge_x), abs(maxx0 - ridge_x))
                                def d_perp(x: float, y: float) -> float:  # noqa: ARG001
                                    return abs(x - ridge_x)
                            else:
                                d0 = max(abs(miny0 - ridge_y), abs(maxy0 - ridge_y))
                                def d_perp(x: float, y: float) -> float:  # noqa: ARG001
                                    return abs(y - ridge_y)

                            peak_z = max(float(p[2]) for face in faces for p in face)
                            for face in faces:
                                for p in face:
                                    x, y = float(p[0]), float(p[1])
                                    d1 = float(d_perp(x, y))
                                    if d1 > d0 + 1e-6:
                                        p[2] = peak_z - tanv * d1
                    except Exception:
                        pass

                for ax in axs:
                    for idx, face in enumerate(faces):
                        col = colors[idx % len(colors)]
                        for vx, vy, _vz in face:
                            xs_all.append(float(vx))
                            ys_all.append(float(vy))
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
                    # Streașină colectoare – pyramid etaje inferioare (exclude secțiuni fără acoperiș expus)
                    if overhang_px > 0 and secs_t and faces:
                        pyr_include_mask = [
                            any(
                                (free_ends1[idx] if idx < len(free_ends1) else {}).get(s, True)
                                and s not in _ends_adjacent_to_upper_floor(sec, upper_secs_parter or [])
                                for s in ("top", "bottom", "left", "right")
                            )
                            for idx, sec in enumerate(secs_t)
                        ]
                        if any(pyr_include_mask):
                            import math as _m_pyr_lo
                            roof_shift_dz = float(_m_pyr_lo.tan(_m_pyr_lo.radians(roof_angle_deg)) * float(overhang_px)) if overhang_shift_whole_roof_down else 0.0
                            all_faces_lo = [{"vertices_3d": f} for f in faces]
                            for gf in get_gutter_faces_3d(
                                secs_t, float(z_base), overhang_px, roof_angle_deg,
                                roof_shift_dz=roof_shift_dz, roof_faces=all_faces_lo,
                                pyramid_all_sides=True,
                                pyramid_extend=True,
                                eaves_z_lift=overhang_px * 0.60,
                                sections_include_mask=pyr_include_mask,
                                interior_reference_sections=upper_secs_parter if upper_secs_parter else None,
                            ):
                                v = gf.get("vertices_3d")
                                if v:
                                    for vx, vy, _ in v:
                                        xs_all.append(float(vx))
                                        ys_all.append(float(vy))
                                    gutter_coll = Poly3DCollection(
                                        [v],
                                        facecolors="#6B7280",
                                        edgecolors="none",
                                        alpha=1.0,
                                        zsort="average",
                                    )
                                    gutter_coll.set_antialiased(False)
                                    ax.add_collection3d(gutter_coll)
                            ep = get_gutter_endpoints_3d(
                                secs_t, float(z_base), overhang_px, roof_angle_deg,
                                roof_shift_dz=roof_shift_dz, roof_faces=all_faces_lo,
                                pyramid_all_sides=True,
                                eaves_z_lift=overhang_px * 0.60,
                                sections_include_mask=pyr_include_mask,
                            )
                            gutter_endpoints_pyr.extend(ep)
                            for _ in range(len(ep) // 2):
                                gutter_segment_sections_pyr.append(secs_t)
                            pyramid_gutter_closure_calls.append(
                                (secs_t, float(z_base), roof_shift_dz, all_faces_lo)
                            )
        except Exception:
            pass

    # Burlani (cilindri verticali + legături la streașină) la fiecare colț exterior
    gutter_radius_pyr = max(2.0, overhang_px * 0.24) * 0.70 if overhang_px > 0 else None
    downspout_result_pyr = get_downspout_faces_pyramid(
        floors_payload,
        wall_height,
        cylinder_radius=gutter_radius_pyr,
        gutter_endpoints=gutter_endpoints_pyr,
        gutter_segment_sections=gutter_segment_sections_pyr if gutter_segment_sections_pyr else None,
        return_used_gutter_endpoints=True,
    )
    downspout_faces_pyr = downspout_result_pyr[0] if isinstance(downspout_result_pyr, tuple) else downspout_result_pyr
    used_gutter_endpoints_pyr = downspout_result_pyr[1] if isinstance(downspout_result_pyr, tuple) else []
    for ax in axs:
        for cf in get_pyramid_corner_hemispheres_3d(
            gutter_endpoints_pyr,
            gutter_segment_sections_pyr if gutter_segment_sections_pyr else [],
            overhang_px,
            downspout_endpoints=used_gutter_endpoints_pyr,
            sections_for_centroid=pyramid_gutter_closure_calls[0][0] if pyramid_gutter_closure_calls else None,
            interior_reference_sections=interior_refs_pyr if interior_refs_pyr else None,
        ):
            v = cf.get("vertices_3d")
            if v:
                for vx, vy, _ in v:
                    xs_all.append(float(vx))
                    ys_all.append(float(vy))
                closure_coll = Poly3DCollection(
                    [v],
                    facecolors="#6B7280",
                    edgecolors="none",
                    alpha=1.0,
                    zsort="average",
                )
                closure_coll.set_antialiased(False)
                ax.add_collection3d(closure_coll)
        for df in downspout_faces_pyr:
            v = df.get("vertices_3d")
            if v:
                for vx, vy, _ in v:
                    xs_all.append(float(vx))
                    ys_all.append(float(vy))
                dp_coll = Poly3DCollection(
                    [v],
                    facecolors=df.get("color", "#6B7280"),
                    edgecolors="none",
                    alpha=1.0,
                    zsort="average",
                )
                dp_coll.set_antialiased(False)
                ax.add_collection3d(dp_coll)

    if xs_all and ys_all:
        minx, maxx = float(min(xs_all)), float(max(xs_all))
        miny, maxy = float(min(ys_all)), float(max(ys_all))
        pad_ratio = float(config.get("bounds_pad_ratio", 0.08))
        pad_min = float(config.get("bounds_pad_px", 60.0))
        span = max(maxx - minx, maxy - miny, 1.0)
        pad = max(pad_min, pad_ratio * span)
        minx, maxx = (minx - pad, maxx + pad)
        miny, maxy = (miny - pad, maxy + pad)
        for ax in axs:
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            zmax = num_floors * wall_height + 500 + float(config.get("bounds_pad_z", 0.0))
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


def visualize_3d_a_frame_matplotlib(
    output_path: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    all_floor_paths: Optional[List[str]] = None,
    floor_roof_results: Optional[List[Dict[str, Any]]] = None,
    roof_levels: Optional[List[Tuple[float, Dict[str, Any], float, float, int]]] = None,
    use_a_lines_structure: bool = True,
) -> bool:
    """
    Randare 3D A-frame (wireframe): pereți + segmente din get_roof_segments_3d.
    Fallback pentru house_a_frame.png când Plotly nu e disponibil.
    """
    _ensure_matplotlib_cache_dirs()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union
    from shapely import affinity as shapely_affinity

    from roof_calc.overhang import (
        extend_sections_to_connect,
        extend_secondary_sections_to_main_ridge,
        ridge_intersection_corner_lines,
    )
    from roof_calc.roof_segments_3d import get_faces_3d_from_segments

    if not all_floor_paths or not floor_roof_results or len(all_floor_paths) != len(floor_roof_results):
        return False

    config = config or {}
    roof_angle_deg = float(config.get("roof_angle", 30.0))
    wall_height = float(config.get("wall_height", 300.0))
    views = config.get("views", [{"elev": 30, "azim": 45, "title": "Sud-Est"}, {"elev": 20, "azim": 225, "title": "Nord-Vest"}])

    offsets = _compute_offsets(all_floor_paths, floor_roof_results)
    floors_payload: List[Tuple[str, Any, Dict[str, Any], Tuple[int, int], int]] = []
    for idx, (p, rr) in enumerate(zip(all_floor_paths, floor_roof_results)):
        poly0 = _polygon_from_path(p)
        if poly0 is None:
            continue
        ox, oy = offsets.get(p, (0, 0))
        poly_t = shapely_affinity.translate(poly0, xoff=ox, yoff=oy)
        floors_payload.append((p, poly_t, rr, (ox, oy), idx))
    if not floors_payload:
        return False

    floors_payload.sort(key=lambda t: (-(float(getattr(t[1], "area", 0.0) or 0.0)), t[4]))
    num_floors = len(floors_payload)
    z_max = num_floors * wall_height + 500

    # Încarcă fețe din a_faces – pentru fiecare etaj, exact ce e în a_faces.png (a_faces_faces.json)
    a_faces_by_etaj: Dict[int, List[Dict[str, Any]]] = {}
    try:
        import json
        out_dir = Path(output_path).parent
        for etaj_dir in sorted(out_dir.glob("etaj_*")):
            if not etaj_dir.is_dir():
                continue
            try:
                etaj_idx = int(etaj_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            faces_path = etaj_dir / "a_faces_faces.json"
            if not faces_path.exists():
                continue
            data = json.loads(faces_path.read_text(encoding="utf-8"))
            faces = data.get("faces", [])
            if faces:
                a_faces_by_etaj[etaj_idx] = faces
    except Exception:
        pass

    wall_faces: List[List[List[float]]] = []
    roof_faces: List[List[List[float]]] = []
    roof_face_edges: List[Tuple[List[float], List[float]]] = []
    xs_all: List[float] = []
    ys_all: List[float] = []
    RED = "#FF0000"
    GRAY = "#95A5A6"

    for floor_idx, payload_item in enumerate(floors_payload):
        _p, floor_poly, rr, (ox, oy) = payload_item[0], payload_item[1], payload_item[2], payload_item[3]
        _orig_idx = payload_item[4] if len(payload_item) > 4 else floor_idx
        z0 = floor_idx * wall_height
        z1 = (floor_idx + 1) * wall_height
        coords = list(floor_poly.exterior.coords) if hasattr(floor_poly, "exterior") else []
        for c in coords:
            xs_all.append(float(c[0]))
            ys_all.append(float(c[1]))
        for i in range(len(coords) - 1):
            x1, y1 = float(coords[i][0]), float(coords[i][1])
            x2, y2 = float(coords[i + 1][0]), float(coords[i + 1][1])
            wall_faces.append([[x1, y1, z0], [x2, y2, z0], [x2, y2, z1], [x1, y1, z1]])

        draw_roof = True
        union_above = None
        if floor_idx + 1 < len(floors_payload):
            union_above = unary_union([fp[1] for fp in floors_payload[floor_idx + 1:]])
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
        kept_extended = extend_sections_to_connect(kept, conns_kept) if (draw_roof and kept and conns_kept) else kept
        if draw_roof and kept_extended:
            kept_extended = extend_secondary_sections_to_main_ridge(kept_extended)

        if draw_roof and kept_extended:
            # Fețe din a_faces – exact ce e în a_faces.png pentru acest etaj
            roof_data: List[Dict[str, Any]] = []
            raw_faces_from_etaj = a_faces_by_etaj.get(_orig_idx) if _orig_idx is not None else None
            if raw_faces_from_etaj:
                for f in raw_faces_from_etaj:
                    vs = f.get("vertices_3d") or []
                    if vs:
                        roof_data.append({
                            "vertices_3d": [[float(v[0]) + ox, float(v[1]) + oy, float(v[2])] for v in vs],
                        })
            if not roof_data:
                cl = ridge_intersection_corner_lines(kept_extended, floor_polygon=floor_poly)
                roof_data = get_faces_3d_from_segments(
                    kept_extended, floor_poly, wall_height=z1,
                    roof_angle_deg=roof_angle_deg, corner_lines=cl,
                    use_section_rect_eaves=False,
                )
            seen_edge: set = set()
            tol = 1e-4

            def _edge_key(a: List[float], b: List[float]) -> tuple:
                ka = (round(a[0] / tol) * tol, round(a[1] / tol) * tol, round(a[2] / tol) * tol)
                kb = (round(b[0] / tol) * tol, round(b[1] / tol) * tol, round(b[2] / tol) * tol)
                return (ka, kb) if ka <= kb else (kb, ka)

            for f in roof_data:
                vs = f.get("vertices_3d") or []
                if vs:
                    roof_faces.append(vs)
                    for i in range(len(vs)):
                        p1, p2 = list(vs[i]), list(vs[(i + 1) % len(vs)])
                        key = _edge_key(p1, p2)
                        if key not in seen_edge:
                            seen_edge.add(key)
                            roof_face_edges.append((p1, p2))

    if not xs_all and not ys_all:
        xs_all = [0.0]
        ys_all = [0.0]
    minx_b = float(min(xs_all))
    maxx_b = float(max(xs_all))
    miny_b = float(min(ys_all))
    maxy_b = float(max(ys_all))
    pad = max(60.0, 0.08 * max(maxx_b - minx_b, maxy_b - miny_b, 1.0))
    minx_b, maxx_b = minx_b - pad, maxx_b + pad
    miny_b, maxy_b = miny_b - pad, maxy_b + pad

    fig = plt.figure(figsize=(16, 7))
    axs = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(min(2, len(views)))]

    for ax, v in zip(axs, views[: len(axs)]):
        ax.set_title(v.get("title", "View"))
        ax.view_init(elev=float(v.get("elev", 30)), azim=float(v.get("azim", 45)))
        ax.set_xlim(minx_b, maxx_b)
        ax.set_ylim(miny_b, maxy_b)
        ax.set_zlim(0, z_max)
        ax.set_facecolor("white")

        wall_tris: List[List[List[float]]] = []
        for face in wall_faces:
            wall_tris.extend(_triangulate_face(face))
        roof_tris: List[List[List[float]]] = []
        for face in roof_faces:
            t = _triangulate_face(face)
            if not t:
                t = _triangulate_fan(face)
            roof_tris.extend(t)

        shadow_tris: List[List[List[float]]] = []
        for tri in wall_tris + roof_tris:
            shadow_tris.append([[vx, vy, 0.0] for vx, vy, vz in tri])
        if shadow_tris:
            sc = Poly3DCollection(
                shadow_tris,
                facecolors=(0.0, 0.0, 0.0, 0.25),
                edgecolors="none",
                alpha=0.25,
                zsort="average",
            )
            sc.set_antialiased(False)
            ax.add_collection3d(sc)

        if wall_tris:
            wc = Poly3DCollection(
                wall_tris,
                facecolors=GRAY,
                edgecolors="none",
                alpha=1.0,
                zsort="average",
            )
            wc.set_antialiased(False)
            ax.add_collection3d(wc)
        if roof_tris:
            rc = Poly3DCollection(
                roof_tris,
                facecolors=RED,
                edgecolors="none",
                alpha=1.0,
                zsort="average",
            )
            rc.set_antialiased(False)
            ax.add_collection3d(rc)
        for p1, p2 in roof_face_edges:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="black", linewidth=2)
        for floor_idx, (_path, floor_poly, _rr, _offset) in enumerate(floors_payload):
            z0 = floor_idx * wall_height
            z1 = (floor_idx + 1) * wall_height
            coords = list(floor_poly.exterior.coords) if hasattr(floor_poly, "exterior") else []
            for i in range(len(coords) - 1):
                x1, y1 = float(coords[i][0]), float(coords[i][1])
                x2, y2 = float(coords[i + 1][0]), float(coords[i + 1][1])
                for (a1, b1, za), (a2, b2, zb) in [
                    ((x1, y1, z0), (x2, y2, z0)),
                    ((x1, y1, z1), (x2, y2, z1)),
                    ((x1, y1, z0), (x1, y1, z1)),
                    ((x2, y2, z0), (x2, y2, z1)),
                ]:
                    ax.plot([a1, a2], [b1, b2], [za, zb], color="black", linewidth=2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return True
