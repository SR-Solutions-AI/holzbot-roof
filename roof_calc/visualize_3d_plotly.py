"""
Vizualizare 3D (piramidă) cu Plotly + Kaleido: export PNG direct (fără browser).
Folosit ca fallback când PyVista/VTK nu poate fi importat (ex. Python 3.14 + VTK wheels).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _hex_to_rgb01(hex_color: str) -> Tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def _triangulate_face(face_pts: List[List[float]]) -> List[List[List[float]]]:
    """Triangulează o față (tri sau quad) în 1-2 triunghiuri."""
    if len(face_pts) == 3:
        return [face_pts]
    if len(face_pts) == 4:
        a, b, c, d = face_pts
        return [[a, b, c], [a, c, d]]
    return []


def _camera_eye_from_elev_azim(elev_deg: float, azim_deg: float, scale: float = 1.8) -> Dict[str, float]:
    az = np.radians(azim_deg)
    el = np.radians(elev_deg)
    return {
        # Plotly camera.eye e un vector relativ (tipic ~1–2), nu în unități de coordonate ale scenei
        "x": float(scale * np.cos(el) * np.cos(az)),
        "y": float(scale * np.cos(el) * np.sin(az)),
        "z": float(scale * np.sin(el)),
    }


def _round_v3(pt: List[float], nd: int = 3) -> Tuple[float, float, float]:
    return (round(float(pt[0]), nd), round(float(pt[1]), nd), round(float(pt[2]), nd))


def _edge_lines_from_tris(
    tris: List[List[List[float]]],
    round_ndigits: int = 3,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Construiește un set de segmente (x,y,z cu None separators) pentru contur/wireframe,
    deduplicând muchiile comune între triunghiuri.
    """
    edges: set[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = set()
    for tri in tris:
        if len(tri) != 3:
            continue
        a, b, c = tri[0], tri[1], tri[2]
        va = _round_v3(a, round_ndigits)
        vb = _round_v3(b, round_ndigits)
        vc = _round_v3(c, round_ndigits)
        for e0, e1 in ((va, vb), (vb, vc), (vc, va)):
            if e0 == e1:
                continue
            edges.add((e0, e1) if e0 < e1 else (e1, e0))

    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for p0, p1 in edges:
        xs.extend([p0[0], p1[0], None])  # type: ignore[arg-type]
        ys.extend([p0[1], p1[1], None])  # type: ignore[arg-type]
        zs.extend([p0[2], p1[2], None])  # type: ignore[arg-type]
    return xs, ys, zs


def visualize_3d_standard_plotly(
    output_path: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    all_floor_paths: Optional[List[str]] = None,
    floor_roof_results: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """
    Randare 3D standard (gable) cu Plotly + Kaleido, salvează PNG la output_path.
    Folosește pipeline-ul multi-etaj bazat pe `rectangles_floor` (similar cu Matplotlib),
    dar cu export robust (fără probleme de z-order).
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        logger.debug("Plotly not available: %s", e)
        return False

    import cv2
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union
    from shapely import affinity as shapely_affinity

    from roof_calc.flood_fill import flood_fill_interior, get_house_shape_mask
    from roof_calc.geometry import extract_polygon
    import roof_calc.roof_faces_3d as rf

    config = config or {}
    roof_angle_deg = float(config.get("roof_angle", 30.0))
    wall_height = float(config.get("wall_height", 300.0))
    colors = config.get("colors", ["#FF6B6B", "#4ECDC4", "#45B7D1"])
    show_outlines = bool(config.get("show_outlines", True))
    outline_color = str(config.get("outline_color", "rgba(0,0,0,0.65)"))
    outline_width = int(config.get("outline_width", 3))
    views = config.get(
        "views",
        [{"elev": 30, "azim": 45, "title": "Sud-Est"}, {"elev": 20, "azim": 225, "title": "Nord-Vest"}],
    )

    if not all_floor_paths or not floor_roof_results or len(all_floor_paths) != len(floor_roof_results):
        return False

    def _polygon_from_path(path: str) -> Optional[Any]:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        filled0 = flood_fill_interior(im)
        house0 = get_house_shape_mask(filled0)
        poly0 = extract_polygon(house0)
        if poly0 is None or poly0.is_empty:
            return None
        return poly0

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

    def _translate_sections(secs: List[Dict[str, Any]], dx: float, dy: float) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for sec in secs:
            rect = sec.get("bounding_rect", [])
            ridge = sec.get("ridge_line", [])
            out.append(
                {
                    **sec,
                    "bounding_rect": [(float(p[0]) + dx, float(p[1]) + dy) for p in rect] if rect else [],
                    "ridge_line": [(float(p[0]) + dx, float(p[1]) + dy) for p in ridge]
                    if ridge and len(ridge) >= 2
                    else [],
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
    z_max = num_floors * wall_height + 500

    tri_by_color: Dict[str, List[List[List[float]]]] = {}

    def add_face(face: List[List[float]], color_hex: str) -> None:
        for tri in _triangulate_face(face):
            tri_by_color.setdefault(color_hex, []).append(tri)

    gray = "#B0B0B0"
    xs_all: List[float] = []
    ys_all: List[float] = []

    for floor_idx, (_p, floor_poly, rr, (ox, oy)) in enumerate(floors_payload):
        z0 = floor_idx * wall_height
        z1 = (floor_idx + 1) * wall_height

        floor_coords = list(floor_poly.exterior.coords) if hasattr(floor_poly, "exterior") else []
        xs_all.extend([c[0] for c in floor_coords])
        ys_all.extend([c[1] for c in floor_coords])

        for i in range(len(floor_coords) - 1):
            p1, p2 = floor_coords[i], floor_coords[i + 1]
            add_face([[p1[0], p1[1], z0], [p2[0], p2[1], z0], [p2[0], p2[1], z1], [p1[0], p1[1], z1]], gray)

        union_above = None
        if floor_idx + 1 < len(floors_payload):
            union_above = unary_union([fp[1] for fp in floors_payload[floor_idx + 1 :]])

        secs = rr.get("sections") or []
        conns = rr.get("connections") or []

        kept: List[Dict[str, Any]] = []
        keep_ids: set[int] = set()
        for sec in secs:
            br = sec.get("bounding_rect", [])
            if len(br) < 3:
                continue
            try:
                sp = ShapelyPolygon(br)
                sp = shapely_affinity.translate(sp, xoff=ox, yoff=oy)
            except Exception:
                continue
            if _covered_by_upper(sp, union_above, area_thresh=500.0):
                continue
            sec_t = _translate_sections([sec], float(ox), float(oy))[0]
            kept.append(sec_t)
            try:
                keep_ids.add(int(sec_t.get("section_id")))
            except Exception:
                pass

        conns_kept = _filter_connections_by_sections(conns, keep_ids)
        roof_faces = rf.get_faces_3d_standard(kept, conns_kept, roof_angle_deg=roof_angle_deg, wall_height=z1)
        for idx, f in enumerate(roof_faces or []):
            verts = f.get("vertices_3d")
            if not verts:
                continue
            color_hex = colors[idx % len(colors)]
            add_face(verts, color_hex)

        # bounds helper from roof rects too
        for sec in rr.get("sections") or []:
            br = sec.get("bounding_rect", [])
            if br:
                xs_all.extend([float(p[0]) + ox for p in br])
                ys_all.extend([float(p[1]) + oy for p in br])

    if not xs_all or not ys_all:
        return False
    minx_b, maxx_b = float(min(xs_all)), float(max(xs_all))
    miny_b, maxy_b = float(min(ys_all)), float(max(ys_all))

    n = max(1, len(views))
    fig = make_subplots(
        rows=1,
        cols=n,
        specs=[[{"type": "scene"} for _ in range(n)]],
        subplot_titles=[v.get("title", f"View {i+1}") for i, v in enumerate(views[:n])],
    )

    for col_idx in range(n):
        for color_hex, tris in tri_by_color.items():
            xs: List[float] = []
            ys: List[float] = []
            zs: List[float] = []
            ii: List[int] = []
            jj: List[int] = []
            kk: List[int] = []
            for tri in tris:
                base = len(xs)
                for vx, vy, vz in tri:
                    xs.append(float(vx))
                    ys.append(float(vy))
                    zs.append(float(vz))
                ii.append(base + 0)
                jj.append(base + 1)
                kk.append(base + 2)
            r01, g01, b01 = _hex_to_rgb01(color_hex)
            fig.add_trace(
                go.Mesh3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    i=ii,
                    j=jj,
                    k=kk,
                    color=f"rgb({int(r01*255)},{int(g01*255)},{int(b01*255)})",
                    opacity=1.0,
                    flatshading=True,
                    lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.9, specular=0.1),
                ),
                row=1,
                col=col_idx + 1,
            )

        if show_outlines:
            tris_all: List[List[List[float]]] = []
            for _c, tris in tri_by_color.items():
                tris_all.extend(tris)
            lx, ly, lz = _edge_lines_from_tris(tris_all, round_ndigits=3)
            if len(lx) > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=lx,
                        y=ly,
                        z=lz,
                        mode="lines",
                        line=dict(color=outline_color, width=outline_width),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=1,
                    col=col_idx + 1,
                )

        v = views[col_idx] if col_idx < len(views) else {"elev": 30, "azim": 45, "title": f"View {col_idx+1}"}
        eye = _camera_eye_from_elev_azim(float(v.get("elev", 30)), float(v.get("azim", 45)), scale=1.8)
        scene_name = "scene" if col_idx == 0 else f"scene{col_idx+1}"
        fig.update_layout(
            **{
                scene_name: dict(
                    xaxis=dict(range=[minx_b, maxx_b], backgroundcolor="white", gridcolor="lightgrey"),
                    yaxis=dict(range=[miny_b, maxy_b], backgroundcolor="white", gridcolor="lightgrey"),
                    zaxis=dict(range=[0, z_max], backgroundcolor="white", gridcolor="lightgrey"),
                    aspectmode="data",
                    camera=dict(
                        eye=eye,
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        projection=dict(type="perspective"),
                    ),
                )
            }
        )

    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(output_path, width=1200 * n, height=900, scale=2)
        logger.info("Saved Plotly 3D standard to %s", output_path)
        return True
    except Exception as e:  # pragma: no cover
        logger.warning("Plotly write_image failed (standard): %s", e)
        return False


def visualize_3d_pyramid_plotly(
    wall_mask: Any,
    roof_data: Dict[str, Any],
    output_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    all_floor_paths: Optional[List[str]] = None,
    floor_roof_results: Optional[List[Dict[str, Any]]] = None,
    roof_levels: Optional[List[Tuple[float, Dict[str, Any], float, float]]] = None,
) -> bool:
    """
    Randare 3D piramidă cu Plotly + Kaleido, salvează PNG la output_path.
    Returnează True dacă a reușit.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        logger.debug("Plotly not available: %s", e)
        return False

    from roof_calc.visualize import _free_roof_ends
    # Refolosim generatorul de fețe din modulul PyVista (nu importă pyvista la import-time)
    from roof_calc.visualize_3d_pyvista import _roof_section_faces_pyramid  # type: ignore

    import cv2
    from roof_calc.flood_fill import flood_fill_interior, get_house_shape_mask
    from roof_calc.geometry import extract_polygon
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union
    from shapely import affinity as shapely_affinity

    config = config or {}
    roof_angle_deg = config.get("roof_angle", 30.0)
    wall_height = config.get("wall_height", 300)
    colors = config.get("colors", ["#FF6B6B", "#4ECDC4", "#45B7D1"])
    show_outlines = bool(config.get("show_outlines", True))
    outline_color = str(config.get("outline_color", "rgba(0,0,0,0.65)"))
    outline_width = int(config.get("outline_width", 3))
    views = config.get(
        "views",
        [{"elev": 30, "azim": 45, "title": "Sud-Est"}, {"elev": 20, "azim": 225, "title": "Nord-Vest"}],
    )
    roof_angle_rad = np.radians(roof_angle_deg)

    wall_mask_path = wall_mask if isinstance(wall_mask, str) else None

    def _polygon_from_path(path: str) -> Optional[Any]:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        filled0 = flood_fill_interior(im)
        house0 = get_house_shape_mask(filled0)
        poly0 = extract_polygon(house0)
        if poly0 is None or poly0.is_empty:
            return None
        return poly0

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

    def _translate_sections(secs: List[Dict[str, Any]], dx: float, dy: float) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for sec in secs:
            rect = sec.get("bounding_rect", [])
            ridge = sec.get("ridge_line", [])
            out.append(
                {
                    "bounding_rect": [(float(p[0]) + dx, float(p[1]) + dy) for p in rect] if rect else [],
                    "ridge_line": [(float(p[0]) + dx, float(p[1]) + dy) for p in ridge] if ridge and len(ridge) >= 2 else [],
                    "ridge_orientation": sec.get("ridge_orientation", "horizontal"),
                    "is_main": sec.get("is_main", False),
                }
            )
        return out

    # New multi-floor path: use per-floor roof_results (rectangles_floor) + overlay offsets
    floors_payload: List[Tuple[str, Any, Dict[str, Any], Tuple[int, int]]] = []
    if (
        wall_mask_path
        and all_floor_paths
        and floor_roof_results
        and len(all_floor_paths) == len(floor_roof_results)
    ):
        offsets = _compute_offsets(all_floor_paths, floor_roof_results)
        for p, rr in zip(all_floor_paths, floor_roof_results):
            poly0 = _polygon_from_path(p)
            if poly0 is None:
                continue
            ox, oy = offsets.get(p, (0, 0))
            poly_t = shapely_affinity.translate(poly0, xoff=ox, yoff=oy)
            floors_payload.append((p, poly_t, rr, (ox, oy)))

    # Fallback single-floor polygon
    if not floors_payload:
        if isinstance(wall_mask, str):
            img = cv2.imread(wall_mask, cv2.IMREAD_GRAYSCALE)
        else:
            img = np.asarray(wall_mask, dtype=np.uint8)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is None:
            return False
        filled = flood_fill_interior(img)
        house_mask = get_house_shape_mask(filled)
        polygon = extract_polygon(house_mask)
        if polygon is None or polygon.is_empty:
            return False
        floors_payload = [(wall_mask_path or "floor0", polygon, roof_data, (0, 0))]

    # Sort floors bottom->top by footprint area (largest first)
    floors_payload.sort(key=lambda t: float(getattr(t[1], "area", 0.0) or 0.0), reverse=True)
    num_floors = len(floors_payload)
    z_max = num_floors * wall_height + 500

    # Scene bounds (from all polygons + all section rects)
    xs_all: List[float] = []
    ys_all: List[float] = []
    for _p, poly, rr, (ox, oy) in floors_payload:
        coords = list(poly.exterior.coords) if hasattr(poly, "exterior") else []
        xs_all.extend([c[0] for c in coords])
        ys_all.extend([c[1] for c in coords])
        for sec in rr.get("sections") or []:
            rect = sec.get("bounding_rect", [])
            xs_all.extend([float(p[0]) + ox for p in rect] if rect else [])
            ys_all.extend([float(p[1]) + oy for p in rect] if rect else [])
    if not xs_all or not ys_all:
        return False
    minx_b, maxx_b = float(min(xs_all)), float(max(xs_all))
    miny_b, maxy_b = float(min(ys_all)), float(max(ys_all))

    # Construim fețe (triunghiuri) grupate pe culoare
    tri_by_color: Dict[str, List[List[List[float]]]] = {}

    def add_face(face: List[List[float]], color_hex: str) -> None:
        for tri in _triangulate_face(face):
            tri_by_color.setdefault(color_hex, []).append(tri)

    gray = "#B0B0B0"

    # Pereți + acoperișuri per etaj, bazat pe `rectangles_floor` (floor_roof_results)
    for floor_idx, (_p, floor_poly, rr, (ox, oy)) in enumerate(floors_payload):
        z0 = floor_idx * wall_height
        z1 = (floor_idx + 1) * wall_height
        coords = list(floor_poly.exterior.coords) if hasattr(floor_poly, "exterior") else []
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            add_face([[p1[0], p1[1], z0], [p2[0], p2[1], z0], [p2[0], p2[1], z1], [p1[0], p1[1], z1]], gray)

        # union of upper floors footprint (overlay coords)
        union_above = None
        if floor_idx + 1 < len(floors_payload):
            union_above = unary_union([fp[1] for fp in floors_payload[floor_idx + 1 :]])

        # upper floor sections translated into this floor coords (for end suppression)
        upper_secs_all: List[Dict[str, Any]] = []
        for _pp, _poly_u, rr_u, (ox_u, oy_u) in floors_payload[floor_idx + 1 :]:
            dxu = float(ox_u - ox)
            dyu = float(oy_u - oy)
            upper_secs_all.extend(_translate_sections(rr_u.get("sections") or [], dxu, dyu))

        secs = rr.get("sections") or []
        conns = rr.get("connections") or []
        free_ends = _free_roof_ends(secs, conns)

        # choose which sections to draw: skip those fully covered by upper floors
        for s_idx, sec in enumerate(secs):
            br = sec.get("bounding_rect", [])
            if len(br) < 3:
                continue
            try:
                sp = ShapelyPolygon(br)
                sp = shapely_affinity.translate(sp, xoff=ox, yoff=oy)
            except Exception:
                continue
            if union_above is not None and not getattr(union_above, "is_empty", True):
                try:
                    if float(sp.difference(union_above).area) < 500.0:
                        continue
                except Exception:
                    pass

            # draw with translated section coords
            sec_t = {
                **sec,
                "bounding_rect": [(float(p[0]) + ox, float(p[1]) + oy) for p in br],
                "ridge_line": [
                    (float(p[0]) + ox, float(p[1]) + oy) for p in (sec.get("ridge_line") or [])
                ],
            }
            color_hex = colors[s_idx % len(colors)]
            fe = free_ends[s_idx] if s_idx < len(free_ends) else {}
            for f in _roof_section_faces_pyramid(sec_t, z1, roof_angle_rad, fe, upper_secs_all):
                add_face(f, color_hex)

    # Build subplot fig
    n = max(1, len(views))
    fig = make_subplots(
        rows=1,
        cols=n,
        specs=[[{"type": "scene"} for _ in range(n)]],
        subplot_titles=[v.get("title", f"View {i+1}") for i, v in enumerate(views[:n])],
    )

    # Add traces to each scene
    for col_idx in range(n):
        for color_hex, tris in tri_by_color.items():
            xs: List[float] = []
            ys: List[float] = []
            zs: List[float] = []
            ii: List[int] = []
            jj: List[int] = []
            kk: List[int] = []
            for tri in tris:
                base = len(xs)
                for vx, vy, vz in tri:
                    xs.append(float(vx))
                    ys.append(float(vy))
                    zs.append(float(vz))
                ii.append(base + 0)
                jj.append(base + 1)
                kk.append(base + 2)
            r01, g01, b01 = _hex_to_rgb01(color_hex)
            fig.add_trace(
                go.Mesh3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    i=ii,
                    j=jj,
                    k=kk,
                    color=f"rgb({int(r01*255)},{int(g01*255)},{int(b01*255)})",
                    opacity=1.0,
                    flatshading=True,
                    lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.9, specular=0.1),
                ),
                row=1,
                col=col_idx + 1,
            )

        if show_outlines:
            tris_all: List[List[List[float]]] = []
            for _c, tris in tri_by_color.items():
                tris_all.extend(tris)
            lx, ly, lz = _edge_lines_from_tris(tris_all, round_ndigits=3)
            if len(lx) > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=lx,
                        y=ly,
                        z=lz,
                        mode="lines",
                        line=dict(color=outline_color, width=outline_width),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=1,
                    col=col_idx + 1,
                )

        v = views[col_idx] if col_idx < len(views) else {"elev": 30, "azim": 45, "title": f"View {col_idx+1}"}
        eye = _camera_eye_from_elev_azim(float(v.get("elev", 30)), float(v.get("azim", 45)), scale=1.8)
        scene_name = "scene" if col_idx == 0 else f"scene{col_idx+1}"
        fig.update_layout(
            **{
                scene_name: dict(
                    xaxis=dict(range=[minx_b, maxx_b], backgroundcolor="white", gridcolor="lightgrey"),
                    yaxis=dict(range=[miny_b, maxy_b], backgroundcolor="white", gridcolor="lightgrey"),
                    zaxis=dict(range=[0, z_max], backgroundcolor="white", gridcolor="lightgrey"),
                    aspectmode="data",
                    camera=dict(
                        eye=eye,
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        projection=dict(type="perspective"),
                    ),
                )
            }
        )

    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(output_path, width=1200 * n, height=900, scale=2)
            logger.info("Saved Plotly 3D pyramid to %s", output_path)
            return True
        except Exception as e:  # pragma: no cover
            logger.warning("Plotly write_image failed (pyramid): %s", e)
            return False
    return True

