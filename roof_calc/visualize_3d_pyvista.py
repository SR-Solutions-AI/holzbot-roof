"""
Vizualizare 3D cu PyVista (VTK): randare cu z-buffer corect, export direct în PNG.
Rezolvă problema acoperișului de la etajul inferior acoperit de peretele etajului superior.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from roof_calc.visualize import (
    _ends_adjacent_to_upper_floor,
    _free_roof_ends,
    _get_largest_rect_bbox,
    _ordered_floor_polygons,
    _pyramid_apex_3d,
    _scene_bounds_3d,
    _translate_roof_data,
)

logger = logging.getLogger(__name__)


def _hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Convertește '#RRGGBB' în (r, g, b) în [0, 1]."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _poly_to_pyvista_cells(pts: List[List[float]]) -> List[int]:
    """Convertește un poligon (3 sau 4 puncte) în format PyVista cells: [n, i0, i1, ...]."""
    n = len(pts)
    return [n] + list(range(n))


def _build_pyramid_end_tris(
    orientation: str,
    end_side: str,
    minx: float,
    maxx: float,
    miny: float,
    maxy: float,
    ridge_mid_x: float,
    ridge_mid_y: float,
    ridge_height: float,
    base_z: float,
) -> List[List[List[float]]]:
    """Returnează [tri1, tri2] pentru capătul piramidei (eave la base_z)."""
    apex_3d = _pyramid_apex_3d(
        orientation, end_side, minx, maxx, miny, maxy,
        ridge_mid_x, ridge_mid_y, ridge_height,
    )
    if not apex_3d:
        return []
    apex_x, apex_y, apex_z = apex_3d
    apex = [apex_x, apex_y, apex_z]
    if orientation == "horizontal":
        if end_side == "left":
            eave_corner1 = [minx, miny, base_z]
            eave_corner2 = [minx, maxy, base_z]
            eave_mid = [minx, ridge_mid_y, base_z]
        else:
            eave_corner1 = [maxx, miny, base_z]
            eave_corner2 = [maxx, maxy, base_z]
            eave_mid = [maxx, ridge_mid_y, base_z]
    else:
        if end_side == "top":
            eave_corner1 = [minx, miny, base_z]
            eave_corner2 = [maxx, miny, base_z]
            eave_mid = [ridge_mid_x, miny, base_z]
        else:
            eave_corner1 = [minx, maxy, base_z]
            eave_corner2 = [maxx, maxy, base_z]
            eave_mid = [ridge_mid_x, maxy, base_z]
    return [[apex, eave_corner1, eave_mid], [apex, eave_mid, eave_corner2]]


def _roof_section_faces_pyramid(
    sec: Dict[str, Any],
    base_z: float,
    roof_angle_rad: float,
    free_ends: Dict[str, bool],
    upper_floor_sections: Optional[List[Dict[str, Any]]],
) -> List[List[List[float]]]:
    """Returnează listă de fețe (fiecare față = listă de 3 sau 4 puncte [x,y,z])."""
    faces: List[List[List[float]]] = []
    bounding_rect = sec.get("bounding_rect", [])
    if len(bounding_rect) < 3:
        return faces
    minx = min(p[0] for p in bounding_rect)
    maxx = max(p[0] for p in bounding_rect)
    miny = min(p[1] for p in bounding_rect)
    maxy = max(p[1] for p in bounding_rect)
    ridge_line = sec.get("ridge_line", [])
    if len(ridge_line) >= 2:
        ridge_mid_x = (ridge_line[0][0] + ridge_line[1][0]) / 2
        ridge_mid_y = (ridge_line[0][1] + ridge_line[1][1]) / 2
    else:
        ridge_mid_x = (minx + maxx) / 2
        ridge_mid_y = (miny + maxy) / 2
    orientation = sec.get("ridge_orientation", "horizontal")
    span_width = (maxy - miny) / 2 if orientation == "horizontal" else (maxx - minx) / 2
    ridge_height = base_z + span_width * np.tan(roof_angle_rad)
    ends_adjacent_upper = _ends_adjacent_to_upper_floor(sec, upper_floor_sections or [])

    def ridge_pt(side: str) -> List[float]:
        if orientation == "horizontal":
            if side == "left":
                if free_ends.get("left", True) and "left" not in ends_adjacent_upper:
                    a = _pyramid_apex_3d(
                        orientation, "left", minx, maxx, miny, maxy,
                        ridge_mid_x, ridge_mid_y, ridge_height,
                    )
                    if a:
                        return [a[0], a[1], a[2]]
                return [minx, ridge_mid_y, ridge_height]
            else:
                if free_ends.get("right", True) and "right" not in ends_adjacent_upper:
                    a = _pyramid_apex_3d(
                        orientation, "right", minx, maxx, miny, maxy,
                        ridge_mid_x, ridge_mid_y, ridge_height,
                    )
                    if a:
                        return [a[0], a[1], a[2]]
                return [maxx, ridge_mid_y, ridge_height]
        else:
            if side == "top":
                if free_ends.get("top", True) and "top" not in ends_adjacent_upper:
                    a = _pyramid_apex_3d(
                        orientation, "top", minx, maxx, miny, maxy,
                        ridge_mid_x, ridge_mid_y, ridge_height,
                    )
                    if a:
                        return [a[0], a[1], a[2]]
                return [ridge_mid_x, miny, ridge_height]
            else:
                if free_ends.get("bottom", True) and "bottom" not in ends_adjacent_upper:
                    a = _pyramid_apex_3d(
                        orientation, "bottom", minx, maxx, miny, maxy,
                        ridge_mid_x, ridge_mid_y, ridge_height,
                    )
                    if a:
                        return [a[0], a[1], a[2]]
                return [ridge_mid_x, maxy, ridge_height]

    if orientation == "horizontal":
        left_r = ridge_pt("left")
        right_r = ridge_pt("right")
        faces.append([[minx, maxy, base_z], [maxx, maxy, base_z], right_r, left_r])
        faces.append([left_r, right_r, [maxx, miny, base_z], [minx, miny, base_z]])
        for end_side in ("left", "right"):
            if free_ends.get(end_side, True) and end_side not in ends_adjacent_upper:
                for tri in _build_pyramid_end_tris(
                    orientation, end_side, minx, maxx, miny, maxy,
                    ridge_mid_x, ridge_mid_y, ridge_height, base_z,
                ):
                    faces.append(tri)
    else:
        top_r = ridge_pt("top")
        bottom_r = ridge_pt("bottom")
        faces.append([[minx, miny, base_z], [minx, maxy, base_z], bottom_r, top_r])
        faces.append([top_r, bottom_r, [maxx, maxy, base_z], [maxx, miny, base_z]])
        for end_side in ("top", "bottom"):
            if free_ends.get(end_side, True) and end_side not in ends_adjacent_upper:
                for tri in _build_pyramid_end_tris(
                    orientation, end_side, minx, maxx, miny, maxy,
                    ridge_mid_x, ridge_mid_y, ridge_height, base_z,
                ):
                    faces.append(tri)
    return faces


def visualize_3d_pyramid_pyvista(
    wall_mask: Any,
    roof_data: Dict[str, Any],
    output_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    all_floor_paths: Optional[List[str]] = None,
    floor_roof_results: Optional[List[Dict[str, Any]]] = None,
    roof_levels: Optional[List[Tuple[float, Dict[str, Any], float, float]]] = None,
) -> bool:
    """
    Randare 3D piramidă cu PyVista (z-buffer corect). Salvează PNG la output_path.
    Returnează True dacă a reușit, False altfel (ex. PyVista indisponibil).
    """
    try:
        import pyvista as pv
    except ImportError as e:
        # PyVista poate fi instalat, dar importul poate eșua din cauza VTK (dlopen/Qt/OpenGL/etc.)
        raise RuntimeError(f"Nu pot importa PyVista/VTK: {e}") from e

    config = config or {}
    roof_angle_deg = config.get("roof_angle", 30.0)
    wall_height = config.get("wall_height", 300)
    colors = config.get("colors", ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"])
    views = config.get("views", [{"elev": 30, "azim": 45, "title": "Sud-Est"}])
    roof_angle_rad = np.radians(roof_angle_deg)

    import cv2
    from roof_calc.flood_fill import flood_fill_interior, get_house_shape_mask
    from roof_calc.geometry import extract_polygon
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union
    from shapely import affinity as shapely_affinity

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

    if not floors_payload:
        # fallback single floor
        if isinstance(wall_mask, str):
            img = cv2.imread(wall_mask, cv2.IMREAD_GRAYSCALE)
        else:
            img = np.asarray(wall_mask, dtype=np.uint8)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is None:
            raise FileNotFoundError(f"Nu pot citi imaginea wall_mask: {wall_mask!r}")
        filled = flood_fill_interior(img)
        house_mask = get_house_shape_mask(filled)
        polygon = extract_polygon(house_mask)
        if polygon is None or polygon.is_empty:
            raise RuntimeError("Nu pot extrage poligonul casei din maskă (extract_polygon a eșuat).")
        floors_payload = [(wall_mask_path or "floor0", polygon, roof_data, (0, 0))]

    floors_payload.sort(key=lambda t: float(getattr(t[1], "area", 0.0) or 0.0), reverse=True)
    num_floors = len(floors_payload)
    z_max = num_floors * wall_height + 500
    roof_base_z = num_floors * wall_height

    meshes_to_add: List[Tuple[List[List[float]], str]] = []
    gray = "#B0B0B0"
    # walls + per-floor roofs from rectangles_floor
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
            quad = [[p1[0], p1[1], z0], [p2[0], p2[1], z0], [p2[0], p2[1], z1], [p1[0], p1[1], z1]]
            meshes_to_add.append((quad, gray))

        union_above = None
        if floor_idx + 1 < len(floors_payload):
            union_above = unary_union([fp[1] for fp in floors_payload[floor_idx + 1 :]])

        upper_secs_all: List[Dict[str, Any]] = []
        for _pp, _poly_u, rr_u, (ox_u, oy_u) in floors_payload[floor_idx + 1 :]:
            dxu = float(ox_u - ox)
            dyu = float(oy_u - oy)
            upper_secs_all.extend(_translate_sections(rr_u.get("sections") or [], dxu, dyu))

        secs = rr.get("sections") or []
        conns = rr.get("connections") or []
        free_ends = _free_roof_ends(secs, conns)

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

            sec_t = {
                **sec,
                "bounding_rect": [(float(p[0]) + ox, float(p[1]) + oy) for p in br],
                "ridge_line": [
                    (float(p[0]) + ox, float(p[1]) + oy) for p in (sec.get("ridge_line") or [])
                ],
            }
            color = colors[s_idx % len(colors)]
            fe = free_ends[s_idx] if s_idx < len(free_ends) else {}
            for face in _roof_section_faces_pyramid(sec_t, z1, roof_angle_rad, fe, upper_secs_all):
                meshes_to_add.append((face, color))

    # scene bounds from accumulated xy coords
    if not xs_all or not ys_all:
        # fallback to polygon bounds (robust even if no coords were accumulated)
        bxs: List[float] = []
        bys: List[float] = []
        for _p, poly, _rr, _off in floors_payload:
            try:
                minx0, miny0, maxx0, maxy0 = poly.bounds
                bxs.extend([float(minx0), float(maxx0)])
                bys.extend([float(miny0), float(maxy0)])
            except Exception:
                continue
        if not bxs or not bys:
            raise RuntimeError("Nu pot determina limitele scenei (bounds) pentru randarea 3D.")
        minx_b, maxx_b = min(bxs), max(bxs)
        miny_b, maxy_b = min(bys), max(bys)
    else:
        minx_b, maxx_b = float(min(xs_all)), float(max(xs_all))
        miny_b, maxy_b = float(min(ys_all)), float(max(ys_all))

    pv.set_plot_theme("document")

    cx, cy = (minx_b + maxx_b) / 2, (miny_b + maxy_b) / 2
    focal = (cx, cy, roof_base_z / 2)
    r = z_max * 2.0

    def _render_view(elev_deg: float, azim_deg: float) -> np.ndarray:
        plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
        plotter.background_color = "white"
        for face_pts, color_hex in meshes_to_add:
            pts = np.array(face_pts, dtype=np.float64)
            cells = np.array(_poly_to_pyvista_cells(face_pts), dtype=np.int32)
            mesh = pv.PolyData(pts, cells)
            plotter.add_mesh(
                mesh,
                color=_hex_to_rgb(color_hex),
                show_edges=True,
                edge_color="black",
            )

        azim_rad = np.radians(azim_deg)
        elev_rad = np.radians(elev_deg)
        cam_x = r * np.cos(elev_rad) * np.cos(azim_rad) + cx
        cam_y = r * np.cos(elev_rad) * np.sin(azim_rad) + cy
        cam_z = r * np.sin(elev_rad)
        plotter.camera.position = (cam_x, cam_y, cam_z)
        plotter.camera.focal_point = focal
        plotter.camera.view_up = (0, 0, 1)
        plotter.reset_camera_clipping_range()
        plotter.render()
        img = plotter.screenshot(None, return_img=True)
        plotter.close()
        if img is None:
            raise RuntimeError("PyVista screenshot a returnat None (randare offscreen eșuată).")
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr

    view_list = views if views else [{"elev": 30, "azim": 45, "title": "Default"}]
    imgs: List[np.ndarray] = []
    for v in view_list:
        imgs.append(_render_view(float(v.get("elev", 30)), float(v.get("azim", 45))))

    imgs = [im[:, :, :3] if (im.ndim == 3 and im.shape[2] >= 3) else im for im in imgs]
    max_h = max(im.shape[0] for im in imgs)
    padded: List[np.ndarray] = []
    for im in imgs:
        if im.shape[0] < max_h:
            pad = max_h - im.shape[0]
            im = np.pad(im, ((0, pad), (0, 0), (0, 0)), mode="constant", constant_values=255)
        padded.append(im)
    combined = np.concatenate(padded, axis=1) if len(padded) > 1 else padded[0]

    if output_path:
        from PIL import Image

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(combined).save(output_path)
        logger.info("Saved PyVista 3D pyramid to %s", output_path)
    return True

