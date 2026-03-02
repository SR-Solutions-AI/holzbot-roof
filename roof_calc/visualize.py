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

from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Hotfix: ensure section area_px for rectangle debug images ("Area: 0.0 px??")
# ---------------------------------------------------------------------------
_original_visualize_individual_rectangles = visualize_individual_rectangles  # type: ignore[name-defined]


def _section_area_from_bounding_rect(sec: Dict[str, Any]) -> float:
    br = sec.get("bounding_rect") or []
    if len(br) < 3:
        return 0.0
    xs = [float(p[0]) for p in br]
    ys = [float(p[1]) for p in br]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def visualize_individual_rectangles(path: str, roof_result: Dict[str, Any], *, output_dir: Optional[str] = None):  # type: ignore[no-redef]
    """Override: inject area_px from bounding_rect so debug image shows correct area."""
    sections = roof_result.get("sections") or []
    for s in sections:
        if "area_px" not in s or (isinstance(s.get("area_px"), (int, float)) and s.get("area_px") == 0):
            s["area_px"] = _section_area_from_bounding_rect(s)
    return _original_visualize_individual_rectangles(path, roof_result, output_dir=output_dir)


# ---------------------------------------------------------------------------
# Hotfix: multi-floor alignment
#
# We observed misaligned floor polygons (different coordinate systems per floor),
# which breaks 3D rendering (roofs/walls appear far apart). We override
# `_ordered_floor_polygons` to align all floors to the SAME reference center
# (derived from the roof floor's largest rectangle bbox when available).
# ---------------------------------------------------------------------------
#
# Hotfix: a_lines.png – ridge-urile secundare trebuie extinse până ating ridge-ul principal.
# ---------------------------------------------------------------------------

from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

# Save original before override (loaded from pyc)
_original_visualize_a_frame_lines = visualize_a_frame_lines  # type: ignore[name-defined]


def _extend_secondary_ridges_to_main(roof_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extinde ridge-urile secundare până ating ridge-ul principal al etajului respectiv."""
    from roof_calc.overhang import extend_secondary_sections_to_main_ridge

    sections = roof_result.get("sections") or []
    extended = extend_secondary_sections_to_main_ridge(sections)
    return {**roof_result, "sections": extended}


def _ridge_intersection_corner_lines(
    sections: List[Dict[str, Any]],
) -> List[Tuple[Tuple[float, float], List[Tuple[float, float]]]]:
    """Delegatează la overhang.ridge_intersection_corner_lines."""
    from roof_calc.overhang import ridge_intersection_corner_lines
    return ridge_intersection_corner_lines(sections)


def visualize_a_frame_lines(  # type: ignore[no-redef]
    wall_mask_path: str,
    roof_result: Dict[str, Any],
    *,
    output_path: str,
    roof_angle_deg: float = 30.0,
    wall_height: float = 300.0,
    upper_floor_roof_sections: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Override: linii mov doar în interiorul fiecărui dreptunghi, ridge propriu, contur exterior."""
    from pathlib import Path

    sections = roof_result.get("sections") or []
    if not sections:
        return _original_visualize_a_frame_lines(
            wall_mask_path,
            roof_result,
            output_path=output_path,
            roof_angle_deg=roof_angle_deg,
            wall_height=wall_height,
            upper_floor_roof_sections=upper_floor_roof_sections,
        )
    from roof_calc.overhang import extend_secondary_sections_to_main_ridge
    sections = extend_secondary_sections_to_main_ridge(sections)

    import cv2

    img = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    h, w = img.shape[:2]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        return False

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f"Trasarea liniilor pentru acoperiș tip A-frame (gable, {roof_angle_deg:.0f}°)")
    ax.set_aspect("equal")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)

    main_sec = next((s for s in sections if s.get("is_main")), None)
    if main_sec is None:
        def _ar(s: Dict[str, Any]) -> float:
            br = s.get("bounding_rect") or []
            if len(br) < 3:
                return 0.0
            xs = [float(p[0]) for p in br]
            ys = [float(p[1]) for p in br]
            return (max(xs) - min(xs)) * (max(ys) - min(ys))
        main_sec = max(sections, key=_ar)

    from roof_calc.flood_fill import flood_fill_interior, get_house_shape_mask

    filled = flood_fill_interior(img)
    house_mask = get_house_shape_mask(filled)

    def _contours_from_mask(mask: Any) -> List[List[Tuple[float, float]]]:
        """Extrage toate contururile exterioare din mască (fiecare regiune albă)."""
        import cv2 as _cv2
        import numpy as _np
        if mask is None or not hasattr(mask, "shape"):
            return []
        arr = _np.asarray(mask, dtype=_np.uint8)
        if arr.size == 0:
            return []
        contours, _ = _cv2.findContours(arr, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            if len(c) < 3:
                continue
            pts = [(float(p[0][0]), float(p[0][1])) for p in c]
            pts.append(pts[0])
            out.append(pts)
        return out

    contour_pts = _contours_from_mask(house_mask)
    segments_ridge: List[List[List[float]]] = []
    segments_magenta: List[List[List[float]]] = []
    segments_contour: List[List[List[float]]] = []

    for pts in contour_pts:
        if len(pts) >= 2:
            xs_g = [p[0] for p in pts]
            ys_g = [p[1] for p in pts]
            ax.plot(xs_g, ys_g, color="green", linewidth=2, linestyle="-")
            for i in range(len(pts) - 1):
                segments_contour.append([[float(pts[i][0]), float(pts[i][1])], [float(pts[i + 1][0]), float(pts[i + 1][1])]])

    for sec in sections:
        ridge = sec.get("ridge_line") or []
        if len(ridge) >= 2:
            lw = 2.5 if sec is main_sec else 1.5
            ax.plot([ridge[0][0], ridge[1][0]], [ridge[0][1], ridge[1][1]], color="darkred", linewidth=lw)
            segments_ridge.append([[float(ridge[0][0]), float(ridge[0][1])], [float(ridge[1][0]), float(ridge[1][1])]])

    from roof_calc.overhang import ridge_intersection_corner_lines
    corner_lines = ridge_intersection_corner_lines(sections, per_section=True)
    for item in corner_lines:
        pt = item[0]
        corners = item[1]
        ix, iy = float(pt[0]), float(pt[1])
        for cx, cy in corners:
            ax.plot([ix, cx], [iy, cy], color="#CC00FF", linewidth=2, linestyle="-", zorder=10)
            segments_magenta.append([[ix, iy], [float(cx), float(cy)]])

    try:
        import json
        out_dir = Path(output_path).parent
        segments_path = out_dir / "a_lines_segments.json"
        payload = {
            "ridge": segments_ridge,
            "magenta": segments_magenta,
            "contour": segments_contour,
        }
        segments_path.write_text(json.dumps(payload, indent=0), encoding="utf-8")
    except Exception:
        pass

    handles = [
        Line2D([0], [0], color="darkred", lw=2, label="Ridge"),
        Line2D([0], [0], color="green", lw=2, ls="-", label="Contur exterior"),
        Line2D([0], [0], color="#CC00FF", lw=2, label="Linii intersecție → colțuri"),
    ]
    ax.legend(handles=handles, loc="upper right")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        plt.close()
        return False


def _get_polygons_from_mask(mask: Any) -> List[Any]:
    """Extrage toate poligoanele din mască (fiecare regiune albă = un poligon)."""
    import cv2 as _cv2
    import numpy as _np
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
    except ImportError:
        return []
    if mask is None or not hasattr(mask, "shape"):
        return []
    arr = _np.asarray(mask, dtype=_np.uint8)
    if arr.size == 0:
        return []
    contours, _ = _cv2.findContours(arr, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if len(c) < 3:
            continue
        try:
            coords = [(float(p[0][0]), float(p[0][1])) for p in c]
            poly = ShapelyPolygon(coords)
            if poly.is_empty:
                continue
            coords_list = list(poly.exterior.coords)
            if len(coords_list) >= 3:
                area = 0.0
                for i in range(len(coords_list) - 1):
                    area += coords_list[i][0] * coords_list[i + 1][1] - coords_list[i + 1][0] * coords_list[i][1]
                if area < 0:
                    poly = ShapelyPolygon(coords_list[::-1])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty and poly.is_valid:
                polys.append(poly)
        except Exception:
            pass
    return polys


def visualize_a_frame_faces(
    wall_mask_path: str,
    roof_result: Dict[str, Any],
    *,
    output_path: str,
    roof_angle_deg: float = 30.0,
    wall_height: float = 300.0,
    upper_floor_roof_sections: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """
    După a_lines.png: folosește exact cele 3 tipuri de segmente (ridge, contur exterior, magenta)
    și umple toate golurile dintre ele – fiecare regiune închisă devine o față cu culoare unică.
    Fără fețe suprapuse, delimitate strict de aceste segmente.
    """
    import random as _random

    sections = roof_result.get("sections") or []
    if not sections:
        return False
    from roof_calc.overhang import extend_secondary_sections_to_main_ridge
    sections = extend_secondary_sections_to_main_ridge(sections)

    import cv2

    img = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    h, w = img.shape[:2]

    from roof_calc.flood_fill import flood_fill_interior, get_house_shape_mask
    from roof_calc.overhang import ridge_intersection_corner_lines
    from roof_calc.roof_segments_3d import get_faces_3d_from_segments

    filled = flood_fill_interior(img)
    house_mask = get_house_shape_mask(filled)
    floor_polys = _get_polygons_from_mask(house_mask)
    if not floor_polys:
        from roof_calc.geometry import extract_polygon
        fp = extract_polygon(house_mask)
        if fp is not None and not getattr(fp, "is_empty", True):
            if hasattr(fp, "geoms"):
                floor_polys = [g for g in fp.geoms if not getattr(g, "is_empty", True)]
            elif hasattr(fp, "exterior"):
                floor_polys = [fp]
        if not floor_polys:
            return False

    def _section_center(sec: Dict[str, Any]) -> Tuple[float, float]:
        br = sec.get("bounding_rect") or []
        if len(br) < 3:
            return (0.0, 0.0)
        xs = [float(p[0]) for p in br]
        ys = [float(p[1]) for p in br]
        return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)

    from shapely.geometry import Point as ShapelyPoint
    from shapely.geometry import Polygon as ShapelyPolygon

    faces_data: List[Dict[str, Any]] = []
    out_dir = Path(output_path).parent
    # Fețe doar din get_faces_3d_from_segments per poligon – fără a_lines_segments (produce fețe în plus)
    processed_sec_ids: set = set()
    if True:
        for fp in floor_polys:
            try:
                fp = fp.buffer(0) if not fp.is_valid else fp
            except Exception:
                pass
            secs_in = [s for s in sections if fp.contains(ShapelyPoint(*_section_center(s)))]
            if not secs_in:
                for s in sections:
                    br = s.get("bounding_rect") or []
                    if len(br) < 3:
                        continue
                    try:
                        rp = ShapelyPolygon(br)
                        if not rp.is_empty and fp.intersects(rp):
                            secs_in.append(s)
                    except Exception:
                        pass
            if not secs_in:
                continue
            for s in secs_in:
                bid = tuple(tuple(p) for p in (s.get("bounding_rect") or [])[:4])
                if bid:
                    processed_sec_ids.add(bid)
            cl = ridge_intersection_corner_lines(secs_in, floor_polygon=fp, per_section=True)
            try:
                fdata = get_faces_3d_from_segments(
                    secs_in,
                    fp,
                    wall_height=wall_height,
                    roof_angle_deg=roof_angle_deg,
                    corner_lines=cl,
                    use_section_rect_eaves=False,
                )
                faces_data.extend(fdata)
            except Exception:
                pass

        # Secțiuni neprocesate: folosim bounding_rect ca floor polygon
        for s in sections:
            br = s.get("bounding_rect") or []
            if len(br) < 3:
                continue
            bid = tuple(tuple(p) for p in br[:4])
            if bid in processed_sec_ids:
                continue
            try:
                fp = ShapelyPolygon(br)
                if fp.is_empty or not fp.is_valid:
                    fp = fp.buffer(0)
                if fp.is_empty:
                    continue
                secs_in = [s]
                cl = ridge_intersection_corner_lines(secs_in, floor_polygon=fp, per_section=True)
                fdata = get_faces_3d_from_segments(
                    secs_in,
                    fp,
                    wall_height=wall_height,
                    roof_angle_deg=roof_angle_deg,
                    corner_lines=cl,
                    use_section_rect_eaves=False,
                )
                faces_data.extend(fdata)
            except Exception:
                pass

    # Salvare fețe pentru house_a_frame (folosește exact aceleași fețe)
    try:
        import json
        out_dir = Path(output_path).parent
        faces_path = out_dir / "a_faces_faces.json"
        payload = {
            "floor_path": str(Path(wall_mask_path).resolve()),
            "faces": [{"vertices_3d": f.get("vertices_3d", [])} for f in faces_data],
        }
        faces_path.write_text(json.dumps(payload, indent=0), encoding="utf-8")
    except Exception:
        pass

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f"Fețe acoperiș A-frame (ridge, contur exterior, magenta)")
    ax.set_aspect("equal")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)

    rng = _random.Random(42)
    # Desenăm fețele de la spate în față (după centroid y) ca să evităm artefacte de suprapunere
    faces_with_centroid = []
    for f in faces_data:
        vs = f.get("vertices_3d") or []
        if len(vs) < 3:
            continue
        cy = sum(float(v[1]) for v in vs) / len(vs)
        faces_with_centroid.append((cy, vs, f))
    faces_with_centroid.sort(key=lambda t: t[0], reverse=True)

    for _cy, vs, _f in faces_with_centroid:
        xs = [float(v[0]) for v in vs]
        ys = [float(v[1]) for v in vs]
        color = (rng.random(), rng.random(), rng.random(), 0.5)
        ax.fill(xs, ys, facecolor=color, edgecolor="black", linewidth=0.5)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        plt.close()
        return False


def _ordered_floor_polygons(  # type: ignore[override]
    all_floor_paths: List[str],
    roof_floor_path: Optional[str],
    floor_roof_results: Optional[List[Dict[str, Any]]],
) -> List[Tuple[str, Any]]:
    import cv2  # local import
    import numpy as np
    from shapely import affinity as shapely_affinity
    from shapely.geometry import box as shapely_box

    from roof_calc.flood_fill import flood_fill_interior, get_house_shape_mask
    from roof_calc.geometry import extract_polygon

    # 1) Build polygons for each floor – nu omitem niciun etaj (păstrăm ordinea și numărul)
    floors: List[Tuple[str, Any]] = []
    for p in all_floor_paths or []:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        poly = None
        if img is not None:
            filled = flood_fill_interior(img)
            house_mask = get_house_shape_mask(filled)
            poly = extract_polygon(house_mask)
        if poly is None or poly.is_empty:
            poly = shapely_box(0.0, 0.0, 1.0, 1.0)
        floors.append((p, poly))

    if not floors:
        return []

    # 2) Păstrăm ordinea engine (floor_0 = beci la baza 3D) dacă path-urile sunt floor_NN
    import re
    def _floor_idx(p: str) -> int:
        stem = Path(p).stem.lower()
        m = re.match(r"floor_?(\d+)", stem)
        return int(m.group(1)) if m else -1
    indices = [_floor_idx(p) for p, _ in floors]
    if all(i >= 0 for i in indices) and len(set(indices)) == len(indices):
        floors.sort(key=lambda t: _floor_idx(t[0]))
    else:
        # Sort bottom->top by footprint area (largest first) doar când nu avem nume floor_NN
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


