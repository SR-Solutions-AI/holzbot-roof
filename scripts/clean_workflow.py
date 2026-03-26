#!/usr/bin/env python3
"""
Workflow curat:
- output/rectangles/floor_X/ - măști dreptunghiuri rămase (după eliminare suprapuneri)
- output/roof_types/floor_X/{0_w,1_w,2_w,4_w,4.5_w}/ - lines.png, faces.png, frame.html per tip
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")
(ROOT / ".mplconfig").mkdir(exist_ok=True)
(ROOT / ".cache").mkdir(exist_ok=True)

from roof_calc import calculate_roof_from_walls
from roof_calc.visualize import visualize_individual_rectangles, _ordered_floor_polygons
from roof_calc.roof_types_workflow import (
    remove_overlapping_rectangles,
    generate_roof_type_outputs,
    generate_entire_frame_html,
    populate_mixed_unfold_and_metrics,
)

import cv2


def _footprint_section_from_wall_mask(wall_mask_path: str) -> Optional[dict]:
    """Pentru acoperiș plat (unghi 0°): un singur dreptunghi = footprint-ul clădirii din mască."""
    img = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    binary = (img > 10).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, rw, rh = cv2.boundingRect(largest)
    if rw < 2 or rh < 2:
        return None
    bounding_rect = [
        (float(x), float(y)),
        (float(x + rw), float(y)),
        (float(x + rw), float(y + rh)),
        (float(x), float(y + rh)),
    ]
    return {
        "bounding_rect": bounding_rect,
        "ridge_line": [],
        "ridge_orientation": "horizontal",
        "is_main": True,
    }


def _translate_sections(sections: list, dx: float, dy: float) -> list:
    """Translatează bounding_rect + ridge_line cu (dx, dy)."""
    out = []
    for sec in sections:
        rect = sec.get("bounding_rect", [])
        ridge = sec.get("ridge_line", [])
        new_rect = [(float(p[0]) + dx, float(p[1]) + dy) for p in rect] if rect else []
        new_ridge = [(float(p[0]) + dx, float(p[1]) + dy) for p in ridge] if ridge and len(ridge) >= 2 else []
        out.append({
            "bounding_rect": new_rect,
            "ridge_line": new_ridge,
            "ridge_orientation": sec.get("ridge_orientation", "horizontal"),
            "is_main": sec.get("is_main", False),
        })
    return out


def _polygon_to_mask(poly, width: int, height: int):
    """Rasterizează un poligon Shapely la mască binară (height, width). Coordonatele poly sunt în (x,y) = (col, row)."""
    import numpy as np
    if poly is None or getattr(poly, "is_empty", True):
        return np.zeros((height, width), dtype=np.uint8)
    try:
        coords = []
        if hasattr(poly, "exterior") and poly.exterior is not None:
            coords = list(poly.exterior.coords)
        if not coords:
            return np.zeros((height, width), dtype=np.uint8)
        pts = np.array([[float(x), float(y)] for x, y in coords], dtype=np.int32)
        pts = np.clip(pts, [0, 0], [width - 1, height - 1])
        pts = pts.reshape(-1, 1, 2)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return mask
    except Exception:
        return np.zeros((height, width), dtype=np.uint8)


def _overlap_area_at_offset(
    ref_mask, ref_w: int, ref_h: int,
    other_mask, other_w: int, other_h: int,
    dx: int, dy: int,
) -> int:
    """
    Suprafața suprapusă (în pixeli) când etajul 'other' e translat cu (dx, dy) în coordonatele etajului de referință.
    ref_mask: (ref_h, ref_w), other_mask: (other_h, other_w).
    """
    x_lo = max(0, dx)
    x_hi = min(ref_w, dx + other_w)
    y_lo = max(0, dy)
    y_hi = min(ref_h, dy + other_h)
    if x_lo >= x_hi or y_lo >= y_hi:
        return 0
    ref_crop = ref_mask[y_lo:y_hi, x_lo:x_hi]
    other_crop = other_mask[y_lo - dy : y_hi - dy, x_lo - dx : x_hi - dx]
    if ref_crop.shape != other_crop.shape:
        return 0
    return int(((ref_crop > 0) & (other_crop > 0)).sum())


def _trim_interior_mask_to_target_area(
    interior_path: str,
    target_area: float,
    tolerance_ratio: float = 0.05,
    max_tries: int = 250,
) -> Optional[Tuple[Any, int, int]]:
    """
    Încarcă masca 09_interior.png (portocaliu pe negru), o descompune în componente conexe,
    și încearcă să elimine componente (random sau pe rând) până când aria rămasă e ≈ target_area.
    Returnează (trimmed_mask, w, h) sau None dacă nu s-a reușit.
    """
    import numpy as np
    img = cv2.imread(interior_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    binary = (gray > 10).astype(np.uint8) * 255
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_cc < 2:
        return None
    total_area = int((binary > 0).sum())
    if total_area <= 0:
        return None
    component_areas = []
    for i in range(1, n_cc):
        area = int((labels == i).sum())
        if area > 0:
            component_areas.append((i, area))
    if not component_areas:
        return None
    component_areas.sort(key=lambda x: x[1], reverse=True)
    low = int(target_area * (1 - tolerance_ratio))
    high = int(target_area * (1 + tolerance_ratio))
    if low <= total_area <= high:
        return (binary, w, h)
    import random
    rng = random.Random(42)
    for _ in range(max_tries):
        n_remove = rng.randint(1, min(len(component_areas), max(1, len(component_areas) // 2 + 1)))
        to_remove = set(rng.sample(range(len(component_areas)), n_remove))
        trimmed = binary.copy()
        for idx in to_remove:
            cc_id = component_areas[idx][0]
            trimmed[labels == cc_id] = 0
        remaining = int((trimmed > 0).sum())
        if low <= remaining <= high:
            return (trimmed, w, h)
    from itertools import combinations
    for n_remove in range(1, min(7, len(component_areas) + 1)):
        for combo in combinations(range(len(component_areas)), n_remove):
            trimmed = binary.copy()
            for idx in combo:
                cc_id = component_areas[idx][0]
                trimmed[labels == cc_id] = 0
            remaining = int((trimmed > 0).sum())
            if low <= remaining <= high:
                return (trimmed, w, h)
    return None


def _align_rects_by_edges(
    ref_sec: dict,
    other_sec: dict,
    ref_w: int, ref_h: int,
    other_w: int, other_h: int,
) -> Tuple[int, int]:
    """Găsește (dx, dy) care maximizează suprapunerea celor două dreptunghiuri (bounding box)."""
    def _bbox(sec):
        br = sec.get("bounding_rect") or []
        if len(br) < 3:
            return (0, 0, 0, 0)
        xs, ys = [float(p[0]) for p in br], [float(p[1]) for p in br]
        return (min(xs), min(ys), max(xs), max(ys))
    r = _bbox(ref_sec)
    o = _bbox(other_sec)
    rw, rh = r[2] - r[0], r[3] - r[1]
    ow, oh = o[2] - o[0], o[3] - o[1]
    if rw <= 0 or rh <= 0 or ow <= 0 or oh <= 0:
        return (0, 0)
    step = int(max(1, min(rw, rh, ow, oh) // 30))
    range_x = int(min(80, max(20, (ref_w + other_w) // 4)))
    range_y = int(min(80, max(20, (ref_h + other_h) // 4)))
    best_overlap = -1
    best_dx, best_dy = 0, 0
    for dx in range(-range_x, range_x + 1, step):
        for dy in range(-range_y, range_y + 1, step):
            inter_lo_x = max(r[0], o[0] + dx)
            inter_lo_y = max(r[1], o[1] + dy)
            inter_hi_x = min(r[2], o[2] + dx)
            inter_hi_y = min(r[3], o[3] + dy)
            if inter_lo_x < inter_hi_x and inter_lo_y < inter_hi_y:
                overlap = (inter_hi_x - inter_lo_x) * (inter_hi_y - inter_lo_y)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_dx, best_dy = dx, dy
    if step > 1:
        for dx in range(best_dx - step, best_dx + step + 1):
            for dy in range(best_dy - step, best_dy + step + 1):
                inter_lo_x = max(r[0], o[0] + dx)
                inter_lo_y = max(r[1], o[1] + dy)
                inter_hi_x = min(r[2], o[2] + dx)
                inter_hi_y = min(r[3], o[3] + dy)
                if inter_lo_x < inter_hi_x and inter_lo_y < inter_hi_y:
                    overlap = (inter_hi_x - inter_lo_x) * (inter_hi_y - inter_lo_y)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_dx, best_dy = dx, dy
    return (best_dx, best_dy)


def _compute_overlay_offsets(floor_paths: List[str], roof_results: List[dict], floors_meta: Optional[List[dict]] = None, debug_output_dir: Optional[Path] = None) -> dict:
    """
    Offsets pentru alinierea etajelor. Când dreptunghiurile nu au aceeași suprafață,
    aliniem cel mai mare dreptunghi de la etajul de referință cu cel mai mare dreptunghi
    de la fiecare alt etaj (prin translații brute-force), astfel încât etajele să fie aliniate
    cât mai bine posibil.
    Dacă debug_output_dir e setat, salvează imagini de debug când se folosește 09_interior (masca tăiată + overlay).
    """
    from roof_calc.flood_fill import flood_fill_interior, get_house_shape_mask
    from roof_calc.geometry import extract_polygon

    def _poly(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        filled = flood_fill_interior(img)
        mask = get_house_shape_mask(filled)
        return extract_polygon(mask)

    def _sec_area(sec):
        br = sec.get("bounding_rect") or []
        if len(br) < 3:
            return 0.0
        xs, ys = [float(p[0]) for p in br], [float(p[1]) for p in br]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    def _sec_center(sec):
        br = sec.get("bounding_rect") or []
        if len(br) < 3:
            return (0.0, 0.0)
        xs, ys = [float(p[0]) for p in br], [float(p[1]) for p in br]
        return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)

    def _mask_center(mask):
        if mask is None:
            return (0.0, 0.0)
        m = cv2.moments((mask > 0).astype(np.uint8))
        if m.get("m00", 0.0) == 0.0:
            return (0.0, 0.0)
        return (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"]))

    # Referință: etajul cu cea mai mare suprafață (polygon house shape)
    polys = [_poly(p) for p in floor_paths]
    best_i = 0
    best_a = 0.0
    for i, poly in enumerate(polys):
        if poly is not None and not poly.is_empty:
            a = float(poly.area)
            if a > best_a:
                best_a = a
                best_i = i

    ref_path = floor_paths[best_i]
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    ref_h, ref_w = (ref_img.shape[0], ref_img.shape[1]) if ref_img is not None else (0, 0)
    ref_mask = _polygon_to_mask(polys[best_i], ref_w, ref_h) if ref_w and ref_h else None
    rr_ref = roof_results[best_i] if best_i < len(roof_results) else {}

    # Cel mai mare dreptunghi la etajul de referință (pentru aliniere explicită)
    ref_sections = rr_ref.get("sections") or []
    ref_main_sec = max(ref_sections, key=_sec_area) if ref_sections else None
    cx_ref, cy_ref = _sec_center(ref_main_sec) if ref_main_sec else (0.0, 0.0)
    if cx_ref == 0 and cy_ref == 0 and polys[best_i]:
        c = polys[best_i].centroid
        cx_ref, cy_ref = float(c.x), float(c.y)

    def _areas_match_rect(a: float, b: float) -> bool:
        if a <= 0 or b <= 0:
            return False
        return 0.9 * b <= a <= 1.1 * b or 0.9 * a <= b <= 1.1 * a

    meta_by_basename = {}
    if floors_meta:
        for m in floors_meta:
            fp = m.get("floor_path")
            if fp:
                meta_by_basename[Path(fp).name] = m

    out = {}
    for i, p in enumerate(floor_paths):
        if i == best_i:
            out[p] = (0, 0)
            continue
        rr = roof_results[i] if i < len(roof_results) else {}
        other_img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        other_h, other_w = (other_img.shape[0], other_img.shape[1]) if other_img is not None else (0, 0)
        other_mask = _polygon_to_mask(polys[i], other_w, other_h) if other_w and other_h and i < len(polys) and polys[i] else None

        # 1) Aliniere după cel mai mare dreptunghi: translație astfel încât centrul celui mai mare rect al etajului curent să coincidă cu centrul celui mai mare rect al referinței
        other_sections = rr.get("sections") or []
        other_main_sec = max(other_sections, key=_sec_area) if other_sections else None
        use_largest_rect_align = ref_main_sec is not None and other_main_sec is not None and _sec_area(ref_main_sec) > 0 and _sec_area(other_main_sec) > 0

        if use_largest_rect_align:
            area_ref = _sec_area(ref_main_sec)
            area_other = _sec_area(other_main_sec)
            areas_within_10 = _areas_match_rect(area_ref, area_other)
            if areas_within_10:
                cxi, cyi = _sec_center(other_main_sec)
                dx0 = int(round(cx_ref - cxi))
                dy0 = int(round(cy_ref - cyi))
                # Prefer geometric centroid alignment when masks are available.
                mx_ref, my_ref = _mask_center(ref_mask)
                mx_other, my_other = _mask_center(other_mask)
                if mx_ref or my_ref or mx_other or my_other:
                    dx0 = int(round(mx_ref - mx_other))
                    dy0 = int(round(my_ref - my_other))
                step = max(1, min(ref_w, ref_h, other_w, other_h) // 80)
                range_fine = min(180, max(30, step * 8))
                best_overlap = -1
                best_dx, best_dy = dx0, dy0
                if ref_mask is not None and other_mask is not None and ref_mask.any() and other_mask.any():
                    for dx in range(dx0 - range_fine, dx0 + range_fine + 1, step):
                        for dy in range(dy0 - range_fine, dy0 + range_fine + 1, step):
                            overlap = _overlap_area_at_offset(
                                ref_mask, ref_w, ref_h,
                                other_mask, other_w, other_h,
                                dx, dy,
                            )
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_dx, best_dy = dx, dy
                    if step > 1:
                        for dx in range(best_dx - step, best_dx + step + 1):
                            for dy in range(best_dy - step, best_dy + step + 1):
                                overlap = _overlap_area_at_offset(
                                    ref_mask, ref_w, ref_h,
                                    other_mask, other_w, other_h,
                                    dx, dy,
                                )
                                if overlap > best_overlap:
                                    best_overlap = overlap
                                    best_dx, best_dy = dx, dy
                out[p] = (best_dx, best_dy)
                if debug_output_dir is not None and ref_mask is not None and other_mask is not None:
                    try:
                        debug_output_dir.mkdir(parents=True, exist_ok=True)
                        other_basename = Path(p).stem
                        if not (debug_output_dir / "ref_floor.png").exists():
                            cv2.imwrite(str(debug_output_dir / "ref_floor.png"), ref_mask)
                        cv2.imwrite(str(debug_output_dir / f"other_{other_basename}.png"), other_mask)
                        x_ref = max(0, -best_dx)
                        y_ref = max(0, -best_dy)
                        x_other = max(0, best_dx)
                        y_other = max(0, best_dy)
                        h_canvas = max(y_ref + ref_h, y_other + other_h)
                        w_canvas = max(x_ref + ref_w, x_other + other_w)
                        canvas = np.zeros((h_canvas, w_canvas, 3), dtype=np.uint8)
                        canvas[y_ref:y_ref + ref_h, x_ref:x_ref + ref_w, 1] = ref_mask
                        canvas[y_other:y_other + other_h, x_other:x_other + other_w, 2] = np.maximum(
                            canvas[y_other:y_other + other_h, x_other:x_other + other_w, 2], other_mask
                        )
                        overlap_zone = (canvas[:, :, 1] > 0) & (canvas[:, :, 2] > 0)
                        canvas[:, :, 0][overlap_zone] = 0
                        canvas[:, :, 1][overlap_zone] = 255
                        canvas[:, :, 2][overlap_zone] = 255
                        cv2.imwrite(str(debug_output_dir / f"overlay_{other_basename}.png"), canvas)
                        print(f"   ✓ [overlay_debug] Salvat (ariile ~egale, centru): {debug_output_dir}/ overlay_{other_basename}.png")
                    except Exception as e:
                        print(f"   ⚠ [overlay_debug] Nu am putut salva imagini: {e}")
                continue
            big_area = max(area_ref, area_other)
            small_area = min(area_ref, area_other)
            # Folosim 09_interior pe etajul cu suprafața mai mare (îl trim la aria mică)
            larger_is_other = area_other >= area_ref
            interior_path = None
            if meta_by_basename:
                other_basename = Path(p).name
                ref_basename = Path(ref_path).name
                meta_other = meta_by_basename.get(other_basename, {})
                meta_ref = meta_by_basename.get(ref_basename, {})
                if larger_is_other:
                    interior_path = (meta_other.get("interior_mask_path") or "").strip() or None
                else:
                    interior_path = (meta_ref.get("interior_mask_path") or "").strip() or None
            trimmed_result = None
            if interior_path and Path(interior_path).exists():
                trimmed_result = _trim_interior_mask_to_target_area(
                    interior_path,
                    target_area=small_area,
                    tolerance_ratio=0.05,
                    max_tries=250,
                )
            if trimmed_result is None and use_largest_rect_align and debug_output_dir is not None:
                why = (
                    "09_interior lipsește sau fișierul nu există" if not (interior_path and Path(interior_path).exists()) else "trim eșuat (nu s-a găsit combinație de camere pentru aria țintă)"
                )
                print(f"   ℹ️ [overlay_debug] Nu folosesc 09_interior: {why}")
            if trimmed_result is not None:
                trimmed_mask, tw, th = trimmed_result
                if larger_is_other:
                    # Etajul „other” e mai mare: am trimat 09_interior al other la small_area; îl suprapunem pe ref
                    use_ref_mask = ref_mask
                    use_other_mask = trimmed_mask
                    use_ow, use_oh = tw, th
                else:
                    # Referința e mai mare: am trimat 09_interior al ref la small_area; suprapunem other pe masca trimată
                    use_ref_mask = trimmed_mask
                    use_other_mask = other_mask
                    use_ow, use_oh = other_w, other_h
                if ref_mask is not None and use_ref_mask is not None and use_other_mask is not None and use_ref_mask.any() and (use_other_mask > 0).any():
                    step = max(1, min(ref_w, ref_h, use_ow, use_oh) // 80)
                    range_fine = min(50, max(15, step * 5))
                    cxi, cyi = _sec_center(other_main_sec)
                    dx0 = int(round(cx_ref - cxi))
                    dy0 = int(round(cy_ref - cyi))
                    mx_ref, my_ref = _mask_center(use_ref_mask)
                    mx_other, my_other = _mask_center(use_other_mask)
                    if mx_ref or my_ref or mx_other or my_other:
                        dx0 = int(round(mx_ref - mx_other))
                        dy0 = int(round(my_ref - my_other))
                    best_overlap = -1
                    best_dx, best_dy = dx0, dy0
                    for dx in range(dx0 - range_fine, dx0 + range_fine + 1, step):
                        for dy in range(dy0 - range_fine, dy0 + range_fine + 1, step):
                            overlap = _overlap_area_at_offset(
                                use_ref_mask, ref_w, ref_h,
                                use_other_mask, use_ow, use_oh,
                                dx, dy,
                            )
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_dx, best_dy = dx, dy
                    if step > 1:
                        for dx in range(best_dx - step, best_dx + step + 1):
                            for dy in range(best_dy - step, best_dy + step + 1):
                                overlap = _overlap_area_at_offset(
                                    use_ref_mask, ref_w, ref_h,
                                    use_other_mask, use_ow, use_oh,
                                    dx, dy,
                                )
                                if overlap > best_overlap:
                                    best_overlap = overlap
                                    best_dx, best_dy = dx, dy
                    out[p] = (best_dx, best_dy)
                    if debug_output_dir is not None:
                        try:
                            debug_output_dir.mkdir(parents=True, exist_ok=True)
                            other_basename = Path(p).stem
                            if not (debug_output_dir / "ref_floor.png").exists():
                                cv2.imwrite(str(debug_output_dir / "ref_floor.png"), ref_mask if larger_is_other else use_ref_mask)
                            if larger_is_other:
                                cv2.imwrite(str(debug_output_dir / f"trimmed_{other_basename}.png"), use_other_mask)
                            else:
                                cv2.imwrite(str(debug_output_dir / "trimmed_ref_floor.png"), use_ref_mask)
                            use_rh, use_rw = use_ref_mask.shape[:2]
                            x_ref = max(0, -best_dx)
                            y_ref = max(0, -best_dy)
                            x_other = max(0, best_dx)
                            y_other = max(0, best_dy)
                            h_canvas = max(y_ref + use_rh, y_other + use_oh)
                            w_canvas = max(x_ref + use_rw, x_other + use_ow)
                            canvas = np.zeros((h_canvas, w_canvas, 3), dtype=np.uint8)
                            canvas[y_ref:y_ref + use_rh, x_ref:x_ref + use_rw, 1] = use_ref_mask
                            canvas[y_other:y_other + use_oh, x_other:x_other + use_ow, 2] = np.maximum(
                                canvas[y_other:y_other + use_oh, x_other:x_other + use_ow, 2], use_other_mask
                            )
                            overlap_zone = (canvas[:, :, 1] > 0) & (canvas[:, :, 2] > 0)
                            canvas[:, :, 0][overlap_zone] = 0
                            canvas[:, :, 1][overlap_zone] = 255
                            canvas[:, :, 2][overlap_zone] = 255
                            cv2.imwrite(str(debug_output_dir / f"overlay_{other_basename}.png"), canvas)
                            print(f"   ✓ [overlay_debug] Salvat: {debug_output_dir}/ (trimmed_{other_basename}.png, overlay_{other_basename}.png)")
                        except Exception as e:
                            print(f"   ⚠ [overlay_debug] Nu am putut salva imagini: {e}")
                    continue
            dx_edges, dy_edges = _align_rects_by_edges(
                ref_main_sec, other_main_sec,
                ref_w, ref_h, other_w, other_h,
            )
            out[p] = (dx_edges, dy_edges)
            if debug_output_dir is not None and ref_mask is not None and other_mask is not None:
                try:
                    debug_output_dir.mkdir(parents=True, exist_ok=True)
                    other_basename = Path(p).stem
                    if not (debug_output_dir / "ref_floor.png").exists():
                        cv2.imwrite(str(debug_output_dir / "ref_floor.png"), ref_mask)
                    cv2.imwrite(str(debug_output_dir / f"other_{other_basename}.png"), other_mask)
                    best_dx, best_dy = dx_edges, dy_edges
                    x_ref = max(0, -best_dx)
                    y_ref = max(0, -best_dy)
                    x_other = max(0, best_dx)
                    y_other = max(0, best_dy)
                    h_canvas = max(y_ref + ref_h, y_other + other_h)
                    w_canvas = max(x_ref + ref_w, x_other + other_w)
                    canvas = np.zeros((h_canvas, w_canvas, 3), dtype=np.uint8)
                    canvas[y_ref:y_ref + ref_h, x_ref:x_ref + ref_w, 1] = ref_mask
                    canvas[y_other:y_other + other_h, x_other:x_other + other_w, 2] = np.maximum(
                        canvas[y_other:y_other + other_h, x_other:x_other + other_w, 2], other_mask
                    )
                    overlap_zone = (canvas[:, :, 1] > 0) & (canvas[:, :, 2] > 0)
                    canvas[:, :, 0][overlap_zone] = 0
                    canvas[:, :, 1][overlap_zone] = 255
                    canvas[:, :, 2][overlap_zone] = 255
                    cv2.imwrite(str(debug_output_dir / f"overlay_{other_basename}.png"), canvas)
                    print(f"   ✓ [overlay_debug] Salvat (aliniere muchii): {debug_output_dir}/ overlay_{other_basename}.png")
                except Exception as e:
                    print(f"   ⚠ [overlay_debug] Nu am putut salva imagini: {e}")
            continue

        # 2) Fallback: brute-force pe măști full (comportament original)
        use_brute_force = (
            ref_mask is not None
            and other_mask is not None
            and ref_mask.any()
            and other_mask.any()
        )
        if use_brute_force:
            step = int(max(1, min(ref_w, ref_h, other_w, other_h) // 50))
            mx_ref, my_ref = _mask_center(ref_mask)
            mx_other, my_other = _mask_center(other_mask)
            dx0 = int(round(mx_ref - mx_other))
            dy0 = int(round(my_ref - my_other))
            range_x = int(min(260, max(60, (min(ref_w, other_w) + 1) // 2)))
            range_y = int(min(260, max(60, (min(ref_h, other_h) + 1) // 2)))
            best_overlap = -1
            best_dx, best_dy = dx0, dy0
            for dx in range(dx0 - range_x, dx0 + range_x + 1, step):
                for dy in range(dy0 - range_y, dy0 + range_y + 1, step):
                    overlap = _overlap_area_at_offset(
                        ref_mask, ref_w, ref_h,
                        other_mask, other_w, other_h,
                        dx, dy,
                    )
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_dx, best_dy = dx, dy
            if step > 1:
                for dx in range(best_dx - step, best_dx + step + 1):
                    for dy in range(best_dy - step, best_dy + step + 1):
                        overlap = _overlap_area_at_offset(
                            ref_mask, ref_w, ref_h,
                            other_mask, other_w, other_h,
                            dx, dy,
                        )
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_dx, best_dy = dx, dy
            out[p] = (best_dx, best_dy)
        else:
            found = False
            for s in rr.get("sections") or []:
                if _sec_area(s) <= 0:
                    continue
                for s_ref in rr_ref.get("sections") or []:
                    if _sec_area(s_ref) <= 0:
                        continue
                    if min(_sec_area(s), _sec_area(s_ref)) / max(_sec_area(s), _sec_area(s_ref)) >= 0.9:
                        cxi, cyi = _sec_center(s)
                        out[p] = (int(round(cx_ref - cxi)), int(round(cy_ref - cyi)))
                        found = True
                        break
                if found:
                    break
            if not found:
                poly = polys[i] if i < len(polys) else None
                if poly is None or getattr(poly, "is_empty", True):
                    out[p] = (0, 0)
                else:
                    c = poly.centroid
                    out[p] = (int(round(cx_ref - c.x)), int(round(cy_ref - c.y)))
    return out


def detect_floors_from_folder(folder_path: str) -> Tuple[List[str], Optional[str]]:
    folder = Path(folder_path)
    if not folder.is_dir():
        return [], None
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    paths = [str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not paths:
        return [], None
    if len(paths) == 1:
        return paths, paths[0]
    # Păstrăm ordinea engine: floor_00, floor_01, ... (beci = 0 la baza 3D)
    import re
    def _floor_index(path: str) -> int:
        stem = Path(path).stem.lower()
        m = re.match(r"floor_?(\d+)", stem)
        return int(m.group(1)) if m else 999
    matching = [p for p in paths if _floor_index(p) != 999]
    if matching:
        matching.sort(key=_floor_index)
        return matching, matching[0]
    sizes = [(p, cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in paths]
    sizes = [(p, img.shape[0] * img.shape[1] if img is not None else 0) for p, img in sizes]
    sizes.sort(key=lambda x: x[1])
    return [x[0] for x in sizes], sizes[0][0]


def _sections_from_edited_json(
    edited_json_path: Path,
    num_floors: int,
) -> Dict[int, list]:
    """
    Convert roof_rectangles_edited.json (editor polygons) into roof sections for roof_types_workflow.
    Returns mapping: floor_idx -> list[section].
    """
    out: Dict[int, list] = {}
    try:
        raw = json.loads(edited_json_path.read_text(encoding="utf-8"))
    except Exception:
        return out
    if not isinstance(raw, dict):
        return out

    for k, rects in raw.items():
        try:
            floor_idx = int(k)
        except Exception:
            continue
        if floor_idx < 0 or floor_idx >= num_floors:
            continue
        if not isinstance(rects, list):
            continue
        sections = []
        for r in rects:
            pts = r.get("points") if isinstance(r, dict) else None
            if not isinstance(pts, list) or len(pts) < 3:
                continue
            arr = []
            for p in pts:
                if isinstance(p, list) and len(p) >= 2:
                    arr.append([float(p[0]), float(p[1])])
            if len(arr) < 3:
                continue
            np_pts = np.array(arr, dtype=np.float32)
            rect = cv2.minAreaRect(np_pts)
            box = cv2.boxPoints(rect).tolist()
            (cx, cy), (rw, rh), _ang = rect
            if rw >= rh:
                ridge = [(float(cx - rw * 0.5), float(cy)), (float(cx + rw * 0.5), float(cy))]
                orientation = "horizontal"
            else:
                ridge = [(float(cx), float(cy - rh * 0.5)), (float(cx), float(cy + rh * 0.5))]
                orientation = "vertical"
            sections.append(
                {
                    "bounding_rect": [(float(p[0]), float(p[1])) for p in box],
                    "ridge_line": ridge,
                    "ridge_orientation": orientation,
                    "is_main": False,
                    "roofAngleDeg": r.get("roofAngleDeg") if isinstance(r, dict) else None,
                    "roofType": r.get("roofType") if isinstance(r, dict) else None,
                    "roomName": r.get("roomName") if isinstance(r, dict) else None,
                }
            )
        out[floor_idx] = sections
    return out


def _generate_per_rectangle_3d_unfold(
    output_path: Path,
    floor_idx: int,
    floor_path: str,
    sections: list,
    roof_angle_deg: float,
) -> None:
    """
    Generate per-rectangle 3D unfold outputs:
      roof_3d/rectangles_3d/floor_X/rectangle_SY/unfold_roof/*.png
      roof_3d/rectangles_3d/floor_X/rectangle_SY/unfold_overhang/*.png
    """
    rect3d_root = output_path / "rectangles_3d" / f"floor_{floor_idx}"
    rect3d_root.mkdir(parents=True, exist_ok=True)
    for rect_idx, sec in enumerate(sections or []):
        if not isinstance(sec, dict):
            continue
        sec_list = [sec]
        rect_dir = rect3d_root / f"rectangle_S{rect_idx}"
        rect_dir.mkdir(parents=True, exist_ok=True)
        try:
            generate_roof_type_outputs(
                floor_path,
                {"sections": sec_list},
                sec_list,
                rect_dir,
                roof_angle_deg=float(sec.get("roofAngleDeg") or roof_angle_deg),
                wall_height=300.0,
                upper_floor_sections=None,
            )
        except Exception as e:
            print(f"   ⚠ rectangles_3d/floor_{floor_idx}/rectangle_S{rect_idx}: {e}")

def run_clean_workflow(
    wall_mask_path: str,
    output_dir: str = "output",
    rectangles_only: bool = False,
    edited_json_path: Optional[str] = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Încarcă unghiuri per etaj din floor_roof_angles.json (extrase de Gemini din side_view)
    floor_roof_angles: dict = {}
    fra_path = output_path / "floor_roof_angles.json"
    if fra_path.exists():
        try:
            data = json.loads(fra_path.read_text(encoding="utf-8"))
            floor_roof_angles = {int(k): float(v) for k, v in (data or {}).items()}
        except Exception:
            pass

    input_path = Path(wall_mask_path)
    all_floor_paths: List[str] = []

    if input_path.is_dir():
        all_floor_paths, _ = detect_floors_from_folder(wall_mask_path)
        if not all_floor_paths:
            print("⚠ Nu s-au găsit imagini în folder!")
            return
        print(f"📁 {len(all_floor_paths)} etaje detectate")
    else:
        if not input_path.exists():
            print(f"⚠ Fișier nu există: {wall_mask_path}")
            return
        all_floor_paths = [wall_mask_path]

    # Încarcă unghiuri per etaj din floor_roof_angles.json (din Gemini/side_view)
    floor_roof_angles: dict[int, float] = {}
    fra_path = output_path / "floor_roof_angles.json"
    if fra_path.exists():
        try:
            fra_data = json.loads(fra_path.read_text(encoding="utf-8"))
            floor_roof_angles = {int(k): float(v) for k, v in (fra_data or {}).items()}
        except Exception:
            pass
    default_angle = 30.0

    floor_roof_results: List[dict] = []
    for i, fp in enumerate(all_floor_paths):
        angle = floor_roof_angles.get(i, default_angle)
        # Validation allows roof_angle in [15, 60]; use clamped value for API, keep angle for footprint logic
        api_angle = angle if 15 <= angle <= 60 else default_angle
        try:
            res = calculate_roof_from_walls(fp, roof_angle=api_angle, overhang_px=2.0)
            floor_roof_results.append(res)
        except Exception as e:
            print(f"⚠ {Path(fp).name}: {e}", flush=True)
            floor_roof_results = []
            break

    if len(floor_roof_results) != len(all_floor_paths):
        floor_roof_results = []
        for i, fp in enumerate(all_floor_paths):
            angle = floor_roof_angles.get(i, default_angle)
            api_angle = angle if 15 <= angle <= 60 else default_angle
            try:
                floor_roof_results.append(calculate_roof_from_walls(fp, roof_angle=api_angle, overhang_px=2.0))
            except Exception:
                floor_roof_results.append({"sections": []})

    # La unghi 0° (acoperiș plat), calculate_roof_from_walls poate returna sections goale → generăm 1 dreptunghi footprint
    for i, fp in enumerate(all_floor_paths):
        if i >= len(floor_roof_results):
            continue
        angle = floor_roof_angles.get(i, default_angle)
        res = floor_roof_results[i]
        sections = res.get("sections") or []
        if angle == 0.0 and not sections:
            footprint = _footprint_section_from_wall_mask(str(fp))
            if footprint:
                floor_roof_results[i] = {**res, "sections": [footprint]}

    floors_ordered = _ordered_floor_polygons(all_floor_paths, None, floor_roof_results)
    if not floors_ordered:
        floors_ordered = [(p, None) for p in all_floor_paths]

    path_to_result = {all_floor_paths[i]: floor_roof_results[i] for i in range(len(all_floor_paths))}

    # Compară dreptunghiuri între etaje: suprafață în pixeli, marjă ±10%
    def _rect_area(sec):
        br = sec.get("bounding_rect") or []
        if len(br) < 3:
            return 0.0
        xs = [float(p[0]) for p in br]
        ys = [float(p[1]) for p in br]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    def _areas_match(a: float, b: float) -> bool:
        if a <= 0 or b <= 0:
            return False
        return 0.9 * b <= a <= 1.1 * b or 0.9 * a <= b <= 1.1 * a

    basement_floor_index: Optional[int] = None
    bfi_path = output_path / "basement_floor_index.json"
    if bfi_path.exists():
        try:
            bfi_data = json.loads(bfi_path.read_text(encoding="utf-8"))
            raw_bfi = bfi_data.get("basement_floor_index")
            if raw_bfi is not None:
                basement_floor_index = int(raw_bfi)
        except Exception:
            basement_floor_index = None

    floors_with_rects: List[Tuple[int, str, list]] = []
    for floor_idx, (floor_path, _) in enumerate(floors_ordered):
        # Beciul nu are acoperiș: nu extragem dreptunghiuri de acoperiș pentru acest etaj.
        if basement_floor_index is not None and floor_idx == basement_floor_index:
            continue
        res = path_to_result.get(floor_path)
        if res is None:
            continue
        sections = res.get("sections") or []
        if not sections:
            continue
        # Group by component_index: merge overlapping rects only within same component (L vs big rect stay separate)
        by_comp: dict = defaultdict(list)
        for s in sections:
            cid = s.get("component_index", 0)
            by_comp[cid].append(s)
        merged = []
        for _cid in sorted(by_comp.keys()):
            merged.extend(remove_overlapping_rectangles(by_comp[_cid], iou_threshold=0.3))
        # Drop very small sections (e.g. meaningless S2 from decomposition tail)
        total_area = sum(_rect_area(s) for s in merged)
        MIN_SECTION_FRAC = 0.02
        remaining = [s for s in merged if total_area <= 0 or _rect_area(s) >= MIN_SECTION_FRAC * total_area]
        for s in remaining:
            s.pop("component_index", None)
        if not remaining:
            continue
        floors_with_rects.append((floor_idx, floor_path, remaining))

    # Keep rectangles per-floor as detected (except basement filter above).
    # Cross-floor area matching/dedup can incorrectly drop valid sections on non-flat levels.
    floors_to_output: List[Tuple[int, str, list]] = sorted(floors_with_rects, key=lambda t: t[0])

    # Pentru foldere rectangles/ și roof_types/: procesăm TOATE etajele (inclusiv 0_w fără secțiuni)
    # Asociem remaining STRICT după path (nu după poziție), apoi iterăm all_floor_paths ca output_idx să fie mereu 0,1,2...
    def _norm_path(p: str) -> str:
        return str(Path(p).resolve())
    path_to_remaining: dict = {_norm_path(fp): rem for _fi, fp, rem in floors_to_output}
    if edited_json_path:
        edited_p = Path(edited_json_path)
        if edited_p.exists():
            edited_sections = _sections_from_edited_json(edited_p, len(all_floor_paths))
            if edited_sections:
                for output_idx, floor_path in enumerate(all_floor_paths):
                    path_to_remaining[_norm_path(floor_path)] = edited_sections.get(output_idx, [])
                print(f"   ✓ Folosesc secțiuni editate din: {edited_p}")
    # Ordinea pentru overlay/offsets trebuie să fie all_floor_paths ca "0"/"1" să corespundă floor_0/floor_1
    ordered_paths = list(all_floor_paths)
    roof_results_ordered = []
    for p in ordered_paths:
        base_res = path_to_result.get(p, {"sections": []})
        roof_results_ordered.append({**base_res, "sections": path_to_remaining.get(_norm_path(p), [])})
    floors_meta = None
    if input_path.is_dir():
        meta_path = input_path / "floors_meta.json"
        if meta_path.exists():
            try:
                floors_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
    offsets_by_path = _compute_overlay_offsets(
        ordered_paths, roof_results_ordered, floors_meta,
        debug_output_dir=output_path / "overlay_debug",
    ) if len(ordered_paths) >= 2 else {}

    # Pairwise top-down:
    # pentru fiecare pereche vecină de etaje, aliniem BRUTE FORCE doar cel mai mare
    # dreptunghi de sus cu cel mai mare dreptunghi de jos (fără rotații),
    # apoi eliminăm dreptunghiul de la etajul inferior.
    # Cumulativ: ce se șterge la un pas rămâne șters în pașii următori.
    def _sec_bbox(sec):
        br = sec.get("bounding_rect") or []
        if len(br) < 3:
            return None
        xs = [float(p[0]) for p in br]
        ys = [float(p[1]) for p in br]
        return (min(xs), min(ys), max(xs), max(ys))

    def _section_mask(sec, w: int, h: int):
        mask = np.zeros((h, w), dtype=np.uint8)
        bb = _sec_bbox(sec)
        if bb is None:
            return mask
        x1, y1, x2, y2 = bb
        xi1 = max(0, min(w - 1, int(round(x1))))
        yi1 = max(0, min(h - 1, int(round(y1))))
        xi2 = max(0, min(w - 1, int(round(x2))))
        yi2 = max(0, min(h - 1, int(round(y2))))
        if xi2 <= xi1 or yi2 <= yi1:
            return mask
        cv2.rectangle(mask, (xi1, yi1), (xi2, yi2), 255, thickness=-1)
        return mask

    def _largest_section_idx(sections):
        best_idx = None
        best_area = -1.0
        for i, s in enumerate(sections or []):
            bb = _sec_bbox(s)
            if bb is None:
                continue
            area = max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1])
            if area > best_area:
                best_area = area
                best_idx = i
        return best_idx

    # Cache dimensiuni per etaj
    dims_by_path: dict[str, tuple[int, int]] = {}
    for fp in all_floor_paths:
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            dims_by_path[fp] = (0, 0)
        else:
            dims_by_path[fp] = (int(img.shape[1]), int(img.shape[0]))

    for upper_idx in range(0, len(all_floor_paths) - 1):
        lower_idx = upper_idx + 1
        up_path = all_floor_paths[upper_idx]
        lo_path = all_floor_paths[lower_idx]
        up_key = _norm_path(up_path)
        lo_key = _norm_path(lo_path)
        up_sections = path_to_remaining.get(up_key, [])
        lo_sections = path_to_remaining.get(lo_key, [])
        if not up_sections or not lo_sections:
            continue

        up_w, up_h = dims_by_path.get(up_path, (0, 0))
        lo_w, lo_h = dims_by_path.get(lo_path, (0, 0))
        if up_w <= 0 or up_h <= 0 or lo_w <= 0 or lo_h <= 0:
            continue

        up_largest_idx = _largest_section_idx(up_sections)
        lo_largest_idx = _largest_section_idx(lo_sections)
        if up_largest_idx is None or lo_largest_idx is None:
            continue
        up_mask = _section_mask(up_sections[up_largest_idx], up_w, up_h)
        lo_mask = _section_mask(lo_sections[lo_largest_idx], lo_w, lo_h)
        if not up_mask.any() or not lo_mask.any():
            continue

        # Seed din offsets globale existente; apoi brute-force local pentru pereche.
        up_off = offsets_by_path.get(up_path, (0.0, 0.0))
        lo_off = offsets_by_path.get(lo_path, (0.0, 0.0))
        dx0 = int(round(up_off[0] - lo_off[0]))
        dy0 = int(round(up_off[1] - lo_off[1]))
        step = max(1, min(up_w, up_h, lo_w, lo_h) // 90)
        search = max(20, min(220, int(max(up_w, up_h, lo_w, lo_h) * 0.12)))
        best_overlap = -1
        best_dx, best_dy = dx0, dy0
        for dx in range(dx0 - search, dx0 + search + 1, step):
            for dy in range(dy0 - search, dy0 + search + 1, step):
                ov = _overlap_area_at_offset(up_mask, up_w, up_h, lo_mask, lo_w, lo_h, dx, dy)
                if ov > best_overlap:
                    best_overlap = ov
                    best_dx, best_dy = dx, dy
        if step > 1:
            for dx in range(best_dx - step, best_dx + step + 1):
                for dy in range(best_dy - step, best_dy + step + 1):
                    ov = _overlap_area_at_offset(up_mask, up_w, up_h, lo_mask, lo_w, lo_h, dx, dy)
                    if ov > best_overlap:
                        best_overlap = ov
                        best_dx, best_dy = dx, dy

        # După aliniere (fără rotații), eliminăm dreptunghiul inferior folosit la comparație.
        lower_filtered = [s for i, s in enumerate(lo_sections) if i != lo_largest_idx]
        path_to_remaining[lo_key] = lower_filtered

    rectangles_dir = output_path / "rectangles"
    roof_types_dir = output_path / "roof_types"
    roof_types_dir.mkdir(parents=True, exist_ok=True)
    overlay_offsets = {
        str(i): {"dx": float(dx), "dy": float(dy)}
        for i, fp in enumerate(ordered_paths)
        for (dx, dy) in [offsets_by_path.get(fp, (0, 0))]
    }
    (roof_types_dir / "overlay_offsets.json").write_text(
        json.dumps(overlay_offsets, indent=0), encoding="utf-8"
    )
    # Ordine de scriere: all_floor_paths (primul plan = floor_0, al doilea = floor_1)
    all_floors_with_remaining: List[Tuple[int, str, list]] = []
    for output_idx, floor_path in enumerate(all_floor_paths):
        remaining = path_to_remaining.get(_norm_path(floor_path), [])
        all_floors_with_remaining.append((output_idx, floor_path, remaining))
    floors_info = [{"floor_idx": output_idx, "path": str(Path(p).resolve())} for output_idx, p, _ in all_floors_with_remaining]
    (roof_types_dir / "floors_info.json").write_text(
        json.dumps(floors_info, indent=0), encoding="utf-8"
    )

    for output_idx, floor_path, remaining in all_floors_with_remaining:
        res = path_to_result.get(floor_path)
        if res is None:
            continue

        filtered_result = {**res, "sections": remaining}

        rect_floor_dir = rectangles_dir / f"floor_{output_idx}"
        rect_floor_dir.mkdir(parents=True, exist_ok=True)
        if remaining:
            try:
                visualize_individual_rectangles(
                    floor_path,
                    filtered_result,
                    output_dir=str(rect_floor_dir),
                )
                print(f"   ✓ rectangles/floor_{output_idx}/: {len(remaining)} dreptunghiuri")
            except Exception as e:
                print(f"   ⚠ rectangles/floor_{output_idx}: {e}")
        else:
            print(f"   ✓ rectangles/floor_{output_idx}/: 0 dreptunghiuri (acoperiș plat / 0_w)")

        if rectangles_only:
            continue
        # Secțiuni etaj superior, traduse în coordonatele etajului curent (doar etaje cu secțiuni)
        upper_sections: list = []
        ox_cur, oy_cur = offsets_by_path.get(floor_path, (0, 0))
        for up_output_idx, up_path, up_remaining in all_floors_with_remaining:
            if up_output_idx <= output_idx or not up_remaining:
                continue
            ox_up, oy_up = offsets_by_path.get(up_path, (0, 0))
            dx = float(ox_up - ox_cur)
            dy = float(oy_up - oy_cur)
            upper_sections.extend(_translate_sections(up_remaining, dx, dy))

        roof_floor_dir = roof_types_dir / f"floor_{output_idx}"
        roof_floor_dir.mkdir(parents=True, exist_ok=True)
        if output_idx == basement_floor_index:
            print(f"   ○ roof_types/floor_{output_idx}/: beci – fără acoperiș")
        else:
            roof_angle_floor = floor_roof_angles.get(output_idx, default_angle)
            try:
                generate_roof_type_outputs(
                    floor_path,
                    filtered_result,
                    remaining,
                    roof_floor_dir,
                    roof_angle_deg=roof_angle_floor,
                    wall_height=300.0,
                    upper_floor_sections=upper_sections if upper_sections else None,
                )
                print(f"   ✓ roof_types/floor_{output_idx}/: 0_w, 1_w, 2_w, 4_w, 4.5_w (lines.png, faces.png, frame.html, unfold_roof/, unfold_overhang/)")
                # Per rectangle 3D renders (single-section run) used for rectangle-level quantities.
                _generate_per_rectangle_3d_unfold(
                    output_path=output_path,
                    floor_idx=output_idx,
                    floor_path=floor_path,
                    sections=remaining,
                    roof_angle_deg=roof_angle_floor,
                )
            except Exception as e:
                print(f"   ⚠ roof_types/floor_{output_idx}: {e}")

    if rectangles_only:
        print("\n   [rectangles-only] Generare dreptunghiuri finalizată.")
        return

    for rt in ("0_w", "1_w", "2_w", "4_w", "4.5_w"):
        try:
            generate_entire_frame_html(
                roof_types_dir,
                output_path,
                wall_height=300.0,
                roof_type=rt,
            )
            print(f"   ✓ entire/{rt}/frame.html, filled.html, roof_types/floor_X/{rt}/unfold_roof/, unfold_overhang/")
        except Exception as e:
            print(f"   ⚠ entire/{rt}/: {e}")

    floor_roof_types: Optional[dict] = None
    frt_path = output_path / "floor_roof_types.json"
    if frt_path.exists():
        try:
            data = json.loads(frt_path.read_text(encoding="utf-8"))
            # Păstrăm None pentru etajul beci (fără acoperiș)
            floor_roof_types = {int(k): (v if v is not None else None) for k, v in (data or {}).items()}
        except Exception:
            pass
    if floor_roof_types:
        try:
            generate_entire_frame_html(
                roof_types_dir,
                output_path,
                wall_height=300.0,
                floor_roof_types=floor_roof_types,
            )
            print(f"   ✓ entire/mixed/frame.html, filled.html (tipuri per etaj)")
            mixed_dir = output_path / "entire" / "mixed"
            if mixed_dir.exists():
                try:
                    populate_mixed_unfold_and_metrics(
                        mixed_dir, roof_types_dir, floor_roof_types, output_path
                    )
                    print(f"   ✓ entire/mixed/unfold_roof/, unfold_overhang/ + roof_metrics.json")
                except Exception as e:
                    print(f"   ⚠ entire/mixed/unfold_roof+unfold_overhang+metrics: {e}")
        except Exception as e:
            print(f"   ⚠ entire/mixed/: {e}")

    print("\n✅ Workflow curat finalizat.")
    print(f"   - {rectangles_dir}/floor_X/ - măști dreptunghiuri")
    print(f"   - {roof_types_dir}/floor_X/{{0_w,1_w,2_w,4_w,4.5_w}}/ - lines.png, faces.png, frame.html")
    print(f"   - {output_path}/entire/{{0_w,1_w,2_w,4_w,4.5_w}}/ - frame.html, filled.html")


if __name__ == "__main__":
    edited_json_path = None
    cli_args = []
    for a in sys.argv[1:]:
        if a.startswith("--edited-json="):
            edited_json_path = a.split("=", 1)[1].strip() or None
            continue
        cli_args.append(a)
    args = [a for a in cli_args if a != "--rectangles-only"]
    rectangles_only = "--rectangles-only" in sys.argv
    if len(args) < 1:
        print("Usage: python clean_workflow.py [--rectangles-only] [--edited-json=/path/to/roof_rectangles_edited.json] <wall_mask_path_or_folder> [output_dir]")
        sys.exit(1)
    path = args[0]
    out = args[1] if len(args) > 1 else "output"
    run_clean_workflow(path, out, rectangles_only=rectangles_only, edited_json_path=edited_json_path)
