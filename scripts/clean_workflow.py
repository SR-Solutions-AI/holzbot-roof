#!/usr/bin/env python3
"""
Workflow curat:
- output/rectangles/floor_X/ - măști dreptunghiuri rămase (după eliminare suprapuneri)
- output/roof_types/floor_X/{1_w,2_w,4_w,4.5_w}/ - lines.png, faces.png, frame.html per tip
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple

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
)

import cv2


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


def _compute_overlay_offsets(floor_paths: List[str], roof_results: List[dict]) -> dict:
    """Offsets pentru alinierea etajelor (centroid / dreptunghi comun)."""
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

    polys = [_poly(p) for p in floor_paths]
    best_i = 0
    best_a = 0.0
    for i, poly in enumerate(polys):
        if poly is not None and not poly.is_empty:
            a = float(poly.area)
            if a > best_a:
                best_a = a
                best_i = i
    cx_ref, cy_ref = 0.0, 0.0
    rr_ref = roof_results[best_i] if best_i < len(roof_results) else {}
    for s in rr_ref.get("sections") or []:
        if _sec_area(s) > 0:
            cx_ref, cy_ref = _sec_center(s)
            break
    if cx_ref == 0 and cy_ref == 0 and polys[best_i]:
        c = polys[best_i].centroid
        cx_ref, cy_ref = float(c.x), float(c.y)

    out = {}
    for i, p in enumerate(floor_paths):
        if i == best_i:
            out[p] = (0, 0)
            continue
        rr = roof_results[i] if i < len(roof_results) else {}
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
                out[p] = (int(round(cx_ref - c.x)), int(round(cy_ref - cy.y)))
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
    sizes = [(p, cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in paths]
    sizes = [(p, img.shape[0] * img.shape[1] if img is not None else 0) for p, img in sizes]
    sizes.sort(key=lambda x: x[1])
    return [x[0] for x in sizes], sizes[0][0]


def run_clean_workflow(wall_mask_path: str, output_dir: str = "output") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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

    floor_roof_results: List[dict] = []
    for fp in all_floor_paths:
        try:
            res = calculate_roof_from_walls(fp, roof_angle=30, overhang_px=2.0)
            floor_roof_results.append(res)
        except Exception as e:
            print(f"⚠ {Path(fp).name}: {e}")
            floor_roof_results = []
            break

    if len(floor_roof_results) != len(all_floor_paths):
        floor_roof_results = []
        for fp in all_floor_paths:
            try:
                floor_roof_results.append(calculate_roof_from_walls(fp, roof_angle=30, overhang_px=2.0))
            except Exception:
                floor_roof_results.append({"sections": []})

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

    floors_with_rects: List[Tuple[int, str, list]] = []
    for floor_idx, (floor_path, _) in enumerate(floors_ordered):
        res = path_to_result.get(floor_path)
        if res is None:
            continue
        sections = res.get("sections") or []
        if not sections:
            continue
        remaining = remove_overlapping_rectangles(sections, iou_threshold=0.3)
        if not remaining:
            continue
        floors_with_rects.append((floor_idx, floor_path, remaining))

    # Pentru fiecare etaj: păstrăm doar dreptunghiurile care NU au match (arie ±10%) la un etaj superior
    floors_to_output: List[Tuple[int, str, list]] = []
    for i, (fi, fp, rem) in enumerate(floors_with_rects):
        upper_areas = []
        for j in range(i + 1, len(floors_with_rects)):
            upper_areas.extend([_rect_area(s) for s in floors_with_rects[j][2]])
        filtered = [
            s for s in rem
            if not any(_areas_match(_rect_area(s), ua) for ua in upper_areas)
        ]
        if filtered:
            floors_to_output.append((fi, fp, filtered))

    ordered_paths = [p for p, _ in floors_ordered]
    roof_results_ordered = [path_to_result.get(p, {"sections": []}) for p in ordered_paths]
    offsets_by_path = _compute_overlay_offsets(ordered_paths, roof_results_ordered) if len(ordered_paths) >= 2 else {}

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
    floors_info = [{"floor_idx": i, "path": str(Path(p).resolve())} for i, (p, _) in enumerate(floors_ordered)]
    (roof_types_dir / "floors_info.json").write_text(
        json.dumps(floors_info, indent=0), encoding="utf-8"
    )

    for floor_idx, floor_path, remaining in floors_to_output:
        res = path_to_result.get(floor_path)
        if res is None:
            continue

        filtered_result = {**res, "sections": remaining}

        rect_floor_dir = rectangles_dir / f"floor_{floor_idx}"
        rect_floor_dir.mkdir(parents=True, exist_ok=True)
        try:
            visualize_individual_rectangles(
                floor_path,
                filtered_result,
                output_dir=str(rect_floor_dir),
            )
            print(f"   ✓ rectangles/floor_{floor_idx}/: {len(remaining)} dreptunghiuri")
        except Exception as e:
            print(f"   ⚠ rectangles/floor_{floor_idx}: {e}")

        # Secțiuni etaj superior, traduse în coordonatele etajului curent
        upper_sections: list = []
        ox_cur, oy_cur = offsets_by_path.get(floor_path, (0, 0))
        for up_idx, up_path, up_remaining in floors_to_output:
            if up_idx <= floor_idx:
                continue
            ox_up, oy_up = offsets_by_path.get(up_path, (0, 0))
            dx = float(ox_up - ox_cur)
            dy = float(oy_up - oy_cur)
            upper_sections.extend(_translate_sections(up_remaining, dx, dy))

        roof_floor_dir = roof_types_dir / f"floor_{floor_idx}"
        roof_floor_dir.mkdir(parents=True, exist_ok=True)
        try:
            generate_roof_type_outputs(
                floor_path,
                filtered_result,
                remaining,
                roof_floor_dir,
                roof_angle_deg=30.0,
                wall_height=300.0,
                upper_floor_sections=upper_sections if upper_sections else None,
            )
            print(f"   ✓ roof_types/floor_{floor_idx}/: 1_w, 2_w, 4_w, 4.5_w (lines.png, faces.png, frame.html, unfold/)")
        except Exception as e:
            print(f"   ⚠ roof_types/floor_{floor_idx}: {e}")

    for rt in ("1_w", "2_w", "4_w", "4.5_w"):
        try:
            generate_entire_frame_html(
                roof_types_dir,
                output_path,
                wall_height=300.0,
                roof_type=rt,
            )
            print(f"   ✓ entire/{rt}/frame.html, filled.html, roof_types/floor_X/{rt}/unfold/")
        except Exception as e:
            print(f"   ⚠ entire/{rt}/: {e}")

    floor_roof_types: Optional[dict] = None
    frt_path = output_path / "floor_roof_types.json"
    if frt_path.exists():
        try:
            data = json.loads(frt_path.read_text(encoding="utf-8"))
            floor_roof_types = {int(k): str(v) for k, v in (data or {}).items()}
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
        except Exception as e:
            print(f"   ⚠ entire/mixed/: {e}")

    print("\n✅ Workflow curat finalizat.")
    print(f"   - {rectangles_dir}/floor_X/ - măști dreptunghiuri")
    print(f"   - {roof_types_dir}/floor_X/{{1_w,2_w,4_w,4.5_w}}/ - lines.png, faces.png, frame.html")
    print(f"   - {output_path}/entire/{{1_w,2_w,4_w,4.5_w}}/ - frame.html, filled.html")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_workflow.py <wall_mask_path_or_folder> [output_dir]")
        sys.exit(1)
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "output"
    run_clean_workflow(path, out)
