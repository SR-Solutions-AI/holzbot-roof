#!/usr/bin/env python3
"""
Generează o imagine de debug: arată dreptunghiurile (bbox) extrase de pe fiecare etaj
și cum le suprapunem. Vezi exact ce dreptunghiuri comparăm și unde ajung pe canvas.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from roof_calc.flood_fill import flood_fill_interior, get_house_shape_mask
from roof_calc.geometry import extract_polygon

SAME_RECT_TOLERANCE = 2


def get_bbox_from_roof_result(roof_data: dict) -> Optional[Tuple[int, int, int, int]]:
    """Bbox (union) din sections (decomposiția rectangle_Sx)."""
    sections = roof_data.get('sections') or []
    if not sections:
        return None
    all_x, all_y = [], []
    for sec in sections:
        rect = sec.get('bounding_rect', [])
        if len(rect) >= 3:
            for p in rect:
                all_x.append(p[0])
                all_y.append(p[1])
    if not all_x or not all_y:
        return None
    return (int(min(all_x)), int(min(all_y)), int(max(all_x)), int(max(all_y)))


def get_largest_bbox_from_floor(floor_path: str) -> Optional[Tuple[int, int, int, int]]:
    img = cv2.imread(floor_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    filled = flood_fill_interior(img)
    house_mask = get_house_shape_mask(filled)
    polygon = extract_polygon(house_mask)
    if polygon is None or polygon.is_empty:
        return None
    minx, miny, maxx, maxy = polygon.bounds
    return (int(minx), int(miny), int(maxx), int(maxy))


def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    minx, miny, maxx, maxy = bbox
    return ((minx + maxx) / 2.0, (miny + maxy) / 2.0)


def _bbox_center_int(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Centre întreg (pixel-perfect)."""
    minx, miny, maxx, maxy = bbox
    return ((minx + maxx) // 2, (miny + maxy) // 2)


def _bbox_same_size(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int], tol: int = SAME_RECT_TOLERANCE) -> bool:
    w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
    w2, h2 = b2[2] - b2[0], b2[3] - b2[1]
    return abs(w1 - w2) <= tol and abs(h1 - h2) <= tol


def generate_overlay_debug_image(
    floor_paths: List[str],
    output_path: str,
    padding: int = 50,
    roof_results: Optional[List[dict]] = None,
) -> bool:
    """
    Generează o imagine de debug:
    - Rând 1: fiecare etaj cu dreptunghiul (bbox) desenat și etichetat (w×h, centru).
    - Rând 2: suprapunerea finală cu fiecare bbox desenat pe canvas (să vezi cum se aliniază).
    """
    if not floor_paths:
        print("   ⚠ Nu s-au găsit etaje!")
        return False

    bboxes: List[Tuple[int, int, int, int]] = []
    floor_images: List[np.ndarray] = []
    used_paths: List[str] = []

    use_rectangles = roof_results is not None and len(roof_results) == len(floor_paths)
    for idx, floor_path in enumerate(floor_paths):
        if use_rectangles:
            bbox = get_bbox_from_roof_result(roof_results[idx])
        else:
            bbox = get_largest_bbox_from_floor(floor_path)
        if bbox is None:
            print(f"   ⚠ Nu s-a putut extrage bbox din {Path(floor_path).name}")
            continue
        img = cv2.imread(floor_path, cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(floor_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img is None:
            continue
        bboxes.append(bbox)
        floor_images.append(img)
        used_paths.append(floor_path)

    if not bboxes:
        print("   ⚠ Nu s-au găsit etaje valide!")
        return False

    colors = [
        (0, 0, 255),
        (0, 180, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]
    colors = colors[:len(bboxes)]

    max_h = max(img.shape[0] for img in floor_images)
    max_w = max(img.shape[1] for img in floor_images)
    cell_w = max_w + 40
    cell_h = max_h + 120
    row1_w = len(floor_images) * cell_w
    row1_h = cell_h

    row1 = np.ones((row1_h, row1_w, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, (path, img, bbox, color) in enumerate(zip(used_paths, floor_images, bboxes, colors)):
        minx, miny, maxx, maxy = bbox
        w = maxx - minx
        h = maxy - miny
        cx, cy = _bbox_center(bbox)

        x0 = idx * cell_w
        rh, rw = img.shape[:2]
        row1[20:20+rh, x0+20:x0+20+rw] = img
        cv2.rectangle(row1, (x0+20+minx, 20+miny), (x0+20+maxx, 20+maxy), color, 3)
        cv2.circle(row1, (x0+20+int(cx), 20+int(cy)), 8, color, -1)
        cv2.circle(row1, (x0+20+int(cx), 20+int(cy)), 10, (0, 0, 0), 2)

        y_text = 20 + rh + 22
        cv2.putText(row1, f"Etaj {idx+1}: {Path(path).name[:25]}...", (x0+20, y_text), font, 0.45, (0,0,0), 1)
        cv2.putText(row1, f"bbox=({minx},{miny})-({maxx},{maxy})", (x0+20, y_text+18), font, 0.4, (0,0,0), 1)
        cv2.putText(row1, f"w={w} h={h}  centru=({int(cx)},{int(cy)})", (x0+20, y_text+36), font, 0.4, (0,0,0), 1)
        same_with = [j+1 for j in range(len(bboxes)) if j != idx and _bbox_same_size(bboxes[j], bbox)]
        if same_with:
            cv2.putText(row1, f"LA FEL ca etajele: {same_with}", (x0+20, y_text+54), font, 0.45, (0, 100, 0), 2)

    cv2.putText(row1, "Dreptunghiuri extrase (bbox) pe fiecare etaj", (20, 16), font, 0.7, (0,0,0), 2)

    bbox_widths = [b[2]-b[0] for b in bboxes]
    bbox_heights = [b[3]-b[1] for b in bboxes]
    cw = max(bbox_widths) + 2 * padding
    ch = max(bbox_heights) + 2 * padding
    ref_cx = int(round(cw / 2))
    ref_cy = int(round(ch / 2))

    canvas = np.ones((ch, cw, 3), dtype=np.uint8) * 255
    cv2.line(canvas, (ref_cx-25, ref_cy), (ref_cx+25, ref_cy), (100, 100, 100), 2)
    cv2.line(canvas, (ref_cx, ref_cy-25), (ref_cx, ref_cy+25), (100, 100, 100), 2)
    cv2.putText(canvas, "ref (centru canvas)", (ref_cx-60, ref_cy-35), font, 0.5, (80, 80, 80), 1)

    for idx, (bbox, img, color) in enumerate(zip(bboxes, floor_images, colors)):
        minx, miny, maxx, maxy = bbox
        bbox_cx, bbox_cy = _bbox_center_int(bbox)
        ox = ref_cx - bbox_cx
        oy = ref_cy - bbox_cy

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        filled = flood_fill_interior(gray)
        house_mask = get_house_shape_mask(filled)
        contours, _ = cv2.findContours(house_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            cnt_shifted = cnt + np.array([[ox, oy]], dtype=np.int32)
            cv2.drawContours(canvas, [cnt_shifted], -1, color, 2)

        rx1, ry1 = ox + minx, oy + miny
        rx2, ry2 = ox + maxx, oy + maxy
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), color, 2)
        cv2.circle(canvas, (ox + bbox_cx, oy + bbox_cy), 6, color, -1)
        cv2.putText(canvas, f"E{idx+1}", (rx1, ry1 - 5), font, 0.5, color, 1)

    scale = 1.0
    if cw > row1_w or ch > 400:
        scale = min(row1_w / cw, 400 / ch, 1.0)
        new_cw, new_ch = int(cw * scale), int(ch * scale)
        canvas = cv2.resize(canvas, (new_cw, new_ch), interpolation=cv2.INTER_AREA)

    h2 = canvas.shape[0]
    w2 = canvas.shape[1]
    row2 = np.ones((h2 + 50, max(row1_w, w2), 3), dtype=np.uint8) * 250
    row2[40:40+h2, 0:w2] = canvas
    cv2.putText(row2, "Suprapunere: fiecare bbox aliniat la același centru (ref). Dacă w,h la fel -> exact unul peste altul.", (10, 22), font, 0.55, (0,0,0), 1)

    gap = 10
    total_w = max(row1_w, row2.shape[1])
    total_h = row1_h + gap + row2.shape[0]
    out = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
    out[0:row1_h, 0:row1_w] = row1
    out[row1_h+gap:row1_h+gap+row2.shape[0], 0:row2.shape[1]] = row2

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out)
    print(f"   ✓ Debug salvat: {Path(output_path).name}")
    return True


def main():
    if len(sys.argv) < 3:
        print("Utilizare: python overlay_floors_2d_debug.py <folder_cu_etaje> <output_path>")
        print("  ex: python overlay_floors_2d_debug.py test2 output/overlay_debug.png")
        sys.exit(1)
    folder_path = sys.argv[1]
    output_path = sys.argv[2]
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"   ⚠ Folder nu există: {folder_path}")
        sys.exit(1)
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    floor_paths = [str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not floor_paths:
        print(f"   ⚠ Nu s-au găsit imagini în {folder_path}")
        sys.exit(1)
    print(f"📁 Folder: {folder_path}, {len(floor_paths)} imagini")
    ok = generate_overlay_debug_image(floor_paths, output_path)
    if not ok:
        sys.exit(1)
    print(f"✅ Gata: {output_path}")


if __name__ == '__main__':
    main()

