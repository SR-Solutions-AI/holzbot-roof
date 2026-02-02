#!/usr/bin/env python3
"""
Suprapune etajele în top view 2D, aliniind cele mai mari dreptunghiuri (bounding box-uri).
Dacă două etaje au dreptunghiuri de aceeași dimensiune, le suprapune astfel încât
acele dreptunghiuri să fie exact unul peste altul (același centru pe canvas).
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np

# Add project root so "roof_calc" is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from roof_calc.flood_fill import flood_fill_interior, get_house_shape_mask
from roof_calc.geometry import extract_polygon

# Toleranță (px) pentru compatibilitate
SAME_RECT_TOLERANCE = 2
# Marjă de eroare (1%) pentru a considera două dreptunghiuri "la fel" după lățime/înălțime
SAME_RECT_PCT = 0.01


def _merge_rectangle_images_from_folder(folder: str) -> Optional[np.ndarray]:
    """
    Încarcă toate rectangle_S*.png dintr-un folder și le combină într-o singură imagine.
    Returnează imaginea BGR sau None dacă nu există fișiere.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        return None
    paths = sorted(
        folder_path.glob("rectangle_S*.png"),
        key=lambda p: int(m.group(1)) if (m := re.search(r"S(\d+)", p.stem)) else 0,
    )
    if not paths:
        return None
    result = None
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if result is None:
            result = np.zeros_like(img)
        # Dacă în același folder există imagini cu dimensiuni diferite, normalizăm prin padding
        # (evităm boolean index mismatch). Originea rămâne (0,0).
        if result.shape[:2] != img.shape[:2]:
            new_h = max(result.shape[0], img.shape[0])
            new_w = max(result.shape[1], img.shape[1])
            new_result = np.zeros((new_h, new_w, 3), dtype=result.dtype)
            new_result[: result.shape[0], : result.shape[1]] = result
            result = new_result
            if img.shape[0] != new_h or img.shape[1] != new_w:
                new_img = np.zeros((new_h, new_w, 3), dtype=img.dtype)
                new_img[: img.shape[0], : img.shape[1]] = img
                img = new_img
        # Overlay: unde img nu e negru, copiază pe result
        mask = (img[:, :, 0] > 0) | (img[:, :, 1] > 0) | (img[:, :, 2] > 0)
        result[mask] = img[mask]
    return result


def get_bbox_from_roof_result(roof_data: dict) -> Optional[Tuple[int, int, int, int]]:
    """
    Extrage bounding box-ul (union) al tuturor dreptunghiurilor din decomposiția acoperișului.
    """
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


def _section_bboxes_with_size(roof_data: dict) -> List[Tuple[Tuple[int, int, int, int], int, int]]:
    """Pentru fiecare secțiune returnează (bbox, width, height)."""
    out = []
    for sec in roof_data.get('sections') or []:
        rect = sec.get('bounding_rect', [])
        if len(rect) < 3:
            continue
        xs = [p[0] for p in rect]
        ys = [p[1] for p in rect]
        minx, maxx = int(min(xs)), int(max(xs))
        miny, maxy = int(min(ys)), int(max(ys))
        w, h = maxx - minx, maxy - miny
        out.append(((minx, miny, maxx, maxy), w, h))
    return out


def _same_size(w1: int, h1: int, w2: int, h2: int, tol: int = SAME_RECT_TOLERANCE) -> bool:
    return abs(w1 - w2) <= tol and abs(h1 - h2) <= tol


def _same_size_pct(
    w1: int, h1: int, w2: int, h2: int, pct: float = SAME_RECT_PCT
) -> bool:
    """True dacă lățimea și înălțimea sunt în marja de eroare pct (ex. 1% = 0.01)."""
    if w1 <= 0 and w2 <= 0 and h1 <= 0 and h2 <= 0:
        return True
    w_ok = abs(w1 - w2) <= pct * max(w1, w2, 1)
    h_ok = abs(h1 - h2) <= pct * max(h1, h2, 1)
    return w_ok and h_ok


def _get_largest_rect_bbox_per_floor(
    roof_results: List[dict],
) -> List[Tuple[int, int, int, int]]:
    """
    Pentru fiecare etaj returnează bbox-ul **celui mai mare** dreptunghi (după arie).
    Suprapunerea aliniază centrele acestor dreptunghiuri → dacă sunt egale, se suprapun perfect.
    """
    result: List[Tuple[int, int, int, int]] = []
    for roof_data in roof_results:
        per_floor = _section_bboxes_with_size(roof_data)
        if not per_floor:
            bbox = get_bbox_from_roof_result(roof_data)
            result.append(bbox)  # poate fi None, dar îl tratăm în caller
            continue
        # Cel mai mare după arie (w * h)
        largest = max(per_floor, key=lambda x: x[1] * x[2])  # (bbox, w, h) -> w*h
        result.append(largest[0])
    return result


def get_largest_bbox_from_floor(floor_path: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Extrage cel mai mare bounding box (dreptunghi) dintr-un etaj.
    
    Returns:
        (minx, miny, maxx, maxy) sau None dacă nu se poate extrage
    """
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
    """Centre (cx, cy) al bbox (minx, miny, maxx, maxy)."""
    minx, miny, maxx, maxy = bbox
    return ((minx + maxx) / 2.0, (miny + maxy) / 2.0)


def _bbox_same_size(
    bbox1: Tuple[int, int, int, int],
    bbox2: Tuple[int, int, int, int],
    tol: int = SAME_RECT_TOLERANCE,
) -> bool:
    """True dacă cele două bbox-uri au aceeași lățime și înălțime (în toleranță)."""
    w1 = bbox1[2] - bbox1[0]
    h1 = bbox1[3] - bbox1[1]
    w2 = bbox2[2] - bbox2[0]
    h2 = bbox2[3] - bbox2[1]
    return abs(w1 - w2) <= tol and abs(h1 - h2) <= tol


def overlay_floors_2d(
    floor_paths: List[str],
    output_path: str,
    alignment: str = 'center',
    background_color: Tuple[int, int, int] = (255, 255, 255),
    floor_colors: Optional[List[Tuple[int, int, int]]] = None,
    roof_results: Optional[List[dict]] = None,
    rectangle_folders: Optional[List[str]] = None,
) -> bool:
    """
    Suprapune etajele în top view 2D, aliniind dreptunghiurile (bbox).
    Dacă roof_results e dat, bbox-ul vine din decomposiția acoperișului (rectangle_Sx) per etaj.
    Dacă rectangle_folders e dat, folosește imaginile rectangle_Sx.png din fiecare folder pentru suprapunere.
    Altfel folosește imaginile de etaj (floor_paths) și cel mai mare bbox din poligon.
    """
    if not floor_paths:
        print("   ⚠ Nu s-au găsit etaje!")
        return False
    
    use_rectangles = roof_results is not None and len(roof_results) == len(floor_paths)
    use_rectangle_images = (
        rectangle_folders is not None
        and len(rectangle_folders) == len(floor_paths)
        and use_rectangles
    )
    
    # Când avem roof_results, aliniem după dreptunghiul cu aceleași dimensiuni (nu union bbox)
    if use_rectangles:
        align_bboxes_per_floor = _get_largest_rect_bbox_per_floor(roof_results)
        # Dacă cele mai mari dreptunghiuri per etaj sunt egale (marjă 1%), se suprapun perfect
        if align_bboxes_per_floor and len(align_bboxes_per_floor) >= 2:
            b0 = align_bboxes_per_floor[0]
            if b0 and all(
                b and _same_size_pct(
                    b0[2] - b0[0], b0[3] - b0[1],
                    b[2] - b[0], b[3] - b[1],
                    SAME_RECT_PCT,
                )
                for b in align_bboxes_per_floor[1:]
            ):
                print("   ✓ Cel mai mare dreptunghi per etaj: egale (≤1%) → suprapunere perfectă")
    else:
        align_bboxes_per_floor = None

    bboxes: List[Tuple[int, int, int, int]] = []
    floor_images: List[np.ndarray] = []
    
    for idx, floor_path in enumerate(floor_paths):
        if use_rectangles and align_bboxes_per_floor and idx < len(align_bboxes_per_floor):
            bbox = align_bboxes_per_floor[idx]
        elif use_rectangles:
            bbox = get_bbox_from_roof_result(roof_results[idx])
        else:
            bbox = get_largest_bbox_from_floor(floor_path)
        if bbox is None:
            print(f"   ⚠ Nu s-a putut extrage bbox din {Path(floor_path).name}")
            continue
        
        if use_rectangle_images:
            img = _merge_rectangle_images_from_folder(rectangle_folders[idx])
            if img is None:
                # Fallback la imaginea etajului
                img = cv2.imread(floor_path, cv2.IMREAD_COLOR)
                if img is None:
                    img = cv2.imread(floor_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.imread(floor_path, cv2.IMREAD_COLOR)
            if img is None:
                img = cv2.imread(floor_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        if img is None:
            print(f"   ⚠ Nu s-a putut încărca {Path(floor_path).name}")
            continue
        
        bboxes.append(bbox)
        floor_images.append(img)
    
    if not bboxes:
        print("   ⚠ Nu s-au găsit etaje valide!")
        return False
    
    print(f"   ✓ {len(bboxes)} etaje găsite")
    
    # Calculează dimensiunile pentru canvas-ul final
    bbox_widths = [bbox[2] - bbox[0] for bbox in bboxes]
    bbox_heights = [bbox[3] - bbox[1] for bbox in bboxes]
    
    max_width = max(bbox_widths)
    max_height = max(bbox_heights)
    padding = 50
    ref_cx = (max_width + 2 * padding) / 2.0
    ref_cy = (max_height + 2 * padding) / 2.0

    draw_positions: List[Tuple[int, int]] = []
    for bbox in bboxes:
        bbox_cx, bbox_cy = _bbox_center(bbox)
        ox = int(round(ref_cx - bbox_cx))
        oy = int(round(ref_cy - bbox_cy))
        draw_positions.append((ox, oy))

    min_ox = min(ox for ox, _ in draw_positions)
    min_oy = min(oy for _, oy in draw_positions)
    max_right = max(ox + img.shape[1] for (ox, _), img in zip(draw_positions, floor_images))
    max_bottom = max(oy + img.shape[0] for (_, oy), img in zip(draw_positions, floor_images))
    canvas_width = max_right - min_ox + 2 * padding
    canvas_height = max_bottom - min_oy + 2 * padding
    shift_x = padding - min_ox
    shift_y = padding - min_oy

    canvas = np.full((canvas_height, canvas_width, 3), background_color, dtype=np.uint8)
    
    # Culori default pentru etaje (BGR)
    if floor_colors is None:
        default_colors = [
            (255, 100, 100),  # Albastru deschis
            (100, 255, 100),  # Verde deschis
            (100, 100, 255),  # Roșu deschis
            (255, 255, 100),  # Cyan
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Galben
        ]
        floor_colors = default_colors[:len(bboxes)]
    
    # Suprapune fiecare etaj: centrele dreptunghiurilor de aliniere la același punct
    for idx, (bbox, img, color) in enumerate(zip(bboxes, floor_images, floor_colors)):
        minx, miny, maxx, maxy = bbox
        ox, oy = draw_positions[idx]
        ox, oy = ox + shift_x, oy + shift_y
        
        if use_rectangle_images:
            h_img, w_img = img.shape[:2]
            y1_dst = max(0, oy)
            x1_dst = max(0, ox)
            y2_dst = min(canvas_height, oy + h_img)
            x2_dst = min(canvas_width, ox + w_img)
            y1_src = y1_dst - oy
            x1_src = x1_dst - ox
            y2_src = y1_src + (y2_dst - y1_dst)
            x2_src = x1_src + (x2_dst - x1_dst)
            if y2_src > y1_src and x2_src > x1_src:
                patch = img[y1_src:y2_src, x1_src:x2_src]
                non_black = (patch[:, :, 0] > 0) | (patch[:, :, 1] > 0) | (patch[:, :, 2] > 0)
                roi = canvas[y1_dst:y2_dst, x1_dst:x2_dst]
                roi[non_black] = cv2.addWeighted(roi[non_black], 0.4, patch[non_black], 0.6, 0)
        else:
            filled = flood_fill_interior(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img)
            house_mask = get_house_shape_mask(filled)
            contours, _ = cv2.findContours(house_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                contour_shifted = largest_contour + np.array([[ox, oy]], dtype=np.int32)
                cv2.drawContours(canvas, [contour_shifted], -1, color, thickness=3)
                mask_overlay = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
                cv2.drawContours(mask_overlay, [contour_shifted], -1, 255, thickness=cv2.FILLED)
                overlay = canvas.copy()
                overlay[mask_overlay > 0] = color
                alpha = 0.3
                canvas = cv2.addWeighted(canvas, 1 - alpha, overlay, alpha, 0)
        
        label = f"Etaj {idx + 1}"
        label_pos = (ox + minx + 10, oy + miny + 30)
        cv2.putText(canvas, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_file), canvas)
    print(f"   ✓ Suprapunere salvată: {output_file.name}")
    return True


def main():
    """Entry point pentru script."""
    if len(sys.argv) < 3:
        print("Utilizare: python overlay_floors_2d.py <folder_cu_etaje> <output_path> [alignment]")
        print("  alignment: 'center' (default) sau 'topleft'")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    output_path = sys.argv[2]
    alignment = sys.argv[3] if len(sys.argv) > 3 else 'center'
    
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"   ⚠ Folder nu există: {folder_path}")
        sys.exit(1)
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    floor_paths = [
        str(p) for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]
    
    if not floor_paths:
        print(f"   ⚠ Nu s-au găsit imagini în {folder_path}")
        sys.exit(1)
    
    print(f"📁 Folder: {folder_path}")
    print(f"   ✓ {len(floor_paths)} imagini găsite")
    
    success = overlay_floors_2d(floor_paths, output_path, alignment=alignment)
    
    if success:
        print(f"\n✅ Suprapunere completă: {output_path}")
    else:
        print(f"\n❌ Eroare la suprapunere")
        sys.exit(1)


if __name__ == '__main__':
    main()

