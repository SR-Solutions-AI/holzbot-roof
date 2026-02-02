"""
Decompoziție poligoane în dreptunghiuri orientate.

Motivație: versiunea restaurată din bytecode producea adesea un singur dreptunghi
pentru forme complexe → secțiuni de acoperiș greșite și randare 3D confuză.

Implementarea de mai jos urmează logica din codul vechi din transcript:
- detectează colțuri concave
- încearcă tăieturi axate (vertical/horizontal) prin colțuri concave
- recursiv, produce dreptunghiuri orientate (min-area bbox) pentru bucăți
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Dict

import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.ops import split

from roof_calc.geometry import classify_shape, find_concave_corners

logger = logging.getLogger(__name__)

MIN_RECTANGLE_AREA_PX = 400  # ~20x20
MAX_RECURSION_DEPTH = 10

# Grid decomposition defaults (inspired from user's script)
GRID_MIN_RECT_SIZE_PX = 50
GRID_OVERLAP_THRESH = 0.85
GRID_STRICT_OVERLAP = 0.90
MAX_GRID_RECTS = 12


def _oriented_bounding_rect(polygon: Polygon) -> Tuple[Polygon, float, float, float]:
    """
    Oriented bounding box (min area). Returns (box_polygon, width, height, angle_deg).
    """
    if polygon.is_empty or not polygon.is_valid or polygon.exterior is None:
        return polygon, 0.0, 0.0, 0.0

    coords = np.array(polygon.exterior.coords[:-1], dtype=np.float64)
    if len(coords) < 3:
        return polygon, 0.0, 0.0, 0.0

    best_area = float("inf")
    best_rect: Polygon | None = None
    best_w = 0.0
    best_h = 0.0
    best_angle = 0.0

    for i in range(len(coords)):
        p0 = coords[i]
        p1 = coords[(i + 1) % len(coords)]
        ang = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        c = float(np.cos(-ang))
        s = float(np.sin(-ang))
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        rcoords = coords @ rot.T
        minx, miny = rcoords.min(axis=0)
        maxx, maxy = rcoords.max(axis=0)
        area = float((maxx - minx) * (maxy - miny))
        if 0 < area < best_area:
            best_area = area
            best_rect = box(float(minx), float(miny), float(maxx), float(maxy))
            best_w = float(maxx - minx)
            best_h = float(maxy - miny)
            best_angle = float(np.degrees(ang))

    if best_rect is None:
        minx, miny, maxx, maxy = polygon.bounds
        best_rect = box(minx, miny, maxx, maxy)
        best_w = float(maxx - minx)
        best_h = float(maxy - miny)
        best_angle = 0.0

    return best_rect, best_w, best_h, best_angle


def _split_once(poly: Polygon) -> List[Polygon]:
    """
    Încearcă o singură tăietură printr-un colț concav (vertical/horizontal).
    Returnează lista de părți (>=2) sau [] dacă nu reușește.
    """
    concave = find_concave_corners(poly)
    if not concave:
        return []
    cx, cy = concave[0]
    minx, miny, maxx, maxy = poly.bounds

    candidates = [
        LineString([(cx, miny - 1), (cx, maxy + 1)]),
        LineString([(minx - 1, cy), (maxx + 1, cy)]),
    ]
    for line in candidates:
        try:
            parts = [p for p in split(poly, line).geoms if isinstance(p, Polygon)]
            parts = [p for p in parts if (not p.is_empty and float(p.area) >= MIN_RECTANGLE_AREA_PX)]
            if len(parts) >= 2:
                return parts
        except Exception:
            continue
    return []


def _partition_recursive(poly: Polygon, out: List[Tuple[Polygon, float]], depth: int) -> None:
    if depth <= 0 or poly.is_empty or float(poly.area) < MIN_RECTANGLE_AREA_PX:
        return

    st = classify_shape(poly)
    if st == "rectangle":
        rect, _w, _h, angle = _oriented_bounding_rect(poly)
        if rect is not None and (not rect.is_empty) and float(rect.area) >= MIN_RECTANGLE_AREA_PX:
            out.append((rect, float(angle)))
        return

    parts = _split_once(poly)
    if not parts:
        rect, _w, _h, angle = _oriented_bounding_rect(poly)
        if rect is not None and (not rect.is_empty) and float(rect.area) >= MIN_RECTANGLE_AREA_PX:
            out.append((rect, float(angle)))
        return

    # recurse on each part
    for p in parts:
        _partition_recursive(p, out, depth - 1)


def _check_mask_overlap(rect: Polygon, filled_mask: np.ndarray) -> float:
    minx, miny, maxx, maxy = map(int, rect.bounds)
    minx = max(0, minx)
    miny = max(0, miny)
    maxx = min(filled_mask.shape[1], maxx)
    maxy = min(filled_mask.shape[0], maxy)
    region = filled_mask[miny:maxy, minx:maxx]
    if region.size == 0:
        return 0.0
    return float(np.sum(region > 0) / region.size)


def _merge_overlapping_rectangles(rectangles: List[Polygon], overlap_threshold: float = 0.20) -> List[Polygon]:
    if len(rectangles) <= 1:
        return rectangles
    merged = True
    current = rectangles[:]
    while merged:
        merged = False
        new_rects: List[Polygon] = []
        used: set[int] = set()
        for i in range(len(current)):
            if i in used:
                continue
            r1 = current[i]
            did = False
            for j in range(i + 1, len(current)):
                if j in used:
                    continue
                r2 = current[j]
                inter = r1.intersection(r2)
                if inter.is_empty:
                    continue
                o1 = float(inter.area / r1.area) if r1.area else 0.0
                o2 = float(inter.area / r2.area) if r2.area else 0.0
                if o1 > overlap_threshold or o2 > overlap_threshold:
                    minx, miny, maxx, maxy = r1.union(r2).bounds
                    new_rects.append(box(minx, miny, maxx, maxy))
                    used.add(i)
                    used.add(j)
                    merged = True
                    did = True
                    break
            if not did:
                new_rects.append(r1)
                used.add(i)
        current = new_rects
    return current


def _ensure_full_coverage(rectangles: List[Polygon], filled_mask: np.ndarray, min_rect_size: int = GRID_MIN_RECT_SIZE_PX) -> List[Polygon]:
    import cv2

    covered = np.zeros_like(filled_mask)
    for rect in rectangles:
        minx, miny, maxx, maxy = map(int, rect.bounds)
        minx = max(0, minx)
        miny = max(0, miny)
        maxx = min(filled_mask.shape[1], maxx)
        maxy = min(filled_mask.shape[0], maxy)
        region = filled_mask[miny:maxy, minx:maxx]
        covered[miny:maxy, minx:maxx] = np.maximum(covered[miny:maxy, minx:maxx], region)

    uncovered = (filled_mask > 0) & (covered == 0)
    if int(np.sum(uncovered)) == 0:
        return rectangles

    contours, _ = cv2.findContours(uncovered.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = rectangles[:]
    for c in contours:
        if cv2.contourArea(c) < float(min_rect_size):
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w >= min_rect_size and h >= min_rect_size:
            out.append(box(x, y, x + w, y + h))
    return out


def _grid_decompose(polygon: Polygon, filled_mask: np.ndarray, min_rect_size: int = GRID_MIN_RECT_SIZE_PX) -> List[Polygon]:
    """
    Grid-based candidates + greedy coverage (inspired by user's pipeline).
    Produces axis-aligned rectangles in image coordinates.
    """
    minx, miny, maxx, maxy = map(int, polygon.bounds)
    coords = list(polygon.exterior.coords[:-1]) if polygon.exterior is not None else []
    xs = sorted(set([minx, maxx] + [int(c[0]) for c in coords]))
    ys = sorted(set([miny, maxy] + [int(c[1]) for c in coords]))

    candidates: List[Dict[str, object]] = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            x1, x2 = xs[i], xs[i + 1]
            y1, y2 = ys[j], ys[j + 1]
            if (x2 - x1) < min_rect_size or (y2 - y1) < min_rect_size:
                continue
            rect = box(x1, y1, x2, y2)
            # Overlap vs mask region
            rx1, ry1, rx2, ry2 = map(int, rect.bounds)
            rx1 = max(0, rx1); ry1 = max(0, ry1)
            rx2 = min(filled_mask.shape[1], rx2); ry2 = min(filled_mask.shape[0], ry2)
            region = filled_mask[ry1:ry2, rx1:rx2]
            if region.size == 0:
                continue
            overlap = float(np.sum(region > 0) / region.size)
            if overlap > GRID_OVERLAP_THRESH:
                candidates.append({"rect": rect, "area": float(rect.area), "overlap": overlap})

    if not candidates:
        return [box(minx, miny, maxx, maxy)]

    candidates.sort(key=lambda d: float(d["area"]), reverse=True)

    selected: List[Polygon] = []
    covered = np.zeros_like(filled_mask)
    total = int(np.sum(filled_mask > 0))
    covered_pixels = 0
    if total == 0:
        return [box(minx, miny, maxx, maxy)]

    for cand in candidates:
        rect = cand["rect"]  # type: ignore[assignment]
        assert isinstance(rect, Polygon)
        rx1, ry1, rx2, ry2 = map(int, rect.bounds)
        rx1 = max(0, rx1); ry1 = max(0, ry1)
        rx2 = min(filled_mask.shape[1], rx2); ry2 = min(filled_mask.shape[0], ry2)
        region = filled_mask[ry1:ry2, rx1:rx2]
        cur = covered[ry1:ry2, rx1:rx2]
        new_pixels = int(np.sum((region > 0) & (cur == 0)))
        new_ratio = new_pixels / total
        cur_ratio = covered_pixels / total
        if new_ratio > 0.05 or (cur_ratio < 0.99 and new_ratio > 0.01):
            selected.append(rect)
            covered[ry1:ry2, rx1:rx2] = np.maximum(cur, region)
            covered_pixels = int(np.sum((covered > 0) & (filled_mask > 0)))
            if covered_pixels >= int(total * 0.999):
                break

    return selected


def _largest_rectangle_in_mask(bin_mask: np.ndarray) -> Optional[Tuple[int, int, int, int, int]]:
    """
    Largest axis-aligned rectangle of 1s in a binary mask.
    Returns (x1, y1, x2, y2, area) where x2/y2 are inclusive.
    """
    if bin_mask.size == 0:
        return None
    h, w = bin_mask.shape[:2]
    heights = np.zeros((w,), dtype=np.int32)
    best: Optional[Tuple[int, int, int, int, int]] = None

    for y in range(h):
        row = bin_mask[y]
        # update histogram heights
        heights = np.where(row > 0, heights + 1, 0)

        # largest rectangle in histogram via stack
        stack: List[int] = []
        for i in range(w + 1):
            cur = int(heights[i]) if i < w else 0
            while stack and int(heights[stack[-1]]) > cur:
                top = stack.pop()
                height = int(heights[top])
                if height <= 0:
                    continue
                left = (stack[-1] + 1) if stack else 0
                right = i - 1
                width = right - left + 1
                area = height * width
                if best is None or area > best[4]:
                    x1 = left
                    x2 = right
                    y2 = y
                    y1 = y - height + 1
                    best = (x1, y1, x2, y2, area)
            stack.append(i)
    return best


def _greedy_max_rectangles(
    filled_mask: np.ndarray,
    min_rect_size: int = GRID_MIN_RECT_SIZE_PX,
    max_rects: int = MAX_GRID_RECTS,
) -> List[Polygon]:
    """
    Greedy cover by repeatedly extracting the largest rectangle fully inside the mask.
    This tends to pick "cel mai mare posibil" rectangles first.
    """
    bin_mask = (filled_mask > 0).astype(np.uint8)
    rects: List[Polygon] = []

    for _ in range(max_rects):
        best = _largest_rectangle_in_mask(bin_mask)
        if best is None:
            break
        x1, y1, x2, y2, area = best
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        if area < MIN_RECTANGLE_AREA_PX or w < min_rect_size or h < min_rect_size:
            break
        # shapely box uses half-open style; add 1 to include last pixel
        rects.append(box(float(x1), float(y1), float(x2 + 1), float(y2 + 1)))
        # remove covered area
        bin_mask[y1 : y2 + 1, x1 : x2 + 1] = 0

        # early stop if almost nothing left
        if int(np.sum(bin_mask > 0)) < MIN_RECTANGLE_AREA_PX:
            break

    return rects


def partition_into_rectangles(polygon: Polygon, filled_mask: Optional[np.ndarray] = None) -> List[Tuple[Polygon, float]]:
    """
    Returnează listă de (rect_poly, angle_deg) pentru secțiuni.
    """
    if polygon is None or polygon.is_empty or not polygon.is_valid:
        return []

    # Prefer grid+coverage method when we have a filled mask
    if filled_mask is not None:
        # 1) Try "largest possible" rectangles first (greedy maximal-rectangle cover)
        rects = _greedy_max_rectangles(filled_mask, min_rect_size=GRID_MIN_RECT_SIZE_PX, max_rects=MAX_GRID_RECTS)
        # 2) If that fails (rare), fallback to the grid candidates method
        if not rects:
            rects = _grid_decompose(polygon, filled_mask, min_rect_size=GRID_MIN_RECT_SIZE_PX)
        # strict filter (>=90%) like in user's pipeline
        rects = [r for r in rects if _check_mask_overlap(r, filled_mask) >= GRID_STRICT_OVERLAP]
        # ensure coverage close to 100%
        rects = _ensure_full_coverage(rects, filled_mask, min_rect_size=GRID_MIN_RECT_SIZE_PX)
        rects = [r for r in rects if _check_mask_overlap(r, filled_mask) >= GRID_STRICT_OVERLAP]
        rects = _merge_overlapping_rectangles(rects, overlap_threshold=0.20)
        # axis-aligned => angle 0
        return [(r, 0.0) for r in rects if r is not None and not r.is_empty and float(r.area) >= MIN_RECTANGLE_AREA_PX]

    out: List[Tuple[Polygon, float]] = []
    _partition_recursive(polygon, out, MAX_RECURSION_DEPTH)
    if not out:
        rect, _w, _h, angle = _oriented_bounding_rect(polygon)
        if rect is not None and (not rect.is_empty):
            out = [(rect, float(angle))]
    return out


def medial_axis_decomposition(polygon: Polygon) -> List[Tuple[float, float, float, float]]:
    """
    Păstrat doar ca fallback API: întoarce un singur box ca (cx, cy, length, width).
    """
    if polygon is None or polygon.is_empty:
        return []
    rect, w, h, _angle = _oriented_bounding_rect(polygon)
    if rect is None or rect.is_empty:
        return []
    minx, miny, maxx, maxy = rect.bounds
    cx = float((minx + maxx) / 2.0)
    cy = float((miny + maxy) / 2.0)
    return [(cx, cy, float(max(w, h)), float(min(w, h)))]


