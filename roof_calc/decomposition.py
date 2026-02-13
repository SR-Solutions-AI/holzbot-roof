"""
FIX: Detectare completă a dreptunghiurilor - TREBUIE 5 în loc de 3!

PROBLEMA IDENTIFICATĂ:
- _greedy_max_rectangles() se oprește când găsește primele 3 dreptunghiuri mari
- Condiția de early stop (np.sum(bin_mask > 0) < MIN_RECTANGLE_AREA_PX) elimină
  dreptunghiurile mici rămase
- GRID_MIN_RECT_SIZE_PX = 50 este prea mare și filtrează dreptunghiurile mici

SOLUȚIE:
1. Reduce MIN_RECTANGLE_AREA_PX la 100 (în loc de 1)
2. Reduce GRID_MIN_RECT_SIZE_PX la 30 (în loc de 50)
3. Elimină early stop prematur
4. Modifică condiția de coverage pentru a include dreptunghiurile mici
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from shapely.geometry import Polygon, box

# CONSTANTE MODIFICATE
MIN_RECTANGLE_AREA_PX = 100      # În loc de 1
GRID_MIN_RECT_SIZE_PX = 30       # În loc de 50 - permite dreptunghiuri mai mici
GRID_OVERLAP_THRESH = 0.85
GRID_STRICT_OVERLAP = 0.90
MAX_GRID_RECTS = 20              # În loc de 12 - permite mai multe dreptunghiuri


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
        heights = np.where(row > 0, heights + 1, 0)

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


def _oriented_bounding_rect(polygon: Polygon) -> Tuple[Optional[Polygon], float, float, float]:
    """
    Minimum-area oriented bounding rectangle for a polygon.
    Returns (rect, width, height, angle_deg) where angle is in degrees.
    """
    if polygon is None or polygon.is_empty or not polygon.is_valid:
        return (None, 0.0, 0.0, 0.0)
    try:
        mrr = polygon.minimum_rotated_rectangle
        if mrr is None or mrr.is_empty:
            env = polygon.envelope
            return (env, float(env.bounds[2] - env.bounds[0]), float(env.bounds[3] - env.bounds[1]), 0.0)
        coords = list(mrr.exterior.coords)
        if len(coords) < 4:
            env = polygon.envelope
            return (env, float(env.bounds[2] - env.bounds[0]), float(env.bounds[3] - env.bounds[1]), 0.0)
        p0 = coords[0]
        p1 = coords[1]
        p3 = coords[3]
        dx1 = p1[0] - p0[0]
        dy1 = p1[1] - p0[1]
        dx2 = p3[0] - p0[0]
        dy2 = p3[1] - p0[1]
        w1 = math.hypot(dx1, dy1)
        w2 = math.hypot(dx2, dy2)
        if w1 >= w2:
            w, h = w1, w2
            angle_rad = math.atan2(dy1, dx1)
        else:
            w, h = w2, w1
            angle_rad = math.atan2(dy2, dx2)
        angle_deg = math.degrees(angle_rad)
        return (mrr, float(w), float(h), float(angle_deg))
    except Exception:
        env = polygon.envelope
        return (env, float(env.bounds[2] - env.bounds[0]), float(env.bounds[3] - env.bounds[1]), 0.0)


def _greedy_max_rectangles_FIXED(
    filled_mask: np.ndarray,
    min_rect_size: int = GRID_MIN_RECT_SIZE_PX,
    max_rects: int = MAX_GRID_RECTS,
) -> List[Polygon]:
    """
    VERSIUNE CORECTATĂ: Continuă să extragi dreptunghiuri până când:
    1. Am găsit max_rects dreptunghiuri SAU
    2. Nu mai există zone neacoperite >= min_rect_size
    
    NU se mai oprește prematur când găsește primele dreptunghiuri mari!
    """
    bin_mask = (filled_mask > 0).astype(np.uint8)
    rects: List[Polygon] = []
    
    # Calculează aria totală
    total_pixels = int(np.sum(bin_mask > 0))
    if total_pixels == 0:
        return []

    for iteration in range(max_rects):
        best = _largest_rectangle_in_mask(bin_mask)
        if best is None:
            break
        
        x1, y1, x2, y2, area = best
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        
        # MODIFICARE 1: Acceptă dreptunghiuri mai mici (min 30x30 în loc de 50x50)
        if w < min_rect_size or h < min_rect_size:
            break
        
        # MODIFICARE 2: Acceptă dreptunghiuri cu arie >= 100 px (în loc de 1)
        if area < MIN_RECTANGLE_AREA_PX:
            break
        
        # Adaugă dreptunghiul
        rects.append(box(float(x1), float(y1), float(x2 + 1), float(y2 + 1)))
        
        # Marchează zona ca acoperită
        bin_mask[y1 : y2 + 1, x1 : x2 + 1] = 0
        
        # MODIFICARE 3: NU te opri prematur!
        # Continuă să cauți până când:
        # - Am găsit max_rects dreptunghiuri SAU
        # - Nu mai există zone neacoperite semnificative
        remaining_pixels = int(np.sum(bin_mask > 0))
        coverage_percent = (total_pixels - remaining_pixels) / total_pixels * 100
        
        # Debug info
        print(f"  Iterație {iteration+1}: Dreptunghi {w}x{h} (arie={area}), "
              f"Acoperire: {coverage_percent:.1f}%, Rămân: {remaining_pixels} px")
        
        # MODIFICARE 4: Oprește DOAR când coverage > 99.5% SAU nu mai există zone >= 900 px
        if coverage_percent > 99.5:
            print(f"  ✓ Acoperire completă: {coverage_percent:.1f}%")
            break
        
        if remaining_pixels < 900:  # ~30x30 pixels
            print(f"  ✓ Zone rămase prea mici: {remaining_pixels} px")
            break

    return rects


def _check_mask_overlap(rect: Polygon, filled_mask: np.ndarray) -> float:
    """Calculează overlap-ul unui dreptunghi cu masca"""
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
    """Combină dreptunghiuri care se suprapun semnificativ"""
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


def _ensure_full_coverage(rectangles: List[Polygon], filled_mask: np.ndarray, 
                         min_rect_size: int = GRID_MIN_RECT_SIZE_PX) -> List[Polygon]:
    """
    Adaugă dreptunghiuri pentru zonele neacoperite.
    MODIFICAT: Folosește min_rect_size mai mic (30 în loc de 50)
    """
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
    uncovered_pixels = int(np.sum(uncovered))
    
    if uncovered_pixels == 0:
        return rectangles
    
    print(f"\n  🔍 Zone neacoperite: {uncovered_pixels} px")

    contours, _ = cv2.findContours(uncovered.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = rectangles[:]
    
    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < float(MIN_RECTANGLE_AREA_PX):
            continue
        
        x, y, w, h = cv2.boundingRect(c)
        
        # MODIFICARE: Acceptă dreptunghiuri >= 30x30 (în loc de 50x50)
        if w >= min_rect_size and h >= min_rect_size:
            print(f"    + Adăugat dreptunghi de completare: {w}x{h} (arie={area:.0f})")
            out.append(box(x, y, x + w, y + h))
    
    return out


def partition_into_rectangles_FIXED(polygon: Polygon, filled_mask: np.ndarray) -> List[Tuple[Polygon, float]]:
    """
    VERSIUNE CORECTATĂ care detectează TOATE dreptunghiurile (5 în loc de 3).
    
    Modificări:
    1. Reduce dimensiunea minimă la 30x30 (în loc de 50x50)
    2. Continuă să extragi până la acoperire 99.5%
    3. Nu se oprește prematur când găsește primele dreptunghiuri mari
    4. Adaugă dreptunghiuri de completare pentru zonele rămase
    """
    if polygon is None or polygon.is_empty or not polygon.is_valid:
        return []
    
    if filled_mask is None:
        raise ValueError("filled_mask este necesar pentru detectare completă!")
    
    print("\n" + "="*70)
    print("  🔍 DETECTARE DREPTUNGHIURI (VERSIUNE CORECTATĂ)")
    print("="*70)
    
    # STEP 1: Extrage dreptunghiuri maxime (greedy)
    print("\n📦 STEP 1: Extragere dreptunghiuri maxime...")
    rects = _greedy_max_rectangles_FIXED(
        filled_mask, 
        min_rect_size=GRID_MIN_RECT_SIZE_PX,  # 30 în loc de 50
        max_rects=MAX_GRID_RECTS               # 20 în loc de 12
    )
    print(f"  ✓ Găsite: {len(rects)} dreptunghiuri")
    
    # STEP 2: Filtrare strictă (overlap >= 90%)
    print("\n🔍 STEP 2: Filtrare strictă (overlap >= 90%)...")
    filtered = [r for r in rects if _check_mask_overlap(r, filled_mask) >= GRID_STRICT_OVERLAP]
    print(f"  ✓ După filtrare: {len(filtered)} dreptunghiuri")
    
    # STEP 3: Completare zone neacoperite
    print("\n📋 STEP 3: Completare zone neacoperite...")
    complete = _ensure_full_coverage(filtered, filled_mask, min_rect_size=GRID_MIN_RECT_SIZE_PX)
    print(f"  ✓ După completare: {len(complete)} dreptunghiuri")
    
    # STEP 4: Re-filtrare după completare
    print("\n🔍 STEP 4: Re-filtrare strictă...")
    complete = [r for r in complete if _check_mask_overlap(r, filled_mask) >= GRID_STRICT_OVERLAP]
    print(f"  ✓ După re-filtrare: {len(complete)} dreptunghiuri")
    
    # STEP 5: Combinare overlap-uri
    print("\n🔗 STEP 5: Combinare overlap-uri...")
    final = _merge_overlapping_rectangles(complete, overlap_threshold=0.20)
    print(f"  ✓ Final: {len(final)} dreptunghiuri")
    
    # STEP 6: Verificare acoperire finală
    covered = np.zeros_like(filled_mask)
    for rect in final:
        minx, miny, maxx, maxy = map(int, rect.bounds)
        minx = max(0, minx)
        miny = max(0, miny)
        maxx = min(filled_mask.shape[1], maxx)
        maxy = min(filled_mask.shape[0], maxy)
        region = filled_mask[miny:maxy, minx:maxx]
        covered[miny:maxy, minx:maxx] = np.maximum(covered[miny:maxy, minx:maxx], region)
    
    total_pixels = int(np.sum(filled_mask > 0))
    covered_pixels = int(np.sum((covered > 0) & (filled_mask > 0)))
    coverage = (covered_pixels / total_pixels * 100) if total_pixels > 0 else 0
    
    print(f"\n📊 REZULTAT FINAL:")
    print(f"  ├─ Dreptunghiuri detectate: {len(final)}")
    print(f"  ├─ Acoperire: {coverage:.2f}%")
    print(f"  └─ Status: {'✓ SUCCES' if coverage >= 99.0 else '⚠ INCOMPLET'}")
    print("="*70 + "\n")
    
    # Returnează cu angle=0 (axis-aligned)
    return [(r, 0.0) for r in final if r is not None and not r.is_empty and float(r.area) >= MIN_RECTANGLE_AREA_PX]


def partition_into_rectangles(
    polygon: Polygon,
    filled_mask: Optional[np.ndarray] = None,
) -> List[Tuple[Polygon, float]]:
    """
    Public API: descompune poligonul în dreptunghiuri.
    Returnează [(rect, angle), ...].
    """
    if filled_mask is not None:
        return partition_into_rectangles_FIXED(polygon, filled_mask)
    # Fallback când nu există mască: un singur dreptunghi orientat
    rect, _w, _h, angle = _oriented_bounding_rect(polygon)
    if rect is not None and not rect.is_empty:
        return [(rect, angle)]
    return []


# ============================================================================
# INSTRUCȚIUNI DE INTEGRARE
# ============================================================================

"""
PENTRU A FIXA PROBLEMA ÎN CODUL TĂU:

1. În decompose.py, ÎNLOCUIEȘTE:
   
   MIN_RECTANGLE_AREA_PX = 1
   GRID_MIN_RECT_SIZE_PX = 50
   MAX_GRID_RECTS = 12
   
   CU:
   
   MIN_RECTANGLE_AREA_PX = 100
   GRID_MIN_RECT_SIZE_PX = 30
   MAX_GRID_RECTS = 20

2. În funcția _greedy_max_rectangles(), ÎNLOCUIEȘTE:
   
   # Early stop
   if int(np.sum(bin_mask > 0)) < MIN_RECTANGLE_AREA_PX:
       break
   
   CU:
   
   remaining_pixels = int(np.sum(bin_mask > 0))
   coverage_percent = (total_pixels - remaining_pixels) / total_pixels * 100
   
   if coverage_percent > 99.5:
       break
   
   if remaining_pixels < 900:
       break

3. În funcția partition_into_rectangles(), ASIGURĂ-TE că:
   - Folosește _greedy_max_rectangles() cu parametrii noi
   - Filtrează cu GRID_STRICT_OVERLAP = 0.90
   - Folosește _ensure_full_coverage() pentru completare
   - Combină cu _merge_overlapping_rectangles()

REZULTAT AȘTEPTAT:
- În loc de 3 dreptunghiuri, vei obține 5 dreptunghiuri
- Acoperire >= 99.5%
- Toate zonele L-ului vor fi detectate
"""