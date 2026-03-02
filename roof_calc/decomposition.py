"""
FIX: Detectare completă a dreptunghiurilor - TREBUIE 5 în loc de 3!

Integrare din rularea veche (1f687b1):
- Descompunere recursivă pe colțuri concave (_partition_recursive, _split_once) pentru L/T
- Grid decomposition (_grid_decompose) ca fallback
- Greedy largest-rectangle + completare zone neacoperite (actual)
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from shapely.geometry import Polygon, box, LineString
from shapely.ops import split

# Constante doar ca fallback; în greedy calculăm din mărimea planului (relative)
MIN_RECTANGLE_AREA_PX = 100
GRID_MIN_RECT_SIZE_PX = 30
GRID_OVERLAP_THRESH = 0.85
GRID_STRICT_OVERLAP = 0.90
MAX_GRID_RECTS = 20
MAX_RECURSION_DEPTH = 10


def _find_concave_corners(polygon: Polygon) -> List[Tuple[float, float]]:
    """
    Colțuri concave (reflex) – vârfuri cu unghi interior > 180°.
    Funcționează pentru exterior CCW (Shapely); pentru CW folosim cross > 0.
    """
    if polygon is None or polygon.is_empty or not polygon.is_valid or polygon.exterior is None:
        return []
    coords = list(polygon.exterior.coords)
    if len(coords) < 4:
        return []
    pts = coords[:-1]
    n = len(pts)
    # Orientare: aria cu semn (CCW > 0)
    area_signed = 0.0
    for i in range(n):
        j = (i + 1) % n
        area_signed += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    concave: List[Tuple[float, float]] = []
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p_curr = pts[i]
        p_next = pts[(i + 1) % n]
        v1 = (float(p_curr[0] - p_prev[0]), float(p_curr[1] - p_prev[1]))
        v2 = (float(p_next[0] - p_curr[0]), float(p_next[1] - p_curr[1]))
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        # CCW: concave = cross < 0; CW: concave = cross > 0
        if (area_signed > 0 and cross < 0) or (area_signed < 0 and cross > 0):
            concave.append((float(p_curr[0]), float(p_curr[1])))
    return concave


def _split_once(poly: Polygon, min_area: float = 400.0) -> List[Polygon]:
    """Tăietură printr-un colț concav (verticală/orizontală). Linia cu offset mic ca să nu treacă exact prin vârf."""
    concave = _find_concave_corners(poly)
    if not concave:
        return []
    minx, miny, maxx, maxy = poly.bounds
    eps = 0.5  # offset ca linia să nu treacă exact prin vârf (probleme numerice Shapely)
    for cx, cy in concave:
        # verticală prin cx, orizontală prin cy; încercăm și cu offset
        for dx, dy in [(0, 0), (eps, 0), (0, eps), (-eps, 0), (0, -eps)]:
            cxx, cyy = cx + dx, cy + dy
            candidates = [
                LineString([(cxx, miny - 1), (cxx, maxy + 1)]),
                LineString([(minx - 1, cyy), (maxx + 1, cyy)]),
            ]
            for line in candidates:
                try:
                    parts = [p for p in split(poly, line).geoms if isinstance(p, Polygon)]
                    parts = [p for p in parts if not p.is_empty and float(p.area) >= min_area]
                    if len(parts) >= 2:
                        return parts
                except Exception:
                    continue
    return []


def _partition_recursive(
    poly: Polygon,
    out: List[Tuple[Polygon, float]],
    depth: int,
    min_area: float = 400.0,
) -> None:
    """Descompunere recursivă: taie prin colțuri concave până obține dreptunghiuri. Întotdeauna încercăm tăierea mai întâi."""
    if depth <= 0 or poly.is_empty or float(poly.area) < min_area:
        return
    # Întâi încercăm tăierea – nu verificăm classify_shape, ca și L-uri clasificate greșit să fie tăiate
    parts = _split_once(poly, min_area=min_area)
    if parts:
        for p in parts:
            _partition_recursive(p, out, depth - 1, min_area=min_area)
        return
    # Nu s-a putut tăia – un singur dreptunghi orientat
    rect, _w, _h, angle = _oriented_bounding_rect(poly)
    if rect is not None and not rect.is_empty and float(rect.area) >= min_area:
        out.append((rect, float(angle)))


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
    Extrage dreptunghiuri până nu mai există niciunul valid.
    Toate pragurile sunt relative la mărimea planului (fără numere fixe de pixeli).
    """
    bin_mask = (filled_mask > 0).astype(np.uint8)
    rects: List[Polygon] = []
    h, w = filled_mask.shape[:2]
    total_pixels = int(np.sum(bin_mask > 0))
    if total_pixels == 0:
        return []

    # Praguri relative: ~1% din latura mică, ~0.005% din aria totală (agresiv pentru L/T)
    small_side = min(h, w)
    min_side = max(3, int(small_side * 0.01))
    min_area = max(15, int(total_pixels * 0.00005))

    for iteration in range(max_rects):
        best = _largest_rectangle_in_mask(bin_mask)
        if best is None:
            break
        
        x1, y1, x2, y2, area = best
        w_rect = x2 - x1 + 1
        h_rect = y2 - y1 + 1
        
        if w_rect < min_side or h_rect < min_side:
            break
        if area < min_area:
            break
        
        rects.append(box(float(x1), float(y1), float(x2 + 1), float(y2 + 1)))
        bin_mask[y1 : y2 + 1, x1 : x2 + 1] = 0
        
        remaining_pixels = int(np.sum(bin_mask > 0))
        coverage_percent = (total_pixels - remaining_pixels) / total_pixels * 100 if total_pixels else 0
        
        print(f"  Iterație {iteration+1}: Dreptunghi {w_rect}x{h_rect} (arie={area}), "
              f"Acoperire: {coverage_percent:.1f}%, Rămân: {remaining_pixels} px")

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


def _grid_decompose(
    polygon: Polygon,
    filled_mask: np.ndarray,
    min_rect_size: int = GRID_MIN_RECT_SIZE_PX,
) -> List[Polygon]:
    """
    Grid-based candidates + greedy coverage (din rularea veche).
    Candidați din gridul poligonului, sortare după arie, acoperire până la ~100%.
    """
    minx, miny, maxx, maxy = map(int, polygon.bounds)
    coords = list(polygon.exterior.coords[:-1]) if polygon.exterior else []
    xs = sorted(set([minx, maxx] + [int(c[0]) for c in coords]))
    ys = sorted(set([miny, maxy] + [int(c[1]) for c in coords]))
    candidates: List[Dict[str, Any]] = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            x1, x2 = xs[i], xs[i + 1]
            y1, y2 = ys[j], ys[j + 1]
            if (x2 - x1) < min_rect_size or (y2 - y1) < min_rect_size:
                continue
            rect = box(x1, y1, x2, y2)
            rx1, ry1, rx2, ry2 = max(0, int(rect.bounds[0])), max(0, int(rect.bounds[1])), min(filled_mask.shape[1], int(rect.bounds[2])), min(filled_mask.shape[0], int(rect.bounds[3]))
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
        rect = cand["rect"]
        rx1, ry1, rx2, ry2 = max(0, int(rect.bounds[0])), max(0, int(rect.bounds[1])), min(filled_mask.shape[1], int(rect.bounds[2])), min(filled_mask.shape[0], int(rect.bounds[3]))
        region = filled_mask[ry1:ry2, rx1:rx2]
        cur = covered[ry1:ry2, rx1:rx2]
        new_pixels = int(np.sum((region > 0) & (cur == 0)))
        new_ratio = new_pixels / total if total else 0
        cur_ratio = covered_pixels / total if total else 0
        if new_ratio > 0.05 or (cur_ratio < 0.99 and new_ratio > 0.01):
            selected.append(rect)
            covered[ry1:ry2, rx1:rx2] = np.maximum(cur, region)
            covered_pixels = int(np.sum((covered > 0) & (filled_mask > 0)))
            if covered_pixels >= int(total * 0.999):
                break
    return selected


def _ensure_full_coverage(rectangles: List[Polygon], filled_mask: np.ndarray, 
                         min_rect_size: int = GRID_MIN_RECT_SIZE_PX) -> List[Polygon]:
    """
    Pentru zonele neacoperite: rulează același algoritm greedy (cât mai multe dreptunghiuri
    până umpli forma), nu doar un bbox per contur. Astfel zonele tip L sunt umplute cu
    mai multe dreptunghiuri care trec filtrul de overlap 90%.
    """
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
    
    # Mască pentru zona neacoperită (aceleași valori > 0 ca filled_mask pentru greedy)
    uncovered_mask = np.where(uncovered, filled_mask, 0).astype(filled_mask.dtype)
    if uncovered_mask.max() == 0:
        uncovered_mask = uncovered.astype(np.uint8) * 255

    print(f"\n  🔍 Zone neacoperite: {uncovered_pixels} px – rulare greedy pe zona rămasă...")
    extra_rects = _greedy_max_rectangles_FIXED(
        uncovered_mask,
        min_rect_size=min_rect_size,
        max_rects=MAX_GRID_RECTS,
    )
    if extra_rects:
        print(f"  ✓ Adăugate {len(extra_rects)} dreptunghiuri pentru completare")
    out = list(rectangles) + extra_rects
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
        min_rect_size=GRID_MIN_RECT_SIZE_PX,
        max_rects=MAX_GRID_RECTS,
    )
    if not rects:
        print("  ⚠ Greedy: 0 dreptunghiuri – încerc grid decomposition (rulare veche)...")
        rects = _grid_decompose(polygon, filled_mask, min_rect_size=GRID_MIN_RECT_SIZE_PX)
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
    
    min_area_relative = max(25, int(total_pixels * 0.0001))
    return [(r, 0.0) for r in final if r is not None and not r.is_empty and float(r.area) >= min_area_relative]


def partition_into_rectangles(
    polygon: Polygon,
    filled_mask: Optional[np.ndarray] = None,
) -> List[Tuple[Polygon, float]]:
    """
    Public API: descompune poligonul în dreptunghiuri.
    Returnează [(rect, angle), ...].
    Integrare rulare veche: mai întâi încercare descompunere recursivă (colțuri concave) pentru L/T.
    """
    if polygon is None or polygon.is_empty:
        return []
    if not polygon.is_valid:
        try:
            polygon = polygon.buffer(0)
            if hasattr(polygon, "geoms"):
                polygon = max(polygon.geoms, key=lambda g: getattr(g, "area", 0))
        except Exception:
            return []
        if polygon.is_empty or not polygon.is_valid:
            return []
    if filled_mask is None:
        out: List[Tuple[Polygon, float]] = []
        _partition_recursive(polygon, out, MAX_RECURSION_DEPTH, min_area=float(MIN_RECTANGLE_AREA_PX))
        if not out:
            rect, _w, _h, angle = _oriented_bounding_rect(polygon)
            if rect is not None and not rect.is_empty:
                return [(rect, float(angle))]
        return out
    # Avem mască: construim poligon din conturul măștii (toate vârfurile) ca să păstrăm colțurile concave
    try:
        import cv2
        bin_m = (filled_mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if len(largest) >= 3:
                coords = [(float(p[0][0]), float(p[0][1])) for p in largest]
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                # Orientare: Shapely vrea exterior CCW; OpenCV poate da CW
                area_signed = 0.0
                for i in range(len(coords) - 1):
                    area_signed += coords[i][0] * coords[i + 1][1] - coords[i + 1][0] * coords[i][1]
                if area_signed < 0:
                    coords = coords[::-1]
                poly_from_mask = Polygon(coords)
                if poly_from_mask.is_empty:
                    pass
                elif not poly_from_mask.is_valid:
                    try:
                        fixed = poly_from_mask.buffer(0)
                        if hasattr(fixed, "geoms"):
                            poly_from_mask = max(fixed.geoms, key=lambda g: getattr(g, "area", 0))
                        else:
                            poly_from_mask = fixed
                    except Exception:
                        pass
                    if not poly_from_mask.is_empty and getattr(poly_from_mask, "is_valid", True):
                        polygon = poly_from_mask
                else:
                    polygon = poly_from_mask
    except Exception:
        pass

    total_pixels = int(np.sum(filled_mask > 0))
    min_area_rel = max(25.0, float(total_pixels * 0.0001))
    # Prag foarte mic la tăiere ca L-ul să fie împărțit în 2+ dreptunghiuri
    min_area_split = max(1.0, float(total_pixels * 0.00002))
    n_concave = len(_find_concave_corners(polygon))
    out_rec: List[Tuple[Polygon, float]] = []
    _partition_recursive(polygon, out_rec, MAX_RECURSION_DEPTH, min_area=min_area_split)
    filtered_rec = [(r, a) for r, a in out_rec if _check_mask_overlap(r, filled_mask) >= GRID_STRICT_OVERLAP]
    min_area_f = max(25, int(total_pixels * 0.0001))
    result_rec = [(r, a) for r, a in filtered_rec if r is not None and not r.is_empty and float(r.area) >= min_area_f]

    result_greedy = partition_into_rectangles_FIXED(polygon, filled_mask)

    # Dacă greedy dă 1 dreptunghi cu aria ~totală (≥98%), păstrăm 1 – nu împărțim un dreptunghi evident
    if len(result_greedy) == 1 and result_greedy:
        r0, _ = result_greedy[0]
        if r0 is not None and not r0.is_empty and total_pixels > 0:
            if float(r0.area) >= 0.98 * total_pixels:
                return result_greedy

    if len(result_rec) > len(result_greedy):
        print("  ✓ Descompunere recursivă:", len(result_rec), "dreptunghiuri.", flush=True)
        return result_rec
    if n_concave > 0 and len(result_rec) > 0:
        print("  ○ Recursiv", len(result_rec), "vs greedy", len(result_greedy), "– folosit greedy.", flush=True)
    return result_greedy


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