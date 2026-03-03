"""
Workflow curat: rectangles (cu eliminare suprapuneri) + roof_types (0_w, 1_w, 2_w, 4_w, 4.5_w).
0_w = plat; 1_w = o apă; 2_w = două ape; 4_w = hip; 4.5_w = half-hip. Fiecare tip: lines.png + faces.png.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from roof_calc.roof_unfold import generate_unfold_masks_for_roof_types


def _rect_area(sec: Dict[str, Any]) -> float:
    br = sec.get("bounding_rect") or []
    if len(br) < 3:
        return 0.0
    xs = [float(p[0]) for p in br]
    ys = [float(p[1]) for p in br]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _section_rect_segments(section: Dict[str, Any]) -> List[List[List[float]]]:
    """Returnează cele 4 segmente ale dreptunghiului de frontieră al secțiunii (inel închis)."""
    br = section.get("bounding_rect") or []
    if len(br) < 3:
        return []
    pts = [[float(p[0]), float(p[1])] for p in br]
    if pts[0][0] == pts[-1][0] and pts[0][1] == pts[-1][1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return []
    segs: List[List[List[float]]] = []
    for i in range(len(pts)):
        p0 = pts[i]
        p1 = pts[(i + 1) % len(pts)]
        segs.append([p0, p1])
    return segs


def _rect_polygon(sec: Dict[str, Any]):
    from shapely.geometry import Polygon as ShapelyPolygon

    br = sec.get("bounding_rect") or []
    if len(br) < 3:
        return None
    try:
        return ShapelyPolygon([(float(p[0]), float(p[1])) for p in br])
    except Exception:
        return None


def _section_connected_components(sections: List[Dict[str, Any]]) -> List[List[int]]:
    """
    Grupează secțiunile în componente conectate: două secțiuni sunt în aceeași componentă
    dacă se ating (sunt lipite, ex. 2 dreptunghiuri care formează L).
    Returnează listă de liste de indici: [ [0, 1], [2] ] = secțiunile 0,1 sunt lipite, 2 e separat.
    """
    if not sections:
        return []
    n = len(sections)
    polys: List[Any] = []
    for sec in sections:
        p = _rect_polygon(sec)
        if p is None or getattr(p, "is_empty", True):
            polys.append(None)
        else:
            if not getattr(p, "is_valid", True):
                try:
                    p = p.buffer(0)
                except Exception:
                    pass
            polys.append(p)
    # Union-find: i și j în același set dacă poly i și poly j se ating
    parent = list(range(n))
    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    try:
        for i in range(n):
            if polys[i] is None:
                continue
            for j in range(i + 1, n):
                if polys[j] is None:
                    continue
                if polys[i].touches(polys[j]):
                    union(i, j)
                elif float(polys[i].distance(polys[j])) < 1.0:
                    # Aproape lipite (ex. L cu gap numeric) → considerăm aceeași componentă
                    union(i, j)
    except Exception:
        pass
    comps: Dict[int, List[int]] = {}
    for i in range(n):
        if polys[i] is None:
            continue
        r = find(i)
        comps.setdefault(r, []).append(i)
    return list(comps.values())


def remove_overlapping_rectangles(sections: List[Dict[str, Any]], iou_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """
    Elimină dreptunghiurile care se suprapun. Păstrează cel mai mare din fiecare pereche suprapusă.
    """
    if len(sections) <= 1:
        return list(sections)

    polys: List[Tuple[Dict[str, Any], Any]] = []
    for sec in sections:
        poly = _rect_polygon(sec)
        if poly is not None and not poly.is_empty:
            polys.append((sec, poly))

    to_remove: set = set()
    for i in range(len(polys)):
        if i in to_remove:
            continue
        sec_i, poly_i = polys[i]
        area_i = _rect_area(sec_i)
        for j in range(i + 1, len(polys)):
            if j in to_remove:
                continue
            sec_j, poly_j = polys[j]
            try:
                inter = poly_i.intersection(poly_j)
                inter_area = float(getattr(inter, "area", 0) or 0)
                if inter_area < 1e-6:
                    continue
                area_j = _rect_area(sec_j)
                min_area = min(area_i, area_j)
                if min_area < 1e-6:
                    continue
                iou = inter_area / min_area
                if iou >= iou_threshold:
                    if area_i >= area_j:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break
            except Exception:
                pass

    return [polys[i][0] for i in range(len(polys)) if i not in to_remove]


def _contours_from_mask(mask: np.ndarray) -> List[List[Tuple[float, float]]]:
    if mask is None or mask.size == 0:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if len(c) < 3:
            continue
        pts = [(float(p[0][0]), float(p[0][1])) for p in c]
        pts.append(pts[0])
        out.append(pts)
    return out


def _has_ridge_intersection(sections: List[Dict[str, Any]]) -> bool:
    """True dacă avem ≥2 ridge-uri care se intersectează (ex. L sau T shape)."""
    if len(sections) < 2:
        return False
    from roof_calc.overhang import extend_secondary_sections_to_main_ridge

    extended = extend_secondary_sections_to_main_ridge(sections)
    main_sec = next((s for s in extended if s.get("is_main")), None)
    if main_sec is None:
        main_sec = max(extended, key=lambda s: _rect_area(s))
    main_ridge = main_sec.get("ridge_line") or []
    if len(main_ridge) < 2:
        return False
    ma = (float(main_ridge[0][0]), float(main_ridge[0][1]))
    mb = (float(main_ridge[1][0]), float(main_ridge[1][1]))
    main_orient = str(main_sec.get("ridge_orientation", "horizontal"))

    def _seg_intersect(a1, a2, b1, b2):
        ax1, ay1 = a1[0], a1[1]
        ax2, ay2 = a2[0], a2[1]
        bx1, by1 = b1[0], b1[1]
        bx2, by2 = b2[0], b2[1]
        d = (ax2 - ax1) * (by2 - by1) - (ay2 - ay1) * (bx2 - bx1)
        if abs(d) < 1e-12:
            return None
        t = ((bx1 - ax1) * (by2 - by1) - (by1 - ay1) * (bx2 - bx1)) / d
        u = ((ax2 - ax1) * (by1 - ay1) - (ay2 - ay1) * (bx1 - ax1)) / d
        if 0 <= t <= 1 and 0 <= u <= 1:
            return (ax1 + t * (ax2 - ax1), ay1 + t * (ay2 - ay1))
        return None

    for sec in extended:
        if sec is main_sec:
            continue
        if str(sec.get("ridge_orientation", "horizontal")) == main_orient:
            continue
        ridge = sec.get("ridge_line") or []
        if len(ridge) < 2:
            continue
        sa = (float(ridge[0][0]), float(ridge[0][1]))
        sb = (float(ridge[1][0]), float(ridge[1][1]))
        pt = _seg_intersect(ma, mb, sa, sb)
        if pt is not None:
            return True
    return False


def _ridge_intersections_on_segment(
    sec: Dict[str, Any],
    sections: List[Dict[str, Any]],
    ma: Tuple[float, float],
    mb: Tuple[float, float],
    tol: float = 1e-6,
) -> List[Tuple[float, float]]:
    """Intersecțiile ridge-ului sec cu alte ridge-uri, pe segmentul ma-mb. Sortate de la ma spre mb."""
    ax1, ay1 = ma[0], ma[1]
    ax2, ay2 = mb[0], mb[1]
    intersections: List[Tuple[float, float]] = []
    for osec in sections:
        if osec is sec:
            continue
        oridge = osec.get("ridge_line") or []
        if len(oridge) < 2:
            continue
        sa = (float(oridge[0][0]), float(oridge[0][1]))
        sb = (float(oridge[1][0]), float(oridge[1][1]))
        pt = _seg_intersect_pt(ma, mb, sa, sb, tol)
        if pt is None:
            continue
        px, py = pt
        if abs(ax2 - ax1) > abs(ay2 - ay1):
            t = (px - ax1) / (ax2 - ax1) if abs(ax2 - ax1) > 1e-12 else 0.0
        else:
            t = (py - ay1) / (ay2 - ay1) if abs(ay2 - ay1) > 1e-12 else 0.0
        if tol < t < 1.0 - tol:
            intersections.append(pt)
    if not intersections:
        return []
    if abs(ax2 - ax1) > abs(ay2 - ay1):
        intersections.sort(key=lambda p: p[0])
    else:
        intersections.sort(key=lambda p: p[1])
    return intersections


def _seg_intersect_pt(
    a1: Tuple[float, float], a2: Tuple[float, float],
    b1: Tuple[float, float], b2: Tuple[float, float],
    tol: float = 1e-9,
) -> Optional[Tuple[float, float]]:
    ax1, ay1, ax2, ay2 = a1[0], a1[1], a2[0], a2[1]
    bx1, by1, bx2, by2 = b1[0], b1[1], b2[0], b2[1]
    d = (ax2 - ax1) * (by2 - by1) - (ay2 - ay1) * (bx2 - bx1)
    if abs(d) < tol:
        return None
    t = ((bx1 - ax1) * (by2 - by1) - (by1 - ay1) * (bx2 - bx1)) / d
    u = ((ax2 - ax1) * (by1 - ay1) - (ay2 - ay1) * (bx1 - ax1)) / d
    if -tol <= t <= 1 + tol and -tol <= u <= 1 + tol:
        return (ax1 + t * (ax2 - ax1), ay1 + t * (ay2 - ay1))
    return None


def _edge_overlap(ax1: float, ay1: float, ax2: float, ay2: float,
                  bx1: float, by1: float, bx2: float, by2: float,
                  tol: float = 2.0) -> float:
    """Fracțiunea segmentului A acoperită de B (0..1), dacă sunt coliniare pe aceeași dreaptă."""
    dx_a, dy_a = ax2 - ax1, ay2 - ay1
    len_a_sq = dx_a * dx_a + dy_a * dy_a
    if len_a_sq < 1e-12:
        return 0.0
    dx_b, dy_b = bx2 - bx1, by2 - by1
    cross = abs(dx_a * dy_b - dy_a * dx_b)
    if cross > tol:
        return 0.0
    # Verificăm că B e pe aceeași dreaptă ca A (nu doar paralel)
    len_a = len_a_sq ** 0.5
    perp = (-dy_a, dx_a)
    dist_b1 = abs((bx1 - ax1) * perp[0] + (by1 - ay1) * perp[1]) / len_a
    dist_b2 = abs((bx2 - ax1) * perp[0] + (by2 - ay1) * perp[1]) / len_a
    if dist_b1 > tol or dist_b2 > tol:
        return 0.0
    t_b1 = ((bx1 - ax1) * dx_a + (by1 - ay1) * dy_a) / len_a_sq
    t_b2 = ((bx2 - ax1) * dx_a + (by2 - ay1) * dy_a) / len_a_sq
    lo, hi = min(t_b1, t_b2), max(t_b1, t_b2)
    overlap = max(0, min(1.0, hi) - max(0.0, lo))
    return overlap


def _sides_attached_to_upper(
    sec: Dict[str, Any],
    upper_floor_sections: List[Dict[str, Any]],
    same_floor_sections: Optional[List[Dict[str, Any]]] = None,
    overlap_thresh: float = 0.9,
) -> List[int]:
    """Indicii laturilor (0=top, 1=right, 2=bottom, 3=left) care sunt lipite de etajul superior sau de alt dreptunghi din același etaj."""
    br = sec.get("bounding_rect") or []
    if len(br) < 3:
        return []
    xs = [float(p[0]) for p in br]
    ys = [float(p[1]) for p in br]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    our_edges = [
        (minx, miny, maxx, miny),
        (maxx, miny, maxx, maxy),
        (maxx, maxy, minx, maxy),
        (minx, maxy, minx, miny),
    ]
    other_sections = list(upper_floor_sections)
    for osec in same_floor_sections or []:
        if osec is not sec:
            other_sections.append(osec)
    attached = []
    for ei, (ex1, ey1, ex2, ey2) in enumerate(our_edges):
        our_len = ((ex2 - ex1) ** 2 + (ey2 - ey1) ** 2) ** 0.5
        if our_len < 1e-9:
            continue
        max_overlap = 0.0
        for usec in other_sections:
            ubr = usec.get("bounding_rect") or []
            if len(ubr) < 3:
                continue
            uxs = [float(p[0]) for p in ubr]
            uys = [float(p[1]) for p in ubr]
            uminx, umaxx, uminy, umaxy = min(uxs), max(uxs), min(uys), max(uys)
            u_edges = [
                (uminx, uminy, umaxx, uminy),
                (umaxx, uminy, umaxx, umaxy),
                (umaxx, umaxy, uminx, umaxy),
                (uminx, umaxy, uminx, uminy),
            ]
            for ue in u_edges:
                r = _edge_overlap(ex1, ey1, ex2, ey2, ue[0], ue[1], ue[2], ue[3])
                max_overlap = max(max_overlap, r)
        if max_overlap >= overlap_thresh:
            attached.append(ei)
    return attached


def _get_ridge_intersections_on_segment(
    ma: Tuple[float, float], mb: Tuple[float, float],
    sections: List[Dict[str, Any]], exclude_sec: Optional[Dict[str, Any]] = None,
) -> List[Tuple[float, float]]:
    """Intersecțiile segmentului ma-mb cu ridge-urile altor secțiuni (excluse cele paralele)."""
    pts: List[Tuple[float, float]] = []

    def _seg_intersect(a1: Tuple[float, float], a2: Tuple[float, float],
                       b1: Tuple[float, float], b2: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        ax1, ay1 = a1[0], a1[1]
        ax2, ay2 = a2[0], a2[1]
        bx1, by1 = b1[0], b1[1]
        bx2, by2 = b2[0], b2[1]
        d = (ax2 - ax1) * (by2 - by1) - (ay2 - ay1) * (bx2 - bx1)
        if abs(d) < 1e-12:
            return None
        t = ((bx1 - ax1) * (by2 - by1) - (by1 - ay1) * (bx2 - bx1)) / d
        u = ((ax2 - ax1) * (by1 - ay1) - (ay2 - ay1) * (bx1 - ax1)) / d
        if 0 <= t <= 1 and 0 <= u <= 1:
            return (ax1 + t * (ax2 - ax1), ay1 + t * (ay2 - ay1))
        return None

    for sec in sections:
        if sec is exclude_sec:
            continue
        ridge = sec.get("ridge_line") or []
        if len(ridge) < 2:
            continue
        sa = (float(ridge[0][0]), float(ridge[0][1]))
        sb = (float(ridge[1][0]), float(ridge[1][1]))
        pt = _seg_intersect(ma, mb, sa, sb)
        if pt is not None:
            pts.append(pt)
    return pts


def _get_ridge_clamp_midpoints(
    sec: Dict[str, Any], sections: List[Dict[str, Any]],
    rminx: float, rmaxx: float, rminy: float, rmaxy: float,
    orient: str, ridge_y: Optional[float], ridge_x: Optional[float],
) -> Tuple[float, float, float, float, Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Pentru ridge cu intersecții: mid_x/mid_y devin mijlocul segmentului de la capăt la prima intersecție.
    Returnează (left_mid_x, right_mid_x, top_mid_y, bot_mid_y, first_right_x, first_left_x, first_down_y, first_up_y).
    Ultimele 4 sunt None când nu există intersecții – folosite pentru clamp ca diagonalele să nu traverseze ridge-uri.
    """
    ridge = sec.get("ridge_line") or []
    if len(ridge) < 2:
        mid_x = (rminx + rmaxx) / 2.0
        mid_y = (rminy + rmaxy) / 2.0
        return mid_x, mid_x, mid_y, mid_y, None, None, None, None

    ma = (float(ridge[0][0]), float(ridge[0][1]))
    mb = (float(ridge[1][0]), float(ridge[1][1]))
    ints = _get_ridge_intersections_on_segment(ma, mb, sections, exclude_sec=sec)

    if orient == "horizontal":
        ints_x = [p[0] for p in ints if rminx < p[0] < rmaxx]
        ints_x.sort()
        if not ints_x:
            mid = (rminx + rmaxx) / 2.0
            return mid, mid, (rminy + rmaxy) / 2.0, (rminy + rmaxy) / 2.0, None, None, None, None
        first_right = ints_x[0]
        first_left = ints_x[-1]
        left_mid_x = (rminx + first_right) / 2.0
        right_mid_x = (rmaxx + first_left) / 2.0
        return left_mid_x, right_mid_x, (rminy + rmaxy) / 2.0, (rminy + rmaxy) / 2.0, first_right, first_left, None, None
    else:
        ints_y = [p[1] for p in ints if rminy < p[1] < rmaxy]
        ints_y.sort()
        if not ints_y:
            mid = (rminy + rmaxy) / 2.0
            return (rminx + rmaxx) / 2.0, (rminx + rmaxx) / 2.0, mid, mid, None, None, None, None
        first_down = ints_y[0]
        first_up = ints_y[-1]
        top_mid_y = (rminy + first_down) / 2.0
        bot_mid_y = (rmaxy + first_up) / 2.0
        return (rminx + rmaxx) / 2.0, (rminx + rmaxx) / 2.0, top_mid_y, bot_mid_y, None, None, first_down, first_up


def _get_pyramid_diagonal_segments(
    sections: List[Dict[str, Any]],
    upper_floor_sections: Optional[List[Dict[str, Any]]] = None,
    shorten_to_midpoint: bool = False,
) -> List[List[List[float]]]:
    """
    Diagonale piramidă 4 ape: din colțuri către ridge, max 45°.
    Folosim ridge extins (poate ieși din dreptunghi) – diagonalele pot ieși din dreptunghi până la jumătate până la intersecție.
    shorten_to_midpoint: pentru 4.5_w, desenăm doar jumătatea de la colț la mijlocul diagonalei (nu de la mijloc la ridge).
    """
    from roof_calc.overhang import extend_secondary_sections_to_main_ridge

    upper = upper_floor_sections or []
    extended = extend_secondary_sections_to_main_ridge(sections)

    segs: List[List[List[float]]] = []
    for sec, sec_ext in zip(sections, extended):
        br = sec.get("bounding_rect") or []
        ridge = sec_ext.get("ridge_line") or []
        if len(br) < 3 or len(ridge) < 2:
            continue

        orient = str(sec.get("ridge_orientation", "horizontal"))
        r0 = (float(ridge[0][0]), float(ridge[0][1]))
        r1 = (float(ridge[1][0]), float(ridge[1][1]))
        rminx, rmaxx = min(r0[0], r1[0]), max(r0[0], r1[0])
        rminy, rmaxy = min(r0[1], r1[1]), max(r0[1], r1[1])
        mid_x = (rminx + rmaxx) / 2.0
        mid_y = (rminy + rmaxy) / 2.0
        ridge_y = (r0[1] + r1[1]) / 2.0 if orient == "horizontal" else None
        ridge_x = (r0[0] + r1[0]) / 2.0 if orient == "vertical" else None

        attached_sides = _sides_attached_to_upper(sec, upper, same_floor_sections=sections)
        (
            left_mid_x, right_mid_x, top_mid_y, bot_mid_y,
            first_right_x, first_left_x, first_down_y, first_up_y,
        ) = _get_ridge_clamp_midpoints(
            sec_ext, extended, rminx, rmaxx, rminy, rmaxy, orient, ridge_y, ridge_x
        )

        xs = [float(p[0]) for p in br]
        ys = [float(p[1]) for p in br]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
        corner_to_sides = [(3, 0), (0, 1), (1, 2), (2, 3)]

        for ci, (cx, cy) in enumerate(corners):
            s1, s2 = corner_to_sides[ci]
            if s1 in attached_sides or s2 in attached_sides:
                continue

            if orient == "horizontal":
                if ridge_y is None:
                    continue
                dy = ridge_y - cy
                dx = abs(dy)
                end_y = ridge_y
                ridge_center_x = (rminx + rmaxx) / 2.0
                # 45°: capătul pe ridge; nu trecem niciodată de bulina galbenă; în unele cazuri putem opri mai aproape (scurtăm ridge-ul).
                corner_on_left = abs(cx - minx) < 1e-6
                if corner_on_left:
                    end_x = (cx - dx) if rminx < cx else (cx + dx)
                else:
                    end_x = (cx + dx) if rmaxx > cx else (cx - dx)
                end_x = max(rminx, min(rmaxx, end_x))
                end_x = min(end_x, ridge_center_x) if corner_on_left else max(end_x, ridge_center_x)  # oprire la galben
                if first_left_x is not None and cx <= first_left_x:
                    end_x = min(end_x, first_left_x)
                if first_right_x is not None and cx >= first_right_x:
                    end_x = max(end_x, first_right_x)
            else:
                if ridge_x is None:
                    continue
                dx = ridge_x - cx
                dy = abs(dx)
                end_x = ridge_x
                ridge_center_y = (rminy + rmaxy) / 2.0
                # 45°: capătul pe ridge; nu trecem niciodată de bulina galbenă; în unele cazuri putem opri mai aproape (scurtăm ridge-ul).
                corner_on_top = abs(cy - miny) < 1e-6
                if corner_on_top:
                    end_y = (cy - dy) if rminy < cy else (cy + dy)
                else:
                    end_y = (cy + dy) if rmaxy > cy else (cy - dy)
                end_y = max(rminy, min(rmaxy, end_y))
                end_y = min(end_y, ridge_center_y) if corner_on_top else max(end_y, ridge_center_y)  # oprire la galben
                if first_down_y is not None and cy <= first_down_y:
                    end_y = min(end_y, first_down_y)
                if first_up_y is not None and cy >= first_up_y:
                    end_y = max(end_y, first_up_y)

            if shorten_to_midpoint:
                # 4.5_w: folosim jumătatea de la COLȚ la mijlocul diagonalei (nu de la mijloc la ridge)
                mx = (cx + end_x) / 2.0
                my = (cy + end_y) / 2.0
                segs.append([[cx, cy], [mx, my]])
            else:
                segs.append([[cx, cy], [end_x, end_y]])
    return segs


def _get_contour_segments(mask: np.ndarray) -> List[List[List[float]]]:
    segs: List[List[List[float]]] = []
    for pts in _contours_from_mask(mask):
        for i in range(len(pts) - 1):
            segs.append([[float(pts[i][0]), float(pts[i][1])], [float(pts[i + 1][0]), float(pts[i + 1][1])]])
    return segs


def _get_contour_segments_from_sections(sections: List[Dict[str, Any]], shape: Tuple[int, int]) -> List[List[List[float]]]:
    """Contur exterior din dreptunghiurile curente (nu din masca completă a etajului)."""
    from roof_calc.masks import generate_binary_mask

    rects = []
    for sec in sections:
        poly = _rect_polygon(sec)
        if poly is not None and not poly.is_empty:
            rects.append(poly)
    if not rects:
        return []
    mask = generate_binary_mask(rects, shape)
    return _get_contour_segments(mask)


def _segment_match(a0: Tuple[float, float], a1: Tuple[float, float], b0: Tuple[float, float], b1: Tuple[float, float], tol: float = 1.0) -> bool:
    """True dacă segmentul (a0,a1) coincide cu (b0,b1) sau (b1,b0)."""
    def eq(p, q):
        return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= tol * tol
    return (eq(a0, b0) and eq(a1, b1)) or (eq(a0, b1) and eq(a1, b0))


def _chain_segments_to_rings(
    segments: List[List[List[float]]],
    tol: float = 1e-6,
) -> List[List[Tuple[float, float]]]:
    """
    Lanțuie segmente în unul sau mai multe inele închise (contur exterior cu oricâte colțuri).
    Returnează listă de ring-uri; fiecare ring = listă de (x,y), fără repetarea primului la final.
    """
    if not segments:
        return []
    segs = []
    for s in segments:
        if len(s) < 2:
            continue
        p0 = (float(s[0][0]), float(s[0][1]))
        p1 = (float(s[1][0]), float(s[1][1]))
        if (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 < 1e-20:
            continue
        segs.append((p0, p1))
    if not segs:
        return []
    used = [False] * len(segs)
    rings: List[List[Tuple[float, float]]] = []

    def point_eq(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= tol * tol

    for start_idx in range(len(segs)):
        if used[start_idx]:
            continue
        p0, p1 = segs[start_idx]
        ring = [p0, p1]
        used[start_idx] = True
        current = p1
        while True:
            found = False
            for i, (a, b) in enumerate(segs):
                if used[i]:
                    continue
                if point_eq(current, a):
                    ring.append(b)
                    current = b
                    used[i] = True
                    found = True
                    break
                if point_eq(current, b):
                    ring.append(a)
                    current = a
                    used[i] = True
                    found = True
                    break
            if not found:
                break
            if point_eq(current, ring[0]) and len(ring) >= 3:
                ring.pop()
                rings.append(ring)
                break
        if not found:
            break
        if point_eq(current, ring[0]) and len(ring) >= 3:
            break
        if len(ring) >= 3 and point_eq(ring[-1], ring[0]):
            ring.pop()
            rings.append(ring)
    return rings


def _chain_45w_green_diag_orange_diag_green(
    seg_green: List[List[List[float]]],
    seg_orange: Optional[List[List[List[float]]]],
    seg_pyramid: List[List[List[float]]],
    tol: float = 2.0,
) -> Optional[List[Tuple[float, float]]]:
    """
    Lanț ordonat pentru baza 4.5_w: segmente VERZI → diagonală până la paralelă → PORTOCALIU (paralela) →
    diagonală până la următorul verde → VERDE. Returnează un ring închis sau None. Începe întotdeauna de la verde.
    """
    def pt_eq(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= tol * tol

    def seg_to_pair(s: List[List[float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        if len(s) < 2:
            return ((0.0, 0.0), (0.0, 0.0))
        return ((float(s[0][0]), float(s[0][1])), (float(s[1][0]), float(s[1][1])))

    greens = [seg_to_pair(s) for s in (seg_green or []) if len(s) >= 2]
    oranges = [seg_to_pair(s) for s in (seg_orange or []) if len(s) >= 2]
    pyramids = [seg_to_pair(s) for s in (seg_pyramid or []) if len(s) >= 2]
    if not greens:
        return None

    used_g = [False] * len(greens)
    used_o = [False] * len(oranges)
    used_p = [False] * len(pyramids)

    def find_green_from(pt: Tuple[float, float]) -> Optional[Tuple[Tuple[float, float], int]]:
        for i, (a, b) in enumerate(greens):
            if used_g[i]:
                continue
            if pt_eq(pt, a):
                return (b, i)
            if pt_eq(pt, b):
                return (a, i)
        return None

    def find_pyramid_from_to(pt: Tuple[float, float], target_near: List[Tuple[float, float]]) -> Optional[Tuple[Tuple[float, float], int]]:
        for i, (a, b) in enumerate(pyramids):
            if used_p[i]:
                continue
            if not pt_eq(pt, a) and not pt_eq(pt, b):
                continue
            other = b if pt_eq(pt, a) else a
            for t in target_near:
                if pt_eq(other, t):
                    return (other, i)
        return None

    def find_pyramid_from_to_any(pt: Tuple[float, float]) -> Optional[Tuple[Tuple[float, float], int]]:
        """Pyramid segment that has pt at one end; return the other end."""
        for i, (a, b) in enumerate(pyramids):
            if used_p[i]:
                continue
            if pt_eq(pt, a):
                return (b, i)
            if pt_eq(pt, b):
                return (a, i)
        return None

    orange_ends = []
    for (a, b) in oranges:
        orange_ends.append(a)
        orange_ends.append(b)

    def find_orange_from(pt: Tuple[float, float]) -> Optional[Tuple[Tuple[float, float], int]]:
        for i, (a, b) in enumerate(oranges):
            if used_o[i]:
                continue
            if pt_eq(pt, a):
                return (b, i)
            if pt_eq(pt, b):
                return (a, i)
        return None

    def try_build_ring(start_seg_idx: int, reverse: bool) -> Optional[List[Tuple[float, float]]]:
        used_g[:] = [False] * len(greens)
        used_o[:] = [False] * len(oranges)
        used_p[:] = [False] * len(pyramids)
        a, b = greens[start_seg_idx]
        ring: List[Tuple[float, float]] = [a, b] if not reverse else [b, a]
        current = ring[-1]
        used_g[start_seg_idx] = True

        while True:
            if pt_eq(current, ring[0]) and len(ring) >= 3:
                if len(ring) > 1 and pt_eq(ring[-1], ring[0]):
                    return ring[:-1]
                return ring

            # 1) Prefer continuare pe VERDE
            nxt = find_green_from(current)
            if nxt is not None:
                other, idx = nxt
                ring.append(other)
                current = other
                used_g[idx] = True
                continue

            # 2) Nu mai e verde: mergi pe diagonală până la paralelă, apoi paralelă, apoi diagonală la următorul verde
            pyramid_out = find_pyramid_from_to(current, orange_ends)
            if pyramid_out is None:
                pyramid_out = find_pyramid_from_to_any(current)
            if pyramid_out is None:
                if pt_eq(current, ring[0]) and len(ring) >= 3:
                    if len(ring) > 1 and pt_eq(ring[-1], ring[0]):
                        return ring[:-1]
                    return ring
                return None
            orange_pt, p_idx = pyramid_out
            ring.append(orange_pt)
            used_p[p_idx] = True
            current = orange_pt

            orange_seg = find_orange_from(current)
            if orange_seg is not None:
                other_o, o_idx = orange_seg
                ring.append(other_o)
                used_o[o_idx] = True
                current = other_o

            pyramid_back = find_pyramid_from_to_any(current)
            if pyramid_back is None:
                return None
            green_pt, p2_idx = pyramid_back
            ring.append(green_pt)
            used_p[p2_idx] = True
            current = green_pt

    for start_idx in range(len(greens)):
        for rev in [False, True]:
            ring = try_build_ring(start_idx, rev)
            if ring is not None and len(ring) >= 3:
                return ring
    return None


def _build_base_segments_45w(
    seg_contour: List[List[List[float]]],
    seg_orange: Optional[List[List[List[float]]]],
    seg_pyramid: List[List[List[float]]],
) -> List[List[List[float]]]:
    """
    Baza pentru 4.5_w: ÎNCEPI cu segmentele VERZI. Când nu mai ai segment verde, mergi pe diagonală
    până la paralela între diagonale (portocaliu), apoi pe diagonală până la următorul segment verde.
    Ordine: verde → diagonală → portocaliu → diagonală → verde. Fără linii mov.
    """
    def ring_to_segments(ring: List[Tuple[float, float]]) -> List[List[List[float]]]:
        if len(ring) < 3:
            return []
        out = []
        for i in range(len(ring)):
            a, b = ring[i], ring[(i + 1) % len(ring)]
            out.append([[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]])
        return out

    tol = 1.0
    # Ordine explicită: verde → diagonală → portocaliu → diagonală → verde (începe de la verde)
    if seg_contour and (seg_orange or seg_pyramid):
        ordered_ring = _chain_45w_green_diag_orange_diag_green(
            seg_contour, seg_orange, seg_pyramid, tol=2.0
        )
        if ordered_ring and len(ordered_ring) >= 3:
            return ring_to_segments(ordered_ring)
    # Fallback: doar verzi
    if seg_contour and len(seg_contour) >= 3:
        rings = _chain_segments_to_rings(seg_contour, tol=tol)
        if rings and len(rings[0]) >= 3:
            return ring_to_segments(rings[0])
    # Fallback: verzi + portocaliu
    seg_go = list(seg_contour) if seg_contour else []
    if seg_orange:
        seg_go = seg_go + list(seg_orange)
    if len(seg_go) >= 3:
        rings = _chain_segments_to_rings(seg_go, tol=tol)
        if rings and len(rings[0]) >= 3:
            return ring_to_segments(rings[0])
    # Fallback: verzi + portocaliu + diagonale (polygonize) – apoi ROTIM ring-ul ca primul segment să fie VERDE
    seg_all = list(seg_go) + list(seg_pyramid) if seg_pyramid else seg_go
    if len(seg_all) < 3:
        return list(seg_contour) + list(seg_orange or []) + list(seg_pyramid)
    try:
        from shapely.geometry import LineString
        from shapely.ops import polygonize_full, unary_union
        lines = []
        for s in seg_all:
            if len(s) >= 2 and (float(s[0][0]) - float(s[1][0])) ** 2 + (float(s[0][1]) - float(s[1][1])) ** 2 > 1e-10:
                lines.append(LineString([(float(s[0][0]), float(s[0][1])), (float(s[1][0]), float(s[1][1]))]))
        if len(lines) < 3:
            return seg_all
        polys, _, _, _ = polygonize_full(lines)
        geoms = list(getattr(polys, "geoms", None) or [polys])
        valid = [g for g in geoms if g is not None and not getattr(g, "is_empty", True)]
        if not valid:
            return seg_all
        main = max(valid, key=lambda g: getattr(g, "area", 0) or 0)
        ext = getattr(main, "exterior", None)
        if ext is None:
            return seg_all
        coords = list(getattr(ext, "coords", []))
        if len(coords) < 3:
            return seg_all
        # Începem de la segment VERDE: rotim ring-ul astfel încât prima latură să fie verde
        green_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for s in (seg_contour or []):
            if len(s) >= 2:
                p0 = (float(s[0][0]), float(s[0][1]))
                p1 = (float(s[1][0]), float(s[1][1]))
                green_edges.append((p0, p1))
        tol_sq = tol * tol
        def pt_eq_sq(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= tol_sq
        def edge_is_green(c0: Tuple[float, float], c1: Tuple[float, float]) -> bool:
            for (a, b) in green_edges:
                if (pt_eq_sq(c0, a) and pt_eq_sq(c1, b)) or (pt_eq_sq(c0, b) and pt_eq_sq(c1, a)):
                    return True
            return False
        n = len(coords)
        rotated = None
        for i in range(n - 1):
            c0, c1 = coords[i], coords[i + 1]
            if edge_is_green(c0, c1):
                rotated = list(coords[i:-1]) + list(coords[0:i])
                break
        if rotated is None:
            rotated = list(coords[:-1])
        out = []
        for i in range(len(rotated)):
            a = rotated[i]
            b = rotated[(i + 1) % len(rotated)]
            out.append([
                [float(a[0]), float(a[1])],
                [float(b[0]), float(b[1])],
            ])
        return out
    except Exception:
        return list(seg_contour) + list(seg_orange or []) + list(seg_pyramid)


def _exterior_segments_of_union(
    segments: List[List[List[float]]],
) -> List[List[List[float]]]:
    """
    Polygonize segmente, unary_union poligoane, returnează conturul exterior al union-ului ca listă de segmente.
    Pentru 4.5_w: un singur contur în jurul tuturor secțiunilor (baze lipite).
    """
    if not segments or len(segments) < 3:
        return []
    try:
        from shapely.geometry import LineString
        from shapely.ops import polygonize_full, unary_union
        lines = []
        for s in segments:
            if len(s) >= 2:
                p0 = (float(s[0][0]), float(s[0][1]))
                p1 = (float(s[1][0]), float(s[1][1]))
                if (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 >= 1e-10:
                    lines.append(LineString([p0, p1]))
        if len(lines) < 3:
            return []
        result = polygonize_full(lines)
        polys = result[0] if isinstance(result, (tuple, list)) and result else result
        if polys is None or getattr(polys, "is_empty", True):
            return []
        geoms = getattr(polys, "geoms", None) or ([polys] if polys and not getattr(polys, "is_empty", True) else [])
        if not geoms:
            return []
        union = unary_union(geoms)
        if union is None or getattr(union, "is_empty", True):
            return []
        if hasattr(union, "geoms") and len(union.geoms) > 1:
            union = unary_union(list(union.geoms))
        ext = getattr(union, "exterior", None)
        if ext is None:
            return []
        coords = list(getattr(ext, "coords", []))
        if len(coords) < 3:
            return []
        out = []
        for i in range(len(coords) - 1):
            out.append([
                [float(coords[i][0]), float(coords[i][1])],
                [float(coords[i + 1][0]), float(coords[i + 1][1])],
            ])
        return out
    except Exception:
        return []


def _exterior_segments_per_polygon(
    segments: List[List[List[float]]],
) -> List[List[List[List[float]]]]:
    """
    Polygonize segmente; returnează o listă de contururi (fiecare = listă de segmente),
    câte un contur exterior per poligon. Nu face union — poligoane disjuncte = contururi separate.
    """
    if not segments or len(segments) < 3:
        return []
    try:
        from shapely.geometry import LineString
        from shapely.ops import polygonize_full
        lines = []
        for s in segments:
            if len(s) >= 2:
                p0 = (float(s[0][0]), float(s[0][1]))
                p1 = (float(s[1][0]), float(s[1][1]))
                if (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 >= 1e-10:
                    lines.append(LineString([p0, p1]))
        if len(lines) < 3:
            return []
        result = polygonize_full(lines)
        polys = result[0] if isinstance(result, (tuple, list)) and result else result
        if polys is None or getattr(polys, "is_empty", True):
            return []
        geoms = getattr(polys, "geoms", None) or ([polys] if polys and not getattr(polys, "is_empty", True) else [])
        out_list: List[List[List[List[float]]]] = []
        for poly in geoms:
            if poly is None or getattr(poly, "is_empty", True):
                continue
            ext = getattr(poly, "exterior", None)
            if ext is None:
                continue
            coords = list(getattr(ext, "coords", []))
            if len(coords) < 3:
                continue
            contour = []
            for i in range(len(coords) - 1):
                contour.append([
                    [float(coords[i][0]), float(coords[i][1])],
                    [float(coords[i + 1][0]), float(coords[i + 1][1])],
                ])
            if contour:
                out_list.append(contour)
        return out_list
    except Exception:
        return []


def _overhang_segments_from_contour(
    segments_contour: List[List[List[float]]],
    mpp: float,
    overhang_meters: float = 1.0,
    contour_only_segments: Optional[List[List[List[float]]]] = None,
) -> List[List[List[float]]]:
    """
    Prelungire contur exterior cu overhang_meters (default 1 m).
    Construiește poligoane din segmente (contur + eventual diagonale + paralelă), aplică buffer în afară,
    apoi extrage conturul buffered ca segmente.
    Dacă contour_only_segments e dat, păstrăm doar poligoanele ale căror frontieră conține
    cel puțin un segment din listă (ex.: doar conturul verde) – astfel excludem poligoane generate
    doar din paralelă+diagonale (nu facem overhang suplimentar pentru paralelă).
    """
    if not segments_contour or mpp <= 0 or overhang_meters <= 0:
        return []
    offset_px = float(overhang_meters) / float(mpp)
    out_segs: List[List[List[float]]] = []

    # Cale robustă pentru contur exterior cu >4 laturi: poligon din segmente lanțuite → buffer.
    if contour_only_segments and len(contour_only_segments) >= 3:
        rings = _chain_segments_to_rings(contour_only_segments, tol=1.0)
        if rings:
            try:
                from shapely.geometry import Polygon as ShapelyPolygon
                from shapely.ops import unary_union as _shapely_union
                # Construim poligoane valide din TOATE inelele, nu doar cel mai mare
                valid_polys = []
                for r in rings:
                    if len(r) >= 3:
                        try:
                            p = ShapelyPolygon(r)
                            if not getattr(p, "is_empty", True):
                                if not p.is_valid:
                                    p = p.buffer(0)
                                if not getattr(p, "is_empty", True):
                                    valid_polys.append(p)
                        except Exception:
                            pass
                if valid_polys:
                    merged = _shapely_union(valid_polys) if len(valid_polys) > 1 else valid_polys[0]
                    if not getattr(merged, "is_empty", True):
                        buffered = merged.buffer(offset_px, resolution=2, join_style=2)
                        if buffered and not getattr(buffered, "is_empty", True):
                            # Poate rezulta MultiPolygon dacă dreptunghiurile nu se ating — unim
                            if hasattr(buffered, "geoms"):
                                buffered = _shapely_union(list(buffered.geoms))
                            ext = getattr(buffered, "exterior", None)
                            if ext is not None:
                                coords = list(getattr(ext, "coords", []))
                                for i in range(len(coords) - 1):
                                    out_segs.append([
                                        [float(coords[i][0]), float(coords[i][1])],
                                        [float(coords[i + 1][0]), float(coords[i + 1][1])],
                                    ])
                                if out_segs:
                                    return out_segs
            except Exception:
                pass

    try:
        from shapely.geometry import LineString
        from shapely.ops import polygonize_full

        lines = []
        for seg in segments_contour:
            if len(seg) >= 2:
                p0 = (float(seg[0][0]), float(seg[0][1]))
                p1 = (float(seg[1][0]), float(seg[1][1]))
                if (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 >= 1e-10:
                    lines.append(LineString([p0, p1]))
        if not lines:
            return []
        result = polygonize_full(lines)
        polygons = result[0] if isinstance(result, (tuple, list)) and result else result
        if polygons is None or getattr(polygons, "is_empty", True):
            return []
        geoms = getattr(polygons, "geoms", None) or (
            [polygons] if polygons and not getattr(polygons, "is_empty", True) else []
        )
        # Segmente verzi/portocalii pentru filtrare (contur verde SAU paralelă – păstrăm ambele tipuri ca să includem și portocaliul)
        green_edges: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None
        if contour_only_segments:
            green_edges = []
            for s in contour_only_segments:
                if len(s) >= 2:
                    a = (float(s[0][0]), float(s[0][1]))
                    b = (float(s[1][0]), float(s[1][1]))
                    if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 >= 1e-10:
                        green_edges.append((a, b))

        kept_polys: List[Any] = []
        for poly in geoms:
            if poly is None or getattr(poly, "is_empty", True):
                continue
            if green_edges is not None:
                ext = getattr(poly, "exterior", None)
                if ext is None:
                    continue
                coords_poly = list(getattr(ext, "coords", []))
                has_allowed = False
                for i in range(len(coords_poly) - 1):
                    pa = (float(coords_poly[i][0]), float(coords_poly[i][1]))
                    pb = (float(coords_poly[i + 1][0]), float(coords_poly[i + 1][1]))
                    for (ga, gb) in green_edges:
                        if _segment_match(pa, pb, ga, gb):
                            has_allowed = True
                            break
                    if has_allowed:
                        break
                if not has_allowed:
                    continue
            kept_polys.append(poly)
        # Un singur overhang: unim toate poligoanele păstrate (verde + portocaliu) înainte de buffer
        if not kept_polys:
            return []
        try:
            from shapely.ops import unary_union
            union = unary_union(kept_polys)
            if union is None or getattr(union, "is_empty", True):
                return []
            # Pentru dreptunghiuri lipite: dacă avem MultiPolygon, conectăm părțile cu un buffer mic
            # astfel încât overhang-ul să fie un singur contur în jurul tuturor.
            to_buffer = union
            if hasattr(union, "geoms") and len(union.geoms) > 1:
                try:
                    # buffer mic + negative buffer unește poligoanele apropiate/lipite
                    merge_tol = max(1.0, offset_px * 0.5)
                    merged = union.buffer(merge_tol, resolution=2, join_style=2)
                    if merged and not getattr(merged, "is_empty", True):
                        merged = merged.buffer(-merge_tol, resolution=2, join_style=2)
                        if merged and not getattr(merged, "is_empty", True):
                            to_buffer = merged
                except Exception:
                    pass
            buffered = to_buffer.buffer(offset_px, resolution=2, join_style=2)
            # Dacă buffer-ul dă MultiPolygon (dreptunghiuri fără contact direct), unim rezultatul
            if buffered is not None and not getattr(buffered, "is_empty", True) and hasattr(buffered, "geoms"):
                buffered = unary_union(list(buffered.geoms))
            if buffered is None or getattr(buffered, "is_empty", True):
                return []
            ext = getattr(buffered, "exterior", None)
            if ext is not None:
                coords = list(getattr(ext, "coords", []))
                for i in range(len(coords) - 1):
                    out_segs.append([
                        [float(coords[i][0]), float(coords[i][1])],
                        [float(coords[i + 1][0]), float(coords[i + 1][1])],
                    ])
            elif hasattr(buffered, "geoms"):
                for g in buffered.geoms:
                    e = getattr(g, "exterior", None)
                    if e is None:
                        continue
                    coords = list(getattr(e, "coords", []))
                    for i in range(len(coords) - 1):
                        out_segs.append([
                            [float(coords[i][0]), float(coords[i][1])],
                            [float(coords[i + 1][0]), float(coords[i + 1][1])],
                        ])
        except Exception:
            # Fallback: buffer fiecare poligon și colectează segmente (comportament vechi)
            for poly in kept_polys:
                try:
                    buffered = poly.buffer(offset_px, resolution=2, join_style=2)
                    if buffered is None or getattr(buffered, "is_empty", True):
                        continue
                    ext = getattr(buffered, "exterior", None)
                    if ext is None:
                        continue
                    coords = list(getattr(ext, "coords", []))
                    if len(coords) < 3:
                        continue
                    for i in range(len(coords) - 1):
                        out_segs.append([
                            [float(coords[i][0]), float(coords[i][1])],
                            [float(coords[i + 1][0]), float(coords[i + 1][1])],
                        ])
                except Exception:
                    continue
    except Exception:
        pass
    if out_segs:
        return out_segs
    fallback = _overhang_segments_from_contour_fallback(segments_contour, offset_px)
    return fallback


def _overhang_segments_from_contour_fallback(
    segments_contour: List[List[List[float]]],
    offset_px: float,
) -> List[List[List[float]]]:
    """Fallback fără Shapely: offset per segment cu unire la colțuri prin intersecție."""
    if not segments_contour or offset_px <= 0:
        return []
    segs = [s for s in segments_contour if len(s) >= 2]
    if len(segs) < 2:
        return []
    all_pts = [(float(s[0][0]), float(s[0][1])) for s in segs]
    cx = sum(p[0] for p in all_pts) / len(all_pts)
    cy = sum(p[1] for p in all_pts) / len(all_pts)
    normals: List[Tuple[float, float]] = []
    for seg in segs:
        ax, ay = float(seg[0][0]), float(seg[0][1])
        bx, by = float(seg[1][0]), float(seg[1][1])
        dx, dy = bx - ax, by - ay
        L = (dx * dx + dy * dy) ** 0.5
        if L < 1e-10:
            normals.append((0.0, 0.0))
            continue
        perp_x = -dy / L
        perp_y = dx / L
        mid_x = (ax + bx) * 0.5
        mid_y = (ay + by) * 0.5
        sign = 1.0 if ((mid_x - cx) * perp_x + (mid_y - cy) * perp_y) >= 0 else -1.0
        normals.append((sign * perp_x * offset_px, sign * perp_y * offset_px))
    out_segs = []
    n = len(segs)
    for i in range(n):
        seg = segs[i]
        seg_prev = segs[(i - 1) % n]
        n0 = normals[i]
        n_prev = normals[(i - 1) % n]
        n1 = normals[(i + 1) % n]
        ax, ay = float(seg[0][0]), float(seg[0][1])
        bx, by = float(seg[1][0]), float(seg[1][1])
        ax_prev, ay_prev = float(seg_prev[0][0]), float(seg_prev[0][1])
        p0 = (ax + n0[0], ay + n0[1])
        p1 = (bx + n0[0], by + n0[1])
        p2 = (bx + n1[0], by + n1[1])
        p_prev_a = (ax_prev + n_prev[0], ay_prev + n_prev[1])
        p_prev_b = (ax + n_prev[0], ay + n_prev[1])
        ix_curr, iy_curr = _segment_intersection_2d(p0, p1, p1, p2)
        ix_prev, iy_prev = _segment_intersection_2d(p_prev_a, p_prev_b, p0, p1)
        if ix_curr is not None and ix_prev is not None:
            out_segs.append([[ix_prev, iy_prev], [ix_curr, iy_curr]])
        elif ix_curr is not None:
            out_segs.append([[p0[0], p0[1]], [ix_curr, iy_curr]])
        else:
            out_segs.append([[p0[0], p0[1]], [p1[0], p1[1]]])
    return out_segs


def _segment_intersection_2d(
    a1: Tuple[float, float], a2: Tuple[float, float],
    b1: Tuple[float, float], b2: Tuple[float, float],
    tol: float = 1e-9,
) -> Tuple[Optional[float], Optional[float]]:
    """Intersecția a două segmente 2D; returnează (x, y) sau (None, None)."""
    dxa = a2[0] - a1[0]
    dya = a2[1] - a1[1]
    dxb = b2[0] - b1[0]
    dyb = b2[1] - b1[1]
    denom = dxa * dyb - dya * dxb
    if abs(denom) < tol:
        return (None, None)
    t = ((b1[0] - a1[0]) * dyb - (b1[1] - a1[1]) * dxb) / denom
    s = ((b1[0] - a1[0]) * dya - (b1[1] - a1[1]) * dxa) / denom
    if -tol <= t <= 1 + tol and -tol <= s <= 1 + tol:
        ix = a1[0] + t * dxa
        iy = a1[1] + t * dya
        return (ix, iy)
    return (None, None)


def _merge_overlapping_overhangs(segments_overhang: List[List[List[float]]]) -> List[List[List[float]]]:
    """
    Unifică toate segmentele de overhang într-un singur poligon și returnează doar conturul exterior.
    Un singur overhang per acoperiș: dacă union-ul dă mai multe poligoane disjuncte, păstrăm doar
    poligonul cu aria cea mai mare (conturul principal).
    """
    if not segments_overhang or len(segments_overhang) < 3:
        return segments_overhang
    try:
        from shapely.geometry import LineString
        from shapely.ops import polygonize_full, unary_union

        lines = []
        for seg in segments_overhang:
            if len(seg) >= 2:
                p0 = (float(seg[0][0]), float(seg[0][1]))
                p1 = (float(seg[1][0]), float(seg[1][1]))
                if (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 >= 1e-10:
                    lines.append(LineString([p0, p1]))
        if not lines:
            return segments_overhang
        result = polygonize_full(lines)
        polygons = result[0] if isinstance(result, (tuple, list)) and result else result
        if polygons is None or getattr(polygons, "is_empty", True):
            return segments_overhang
        geoms = getattr(polygons, "geoms", None) or (
            [polygons] if polygons and not getattr(polygons, "is_empty", True) else []
        )
        if not geoms:
            return segments_overhang
        union = unary_union(geoms)
        if union is None or getattr(union, "is_empty", True):
            return segments_overhang
        # Unim TOATE poligoanele într-un singur contur exterior — nu doar cel mai mare.
        # Dreptunghiurile lipite produc MultiPolygon; trebuie unite înainte de a extrage conturul.
        polys_to_consider = (
            list(getattr(union, "geoms", [])) if hasattr(union, "geoms") and len(union.geoms) > 1
            else [union]
        )
        if not polys_to_consider:
            return segments_overhang
        # Unim toate; dacă rămân disjuncte, buffer mic snap topology
        merged_all = unary_union(polys_to_consider) if len(polys_to_consider) > 1 else polys_to_consider[0]
        if hasattr(merged_all, "geoms"):
            # Dreptunghiuri fără contact direct → buffer mic să fuzioneze, apoi negativ să revenim la formă
            merged_all = merged_all.buffer(1.0).buffer(-1.0)
        if hasattr(merged_all, "geoms"):
            # Tot disjuncte → buffer pozitiv fără compensare (overhang mic între ele)
            merged_all = merged_all.buffer(1.0)
        ext = getattr(merged_all, "exterior", None)
        if ext is None:
            return segments_overhang
        coords = list(getattr(ext, "coords", []))
        out_segs: List[List[List[float]]] = []
        for i in range(len(coords) - 1):
            out_segs.append([
                [float(coords[i][0]), float(coords[i][1])],
                [float(coords[i + 1][0]), float(coords[i + 1][1])],
            ])
        if out_segs:
            return out_segs
    except Exception:
        pass
    return segments_overhang


def _filter_overhang_near_base(
    overhang_segments: List[List[List[float]]],
    base_segments: List[List[List[float]]],
    mpp: float,
    max_distance_from_base_m: float = 2.0,
) -> List[List[List[float]]]:
    """
    Păstrează doar segmentele de overhang al căror midpoint e la distanță <= max_distance_from_base_m
    de poligonul bazei (mov). Elimină overhang generat greșit pentru zone fără bază (ex. dreptunghi jos).
    """
    if not overhang_segments or not base_segments or mpp <= 0:
        return overhang_segments
    try:
        from shapely.geometry import LineString, Point
        from shapely.ops import polygonize_full, unary_union
        base_lines = []
        for s in base_segments:
            if len(s) >= 2:
                p0 = (float(s[0][0]), float(s[0][1]))
                p1 = (float(s[1][0]), float(s[1][1]))
                if (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 >= 1e-10:
                    base_lines.append(LineString([p0, p1]))
        if len(base_lines) < 3:
            return overhang_segments
        result = polygonize_full(base_lines)
        polys = result[0] if isinstance(result, (tuple, list)) and result else result
        if polys is None or getattr(polys, "is_empty", True):
            return overhang_segments
        geoms = getattr(polys, "geoms", None) or ([polys] if polys and not getattr(polys, "is_empty", True) else [])
        if not geoms:
            return overhang_segments
        base_union = unary_union(geoms)
        if base_union is None or getattr(base_union, "is_empty", True):
            return overhang_segments
        buffer_px = max_distance_from_base_m / mpp
        allowed_zone = base_union.buffer(buffer_px, resolution=2)
        out = []
        for seg in overhang_segments:
            if len(seg) < 2:
                continue
            mx = (float(seg[0][0]) + float(seg[1][0])) * 0.5
            my = (float(seg[0][1]) + float(seg[1][1])) * 0.5
            pt = Point(mx, my)
            if allowed_zone.contains(pt) or allowed_zone.distance(pt) < 1e-6:
                out.append(seg)
        return out if out else overhang_segments
    except Exception:
        return overhang_segments


def _contour_segments_minus_upper_rect(
    segments: List[List[List[float]]],
    upper_rect_segs: List[List[List[float]]],
    offset_px: float,
) -> List[List[List[float]]]:
    """
    Contur pentru overhang de la început fără zona etajului superior.
    Returnează segmentele care formează frontiera (polygonize(segments) - buffer(upper_rect)).
    Dacă offset_px <= 0 sau upper_rect_segs e gol, returnează segments neschimbat.
    """
    if not segments or not upper_rect_segs or offset_px <= 0:
        return segments
    try:
        from shapely.geometry import LineString
        from shapely.ops import polygonize_full, unary_union

        # Poligon etaj superior, bufferat în afară
        lines_upper = []
        for seg in upper_rect_segs:
            if len(seg) >= 2:
                p0 = (float(seg[0][0]), float(seg[0][1]))
                p1 = (float(seg[1][0]), float(seg[1][1]))
                if (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 >= 1e-10:
                    lines_upper.append(LineString([p0, p1]))
        if not lines_upper:
            return segments
        result = polygonize_full(lines_upper)
        upper_polys = result[0] if isinstance(result, (tuple, list)) and result else result
        if upper_polys is None or getattr(upper_polys, "is_empty", True):
            return segments
        geoms_upper = getattr(upper_polys, "geoms", None) or (
            [upper_polys] if upper_polys and not getattr(upper_polys, "is_empty", True) else []
        )
        upper_union = unary_union(geoms_upper) if geoms_upper else None
        if upper_union is None or getattr(upper_union, "is_empty", True):
            return segments
        try:
            buffered = upper_union.buffer(offset_px, resolution=2, join_style=2)
            if buffered is not None and not getattr(buffered, "is_empty", True):
                upper_union = buffered
        except Exception:
            pass

        # Poligoane din segmente (contur + eventual diagonale/paralelă)
        lines_in = []
        for seg in segments:
            if len(seg) >= 2:
                p0 = (float(seg[0][0]), float(seg[0][1]))
                p1 = (float(seg[1][0]), float(seg[1][1]))
                if (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 >= 1e-10:
                    lines_in.append(LineString([p0, p1]))
        if not lines_in:
            return segments
        result_in = polygonize_full(lines_in)
        polys_in = result_in[0] if isinstance(result_in, (tuple, list)) and result_in else result_in
        if polys_in is None or getattr(polys_in, "is_empty", True):
            return segments
        geoms_in = getattr(polys_in, "geoms", None) or (
            [polys_in] if polys_in and not getattr(polys_in, "is_empty", True) else []
        )
        out_segs: List[List[List[float]]] = []
        for poly in geoms_in:
            if poly is None or getattr(poly, "is_empty", True):
                continue
            try:
                diff = poly.difference(upper_union)
                if diff is None or getattr(diff, "is_empty", True):
                    continue
                to_process = getattr(diff, "geoms", None) or ([diff] if not getattr(diff, "is_empty", True) else [])
                for geom in to_process:
                    if geom is None or getattr(geom, "is_empty", True):
                        continue
                    if hasattr(geom, "exterior") and geom.exterior is not None:
                        coords = list(geom.exterior.coords)
                        for i in range(len(coords) - 1):
                            out_segs.append([
                                [float(coords[i][0]), float(coords[i][1])],
                                [float(coords[i + 1][0]), float(coords[i + 1][1])],
                            ])
            except Exception:
                continue
        if out_segs:
            return out_segs
    except Exception:
        pass
    return segments


def _get_separator_segments(sections: List[Dict[str, Any]]) -> List[List[List[float]]]:
    """
    Segmente care despart dreptunghiuri lipite (același etaj): muchia comună între două
    bounding_rect care se ating. Permite două fețe separate și trasarea liniei de despărțire.
    """
    out: List[List[List[float]]] = []
    polys: List[Any] = []
    for sec in sections:
        poly = _rect_polygon(sec)
        if poly is not None and not poly.is_empty:
            polys.append(poly)
        else:
            polys.append(None)
    tol = 1e-6
    seen: set = set()
    for i in range(len(polys)):
        if polys[i] is None:
            continue
        for j in range(i + 1, len(polys)):
            if polys[j] is None:
                continue
            try:
                bi = getattr(polys[i], "boundary", None)
                bj = getattr(polys[j], "boundary", None)
                if bi is None or bj is None:
                    continue
                inter = bi.intersection(bj)
                if inter is None or getattr(inter, "is_empty", True):
                    continue
                geoms = getattr(inter, "geoms", None) or ([inter] if inter else [])
                for g in geoms:
                    coords = list(getattr(g, "coords", []))
                    if len(coords) < 2:
                        continue
                    p0 = (round(float(coords[0][0]), 6), round(float(coords[0][1]), 6))
                    p1 = (round(float(coords[-1][0]), 6), round(float(coords[-1][1]), 6))
                    if (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 < tol * tol:
                        continue
                    key = (p0, p1) if p0 < p1 else (p1, p0)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append([[float(coords[0][0]), float(coords[0][1])], [float(coords[-1][0]), float(coords[-1][1])]])
            except Exception:
                pass
    return out


def _split_brown_segments_no_containment(
    brown_list: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    tol: float = 1e-6,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Sparge segmentele astfel încât niciun segment să nu conțină alte segmente în interior.
    Dacă (c1,c2) conține capătul altui segment în interior, îl împarte acolo.
    """
    if not brown_list:
        return []
    result: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for c1, c2 in brown_list:
        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]
        len_sq = dx * dx + dy * dy
        if len_sq < 1e-12:
            continue

        def _t_on_seg(p: Tuple[float, float]) -> float:
            return ((p[0] - c1[0]) * dx + (p[1] - c1[1]) * dy) / len_sq

        def _collinear(p: Tuple[float, float]) -> bool:
            cross = (p[0] - c1[0]) * dy - (p[1] - c1[1]) * dx
            return abs(cross) < tol * (len_sq ** 0.5 + 1)

        interior_ts: List[float] = []
        for oc1, oc2 in brown_list:
            if (oc1, oc2) == (c1, c2):
                continue
            for p in (oc1, oc2):
                if not _collinear(p):
                    continue
                t = _t_on_seg(p)
                if tol < t < 1 - tol:
                    interior_ts.append(t)
        interior_ts = sorted(set(interior_ts))
        pts = [c1] + [
            (c1[0] + t * dx, c1[1] + t * dy) for t in interior_ts
        ] + [c2]
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 >= tol * tol:
                result.append((a, b))
    return result


def _segment_overlaps_brown(
    green_seg: List[List[float]],
    brown_p1: Tuple[float, float],
    brown_p2: Tuple[float, float],
    tol: float = 1e-6,
) -> bool:
    """True dacă segmentul verde e coliniar cu segmentul maro și se suprapun."""
    if len(green_seg) < 2:
        return False
    ax, ay = float(green_seg[0][0]), float(green_seg[0][1])
    bx, by = float(green_seg[1][0]), float(green_seg[1][1])
    gdx, gdy = bx - ax, by - ay
    g_len_sq = gdx * gdx + gdy * gdy
    if g_len_sq < 1e-12:
        return False
    cross = (brown_p2[0] - brown_p1[0]) * gdy - (brown_p2[1] - brown_p1[1]) * gdx
    # Toleranță relaxată pentru coliniaritate (laturi comune pot avea mici abateri)
    colinear_tol = max(tol * (g_len_sq ** 0.5 + 1), 2.0)
    if abs(cross) > colinear_tol:
        return False
    t1 = ((brown_p1[0] - ax) * gdx + (brown_p1[1] - ay) * gdy) / g_len_sq
    t2 = ((brown_p2[0] - ax) * gdx + (brown_p2[1] - ay) * gdy) / g_len_sq
    t_lo, t_hi = min(t1, t2), max(t1, t2)
    return t_hi >= -0.05 and t_lo <= 1.05


def _reconstruct_green_from_different_number_points(
    green_seg: List[List[float]],
    brown_endpoint_markers: List[Tuple[Tuple[float, float], int]],
    tol: float = 1e-6,
) -> List[List[List[float]]]:
    """
    Reconstruiește segmentul verde păstrând doar porțiunile între puncte cu numere DIFERITE (coliniare).
    Elimină porțiunile între puncte cu același număr.
    """
    if len(green_seg) < 2:
        return [green_seg]
    ax, ay = float(green_seg[0][0]), float(green_seg[0][1])
    bx, by = float(green_seg[1][0]), float(green_seg[1][1])
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return [green_seg]
    # Puncte coliniare pe segment: (t, set de pair_ids)
    pt_tol = 2.0
    t_to_ids: Dict[float, set] = {}
    for (pt, pair_id) in brown_endpoint_markers:
        tx = (pt[0] - ax) * dx + (pt[1] - ay) * dy
        cross = (pt[0] - ax) * dy - (pt[1] - ay) * dx
        if abs(cross) > 2.0 * (seg_len_sq ** 0.5 + 1):
            continue
        t = tx / seg_len_sq
        if t < -0.01 or t > 1.01:
            continue
        t_round = round(t / tol) * tol
        if t_round not in t_to_ids:
            t_to_ids[t_round] = set()
        t_to_ids[t_round].add(pair_id)
    if not t_to_ids:
        return [green_seg]
    # Include capetele segmentului
    t_to_ids[0.0] = t_to_ids.get(0.0, set()) or {-1}
    t_to_ids[1.0] = t_to_ids.get(1.0, set()) or {-2}
    pts_sorted = sorted(t_to_ids.items(), key=lambda x: x[0])
    # Păstrăm sub-segmente între puncte cu numere DIFERITE (intersecție goală)
    result: List[List[List[float]]] = []
    for i in range(len(pts_sorted) - 1):
        t_a, ids_a = pts_sorted[i]
        t_b, ids_b = pts_sorted[i + 1]
        if t_b - t_a < tol:
            continue
        if ids_a & ids_b:
            continue
        t_a_clamp = max(0.0, min(1.0, t_a))
        t_b_clamp = max(0.0, min(1.0, t_b))
        if t_b_clamp - t_a_clamp < tol:
            continue
        p0 = [ax + t_a_clamp * dx, ay + t_a_clamp * dy]
        p1 = [ax + t_b_clamp * dx, ay + t_b_clamp * dy]
        if (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 >= 1e-10:
            result.append([p0, p1])
    return result


def _clip_green_by_ranges(
    green_seg: List[List[float]],
    cut_ranges: List[Tuple[float, float]],
) -> List[List[List[float]]]:
    """Taie din segmentul verde intervalele date. Păstrează părțile din afara lor."""
    if len(green_seg) < 2 or not cut_ranges:
        return [green_seg]
    ax, ay = float(green_seg[0][0]), float(green_seg[0][1])
    bx, by = float(green_seg[1][0]), float(green_seg[1][1])
    dx, dy = bx - ax, by - ay
    cut_ranges = sorted(cut_ranges, key=lambda x: x[0])
    tol = 1e-6
    result: List[List[List[float]]] = []
    t_cur = 0.0
    for t_lo, t_hi in cut_ranges:
        if t_lo > t_cur + tol:
            p0 = [ax + t_cur * dx, ay + t_cur * dy]
            p1 = [ax + t_lo * dx, ay + t_lo * dy]
            if (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 >= 1e-10:
                result.append([p0, p1])
        t_cur = max(t_cur, t_hi)
    if t_cur < 1.0 - tol:
        p0 = [ax + t_cur * dx, ay + t_cur * dy]
        p1 = [bx, by]
        if (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 >= 1e-10:
            result.append([p0, p1])
    return result


def _segment_key(a: List[float], b: List[float], tol: float = 2.0) -> Tuple[float, float, float, float]:
    """Cheie canonică pentru segment: punctele sortate, rotunjite. tol=2 pentru match pe laturi comune."""
    ax, ay = round(a[0] / tol) * tol, round(a[1] / tol) * tol
    bx, by = round(b[0] / tol) * tol, round(b[1] / tol) * tol
    if (ax, ay) < (bx, by):
        return (ax, ay, bx, by)
    return (bx, by, ax, ay)


def _merge_diagonal_segments_with_chamfer(
    pyramid_segs: List[List[List[float]]],
    chamfer_diag_segs: List[List[List[float]]],
    tol: float = 0.5,
) -> List[List[List[float]]]:
    """
    Contopește segmentele pyramid cu cele chamfer (corner→m, m→corner).
    Elimină duplicate: (A,B) și (B,A) sunt același segment.
    Returnează diagonale unice cu câte 2 capete.
    """
    seen: Dict[Tuple[float, float, float, float], List[List[float]]] = {}
    for seg in pyramid_segs + chamfer_diag_segs:
        if len(seg) < 2:
            continue
        p1, p2 = [float(seg[0][0]), float(seg[0][1])], [float(seg[1][0]), float(seg[1][1])]
        k = _segment_key(p1, p2, tol=tol)
        if k not in seen:
            seen[k] = [p1, p2]
    return list(seen.values())


def _get_contour_segments_45w_chamfered(
    sections: List[Dict[str, Any]],
    shape: Tuple[int, int],
    upper_floor_sections: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[List[List[float]]], List[List[List[float]]]]:
    """
    Contur exterior 4.5_w: conectăm paralela dintre diagonale (la mijlocul lor) cu colțurile,
    eliminând laturile verticale/stânga-dreapta și înlocuindu-le cu diagonalele către midpoints.
    Returnează (segs_green, segs_pink): contur normal (verde) și segmentele care merg spre
    diagonalele 45° (roz – corner→midpoint, midpoint→midpoint, midpoint→corner).
    """
    from roof_calc.overhang import extend_secondary_sections_to_main_ridge

    upper = upper_floor_sections or []
    extended = extend_secondary_sections_to_main_ridge(sections)
    segs_green: List[List[List[float]]] = []
    segs_pink_diag: List[List[List[float]]] = []  # corner→midpoint, midpoint→corner (add to pyramid)
    segs_pink_mid: List[List[List[float]]] = []   # midpoint→midpoint (paralelă, NU la pyramid)
    seen_keys: set = set()
    skip_edge_keys: set = set()

    def _add_seg(p1: List[float], p2: List[float], allow_skip: bool = False, pink_diag: bool = False, pink_mid: bool = False) -> None:
        k = _segment_key(p1, p2)
        if allow_skip and k in skip_edge_keys:
            return
        if k not in seen_keys:
            seen_keys.add(k)
            if pink_mid:
                segs_pink_mid.append([p1, p2])
            elif pink_diag:
                segs_pink_diag.append([p1, p2])
            else:
                segs_green.append([p1, p2])

    for sec, sec_ext in zip(sections, extended):
        br = sec.get("bounding_rect") or []
        ridge = sec_ext.get("ridge_line") or []
        if len(br) < 3 or len(ridge) < 2:
            continue
        orient = str(sec.get("ridge_orientation", "horizontal"))
        r0 = (float(ridge[0][0]), float(ridge[0][1]))
        r1 = (float(ridge[1][0]), float(ridge[1][1]))
        rminx, rmaxx = min(r0[0], r1[0]), max(r0[0], r1[0])
        rminy, rmaxy = min(r0[1], r1[1]), max(r0[1], r1[1])
        ridge_y = (r0[1] + r1[1]) / 2.0 if orient == "horizontal" else (r0[1] + r1[1]) / 2.0
        ridge_x = (r0[0] + r1[0]) / 2.0 if orient == "vertical" else (r0[0] + r1[0]) / 2.0
        (
            left_mid_x, right_mid_x, top_mid_y, bot_mid_y,
            _frx, _flx, _fdy, _fuy,
        ) = _get_ridge_clamp_midpoints(
            sec_ext, extended, rminx, rmaxx, rminy, rmaxy, orient, ridge_y, ridge_x
        )

        xs = [float(p[0]) for p in br]
        ys = [float(p[1]) for p in br]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
        corner_to_sides = [(3, 0), (0, 1), (1, 2), (2, 3)]
        attached_sides = _sides_attached_to_upper(sec, upper, same_floor_sections=sections)
        clamp_to_mid = len(attached_sides) == 0

        ridge_center_x = (rminx + rmaxx) / 2.0
        ridge_center_y = (rminy + rmaxy) / 2.0

        def ridge_end(ci: int) -> Optional[Tuple[float, float]]:
            s1, s2 = corner_to_sides[ci]
            if s1 in attached_sides or s2 in attached_sides:
                return None
            cx, cy = corners[ci]
            if orient == "horizontal":
                dy = ridge_y - cy
                dx = abs(dy)
                corner_on_left = abs(cx - minx) < 1e-6
                if corner_on_left:
                    end_x = (cx - dx) if rminx < cx else (cx + dx)
                    if clamp_to_mid:
                        end_x = max(rminx, min(left_mid_x, ridge_center_x, end_x))
                    else:
                        end_x = max(rminx, min(rmaxx, end_x))
                    end_x = min(end_x, ridge_center_x)
                else:
                    end_x = (cx + dx) if rmaxx > cx else (cx - dx)
                    if clamp_to_mid:
                        end_x = max(right_mid_x, ridge_center_x, min(rmaxx, end_x))
                    else:
                        end_x = max(rminx, min(rmaxx, end_x))
                    end_x = max(end_x, ridge_center_x)
                if _flx is not None and cx <= _flx:
                    end_x = min(end_x, _flx)
                if _frx is not None and cx >= _frx:
                    end_x = max(end_x, _frx)
                return (end_x, ridge_y)
            else:
                dx = ridge_x - cx
                dy = abs(dx)
                corner_on_top = abs(cy - miny) < 1e-6
                if corner_on_top:
                    end_y = (cy - dy) if rminy < cy else (cy + dy)
                    if clamp_to_mid:
                        end_y = max(rminy, min(top_mid_y, ridge_center_y, end_y))
                    else:
                        end_y = max(rminy, min(rmaxy, end_y))
                    end_y = min(end_y, ridge_center_y)
                else:
                    end_y = (cy + dy) if rmaxy > cy else (cy - dy)
                    if clamp_to_mid:
                        end_y = max(bot_mid_y, ridge_center_y, min(rmaxy, end_y))
                    else:
                        end_y = max(rminy, min(rmaxy, end_y))
                    end_y = max(end_y, ridge_center_y)
                if _fdy is not None and cy <= _fdy:
                    end_y = min(end_y, _fdy)
                if _fuy is not None and cy >= _fuy:
                    end_y = max(end_y, _fuy)
                return (ridge_x, end_y)

        def midpoint(ci: int, re: Tuple[float, float]) -> Tuple[float, float]:
            cx, cy = corners[ci]
            return ((cx + re[0]) / 2.0, (cy + re[1]) / 2.0)

        re0, re1, re2, re3 = ridge_end(0), ridge_end(1), ridge_end(2), ridge_end(3)
        edges = [
            ([minx, miny], [maxx, miny]),
            ([maxx, miny], [maxx, maxy]),
            ([maxx, maxy], [minx, maxy]),
            ([minx, maxy], [minx, miny]),
        ]
        if orient == "horizontal":
            if re0 is not None and re3 is not None and 1 in attached_sides:
                skip_edge_keys.add(_segment_key(edges[1][0], edges[1][1]))
            if re1 is not None and re2 is not None and 3 in attached_sides:
                skip_edge_keys.add(_segment_key(edges[3][0], edges[3][1]))
        else:
            if re0 is not None and re1 is not None and 2 in attached_sides:
                skip_edge_keys.add(_segment_key(edges[2][0], edges[2][1]))
            if re2 is not None and re3 is not None and 0 in attached_sides:
                skip_edge_keys.add(_segment_key(edges[0][0], edges[0][1]))

        # FIX v4: skip teșire când segmentul paralel (portocaliu) ar fi prea scurt (triunghi degenerat)
        _min_chamfer_len_sq = 16.0  # minim 4px pentru segmentul de mijloc

        if orient == "horizontal":
            # Stânga: eliminăm latura verticală (minx,*), conectăm colțuri la paralelă
            if re0 is not None and re3 is not None:
                m_left_top = midpoint(0, re0)
                m_left_bot = midpoint(3, re3)
                _mid_sq = (m_left_top[0]-m_left_bot[0])**2 + (m_left_top[1]-m_left_bot[1])**2
                if _mid_sq >= _min_chamfer_len_sq:
                    _add_seg([minx, miny], [m_left_top[0], m_left_top[1]], pink_diag=True)
                    _add_seg([m_left_top[0], m_left_top[1]], [m_left_bot[0], m_left_bot[1]], pink_mid=True)
                    _add_seg([m_left_bot[0], m_left_bot[1]], [minx, maxy], pink_diag=True)
                else:
                    _add_seg([minx, miny], [minx, maxy], allow_skip=True)
            else:
                _add_seg([minx, miny], [minx, maxy], allow_skip=True)

            _add_seg([minx, maxy], [maxx, maxy])

            if re1 is not None and re2 is not None:
                m_right_top = midpoint(1, re1)
                m_right_bot = midpoint(2, re2)
                _mid_sq = (m_right_top[0]-m_right_bot[0])**2 + (m_right_top[1]-m_right_bot[1])**2
                if _mid_sq >= _min_chamfer_len_sq:
                    _add_seg([maxx, maxy], [m_right_bot[0], m_right_bot[1]], pink_diag=True)
                    _add_seg([m_right_bot[0], m_right_bot[1]], [m_right_top[0], m_right_top[1]], pink_mid=True)
                    _add_seg([m_right_top[0], m_right_top[1]], [maxx, miny], pink_diag=True)
                else:
                    _add_seg([maxx, maxy], [maxx, miny], allow_skip=True)
            else:
                _add_seg([maxx, maxy], [maxx, miny], allow_skip=True)

            _add_seg([maxx, miny], [minx, miny])
        else:
            if re0 is not None and re1 is not None:
                m_top_left = midpoint(0, re0)
                m_top_right = midpoint(1, re1)
                _mid_sq = (m_top_left[0]-m_top_right[0])**2 + (m_top_left[1]-m_top_right[1])**2
                if _mid_sq >= _min_chamfer_len_sq:
                    _add_seg([minx, miny], [m_top_left[0], m_top_left[1]], pink_diag=True)
                    _add_seg([m_top_left[0], m_top_left[1]], [m_top_right[0], m_top_right[1]], pink_mid=True)
                    _add_seg([m_top_right[0], m_top_right[1]], [maxx, miny], pink_diag=True)
                else:
                    _add_seg([minx, miny], [maxx, miny], allow_skip=True)
            else:
                _add_seg([minx, miny], [maxx, miny], allow_skip=True)

            _add_seg([maxx, miny], [maxx, maxy])

            if re2 is not None and re3 is not None:
                m_bot_right = midpoint(2, re2)
                m_bot_left = midpoint(3, re3)
                _mid_sq = (m_bot_right[0]-m_bot_left[0])**2 + (m_bot_right[1]-m_bot_left[1])**2
                if _mid_sq >= _min_chamfer_len_sq:
                    _add_seg([maxx, maxy], [m_bot_right[0], m_bot_right[1]], pink_diag=True)
                    _add_seg([m_bot_right[0], m_bot_right[1]], [m_bot_left[0], m_bot_left[1]], pink_mid=True)
                    _add_seg([m_bot_left[0], m_bot_left[1]], [minx, maxy], pink_diag=True)
                else:
                    _add_seg([maxx, maxy], [minx, maxy], allow_skip=True)
            else:
                _add_seg([maxx, maxy], [minx, maxy], allow_skip=True)

            _add_seg([minx, maxy], [minx, miny])

    if segs_green or segs_pink_diag or segs_pink_mid:
        return (segs_green, segs_pink_diag)
    fallback = _get_contour_segments_from_sections(sections, shape)
    return (fallback, [])


def _get_opposite_side_segments_to_eliminate_45w(
    sections: List[Dict[str, Any]],
    upper_floor_sections: Optional[List[Dict[str, Any]]] = None,
) -> List[Tuple[List[List[float]], Tuple[float, float], Tuple[float, float]]]:
    """
    Segmente de eliminat: latura opusă teșirii, când e lipită de alt dreptunghi la același etaj.
    Returnează (segment, colț1, colț2) – colțurile dreptunghiului care delimitează latura.
    """
    from roof_calc.overhang import extend_secondary_sections_to_main_ridge

    upper = upper_floor_sections or []
    extended = extend_secondary_sections_to_main_ridge(sections)
    out: List[Tuple[List[List[float]], Tuple[float, float], Tuple[float, float]]] = []
    seen: set = set()

    def _add(seg: List[List[float]], c1: Tuple[float, float], c2: Tuple[float, float]) -> None:
        k = _segment_key(seg[0], seg[1])
        if k not in seen:
            seen.add(k)
            out.append((seg, c1, c2))

    for sec, sec_ext in zip(sections, extended):
        br = sec.get("bounding_rect") or []
        ridge = sec_ext.get("ridge_line") or []
        if len(br) < 3 or len(ridge) < 2:
            continue
        orient = str(sec.get("ridge_orientation", "horizontal"))
        r0 = (float(ridge[0][0]), float(ridge[0][1]))
        r1 = (float(ridge[1][0]), float(ridge[1][1]))
        rminx, rmaxx = min(r0[0], r1[0]), max(r0[0], r1[0])
        rminy, rmaxy = min(r0[1], r1[1]), max(r0[1], r1[1])
        ridge_y = (r0[1] + r1[1]) / 2.0
        ridge_x = (r0[0] + r1[0]) / 2.0
        (
            left_mid_x, right_mid_x, top_mid_y, bot_mid_y,
            _frx, _flx, _fdy, _fuy,
        ) = _get_ridge_clamp_midpoints(
            sec_ext, extended, rminx, rmaxx, rminy, rmaxy, orient, ridge_y, ridge_x
        )
        xs = [float(p[0]) for p in br]
        ys = [float(p[1]) for p in br]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
        corner_to_sides = [(3, 0), (0, 1), (1, 2), (2, 3)]
        attached_sides = _sides_attached_to_upper(sec, upper, same_floor_sections=sections)
        attached_same_floor = _sides_attached_to_upper(sec, [], same_floor_sections=sections)
        clamp_to_mid = len(attached_sides) == 0

        def ridge_end(ci: int) -> Optional[Tuple[float, float]]:
            s1, s2 = corner_to_sides[ci]
            if s1 in attached_sides or s2 in attached_sides:
                return None
            cx, cy = corners[ci]
            if orient == "horizontal":
                dy = ridge_y - cy
                dx = abs(dy)
                corner_on_left = abs(cx - minx) < 1e-6
                if corner_on_left:
                    end_x = (cx - dx) if rminx < cx else (cx + dx)
                    if clamp_to_mid:
                        end_x = max(rminx, min(left_mid_x, ridge_x, end_x))
                    else:
                        end_x = max(rminx, min(rmaxx, end_x))
                    end_x = min(end_x, ridge_x)
                    if _flx is not None and cx <= _flx:
                        end_x = min(end_x, _flx)
                    if _frx is not None and cx >= _frx:
                        end_x = max(end_x, _frx)
                else:
                    end_x = (cx + dx) if rmaxx > cx else (cx - dx)
                    if clamp_to_mid:
                        end_x = max(right_mid_x, ridge_x, min(rmaxx, end_x))
                    else:
                        end_x = max(rminx, min(rmaxx, end_x))
                    end_x = max(end_x, ridge_x)
                    if _flx is not None and cx <= _flx:
                        end_x = min(end_x, _flx)
                    if _frx is not None and cx >= _frx:
                        end_x = max(end_x, _frx)
                return (end_x, ridge_y)
            else:
                dx = ridge_x - cx
                dy = abs(dx)
                corner_on_top = abs(cy - miny) < 1e-6
                if corner_on_top:
                    end_y = (cy - dy) if rminy < cy else (cy + dy)
                    if clamp_to_mid:
                        end_y = max(rminy, min(top_mid_y, ridge_y, end_y))
                    else:
                        end_y = max(rminy, min(rmaxy, end_y))
                    end_y = min(end_y, ridge_y)
                    if _fdy is not None and cy <= _fdy:
                        end_y = min(end_y, _fdy)
                    if _fuy is not None and cy >= _fuy:
                        end_y = max(end_y, _fuy)
                else:
                    end_y = (cy + dy) if rmaxy > cy else (cy - dy)
                    if clamp_to_mid:
                        end_y = max(bot_mid_y, ridge_y, min(rmaxy, end_y))
                    else:
                        end_y = max(rminy, min(rmaxy, end_y))
                    end_y = max(end_y, ridge_y)
                    if _fdy is not None and cy <= _fdy:
                        end_y = min(end_y, _fdy)
                    if _fuy is not None and cy >= _fuy:
                        end_y = max(end_y, _fuy)
                return (ridge_x, end_y)

        re0, re1, re2, re3 = ridge_end(0), ridge_end(1), ridge_end(2), ridge_end(3)
        edges = [
            ([minx, miny], [maxx, miny]),
            ([maxx, miny], [maxx, maxy]),
            ([maxx, maxy], [minx, maxy]),
            ([minx, maxy], [minx, miny]),
        ]
        side_to_corners = [
            ((minx, miny), (maxx, miny)),
            ((maxx, miny), (maxx, maxy)),
            ((maxx, maxy), (minx, maxy)),
            ((minx, maxy), (minx, miny)),
        ]
        for (our_side, neighbor_side), (del0, del1) in [
            ((1, 3), (re0, re3)),
            ((3, 1), (re1, re2)),
            ((2, 0), (re0, re1)),
            ((0, 2), (re2, re3)),
        ]:
            if del0 is not None and del1 is not None and our_side in attached_same_floor:
                es = edges[our_side]
                c1, c2 = side_to_corners[our_side]
                _add(es, c1, c2)
                for osec in sections:
                    if osec is sec:
                        continue
                    obr = osec.get("bounding_rect") or []
                    if len(obr) < 3:
                        continue
                    oxs = [float(p[0]) for p in obr]
                    oys = [float(p[1]) for p in obr]
                    ominx, omaxx, ominy, omaxy = min(oxs), max(oxs), min(oys), max(oys)
                    o_edges = [
                        ([ominx, ominy], [omaxx, ominy]),
                        ([omaxx, ominy], [omaxx, omaxy]),
                        ([omaxx, omaxy], [ominx, omaxy]),
                        ([ominx, omaxy], [ominx, ominy]),
                    ]
                    ea = edges[our_side]
                    eb = o_edges[neighbor_side]
                    r = _edge_overlap(
                        ea[0][0], ea[0][1], ea[1][0], ea[1][1],
                        eb[0][0], eb[0][1], eb[1][0], eb[1][1],
                    )
                    if r >= 0.5:
                        oe = o_edges[neighbor_side]
                        o_corners = [
                            ((ominx, ominy), (omaxx, ominy)),
                            ((omaxx, ominy), (omaxx, omaxy)),
                            ((omaxx, omaxy), (ominx, omaxy)),
                            ((ominx, omaxy), (ominx, ominy)),
                        ]
                        oc1, oc2 = o_corners[neighbor_side]
                        _add(oe, oc1, oc2)
    return out


def _get_orange_midpoint_segments_45w(
    sections: List[Dict[str, Any]],
    upper_floor_sections: Optional[List[Dict[str, Any]]] = None,
) -> List[List[List[float]]]:
    """
    Segmente portocalii: capetele la mijlocul diagonalelor. Nu afișăm unde nu afișăm diagonale (laturi lipite).
    Ridge extins – diagonalele pot ieși din dreptunghi.
    """
    from roof_calc.overhang import extend_secondary_sections_to_main_ridge

    upper = upper_floor_sections or []
    extended = extend_secondary_sections_to_main_ridge(sections)
    segs: List[List[List[float]]] = []
    for sec, sec_ext in zip(sections, extended):
        br = sec.get("bounding_rect") or []
        ridge = sec_ext.get("ridge_line") or []
        if len(br) < 3 or len(ridge) < 2:
            continue
        orient = str(sec.get("ridge_orientation", "horizontal"))
        r0 = (float(ridge[0][0]), float(ridge[0][1]))
        r1 = (float(ridge[1][0]), float(ridge[1][1]))
        rminx, rmaxx = min(r0[0], r1[0]), max(r0[0], r1[0])
        rminy, rmaxy = min(r0[1], r1[1]), max(r0[1], r1[1])
        ridge_y = (r0[1] + r1[1]) / 2.0 if orient == "horizontal" else (r0[1] + r1[1]) / 2.0
        ridge_x = (r0[0] + r1[0]) / 2.0 if orient == "vertical" else (r0[0] + r1[0]) / 2.0
        (
            left_mid_x, right_mid_x, top_mid_y, bot_mid_y,
            _frx, _flx, _fdy, _fuy,
        ) = _get_ridge_clamp_midpoints(
            sec_ext, extended, rminx, rmaxx, rminy, rmaxy, orient, ridge_y, ridge_x
        )

        xs = [float(p[0]) for p in br]
        ys = [float(p[1]) for p in br]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
        corner_to_sides = [(3, 0), (0, 1), (1, 2), (2, 3)]
        attached_sides = _sides_attached_to_upper(sec, upper, same_floor_sections=sections)
        clamp_to_mid = len(attached_sides) == 0

        ridge_center_x = (rminx + rmaxx) / 2.0
        ridge_center_y = (rminy + rmaxy) / 2.0

        def ridge_end(ci: int) -> Optional[Tuple[float, float]]:
            s1, s2 = corner_to_sides[ci]
            if s1 in attached_sides or s2 in attached_sides:
                return None
            cx, cy = corners[ci]
            if orient == "horizontal":
                dy = ridge_y - cy
                dx = abs(dy)
                corner_on_left = abs(cx - minx) < 1e-6
                if corner_on_left:
                    end_x = (cx - dx) if rminx < cx else (cx + dx)
                    if clamp_to_mid:
                        end_x = max(rminx, min(left_mid_x, ridge_center_x, end_x))
                    else:
                        end_x = max(rminx, min(rmaxx, end_x))
                    end_x = min(end_x, ridge_center_x)
                else:
                    end_x = (cx + dx) if rmaxx > cx else (cx - dx)
                    if clamp_to_mid:
                        end_x = max(right_mid_x, ridge_center_x, min(rmaxx, end_x))
                    else:
                        end_x = max(rminx, min(rmaxx, end_x))
                    end_x = max(end_x, ridge_center_x)
                if _flx is not None and cx <= _flx:
                    end_x = min(end_x, _flx)
                if _frx is not None and cx >= _frx:
                    end_x = max(end_x, _frx)
                return (end_x, ridge_y)
            else:
                dx = ridge_x - cx
                dy = abs(dx)
                corner_on_top = abs(cy - miny) < 1e-6
                if corner_on_top:
                    end_y = (cy - dy) if rminy < cy else (cy + dy)
                    if clamp_to_mid:
                        end_y = max(rminy, min(top_mid_y, ridge_center_y, end_y))
                    else:
                        end_y = max(rminy, min(rmaxy, end_y))
                    end_y = min(end_y, ridge_center_y)
                else:
                    end_y = (cy + dy) if rmaxy > cy else (cy - dy)
                    if clamp_to_mid:
                        end_y = max(bot_mid_y, ridge_center_y, min(rmaxy, end_y))
                    else:
                        end_y = max(rminy, min(rmaxy, end_y))
                    end_y = max(end_y, ridge_center_y)
                if _fdy is not None and cy <= _fdy:
                    end_y = min(end_y, _fdy)
                if _fuy is not None and cy >= _fuy:
                    end_y = max(end_y, _fuy)
                return (ridge_x, end_y)

        if orient == "horizontal":
            re0 = ridge_end(0)  # top-left
            re3 = ridge_end(3)  # bottom-left
            if re0 is not None and re3 is not None:
                m0 = ((corners[0][0] + re0[0]) / 2.0, (corners[0][1] + re0[1]) / 2.0)
                m3 = ((corners[3][0] + re3[0]) / 2.0, (corners[3][1] + re3[1]) / 2.0)
                segs.append([[m0[0], m0[1]], [m3[0], m3[1]]])
            re1 = ridge_end(1)
            re2 = ridge_end(2)
            if re1 is not None and re2 is not None:
                m1 = ((corners[1][0] + re1[0]) / 2.0, (corners[1][1] + re1[1]) / 2.0)
                m2 = ((corners[2][0] + re2[0]) / 2.0, (corners[2][1] + re2[1]) / 2.0)
                segs.append([[m1[0], m1[1]], [m2[0], m2[1]]])
        else:
            re0, re1 = ridge_end(0), ridge_end(1)
            if re0 is not None and re1 is not None:
                m0 = ((corners[0][0] + re0[0]) / 2.0, (corners[0][1] + re0[1]) / 2.0)
                m1 = ((corners[1][0] + re1[0]) / 2.0, (corners[1][1] + re1[1]) / 2.0)
                segs.append([[m0[0], m0[1]], [m1[0], m1[1]]])
            re2, re3 = ridge_end(2), ridge_end(3)
            if re2 is not None and re3 is not None:
                m2 = ((corners[2][0] + re2[0]) / 2.0, (corners[2][1] + re2[1]) / 2.0)
                m3 = ((corners[3][0] + re3[0]) / 2.0, (corners[3][1] + re3[1]) / 2.0)
                segs.append([[m2[0], m2[1]], [m3[0], m3[1]]])
    return segs


def _get_upper_rect_segments(upper_floor_sections: List[Dict[str, Any]]) -> List[List[List[float]]]:
    """Segmente pentru conturul dreptunghiurilor etajului superior (galben)."""
    segs: List[List[List[float]]] = []
    for sec in upper_floor_sections:
        br = sec.get("bounding_rect") or []
        if len(br) < 3:
            continue
        xs = [float(p[0]) for p in br]
        ys = [float(p[1]) for p in br]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            segs.append([[a[0], a[1]], [b[0], b[1]]])
    return segs


def _section_covered_by_upper(
    sec: Dict[str, Any],
    upper_rect_segs: List[List[List[float]]],
    area_ratio_thresh: float = 0.5,
) -> bool:
    """True dacă secțiunea (poligonul) are deasupra etajul superior – centroidul sau majoritatea ariei e în interiorul poligonului upper."""
    if not upper_rect_segs or len(upper_rect_segs) < 3:
        return False
    br = sec.get("bounding_rect") or []
    if len(br) < 3:
        return False
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.ops import polygonize_full, unary_union
        lines_upper = []
        for s in upper_rect_segs:
            if len(s) >= 2 and (float(s[0][0]) - float(s[1][0])) ** 2 + (float(s[0][1]) - float(s[1][1])) ** 2 >= 1e-10:
                lines_upper.append(((float(s[0][0]), float(s[0][1])), (float(s[1][0]), float(s[1][1]))))
        if len(lines_upper) < 3:
            return False
        from shapely.geometry import LineString
        ls = [LineString([a, b]) for a, b in lines_upper]
        polys, _, _, _ = polygonize_full(ls)
        geoms = list(getattr(polys, "geoms", None) or [polys])
        upper_poly = unary_union([g for g in geoms if g and not getattr(g, "is_empty", True)]) if geoms else None
        if upper_poly is None or getattr(upper_poly, "is_empty", True):
            return False
        sec_poly = ShapelyPolygon([(float(p[0]), float(p[1])) for p in br])
        if sec_poly.is_empty or not sec_poly.is_valid:
            sec_poly = sec_poly.buffer(0) if sec_poly else sec_poly
        if sec_poly is None or getattr(sec_poly, "is_empty", True):
            return False
        inter = sec_poly.intersection(upper_poly)
        inter_area = float(getattr(inter, "area", 0) or 0)
        sec_area = float(getattr(sec_poly, "area", 1) or 1)
        if sec_area < 1e-10:
            return False
        return (inter_area / sec_area) >= area_ratio_thresh
    except Exception:
        return False


def _get_ridge_segments(sections: List[Dict[str, Any]]) -> List[List[List[float]]]:
    from roof_calc.overhang import extend_secondary_sections_to_main_ridge

    extended = extend_secondary_sections_to_main_ridge(sections)
    segs: List[List[List[float]]] = []
    for sec in extended:
        ridge = sec.get("ridge_line") or []
        if len(ridge) >= 2:
            segs.append([
                [float(ridge[0][0]), float(ridge[0][1])],
                [float(ridge[1][0]), float(ridge[1][1])],
            ])
    return segs


def _seg_intersect(
    a1: Tuple[float, float], a2: Tuple[float, float],
    b1: Tuple[float, float], b2: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """Intersecția segmentelor a1-a2 și b1-b2, sau None."""
    ax1, ay1, ax2, ay2 = a1[0], a1[1], a2[0], a2[1]
    bx1, by1, bx2, by2 = b1[0], b1[1], b2[0], b2[1]
    d = (ax2 - ax1) * (by2 - by1) - (ay2 - ay1) * (bx2 - bx1)
    if abs(d) < 1e-12:
        return None
    t = ((bx1 - ax1) * (by2 - by1) - (by1 - ay1) * (bx2 - bx1)) / d
    u = ((ax2 - ax1) * (by1 - ay1) - (ay2 - ay1) * (bx1 - ax1)) / d
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (ax1 + t * (ax2 - ax1), ay1 + t * (ay2 - ay1))
    return None


def _get_main_ridge_connection(
    sec: Dict[str, Any],
    main_sec: Dict[str, Any],
    extended: List[Dict[str, Any]],
) -> Optional[Tuple[float, float]]:
    """Punctul unde ridge-ul secțiunii sec se întâlnește cu ridge-ul principal."""
    main_ridge = main_sec.get("ridge_line") or []
    ridge = sec.get("ridge_line") or []
    if len(main_ridge) < 2 or len(ridge) < 2:
        return None
    main_orient = str(main_sec.get("ridge_orientation", "horizontal"))
    orient = str(sec.get("ridge_orientation", "horizontal"))
    if orient == main_orient:
        return None
    ma = (float(main_ridge[0][0]), float(main_ridge[0][1]))
    mb = (float(main_ridge[1][0]), float(main_ridge[1][1]))
    sa = (float(ridge[0][0]), float(ridge[0][1]))
    sb = (float(ridge[1][0]), float(ridge[1][1]))
    pt = _seg_intersect(ma, mb, sa, sb)
    return pt


def _point_on_segment(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float], tol: float = 2.0) -> bool:
    """True dacă p e pe segmentul a-b (în interior sau capete)."""
    px, py = p[0], p[1]
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    if abs(bx - ax) < 1e-9 and abs(by - ay) < 1e-9:
        return abs(px - ax) < tol and abs(py - ay) < tol
    seg_len_sq = (bx - ax) ** 2 + (by - ay) ** 2
    if seg_len_sq < 1e-12:
        return (px - ax) ** 2 + (py - ay) ** 2 < tol * tol
    t = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / seg_len_sq
    if t < -0.01 or t > 1.01:
        return False
    proj = (ax + t * (bx - ax), ay + t * (by - ay))
    return (px - proj[0]) ** 2 + (py - proj[1]) ** 2 < tol * tol


def _closest_point_on_segment(
    p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]
) -> Tuple[float, float, float]:
    """Returnează (x, y, dist_sq) al punctului cel mai apropiat de p pe segmentul [a,b]."""
    px, py = p[0], p[1]
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-20:
        d2 = (px - ax) ** 2 + (py - ay) ** 2
        return (ax, ay, d2)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    qx, qy = ax + t * dx, ay + t * dy
    d2 = (px - qx) ** 2 + (py - qy) ** 2
    return (qx, qy, d2)


def _ridge_intersection_points(
    segments_ridge: List[List[List[float]]],
    tol: float = 5.0,
) -> List[Tuple[float, float]]:
    """Puncte unde cel puțin două ridge-uri se întâlnesc (ex. T/L). Folosit pentru a nu desena buline albastre acolo."""
    if not segments_ridge or len(segments_ridge) < 2:
        return []
    # Fiecare endpoint de ridge: (x, y), segment_index
    endpoints: List[Tuple[Tuple[float, float], int]] = []
    for idx, seg in enumerate(segments_ridge):
        if len(seg) >= 2:
            endpoints.append(((float(seg[0][0]), float(seg[0][1])), idx))
            endpoints.append(((float(seg[1][0]), float(seg[1][1])), idx))
    out: List[Tuple[float, float]] = []
    used = [False] * len(endpoints)
    for i, (p, seg_i) in enumerate(endpoints):
        if used[i]:
            continue
        cluster_pts = [p]
        cluster_segs = {seg_i}
        used[i] = True
        for j, (q, seg_j) in enumerate(endpoints):
            if used[j]:
                continue
            if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= tol * tol:
                cluster_pts.append(q)
                cluster_segs.add(seg_j)
                used[j] = True
        # Intersecție = punct unde se întâlnesc ≥2 segmente ridge (capete din segmente diferite)
        if len(cluster_segs) >= 2:
            cx = sum(c[0] for c in cluster_pts) / len(cluster_pts)
            cy = sum(c[1] for c in cluster_pts) / len(cluster_pts)
            out.append((cx, cy))
    return out


def _compute_overhang_ridge_pts(
    segments_ridge: List[List[List[float]]],
    segments_overhang: List[List[List[float]]],
    roof_type: str,
    tol: float = 5.0,
) -> List[Tuple[float, float]]:
    """Puncte unde ridge-ul prelungit intersectează overhang-ul (buline albastre închis). Folosit pentru 2_w."""
    if not segments_overhang or not segments_ridge:
        return []
    ridge_int_pts = _ridge_intersection_points(segments_ridge, tol=tol)

    def _near_ridge_intersection(p: Tuple[float, float]) -> bool:
        for r in ridge_int_pts:
            if (p[0] - r[0]) ** 2 + (p[1] - r[1]) ** 2 <= tol * tol:
                return True
        return False

    def _segment_segment_intersection(
        a0: Tuple[float, float], a1: Tuple[float, float],
        b0: Tuple[float, float], b1: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        ax0, ay0, ax1, ay1 = a0[0], a0[1], a1[0], a1[1]
        bx0, by0, bx1, by1 = b0[0], b0[1], b1[0], b1[1]
        dx_a, dy_a = ax1 - ax0, ay1 - ay0
        dx_b, dy_b = bx1 - bx0, by1 - by0
        denom = dx_a * dy_b - dy_a * dx_b
        if abs(denom) < 1e-12:
            return None
        t = ((bx0 - ax0) * dy_b - (by0 - ay0) * dx_b) / denom
        s = ((bx0 - ax0) * dy_a - (by0 - ay0) * dx_a) / denom
        if not (0 <= t <= 1 and 0 <= s <= 1):
            return None
        return (ax0 + t * dx_a, ay0 + t * dy_a)

    out_pts: List[Tuple[float, float]] = []
    for rseg in segments_ridge:
        if len(rseg) < 2:
            continue
        r0 = (float(rseg[0][0]), float(rseg[0][1]))
        r1 = (float(rseg[1][0]), float(rseg[1][1]))
        dx = r1[0] - r0[0]
        dy = r1[1] - r0[1]
        L = (dx * dx + dy * dy) ** 0.5
        if L < 1e-12:
            continue
        half = max(0.1 * L, L)
        e0 = (r0[0] - half * (dx / L), r0[1] - half * (dy / L))
        e1 = (r1[0] + half * (dx / L), r1[1] + half * (dy / L))
        for oseg in segments_overhang:
            if len(oseg) < 2:
                continue
            o0 = (float(oseg[0][0]), float(oseg[0][1]))
            o1 = (float(oseg[1][0]), float(oseg[1][1]))
            pt = _segment_segment_intersection(e0, e1, o0, o1)
            if pt is None:
                continue
            if roof_type == "4_w":
                continue
            dx_o = o1[0] - o0[0]
            dy_o = o1[1] - o0[1]
            len_o_sq = dx_o * dx_o + dy_o * dy_o
            if len_o_sq >= 1e-20:
                dot = dx * dx_o + dy * dy_o
                len_r = (dx * dx + dy * dy) ** 0.5
                len_o = len_o_sq ** 0.5
                thresh = 0.05 if roof_type == "2_w" else 0.17
                if abs(dot) > thresh * len_r * len_o:
                    continue
            if not _near_ridge_intersection(r0) and not _near_ridge_intersection(r1):
                out_pts.append(pt)
    return [p for p in out_pts if not _near_ridge_intersection(p)]


def _point_on_segment_seg(
    pt: Tuple[float, float],
    seg: List[List[float]],
    tol: float = 5.0,
) -> bool:
    """True dacă pt se află pe segmentul seg (proiecție între capete, distanță <= tol)."""
    if len(seg) < 2:
        return False
    ax, ay = float(seg[0][0]), float(seg[0][1])
    bx, by = float(seg[1][0]), float(seg[1][1])
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-20:
        return (pt[0] - ax) ** 2 + (pt[1] - ay) ** 2 <= tol * tol
    t = ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / len_sq
    if not (0 <= t <= 1):
        return False
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return (pt[0] - proj_x) ** 2 + (pt[1] - proj_y) ** 2 <= tol * tol


def _point_near_segment_midpoint(
    pt: Tuple[float, float],
    seg: List[List[float]],
    tol: float = 5.0,
    mid_frac: float = 0.1,
) -> bool:
    """True dacă pt este aproape de mijlocul segmentului seg (90% acuratețe: proiecția t în [0.5-mid_frac/2, 0.5+mid_frac/2], distanță <= tol)."""
    if len(seg) < 2:
        return False
    ax, ay = float(seg[0][0]), float(seg[0][1])
    bx, by = float(seg[1][0]), float(seg[1][1])
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-20:
        return (pt[0] - ax) ** 2 + (pt[1] - ay) ** 2 <= tol * tol
    t = ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / len_sq
    # Mijloc = 0.5; 90% acuratețe = acceptăm în [0.45, 0.55] (mid_frac=0.1)
    if not (0.5 - mid_frac / 2 <= t <= 0.5 + mid_frac / 2):
        return False
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return (pt[0] - proj_x) ** 2 + (pt[1] - proj_y) ** 2 <= tol * tol


def _overhang_segments_with_ridge_pt(
    segments_overhang: List[List[List[float]]],
    overhang_ridge_pts: List[Tuple[float, float]],
    tol: float = 5.0,
    at_midpoint_only: bool = True,
) -> List[List[List[float]]]:
    """Segmente de overhang pe care există o bulină albastră închis la jumătatea segmentului (90% acuratețe). Pentru 2_w: le desenăm mov închis."""
    if not segments_overhang or not overhang_ridge_pts:
        return []
    out: List[List[List[float]]] = []
    for seg in segments_overhang:
        if len(seg) < 2:
            continue
        for pt in overhang_ridge_pts:
            if at_midpoint_only:
                if _point_near_segment_midpoint(pt, seg, tol=tol, mid_frac=0.1):
                    out.append(seg)
                    break
            else:
                if _point_on_segment_seg(pt, seg, tol=tol):
                    out.append(seg)
                    break
    return out


def _seg_key(seg: List[List[float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Cheie pentru segment (ordine capete normalizată) pentru comparație."""
    if len(seg) < 2:
        return ((0.0, 0.0), (0.0, 0.0))
    a = (round(float(seg[0][0]), 2), round(float(seg[0][1]), 2))
    b = (round(float(seg[1][0]), 2), round(float(seg[1][1]), 2))
    return (a, b) if a <= b else (b, a)


def _contour_interior_corners(
    segments_contour: List[List[List[float]]],
) -> List[Tuple[float, float]]:
    """Colțuri re-entrante (interioare) ale conturului verde: unghi > 180."""
    if not segments_contour:
        return []
    # Construim poligon închis: listă ordonată de vârfuri prin lanțuire segmente
    segs = [(float(s[0][0]), float(s[0][1]), float(s[1][0]), float(s[1][1])) for s in segments_contour if len(s) >= 2]
    if not segs:
        return []
    ordered: List[Tuple[float, float]] = []
    (x0, y0, x1, y1) = segs[0]
    ordered.append((x0, y0))
    ordered.append((x1, y1))
    used = {0}
    while len(used) < len(segs):
        tail = ordered[-1]
        found = False
        for idx, (a0, a1, b0, b1) in enumerate(segs):
            if idx in used:
                continue
            tol = 1e-6
            if (abs(tail[0] - a0) < tol and abs(tail[1] - a1) < tol):
                ordered.append((b0, b1))
                used.add(idx)
                found = True
                break
            if (abs(tail[0] - b0) < tol and abs(tail[1] - b1) < tol):
                ordered.append((a0, a1))
                used.add(idx)
                found = True
                break
        if not found:
            break
    if len(ordered) < 3:
        return []
    interior: List[Tuple[float, float]] = []
    for i in range(len(ordered)):
        prev = ordered[(i - 1) % len(ordered)]
        curr = ordered[i]
        nxt = ordered[(i + 1) % len(ordered)]
        v1 = (prev[0] - curr[0], prev[1] - curr[1])
        v2 = (nxt[0] - curr[0], nxt[1] - curr[1])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross < -1e-6:
            interior.append(curr)
    return interior


def _overhang_corner_is_interior(
    pt: Tuple[float, float],
    segments_overhang: List[List[List[float]]],
    tol: float = 3.0,
) -> bool:
    """True dacă pt este un colț re-entrant (interior) al poligonului overhang."""
    if not segments_overhang:
        return False
    segs = [(float(s[0][0]), float(s[0][1]), float(s[1][0]), float(s[1][1])) for s in segments_overhang if len(s) >= 2]
    if not segs:
        return False
    # Ordine vârfuri overhang
    ordered: List[Tuple[float, float]] = []
    (x0, y0, x1, y1) = segs[0]
    ordered.append((x0, y0))
    ordered.append((x1, y1))
    used = {0}
    while len(used) < len(segs):
        tail = ordered[-1]
        found = False
        for idx, (a0, a1, b0, b1) in enumerate(segs):
            if idx in used:
                continue
            t = 1e-6
            if abs(tail[0] - a0) < t and abs(tail[1] - a1) < t:
                ordered.append((b0, b1))
                used.add(idx)
                found = True
                break
            if abs(tail[0] - b0) < t and abs(tail[1] - b1) < t:
                ordered.append((a0, a1))
                used.add(idx)
                found = True
                break
        if not found:
            break
    pk = (round(pt[0], 2), round(pt[1], 2))
    for i, v in enumerate(ordered):
        vk = (round(v[0], 2), round(v[1], 2))
        if (pk[0] - vk[0]) ** 2 + (pk[1] - vk[1]) ** 2 > tol * tol:
            continue
        prev = ordered[(i - 1) % len(ordered)]
        nxt = ordered[(i + 1) % len(ordered)]
        v1 = (prev[0] - v[0], prev[1] - v[1])
        v2 = (nxt[0] - v[0], nxt[1] - v[1])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        return cross < -1e-6
    return False


def _overhang_anchor_segments(
    overhang_corner_pts: List[Tuple[float, float]],
    overhang_ridge_pts: List[Tuple[float, float]],
    segments_ridge: List[List[List[float]]],
    segments_contour: List[List[List[float]]],
    segments_overhang: List[List[List[float]]],
    segments_pyramid: Optional[List[List[List[float]]]] = None,
    roof_type: str = "2_w",
    sections: Optional[List[Dict[str, Any]]] = None,
) -> List[List[List[float]]]:
    """Segmente de la fiecare bulină (albastru deschis/închis) la anchor-ul ei (gri). Pentru polygonize fețe overhang."""
    out: List[List[List[float]]] = []
    ridge_tuples: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for seg in (segments_ridge or []):
        if len(seg) >= 2:
            a = (float(seg[0][0]), float(seg[0][1]))
            b = (float(seg[1][0]), float(seg[1][1]))
            ridge_tuples.append((a, b))
    contour_corners: List[Tuple[float, float]] = []
    seen_cc = set()
    for seg in (segments_contour or []):
        if len(seg) >= 2:
            for p in (seg[0], seg[1]):
                q = (round(float(p[0]), 2), round(float(p[1]), 2))
                if q not in seen_cc:
                    seen_cc.add(q)
                    contour_corners.append((float(p[0]), float(p[1])))
    contour_interior = _contour_interior_corners(segments_contour) if segments_contour else []
    section_corners_list: List[List[Tuple[float, float]]] = []
    contour_corner_to_section: Dict[Tuple[float, float], List[int]] = {}
    if sections and contour_corners:
        for sec in sections:
            br = sec.get("bounding_rect") or []
            if len(br) < 3:
                continue
            xs = [float(p[0]) for p in br]
            ys = [float(p[1]) for p in br]
            mnx, mxx = min(xs), max(xs)
            mny, mxy = min(ys), max(ys)
            section_corners_list.append([(mnx, mny), (mxx, mny), (mxx, mxy), (mnx, mxy)])
        _tol_rect = 2.0
        for c in contour_corners:
            rc = (round(c[0], 2), round(c[1], 2))
            contour_corner_to_section[rc] = []
            for i, sc_list in enumerate(section_corners_list):
                for sc in sc_list:
                    if (c[0] - sc[0]) ** 2 + (c[1] - sc[1]) ** 2 <= _tol_rect * _tol_rect:
                        contour_corner_to_section[rc].append(i)
                        break
    pyramid_tuples: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for seg in (segments_pyramid or []):
        if len(seg) >= 2:
            a = (float(seg[0][0]), float(seg[0][1]))
            b = (float(seg[-1][0]), float(seg[-1][1]))
            pyramid_tuples.append((a, b))
    # Poligon overhang: pentru TOATE tipurile – anchor doar colțuri STRICT ÎN INTERIOR
    overhang_poly_anchor = None
    if segments_overhang and len(segments_overhang) >= 3 and roof_type in ("0_w", "1_w", "2_w", "4_w", "4.5_w"):
        try:
            from shapely.geometry import LineString
            from shapely.ops import polygonize_full, unary_union
            oh_lines = [
                LineString([(float(s[0][0]), float(s[0][1])), (float(s[1][0]), float(s[1][1]))])
                for s in segments_overhang if len(s) >= 2
                and (float(s[0][0]) - float(s[1][0])) ** 2 + (float(s[0][1]) - float(s[1][1])) ** 2 > 1e-6
            ]
            if len(oh_lines) >= 3:
                polys, _, _, _ = polygonize_full(oh_lines)
                geoms = list(getattr(polys, "geoms", None) or [polys])
                valid = [g for g in geoms if g is not None and not getattr(g, "is_empty", True)]
                if valid:
                    overhang_poly_anchor = unary_union(valid) if len(valid) > 1 else valid[0]
        except Exception:
            overhang_poly_anchor = None
    for p in overhang_ridge_pts:
        pt = (float(p[0]), float(p[1]))
        if ridge_tuples:
            best_ax, best_ay, best_d2 = 0.0, 0.0, 1e30
            for (a, b) in ridge_tuples:
                qx, qy, d2 = _closest_point_on_segment(pt, a, b)
                if d2 < best_d2:
                    best_d2 = d2
                    best_ax, best_ay = qx, qy
            out.append([[pt[0], pt[1]], [best_ax, best_ay]])
    for p in overhang_corner_pts:
        pt = (float(p[0]), float(p[1]))
        use_interior = _overhang_corner_is_interior(pt, segments_overhang) and bool(contour_interior)
        anchor_candidates = contour_interior if use_interior else contour_corners
        # FIX v4: Logică unificată anchor (oglindă cu draw_lines)
        best_ax, best_ay = pt[0], pt[1]
        _used_interior_v4 = False
        if not use_interior and contour_interior:
            best_int_d2 = 1e30
            best_int_ax, best_int_ay = pt[0], pt[1]
            for q in contour_interior:
                d2 = (pt[0] - q[0]) ** 2 + (pt[1] - q[1]) ** 2
                if d2 < best_int_d2:
                    best_int_d2 = d2
                    best_int_ax, best_int_ay = q[0], q[1]
            best_ext_d2 = min(
                ((pt[0] - q[0]) ** 2 + (pt[1] - q[1]) ** 2)
                for q in contour_corners
            ) if contour_corners else 1e30
            if best_int_d2 <= best_ext_d2 * 2.5:
                _used_interior_v4 = True
                best_ax, best_ay = best_int_ax, best_int_ay
        if not use_interior and not _used_interior_v4:
            if roof_type in ("4_w", "4.5_w") and (pyramid_tuples or contour_corners):
                # 4_w, 4.5_w: anchor = colț (contur sau piramidă) STRICT ÎN INTERIOR overhang
                anchor_candidates_45 = list(contour_corners) if contour_corners else []
                for (a, b) in (pyramid_tuples or []):
                    anchor_candidates_45.append(a)
                    anchor_candidates_45.append(b)
                anchor_candidates_45 = list(dict.fromkeys([(round(c[0], 4), round(c[1], 4)) for c in anchor_candidates_45]))
                anchor_candidates_45 = [(c[0], c[1]) for c in anchor_candidates_45]
                if overhang_poly_anchor is not None:
                    try:
                        from shapely.geometry import Point as ShapelyPoint
                        _tol = 1e-6
                        def _strictly_inside_45(c):
                            pt = ShapelyPoint(c[0], c[1])
                            return (
                                overhang_poly_anchor.contains(pt)
                                and overhang_poly_anchor.boundary.distance(pt) > _tol
                            )
                        anchor_candidates_45 = [c for c in anchor_candidates_45 if _strictly_inside_45(c)]
                    except Exception:
                        pass
                if anchor_candidates_45:
                    best_d2 = 1e30
                    for q in anchor_candidates_45:
                        d2 = (pt[0] - q[0]) ** 2 + (pt[1] - q[1]) ** 2
                        if d2 < best_d2:
                            best_d2 = d2
                            best_ax, best_ay = q[0], q[1]
                    out.append([[pt[0], pt[1]], [best_ax, best_ay]])
                    continue
                # else: niciun colț în interior → nu adăugăm segment, trecem la următoarea bulină
            elif roof_type in ("4_w", "4.5_w") and pyramid_tuples:
                best_d2 = 1e30
                for (a, b) in pyramid_tuples:
                    qx, qy, d2 = _closest_point_on_segment(pt, a, b)
                    if d2 < best_d2:
                        best_d2 = d2
                        best_ax, best_ay = qx, qy
                out.append([[pt[0], pt[1]], [best_ax, best_ay]])
                continue
            else:
                # 2_w, 1_w, 0_w: la fel ca 4_w — TOATE colțurile conturului, doar cele STRICT ÎN INTERIOR overhang, cel mai apropiat
                anchor_candidates = list(contour_corners) if contour_corners else []
                if not anchor_candidates:
                    continue
                # Doar colțuri STRICT ÎN INTERIORUL overhang-ului
                if overhang_poly_anchor is not None:
                    try:
                        from shapely.geometry import Point as ShapelyPoint
                        _tol = 1e-6
                        def _strictly_inside_oh(c):
                            pt = ShapelyPoint(c[0], c[1])
                            return (
                                overhang_poly_anchor.contains(pt)
                                and overhang_poly_anchor.boundary.distance(pt) > _tol
                            )
                        anchor_candidates = [c for c in anchor_candidates if _strictly_inside_oh(c)]
                    except Exception:
                        pass
                if not anchor_candidates:
                    continue
                best_d2 = 1e30
                for q in anchor_candidates:
                    d2 = (pt[0] - q[0]) ** 2 + (pt[1] - q[1]) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        best_ax, best_ay = q[0], q[1]
        elif use_interior and anchor_candidates:
            # Doar colțuri interioare STRICT ÎN INTERIORUL overhang-ului
            if overhang_poly_anchor is not None:
                try:
                    from shapely.geometry import Point as ShapelyPoint
                    _tol = 1e-6
                    def _strictly_inside_int(c):
                        pt = ShapelyPoint(c[0], c[1])
                        return (
                            overhang_poly_anchor.contains(pt)
                            and overhang_poly_anchor.boundary.distance(pt) > _tol
                        )
                    anchor_candidates = [c for c in anchor_candidates if _strictly_inside_int(c)]
                except Exception:
                    pass
            if not anchor_candidates:
                continue
            best_d2 = 1e30
            for q in anchor_candidates:
                d2 = (pt[0] - q[0]) ** 2 + (pt[1] - q[1]) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_ax, best_ay = q[0], q[1]
        else:
            continue
        out.append([[pt[0], pt[1]], [best_ax, best_ay]])
    return out


def _ordered_overhang_boundary_pts(
    segments_overhang: List[List[List[float]]],
    overhang_ridge_pts: List[List[float]],
    tol: float = 2.0,
) -> List[Tuple[float, float]]:
    """Inelul overhang în ordine: colțuri din segmente + puncte ridge inserate pe segmente. Pentru 3D ca segmentul să treacă prin buline."""
    if not segments_overhang:
        return []
    segs = [(float(s[0][0]), float(s[0][1]), float(s[1][0]), float(s[1][1])) for s in segments_overhang if len(s) >= 2]
    if not segs:
        return []
    # Lanțuire segmente → listă ordonată de colțuri (închis)
    ordered: List[Tuple[float, float]] = []
    (x0, y0, x1, y1) = segs[0]
    ordered.append((x0, y0))
    ordered.append((x1, y1))
    used = {0}
    while len(used) < len(segs):
        tail = ordered[-1]
        found = False
        for idx, (a0, a1, b0, b1) in enumerate(segs):
            if idx in used:
                continue
            t = 1e-6
            if abs(tail[0] - a0) < t and abs(tail[1] - a1) < t:
                ordered.append((b0, b1))
                used.add(idx)
                found = True
                break
            if abs(tail[0] - b0) < t and abs(tail[1] - b1) < t:
                ordered.append((a0, a1))
                used.add(idx)
                found = True
                break
        if not found:
            break
    if len(ordered) < 3:
        return ordered
    # Inserăm punctele ridge pe segmente (parametru t în (0,1))
    ridge_pts = [(float(p[0]), float(p[1])) for p in overhang_ridge_pts if len(p) >= 2]
    if not ridge_pts:
        return ordered
    result: List[Tuple[float, float]] = []
    for i in range(len(ordered)):
        result.append(ordered[i])
        a, b = ordered[i], ordered[(i + 1) % len(ordered)]
        ax, ay = a[0], a[1]
        bx, by = b[0], b[1]
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        on_seg: List[Tuple[float, Tuple[float, float]]] = []
        for (rx, ry) in ridge_pts:
            if seg_len_sq < 1e-20:
                continue
            t = ((rx - ax) * dx + (ry - ay) * dy) / seg_len_sq
            if not (0 < t < 1):
                continue
            proj_x = ax + t * dx
            proj_y = ay + t * dy
            if (rx - proj_x) ** 2 + (ry - proj_y) ** 2 <= tol * tol:
                on_seg.append((t, (rx, ry)))
        for _, pt in sorted(on_seg, key=lambda x: x[0]):
            result.append(pt)
    return result


def _segments_perpendicular(
    a1: Tuple[float, float], a2: Tuple[float, float],
    b1: Tuple[float, float], b2: Tuple[float, float],
    tol: float = 2.0,
) -> bool:
    """True dacă segmentele (a1,a2) și (b1,b2) sunt perpendiculare."""
    ax, ay = a2[0] - a1[0], a2[1] - a1[1]
    bx, by = b2[0] - b1[0], b2[1] - b1[1]
    dot = ax * bx + ay * by
    return abs(dot) < tol * (abs(ax) + abs(ay) + abs(bx) + abs(by) + 1e-9)


def _get_wall_support_segments(
    sections: List[Dict[str, Any]],
    ridge_segments: List[List[List[float]]],
    upper_rect_segments: List[List[List[float]]],
    contour_segments: List[List[List[float]]],
    tol: float = 3.0,
) -> List[List[List[float]]]:
    """
    Segmente suport: de la capul ridge-ului (unde atinge galbenul) până la capetele
    segmentelor verzi care sunt perpendiculare cu peretele galben de etaj superior.
    Capetele suportului trebuie să fie exact pe capetele (ca, cb) ale acelor segmente.
    """
    from roof_calc.overhang import extend_secondary_sections_to_main_ridge

    out: List[List[List[float]]] = []
    seen: set = set()
    extended = extend_secondary_sections_to_main_ridge(sections)

    def _section_for_ridge(rseg_ra: Tuple[float, float], rseg_rb: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        for sec in extended:
            ridge = sec.get("ridge_line") or []
            if len(ridge) < 2:
                continue
            sra = (float(ridge[0][0]), float(ridge[0][1]))
            srb = (float(ridge[1][0]), float(ridge[1][1]))
            if _point_on_segment(rseg_ra, sra, srb, tol) or _point_on_segment(rseg_rb, sra, srb, tol):
                return sec
        return None

    def _point_in_rect(pt: Tuple[float, float], sec: Dict[str, Any], margin: float = 1.0) -> bool:
        br = sec.get("bounding_rect") or []
        if len(br) < 3:
            return False
        xs = [float(p[0]) for p in br]
        ys = [float(p[1]) for p in br]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        return minx - margin <= pt[0] <= maxx + margin and miny - margin <= pt[1] <= maxy + margin

    def _perpendicular_green_segments_for_ridge_on_yellow(
        ridge_pt: Tuple[float, float], sec: Dict[str, Any]
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Găsește segmentele verzi (contur) care sunt perpendiculare pe peretele galben
        de etaj superior pe care ridge_pt atinge. Returnează lista de (ca, cb).
        """
        result: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for useg in upper_rect_segments:
            if len(useg) < 2:
                continue
            ua = (float(useg[0][0]), float(useg[0][1]))
            ub = (float(useg[1][0]), float(useg[1][1]))
            if not _point_on_segment(ridge_pt, ua, ub, tol):
                continue
            for cseg in contour_segments:
                if len(cseg) < 2:
                    continue
                ca = (float(cseg[0][0]), float(cseg[0][1]))
                cb = (float(cseg[1][0]), float(cseg[1][1]))
                if not _point_in_rect(ca, sec) and not _point_in_rect(cb, sec):
                    continue
                if not _segments_perpendicular(ua, ub, ca, cb, tol):
                    continue
                touches_yellow = False
                for us in upper_rect_segments:
                    if len(us) < 2:
                        continue
                    usa, usb = (float(us[0][0]), float(us[0][1])), (float(us[1][0]), float(us[1][1]))
                    if _point_on_segment(ca, usa, usb, tol) or _point_on_segment(cb, usa, usb, tol):
                        touches_yellow = True
                        break
                if not touches_yellow:
                    continue
                result.append((ca, cb))
            if result:
                return result
        return result

    for rseg in ridge_segments:
        if len(rseg) < 2:
            continue
        ra = (float(rseg[0][0]), float(rseg[0][1]))
        rb = (float(rseg[1][0]), float(rseg[1][1]))
        sec = _section_for_ridge(ra, rb)
        if sec is None:
            continue
        for ridge_pt in (ra, rb):
            green_segs = _perpendicular_green_segments_for_ridge_on_yellow(ridge_pt, sec)
            if not green_segs:
                br = sec.get("bounding_rect") or []
                if len(br) >= 3:
                    on_yellow = any(
                        _point_on_segment(ridge_pt, (useg[0][0], useg[0][1]), (useg[1][0], useg[1][1]), tol)
                        for useg in (upper_rect_segments or []) if len(useg) >= 2
                    )
                    if on_yellow:
                        xs = [float(p[0]) for p in br]
                        ys = [float(p[1]) for p in br]
                        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
                        is_horiz = abs(ra[1] - rb[1]) < 1e-6
                        if is_horiz and abs(ra[1] - miny) < abs(ra[1] - maxy):
                            green_segs = [((minx, miny), (minx, maxy)), ((maxx, miny), (maxx, maxy))]
                        elif is_horiz:
                            green_segs = [((minx, miny), (minx, maxy)), ((maxx, miny), (maxx, maxy))]
                        elif abs(ra[0] - minx) < abs(ra[0] - maxx):
                            green_segs = [((minx, miny), (maxx, miny)), ((minx, maxy), (maxx, maxy))]
                        else:
                            green_segs = [((minx, miny), (maxx, miny)), ((minx, maxy), (maxx, maxy))]
            def _endpoint_touches_yellow(pt: Tuple[float, float]) -> bool:
                for useg in (upper_rect_segments or []):
                    if len(useg) < 2:
                        continue
                    ua = (float(useg[0][0]), float(useg[0][1]))
                    ub = (float(useg[1][0]), float(useg[1][1]))
                    if _point_on_segment(pt, ua, ub, tol):
                        return True
                return False

            for ca, cb in green_segs:
                for tgt in (ca, cb):
                    if not _endpoint_touches_yellow(tgt):
                        continue
                    k = _segment_key([ridge_pt[0], ridge_pt[1]], [tgt[0], tgt[1]], tol=tol)
                    if k in seen:
                        continue
                    dist_sq = (ridge_pt[0] - tgt[0]) ** 2 + (ridge_pt[1] - tgt[1]) ** 2
                    if dist_sq < 1e-6:
                        continue
                    seen.add(k)
                    out.append([[ridge_pt[0], ridge_pt[1]], [tgt[0], tgt[1]]])
    return out


def _get_ridge_diagonal_markers_45w(
    sections: List[Dict[str, Any]],
    upper_floor_sections: Optional[List[Dict[str, Any]]] = None,
    pyramid_segments: Optional[List[List[List[float]]]] = None,
    segments_orange: Optional[List[List[List[float]]]] = None,
) -> List[Tuple[float, float]]:
    """
    Pentru fiecare ridge:
    - Bulina 1: exact pe diagonalele albastre, unde vârfurile lor ating ridge-ul
    - Bulina 2: capătul opus al ridge-ului – din punctul unde portocaliul intersectează ridge-ul
                mergem în direcția opusă și luăm capul ridge-ului
    """
    from roof_calc.overhang import extend_secondary_sections_to_main_ridge

    extended = extend_secondary_sections_to_main_ridge(sections)
    if not pyramid_segments:
        return []

    ridge_pts = [(float(seg[1][0]), float(seg[1][1])) for seg in pyramid_segments if len(seg) >= 2]
    if not ridge_pts:
        return []

    orange_segs = segments_orange or []
    out: List[Tuple[float, float]] = []
    tol = 2.0
    for sec in extended:
        ridge = sec.get("ridge_line") or []
        if len(ridge) < 2:
            continue
        ra = (float(ridge[0][0]), float(ridge[0][1]))
        rb = (float(ridge[1][0]), float(ridge[1][1]))
        is_vert = abs(ra[0] - rb[0]) < 1e-6
        rminx, rmaxx = min(ra[0], rb[0]), max(ra[0], rb[0])
        rminy, rmaxy = min(ra[1], rb[1]), max(ra[1], rb[1])
        ridge_y = ra[1]
        ridge_x = ra[0]
        pts_on_ridge = [
            p for p in ridge_pts
            if (is_vert and abs(p[0] - ridge_x) < tol and rminy - tol <= p[1] <= rmaxy + tol)
            or (not is_vert and abs(p[1] - ridge_y) < tol and rminx - tol <= p[0] <= rmaxx + tol)
        ]
        if not pts_on_ridge:
            continue
        x_lo = min(p[0] for p in pts_on_ridge)
        x_hi = max(p[0] for p in pts_on_ridge)
        y_lo = min(p[1] for p in pts_on_ridge)
        y_hi = max(p[1] for p in pts_on_ridge)
        touch = ((x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0)
        orange_int: Optional[Tuple[float, float]] = None
        for oseg in orange_segs:
            if len(oseg) < 2:
                continue
            pt = _seg_intersect(ra, rb, (oseg[0][0], oseg[0][1]), (oseg[1][0], oseg[1][1]))
            if pt is not None:
                orange_int = pt
                break
        attached = _sides_attached_to_upper(sec, upper_floor_sections or [], same_floor_sections=sections)
        if attached:
            uf = upper_floor_sections or []
            if uf:
                ux = []
                uy = []
                for u in uf:
                    ub = u.get("bounding_rect") or []
                    if len(ub) >= 3:
                        ux.extend(float(p[0]) for p in ub)
                        uy.extend(float(p[1]) for p in ub)
                if ux and uy:
                    ucx = sum(ux) / len(ux)
                    ucy = sum(uy) / len(uy)
                    d_ra = (ra[0] - ucx) ** 2 + (ra[1] - ucy) ** 2
                    d_rb = (rb[0] - ucx) ** 2 + (rb[1] - ucy) ** 2
                    opposite = ra if d_ra < d_rb else rb
                else:
                    d_ra = (ra[0] - touch[0]) ** 2 + (ra[1] - touch[1]) ** 2
                    d_rb = (rb[0] - touch[0]) ** 2 + (rb[1] - touch[1]) ** 2
                    opposite = rb if d_rb > d_ra else ra
            else:
                d_ra = (ra[0] - touch[0]) ** 2 + (ra[1] - touch[1]) ** 2
                d_rb = (rb[0] - touch[0]) ** 2 + (rb[1] - touch[1]) ** 2
                opposite = rb if d_rb > d_ra else ra
        elif orange_int is not None:
            d_ra = (ra[0] - orange_int[0]) ** 2 + (ra[1] - orange_int[1]) ** 2
            d_rb = (rb[0] - orange_int[0]) ** 2 + (rb[1] - orange_int[1]) ** 2
            opposite = rb if d_rb > d_ra else ra
        else:
            d_ra = (ra[0] - touch[0]) ** 2 + (ra[1] - touch[1]) ** 2
            d_rb = (rb[0] - touch[0]) ** 2 + (rb[1] - touch[1]) ** 2
            opposite = rb if d_rb > d_ra else ra
        base_pts = [
            (float(seg[0][0]), float(seg[0][1]))
            for seg in pyramid_segments
            if len(seg) >= 2 and any(
                abs(float(seg[1][0]) - q[0]) < tol and abs(float(seg[1][1]) - q[1]) < tol
                for q in pts_on_ridge
            )
        ]

        def _on_green_eliminated(pt: Tuple[float, float]) -> bool:
            for i, bp1 in enumerate(base_pts):
                for bp2 in base_pts[i + 1:]:
                    if _point_on_segment(pt, bp1, bp2, tol):
                        return True
            for oseg in orange_segs:
                if len(oseg) >= 2 and _point_on_segment(pt, (oseg[0][0], oseg[0][1]), (oseg[1][0], oseg[1][1]), tol):
                    return True
            return False

        if _on_green_eliminated(opposite):
            opposite = rb if opposite == ra else ra
        out.append(touch)
        out.append((opposite[0], opposite[1]))

    return out


def _get_ridge_segments_45w_trimmed(
    sections: List[Dict[str, Any]],
    upper_floor_sections: Optional[List[Dict[str, Any]]] = None,
    pyramid_segments: Optional[List[List[List[float]]]] = None,
    segments_orange: Optional[List[List[List[float]]]] = None,
) -> List[List[List[float]]]:
    """
    Diagonalele au capetele pe ridge. Segmentul de ridge parcurs de diagonale e șters.
    Prelungirea către ridge-ul principal rămâne. Pentru segmentele unde ridge-ul s-a eliminat
    complet (extent degenerat), îl reconstruim între cele două buline roz (touch, opposite).
    """
    from roof_calc.overhang import extend_secondary_sections_to_main_ridge

    extended = extend_secondary_sections_to_main_ridge(sections)
    if not pyramid_segments:
        return _get_ridge_segments(sections)

    main_sec = next((s for s in extended if s.get("is_main")), None)
    if main_sec is None:
        main_sec = max(extended, key=lambda s: _rect_area(s))

    ridge_pts = []
    for seg in pyramid_segments:
        if len(seg) >= 2:
            ridge_pts.append((float(seg[1][0]), float(seg[1][1])))

    orange_segs = segments_orange or []
    segs: List[List[List[float]]] = []
    tol = 2.0
    for sec in extended:
        ridge = sec.get("ridge_line") or []
        if len(ridge) < 2:
            continue
        ra = (float(ridge[0][0]), float(ridge[0][1]))
        rb = (float(ridge[1][0]), float(ridge[1][1]))
        is_vert = abs(ra[0] - rb[0]) < 1e-6
        rminx, rmaxx = min(ra[0], rb[0]), max(ra[0], rb[0])
        rminy, rmaxy = min(ra[1], rb[1]), max(ra[1], rb[1])
        ridge_y = ra[1]
        ridge_x = ra[0]
        pts_on_ridge = [
            p for p in ridge_pts
            if (is_vert and abs(p[0] - ridge_x) < tol and rminy - tol <= p[1] <= rmaxy + tol)
            or (not is_vert and abs(p[1] - ridge_y) < tol and rminx - tol <= p[0] <= rmaxx + tol)
        ]
        main_conn = _get_main_ridge_connection(sec, main_sec, extended)
        if not pts_on_ridge:
            segs.append([[ra[0], ra[1]], [rb[0], rb[1]]])
            continue
        if is_vert:
            x_lo, x_hi = min(p[0] for p in pts_on_ridge), max(p[0] for p in pts_on_ridge)
            y_lo, y_hi = min(p[1] for p in pts_on_ridge), max(p[1] for p in pts_on_ridge)
        else:
            x_lo, x_hi = min(p[0] for p in pts_on_ridge), max(p[0] for p in pts_on_ridge)
            y_lo, y_hi = min(p[1] for p in pts_on_ridge), max(p[1] for p in pts_on_ridge)
        touch = ((x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0)
        orange_int: Optional[Tuple[float, float]] = None
        for oseg in orange_segs:
            if len(oseg) < 2:
                continue
            pt = _seg_intersect(ra, rb, (oseg[0][0], oseg[0][1]), (oseg[1][0], oseg[1][1]))
            if pt is not None:
                orange_int = pt
                break
        attached = _sides_attached_to_upper(sec, upper_floor_sections or [], same_floor_sections=sections)
        if attached and (upper_floor_sections or []):
            ux, uy = [], []
            for u in (upper_floor_sections or []):
                ub = u.get("bounding_rect") or []
                if len(ub) >= 3:
                    ux.extend(float(p[0]) for p in ub)
                    uy.extend(float(p[1]) for p in ub)
            if ux and uy:
                ucx, ucy = sum(ux) / len(ux), sum(uy) / len(uy)
                d_ra = (ra[0] - ucx) ** 2 + (ra[1] - ucy) ** 2
                d_rb = (rb[0] - ucx) ** 2 + (rb[1] - ucy) ** 2
                opposite = ra if d_ra < d_rb else rb
            else:
                opposite = rb if (rb[0] - touch[0]) ** 2 + (rb[1] - touch[1]) ** 2 > (ra[0] - touch[0]) ** 2 + (ra[1] - touch[1]) ** 2 else ra
        elif orange_int is not None:
            d_ra = (ra[0] - orange_int[0]) ** 2 + (ra[1] - orange_int[1]) ** 2
            d_rb = (rb[0] - orange_int[0]) ** 2 + (rb[1] - orange_int[1]) ** 2
            opposite = rb if d_rb > d_ra else ra
        else:
            d_ra = (ra[0] - touch[0]) ** 2 + (ra[1] - touch[1]) ** 2
            d_rb = (rb[0] - touch[0]) ** 2 + (rb[1] - touch[1]) ** 2
            opposite = rb if d_rb > d_ra else ra
        base_pts_trim = [
            (float(seg[0][0]), float(seg[0][1]))
            for seg in pyramid_segments
            if len(seg) >= 2 and any(abs(float(seg[1][0]) - q[0]) < tol and abs(float(seg[1][1]) - q[1]) < tol for q in pts_on_ridge)
        ]

        def _on_green(p: Tuple[float, float]) -> bool:
            for i, bp1 in enumerate(base_pts_trim):
                for bp2 in base_pts_trim[i + 1:]:
                    if _point_on_segment(p, bp1, bp2, tol):
                        return True
            for oseg in orange_segs:
                if len(oseg) >= 2 and _point_on_segment(p, (oseg[0][0], oseg[0][1]), (oseg[1][0], oseg[1][1]), tol):
                    return True
            return False

        if _on_green(opposite):
            opposite = rb if opposite == ra else ra
        ext_degen = abs(x_hi - x_lo) < 1e-3 and abs(y_hi - y_lo) < 1e-3
        if ext_degen:
            segs.append([[touch[0], touch[1]], [opposite[0], opposite[1]]])
            continue
        out: List[List[List[float]]] = []
        if is_vert:
            if rmaxy > y_lo + 1e-6 and rminy < y_hi - 1e-6:
                out.append([[ra[0], y_lo], [ra[0], y_hi]])
            if main_conn is not None:
                my_conn = main_conn[1]
                if my_conn < y_lo - 1e-6 and rminy < y_lo - 1e-6:
                    out.append([[ra[0], my_conn], [ra[0], y_lo]])
                elif my_conn > y_hi + 1e-6 and rmaxy > y_hi + 1e-6:
                    out.append([[ra[0], y_hi], [ra[0], my_conn]])
        else:
            if rmaxx > x_lo + 1e-6 and rminx < x_hi - 1e-6:
                out.append([[x_lo, ridge_y], [x_hi, ridge_y]])
            if main_conn is not None:
                mx_conn = main_conn[0]
                if mx_conn < x_lo - 1e-6 and rminx < x_lo - 1e-6:
                    out.append([[mx_conn, ridge_y], [x_lo, ridge_y]])
                elif mx_conn > x_hi + 1e-6 and rmaxx > x_hi + 1e-6:
                    out.append([[x_hi, ridge_y], [mx_conn, ridge_y]])
        if out:
            segs.extend(out)
        else:
            segs.append([[ra[0], ra[1]], [rb[0], rb[1]]])

    return segs if segs else _get_ridge_segments(sections)


def _get_magenta_segments(sections: List[Dict[str, Any]]) -> List[List[List[float]]]:
    """Linii mov doar când ≥2 ridge-uri se intersectează."""
    if not _has_ridge_intersection(sections):
        return []
    from roof_calc.overhang import ridge_intersection_corner_lines

    corner_lines = ridge_intersection_corner_lines(sections, per_section=False)
    segs: List[List[List[float]]] = []
    for item in corner_lines:
        pt, corners, _ = item
        ix, iy = float(pt[0]), float(pt[1])
        for c in corners:
            segs.append([[ix, iy], [float(c[0]), float(c[1])]])
    return segs


def _get_magenta_segments_45w(
    seg_orange: Optional[List[List[List[float]]]],
    seg_contour_green: List[List[List[float]]],
) -> List[List[List[float]]]:
    """
    Linii mov pentru 4.5_w: între un capăt de segment portocaliu și un capăt de segment VERDE
    (colț al conturului verde), NU la punctele de teșire/mijloc care sunt capetele portocaliului.
    Fiecare capăt portocaliu se leagă de cel mai apropiat capăt verde care NU e capăt portocaliu.
    """
    if not seg_orange or not seg_contour_green:
        return []
    orange_pts: List[Tuple[float, float]] = []
    for s in seg_orange:
        if len(s) >= 2:
            orange_pts.append((float(s[0][0]), float(s[0][1])))
            orange_pts.append((float(s[1][0]), float(s[1][1])))
    orange_pts = list(dict.fromkeys([(round(p[0], 4), round(p[1], 4)) for p in orange_pts]))
    orange_pts = [(p[0], p[1]) for p in orange_pts]

    green_pts: List[Tuple[float, float]] = []
    for s in seg_contour_green:
        if len(s) >= 2:
            green_pts.append((float(s[0][0]), float(s[0][1])))
            green_pts.append((float(s[1][0]), float(s[1][1])))

    # Excludem din verde capetele care sunt (aproape) capete portocalii = punctele de teșire/mijloc
    tol_same = 2.0
    tol_same_sq = tol_same * tol_same
    green_only_corners = [
        (gx, gy) for (gx, gy) in green_pts
        if not any((gx - ox) ** 2 + (gy - oy) ** 2 <= tol_same_sq for (ox, oy) in orange_pts)
    ]
    green_pts = green_only_corners if green_only_corners else green_pts

    if not orange_pts or not green_pts:
        return []
    tol_sq = 1.0 ** 2
    out: List[List[List[float]]] = []
    for (ox, oy) in orange_pts:
        best_d2 = 1e30
        best_g = None
        for (gx, gy) in green_pts:
            d2 = (ox - gx) ** 2 + (oy - gy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_g = (gx, gy)
        if best_g is not None and best_d2 > tol_sq:
            out.append([[ox, oy], [best_g[0], best_g[1]]])
    return out


def _get_blue_segment_1w(sections: List[Dict[str, Any]]) -> List[List[List[float]]]:
    """
    Segment albastru pentru 1_w (shed): folosește aceleași reguli ca ridge-ul.
    Ridge-ul vine din decomposiția acoperișului (ridge_line, ridge_orientation).
    """
    return _get_ridge_segments(sections)


def _snap_segment_endpoints(
    segs: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    tol: float = 5.0,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Snap endpoint-uri care sunt la distanță <= tol la același punct pentru a asigura conectare la polygonize."""
    if not segs:
        return []
    all_pts_ordered: List[Tuple[float, float]] = []
    seen_pt: set = set()
    for p1, p2 in segs:
        for p in (p1, p2):
            k = (round(p[0], 2), round(p[1], 2))
            if k not in seen_pt:
                seen_pt.add(k)
                all_pts_ordered.append(p)
    n = len(all_pts_ordered)
    parent = list(range(n))

    def find(i: int) -> int:
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def unite(i: int, j: int) -> None:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        for j in range(i + 1, n):
            d2 = (all_pts_ordered[i][0] - all_pts_ordered[j][0]) ** 2 + (all_pts_ordered[i][1] - all_pts_ordered[j][1]) ** 2
            if d2 <= tol * tol:
                unite(i, j)

    cluster_pts: Dict[int, List[Tuple[float, float]]] = {}
    for i in range(n):
        r = find(i)
        if r not in cluster_pts:
            cluster_pts[r] = []
        cluster_pts[r].append(all_pts_ordered[i])
    centroid: Dict[int, Tuple[float, float]] = {}
    for r, pts in cluster_pts.items():
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        centroid[r] = (cx, cy)

    def _pt_to_snapped(p: Tuple[float, float]) -> Tuple[float, float]:
        best_i, best_d2 = 0, 1e30
        for i in range(n):
            d2 = (p[0] - all_pts_ordered[i][0]) ** 2 + (p[1] - all_pts_ordered[i][1]) ** 2
            if d2 < best_d2:
                best_d2, best_i = d2, i
        return centroid[find(best_i)]

    seen_out: set = set()
    out: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for p1, p2 in segs:
        np1 = _pt_to_snapped(p1)
        np2 = _pt_to_snapped(p2)
        if (np1[0] - np2[0]) ** 2 + (np1[1] - np2[1]) ** 2 < 1e-10:
            continue
        key = (np1, np2) if np1 < np2 else (np2, np1)
        if key in seen_out:
            continue
        seen_out.add(key)
        out.append((np1, np2))
    return out


def _polygons_from_line_segments(
    *segment_lists: List[List[List[float]]],
    decimals: int = 2,
    min_area: float = 0.5,
    max_area_ratio: float = 0.98,
    exclude_interior_of: Optional[List[List[List[float]]]] = None,
) -> List[List[Tuple[float, float]]]:
    """
    Generează poligoane din segmente folosind shapely polygonize_full.
    Fiecare poligon = listă de (x,y) – exteriorul închis.
    exclude_interior_of: segmente (ex. upper_rect) – poligoanele din interiorul acestora sunt excluse.
    """
    seen: set = set()
    all_segs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for lst in segment_lists:
        if not lst:
            continue
        for seg in lst:
            if len(seg) < 2:
                continue
            p1 = (round(float(seg[0][0]), decimals), round(float(seg[0][1]), decimals))
            p2 = (round(float(seg[1][0]), decimals), round(float(seg[1][1]), decimals))
            if (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 < 1e-10:
                continue
            key = (p1, p2) if p1 < p2 else (p2, p1)
            if key in seen:
                continue
            seen.add(key)
            all_segs.append((p1, p2))
    all_segs = _snap_segment_endpoints(all_segs, tol=5.0)
    if not all_segs:
        return []
    try:
        from shapely.geometry import LineString, MultiLineString
        from shapely.ops import polygonize_full

        lines = [LineString([p1, p2]) for p1, p2 in all_segs]
        noded_lines = None
        try:
            from shapely.ops import node as shapely_node
            ml = MultiLineString(lines)
            noded = shapely_node(ml)
            if noded is not None and not getattr(noded, "is_empty", True):
                segs_out = []
                for g in (getattr(noded, "geoms", None) or [noded]):
                    c = list(getattr(g, "coords", []))
                    for i in range(len(c) - 1):
                        segs_out.append(LineString([c[i], c[i + 1]]))
                if segs_out:
                    noded_lines = segs_out
        except (ImportError, AttributeError, Exception):
            pass
        if noded_lines is None:
            try:
                from shapely.ops import unary_union
                merged = unary_union(lines)
                if merged is not None and not getattr(merged, "is_empty", True):
                    segs_out = []
                    for g in (getattr(merged, "geoms", None) or [merged]):
                        c = list(getattr(g, "coords", []))
                        for i in range(len(c) - 1):
                            segs_out.append(LineString([c[i], c[i + 1]]))
                    if segs_out:
                        noded_lines = segs_out
            except Exception:
                pass
        if noded_lines:
            lines = noded_lines
        polygons, _cuts, _dangles, _invalids = polygonize_full(lines)
        all_x = [p[0] for s in all_segs for p in s]
        all_y = [p[1] for s in all_segs for p in s]
        bbox_area = (max(all_x) - min(all_x) + 1) * (max(all_y) - min(all_y) + 1) if all_x and all_y else 1e9
        max_area = bbox_area * max_area_ratio
        result: List[List[Tuple[float, float]]] = []
        geoms = polygons.geoms if hasattr(polygons, "geoms") else [polygons]
        for g in geoms:
            if g is None or getattr(g, "is_empty", True):
                continue
            ext = getattr(g, "exterior", None)
            if ext is None:
                continue
            coords = list(getattr(ext, "coords", []))
            if len(coords) < 4:
                continue
            pts = [(float(c[0]), float(c[1])) for c in coords[:-1]]
            try:
                from shapely.geometry import Polygon as ShapelyPolygon
                p = ShapelyPolygon(pts)
                a = float(p.area)
                if a < min_area or a > max_area:
                    continue
                if exclude_interior_of:
                    try:
                        excl_lines = [
                            LineString([(float(s[0][0]), float(s[0][1])), (float(s[1][0]), float(s[1][1]))])
                            for s in exclude_interior_of if len(s) >= 2
                        ]
                        if excl_lines:
                            excl_polys, _, _, _ = polygonize_full(excl_lines)
                            within_excl = False
                            for ep in (getattr(excl_polys, "geoms", None) or [excl_polys]):
                                if ep is None or getattr(ep, "is_empty", True):
                                    continue
                                if hasattr(ep, "exterior") and ep.exterior and p.within(ep):
                                    within_excl = True
                                    break
                            if within_excl:
                                continue
                    except Exception:
                        pass
                result.append(pts)
            except Exception:
                continue
        return result
    except Exception:
        return []


def _numbered_faces_from_polygons(
    polygons_2d: List[List[Tuple[float, float]]],
    faces_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Returnează doar fețele care corespund poligoanelor numerotate (faces.png).
    Ordinea = ordinea din polygons_2d. Fără fețe suplimentare.
    """
    if not polygons_2d or not faces_data:
        return []
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
    except ImportError:
        return []
    result: List[Dict[str, Any]] = []
    used_face_idx: set = set()
    for poly_pts in polygons_2d:
        if len(poly_pts) < 3:
            continue
        try:
            p_poly = ShapelyPolygon([(float(p[0]), float(p[1])) for p in poly_pts])
            if not getattr(p_poly, "is_valid", True):
                try:
                    p_poly = p_poly.buffer(0)
                except Exception:
                    continue
            if p_poly.is_empty:
                continue
        except Exception:
            continue
        best_idx = -1
        best_area = -1.0
        for fi, f in enumerate(faces_data):
            if fi in used_face_idx:
                continue
            vs = f.get("vertices_3d") or []
            if len(vs) < 3:
                continue
            try:
                f_xy = [(float(v[0]), float(v[1])) for v in vs]
                f_poly = ShapelyPolygon(f_xy)
                if f_poly.is_empty:
                    continue
                inter = p_poly.intersection(f_poly)
                area = float(inter.area) if hasattr(inter, "area") else 0.0
                if area > best_area:
                    best_area = area
                    best_idx = fi
            except Exception:
                continue
        if best_idx >= 0 and best_area > 0.5:
            used_face_idx.add(best_idx)
            result.append(faces_data[best_idx])
    return result


def _draw_faces_png(
    wall_mask: np.ndarray,
    polygons: List[List[Tuple[float, float]]],
    output_path: Path,
    roof_type: str = "",
    seed: int = 42,
) -> bool:
    """Desenează poligoane cu culori random și transparență 50%."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    h, w = wall_mask.shape[:2]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f"Fețe acoperiș {roof_type}" if roof_type else "Fețe din segmente (50% transparență)")
    ax.set_aspect("equal")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.imshow(wall_mask, cmap="gray", vmin=0, vmax=255)
    rng = random.Random(seed)
    for idx, pts in enumerate(polygons):
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = (rng.random(), rng.random(), rng.random(), 0.5)
        ax.fill(xs, ys, facecolor=color, edgecolor="none")
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        ax.text(cx, cy, str(idx + 1), ha="center", va="center", fontsize=12, fontweight="bold", color="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        plt.close()
        return False


def _draw_lines_and_save(
    wall_mask: np.ndarray,
    segments_ridge: List[List[List[float]]],
    segments_contour: List[List[List[float]]],
    segments_magenta: List[List[List[float]]],
    segments_blue: List[List[List[float]]],
    segments_pyramid: List[List[List[float]]],
    segments_upper_rect: List[List[List[float]]],
    roof_type: str,
    output_path: Path,
    segments_orange: Optional[List[List[List[float]]]] = None,
    segments_wall_support: Optional[List[List[List[float]]]] = None,
    ridge_midpoints_from: Optional[List[List[List[float]]]] = None,
    ridge_pink_points: Optional[List[Tuple[float, float]]] = None,
    brown_endpoint_markers: Optional[List[Tuple[Tuple[float, float], int]]] = None,
    segments_brown: Optional[List[List[List[float]]]] = None,
    segments_overhang: Optional[List[List[List[float]]]] = None,
    segments_overhang_inner: Optional[List[List[List[float]]]] = None,
    segments_overhang_ridge: Optional[List[List[List[float]]]] = None,
    sections: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        return False

    h, w = wall_mask.shape[:2]
    # Figură mai mare cu spațiu pentru legendă alături (nu peste imagine)
    fig_w = 14.0
    fig_h = 10.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(1, 1, 1)
    # Axe pentru imagine: ~70% lățime, legenda va fi în dreapta
    ax.set_position([0.05, 0.05, 0.65, 0.9])
    ax.set_title(f"Linii acoperiș {roof_type}")
    ax.set_aspect("equal")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.imshow(wall_mask, cmap="gray", vmin=0, vmax=255)

    # Linii subțiri + offset perpendicular ca segmentele suprapuse să se vadă una lângă alta
    _lw = 0.8
    _off = 1.0  # px offset per „strat” când sunt suprapuse

    def _offset_seg(seg: List[List[float]], d_px: float):
        if len(seg) < 2 or abs(d_px) < 1e-6:
            return [[seg[0][0], seg[0][1]], [seg[1][0], seg[1][1]]]
        x0, y0 = float(seg[0][0]), float(seg[0][1])
        x1, y1 = float(seg[1][0]), float(seg[1][1])
        dx, dy = x1 - x0, y1 - y0
        L = (dx * dx + dy * dy) ** 0.5
        if L < 1e-12:
            return [[x0, y0], [x1, y1]]
        nx, ny = -dy / L, dx / L
        return [[x0 + d_px * nx, y0 + d_px * ny], [x1 + d_px * nx, y1 + d_px * ny]]

    _alpha = 0.5
    for seg in segments_contour:
        if len(seg) >= 2:
            s = _offset_seg(seg, 0)
            ax.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], color="green", linewidth=_lw, alpha=_alpha)

    # Poligon bază overhang (contur pe baza căruia se ia distanța de 1 m) – mov
    if segments_overhang_inner:
        for seg in segments_overhang_inner:
            if len(seg) >= 2:
                s = _offset_seg(seg, _off * 1)
                ax.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], color="magenta", linewidth=_lw, linestyle="-", alpha=0.9, zorder=3)

    if segments_overhang:
        for seg in segments_overhang:
            if len(seg) >= 2:
                s = _offset_seg(seg, _off * 2)
                # 2_w: segmente cu bulină albastră închis la mijloc → mov închis
                is_ridge_seg = False
                if segments_overhang_ridge:
                    sk = _seg_key(seg)
                    for rseg in segments_overhang_ridge:
                        if len(rseg) >= 2 and _seg_key(rseg) == sk:
                            is_ridge_seg = True
                            break
                color = "#4B0082" if is_ridge_seg else "cyan"
                ax.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], color=color, linewidth=_lw, linestyle="--" if not is_ridge_seg else "-", alpha=_alpha if not is_ridge_seg else 0.95, zorder=4)
    # Buline: colțuri overhang = albastru deschis; unde ridge-urile prelungite ating overhang = albastru închis
    overhang_corner_pts: List[Tuple[float, float]] = []
    overhang_ridge_pts: List[Tuple[float, float]] = []
    if segments_overhang:
        # Colțuri overhang = toate capetele segmentelor de overhang
        for seg in segments_overhang:
            if len(seg) >= 2:
                overhang_corner_pts.append((float(seg[0][0]), float(seg[0][1])))
                overhang_corner_pts.append((float(seg[1][0]), float(seg[1][1])))
        # La intersecția de ridge-uri nu generăm buline albastre; nici când capătul ridge-ului e la intersecție
        ridge_int_pts = _ridge_intersection_points(segments_ridge, tol=5.0)
        _bul_tol = 5.0
        def _near_ridge_intersection(p: Tuple[float, float]) -> bool:
            for r in ridge_int_pts:
                if (p[0] - r[0]) ** 2 + (p[1] - r[1]) ** 2 <= _bul_tol * _bul_tol:
                    return True
            return False
        # Intersecții: prelungim fiecare ridge la 2× lungimea lui și căutăm intersecția cu overhang (segment–segment)
        if segments_ridge:
            def _segment_segment_intersection(
                a0: Tuple[float, float], a1: Tuple[float, float],
                b0: Tuple[float, float], b1: Tuple[float, float],
            ) -> Optional[Tuple[float, float]]:
                """Intersecția segment [a0,a1] cu segment [b0,b1]. Returnează punctul dacă există și e pe ambele."""
                ax0, ay0 = a0[0], a0[1]
                ax1, ay1 = a1[0], a1[1]
                bx0, by0 = b0[0], b0[1]
                bx1, by1 = b1[0], b1[1]
                dx_a = ax1 - ax0
                dy_a = ay1 - ay0
                dx_b = bx1 - bx0
                dy_b = by1 - by0
                denom = dx_a * dy_b - dy_a * dx_b
                if abs(denom) < 1e-12:
                    return None
                t = ((bx0 - ax0) * dy_b - (by0 - ay0) * dx_b) / denom
                s = ((bx0 - ax0) * dy_a - (by0 - ay0) * dx_a) / denom
                if not (0 <= t <= 1 and 0 <= s <= 1):
                    return None
                px = ax0 + t * dx_a
                py = ay0 + t * dy_a
                return (px, py)
            for rseg in segments_ridge:
                if len(rseg) < 2:
                    continue
                r0 = (float(rseg[0][0]), float(rseg[0][1]))
                r1 = (float(rseg[1][0]), float(rseg[1][1]))
                dx = r1[0] - r0[0]
                dy = r1[1] - r0[1]
                L = (dx * dx + dy * dy) ** 0.5
                if L < 1e-12:
                    continue
                # Segment de căutare: prelungim ridge-ul suficient ca să atingă overhang-ul (1 m în afară)
                # Extensie mare (1×L pe fiecare cap) ca linia să traverseze și banda de overhang; 10% era prea puțin pentru 2_w
                half = max(0.1 * L, L)
                e0 = (r0[0] - half * (dx / L), r0[1] - half * (dy / L))
                e1 = (r1[0] + half * (dx / L), r1[1] + half * (dy / L))
                for oseg in segments_overhang:
                    if len(oseg) < 2:
                        continue
                    o0 = (float(oseg[0][0]), float(oseg[0][1]))
                    o1 = (float(oseg[1][0]), float(oseg[1][1]))
                    pt = _segment_segment_intersection(e0, e1, o0, o1)
                    if pt is not None:
                        # 4_w: nu punem buline albastre deloc
                        if roof_type == "4_w":
                            continue
                        # Bulina albastru închis pe laturi perpendiculare pe ridge (nu paralele)
                        dx_o = o1[0] - o0[0]
                        dy_o = o1[1] - o0[1]
                        len_o_sq = dx_o * dx_o + dy_o * dy_o
                        if len_o_sq >= 1e-20:
                            dot = dx * dx_o + dy * dy_o
                            len_r = (dx * dx + dy * dy) ** 0.5
                            len_o = len_o_sq ** 0.5
                            # 2_w: nu punem bulină dacă segmentul de overhang e paralel cu ridge-ul DIN CARE IESE (același ridge rseg)
                            # 4.5_w etc.: cos(80°) ≈ 0.17
                            thresh = 0.05 if roof_type == "2_w" else 0.17
                            if abs(dot) > thresh * len_r * len_o:
                                continue
                        # Excludem bulina dacă un cap al ridge-ului e pe alt ridge (T/L etc.)
                        if not _near_ridge_intersection(r0) and not _near_ridge_intersection(r1):
                            overhang_ridge_pts.append(pt)
        # Filtrare suplimentară: punctul pe overhang să nu fie la intersecție (2_w: păstrăm toate colțurile overhang)
        if roof_type != "2_w":
            overhang_corner_pts = [p for p in overhang_corner_pts if not _near_ridge_intersection(p)]
        overhang_ridge_pts = [p for p in overhang_ridge_pts if not _near_ridge_intersection(p)]
        if overhang_corner_pts:
            uniq = list(dict.fromkeys([(round(p[0], 2), round(p[1], 2)) for p in overhang_corner_pts]))
            ax.scatter([p[0] for p in uniq], [p[1] for p in uniq], c="lightblue", s=80, zorder=15, edgecolors="black", linewidths=1)
        if overhang_ridge_pts:
            # Buline albastre închis: ridge prelungit → overhang (2_w, 4_w, 4.5_w)
            uniq = list(dict.fromkeys([(round(p[0], 2), round(p[1], 2)) for p in overhang_ridge_pts]))
            ax.scatter([p[0] for p in uniq], [p[1] for p in uniq], c="darkblue", s=80, zorder=15, edgecolors="black", linewidths=1)
        # Anchor + segment bulină → anchor (ca în 3D)
        if segments_overhang and (overhang_corner_pts or overhang_ridge_pts):
            contour_corners_2d: List[Tuple[float, float]] = []
            seen_cc2 = set()
            for seg in segments_contour:
                if len(seg) >= 2:
                    for p in (seg[0], seg[1]):
                        q = (round(float(p[0]), 2), round(float(p[1]), 2))
                        if q not in seen_cc2:
                            seen_cc2.add(q)
                            contour_corners_2d.append((float(p[0]), float(p[1])))
            # Pentru ancore pe același dreptunghi: colțuri per secțiune și mapare contur -> secțiune
            section_corners_list: List[List[Tuple[float, float]]] = []
            contour_corner_to_section: Dict[Tuple[float, float], List[int]] = {}
            if sections:
                for sec in sections:
                    br = sec.get("bounding_rect") or []
                    if len(br) < 3:
                        continue
                    xs = [float(p[0]) for p in br]
                    ys = [float(p[1]) for p in br]
                    mnx, mxx = min(xs), max(xs)
                    mny, mxy = min(ys), max(ys)
                    corners = [(mnx, mny), (mxx, mny), (mxx, mxy), (mnx, mxy)]
                    section_corners_list.append(corners)
                _tol_rect = 2.0
                for c in contour_corners_2d:
                    rc = (round(c[0], 2), round(c[1], 2))
                    contour_corner_to_section[rc] = []
                    for i, sc_list in enumerate(section_corners_list):
                        for sc in sc_list:
                            if (c[0] - sc[0]) ** 2 + (c[1] - sc[1]) ** 2 <= _tol_rect * _tol_rect:
                                contour_corner_to_section[rc].append(i)
                                break
            contour_interior_corners = _contour_interior_corners(segments_contour)
            pyramid_seg_tuples: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
            for seg in segments_pyramid:
                if len(seg) >= 2:
                    a = (float(seg[0][0]), float(seg[0][1]))
                    b = (float(seg[-1][0]), float(seg[-1][1]))
                    pyramid_seg_tuples.append((a, b))
            # Poligon overhang (cyan): pentru TOATE tipurile – anchor doar colțuri STRICT ÎN INTERIOR
            overhang_poly_for_anchor = None
            if segments_overhang and len(segments_overhang) >= 3 and roof_type in ("0_w", "1_w", "2_w", "4_w", "4.5_w"):
                try:
                    from shapely.geometry import LineString, Point
                    from shapely.ops import polygonize_full, unary_union
                    oh_lines = [
                        LineString([(float(s[0][0]), float(s[0][1])), (float(s[1][0]), float(s[1][1]))])
                        for s in segments_overhang if len(s) >= 2
                        and (float(s[0][0]) - float(s[1][0])) ** 2 + (float(s[0][1]) - float(s[1][1])) ** 2 > 1e-6
                    ]
                    if len(oh_lines) >= 3:
                        polys, _, _, _ = polygonize_full(oh_lines)
                        geoms = list(getattr(polys, "geoms", None) or [polys])
                        valid = [g for g in geoms if g is not None and not getattr(g, "is_empty", True)]
                        if valid:
                            overhang_poly_for_anchor = unary_union(valid) if len(valid) > 1 else valid[0]
                        else:
                            overhang_poly_for_anchor = valid[0] if valid else None
                except Exception:
                    overhang_poly_for_anchor = None
            ridge_seg_tuples_2d: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
            for seg in segments_ridge:
                if len(seg) >= 2:
                    a = (float(seg[0][0]), float(seg[0][1]))
                    b = (float(seg[1][0]), float(seg[1][1]))
                    ridge_seg_tuples_2d.append((a, b))
            uniq_ridge = list(dict.fromkeys([(round(p[0], 2), round(p[1], 2)) for p in overhang_ridge_pts]))
            uniq_corners = list(dict.fromkeys([(round(p[0], 2), round(p[1], 2)) for p in overhang_corner_pts]))
            overhang_ridge_set_2d = set(uniq_ridge)
            drawn_anchors: set = set()
            drawn_anchor_positions: List[Tuple[float, float]] = []
            _anchor_near_tol = 5.0
            def _anchor_near(ax: float, ay: float) -> bool:
                for (ox, oy) in drawn_anchor_positions:
                    if (ax - ox) ** 2 + (ay - oy) ** 2 <= _anchor_near_tol * _anchor_near_tol:
                        return True
                return False
            for pt in uniq_ridge:
                p = (pt[0], pt[1])
                if ridge_seg_tuples_2d:
                    best_ax, best_ay, best_d2 = 0.0, 0.0, 1e30
                    for (a, b) in ridge_seg_tuples_2d:
                        qx, qy, d2 = _closest_point_on_segment(p, a, b)
                        if d2 < best_d2:
                            best_d2 = d2
                            best_ax, best_ay = qx, qy
                    ka = (round(best_ax, 2), round(best_ay, 2))
                    if not _anchor_near(best_ax, best_ay):
                        drawn_anchors.add(ka)
                        drawn_anchor_positions.append((best_ax, best_ay))
                        ax.scatter([best_ax], [best_ay], c="gray", s=50, zorder=14, edgecolors="black", linewidths=1)
                    ax.plot([p[0], best_ax], [p[1], best_ay], color="gray", linewidth=_lw, linestyle=":", alpha=0.8, zorder=12)
            for pt in uniq_corners:
                p = (pt[0], pt[1])
                # FIX v4: Logică unificată pentru anchor lightblue (toate tipurile de acoperiș)
                # Prioritate 1: dacă există colț interior al conturului verde aproape de bulină → anchor acolo
                # (colțul interior verde = concavitatea formei L; bulina externă din overhang îi corespunde)
                use_interior_only = False
                best_ax, best_ay = p[0], p[1]
                if contour_interior_corners:
                    best_int_d2 = 1e30
                    best_int_ax, best_int_ay = p[0], p[1]
                    # Doar colțuri interioare care sunt STRICT ÎN INTERIORUL overhang-ului
                    interior_candidates = list(contour_interior_corners)
                    if overhang_poly_for_anchor is not None:
                        try:
                            from shapely.geometry import Point as ShapelyPoint
                            _tol = 1e-6
                            def _strictly_inside_overhang(c):
                                pt = ShapelyPoint(c[0], c[1])
                                return (
                                    overhang_poly_for_anchor.contains(pt)
                                    and overhang_poly_for_anchor.boundary.distance(pt) > _tol
                                )
                            interior_candidates = [q for q in interior_candidates if _strictly_inside_overhang(q)]
                        except Exception:
                            pass
                    for q in interior_candidates:
                        d2 = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
                        if d2 < best_int_d2:
                            best_int_d2 = d2
                            best_int_ax, best_int_ay = q[0], q[1]
                    # Folosim colțul interior dacă e suficient de aproape de bulină
                    # (mai aproape decât dublul distanței la cel mai apropiat colț exterior)
                    best_ext_d2 = min(
                        ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
                        for q in contour_corners_2d
                    ) if contour_corners_2d else 1e30
                    if best_int_d2 <= best_ext_d2 * 2.5:
                        use_interior_only = True
                        best_ax, best_ay = best_int_ax, best_int_ay
                if not use_interior_only:
                    if roof_type in ("4_w", "4.5_w") and (pyramid_seg_tuples or contour_corners_2d):
                        # 4_w, 4.5_w: anchor = colț (contur sau piramidă) STRICT ÎN INTERIOR overhang
                        anchor_candidates_45 = list(contour_corners_2d) if contour_corners_2d else []
                        for (a, b) in (pyramid_seg_tuples or []):
                            anchor_candidates_45.append(a)
                            anchor_candidates_45.append(b)
                        anchor_candidates_45 = list(dict.fromkeys([(round(c[0], 4), round(c[1], 4)) for c in anchor_candidates_45]))
                        anchor_candidates_45 = [(c[0], c[1]) for c in anchor_candidates_45]
                        if overhang_poly_for_anchor is not None:
                            try:
                                from shapely.geometry import Point as ShapelyPoint
                                _tol = 1e-6
                                def _strictly_inside_45(c):
                                    pt = ShapelyPoint(c[0], c[1])
                                    return (
                                        overhang_poly_for_anchor.contains(pt)
                                        and overhang_poly_for_anchor.boundary.distance(pt) > _tol
                                    )
                                anchor_candidates_45 = [c for c in anchor_candidates_45 if _strictly_inside_45(c)]
                            except Exception:
                                pass
                        if anchor_candidates_45:
                            best_d2 = 1e30
                            for q in anchor_candidates_45:
                                d2 = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
                                if d2 < best_d2:
                                    best_d2 = d2
                                    best_ax, best_ay = q[0], q[1]
                        else:
                            continue  # niciun colț în interiorul overhang → nu desenăm linie
                    elif roof_type in ("4_w", "4.5_w") and pyramid_seg_tuples:
                        # 4_w/4.5_w fără overhang poly: fallback proiecție pe piramidă
                        best_d2 = 1e30
                        for (a, b) in pyramid_seg_tuples:
                            qx, qy, d2 = _closest_point_on_segment(p, a, b)
                            if d2 < best_d2:
                                best_d2 = d2
                                best_ax, best_ay = qx, qy
                    else:
                        # Prioritate 3 (2_w, 1_w, 0_w): la fel ca 4_w — TOATE colțurile conturului, apoi doar cele STRICT ÎN INTERIOR overhang, cel mai apropiat
                        anchor_candidates = list(contour_corners_2d) if contour_corners_2d else []
                        if not anchor_candidates:
                            continue
                        # Doar colțuri STRICT ÎN INTERIORUL overhang-ului (cyan)
                        if overhang_poly_for_anchor is not None:
                            try:
                                from shapely.geometry import Point as ShapelyPoint
                                _tol = 1e-6
                                def _strictly_inside(c):
                                    pt = ShapelyPoint(c[0], c[1])
                                    return (
                                        overhang_poly_for_anchor.contains(pt)
                                        and overhang_poly_for_anchor.boundary.distance(pt) > _tol
                                    )
                                anchor_candidates = [c for c in anchor_candidates if _strictly_inside(c)]
                            except Exception:
                                pass
                        if not anchor_candidates:
                            continue
                        best_d2 = 1e30
                        for q in anchor_candidates:
                            d2 = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
                            if d2 < best_d2:
                                best_d2 = d2
                                best_ax, best_ay = q[0], q[1]
                ka = (round(best_ax, 2), round(best_ay, 2))
                if not _anchor_near(best_ax, best_ay):
                    drawn_anchors.add(ka)
                    drawn_anchor_positions.append((best_ax, best_ay))
                    ax.scatter([best_ax], [best_ay], c="gray", s=50, zorder=14, edgecolors="black", linewidths=1)
                ax.plot([p[0], best_ax], [p[1], best_ay], color="gray", linewidth=_lw, linestyle=":", alpha=0.8, zorder=12)

    if segments_brown:
        for seg in segments_brown:
            if len(seg) >= 2:
                s = _offset_seg(seg, _off * 3)
                ax.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], color="saddlebrown", linewidth=_lw, alpha=_alpha, zorder=8)

    for seg in segments_ridge:
        if len(seg) >= 2:
            s = _offset_seg(seg, _off * 4)
            ax.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], color="darkred", linewidth=_lw, alpha=_alpha)

    for seg in segments_magenta:
        if len(seg) >= 2:
            s = _offset_seg(seg, _off * 5)
            ax.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], color="#CC00FF", linewidth=_lw, alpha=_alpha, zorder=10)

    for seg in segments_blue:
        if len(seg) >= 2:
            s = _offset_seg(seg, _off * 6)
            ax.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], color="blue", linewidth=_lw, alpha=_alpha)

    for seg in segments_pyramid:
        if len(seg) >= 2:
            s = _offset_seg(seg, _off * 7)
            ax.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], color="blue", linewidth=_lw, linestyle="--", alpha=_alpha)

    for seg in segments_upper_rect:
        if len(seg) >= 2:
            s = _offset_seg(seg, _off * 8)
            ax.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], color="yellow", linewidth=_lw, alpha=_alpha, zorder=5)

    if segments_orange:
        for seg in segments_orange:
            if len(seg) >= 2:
                s = _offset_seg(seg, _off * 9)
                ax.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], color="orange", linewidth=_lw, linestyle="-", alpha=_alpha, zorder=6)

    if segments_wall_support:
        for seg in segments_wall_support:
            if len(seg) >= 2:
                s = _offset_seg(seg, _off * 10)
                ax.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], color="deepskyblue", linewidth=_lw, alpha=_alpha, zorder=6)

    # Buline galbene la mijlocul fiecărui ridge (original, netrimat – pentru 4.5_w)
    src = ridge_midpoints_from if ridge_midpoints_from is not None else segments_ridge
    ridge_midpoints: List[Tuple[float, float]] = []
    for seg in src:
        if len(seg) >= 2:
            mx = (seg[0][0] + seg[1][0]) / 2.0
            my = (seg[0][1] + seg[1][1]) / 2.0
            ridge_midpoints.append((mx, my))
    if ridge_midpoints:
        xs = [p[0] for p in ridge_midpoints]
        ys = [p[1] for p in ridge_midpoints]
        ax.scatter(xs, ys, c="yellow", s=80, zorder=15, edgecolors="black", linewidths=1)

    if ridge_pink_points:
        px = [p[0] for p in ridge_pink_points]
        py = [p[1] for p in ridge_pink_points]
        ax.scatter(px, py, c="hotpink", s=80, zorder=15, edgecolors="black", linewidths=1)

    if brown_endpoint_markers:
        pt_to_ids: Dict[Tuple[float, float], List[int]] = {}
        tol = 3.0
        for (pt, pair_id) in brown_endpoint_markers:
            px, py = round(pt[0] / tol) * tol, round(pt[1] / tol) * tol
            key = (px, py)
            if key not in pt_to_ids:
                pt_to_ids[key] = []
            if pair_id not in pt_to_ids[key]:
                pt_to_ids[key].append(pair_id)
        unique_pts: List[Tuple[Tuple[float, float], str]] = []
        for key, ids in pt_to_ids.items():
            ids_sorted = sorted(set(ids))
            label = ",".join(str(i) for i in ids_sorted)
            pt = (key[0], key[1])
            unique_pts.append((pt, label))
        bx = [p[0][0] for p in unique_pts]
        by = [p[0][1] for p in unique_pts]
        ax.scatter(bx, by, c="saddlebrown", s=80, zorder=15, edgecolors="black", linewidths=1)
        for (pt, label) in unique_pts:
            ax.text(pt[0], pt[1], label, color="white", fontsize=8, ha="center", va="center", zorder=16)

    handles = []
    if ridge_midpoints:
        handles.append(Line2D([0], [0], color="yellow", marker="o", markersize=6, linestyle="", label="Mijloc ridge"))
    if ridge_pink_points:
        handles.append(Line2D([0], [0], color="hotpink", marker="o", markersize=6, linestyle="", label="Atins de diagonale / capăt opus"))
    if brown_endpoint_markers:
        handles.append(Line2D([0], [0], color="saddlebrown", marker="o", markersize=6, linestyle="", label="Capete segment maro"))
    handles.append(Line2D([0], [0], color="green", lw=_lw, alpha=_alpha, label="Contur exterior"))
    if segments_overhang_inner:
        handles.append(Line2D([0], [0], color="magenta", lw=_lw, alpha=0.9, label="Bază overhang (1 m)"))
    if segments_overhang:
        handles.append(Line2D([0], [0], color="cyan", lw=_lw, ls="--", alpha=_alpha, label="Overhang 1 m"))
    if segments_overhang_ridge:
        handles.append(Line2D([0], [0], color="#4B0082", lw=_lw, ls="-", alpha=0.95, label="Segment ridge–overhang"))
    if overhang_corner_pts:
        handles.append(Line2D([0], [0], color="lightblue", marker="o", markersize=6, linestyle="", label="Colțuri overhang"))
    if overhang_ridge_pts:
        handles.append(Line2D([0], [0], color="darkblue", marker="o", markersize=6, linestyle="", label="Ridge prelungit → overhang"))
    if segments_ridge:
        handles.append(Line2D([0], [0], color="darkred", lw=_lw, alpha=_alpha, label="Ridge"))
    if segments_magenta:
        handles.append(Line2D([0], [0], color="#CC00FF", lw=_lw, alpha=_alpha, label="Linii intersecție → colțuri"))
    if segments_blue and not segments_pyramid:
        handles.append(Line2D([0], [0], color="blue", lw=_lw, alpha=_alpha, label="Segment albastru (shed)"))
    elif segments_pyramid:
        handles.append(Line2D([0], [0], color="blue", lw=_lw, ls="--", alpha=_alpha, label="Diagonale 45° (piramidă)"))
    if segments_upper_rect:
        handles.append(Line2D([0], [0], color="yellow", lw=_lw, alpha=_alpha, label="Etaj superior"))
    if segments_orange:
        handles.append(Line2D([0], [0], color="orange", lw=_lw, ls="-", alpha=_alpha, label="Paralelă între diagonale (4.5_w)"))
    if segments_brown:
        handles.append(Line2D([0], [0], color="saddlebrown", lw=_lw, alpha=_alpha, label="Laturi opuse teșirii"))
    if segments_wall_support:
        handles.append(Line2D([0], [0], color="deepskyblue", lw=_lw, alpha=_alpha, label="Suport perete"))
    if handles:
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
            fontsize=9,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        plt.close()
        return False


def _sections_for_1w_shed(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Secțiuni sintetice pentru 1_w: ridge_line = segment albastru (centru)."""
    out = []
    for sec in sections:
        br = sec.get("bounding_rect") or []
        if len(br) < 3:
            continue
        xs = [float(p[0]) for p in br]
        ys = [float(p[1]) for p in br]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        w, h = maxx - minx, maxy - miny
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        if w >= h:
            ridge = [[minx, cy], [maxx, cy]]
            orient = "horizontal"
        else:
            ridge = [[cx, miny], [cx, maxy]]
            orient = "vertical"
        out.append({
            **sec,
            "ridge_line": ridge,
            "ridge_orientation": orient,
        })
    return out


def _faces_from_segments(
    roof_type: str,
    sections: List[Dict[str, Any]],
    wall_mask: np.ndarray,
    wall_height: float = 300.0,
    roof_angle_deg: float = 30.0,
) -> List[Dict[str, Any]]:
    """Generează fețe 3D. Pentru 1_w folosește secțiuni shed (ridge=albastru)."""
    from roof_calc.roof_segments_3d import get_faces_3d_from_segments
    from roof_calc.flood_fill import flood_fill_interior, get_house_shape_mask
    from roof_calc.geometry import extract_polygon

    filled = flood_fill_interior(wall_mask)
    house_mask = get_house_shape_mask(filled)
    floor_poly = extract_polygon(house_mask)
    if floor_poly is None or floor_poly.is_empty:
        return []

    secs = _sections_for_1w_shed(sections) if roof_type in ("0_w", "1_w") else sections
    corner_lines = None
    if roof_type not in ("0_w", "1_w") and _has_ridge_intersection(sections):
        from roof_calc.overhang import ridge_intersection_corner_lines
        corner_lines = ridge_intersection_corner_lines(sections, per_section=False)

    try:
        return get_faces_3d_from_segments(
            secs,
            floor_poly,
            wall_height=wall_height,
            roof_angle_deg=roof_angle_deg,
            corner_lines=corner_lines,
            use_section_rect_eaves=False,
        )
    except Exception:
        return []


def _z_roof_at(roof_faces: List[Dict[str, Any]], x: float, y: float, default_z: float, tol: float = 15.0) -> float:
    """Înălțimea acoperișului la (x,y). Interpolează din fețele acoperișului."""
    best: Optional[float] = None
    best_dist: Optional[float] = None
    try:
        from shapely.geometry import Point
        from shapely.geometry import Polygon as ShapelyPolygon
    except Exception:
        return default_z
    pt = Point(x, y)
    for f in roof_faces:
        vs = f.get("vertices_3d") or []
        if len(vs) < 3:
            continue
        xy = [(float(v[0]), float(v[1])) for v in vs]
        xs, ys = [p[0] for p in xy], [p[1] for p in xy]
        if x < min(xs) - tol or x > max(xs) + tol or y < min(ys) - tol or y > max(ys) + tol:
            continue
        try:
            poly = ShapelyPolygon(xy)
            if poly.is_empty:
                continue
            inside = poly.contains(pt) or poly.buffer(tol).contains(pt)
            dist = float(poly.distance(pt)) if not inside else 0.0
            if not inside and dist > tol:
                continue
        except Exception:
            continue
        v0, v1, v2 = vs[0], vs[1], vs[2]
        x0, y0, z0 = float(v0[0]), float(v0[1]), float(v0[2])
        x1, y1, z1 = float(v1[0]), float(v1[1]), float(v1[2])
        x2, y2, z2 = float(v2[0]), float(v2[1]), float(v2[2])
        denom = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(denom) < 1e-12:
            z = max(z0, z1, z2)
        else:
            w1 = ((x - x0) * (y2 - y0) - (x2 - x0) * (y - y0)) / denom
            w2 = ((x1 - x0) * (y - y0) - (x - x0) * (y1 - y0)) / denom
            z = (1.0 - w1 - w2) * z0 + w1 * z1 + w2 * z2
        if best is None or (dist <= (best_dist or 1e9) and z > (best or -1e9)):
            best, best_dist = z, dist
    return float(best) if best is not None else default_z


def _generate_frame_html(subdir: Path, wall_height: float = 300.0) -> None:
    """
    Generează frame.html cu randare 3D: exact segmentele din lines.png (culori, legende)
    + liniile de contur ale pereților.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    ff_path = subdir / "faces_faces.json"
    if not ff_path.exists():
        return
    payload = json.loads(ff_path.read_text(encoding="utf-8"))
    floor_path = payload.get("floor_path")
    faces = payload.get("faces") or []
    wh = float(payload.get("wall_height", wall_height))
    segs_data = payload.get("segments") or {}
    markers_data = payload.get("markers") or {}
    has_segments = bool(segs_data)
    roof_type_frame = subdir.name

    def _seg_to_plotly(segs_2d: List[Any], z_fn, endpoints_only: bool = False) -> Tuple[List[float], List[float], List[float]]:
        lx, ly, lz = [], [], []
        for seg in segs_2d:
            if not seg or len(seg) < 2:
                continue
            pts = seg if isinstance(seg[0], (list, tuple)) else []
            if not pts or len(pts) < 2:
                continue
            if endpoints_only:
                a, b = pts[0], pts[-1]
                x1, y1 = float(a[0]), float(a[1])
                x2, y2 = float(b[0]), float(b[1])
                z1, z2 = z_fn(x1, y1), z_fn(x2, y2)
                lx.extend([x1, x2, None])
                ly.extend([y1, y2, None])
                lz.extend([z1, z2, None])
            else:
                for i in range(len(pts) - 1):
                    a, b = pts[i], pts[i + 1]
                    x1, y1 = float(a[0]), float(a[1])
                    x2, y2 = float(b[0]), float(b[1])
                    z1, z2 = z_fn(x1, y1), z_fn(x2, y2)
                    lx.extend([x1, x2, None])
                    ly.extend([y1, y2, None])
                    lz.extend([z1, z2, None])
        return lx, ly, lz

    def _add_trace(lx: List[float], ly: List[float], lz: List[float], color: str, name: str, width: int = 2, dash: Optional[str] = None) -> None:
        if not lx:
            return
        ld = dict(color=color, width=width)
        if dash:
            ld["dash"] = dash
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", line=ld, name=name, legendgroup=name))

    z_fn_roof = lambda x, y: _z_roof_at(faces, x, y, wh, tol=40.0)
    # Ridge: sus (max z din ridge)
    ridge_segs = segs_data.get("ridge") or []
    ridge_z = wh
    ridge_pts: List[Tuple[float, float]] = []
    for seg in ridge_segs:
        pts = seg if isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)) else []
        for p in pts:
            if len(p) >= 2:
                xp, yp = float(p[0]), float(p[1])
                ridge_pts.append((xp, yp))
                z = z_fn_roof(xp, yp)
                ridge_z = max(ridge_z, z)
    z_fn_ridge = lambda x, y: ridge_z
    has_upper_floor = bool(payload.get("has_upper_floor", False))
    upper_rect_pts: List[Tuple[float, float]] = []
    contour_segs_list = segs_data.get("contour") or []
    for seg in (segs_data.get("upper_rect") or []):
        pts = seg if isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)) else []
        for p in pts:
            if len(p) >= 2:
                upper_rect_pts.append((float(p[0]), float(p[1])))
    best_contour_d2: Optional[float] = None
    if roof_type_frame in ("0_w", "1_w") and has_upper_floor and upper_rect_pts and contour_segs_list:
        best_contour_d2 = 1e30
        for cseg in contour_segs_list:
            pts = cseg if isinstance(cseg, (list, tuple)) and cseg and isinstance(cseg[0], (list, tuple)) else []
            if len(pts) >= 2:
                a, b = (float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1]))
                mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
                d2 = min((mid[0] - q[0]) ** 2 + (mid[1] - q[1]) ** 2 for q in upper_rect_pts)
                best_contour_d2 = min(best_contour_d2, d2)
        best_contour_d2 = (best_contour_d2 or 0) + 100

    def _point_on_contour_near_yellow(px: float, py: float, tol: float = 5.0) -> bool:
        if not upper_rect_pts or roof_type_frame not in ("0_w", "1_w") or not has_upper_floor or best_contour_d2 is None:
            return False
        pt = (px, py)
        for cseg in contour_segs_list:
            pts = cseg if isinstance(cseg, (list, tuple)) and cseg and isinstance(cseg[0], (list, tuple)) else []
            if len(pts) >= 2:
                a, b = (float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1]))
                if not _point_on_segment(pt, a, b, tol):
                    continue
                mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
                d2 = min((mid[0] - q[0]) ** 2 + (mid[1] - q[1]) ** 2 for q in upper_rect_pts)
                if d2 <= best_contour_d2:
                    return True
        return False

    def z_fn_contour(x: float, y: float) -> float:
        # 0_w: contur la nivelul pereților (wh). 1_w: latura lipită de etajul superior la ridge_z, restul la wh.
        if roof_type_frame == "1_w" and has_upper_floor and _point_on_contour_near_yellow(x, y):
            return ridge_z
        return wh
    # Portocalii: la mijlocul diagonalelor (paralele între diagonale)
    z_fn_orange = lambda x, y: (ridge_z + wh) / 2.0
    contour_pts_for_z: List[Tuple[float, float]] = []
    for seg in (segs_data.get("contour") or []):
        pts = seg if isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)) else []
        for p in pts:
            if len(p) >= 2:
                contour_pts_for_z.append((float(p[0]), float(p[1])))
    ridge_segs_list = segs_data.get("ridge") or []
    def _point_on_ridge(px: float, py: float, tol: float = 5.0) -> bool:
        pt = (px, py)
        for rseg in ridge_segs_list:
            pts = rseg if isinstance(rseg, (list, tuple)) and rseg and isinstance(rseg[0], (list, tuple)) else []
            if len(pts) >= 2:
                a = (float(pts[0][0]), float(pts[0][1]))
                b = (float(pts[1][0]), float(pts[1][1]))
                if _point_on_segment(pt, a, b, tol):
                    return True
        return False
    for key in ("magenta", "pyramid"):
        for seg in (segs_data.get(key) or []):
            pts = seg if isinstance(seg, (list, tuple)) and seg and isinstance(seg[0], (list, tuple)) else []
            for p in pts:
                if len(p) >= 2:
                    px, py = float(p[0]), float(p[1])
                    if not _point_on_ridge(px, py):
                        contour_pts_for_z.append((px, py))
    if not contour_pts_for_z and floor_path:
        mask = cv2.imread(floor_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            for seg in _get_contour_segments(mask):
                if len(seg) >= 2:
                    for p in seg:
                        if len(p) >= 2:
                            contour_pts_for_z.append((float(p[0]), float(p[1])))
    # Restul segmentelor: trasate între capete (ridge sus, contur jos)
    def _min_dist_sq(pt: Tuple[float, float], lst: List[Tuple[float, float]]) -> float:
        if not lst:
            return 1e30
        return min((pt[0] - q[0]) ** 2 + (pt[1] - q[1]) ** 2 for q in lst)

    def z_fn_between(x: float, y: float) -> float:
        d_ridge = _min_dist_sq((x, y), ridge_pts)
        d_contour = _min_dist_sq((x, y), contour_pts_for_z)
        return ridge_z if d_ridge <= d_contour else wh

    def z_fn_magenta(x: float, y: float) -> float:
        """Capete pe ridge: ridge_z; capete pe contur: același z ca conturul."""
        return z_fn_contour(x, y) if not _point_on_ridge(x, y) else z_fn_between(x, y)

    def z_fn_pyramid(x: float, y: float) -> float:
        """Capete pe ridge: ridge_z; capete pe contur: același z ca conturul."""
        return z_fn_contour(x, y) if not _point_on_ridge(x, y) else z_fn_between(x, y)

    z_fn_pyramid_use = z_fn_pyramid if roof_type_frame in ("4_w", "2_w") else z_fn_between

    # Overhang 3D: înălțimi conform prelungirii (ridge = ridge_z; colțuri = pantă acoperiș din anchor)
    # Pentru acoperiș plat (0_w) nu aplicăm pantă – bulinele rămân la wh
    roof_angle_deg = float(payload.get("roof_angle_deg") or 30.0)
    roof_angle_rad = math.radians(roof_angle_deg)
    tan_roof = 0.0 if roof_type_frame == "0_w" else math.tan(roof_angle_rad)

    overhang_ridge_set: set = set()
    for pt in (markers_data.get("overhang_ridge_pts") or []):
        if pt and len(pt) >= 2:
            overhang_ridge_set.add((round(float(pt[0]), 2), round(float(pt[1]), 2)))

    contour_corners: List[Tuple[float, float]] = []
    seen_cc = set()
    for seg in (segs_data.get("contour") or []):
        pts = seg if isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)) else []
        for p in pts:
            if len(p) >= 2:
                q = (round(float(p[0]), 2), round(float(p[1]), 2))
                if q not in seen_cc:
                    seen_cc.add(q)
                    contour_corners.append((float(p[0]), float(p[1])))

    pyramid_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for seg in (segs_data.get("pyramid") or []):
        pts = seg if isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)) else []
        if len(pts) >= 2:
            a = (float(pts[0][0]), float(pts[0][1]))
            b = (float(pts[-1][0]), float(pts[-1][1]))
            pyramid_segments.append((a, b))

    overhang_boundary_pts: List[Tuple[float, float]] = []
    for pt in (markers_data.get("overhang_corners") or []):
        if pt and len(pt) >= 2:
            overhang_boundary_pts.append((float(pt[0]), float(pt[1])))
    for pt in (markers_data.get("overhang_ridge_pts") or []):
        if pt and len(pt) >= 2:
            overhang_boundary_pts.append((float(pt[0]), float(pt[1])))
    for seg in (segs_data.get("overhang") or []):
        if not seg or len(seg) < 2:
            continue
        for p in (seg[0], seg[-1]):
            if len(p) >= 2:
                overhang_boundary_pts.append((float(p[0]), float(p[1])))

    overhang_z_map: Dict[Tuple[float, float], float] = {}
    overhang_anchor_map: Dict[Tuple[float, float], Tuple[float, float]] = {}
    ridge_seg_tuples: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for seg in (segs_data.get("ridge") or []):
        pts = seg if isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)) else []
        if len(pts) >= 2:
            a = (float(pts[0][0]), float(pts[0][1]))
            b = (float(pts[1][0]), float(pts[1][1]))
            ridge_seg_tuples.append((a, b))

    for (x, y) in overhang_boundary_pts:
        key = (round(x, 2), round(y, 2))
        if key in overhang_z_map:
            continue
        if key in overhang_ridge_set:
            overhang_z_map[key] = ridge_z
            if ridge_seg_tuples:
                best_ax, best_ay, best_d2 = 0.0, 0.0, 1e30
                for (a, b) in ridge_seg_tuples:
                    qx, qy, d2 = _closest_point_on_segment((x, y), a, b)
                    if d2 < best_d2:
                        best_d2 = d2
                        best_ax, best_ay = qx, qy
                overhang_anchor_map[key] = (best_ax, best_ay)
            continue
        if roof_type_frame in ("4_w", "4.5_w") and pyramid_segments:
            best_d2 = 1e30
            best_z_anchor = wh
            best_ax, best_ay = x, y
            for (a, b) in pyramid_segments:
                qx, qy, d2 = _closest_point_on_segment((x, y), a, b)
                if d2 < best_d2:
                    best_d2 = d2
                    best_z_anchor = z_fn_pyramid_use(qx, qy)
                    best_ax, best_ay = qx, qy
            d = math.sqrt(best_d2)
            z_val = best_z_anchor - d * tan_roof
            overhang_z_map[key] = max(0.0, z_val)
            overhang_anchor_map[key] = (best_ax, best_ay)
        elif contour_corners:
            best_d2 = 1e30
            best_q = (x, y)
            for q in contour_corners:
                d2 = (x - q[0]) ** 2 + (y - q[1]) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_q = q
            d = math.sqrt(best_d2)
            z_val = wh - d * tan_roof
            overhang_z_map[key] = max(0.0, z_val)
            overhang_anchor_map[key] = best_q
        else:
            overhang_z_map[key] = z_fn_contour(x, y) if roof_type_frame == "0_w" else z_fn_between(x, y)

    def z_fn_overhang(x: float, y: float) -> float:
        k = (round(x, 2), round(y, 2))
        if k in overhang_z_map:
            return overhang_z_map[k]
        return z_fn_contour(x, y) if roof_type_frame == "0_w" else z_fn_between(x, y)


    fig = go.Figure()

    if not has_segments:
        # Fallback: muchii din fețe (pentru faces_faces.json vechi)
        seen_edges: set = set()
        for f in faces:
            vs = f.get("vertices_3d") or []
            n = len(vs)
            for i in range(n):
                v1 = [float(vs[i][0]), float(vs[i][1]), float(vs[i][2])]
                v2 = [float(vs[(i + 1) % n][0]), float(vs[(i + 1) % n][1]), float(vs[(i + 1) % n][2])]
                k1 = (round(v1[0], 2), round(v1[1], 2), round(v1[2], 2))
                k2 = (round(v2[0], 2), round(v2[1], 2), round(v2[2], 2))
                key = (k1, k2) if k1 <= k2 else (k2, k1)
                if key not in seen_edges:
                    seen_edges.add(key)
                    lx, ly, lz = [v1[0], v2[0], None], [v1[1], v2[1], None], [v1[2], v2[2], None]
                    _add_trace(lx, ly, lz, "#E74C3C", "Acoperiș (muchii fețe)", width=3)

    # Segmente din lines.png – culori exacte ca în legendă
    # Ridge sus, contur la pereți, restul între capete (ridge ↔ contur)
    segment_config = [
        ("contour", "green", "Contur exterior", 2, None, z_fn_contour, False),
        ("ridge", "darkred", "Ridge", 2.5, None, z_fn_ridge, False),
        ("magenta", "#CC00FF", "Linii intersecție → colțuri", 2, None, z_fn_magenta, False),
        ("blue", "blue", "Segment albastru (shed)", 2.5, None, z_fn_between, False),
        ("pyramid", "blue", "Diagonale 45° (piramidă)", 1.5, "dash", z_fn_pyramid_use, True),
        ("orange", "orange", "Paralelă între diagonale (4.5_w)", 2, None, z_fn_orange, False),
        ("overhang", "cyan", "Overhang 1 m", 2, "dash", z_fn_overhang, False),
        ("upper_rect", "yellow", "Etaj superior", 2, None, z_fn_orange, False),
        ("wall_support", "deepskyblue", "Suport perete", 2, None, z_fn_between, False),
        ("brown", "saddlebrown", "Laturi opuse teșirii", 2.5, None, z_fn_between, False),
    ]
    for key, color, name, width, dash, z_fn, endpoints_only in segment_config:
        if key == "overhang" and not faces:
            continue
        if key == "overhang" and faces:
            # Segment overhang în 3D trece prin toate bulinele (colțuri + ridge); inel închis
            overhang_segs = segs_data.get("overhang") or []
            ridge_pts = markers_data.get("overhang_ridge_pts") or []
            ordered_ring = _ordered_overhang_boundary_pts(overhang_segs, ridge_pts)
            if len(ordered_ring) >= 2:
                lx = [p[0] for p in ordered_ring] + [ordered_ring[0][0]]
                ly = [p[1] for p in ordered_ring] + [ordered_ring[0][1]]
                lz = [overhang_z_map.get((round(p[0], 2), round(p[1], 2)), z_fn_overhang(p[0], p[1])) for p in ordered_ring]
                lz.append(overhang_z_map.get((round(ordered_ring[0][0], 2), round(ordered_ring[0][1], 2)), z_fn_overhang(ordered_ring[0][0], ordered_ring[0][1])))
                _add_trace(lx, ly, lz, "cyan", "Overhang 1 m", width=width, dash=dash)
            continue
        segs = segs_data.get(key) or []
        lx, ly, lz = _seg_to_plotly(segs, z_fn, endpoints_only=endpoints_only)
        _add_trace(lx, ly, lz, color, name, width=width, dash=dash)
        if key == "upper_rect" and segs and contour_pts_for_z:
            seen_upper_pt: set = set()
            for seg in segs:
                pts = seg if isinstance(seg, (list, tuple)) and seg and isinstance(seg[0], (list, tuple)) else []
                for p in pts:
                    if len(p) < 2:
                        continue
                    x, y = float(p[0]), float(p[1])
                    k = (round(x, 1), round(y, 1))
                    if k in seen_upper_pt:
                        continue
                    d = _min_dist_sq((x, y), contour_pts_for_z)
                    if d > 500:
                        continue
                    seen_upper_pt.add(k)
                    z_top = z_fn(x, y)
                    lx_v = [x, x, None]
                    ly_v = [y, y, None]
                    lz_v = [wh, z_top, None]
                    _add_trace(lx_v, ly_v, lz_v, "yellow", "Etaj superior (pereți)", width=1, dash=None)

    # Markeri (buline)
    def _add_markers(pts: List[List[float]], color: str, name: str, z_fn_m: Any) -> None:
        if not pts:
            return
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        zs = [z_fn_m(x, y) for x, y in zip(xs, ys)]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="markers",
            marker=dict(size=6, color=color, symbol="circle", line=dict(width=1, color="black")),
            name=name, legendgroup=name,
        ))
    _add_markers(markers_data.get("ridge_midpoints") or [], "yellow", "Mijloc ridge", z_fn_ridge)
    _add_markers(markers_data.get("ridge_pink") or [], "hotpink", "Atins de diagonale / capăt opus", z_fn_between)
    brown_pts = [p[0] for p in (markers_data.get("brown_endpoints") or [])]
    _add_markers(brown_pts, "saddlebrown", "Capete segment maro", z_fn_contour)
    _add_markers(markers_data.get("overhang_corners") or [], "lightblue", "Colțuri overhang", z_fn_overhang)
    _add_markers(markers_data.get("overhang_ridge_pts") or [], "darkblue", "Ridge prelungit → overhang", z_fn_overhang)
    # Puncte gri (anchor) – deduplicate, vizibile în 3D
    drawn_anchor_3d: set = set()
    anchor_pts_3d: List[Tuple[float, float, float]] = []
    # Bulină la anchor + segment bulină → anchor pentru fiecare punct overhang
    for (x, y) in overhang_boundary_pts:
        key = (round(x, 2), round(y, 2))
        anchor = overhang_anchor_map.get(key)
        if anchor is None:
            continue
        ax, ay = anchor[0], anchor[1]
        z_bul = overhang_z_map.get(key)
        if z_bul is None:
            continue
        z_anchor = ridge_z if key in overhang_ridge_set else (
            z_fn_pyramid_use(ax, ay) if roof_type_frame in ("4_w", "4.5_w") else z_fn_contour(ax, ay)
        )
        ka = (round(ax, 2), round(ay, 2))
        if ka not in drawn_anchor_3d:
            drawn_anchor_3d.add(ka)
            anchor_pts_3d.append((ax, ay, z_anchor))
        fig.add_trace(go.Scatter3d(
            x=[x, ax, None], y=[y, ay, None], z=[z_bul, z_anchor, None],
            mode="lines", line=dict(color="gray", width=1, dash="dot"),
            name="Overhang → anchor", legendgroup="overhang_anchor",
        ))

    if anchor_pts_3d:
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in anchor_pts_3d],
            y=[p[1] for p in anchor_pts_3d],
            z=[p[2] for p in anchor_pts_3d],
            mode="markers",
            marker=dict(size=8, color="gray", symbol="circle", line=dict(width=1, color="black")),
            name="Anchor (gri)", legendgroup="anchor_gri",
        ))

    # Contur pereți: din mască sau fallback din segmentele "contour" din payload
    wall_segs_2d = []
    if floor_path:
        mask = cv2.imread(floor_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            wall_segs_2d = _get_contour_segments(mask)
    if not wall_segs_2d:
        # Fallback: pereți din conturul exterior (segmente contour) – linii orizontale la z=0 și z=wh + verticale la capete
        contour_segs = segs_data.get("contour") or []
        for seg in contour_segs:
            if not seg or len(seg) < 2:
                continue
            pts = seg if isinstance(seg[0], (list, tuple)) else []
            if len(pts) < 2:
                continue
            p1, p2 = pts[0], pts[1]
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])
            wall_segs_2d.append([[x1, y1], [x2, y2]])
    lx_w, ly_w, lz_w = [], [], []
    contour_pts: List[Tuple[float, float]] = []
    for seg in wall_segs_2d:
        if len(seg) < 2:
            continue
        p1, p2 = seg[0], seg[1]
        x1, y1, x2, y2 = float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])
        contour_pts.extend([(x1, y1), (x2, y2)])
        for z_val in (0.0, wh):
            lx_w.extend([x1, x2, None])
            ly_w.extend([y1, y2, None])
            lz_w.extend([z_val, z_val, None])
    seen_pt: set = set()
    for pt in contour_pts:
        k = (round(pt[0], 1), round(pt[1], 1))
        if k not in seen_pt:
            seen_pt.add(k)
            x, y = pt[0], pt[1]
            lx_w.extend([x, x, None])
            ly_w.extend([y, y, None])
            lz_w.extend([0.0, wh, None])
    _add_trace(lx_w, ly_w, lz_w, "#3498DB", "Contur pereți", width=2)

    fig.update_layout(
        title=f"3D – {subdir.name}",
        scene=dict(aspectmode="data", xaxis=dict(title="x"), yaxis=dict(title="y"), zaxis=dict(title="z")),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=True,
    )
    fig.write_html(str(subdir / "frame.html"), include_plotlyjs="cdn", full_html=True)


def generate_frames_for_roof_types_dir(base_dir: Path, wall_height: float = 300.0) -> None:
    """
    Generează frame.html pentru toate subfolderele de tip acoperiș (1_w, 2_w, 4_w, 4.5_w)
    dintr-un director. Util pentru output-uri existente.
    """
    for name in ("0_w", "1_w", "2_w", "4_w", "4.5_w"):
        subdir = base_dir / name
        if subdir.is_dir() and (subdir / "faces_faces.json").exists():
            try:
                _generate_frame_html(subdir, wall_height=wall_height)
            except Exception:
                pass


def generate_entire_frame_html(
    roof_types_dir: Path,
    output_path: Path,
    wall_height: float = 300.0,
    roof_type: str = "4.5_w",
    overlay_offsets: Optional[Dict[str, Dict[str, int]]] = None,
    floor_roof_types: Optional[Dict[int, str]] = None,
) -> None:
    """
    Generează output/entire/{roof_type}/frame.html și filled.html cu randarea 3D a tuturor
    etajelor casei. Dacă floor_roof_types e dat, fiecare etaj poate avea alt tip (output: entire/mixed/).
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    roof_types_dir = Path(roof_types_dir)
    floor_dirs = sorted(
        [d for d in roof_types_dir.iterdir() if d.is_dir() and d.name.startswith("floor_")],
        key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0,
    )
    floors_info: List[Dict[str, Any]] = []
    fi_path = Path(roof_types_dir) / "floors_info.json"
    if fi_path.exists():
        try:
            floors_info = json.loads(fi_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not floor_dirs and not floors_info:
        return

    floor_payloads: List[Tuple[int, Dict[str, Any], str, str]] = []
    if floor_roof_types:
        for floor_dir in floor_dirs:
            fidx = int(floor_dir.name.split("_")[1]) if floor_dir.name.split("_")[1].isdigit() else 0
            rt = floor_roof_types.get(fidx)
            if not rt or rt not in ("0_w", "1_w", "2_w", "4_w", "4.5_w"):
                continue
            ff_path = floor_dir / rt / "faces_faces.json"
            if ff_path.exists():
                try:
                    payload = json.loads(ff_path.read_text(encoding="utf-8"))
                    floor_payloads.append((fidx, payload, str(ff_path.parent), rt))
                except Exception:
                    pass
        out_subdir = "mixed"
    else:
        for floor_dir in floor_dirs:
            fidx = int(floor_dir.name.split("_")[1]) if floor_dir.name.split("_")[1].isdigit() else 0
            ff_path = floor_dir / roof_type / "faces_faces.json"
            if ff_path.exists():
                try:
                    payload = json.loads(ff_path.read_text(encoding="utf-8"))
                    floor_payloads.append((fidx, payload, str(ff_path.parent), roof_type))
                except Exception:
                    pass
        out_subdir = roof_type

    if not floor_payloads and not floors_info:
        return

    floors_with_roof = {fidx for fidx, _, _, _ in floor_payloads}
    entire_dir = Path(output_path) / "entire" / out_subdir
    entire_dir.mkdir(parents=True, exist_ok=True)

    if overlay_offsets is None:
        offsets_path = Path(roof_types_dir) / "overlay_offsets.json"
        overlay_offsets = {}
        if offsets_path.exists():
            try:
                overlay_offsets = json.loads(offsets_path.read_text(encoding="utf-8"))
            except Exception:
                pass

    wh = float(wall_height)

    def _z_roof_at(faces: List[Dict], x: float, y: float, base_z: float, z_offset: float, tol: float = 40.0) -> float:
        from shapely.geometry import Point as ShapelyPoint
        pt = ShapelyPoint(x, y)
        best, best_dist = None, None
        for f in (faces or []):
            vs = f.get("vertices_3d") or []
            if len(vs) < 3:
                continue
            try:
                from shapely.geometry import Polygon as ShapelyPolygon
                poly = ShapelyPolygon([(float(v[0]), float(v[1])) for v in vs])
                if poly.is_empty:
                    continue
                inside = poly.contains(pt) or poly.buffer(tol).contains(pt)
                dist = float(poly.distance(pt)) if not inside else 0.0
                if not inside and dist > tol:
                    continue
            except Exception:
                continue
            v0, v1, v2 = vs[0], vs[1], vs[2]
            x0, y0, z0 = float(v0[0]), float(v0[1]), float(v0[2]) + z_offset
            x1, y1, z1 = float(v1[0]), float(v1[1]), float(v1[2]) + z_offset
            x2, y2, z2 = float(v2[0]), float(v2[1]), float(v2[2]) + z_offset
            denom = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if abs(denom) < 1e-12:
                z = max(z0, z1, z2)
            else:
                w1 = ((x - x0) * (y2 - y0) - (x2 - x0) * (y - y0)) / denom
                w2 = ((x1 - x0) * (y - y0) - (x - x0) * (y1 - y0)) / denom
                z = (1.0 - w1 - w2) * z0 + w1 * z1 + w2 * z2
            if best is None or (dist <= (best_dist or 1e9) and z > (best or -1e9)):
                best, best_dist = z, dist
        return float(best) if best is not None else base_z + z_offset

    fig = go.Figure()

    for floor_idx, payload, _, rt_floor in floor_payloads:
        z_off = floor_idx * wh
        off = overlay_offsets.get(str(floor_idx), {})
        dx_off = float(off.get("dx", 0))
        dy_off = float(off.get("dy", 0))

        faces = payload.get("faces") or []
        segs_data = payload.get("segments") or {}
        markers_data = payload.get("markers") or {}
        floor_path = payload.get("floor_path")
        fwh = float(payload.get("wall_height", wh))

        ridge_segs = segs_data.get("ridge") or []
        ridge_z_local = fwh
        ridge_pts: List[Tuple[float, float]] = []
        for seg in ridge_segs:
            pts = seg if isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)) else []
            for p in pts:
                if len(p) >= 2:
                    xp, yp = float(p[0]), float(p[1])
                    ridge_pts.append((xp, yp))
                    z = _z_roof_at(faces, xp, yp, fwh, z_off)
                    ridge_z_local = max(ridge_z_local, z - z_off)
            # Pentru acoperiș cu pantă (2_w, 4_w etc.): capetele ridge sunt la streașină (z=fwh).
            # Eșantionăm și la mijlocul segmentului ca ridge_z să fie creasta ridicată.
            if len(pts) >= 2:
                x0, y0 = float(pts[0][0]), float(pts[0][1])
                x1, y1 = float(pts[1][0]), float(pts[1][1])
                for t in (0.5, 0.25, 0.75):
                    xm, ym = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
                    z = _z_roof_at(faces, xm, ym, fwh, z_off)
                    ridge_z_local = max(ridge_z_local, z - z_off)
        ridge_z = ridge_z_local + z_off

        def _z_fn_ridge(x, y):
            return ridge_z

        has_upper = bool(payload.get("has_upper_floor", False))
        upper_rect_pts_entire: List[Tuple[float, float]] = []
        for seg in (segs_data.get("upper_rect") or []):
            pts = seg if isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)) else []
            for p in pts:
                if len(p) >= 2:
                    upper_rect_pts_entire.append((float(p[0]), float(p[1])))

        best_contour_d2_entire: Optional[float] = None
        contour_segs_entire = segs_data.get("contour") or []
        if rt_floor in ("0_w", "1_w") and has_upper and upper_rect_pts_entire and contour_segs_entire:
            def _min_d2_early(pt, lst):
                if not lst:
                    return 1e30
                return min((pt[0] - q[0]) ** 2 + (pt[1] - q[1]) ** 2 for q in lst)
            best_contour_d2_entire = 1e30
            for cseg in contour_segs_entire:
                pts = cseg if isinstance(cseg, (list, tuple)) and cseg and isinstance(cseg[0], (list, tuple)) else []
                if len(pts) >= 2:
                    a, b = (float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1]))
                    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
                    best_contour_d2_entire = min(best_contour_d2_entire, _min_d2_early(mid, upper_rect_pts_entire))
            best_contour_d2_entire = (best_contour_d2_entire or 0) + 100

        def _point_on_contour_near_yellow_entire(px: float, py: float, tol: float = 3.0) -> bool:
            if not upper_rect_pts_entire or rt_floor not in ("0_w", "1_w") or not has_upper or best_contour_d2_entire is None:
                return False
            pt = (px, py)
            for cseg in contour_segs_entire:
                pts = cseg if isinstance(cseg, (list, tuple)) and cseg and isinstance(cseg[0], (list, tuple)) else []
                if len(pts) >= 2:
                    a, b = (float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1]))
                    if not _point_on_segment(pt, a, b, tol):
                        continue
                    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
                    d2 = _min_dist_sq(mid, upper_rect_pts_entire)
                    if d2 <= best_contour_d2_entire:
                        return True
            return False

        def _z_fn_contour(x, y):
            # 0_w: contur la nivelul pereților. 1_w: latura lipită de etajul superior la ridge_z.
            if rt_floor == "1_w" and has_upper and _point_on_contour_near_yellow_entire(x, y):
                return ridge_z_local + z_off
            return fwh + z_off

        def _z_fn_orange(x, y):
            return (ridge_z_local + fwh) / 2.0 + z_off

        contour_pts_for_z: List[Tuple[float, float]] = []
        for seg in (segs_data.get("contour") or []):
            pts = seg if isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)) else []
            for p in pts:
                if len(p) >= 2:
                    contour_pts_for_z.append((float(p[0]), float(p[1])))
        if not contour_pts_for_z and floor_path:
            mask = cv2.imread(floor_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                for seg in _get_contour_segments(mask):
                    if len(seg) >= 2:
                        for p in seg:
                            if len(p) >= 2:
                                contour_pts_for_z.append((float(p[0]), float(p[1])))

        def _min_dist_sq(pt, lst):
            if not lst:
                return 1e30
            return min((pt[0] - q[0]) ** 2 + (pt[1] - q[1]) ** 2 for q in lst)

        def _z_fn_between(x, y):
            d_ridge = _min_dist_sq((x, y), ridge_pts)
            d_contour = _min_dist_sq((x, y), contour_pts_for_z)
            base = ridge_z_local if d_ridge <= d_contour else fwh
            return base + z_off

        ridge_segs_list = segs_data.get("ridge") or []
        def _point_on_ridge(px: float, py: float, tol: float = 5.0) -> bool:
            pt = (px, py)
            for rseg in ridge_segs_list:
                pts = rseg if isinstance(rseg, (list, tuple)) and rseg and isinstance(rseg[0], (list, tuple)) else []
                if len(pts) >= 2:
                    a, b = (float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1]))
                    if _point_on_segment(pt, a, b, tol):
                        return True
            return False

        for key in ("magenta", "pyramid"):
            for seg in (segs_data.get(key) or []):
                pts = seg if isinstance(seg, (list, tuple)) and seg and isinstance(seg[0], (list, tuple)) else []
                for p in pts:
                    if len(p) >= 2:
                        px, py = float(p[0]), float(p[1])
                        if not _point_on_ridge(px, py):
                            contour_pts_for_z.append((px, py))

        def _z_fn_magenta(x, y):
            return _z_fn_contour(x, y) if not _point_on_ridge(x, y) else _z_fn_between(x, y)

        def _z_fn_pyramid(x, y):
            return _z_fn_contour(x, y) if not _point_on_ridge(x, y) else _z_fn_between(x, y)

        def _seg_to_plotly(segs_2d, z_fn, endpoints_only=False):
            lx, ly, lz = [], [], []
            for seg in (segs_2d or []):
                if not seg or len(seg) < 2:
                    continue
                pts = seg if isinstance(seg[0], (list, tuple)) else []
                if not pts or len(pts) < 2:
                    continue
                if endpoints_only:
                    a, b = pts[0], pts[-1]
                    x1, y1 = float(a[0]) + dx_off, float(a[1]) + dy_off
                    x2, y2 = float(b[0]) + dx_off, float(b[1]) + dy_off
                    z1, z2 = z_fn(float(a[0]), float(a[1])), z_fn(float(b[0]), float(b[1]))
                    lx.extend([x1, x2, None])
                    ly.extend([y1, y2, None])
                    lz.extend([z1, z2, None])
                else:
                    for i in range(len(pts) - 1):
                        a, b = pts[i], pts[i + 1]
                        ax, ay = float(a[0]), float(a[1])
                        bx, by = float(b[0]), float(b[1])
                        x1, y1 = ax + dx_off, ay + dy_off
                        x2, y2 = bx + dx_off, by + dy_off
                        z1, z2 = z_fn(ax, ay), z_fn(bx, by)
                        lx.extend([x1, x2, None])
                        ly.extend([y1, y2, None])
                        lz.extend([z1, z2, None])
            return lx, ly, lz

        def _add_trace(lx, ly, lz, color, name, width=2, dash=None):
            if not lx:
                return
            ld = dict(color=color, width=width)
            if dash:
                ld["dash"] = dash
            lbl = f"{name}" if len(floor_payloads) <= 1 else f"{name} (etaj {floor_idx})"
            fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", line=ld, name=lbl, legendgroup=lbl))

        _z_fn_pyramid_use = _z_fn_pyramid if rt_floor in ("4_w", "2_w") else _z_fn_between
        # Overhang 3D: înălțimi conform prelungirii; pentru acoperiș plat (0_w) nu aplicăm pantă
        roof_angle_deg_entire = float(payload.get("roof_angle_deg") or 30.0)
        roof_angle_rad_entire = math.radians(roof_angle_deg_entire)
        tan_roof_entire = 0.0 if rt_floor == "0_w" else math.tan(roof_angle_rad_entire)

        overhang_ridge_set_entire: set = set()
        for pt in (markers_data.get("overhang_ridge_pts") or []):
            if pt and len(pt) >= 2:
                overhang_ridge_set_entire.add((round(float(pt[0]), 2), round(float(pt[1]), 2)))

        contour_corners_entire: List[Tuple[float, float]] = []
        seen_cc_e = set()
        for seg in (segs_data.get("contour") or []):
            pts = seg if isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)) else []
            for p in pts:
                if len(p) >= 2:
                    q = (round(float(p[0]), 2), round(float(p[1]), 2))
                    if q not in seen_cc_e:
                        seen_cc_e.add(q)
                        contour_corners_entire.append((float(p[0]), float(p[1])))

        pyramid_segments_entire: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for seg in (segs_data.get("pyramid") or []):
            pts = seg if isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)) else []
            if len(pts) >= 2:
                a = (float(pts[0][0]), float(pts[0][1]))
                b = (float(pts[-1][0]), float(pts[-1][1]))
                pyramid_segments_entire.append((a, b))

        overhang_boundary_pts_entire: List[Tuple[float, float]] = []
        for pt in (markers_data.get("overhang_corners") or []):
            if pt and len(pt) >= 2:
                overhang_boundary_pts_entire.append((float(pt[0]), float(pt[1])))
        for pt in (markers_data.get("overhang_ridge_pts") or []):
            if pt and len(pt) >= 2:
                overhang_boundary_pts_entire.append((float(pt[0]), float(pt[1])))
        for seg in (segs_data.get("overhang") or []):
            if not seg or len(seg) < 2:
                continue
            for p in (seg[0], seg[-1]):
                if len(p) >= 2:
                    overhang_boundary_pts_entire.append((float(p[0]), float(p[1])))

        overhang_z_map_entire: Dict[Tuple[float, float], float] = {}
        overhang_anchor_map_entire: Dict[Tuple[float, float], Tuple[float, float]] = {}
        ridge_seg_tuples_entire: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for seg in (segs_data.get("ridge") or []):
            pts = seg if isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)) else []
            if len(pts) >= 2:
                a = (float(pts[0][0]), float(pts[0][1]))
                b = (float(pts[1][0]), float(pts[1][1]))
                ridge_seg_tuples_entire.append((a, b))

        for (x, y) in overhang_boundary_pts_entire:
            key = (round(x, 2), round(y, 2))
            if key in overhang_z_map_entire:
                continue
            if key in overhang_ridge_set_entire:
                overhang_z_map_entire[key] = ridge_z
                if ridge_seg_tuples_entire:
                    best_ax, best_ay, best_d2 = 0.0, 0.0, 1e30
                    for (a, b) in ridge_seg_tuples_entire:
                        qx, qy, d2 = _closest_point_on_segment((x, y), a, b)
                        if d2 < best_d2:
                            best_d2 = d2
                            best_ax, best_ay = qx, qy
                    overhang_anchor_map_entire[key] = (best_ax, best_ay)
                continue
            if rt_floor in ("4_w", "4.5_w") and pyramid_segments_entire:
                best_d2_e = 1e30
                best_z_anchor_e = fwh + z_off
                best_ax_e, best_ay_e = x, y
                for (a, b) in pyramid_segments_entire:
                    qx, qy, d2 = _closest_point_on_segment((x, y), a, b)
                    if d2 < best_d2_e:
                        best_d2_e = d2
                        best_z_anchor_e = _z_fn_pyramid_use(qx, qy)
                        best_ax_e, best_ay_e = qx, qy
                d_e = math.sqrt(best_d2_e)
                z_val_e = best_z_anchor_e - d_e * tan_roof_entire
                overhang_z_map_entire[key] = max(0.0, z_val_e)
                overhang_anchor_map_entire[key] = (best_ax_e, best_ay_e)
            elif contour_corners_entire:
                best_d2_e = 1e30
                best_q_e = (x, y)
                for q in contour_corners_entire:
                    d2 = (x - q[0]) ** 2 + (y - q[1]) ** 2
                    if d2 < best_d2_e:
                        best_d2_e = d2
                        best_q_e = q
                d_e = math.sqrt(best_d2_e)
                z_val_e = (fwh - d_e * tan_roof_entire) + z_off
                overhang_z_map_entire[key] = max(0.0, z_val_e)
                overhang_anchor_map_entire[key] = best_q_e
            else:
                overhang_z_map_entire[key] = _z_fn_contour(x, y) if rt_floor == "0_w" else _z_fn_between(x, y)

        def _z_fn_overhang(x: float, y: float) -> float:
            k = (round(x, 2), round(y, 2))
            if k in overhang_z_map_entire:
                return overhang_z_map_entire[k]
            return _z_fn_contour(x, y) if rt_floor == "0_w" else _z_fn_between(x, y)
        # Nu randăm segmente galbene (upper_rect / Etaj superior) în 3D
        seg_cfg = [
            ("contour", "green", "Contur exterior", 2, None, _z_fn_contour, False),
            ("ridge", "darkred", "Ridge", 2.5, None, _z_fn_ridge, False),
            ("magenta", "#CC00FF", "Linii intersecție", 2, None, _z_fn_magenta, False),
            ("blue", "blue", "Segment albastru", 2.5, None, _z_fn_between, False),
            ("pyramid", "blue", "Diagonale 45°", 1.5, "dash", _z_fn_pyramid_use, True),
            ("orange", "orange", "Paralelă diagonale", 2, None, _z_fn_orange, False),
            ("overhang", "cyan", "Overhang 1 m", 2, "dash", _z_fn_overhang, False),
            ("wall_support", "deepskyblue", "Suport perete", 2, None, _z_fn_between, False),
            ("brown", "saddlebrown", "Laturi opuse", 2.5, None, _z_fn_between, False),
        ]
        for key, color, name, width, dash, z_fn, ep in seg_cfg:
            if key == "overhang" and not (payload.get("faces") or []):
                continue
            if key == "overhang":
                overhang_segs = segs_data.get("overhang") or []
                ridge_pts = markers_data.get("overhang_ridge_pts") or []
                ordered_ring = _ordered_overhang_boundary_pts(overhang_segs, ridge_pts)
                if len(ordered_ring) >= 2:
                    lx = [p[0] + dx_off for p in ordered_ring] + [ordered_ring[0][0] + dx_off]
                    ly = [p[1] + dy_off for p in ordered_ring] + [ordered_ring[0][1] + dy_off]
                    lz = [overhang_z_map_entire.get((round(p[0], 2), round(p[1], 2)), _z_fn_overhang(p[0], p[1])) for p in ordered_ring]
                    lz.append(overhang_z_map_entire.get((round(ordered_ring[0][0], 2), round(ordered_ring[0][1], 2)), _z_fn_overhang(ordered_ring[0][0], ordered_ring[0][1])))
                    _add_trace(lx, ly, lz, "cyan", name, width=width, dash=dash)
                continue
            segs = segs_data.get(key) or []
            lx, ly, lz = _seg_to_plotly(segs, z_fn, endpoints_only=ep)
            _add_trace(lx, ly, lz, color, name, width=width, dash=dash)

        for pts in (markers_data.get("ridge_midpoints") or []):
            if pts and len(pts) >= 2:
                xs, ys = [float(pts[0]) + dx_off], [float(pts[1]) + dy_off]
                zs = [_z_fn_ridge(float(pts[0]), float(pts[1]))]
                lbl = "Mijloc ridge" if len(floor_payloads) <= 1 else f"Mijloc ridge (etaj {floor_idx})"
                fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="markers",
                    marker=dict(size=6, color="yellow", symbol="circle", line=dict(width=1, color="black")),
                    name=lbl, legendgroup=lbl))
        for pts in (markers_data.get("ridge_pink") or []):
            if pts and len(pts) >= 2:
                x0, y0 = float(pts[0]), float(pts[1])
                xs, ys = [x0 + dx_off], [y0 + dy_off]
                zs = [_z_fn_between(x0, y0)]
                lbl = "Atins diagonale" if len(floor_payloads) <= 1 else f"Atins (etaj {floor_idx})"
                fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="markers",
                    marker=dict(size=6, color="hotpink", symbol="circle", line=dict(width=1, color="black")),
                    name=lbl, legendgroup=lbl))
        for item in (markers_data.get("brown_endpoints") or []):
            p = item[0] if isinstance(item, (list, tuple)) and item else []
            if p and len(p) >= 2:
                x0, y0 = float(p[0]), float(p[1])
                xs, ys = [x0 + dx_off], [y0 + dy_off]
                zs = [_z_fn_contour(x0, y0)]
                lbl = "Capete maro" if len(floor_payloads) <= 1 else f"Capete maro (etaj {floor_idx})"
                fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="markers",
                    marker=dict(size=6, color="saddlebrown", symbol="circle", line=dict(width=1, color="black")),
                    name=lbl, legendgroup=lbl))
        for pts in (markers_data.get("overhang_corners") or []):
            if pts and len(pts) >= 2:
                x0, y0 = float(pts[0]), float(pts[1])
                xs, ys = [x0 + dx_off], [y0 + dy_off]
                zs = [_z_fn_overhang(x0, y0)]
                lbl = "Colțuri overhang" if len(floor_payloads) <= 1 else f"Colțuri overhang (etaj {floor_idx})"
                fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="markers",
                    marker=dict(size=6, color="lightblue", symbol="circle", line=dict(width=1, color="black")),
                    name=lbl, legendgroup=lbl))
        for pts in (markers_data.get("overhang_ridge_pts") or []):
            if pts and len(pts) >= 2:
                x0, y0 = float(pts[0]), float(pts[1])
                xs, ys = [x0 + dx_off], [y0 + dy_off]
                zs = [_z_fn_overhang(x0, y0)]
                lbl = "Ridge → overhang" if len(floor_payloads) <= 1 else f"Ridge → overhang (etaj {floor_idx})"
                fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="markers",
                    marker=dict(size=6, color="darkblue", symbol="circle", line=dict(width=1, color="black")),
                    name=lbl, legendgroup=lbl))

        for (x, y) in overhang_boundary_pts_entire:
            key = (round(x, 2), round(y, 2))
            anchor = overhang_anchor_map_entire.get(key)
            if anchor is None:
                continue
            z_bul = overhang_z_map_entire.get(key)
            if z_bul is None:
                continue
            ax, ay = anchor[0], anchor[1]
            z_anchor = ridge_z if key in overhang_ridge_set_entire else (
                _z_fn_pyramid_use(ax, ay) if rt_floor in ("4_w", "4.5_w") else _z_fn_contour(ax, ay)
            )
            fig.add_trace(go.Scatter3d(
                x=[ax + dx_off], y=[ay + dy_off], z=[z_anchor], mode="markers",
                marker=dict(size=5, color="gray", symbol="circle", line=dict(width=1, color="black")),
                name="Anchor overhang", legendgroup="anchor_overhang",
            ))
            fig.add_trace(go.Scatter3d(
                x=[x + dx_off, ax + dx_off, None], y=[y + dy_off, ay + dy_off, None], z=[z_bul, z_anchor, None],
                mode="lines", line=dict(color="gray", width=1, dash="dot"),
                name="Overhang → anchor", legendgroup="overhang_anchor",
            ))

        wall_segs_2d = []
        if floor_path:
            mask = cv2.imread(floor_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                wall_segs_2d = _get_contour_segments(mask)
        if not wall_segs_2d:
            contour_segs = (payload.get("segments") or {}).get("contour") or []
            for seg in contour_segs:
                if not seg or len(seg) < 2:
                    continue
                pts = seg if isinstance(seg[0], (list, tuple)) else []
                if len(pts) < 2:
                    continue
                p1, p2 = pts[0], pts[1]
                wall_segs_2d.append([[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]])
        lx_w, ly_w, lz_w = [], [], []
        contour_pts = []
        for seg in wall_segs_2d:
            if len(seg) < 2:
                continue
            p1, p2 = seg[0], seg[1]
            x1, y1 = float(p1[0]) + dx_off, float(p1[1]) + dy_off
            x2, y2 = float(p2[0]) + dx_off, float(p2[1]) + dy_off
            contour_pts.extend([(x1, y1), (x2, y2)])
            for z_val in (z_off, fwh + z_off):
                lx_w.extend([x1, x2, None])
                ly_w.extend([y1, y2, None])
                lz_w.extend([z_val, z_val, None])
        seen_pt = set()
        for pt in contour_pts:
            k = (round(pt[0], 1), round(pt[1], 1))
            if k not in seen_pt:
                seen_pt.add(k)
                x, y = pt[0], pt[1]  # pt already has offset applied
                lx_w.extend([x, x, None])
                ly_w.extend([y, y, None])
                lz_w.extend([z_off, fwh + z_off, None])
        lbl = "Contur pereți" if len(floor_payloads) <= 1 else f"Contur pereți (etaj {floor_idx})"
        if lx_w:
            fig.add_trace(go.Scatter3d(x=lx_w, y=ly_w, z=lz_w, mode="lines",
                line=dict(color="#3498DB", width=2), name=lbl, legendgroup=lbl))

    for item in floors_info:
        floor_idx = int(item.get("floor_idx", 0))
        if floor_idx in floors_with_roof:
            continue
        floor_path = item.get("path") or ""
        if not floor_path:
            continue
        z_off = floor_idx * wh
        off = overlay_offsets.get(str(floor_idx), {})
        dx_off = float(off.get("dx", 0))
        dy_off = float(off.get("dy", 0))
        mask = cv2.imread(floor_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        wall_segs_2d = _get_contour_segments(mask)
        lx_w, ly_w, lz_w = [], [], []
        contour_pts = []
        for seg in wall_segs_2d:
            if len(seg) < 2:
                continue
            p1, p2 = seg[0], seg[1]
            x1, y1 = float(p1[0]) + dx_off, float(p1[1]) + dy_off
            x2, y2 = float(p2[0]) + dx_off, float(p2[1]) + dy_off
            contour_pts.extend([(x1, y1), (x2, y2)])
            for z_val in (z_off, wh + z_off):
                lx_w.extend([x1, x2, None])
                ly_w.extend([y1, y2, None])
                lz_w.extend([z_val, z_val, None])
        seen_pt = set()
        for pt in contour_pts:
            k = (round(pt[0], 1), round(pt[1], 1))
            if k not in seen_pt:
                seen_pt.add(k)
                x, y = pt[0], pt[1]
                lx_w.extend([x, x, None])
                ly_w.extend([y, y, None])
                lz_w.extend([z_off, wh + z_off, None])
        lbl = f"Contur pereți (etaj {floor_idx}, fără acoperiș)"
        if lx_w:
            fig.add_trace(go.Scatter3d(x=lx_w, y=ly_w, z=lz_w, mode="lines",
                line=dict(color="#3498DB", width=2), name=lbl, legendgroup=lbl))

    n_floors = len(floors_info) if floors_info else len(floor_payloads)
    title_roof = "mixed" if floor_roof_types else roof_type
    fig.update_layout(
        title=f"3D – Casa întreagă ({title_roof})" + (" (toate etajele)" if n_floors > 1 else ""),
        scene=dict(aspectmode="data", xaxis=dict(title="x"), yaxis=dict(title="y"), zaxis=dict(title="z")),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=True,
    )
    fig.write_html(str(entire_dir / "frame.html"), include_plotlyjs="cdn", full_html=True)

    seed = hash(str(floor_roof_types or roof_type)) % (2**32)
    rng = random.Random(seed)
    for floor_idx, payload, _, rt_floor in floor_payloads:
        z_off = floor_idx * wh
        off = overlay_offsets.get(str(floor_idx), {})
        dx_off = float(off.get("dx", 0))
        dy_off = float(off.get("dy", 0))
        fwh = float(payload.get("wall_height", wh))
        faces = payload.get("faces") or []
        ridge_pts: List[Tuple[float, float]] = []
        for seg in (payload.get("segments") or {}).get("ridge") or []:
            pts = seg if isinstance(seg, list) and isinstance(seg[0] if seg else None, (list, tuple)) else []
            for p in (pts or []):
                if len(p) >= 2:
                    ridge_pts.append((float(p[0]), float(p[1])))
        contour_pts_for_z: List[Tuple[float, float]] = []
        contour_segs_for_z = (payload.get("segments") or {}).get("contour") or []
        for seg in contour_segs_for_z:
            pts = seg if isinstance(seg, list) and isinstance(seg[0] if seg else None, (list, tuple)) else []
            for p in (pts or []):
                if len(p) >= 2:
                    contour_pts_for_z.append((float(p[0]), float(p[1])))
        ridge_segs_list_z = (payload.get("segments") or {}).get("ridge") or []
        def _point_on_ridge_z(px: float, py: float, tol: float = 5.0) -> bool:
            pt = (px, py)
            for rseg in ridge_segs_list_z:
                pts = rseg if isinstance(rseg, (list, tuple)) and rseg and isinstance(rseg[0], (list, tuple)) else []
                if len(pts) >= 2:
                    a, b = (float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1]))
                    if _point_on_segment(pt, a, b, tol):
                        return True
            return False
        for key in ("magenta", "pyramid"):
            for seg in (payload.get("segments") or {}).get(key) or []:
                pts = seg if isinstance(seg, (list, tuple)) and seg and isinstance(seg[0], (list, tuple)) else []
                for p in pts:
                    if len(p) >= 2:
                        px, py = float(p[0]), float(p[1])
                        if not _point_on_ridge_z(px, py):
                            contour_pts_for_z.append((px, py))
        ridge_z_local = fwh
        for xp, yp in ridge_pts:
            z = _z_roof_at(faces, xp, yp, fwh, 0.0)
            ridge_z_local = max(ridge_z_local, z)
        def _min_d2(pt, lst):
            if not lst:
                return 1e30
            return min((pt[0] - q[0]) ** 2 + (pt[1] - q[1]) ** 2 for q in lst)

        orange_segs = (payload.get("segments") or {}).get("orange") or []
        z_orange = (ridge_z_local + fwh) / 2.0 + z_off
        tol_on = 15.0

        has_upper_filled = bool(payload.get("has_upper_floor", False))
        upper_rect_pts_filled: List[Tuple[float, float]] = []
        contour_segs_filled = (payload.get("segments") or {}).get("contour") or []
        for seg in (payload.get("segments") or {}).get("upper_rect") or []:
            pts = seg if isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)) else []
            for p in pts:
                if len(p) >= 2:
                    upper_rect_pts_filled.append((float(p[0]), float(p[1])))
        best_contour_d2_filled: Optional[float] = None
        if rt_floor in ("0_w", "1_w") and has_upper_filled and upper_rect_pts_filled and contour_segs_filled:
            best_contour_d2_filled = 1e30
            for cseg in contour_segs_filled:
                pts = cseg if isinstance(cseg, (list, tuple)) and cseg and isinstance(cseg[0], (list, tuple)) else []
                if len(pts) >= 2:
                    a, b = (float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1]))
                    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
                    best_contour_d2_filled = min(best_contour_d2_filled, _min_d2(mid, upper_rect_pts_filled))
            best_contour_d2_filled = (best_contour_d2_filled or 0) + 100

        def _point_on_contour_near_yellow_filled(px: float, py: float, tol: float = 5.0) -> bool:
            if not upper_rect_pts_filled or rt_floor not in ("0_w", "1_w") or not has_upper_filled or best_contour_d2_filled is None:
                return False
            pt = (px, py)
            for cseg in contour_segs_filled:
                pts = cseg if isinstance(cseg, (list, tuple)) and cseg and isinstance(cseg[0], (list, tuple)) else []
                if len(pts) >= 2:
                    a, b = (float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1]))
                    if not _point_on_segment(pt, a, b, tol):
                        continue
                    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
                    if _min_d2(mid, upper_rect_pts_filled) <= best_contour_d2_filled:
                        return True
            return False

        def _point_on_contour_segment(px: float, py: float, tol: float = 10.0) -> bool:
            """True dacă (px,py) se află pe un segment de contur."""
            pt = (px, py)
            for cseg in contour_segs_filled:
                pts = cseg if isinstance(cseg, (list, tuple)) and cseg and isinstance(cseg[0], (list, tuple)) else []
                if len(pts) >= 2:
                    a, b = (float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1]))
                    if _point_on_segment(pt, a, b, tol):
                        return True
            return False

        def _point_on_magenta_pyramid_non_ridge(px: float, py: float, tol: float = 10.0) -> bool:
            """True dacă (px,py) e pe magenta/pyramid și nu pe ridge (→ nivel contur)."""
            if _point_on_ridge_z(px, py):
                return False
            pt = (px, py)
            for key in ("magenta", "pyramid"):
                for seg in (payload.get("segments") or {}).get(key) or []:
                    pts = seg if isinstance(seg, (list, tuple)) and seg and isinstance(seg[0], (list, tuple)) else []
                    if len(pts) >= 2:
                        a, b = (float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1]))
                        if _point_on_segment(pt, a, b, tol):
                            return True
            return False

        def _z_fn(x: float, y: float) -> float:
            # Pentru 0_w (plat) tot acoperișul la același nivel – niciun colț ridicat
            if rt_floor == "0_w":
                return fwh + z_off
            if rt_floor == "1_w" and has_upper_filled and _point_on_contour_near_yellow_filled(x, y):
                return ridge_z_local + z_off
            if rt_floor == "4.5_w" and orange_segs:
                for oseg in orange_segs:
                    if len(oseg) >= 2:
                        pts = oseg if isinstance(oseg[0], (list, tuple)) else []
                        if len(pts) >= 2:
                            a, b = (float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1]))
                            if _point_on_segment((x, y), a, b, tol_on):
                                return z_orange
            if _point_on_ridge_z(x, y):
                return ridge_z_local + z_off
            if _point_on_contour_segment(x, y) or _point_on_magenta_pyramid_non_ridge(x, y):
                return fwh + z_off
            d_ridge = _min_d2((x, y), ridge_pts)
            d_contour = _min_d2((x, y), contour_pts_for_z)
            base = ridge_z_local if d_ridge <= d_contour else fwh
            return base + z_off

        def _overlap_length_edge_with_rect(ax: float, ay: float, bx: float, by: float) -> float:
            """Lungimea suprapunerii segmentului (a,b) cu marginile upper_rect (suprafața lipită de perete)."""
            if not upper_rect_pts_filled or len(upper_rect_pts_filled) < 3:
                return 0.0
            xs = [q[0] for q in upper_rect_pts_filled]
            ys = [q[1] for q in upper_rect_pts_filled]
            uminx, umaxx, uminy, umaxy = min(xs), max(xs), min(ys), max(ys)
            rect_edges = [
                (uminx, uminy, umaxx, uminy),
                (umaxx, uminy, umaxx, umaxy),
                (umaxx, umaxy, uminx, umaxy),
                (uminx, umaxy, uminx, uminy),
            ]
            tol_par, tol_dist = 15.0, 12.0
            total = 0.0
            dx_ab, dy_ab = bx - ax, by - ay
            len_ab = (dx_ab * dx_ab + dy_ab * dy_ab) ** 0.5
            if len_ab < 1e-9:
                return 0.0
            for ex1, ey1, ex2, ey2 in rect_edges:
                cross = abs(dx_ab * (ey2 - ey1) - dy_ab * (ex2 - ex1))
                len_rect = ((ex2 - ex1) ** 2 + (ey2 - ey1) ** 2) ** 0.5
                if len_rect < 1e-9:
                    continue
                if cross > tol_par:
                    continue
                proj_a = (ax - ex1) * (ex2 - ex1) + (ay - ey1) * (ey2 - ey1)
                proj_b = (bx - ex1) * (ex2 - ex1) + (by - ey1) * (ey2 - ey1)
                t_a, t_b = proj_a / len_rect, proj_b / len_rect
                t_lo, t_hi = min(t_a, t_b), max(t_a, t_b)
                overlap_1d = max(0, min(len_rect, t_hi) - max(0, t_lo))
                dist_a = abs((ax - ex1) * (ey2 - ey1) - (ay - ey1) * (ex2 - ex1)) / len_rect
                dist_b = abs((bx - ex1) * (ey2 - ey1) - (by - ey1) * (ex2 - ex1)) / len_rect
                if min(dist_a, dist_b) <= tol_dist:
                    total += overlap_1d
            return total

        polygons_2d = payload.get("polygons_2d") or []
        faces_list = payload.get("faces") or []
        # Folosim fețele din payload (aceleași ca în lines.png/faces.png); z din vertices_3d
        poly_and_zs: List[Tuple[List[List[float]], Optional[List[float]]]] = []
        if faces_list:
            for f in faces_list:
                vs = f.get("vertices_3d") or []
                if len(vs) >= 3:
                    poly_2d = [[float(v[0]), float(v[1])] for v in vs]
                    zs_face = [float(v[2]) for v in vs]
                    poly_and_zs.append((poly_2d, zs_face))
        elif polygons_2d:
            for poly in polygons_2d:
                poly_and_zs.append((poly, None))
        faces_for_unfold: List[Dict[str, Any]] = []
        poly_idx = 0
        for poly, zs_from_face in poly_and_zs:
            if not poly or len(poly) < 3:
                continue
            poly_idx += 1
            xs = [float(p[0]) + dx_off for p in poly]
            ys = [float(p[1]) + dy_off for p in poly]
            if zs_from_face is not None and len(zs_from_face) == len(poly):
                zs = [z + z_off for z in zs_from_face]
            else:
                # Fallback: _z_fn (0_w plat sau 1_w cu muchie ridicată)
                if rt_floor == "1_w" and has_upper_filled and upper_rect_pts_filled:
                    best_edge_idx = -1
                    best_overlap = -1.0
                    n = len(poly)
                    for ei in range(n):
                        ax, ay = float(poly[ei][0]), float(poly[ei][1])
                        bx, by = float(poly[(ei + 1) % n][0]), float(poly[(ei + 1) % n][1])
                        ov = _overlap_length_edge_with_rect(ax, ay, bx, by)
                        if ov > best_overlap:
                            best_overlap, best_edge_idx = ov, ei
                    if best_overlap <= 0:
                        best_edge_d2, best_edge_idx = 1e30, 0
                        for ei in range(n):
                            ax, ay = float(poly[ei][0]), float(poly[ei][1])
                            bx, by = float(poly[(ei + 1) % n][0]), float(poly[(ei + 1) % n][1])
                            mid = ((ax + bx) / 2.0, (ay + by) / 2.0)
                            d2 = _min_d2(mid, upper_rect_pts_filled)
                            if d2 < best_edge_d2:
                                best_edge_d2, best_edge_idx = d2, ei
                    raised_z = ridge_z_local + z_off
                    zs = []
                    for vi in range(n):
                        on_raised_edge = vi == best_edge_idx or vi == (best_edge_idx + 1) % n
                        zs.append(raised_z if on_raised_edge else _z_fn(float(poly[vi][0]), float(poly[vi][1])))
                else:
                    zs = [_z_fn(float(p[0]), float(p[1])) for p in poly]
            verts_3d = [[float(poly[i][0]), float(poly[i][1]), zs[i]] for i in range(len(poly))]
            faces_for_unfold.append({"vertices_3d": verts_3d})
            i_arr, j_arr, k_arr = [], [], []
            n = len(poly)
            for t in range(1, n - 1):
                i_arr.append(0)
                j_arr.append(t)
                k_arr.append(t + 1)
            r, g, b = rng.random(), rng.random(), rng.random()
            color_hex = f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.6)"
            fig.add_trace(go.Mesh3d(
                x=xs, y=ys, z=zs, i=i_arr, j=j_arr, k=k_arr,
                color=color_hex, opacity=0.6, showlegend=False,
            ))
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            cz = sum(zs) / len(zs)
            fig.add_trace(go.Scatter3d(
                x=[cx], y=[cy], z=[cz],
                mode="text", text=[str(poly_idx)],
                textfont=dict(size=14, color="black"),
                showlegend=False, hoverinfo="skip",
            ))
        if faces_for_unfold:
            floor_path_filled = payload.get("floor_path")
            plan_h, plan_w = 0, 0
            if floor_path_filled:
                img_plan = cv2.imread(floor_path_filled, cv2.IMREAD_GRAYSCALE)
                if img_plan is not None:
                    plan_h, plan_w = img_plan.shape[:2]
            if plan_h > 0 and plan_w > 0:
                num_roof_faces = int(payload.get("num_roof_faces", len(faces_for_unfold)))
                num_roof_faces = min(max(0, num_roof_faces), len(faces_for_unfold))
                roof_faces_for_unfold = faces_for_unfold[:num_roof_faces]
                overhang_faces_for_unfold = faces_for_unfold[num_roof_faces:]
                base_unfold = roof_types_dir / f"floor_{floor_idx}" / rt_floor
                unfold_roof_dir = base_unfold / "unfold_roof"
                unfold_overhang_dir = base_unfold / "unfold_overhang"
                try:
                    if roof_faces_for_unfold:
                        generate_unfold_masks_for_roof_types(roof_faces_for_unfold, plan_h, plan_w, unfold_roof_dir)
                    if overhang_faces_for_unfold:
                        generate_unfold_masks_for_roof_types(overhang_faces_for_unfold, plan_h, plan_w, unfold_overhang_dir)
                except Exception:
                    pass
    fig.update_layout(title=f"3D – Casa întreagă ({title_roof}) – umplut")
    fig.write_html(str(entire_dir / "filled.html"), include_plotlyjs="cdn", full_html=True)
    try:
        fig.write_image(str(entire_dir / "filled.png"), width=900, height=700, scale=2)
    except Exception:
        pass


def _mask_area_and_contour_px(mask: np.ndarray) -> Tuple[float, float]:
    """Returnează (area_px, contour_px) pentru o mască binară."""
    if mask is None or mask.size == 0:
        return 0.0, 0.0
    area_px = float(np.count_nonzero(mask > 0))
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_px = 0.0
    for c in contours:
        if len(c) >= 3:
            contour_px += float(cv2.arcLength(c, closed=True))
    return area_px, contour_px


def _load_meters_per_pixel_per_floor(output_path: Path, num_floors: int) -> Dict[int, float]:
    """Încearcă să încarce scale_m_per_px per etaj din scale/ (când rulăm din holzbot-engine)."""
    result: Dict[int, float] = {}
    try:
        scale_root = output_path.resolve().parent.parent / "scale"
        if not scale_root.is_dir():
            return result
        scale_files = sorted(scale_root.glob("*/cubicasa_result.json"))
        if not scale_files:
            scale_files = sorted(scale_root.glob("*/scale_result.json"))
        for floor_idx in range(min(num_floors, len(scale_files))):
            try:
                data = json.loads(scale_files[floor_idx].read_text(encoding="utf-8"))
                mpp = (
                    data.get("measurements", {}).get("metrics", {}).get("scale_m_per_px")
                    or data.get("meters_per_pixel")
                    or data.get("scale_m_per_px")
                )
                if mpp is not None and float(mpp) > 0:
                    result[floor_idx] = float(mpp)
            except Exception:
                pass
    except Exception:
        pass
    return result


def populate_mixed_unfold_and_metrics(
    entire_mixed_dir: Path,
    roof_types_dir: Path,
    floor_roof_types: Dict[int, str],
    output_path: Path,
) -> None:
    """
    Pentru entire/mixed/: creează unfold_roof/ (doar fețe acoperiș) și unfold_overhang/ (doar fețe overhang).
    roof_metrics.json: area/contour per față, per etaj, total – separat pentru roof și overhang.
    Pentru preț: izolație etc. = doar unfold_roof; restul = măsurători din roof + overhang.
    """
    entire_mixed_dir = Path(entire_mixed_dir)
    roof_types_dir = Path(roof_types_dir)
    unfold_roof_dir = entire_mixed_dir / "unfold_roof"
    unfold_overhang_dir = entire_mixed_dir / "unfold_overhang"
    unfold_roof_dir.mkdir(parents=True, exist_ok=True)
    unfold_overhang_dir.mkdir(parents=True, exist_ok=True)

    floor_dirs = sorted(
        [d for d in roof_types_dir.iterdir() if d.is_dir() and d.name.startswith("floor_")],
        key=lambda d: int(d.name.split("_")[1]) if len(d.name.split("_")) > 1 and d.name.split("_")[1].isdigit() else 0,
    )

    faces_metrics_roof: List[Dict[str, Any]] = []
    faces_metrics_overhang: List[Dict[str, Any]] = []
    floor_totals_roof: Dict[int, Dict[str, Any]] = {}
    floor_totals_overhang: Dict[int, Dict[str, Any]] = {}
    total_area_px_roof = 0.0
    total_contour_px_roof = 0.0
    total_area_px_overhang = 0.0
    total_contour_px_overhang = 0.0

    mpp_by_floor = _load_meters_per_pixel_per_floor(output_path, len(floor_dirs) + 1)

    def _process_unfold_src(unfold_src: Path, out_dir: Path, fidx: int, faces_metrics: List[Dict[str, Any]], floor_totals: Dict[int, Dict[str, Any]]) -> None:
        if not unfold_src.is_dir():
            return
        mpp = mpp_by_floor.get(fidx)
        floor_area_px = 0.0
        floor_contour_px = 0.0
        for png in sorted(unfold_src.glob("*.png")):
            face_num = png.stem
            if not face_num.isdigit():
                continue
            dst_name = f"floor_{fidx}_{face_num}.png"
            dst_path = out_dir / dst_name
            try:
                shutil.copy2(png, dst_path)
            except Exception:
                continue
            mask = cv2.imread(str(dst_path), cv2.IMREAD_GRAYSCALE)
            area_px, contour_px = _mask_area_and_contour_px(mask)
            floor_area_px += area_px
            floor_contour_px += contour_px
            area_m2 = (area_px * (mpp ** 2)) if (mpp and mpp > 0) else None
            contour_m = (contour_px * mpp) if (mpp and mpp > 0) else None
            faces_metrics.append({
                "floor_idx": fidx,
                "face_id": int(face_num),
                "filename": dst_name,
                "area_px": round(area_px, 2),
                "area_m2": round(area_m2, 6) if area_m2 is not None else None,
                "contour_px": round(contour_px, 2),
                "contour_m": round(contour_m, 4) if contour_m is not None else None,
            })
        floor_area_m2 = (floor_area_px * (mpp ** 2)) if (mpp and mpp > 0) else None
        floor_contour_m = (floor_contour_px * mpp) if (mpp and mpp > 0) else None
        floor_totals[fidx] = {
            "area_px": round(floor_area_px, 2),
            "area_m2": round(floor_area_m2, 6) if floor_area_m2 is not None else None,
            "contour_px": round(floor_contour_px, 2),
            "contour_m": round(floor_contour_m, 4) if floor_contour_m is not None else None,
        }

    for floor_dir in floor_dirs:
        fidx = int(floor_dir.name.split("_")[1]) if len(floor_dir.name.split("_")) > 1 and floor_dir.name.split("_")[1].isdigit() else 0
        rt = floor_roof_types.get(fidx, "2_w")
        if not rt:
            continue
        base_src = floor_dir / rt
        _process_unfold_src(base_src / "unfold_roof", unfold_roof_dir, fidx, faces_metrics_roof, floor_totals_roof)
        _process_unfold_src(base_src / "unfold_overhang", unfold_overhang_dir, fidx, faces_metrics_overhang, floor_totals_overhang)

    total_area_px_roof = sum(f["area_px"] for f in faces_metrics_roof)
    total_contour_px_roof = sum(f["contour_px"] for f in faces_metrics_roof)
    total_area_px_overhang = sum(f["area_px"] for f in faces_metrics_overhang)
    total_contour_px_overhang = sum(f["contour_px"] for f in faces_metrics_overhang)

    def _total_m2(area_px: float, mpp_dict: Dict[int, float]) -> Optional[float]:
        if not mpp_dict:
            return None
        avg_mpp = sum(mpp_dict.values()) / len(mpp_dict)
        return area_px * (avg_mpp ** 2) if avg_mpp > 0 else None

    def _total_contour_m(contour_px: float, mpp_dict: Dict[int, float]) -> Optional[float]:
        if not mpp_dict:
            return None
        avg_mpp = sum(mpp_dict.values()) / len(mpp_dict)
        return contour_px * avg_mpp if avg_mpp > 0 else None

    metrics = {
        "unfold_roof": {
            "faces": faces_metrics_roof,
            "by_floor": floor_totals_roof,
            "total": {
                "area_px": round(total_area_px_roof, 2),
                "area_m2": round(_total_m2(total_area_px_roof, mpp_by_floor), 6) if mpp_by_floor else None,
                "contour_px": round(total_contour_px_roof, 2),
                "contour_m": round(_total_contour_m(total_contour_px_roof, mpp_by_floor), 4) if mpp_by_floor else None,
            },
        },
        "unfold_overhang": {
            "faces": faces_metrics_overhang,
            "by_floor": floor_totals_overhang,
            "total": {
                "area_px": round(total_area_px_overhang, 2),
                "area_m2": round(_total_m2(total_area_px_overhang, mpp_by_floor), 6) if mpp_by_floor else None,
                "contour_px": round(total_contour_px_overhang, 2),
                "contour_m": round(_total_contour_m(total_contour_px_overhang, mpp_by_floor), 4) if mpp_by_floor else None,
            },
        },
        "total_combined": {
            "area_px": round(total_area_px_roof + total_area_px_overhang, 2),
            "area_m2": round((_total_m2(total_area_px_roof + total_area_px_overhang, mpp_by_floor) or 0), 6) if mpp_by_floor else None,
            "contour_px": round(total_contour_px_roof + total_contour_px_overhang, 2),
            "contour_m": round((_total_contour_m(total_contour_px_roof + total_contour_px_overhang, mpp_by_floor) or 0), 4) if mpp_by_floor else None,
        },
        "meters_per_pixel_by_floor": {str(k): v for k, v in mpp_by_floor.items()} if mpp_by_floor else None,
    }
    (entire_mixed_dir / "roof_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _comp_covered_by_upper_direct(
    comp_sections: List[Dict[str, Any]],
    upper_floor_sections: List[Dict[str, Any]],
    area_ratio_thresh: float = 0.5,
) -> bool:
    """Verificare directă bounding_rect fără polygonize."""
    if not upper_floor_sections or not comp_sections:
        return False
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.ops import unary_union
        upper_polys = [_rect_polygon(s) for s in upper_floor_sections]
        upper_polys = [p for p in upper_polys if p and not getattr(p, "is_empty", True)]
        if not upper_polys:
            return False
        upper_union = unary_union(upper_polys)
        for sec in comp_sections:
            sec_poly = _rect_polygon(sec)
            if sec_poly is None or getattr(sec_poly, "is_empty", True):
                continue
            inter = sec_poly.intersection(upper_union)
            sec_area = float(getattr(sec_poly, "area", 0) or 0)
            if sec_area < 1e-6:
                continue
            ratio = float(getattr(inter, "area", 0) or 0) / sec_area
            if ratio < area_ratio_thresh:
                return False
        return True
    except Exception:
        return False


def _section_or_comp_covered(
    sections_to_check: List[Dict[str, Any]],
    upper_rect_segs: List[List[List[float]]],
    upper_floor_sections: List[Dict[str, Any]],
    area_ratio_thresh: float = 0.5,
) -> bool:
    """
    Verificare robustă în două etape:
    1. Încearcă _section_covered_by_upper (polygonize-based) pentru fiecare secțiune
    2. Dacă eșuează (polygonize fail), fallback la _comp_covered_by_upper_direct (bbox-based)
    3. Dacă ambele eșuează, fallback la comparare bbox simplă
    Returnează True dacă TOATE secțiunile din lista dată sunt acoperite.
    """
    if not sections_to_check:
        return False
    if not upper_rect_segs and not upper_floor_sections:
        return False

    for sec in sections_to_check:
        if _section_covered_by_upper(sec, upper_rect_segs, area_ratio_thresh):
            continue
        if upper_floor_sections:
            if _comp_covered_by_upper_direct([sec], upper_floor_sections, area_ratio_thresh):
                continue
        try:
            br = sec.get("bounding_rect") or []
            if len(br) < 3:
                return False
            xs = [float(p[0]) for p in br]
            ys = [float(p[1]) for p in br]
            sec_minx, sec_maxx = min(xs), max(xs)
            sec_miny, sec_maxy = min(ys), max(ys)
            sec_area = (sec_maxx - sec_minx) * (sec_maxy - sec_miny)
            if sec_area < 1e-6:
                return False
            best_overlap = 0.0
            for usec in (upper_floor_sections or []):
                ubr = usec.get("bounding_rect") or []
                if len(ubr) < 3:
                    continue
                uxs = [float(p[0]) for p in ubr]
                uys = [float(p[1]) for p in ubr]
                u_minx, u_maxx = min(uxs), max(uxs)
                u_miny, u_maxy = min(uys), max(uys)
                inter_minx = max(sec_minx, u_minx)
                inter_maxx = min(sec_maxx, u_maxx)
                inter_miny = max(sec_miny, u_miny)
                inter_maxy = min(sec_maxy, u_maxy)
                if inter_maxx > inter_minx and inter_maxy > inter_miny:
                    inter_area = (inter_maxx - inter_minx) * (inter_maxy - inter_miny)
                    ratio = inter_area / sec_area
                    best_overlap = max(best_overlap, ratio)
            if best_overlap >= area_ratio_thresh:
                continue
        except Exception:
            pass
        return False
    return True


def _point_in_section_bbox(
    px: float, py: float, section: Dict[str, Any], tol: float = 1e-6
) -> bool:
    """True dacă punctul (px, py) este în interiorul bbox-ului secțiunii (bounding_rect)."""
    br = section.get("bounding_rect") or []
    if len(br) < 3:
        return False
    xs = [float(p[0]) for p in br]
    ys = [float(p[1]) for p in br]
    return min(xs) - tol <= px <= max(xs) + tol and min(ys) - tol <= py <= max(ys) + tol


def _filter_overhang_segments_not_on_covered_sections(
    segments: List[List[List[float]]],
    sections: List[Dict[str, Any]],
    upper_rect_segs: List[List[List[float]]],
    upper_floor_sections: Optional[List[Dict[str, Any]]],
) -> List[List[List[float]]]:
    """
    Păstrează doar segmentele al căror mijloc NU se află în secțiuni acoperite de etajul superior.
    Nu afișăm baza overhang și overhang pe dreptunghiurile care au pe ele etaj superior.
    """
    if not segments or not sections or not upper_rect_segs:
        return list(segments)
    covered = [
        s for s in sections
        if _section_or_comp_covered([s], upper_rect_segs, upper_floor_sections or [])
    ]
    if not covered:
        return list(segments)
    out = []
    for seg in segments:
        if len(seg) < 2:
            continue
        mx = (float(seg[0][0]) + float(seg[1][0])) / 2
        my = (float(seg[0][1]) + float(seg[1][1])) / 2
        if any(_point_in_section_bbox(mx, my, s) for s in covered):
            continue
        out.append(seg)
    return out


def generate_roof_type_outputs(
    wall_mask_path: str,
    roof_result: Dict[str, Any],
    sections: List[Dict[str, Any]],
    output_dir: Path,
    roof_angle_deg: float = 30.0,
    wall_height: float = 300.0,
    upper_floor_sections: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Generează lines.png și faces.png pentru 0_w, 1_w, 2_w, 4_w, 4.5_w.
    0_w = acoperiș plat (ca 1_w dar cu unghi 0). Conturul exterior = dreptunghiurile curente.
    Pe lines.png se adaugă și marginea la overhang (1 m) când mpp este disponibil din scale.
    """
    output_dir = Path(output_dir)
    roof_3d_path = output_dir.parent.parent
    roof_types_dir = output_dir.parent
    floor_dirs = [d for d in roof_types_dir.iterdir() if d.is_dir() and d.name.startswith("floor_") and d.name.split("_")[-1].isdigit()]
    num_floors = max(1, len(floor_dirs))
    mpp_by_floor = _load_meters_per_pixel_per_floor(roof_3d_path, num_floors)
    try:
        floor_idx = int(output_dir.name.split("_")[1]) if output_dir.name.startswith("floor_") and len(output_dir.name.split("_")) > 1 else 0
    except (ValueError, IndexError):
        floor_idx = 0
    mpp_this_floor = mpp_by_floor.get(floor_idx) if mpp_by_floor else None
    img = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    h, w = img.shape[:2]
    contour_segs = _get_contour_segments_from_sections(sections, (h, w))
    seg_separator = _get_separator_segments(sections)
    # Segmente care despart dreptunghiuri lipite (două fețe separate + linie de despărțire)
    contour_segs = list(contour_segs) + list(seg_separator)
    contour_segs_45w_green, contour_segs_45w_pink = _get_contour_segments_45w_chamfered(
        sections, (h, w), upper_floor_sections
    )
    upper_rect_segs = _get_upper_rect_segments(upper_floor_sections or [])

    roof_types_config = {
        "0_w": {
            "ridge": False,
            "magenta": False,
            "blue": True,
            "pyramid": False,
        },
        "1_w": {
            "ridge": False,
            "magenta": False,
            "blue": True,
            "pyramid": False,
        },
        "2_w": {
            "ridge": True,
            "magenta": _has_ridge_intersection(sections),
            "blue": False,
            "pyramid": False,
        },
        "4_w": {
            "ridge": True,
            "magenta": _has_ridge_intersection(sections),
            "blue": False,
            "pyramid": True,
        },
        "4.5_w": {
            "ridge": True,
            "magenta": _has_ridge_intersection(sections),
            "blue": False,
            "pyramid": True,
        },
    }

    ridge_segs = _get_ridge_segments(sections) if any(c["ridge"] for c in roof_types_config.values()) else []
    magenta_segs = _get_magenta_segments(sections) if any(c["magenta"] for c in roof_types_config.values()) else []
    blue_segs = _get_blue_segment_1w(sections)

    for roof_type, config in roof_types_config.items():
        subdir = output_dir / roof_type
        subdir.mkdir(parents=True, exist_ok=True)

        use_ridge_trimmed = roof_type in ("4.5_w", "4_w") and config["ridge"]
        seg_pyramid_short = (
            _get_pyramid_diagonal_segments(sections, upper_floor_sections, shorten_to_midpoint=True)
            if config["pyramid"] and use_ridge_trimmed
            else []
        )
        seg_pyramid = (
            _get_pyramid_diagonal_segments(sections, upper_floor_sections, shorten_to_midpoint=(roof_type == "4.5_w"))
            if config["pyramid"]
            else []
        )
        seg_orange = _get_orange_midpoint_segments_45w(sections, upper_floor_sections) if roof_type == "4.5_w" else []
        seg_ridge = (
            _get_ridge_segments_45w_trimmed(sections, upper_floor_sections, seg_pyramid_short, seg_orange)
            if use_ridge_trimmed and seg_pyramid_short
            else (ridge_segs if config["ridge"] else [])
        )
        seg_magenta = magenta_segs if config["magenta"] else []
        if roof_type == "4.5_w":
            seg_magenta = []  # 4.5_w: fără linii mov
        seg_blue = blue_segs if config["blue"] else []
        seg_contour = (contour_segs_45w_green if roof_type == "4.5_w" else contour_segs)
        seg_contour = list(seg_contour) + list(seg_separator)
        seg_pyramid_3d = list(seg_pyramid)  # pentru 3D: diagonale cu 2 capete
        brown_endpoint_markers: Optional[List[Tuple[Tuple[float, float], int]]] = None
        seg_brown: Optional[List[List[List[float]]]] = None
        seg_wall_support: List[List[List[float]]] = []
        if roof_type == "4.5_w":
            to_eliminate = _get_opposite_side_segments_to_eliminate_45w(sections, upper_floor_sections)
            seen_brown: set = set()
            brown_list: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
            original_corners: List[Tuple[float, float]] = []
            for s, c1, c2 in to_eliminate:
                if len(s) < 2:
                    continue
                k = (_segment_key(s[0], s[1], tol=5), (round(c1[0], 2), round(c1[1], 2)), (round(c2[0], 2), round(c2[1], 2)))
                if k not in seen_brown:
                    seen_brown.add(k)
                    brown_list.append((c1, c2))
                    original_corners.append((float(c1[0]), float(c1[1])))
                    original_corners.append((float(c2[0]), float(c2[1])))
            # Fără split – segmente maro doar de la colț la colț (latura opusă teșirii)
            # O singură pereche per segment identic (c1,c2) – elimină duplicate
            tol_seg = 1e-3
            seen_keys: set = set()
            brown_dedup: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
            for c1, c2 in brown_list:
                k1 = (round(c1[0] / tol_seg) * tol_seg, round(c1[1] / tol_seg) * tol_seg)
                k2 = (round(c2[0] / tol_seg) * tol_seg, round(c2[1] / tol_seg) * tol_seg)
                key = (k1, k2) if k1 <= k2 else (k2, k1)
                if key not in seen_keys:
                    seen_keys.add(key)
                    brown_dedup.append((c1, c2))
            brown_list = brown_dedup
            brown_list_with_ids = [(c1, c2, i) for i, (c1, c2) in enumerate(brown_list, 1)]
            # Buline strict la capetele originale (colțuri dreptunghi)
            def _is_original(pt: Tuple[float, float], tol: float = 0.5) -> bool:
                for oc in original_corners:
                    if (pt[0] - oc[0]) ** 2 + (pt[1] - oc[1]) ** 2 <= tol * tol:
                        return True
                return False

            def _has_markers_between(a: Tuple[float, float], b: Tuple[float, float], excl_seg_id: int) -> bool:
                """True dacă există capete ale altor segmente strict între a și b pe aceeași linie."""
                tol = 1e-6
                dx, dy = b[0] - a[0], b[1] - a[1]
                len_sq = dx * dx + dy * dy
                if len_sq < 1e-12:
                    return False
                for j, (oc1, oc2) in enumerate(brown_list):
                    if j + 1 == excl_seg_id:
                        continue
                    for p in (oc1, oc2):
                        cross = (p[0] - a[0]) * dy - (p[1] - a[1]) * dx
                        if abs(cross) > 0.5 * (len_sq ** 0.5 + 1):
                            continue
                        t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq
                        if tol < t < 1 - tol:
                            return True
                return False

            brown_list_for_clip: List[Tuple[Tuple[float, float], Tuple[float, float], int]] = []
            brown_endpoint_markers = []
            for pair_id, (c1, c2) in enumerate(brown_list, 1):
                if _has_markers_between(c1, c2, pair_id):
                    continue
                brown_list_for_clip.append((c1, c2, pair_id))
                if _is_original(c1):
                    brown_endpoint_markers.append((c1, pair_id))
                if _is_original(c2):
                    brown_endpoint_markers.append((c2, pair_id))

            def _process_segs_reconstruct(lst: List[List[List[float]]]) -> List[List[List[float]]]:
                """Reconstruiește segmentele verzi păstrând doar între puncte cu numere diferite."""
                out: List[List[List[float]]] = []
                for s in lst:
                    if len(s) < 2:
                        out.append(s)
                        continue
                    reconstructed = _reconstruct_green_from_different_number_points(
                        s, brown_endpoint_markers
                    )
                    out.extend(reconstructed)
                return out

            seg_contour = _process_segs_reconstruct(seg_contour)
            contour_pink_recon = _process_segs_reconstruct(contour_segs_45w_pink)
            # Pentru 3D: diagonale întregi (corner→ridge), un singur segment – nu schimbare de direcție la jumătate
            seg_pyramid_full = _get_pyramid_diagonal_segments(
                sections, upper_floor_sections, shorten_to_midpoint=False
            ) if config["pyramid"] else []
            seg_pyramid_3d = list(seg_pyramid_full)
            seg_pyramid = list(seg_pyramid) + contour_pink_recon
            upper_rect_segs = _process_segs_reconstruct(upper_rect_segs)
            seg_brown = None

        ridge_pink = (
            _get_ridge_diagonal_markers_45w(
                sections, upper_floor_sections,
                seg_pyramid if roof_type == "4.5_w" else seg_pyramid_short,
                seg_orange,
            )
            if roof_type in ("4.5_w", "4_w") and (seg_pyramid if roof_type == "4.5_w" else seg_pyramid_short)
            else None
        )
        if seg_ridge and upper_rect_segs:
            seg_wall_support = _get_wall_support_segments(sections, seg_ridge, upper_rect_segs, seg_contour)
        seg_overhang: Optional[List[List[List[float]]]] = None
        seg_for_overhang_draw: Optional[List[List[List[float]]]] = None
        # Overhang în lines.png: folosim mpp când există (1 m); altfel fallback în px ca linia să apară tot
        mpp_for_overhang = mpp_this_floor
        if mpp_for_overhang is None or mpp_for_overhang <= 0:
            default_overhang_px = max(20.0, 0.02 * min(h, w))
            mpp_for_overhang = 1.0 / default_overhang_px  # 1 m → default_overhang_px
        if mpp_for_overhang > 0:
            # Conturul verde e baza; celelalte segmente (diagonale, paralelă) le folosim doar ca să închidem
            # poligonul format din verde (ex.: L interior, teșituri 4.5_w). Nu înlocuim conturul.
            # De la început excludem zona etajului superior (nu tăiem ulterior).
            # Contur overhang: recreat de la zero din verde + diagonale + paralelă (paralela închide conturul, nu face overhang separat)
            offset_px = 1.0 / mpp_for_overhang
            seg_base_full_45w: List[List[List[float]]] = []
            contour_for_overhang_45w: List[List[List[float]]] = []
            seg_overhang_from_comps: Optional[List[List[List[float]]]] = None  # 4.5_w: nu mai folosim; overhang ca 4_w
            if roof_type == "4.5_w":
                # Overhang EXACT ca 4_w: același contur (verde+portocaliu+piramidă), apoi tăiere cu etajul, apoi _overhang_segments_from_contour.
                # Păstrăm doar baza pentru desen (seg_base_full_45w) cu regulile 4.5_w: diagonale + segmente de mijloc doar pe secțiunile neacoperite.
                seg_for_overhang = list(seg_contour) + list(seg_orange or []) + list(seg_pyramid)
                seg_base_full_45w = list(seg_for_overhang)
                if upper_rect_segs and offset_px > 0:
                    sections_not_covered_for_draw = [
                        s for s in sections
                        if not _section_or_comp_covered(
                            [s], upper_rect_segs, upper_floor_sections or [],
                        )
                    ]
                    if sections_not_covered_for_draw:
                        seg_sep_draw = _get_separator_segments(sections_not_covered_for_draw)
                        seg_45w_draw, _ = _get_contour_segments_45w_chamfered(
                            sections_not_covered_for_draw, (h, w), upper_floor_sections
                        )
                        seg_orange_draw = _get_orange_midpoint_segments_45w(
                            sections_not_covered_for_draw, upper_floor_sections
                        ) or []
                        seg_pyramid_draw = _get_pyramid_diagonal_segments(
                            sections_not_covered_for_draw, upper_floor_sections, shorten_to_midpoint=True
                        ) or []
                        seg_contour_draw = (list(seg_45w_draw) + list(seg_sep_draw)) if seg_45w_draw else []
                        seg_base_full_45w = list(seg_contour_draw) + list(seg_orange_draw) + list(seg_pyramid_draw)
                    else:
                        seg_base_full_45w = []
                contour_for_overhang_45w = []
            elif roof_type == "4_w":
                seg_for_overhang = list(seg_contour) + list(seg_pyramid)
            else:
                seg_for_overhang = seg_contour
            # 2_w: overhang 1 m per dreptunghi (fiecare secțiune în parte) → mai multe noduri pentru L
            if roof_type == "2_w" and sections and mpp_for_overhang and mpp_for_overhang > 0:
                seg_overhang = []
                seg_for_overhang_draw = []
                for sec in sections:
                    if upper_rect_segs and _section_or_comp_covered([sec], upper_rect_segs, upper_floor_sections or []):
                        continue
                    rect_segs = _section_rect_segments(sec)
                    if not rect_segs:
                        continue
                    seg_for_overhang_draw.extend(rect_segs)
                    oh = _overhang_segments_from_contour(
                        rect_segs, mpp_for_overhang, overhang_meters=1.0,
                    )
                    seg_overhang.extend(oh)
            else:
                # Excludem zona etajului superior: la toate tipurile (0_w, 1_w, 2_w, 4_w, 4.5_w) tăiem conturul cu etajul → nu generăm overhang pe dreptunghiurile galbene
                if upper_rect_segs and offset_px > 0:
                    seg_for_overhang = _contour_segments_minus_upper_rect(
                        seg_for_overhang, upper_rect_segs, offset_px
                    )
                if not seg_for_overhang:
                    seg_overhang = []
                elif seg_overhang_from_comps is not None:
                    seg_overhang = list(seg_overhang_from_comps)
                    # 4.5_w: baza pentru desen = doar secțiunile neacoperite (nu verde sub galben); fără etaj = full base
                    seg_for_overhang_draw = (
                        list(seg_base_full_45w) if roof_type == "4.5_w" else list(seg_for_overhang)
                    )
                else:
                    seg_for_overhang_draw = (
                        list(seg_base_full_45w) if roof_type == "4.5_w" else list(seg_for_overhang)
                    )
                    # Aceleași reguli ca 4_w: contour_only pentru filtrare SAU lanț explicit 4.5_w (verde+portocaliu) → un inel pe ambele baze
                    contour_only = (
                        contour_for_overhang_45w if (roof_type == "4.5_w" and contour_for_overhang_45w and len(contour_for_overhang_45w) >= 3)
                        else (None if (upper_rect_segs and offset_px > 0) else (
                            (list(seg_contour) + list(seg_orange or [])) if roof_type in ("4.5_w", "4_w") else None
                        ))
                    )
                    seg_overhang = _overhang_segments_from_contour(
                        seg_for_overhang, mpp_for_overhang, overhang_meters=1.0,
                        contour_only_segments=contour_only,
                    )
            if seg_overhang:
                # Doar 4_w și 4.5_w: merge overhang; 2_w păstrăm toate segmentele (toate colțurile pentru buline)
                if roof_type in ("4_w", "4.5_w"):
                    seg_overhang = _merge_overlapping_overhangs(seg_overhang)
            # Nu afișăm baza overhang și overhang pe dreptunghiurile acoperite de etajul superior
            if seg_overhang and upper_rect_segs and sections:
                seg_overhang = _filter_overhang_segments_not_on_covered_sections(
                    seg_overhang, sections, upper_rect_segs, upper_floor_sections
                )
            if seg_for_overhang_draw and upper_rect_segs and sections:
                seg_for_overhang_draw = _filter_overhang_segments_not_on_covered_sections(
                    seg_for_overhang_draw, sections, upper_rect_segs, upper_floor_sections
                )
            # 2_w: segmente overhang cu bulină albastră închis la mijloc → mov închis
            seg_overhang_ridge: Optional[List[List[List[float]]]] = None
            if roof_type == "2_w" and seg_overhang and seg_ridge:
                overhang_ridge_pts_2w = _compute_overhang_ridge_pts(
                    seg_ridge, seg_overhang, "2_w", tol=5.0,
                )
                seg_overhang_ridge = _overhang_segments_with_ridge_pt(
                    seg_overhang, overhang_ridge_pts_2w, tol=5.0,
                )
        _draw_lines_and_save(
            img,
            seg_ridge,
            seg_contour,
            seg_magenta,
            seg_blue,
            seg_pyramid,
            upper_rect_segs,
            roof_type,
            subdir / "lines.png",
            segments_orange=seg_orange,
            ridge_midpoints_from=ridge_segs if roof_type in ("4.5_w", "4_w") and use_ridge_trimmed else None,
            ridge_pink_points=ridge_pink,
            brown_endpoint_markers=brown_endpoint_markers,
            segments_brown=seg_brown,
            segments_wall_support=seg_wall_support,
            segments_overhang=seg_overhang,
            segments_overhang_inner=seg_for_overhang_draw,
            segments_overhang_ridge=seg_overhang_ridge if roof_type == "2_w" else None,
            sections=sections,
        )

        all_seg_lists = [seg_ridge, seg_contour, seg_magenta, seg_blue, seg_pyramid]
        if roof_type in ("0_w", "1_w"):
            all_seg_lists = [seg_ridge, seg_contour, seg_magenta, seg_pyramid]
        if seg_orange:
            all_seg_lists.append(seg_orange)
        all_seg_lists.append(upper_rect_segs)
        if seg_wall_support:
            all_seg_lists.append(seg_wall_support)
        # Nu adăugăm seg_overhang direct: fețele overhang = banda între contur verde (inner) și overhang (outer), ca să nu suprapunem cu fețele de acoperiș
        polygons_2d = _polygons_from_line_segments(
            *all_seg_lists,
            exclude_interior_of=upper_rect_segs if upper_rect_segs else None,
        )
        num_roof_polygons = len(polygons_2d)  # fețe acoperiș (fără banda overhang) – pentru unfold_roof vs unfold_overhang
        # Fețe overhang: zone între cyan (outer) și magenta (inner), împărțite de liniile gri punctat (anchor).
        # LOGICĂ: polygonize(outer_lines + inner_lines + anchor_lines) → poligoane cu centroid ÎN BANDĂ
        #         = în outer_poly ȘI NU în inner_poly. NU folosim band_geom pentru filtru (prea restrictiv).
        if seg_overhang:
            try:
                from shapely.geometry import LineString, Polygon as ShapelyPolygon
                from shapely.ops import polygonize_full, unary_union

                # ── Calculăm outer_poly (overhang) și inner_poly (baza overhang) ──────────────────
                overhang_lines = [
                    LineString([(float(s[0][0]), float(s[0][1])), (float(s[1][0]), float(s[1][1]))])
                    for s in seg_overhang if len(s) >= 2
                    and (float(s[0][0])-float(s[1][0]))**2+(float(s[0][1])-float(s[1][1]))**2 > 1e-6
                ]
                if not overhang_lines:
                    raise ValueError("no overhang lines")

                # outer_poly = poligonul overhang exterior (cyan)
                over_polys, _, _, _ = polygonize_full(overhang_lines)
                over_geoms = list(getattr(over_polys, "geoms", None) or [over_polys])
                outer_poly = max(
                    [g for g in over_geoms if g is not None and not getattr(g, "is_empty", True)],
                    key=lambda g: getattr(g, "area", 0) or 0,
                    default=None,
                )
                if outer_poly is None or getattr(outer_poly, "is_empty", True):
                    raise ValueError("outer_poly empty")

                # inner_poly = poligonul bazei overhang (magenta / seg_for_overhang_draw / seg_contour)
                inner_poly = None
                base_lines: list = []
                if seg_for_overhang_draw and len(seg_for_overhang_draw) >= 3:
                    base_lines = [
                        LineString([(float(s[0][0]), float(s[0][1])), (float(s[1][0]), float(s[1][1]))])
                        for s in seg_for_overhang_draw if len(s) >= 2
                        and (float(s[0][0])-float(s[1][0]))**2+(float(s[0][1])-float(s[1][1]))**2 > 1e-6
                    ]
                    if base_lines:
                        base_polys_b, _, _, _ = polygonize_full(base_lines)
                        base_geoms_b = list(getattr(base_polys_b, "geoms", None) or [base_polys_b])
                        valid_base = [g for g in base_geoms_b if g is not None and not getattr(g, "is_empty", True)]
                        if valid_base:
                            inner_poly = unary_union(valid_base)
                if inner_poly is None or getattr(inner_poly, "is_empty", True):
                    contour_lines = [
                        LineString([(float(s[0][0]), float(s[0][1])), (float(s[1][0]), float(s[1][1]))])
                        for s in seg_contour if len(s) >= 2
                        and (float(s[0][0])-float(s[1][0]))**2+(float(s[0][1])-float(s[1][1]))**2 > 1e-6
                    ]
                    if contour_lines:
                        cont_polys, _, _, _ = polygonize_full(contour_lines)
                        cont_geoms = list(getattr(cont_polys, "geoms", None) or [cont_polys])
                        valid_cont = [g for g in cont_geoms if g is not None and not getattr(g, "is_empty", True)]
                        if valid_cont:
                            inner_poly = unary_union(valid_cont)
                if inner_poly is None or getattr(inner_poly, "is_empty", True):
                    raise ValueError("inner_poly empty")

                # ── Calculăm anchor segments (gri punctat): bulina → anchor ─────────────────────
                ridge_int_pts_oh = _ridge_intersection_points(seg_ridge, tol=5.0) if seg_ridge else []
                _tol_oh = 5.0

                def _near_ridge_int(p: Tuple[float, float]) -> bool:
                    for r in ridge_int_pts_oh:
                        if (p[0] - r[0]) ** 2 + (p[1] - r[1]) ** 2 <= _tol_oh * _tol_oh:
                            return True
                    return False

                overhang_corner_pts_oh: List[Tuple[float, float]] = []
                overhang_ridge_pts_oh: List[Tuple[float, float]] = []
                for seg in seg_overhang:
                    if len(seg) >= 2:
                        overhang_corner_pts_oh.append((float(seg[0][0]), float(seg[0][1])))
                        overhang_corner_pts_oh.append((float(seg[1][0]), float(seg[1][1])))
                if seg_ridge and seg_overhang:
                    for rseg in seg_ridge:
                        if len(rseg) < 2:
                            continue
                        r0_oh = (float(rseg[0][0]), float(rseg[0][1]))
                        r1_oh = (float(rseg[1][0]), float(rseg[1][1]))
                        if _near_ridge_int(r0_oh) or _near_ridge_int(r1_oh):
                            continue
                        dx_oh = r1_oh[0] - r0_oh[0]
                        dy_oh = r1_oh[1] - r0_oh[1]
                        L_oh = (dx_oh * dx_oh + dy_oh * dy_oh) ** 0.5
                        if L_oh < 1e-12:
                            continue
                        half_oh = 0.1 * L_oh
                        e0_oh = (r0_oh[0] - half_oh * (dx_oh / L_oh), r0_oh[1] - half_oh * (dy_oh / L_oh))
                        e1_oh = (r1_oh[0] + half_oh * (dx_oh / L_oh), r1_oh[1] + half_oh * (dy_oh / L_oh))
                        for oseg in seg_overhang:
                            if len(oseg) < 2:
                                continue
                            o0_oh = (float(oseg[0][0]), float(oseg[0][1]))
                            o1_oh = (float(oseg[1][0]), float(oseg[1][1]))
                            ix, iy = _segment_intersection_2d(e0_oh, e1_oh, o0_oh, o1_oh)
                            if ix is not None and iy is not None:
                                # Bulina albastru închis STRICT pe laturi perpendiculare pe ridge
                                dx_o_oh = o1_oh[0] - o0_oh[0]
                                dy_o_oh = o1_oh[1] - o0_oh[1]
                                len_o_sq_oh = dx_o_oh * dx_o_oh + dy_o_oh * dy_o_oh
                                if len_o_sq_oh >= 1e-20:
                                    dot_oh = dx_oh * dx_o_oh + dy_oh * dy_o_oh
                                    if abs(dot_oh) > 0.17 * L_oh * (len_o_sq_oh ** 0.5):
                                        continue
                                overhang_ridge_pts_oh.append((ix, iy))
                overhang_corner_pts_oh = [p for p in overhang_corner_pts_oh if not _near_ridge_int(p)]
                overhang_ridge_pts_oh = [p for p in overhang_ridge_pts_oh if not _near_ridge_int(p)]
                # 2_w: și extensiile ridge = buline albastre, le includem la anchor
                if roof_type == "2_w" and overhang_ridge_pts_oh:
                    overhang_corner_pts_oh = list(overhang_corner_pts_oh) + list(overhang_ridge_pts_oh)
                anchor_segments = _overhang_anchor_segments(
                    overhang_corner_pts_oh, overhang_ridge_pts_oh,
                    seg_ridge, seg_contour, seg_overhang, seg_pyramid, roof_type,
                    sections=sections,
                )

                # ── Construim toate liniile pentru bandă și polygonize ───────────────────────────
                # outer (cyan) + inner (magenta/verde) + anchors (gri punctat)
                raw_band_lines = list(overhang_lines)
                if base_lines:
                    raw_band_lines.extend(base_lines)
                else:
                    # Fallback: linii din contur verde
                    raw_band_lines.extend([
                        LineString([(float(s[0][0]), float(s[0][1])), (float(s[1][0]), float(s[1][1]))])
                        for s in seg_contour if len(s) >= 2
                        and (float(s[0][0])-float(s[1][0]))**2+(float(s[0][1])-float(s[1][1]))**2 > 1e-6
                    ])
                for s in anchor_segments:
                    if len(s) >= 2 and (float(s[0][0])-float(s[1][0]))**2+(float(s[0][1])-float(s[1][1]))**2 > 1e-6:
                        raw_band_lines.append(LineString([
                            (float(s[0][0]), float(s[0][1])),
                            (float(s[1][0]), float(s[1][1])),
                        ]))

                # NODDING: unary_union împarte liniile la toate intersecțiile — fără asta polygonize nu creează celule mici
                try:
                    noded = unary_union(raw_band_lines)
                    noded_geoms = list(getattr(noded, "geoms", None) or [noded])
                    all_band_lines = []
                    for g in noded_geoms:
                        coords = list(getattr(g, "coords", []))
                        for i in range(len(coords) - 1):
                            seg = LineString([coords[i], coords[i + 1]])
                            if not seg.is_empty:
                                all_band_lines.append(seg)
                except Exception:
                    all_band_lines = raw_band_lines

                # ── Polygonize și filtrare: centroid ÎN outer ȘI NU ÎN inner ──────────────────
                band_polys, _, _, _ = polygonize_full(all_band_lines)
                band_geoms = list(getattr(band_polys, "geoms", None) or [band_polys])
                added_band_count = 0
                # Buffer mic pe inner: 0.2 în loc de 0.5 ca să nu excludem fețe la marginea benzii
                inner_poly_buf = inner_poly.buffer(0.2) if inner_poly is not None else None
                outer_poly_buf = outer_poly.buffer(1.0) if outer_poly is not None else None
                for g in band_geoms:
                    if g is None or getattr(g, "is_empty", True):
                        continue
                    try:
                        a_g = float(getattr(g, "area", 0) or 0)
                        if a_g < 0.5:
                            continue
                        cent = g.centroid
                        if cent is None or getattr(cent, "is_empty", True):
                            continue
                        # Filtru principal: centroid TREBUIE să fie în outer și NU în inner
                        in_outer = (
                            outer_poly_buf.contains(cent)
                            if outer_poly_buf is not None
                            else outer_poly.buffer(2).contains(cent)
                        )
                        if not in_outer:
                            continue
                        in_inner = (
                            inner_poly_buf.contains(cent)
                            if inner_poly_buf is not None
                            else inner_poly.contains(cent)
                        )
                        if in_inner:
                            continue
                        ext = getattr(g, "exterior", None)
                        if ext is None:
                            continue
                        coords = list(getattr(ext, "coords", []))
                        if len(coords) >= 4:
                            pts = [(float(c[0]), float(c[1])) for c in coords[:-1]]
                            polygons_2d.append(pts)
                            added_band_count += 1
                    except Exception:
                        pass

                # Fallback dacă polygonize nu a dat nimic: banda simplă (outer minus inner)
                if added_band_count == 0:
                    try:
                        band = outer_poly.difference(inner_poly.buffer(0))
                        if band is not None and not getattr(band, "is_empty", True):
                            for g in (getattr(band, "geoms", None) or [band]):
                                if g is None or getattr(g, "is_empty", True):
                                    continue
                                ext = getattr(g, "exterior", None)
                                if ext is not None:
                                    coords = list(getattr(ext, "coords", []))
                                    if len(coords) >= 4:
                                        pts = [(float(c[0]), float(c[1])) for c in coords[:-1]]
                                        polygons_2d.append(pts)
                    except Exception:
                        pass

            except Exception:
                # Fallback complet: adăugăm overhang la segmente și polygonize normal
                fallback_lists = list(all_seg_lists) + [seg_overhang]
                polygons_2d = _polygons_from_line_segments(
                    *fallback_lists,
                    exclude_interior_of=upper_rect_segs if upper_rect_segs else None,
                )
        effective_angle = 0.0 if roof_type == "0_w" else roof_angle_deg
        faces_data = _faces_from_segments(
            roof_type, sections, img,
            wall_height=wall_height, roof_angle_deg=effective_angle,
        )
        _draw_faces_png(img, polygons_2d, subdir / "faces.png", roof_type=roof_type, seed=hash(roof_type) % (2**32))

        try:
            from roof_calc.roof_segments_3d import (
                get_roof_segments_3d,
                _xy_to_z_map,
                _lookup_z,
                _round_segment_coords,
            )
            from roof_calc.flood_fill import flood_fill_interior, get_house_shape_mask
            from roof_calc.geometry import extract_polygon
            from roof_calc.overhang import ridge_intersection_corner_lines

            overhang_corner_list: List[List[float]] = []
            overhang_ridge_list: List[List[float]] = []
            filled = flood_fill_interior(img)
            house_mask = get_house_shape_mask(filled)
            floor_poly = extract_polygon(house_mask)
            if floor_poly is not None and not floor_poly.is_empty and polygons_2d:
                secs = _sections_for_1w_shed(sections) if roof_type in ("0_w", "1_w") else sections
                corner_lines = (
                    ridge_intersection_corner_lines(sections, per_section=False)
                    if roof_type not in ("4.5_w", "4_w") and _has_ridge_intersection(sections)
                    else None
                )
                segments_3d = get_roof_segments_3d(
                    secs,
                    floor_poly,
                    wall_height=wall_height,
                    roof_angle_deg=effective_angle,
                    corner_lines=corner_lines,
                    use_section_rect_eaves=False,
                )
                segs_rounded = _round_segment_coords(segments_3d, decimals=2)
                xy_to_z = _xy_to_z_map(segs_rounded, tol=5.0)
                segments_fallback = segs_rounded
                # Puncte overhang pentru 3D (ca în lines.png): colțuri + ridge prelungit (trebuie înainte de overhang_z_override)
                ridge_int_pts_3d = _ridge_intersection_points(seg_ridge, tol=5.0) if seg_ridge else []
                _tol_ridge = 5.0
                def _ridge_endpoint_at_intersection(r: Tuple[float, float]) -> bool:
                    for q in ridge_int_pts_3d:
                        if (r[0] - q[0]) ** 2 + (r[1] - q[1]) ** 2 <= _tol_ridge * _tol_ridge:
                            return True
                    return False
                overhang_corner_list: List[List[float]] = []
                overhang_ridge_list: List[List[float]] = []
                if seg_overhang:
                    for seg in seg_overhang:
                        if len(seg) >= 2:
                            overhang_corner_list.append([float(seg[0][0]), float(seg[0][1])])
                            overhang_corner_list.append([float(seg[1][0]), float(seg[1][1])])
                    if seg_ridge and seg_overhang:
                        for rseg in seg_ridge:
                            if len(rseg) < 2:
                                continue
                            r0 = (float(rseg[0][0]), float(rseg[0][1]))
                            r1 = (float(rseg[1][0]), float(rseg[1][1]))
                            if _ridge_endpoint_at_intersection(r0) or _ridge_endpoint_at_intersection(r1):
                                continue
                            dx = r1[0] - r0[0]
                            dy = r1[1] - r0[1]
                            L = (dx * dx + dy * dy) ** 0.5
                            if L < 1e-12:
                                continue
                            half = 0.1 * L
                            e0 = (r0[0] - half * (dx / L), r0[1] - half * (dy / L))
                            e1 = (r1[0] + half * (dx / L), r1[1] + half * (dy / L))
                            for oseg in seg_overhang:
                                if len(oseg) < 2:
                                    continue
                                o0 = (float(oseg[0][0]), float(oseg[0][1]))
                                o1 = (float(oseg[1][0]), float(oseg[1][1]))
                                ix, iy = _segment_intersection_2d(e0, e1, o0, o1)
                                if ix is not None and iy is not None:
                                    dx_o = o1[0] - o0[0]
                                    dy_o = o1[1] - o0[1]
                                    len_o_sq = dx_o * dx_o + dy_o * dy_o
                                    if len_o_sq >= 1e-20:
                                        dot = dx * dx_o + dy * dy_o
                                        if abs(dot) > 0.17 * L * (len_o_sq ** 0.5):
                                            continue
                                    overhang_ridge_list.append([ix, iy])
                if overhang_corner_list:
                    seen = set()
                    dedup = []
                    for p in overhang_corner_list:
                        k = (round(p[0], 2), round(p[1], 2))
                        if k not in seen:
                            seen.add(k)
                            dedup.append(p)
                    overhang_corner_list = dedup
                if overhang_ridge_list:
                    seen = set()
                    dedup = []
                    for p in overhang_ridge_list:
                        k = (round(p[0], 2), round(p[1], 2))
                        if k not in seen:
                            seen.add(k)
                            dedup.append(p)
                    overhang_ridge_list = dedup
                if ridge_int_pts_3d:
                    def _near(p: List[float]) -> bool:
                        if len(p) < 2:
                            return False
                        for r in ridge_int_pts_3d:
                            if (p[0] - r[0]) ** 2 + (p[1] - r[1]) ** 2 <= _tol_ridge * _tol_ridge:
                                return True
                        return False
                    overhang_corner_list = [p for p in overhang_corner_list if not _near(p)]
                    overhang_ridge_list = [p for p in overhang_ridge_list if not _near(p)]
                # Z pentru vârfurile fețelor overhang: albastru închis (ridge), albastru deschis (colțuri) = același Z ca bulinele (poate fi sub wh), gri (anchor) la wall_height
                ridge_z_val = max(float(p[2]) for s in segments_3d for p in s) if segments_3d else wall_height
                overhang_z_override: Dict[Tuple[float, float], float] = {}
                overhang_boundary_set = set()
                # Colțuri overhang (albastru deschis): Z ca în randarea 3D (buline) – poate fi sub nivel perete (pantă)
                tan_roof = 0.0 if roof_type == "0_w" else math.tan(math.radians(effective_angle))
                contour_corners_workflow: List[Tuple[float, float]] = []
                seen_cc_w = set()
                for seg in (seg_contour or []):
                    if len(seg) >= 2:
                        for pt in (seg[0], seg[1]):
                            q = (round(float(pt[0]), 2), round(float(pt[1]), 2))
                            if q not in seen_cc_w:
                                seen_cc_w.add(q)
                                contour_corners_workflow.append((float(pt[0]), float(pt[1])))
                pyramid_seg_tuples: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
                for seg in (seg_pyramid or []):
                    if len(seg) >= 2:
                        a = (float(seg[0][0]), float(seg[0][1]))
                        b = (float(seg[-1][0]), float(seg[-1][1]))
                        pyramid_seg_tuples.append((a, b))

                def _z_overhang_corner(ox: float, oy: float) -> float:
                    """Z pentru colț overhang (albastru deschis), același ca în frame: poate fi sub wh."""
                    if roof_type == "0_w":
                        return wall_height
                    if roof_type in ("4_w", "4.5_w") and pyramid_seg_tuples:
                        best_d2 = 1e30
                        best_z_anchor = wall_height
                        for (a, b) in pyramid_seg_tuples:
                            qx, qy, d2 = _closest_point_on_segment((ox, oy), a, b)
                            if d2 < best_d2:
                                best_d2 = d2
                                best_z_anchor = _lookup_z(qx, qy, xy_to_z, 5.0, segments_fallback)
                        d = math.sqrt(best_d2)
                        return max(0.0, best_z_anchor - d * tan_roof)
                    if contour_corners_workflow:
                        best_d2 = 1e30
                        for q in contour_corners_workflow:
                            d2 = (ox - q[0]) ** 2 + (oy - q[1]) ** 2
                            if d2 < best_d2:
                                best_d2 = d2
                        d = math.sqrt(best_d2)
                        return max(0.0, wall_height - d * tan_roof)
                    return wall_height

                for p in (overhang_corner_list or []):
                    k = (round(float(p[0]), 2), round(float(p[1]), 2))
                    overhang_boundary_set.add(k)
                    overhang_z_override[k] = _z_overhang_corner(float(p[0]), float(p[1]))
                for p in (overhang_ridge_list or []):
                    k = (round(float(p[0]), 2), round(float(p[1]), 2))
                    overhang_boundary_set.add(k)
                    overhang_z_override[k] = ridge_z_val
                anchor_segs_for_z = []
                if seg_overhang and seg_ridge and (overhang_corner_list or overhang_ridge_list):
                    try:
                        anchor_segs_for_z = _overhang_anchor_segments(
                            [(p[0], p[1]) for p in (overhang_corner_list or [])],
                            [(p[0], p[1]) for p in (overhang_ridge_list or [])],
                            seg_ridge, seg_contour, seg_overhang, seg_pyramid, roof_type,
                            sections=sections,
                        )
                        for s in (anchor_segs_for_z or []):
                            if len(s) < 2:
                                continue
                            for pt in (s[0], s[1]):
                                if len(pt) < 2:
                                    continue
                                k = (round(float(pt[0]), 2), round(float(pt[1]), 2))
                                if k not in overhang_boundary_set:
                                    overhang_z_override[k] = wall_height
                    except Exception:
                        anchor_segs_for_z = []

                def _z_for_vertex(px: float, py: float) -> float:
                    k = (round(px, 2), round(py, 2))
                    if k in overhang_z_override:
                        return overhang_z_override[k]
                    return _lookup_z(px, py, xy_to_z, 5.0, segments_fallback)

                # Puncte canonice = coordonate exacte din segmente și markere (ridge, contur, overhang, ancore)
                # pentru ca fețele 3D să folosească aceleași (x,y,z) ca liniile și bulinele
                canonical_xyz: List[Tuple[float, float, float]] = []
                for seg in segments_3d:
                    for p in seg:
                        canonical_xyz.append((float(p[0]), float(p[1]), float(p[2])))
                for p in (overhang_corner_list or []):
                    canonical_xyz.append((float(p[0]), float(p[1]), overhang_z_override.get((round(float(p[0]), 2), round(float(p[1]), 2)), wall_height)))
                for p in (overhang_ridge_list or []):
                    canonical_xyz.append((float(p[0]), float(p[1]), ridge_z_val))
                for s in (anchor_segs_for_z or []):
                    if len(s) >= 2:
                        for pt in (s[0], s[1]):
                            if len(pt) >= 2:
                                canonical_xyz.append((float(pt[0]), float(pt[1]), wall_height))

                _snap_tol_sq = 4.0  # 2 px

                def _snapped_vertex(px: float, py: float) -> List[float]:
                    """Returnează (x, y, z) snapping la punctul canonic cel mai apropiat, ca fețele să coincidă cu ridge/contur/overhang."""
                    px, py = float(px), float(py)
                    best_x, best_y, best_z = px, py, _z_for_vertex(px, py)
                    best_d2 = 1e30
                    for (cx, cy, cz) in canonical_xyz:
                        d2 = (px - cx) ** 2 + (py - cy) ** 2
                        if d2 <= _snap_tol_sq and d2 < best_d2:
                            best_d2 = d2
                            best_x, best_y, best_z = cx, cy, cz
                    return [best_x, best_y, best_z]

                faces_from_polygons_2d = [
                    {
                        "vertices_3d": [_snapped_vertex(float(p[0]), float(p[1])) for p in poly]
                    }
                    for poly in polygons_2d
                ]
            else:
                faces_from_polygons_2d = [{"vertices_3d": f.get("vertices_3d", [])} for f in faces_data]
            src_ridge = ridge_segs if roof_type in ("4.5_w", "4_w") and use_ridge_trimmed else seg_ridge
            ridge_midpoints = []
            for s in (src_ridge or []):
                if len(s) >= 2:
                    ridge_midpoints.append([
                        (float(s[0][0]) + float(s[1][0])) / 2.0,
                        (float(s[0][1]) + float(s[1][1])) / 2.0,
                    ])
            ridge_pink_list = [[float(p[0]), float(p[1])] for p in (ridge_pink or [])]
            brown_markers = [[[float(pt[0]), float(pt[1])], pid] for pt, pid in (brown_endpoint_markers or [])]
            # overhang_corner_list și overhang_ridge_list sunt deja calculate mai sus (în blocul if floor_poly)
            if not overhang_corner_list and not overhang_ridge_list and seg_overhang:
                overhang_corner_list = []
                overhang_ridge_list = []
                for seg in seg_overhang:
                    if len(seg) >= 2:
                        overhang_corner_list.append([float(seg[0][0]), float(seg[0][1])])
                        overhang_corner_list.append([float(seg[1][0]), float(seg[1][1])])
            def _seg_to_xy(seg: List[List[float]]) -> List[List[float]]:
                if len(seg) < 2:
                    return []
                return [[float(p[0]), float(p[1])] for p in seg]
            polygons_2d_serial = [[[float(p[0]), float(p[1])] for p in poly] for poly in polygons_2d]
            payload = {
                "floor_path": str(Path(wall_mask_path).resolve()),
                "faces": faces_from_polygons_2d,
                "num_roof_faces": num_roof_polygons,
                "polygons_2d": polygons_2d_serial,
                "wall_height": wall_height,
                "roof_angle_deg": effective_angle,
                "has_upper_floor": bool(upper_floor_sections),
                "segments": {
                    "ridge": [_seg_to_xy(seg) for seg in seg_ridge if len(seg) >= 2],
                    "contour": [_seg_to_xy(seg) for seg in seg_contour if len(seg) >= 2],
                    "magenta": [_seg_to_xy(seg) for seg in seg_magenta if len(seg) >= 2],
                    "blue": [_seg_to_xy(seg) for seg in seg_blue if len(seg) >= 2],
                    "pyramid": [_seg_to_xy(seg) for seg in seg_pyramid_3d if len(seg) >= 2],
                    "upper_rect": [_seg_to_xy(seg) for seg in upper_rect_segs if len(seg) >= 2],
                    "orange": [_seg_to_xy(seg) for seg in (seg_orange or []) if len(seg) >= 2],
                    "brown": [_seg_to_xy(seg) for seg in (seg_brown or []) if len(seg) >= 2],
                    "wall_support": [_seg_to_xy(seg) for seg in (seg_wall_support or []) if len(seg) >= 2],
                    "overhang": [_seg_to_xy(seg) for seg in (seg_overhang or []) if len(seg) >= 2],
                },
                "markers": {
                    "ridge_midpoints": ridge_midpoints,
                    "ridge_pink": ridge_pink_list,
                    "brown_endpoints": brown_markers,
                    "overhang_corners": overhang_corner_list,
                    "overhang_ridge_pts": overhang_ridge_list,
                },
            }
            (subdir / "faces_faces.json").write_text(json.dumps(payload, indent=0), encoding="utf-8")
        except Exception:
            pass
        try:
            _generate_frame_html(subdir, wall_height=wall_height)
        except Exception:
            pass

