"""
Workflow curat: rectangles (cu eliminare suprapuneri) + roof_types (1_w, 2_w, 4_w, 4.5_w).
Fiecare tip: lines.png + faces.png.
"""

from __future__ import annotations

import json
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


def _rect_polygon(sec: Dict[str, Any]):
    from shapely.geometry import Polygon as ShapelyPolygon

    br = sec.get("bounding_rect") or []
    if len(br) < 3:
        return None
    try:
        return ShapelyPolygon([(float(p[0]), float(p[1])) for p in br])
    except Exception:
        return None


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
    shorten_to_midpoint: pentru 4.5_w, desenăm doar jumătatea de la mijlocul diagonalei la ridge.
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
        clamp_to_mid = len(attached_sides) == 0
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
                # Clasificare după secțiune (stânga/dreapta); ridge_center asigură că nu trecem de jumătate
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
                # Nu traversăm ridge-uri verticale
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
                # Clasificare după secțiune (sus/jos); când ridge-ul iese din dreptunghi, centrul poate fi în afara secțiunii
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
                # Nu traversăm ridge-uri orizontale
                if first_down_y is not None and cy <= first_down_y:
                    end_y = min(end_y, first_down_y)
                if first_up_y is not None and cy >= first_up_y:
                    end_y = max(end_y, first_up_y)

            if shorten_to_midpoint:
                mx = (cx + end_x) / 2.0
                my = (cy + end_y) / 2.0
                segs.append([[mx, my], [end_x, end_y]])
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

        if orient == "horizontal":
            # Stânga: eliminăm latura verticală (minx,*), conectăm colțuri la paralelă
            if re0 is not None and re3 is not None:
                m_left_top = midpoint(0, re0)
                m_left_bot = midpoint(3, re3)
                _add_seg([minx, miny], [m_left_top[0], m_left_top[1]], pink_diag=True)
                _add_seg([m_left_top[0], m_left_top[1]], [m_left_bot[0], m_left_bot[1]], pink_mid=True)
                _add_seg([m_left_bot[0], m_left_bot[1]], [minx, maxy], pink_diag=True)
            else:
                _add_seg([minx, miny], [minx, maxy], allow_skip=True)

            _add_seg([minx, maxy], [maxx, maxy])

            if re1 is not None and re2 is not None:
                m_right_top = midpoint(1, re1)
                m_right_bot = midpoint(2, re2)
                _add_seg([maxx, maxy], [m_right_bot[0], m_right_bot[1]], pink_diag=True)
                _add_seg([m_right_bot[0], m_right_bot[1]], [m_right_top[0], m_right_top[1]], pink_mid=True)
                _add_seg([m_right_top[0], m_right_top[1]], [maxx, miny], pink_diag=True)
            else:
                _add_seg([maxx, maxy], [maxx, miny], allow_skip=True)

            _add_seg([maxx, miny], [minx, miny])
        else:
            if re0 is not None and re1 is not None:
                m_top_left = midpoint(0, re0)
                m_top_right = midpoint(1, re1)
                _add_seg([minx, miny], [m_top_left[0], m_top_left[1]], pink_diag=True)
                _add_seg([m_top_left[0], m_top_left[1]], [m_top_right[0], m_top_right[1]], pink_mid=True)
                _add_seg([m_top_right[0], m_top_right[1]], [maxx, miny], pink_diag=True)
            else:
                _add_seg([minx, miny], [maxx, miny], allow_skip=True)

            _add_seg([maxx, miny], [maxx, maxy])

            if re2 is not None and re3 is not None:
                m_bot_right = midpoint(2, re2)
                m_bot_left = midpoint(3, re3)
                _add_seg([maxx, maxy], [m_bot_right[0], m_bot_right[1]], pink_diag=True)
                _add_seg([m_bot_right[0], m_bot_right[1]], [m_bot_left[0], m_bot_left[1]], pink_mid=True)
                _add_seg([m_bot_left[0], m_bot_left[1]], [minx, maxy], pink_diag=True)
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
) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        return False

    h, w = wall_mask.shape[:2]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f"Linii acoperiș {roof_type}")
    ax.set_aspect("equal")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.imshow(wall_mask, cmap="gray", vmin=0, vmax=255)

    _alpha = 0.5
    for seg in segments_contour:
        if len(seg) >= 2:
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color="green", linewidth=2, alpha=_alpha)

    if segments_brown:
        for seg in segments_brown:
            if len(seg) >= 2:
                ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color="saddlebrown", linewidth=2.5, alpha=_alpha, zorder=8)

    for seg in segments_ridge:
        if len(seg) >= 2:
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color="darkred", linewidth=2.5, alpha=_alpha)

    for seg in segments_magenta:
        if len(seg) >= 2:
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color="#CC00FF", linewidth=2, alpha=_alpha, zorder=10)

    for seg in segments_blue:
        if len(seg) >= 2:
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color="blue", linewidth=2.5, alpha=_alpha)

    for seg in segments_pyramid:
        if len(seg) >= 2:
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color="blue", linewidth=1.5, linestyle="--", alpha=_alpha)

    for seg in segments_upper_rect:
        if len(seg) >= 2:
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color="yellow", linewidth=2, alpha=_alpha, zorder=5)

    if segments_orange:
        for seg in segments_orange:
            if len(seg) >= 2:
                ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color="orange", linewidth=2.5, linestyle="-", alpha=_alpha, zorder=6)

    if segments_wall_support:
        for seg in segments_wall_support:
            if len(seg) >= 2:
                ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color="deepskyblue", linewidth=2, alpha=_alpha, zorder=6)

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
    handles.append(Line2D([0], [0], color="green", lw=2, alpha=_alpha, label="Contur exterior"))
    if segments_ridge:
        handles.append(Line2D([0], [0], color="darkred", lw=2.5, alpha=_alpha, label="Ridge"))
    if segments_magenta:
        handles.append(Line2D([0], [0], color="#CC00FF", lw=2, alpha=_alpha, label="Linii intersecție → colțuri"))
    if segments_blue and not segments_pyramid:
        handles.append(Line2D([0], [0], color="blue", lw=2.5, alpha=_alpha, label="Segment albastru (shed)"))
    elif segments_pyramid:
        handles.append(Line2D([0], [0], color="blue", lw=1.5, ls="--", alpha=_alpha, label="Diagonale 45° (piramidă)"))
    if segments_upper_rect:
        handles.append(Line2D([0], [0], color="yellow", lw=2, alpha=_alpha, label="Etaj superior"))
    if segments_orange:
        handles.append(Line2D([0], [0], color="orange", lw=2.5, ls="-", alpha=_alpha, label="Paralelă între diagonale (4.5_w)"))
    if segments_brown:
        handles.append(Line2D([0], [0], color="saddlebrown", lw=2.5, alpha=_alpha, label="Laturi opuse teșirii"))
    if segments_wall_support:
        handles.append(Line2D([0], [0], color="deepskyblue", lw=2, alpha=_alpha, label="Suport perete"))
    if handles:
        ax.legend(handles=handles, loc="upper right")

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

    secs = _sections_for_1w_shed(sections) if roof_type == "1_w" else sections
    corner_lines = None
    if roof_type != "1_w" and _has_ridge_intersection(sections):
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
    if roof_type_frame == "1_w" and has_upper_floor and upper_rect_pts and contour_segs_list:
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
        if not upper_rect_pts or roof_type_frame != "1_w" or not has_upper_floor or best_contour_d2 is None:
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
        if roof_type_frame == "1_w" and has_upper_floor and _point_on_contour_near_yellow(x, y):
            return wh + 0.5 * wh
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
        ("upper_rect", "yellow", "Etaj superior", 2, None, z_fn_orange, False),
        ("wall_support", "deepskyblue", "Suport perete", 2, None, z_fn_between, False),
        ("brown", "saddlebrown", "Laturi opuse teșirii", 2.5, None, z_fn_between, False),
    ]
    for key, color, name, width, dash, z_fn, endpoints_only in segment_config:
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

    # Contur pereți
    wall_segs_2d = []
    if floor_path:
        mask = cv2.imread(floor_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            wall_segs_2d = _get_contour_segments(mask)
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
    for name in ("1_w", "2_w", "4_w", "4.5_w"):
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
            if not rt or rt not in ("1_w", "2_w", "4_w", "4.5_w"):
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
        if rt_floor == "1_w" and has_upper and upper_rect_pts_entire and contour_segs_entire:
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
            if not upper_rect_pts_entire or rt_floor != "1_w" or not has_upper or best_contour_d2_entire is None:
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
            if rt_floor == "1_w" and has_upper and _point_on_contour_near_yellow_entire(x, y):
                return fwh + 0.5 * fwh + z_off
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
        seg_cfg = [
            ("contour", "green", "Contur exterior", 2, None, _z_fn_contour, False),
            ("ridge", "darkred", "Ridge", 2.5, None, _z_fn_ridge, False),
            ("magenta", "#CC00FF", "Linii intersecție", 2, None, _z_fn_magenta, False),
            ("blue", "blue", "Segment albastru", 2.5, None, _z_fn_between, False),
            ("pyramid", "blue", "Diagonale 45°", 1.5, "dash", _z_fn_pyramid_use, True),
            ("orange", "orange", "Paralelă diagonale", 2, None, _z_fn_orange, False),
            ("upper_rect", "yellow", "Etaj superior", 2, None, _z_fn_orange, False),
            ("wall_support", "deepskyblue", "Suport perete", 2, None, _z_fn_between, False),
            ("brown", "saddlebrown", "Laturi opuse", 2.5, None, _z_fn_between, False),
        ]
        for key, color, name, width, dash, z_fn, ep in seg_cfg:
            segs = segs_data.get(key) or []
            lx, ly, lz = _seg_to_plotly(segs, z_fn, endpoints_only=ep)
            _add_trace(lx, ly, lz, color, name, width=width, dash=dash)
            if key == "upper_rect" and segs and contour_pts_for_z:
                seen_upper_pt = set()
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
                        lx_v = [x + dx_off, x + dx_off, None]
                        ly_v = [y + dy_off, y + dy_off, None]
                        lz_v = [fwh + z_off, z_top, None]
                        lbl = "Etaj superior (pereți)" if len(floor_payloads) <= 1 else f"Etaj superior (pereți) (etaj {floor_idx})"
                        fig.add_trace(go.Scatter3d(x=lx_v, y=ly_v, z=lz_v, mode="lines",
                            line=dict(color="yellow", width=1), name=lbl, legendgroup=lbl))

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

        wall_segs_2d = []
        if floor_path:
            mask = cv2.imread(floor_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
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
        if rt_floor == "1_w" and has_upper_filled and upper_rect_pts_filled and contour_segs_filled:
            best_contour_d2_filled = 1e30
            for cseg in contour_segs_filled:
                pts = cseg if isinstance(cseg, (list, tuple)) and cseg and isinstance(cseg[0], (list, tuple)) else []
                if len(pts) >= 2:
                    a, b = (float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1]))
                    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
                    best_contour_d2_filled = min(best_contour_d2_filled, _min_d2(mid, upper_rect_pts_filled))
            best_contour_d2_filled = (best_contour_d2_filled or 0) + 100

        def _point_on_contour_near_yellow_filled(px: float, py: float, tol: float = 5.0) -> bool:
            if not upper_rect_pts_filled or rt_floor != "1_w" or not has_upper_filled or best_contour_d2_filled is None:
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
            if rt_floor == "1_w" and has_upper_filled and _point_on_contour_near_yellow_filled(x, y):
                return fwh + 0.5 * fwh + z_off
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
        if not polygons_2d:
            for f in (payload.get("faces") or []):
                vs = f.get("vertices_3d") or []
                if len(vs) >= 3:
                    polygons_2d.append([[float(v[0]), float(v[1])] for v in vs])
        faces_for_unfold: List[Dict[str, Any]] = []
        poly_idx = 0
        for poly in polygons_2d:
            if not poly or len(poly) < 3:
                continue
            poly_idx += 1
            xs = [float(p[0]) + dx_off for p in poly]
            ys = [float(p[1]) + dy_off for p in poly]
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
                raised_z = fwh + 0.5 * fwh + z_off
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
                unfold_dir = roof_types_dir / f"floor_{floor_idx}" / rt_floor / "unfold"
                try:
                    generate_unfold_masks_for_roof_types(faces_for_unfold, plan_h, plan_w, unfold_dir)
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
    Pentru entire/mixed/: creează unfold/ cu toate măștile roof și roof_metrics.json.
    roof_metrics.json: area/contour în pixeli și m² per față, per etaj, total.
    """
    entire_mixed_dir = Path(entire_mixed_dir)
    roof_types_dir = Path(roof_types_dir)
    unfold_dir = entire_mixed_dir / "unfold"
    unfold_dir.mkdir(parents=True, exist_ok=True)

    floor_dirs = sorted(
        [d for d in roof_types_dir.iterdir() if d.is_dir() and d.name.startswith("floor_")],
        key=lambda d: int(d.name.split("_")[1]) if len(d.name.split("_")) > 1 and d.name.split("_")[1].isdigit() else 0,
    )

    faces_metrics: List[Dict[str, Any]] = []
    floor_totals: Dict[int, Dict[str, Any]] = {}
    total_area_px = 0.0
    total_contour_px = 0.0

    mpp_by_floor = _load_meters_per_pixel_per_floor(output_path, len(floor_dirs) + 1)

    for floor_dir in floor_dirs:
        fidx = int(floor_dir.name.split("_")[1]) if len(floor_dir.name.split("_")) > 1 and floor_dir.name.split("_")[1].isdigit() else 0
        rt = floor_roof_types.get(fidx, "2_w")
        unfold_src = floor_dir / rt / "unfold"
        if not unfold_src.is_dir():
            continue

        mpp = mpp_by_floor.get(fidx)
        floor_area_px = 0.0
        floor_contour_px = 0.0
        floor_area_m2: Optional[float] = None
        floor_contour_m: Optional[float] = None

        for png in sorted(unfold_src.glob("*.png")):
            face_num = png.stem
            if not face_num.isdigit():
                continue
            dst_name = f"floor_{fidx}_{face_num}.png"
            dst_path = unfold_dir / dst_name
            try:
                shutil.copy2(png, dst_path)
            except Exception:
                continue

            mask = cv2.imread(str(dst_path), cv2.IMREAD_GRAYSCALE)
            area_px, contour_px = _mask_area_and_contour_px(mask)
            floor_area_px += area_px
            floor_contour_px += contour_px
            total_area_px += area_px
            total_contour_px += contour_px

            area_m2: Optional[float] = None
            contour_m: Optional[float] = None
            if mpp is not None and mpp > 0:
                area_m2 = area_px * (mpp ** 2)
                contour_m = contour_px * mpp

            faces_metrics.append({
                "floor_idx": fidx,
                "face_id": int(face_num),
                "filename": dst_name,
                "area_px": round(area_px, 2),
                "area_m2": round(area_m2, 6) if area_m2 is not None else None,
                "contour_px": round(contour_px, 2),
                "contour_m": round(contour_m, 4) if contour_m is not None else None,
            })

        if mpp is not None and mpp > 0:
            floor_area_m2 = floor_area_px * (mpp ** 2)
            floor_contour_m = floor_contour_px * mpp

        floor_totals[fidx] = {
            "area_px": round(floor_area_px, 2),
            "area_m2": round(floor_area_m2, 6) if floor_area_m2 is not None else None,
            "contour_px": round(floor_contour_px, 2),
            "contour_m": round(floor_contour_m, 4) if floor_contour_m is not None else None,
        }

    total_area_m2: Optional[float] = None
    total_contour_m: Optional[float] = None
    if mpp_by_floor:
        avg_mpp = sum(mpp_by_floor.values()) / len(mpp_by_floor)
        if avg_mpp > 0:
            total_area_m2 = total_area_px * (avg_mpp ** 2)
            total_contour_m = total_contour_px * avg_mpp

    metrics = {
        "faces": faces_metrics,
        "by_floor": floor_totals,
        "total": {
            "area_px": round(total_area_px, 2),
            "area_m2": round(total_area_m2, 6) if total_area_m2 is not None else None,
            "contour_px": round(total_contour_px, 2),
            "contour_m": round(total_contour_m, 4) if total_contour_m is not None else None,
        },
        "meters_per_pixel_by_floor": {str(k): v for k, v in mpp_by_floor.items()} if mpp_by_floor else None,
    }
    (entire_mixed_dir / "roof_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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
    Generează lines.png și faces.png pentru 1_w, 2_w, 4_w, 4.5_w.
    Conturul exterior = dreptunghiurile curente. Etajul superior = galben (dacă e dat).
    """
    img = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    h, w = img.shape[:2]
    contour_segs = _get_contour_segments_from_sections(sections, (h, w))
    contour_segs_45w_green, contour_segs_45w_pink = _get_contour_segments_45w_chamfered(
        sections, (h, w), upper_floor_sections
    )
    upper_rect_segs = _get_upper_rect_segments(upper_floor_sections or [])

    roof_types_config = {
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
        seg_blue = blue_segs if config["blue"] else []
        seg_contour = contour_segs_45w_green if roof_type == "4.5_w" else contour_segs
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
        )

        all_seg_lists = [seg_ridge, seg_contour, seg_magenta, seg_blue, seg_pyramid]
        if roof_type == "1_w":
            all_seg_lists = [seg_ridge, seg_contour, seg_magenta, seg_pyramid]
        if seg_orange:
            all_seg_lists.append(seg_orange)
        all_seg_lists.append(upper_rect_segs)
        if seg_wall_support:
            all_seg_lists.append(seg_wall_support)

        polygons_2d = _polygons_from_line_segments(
            *all_seg_lists,
            exclude_interior_of=upper_rect_segs if upper_rect_segs else None,
        )
        faces_data = _faces_from_segments(
            roof_type, sections, img,
            wall_height=wall_height, roof_angle_deg=roof_angle_deg,
        )
        _draw_faces_png(img, polygons_2d, subdir / "faces.png", roof_type=roof_type, seed=hash(roof_type) % (2**32))

        try:
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
            def _seg_to_xy(seg: List[List[float]]) -> List[List[float]]:
                if len(seg) < 2:
                    return []
                return [[float(p[0]), float(p[1])] for p in seg]
            polygons_2d_serial = [[[float(p[0]), float(p[1])] for p in poly] for poly in polygons_2d]
            payload = {
                "floor_path": str(Path(wall_mask_path).resolve()),
                "faces": [{"vertices_3d": f.get("vertices_3d", [])} for f in faces_data],
                "polygons_2d": polygons_2d_serial,
                "wall_height": wall_height,
                "roof_angle_deg": roof_angle_deg,
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
                },
                "markers": {
                    "ridge_midpoints": ridge_midpoints,
                    "ridge_pink": ridge_pink_list,
                    "brown_endpoints": brown_markers,
                },
            }
            (subdir / "faces_faces.json").write_text(json.dumps(payload, indent=0), encoding="utf-8")
        except Exception:
            pass
        try:
            _generate_frame_html(subdir, wall_height=wall_height)
        except Exception:
            pass

