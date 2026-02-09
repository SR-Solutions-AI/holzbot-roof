from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class RectBounds:
    minx: float
    miny: float
    maxx: float
    maxy: float

    @property
    def w(self) -> float:
        return float(self.maxx - self.minx)

    @property
    def h(self) -> float:
        return float(self.maxy - self.miny)

    @property
    def diag(self) -> float:
        return sqrt(self.w * self.w + self.h * self.h)


def _strip_xy(p: Any) -> Tuple[float, float]:
    return (float(p[0]), float(p[1]))


def rect_bounds_from_bounding_rect(bounding_rect: List[Any]) -> Optional[RectBounds]:
    if not bounding_rect or len(bounding_rect) < 3:
        return None
    xs = [float(_strip_xy(p)[0]) for p in bounding_rect]
    ys = [float(_strip_xy(p)[1]) for p in bounding_rect]
    return RectBounds(min(xs), min(ys), max(xs), max(ys))


def compute_overhang_px_from_roof_results(
    roof_results: Iterable[Dict[str, Any]],
    *,
    ratio: float = 0.10,
) -> float:
    """
    Returns a single overhang distance (in pixels) computed as:
      ratio * diagonal_of_largest_section_rectangle
    where "largest" is by area (w*h) across all provided roof_results.
    """
    best_diag = 0.0
    best_area = -1.0
    for rr in roof_results or []:
        for sec in (rr.get("sections") or []):
            bb = rect_bounds_from_bounding_rect(sec.get("bounding_rect") or [])
            if bb is None:
                continue
            area = bb.w * bb.h
            if area > best_area:
                best_area = area
                best_diag = bb.diag
    if best_diag <= 0:
        return 0.0
    return float(best_diag) * float(ratio)


def compute_free_sides_axis_aligned(
    sections: List[Dict[str, Any]],
    *,
    tol: float = 1e-6,
) -> List[Dict[str, bool]]:
    """
    For each section (assumed axis-aligned rectangle), returns a dict:
      {"left": bool, "right": bool, "top": bool, "bottom": bool}
    marking which sides are free (NOT attached to another section along an edge).
    Attachment requires a near-equal coordinate and a positive overlap along the orthogonal axis.
    """
    bounds: List[Optional[RectBounds]] = [rect_bounds_from_bounding_rect(s.get("bounding_rect") or []) for s in sections]
    attached = [{"left": False, "right": False, "top": False, "bottom": False} for _ in sections]

    def overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
        lo = max(min(a0, a1), min(b0, b1))
        hi = min(max(a0, a1), max(b0, b1))
        return float(hi - lo)

    n = len(sections)
    for i in range(n):
        bi = bounds[i]
        if bi is None:
            continue
        for j in range(i + 1, n):
            bj = bounds[j]
            if bj is None:
                continue

            # i right touches j left
            if abs(bi.maxx - bj.minx) <= tol:
                if overlap_1d(bi.miny, bi.maxy, bj.miny, bj.maxy) > tol:
                    attached[i]["right"] = True
                    attached[j]["left"] = True
            # i left touches j right
            if abs(bi.minx - bj.maxx) <= tol:
                if overlap_1d(bi.miny, bi.maxy, bj.miny, bj.maxy) > tol:
                    attached[i]["left"] = True
                    attached[j]["right"] = True
            # i bottom touches j top
            if abs(bi.maxy - bj.miny) <= tol:
                if overlap_1d(bi.minx, bi.maxx, bj.minx, bj.maxx) > tol:
                    attached[i]["bottom"] = True
                    attached[j]["top"] = True
            # i top touches j bottom
            if abs(bi.miny - bj.maxy) <= tol:
                if overlap_1d(bi.minx, bi.maxx, bj.minx, bj.maxx) > tol:
                    attached[i]["top"] = True
                    attached[j]["bottom"] = True

    free = []
    for a in attached:
        free.append({k: (not v) for k, v in a.items()})
    return free


def compute_overhang_sides_from_free_ends(
    sections: List[Dict[str, Any]],
    free_ends: Optional[List[Dict[str, bool]]],
) -> List[Dict[str, bool]]:
    """
    Builds per-section side masks for overhang.

    Design choice (matches typical roof behavior and user's expectation of "even" extension):
    - Eaves (perpendicular to ridge) ALWAYS get overhang on both sides.
    - Ridge-direction ends get overhang only if marked free in `free_ends`.

    Returns list of {"left","right","top","bottom"} booleans.
    """
    out: List[Dict[str, bool]] = []
    for i, sec in enumerate(sections):
        orient = sec.get("ridge_orientation", "horizontal")
        fe = (free_ends[i] if (free_ends and i < len(free_ends)) else {}) or {}

        if orient == "vertical":
            # Ridge runs along Y; eaves are left/right (X) -> always
            left = True
            right = True
            top = bool(fe.get("top", True))
            bottom = bool(fe.get("bottom", True))
        else:
            # Ridge runs along X; eaves are top/bottom (Y) -> always
            top = True
            bottom = True
            left = bool(fe.get("left", True))
            right = bool(fe.get("right", True))

        out.append({"left": left, "right": right, "top": top, "bottom": bottom})
    return out


def compute_overhang_sides_from_connections(
    sections: List[Dict[str, Any]],
    connections: List[Dict[str, Any]],
) -> List[Dict[str, bool]]:
    """
    Uses roof connections (not pure bbox adjacency) to decide which ridge-ends are free.
    This avoids the "only extended on one side" issue when rectangles touch only partially.
    """
    try:
        from roof_calc.visualize import _free_roof_ends  # type: ignore
    except Exception:
        # Fallback to naive adjacency if visualize shim not available
        return compute_free_sides_axis_aligned(sections)

    free_ends = _free_roof_ends(sections, connections)  # type: ignore[misc]
    return compute_overhang_sides_from_free_ends(sections, free_ends)


def compute_overhang_sides_from_union_boundary(
    sections: List[Dict[str, Any]],
    *,
    tol: float = 1.0,
) -> List[Dict[str, bool]]:
    """
    Robust side detection for partial attachments:
    A side is considered "free" if any portion of that side lies on the OUTER boundary
    of the union of all section rectangles.

    This avoids the case where rectangles touch only on part of a side (and we'd otherwise
    incorrectly mark the whole side as attached and skip overhang).
    """
    try:
        from shapely.geometry import Point, Polygon as ShapelyPolygon
        from shapely.ops import unary_union
    except Exception:
        # If Shapely isn't available, fallback to naive adjacency
        return compute_free_sides_axis_aligned(sections, tol=tol)

    polys = []
    rects: List[Optional[RectBounds]] = []
    for sec in sections:
        bb = rect_bounds_from_bounding_rect(sec.get("bounding_rect") or [])
        rects.append(bb)
        if bb is None:
            polys.append(None)
            continue
        try:
            polys.append(ShapelyPolygon([(bb.minx, bb.miny), (bb.maxx, bb.miny), (bb.maxx, bb.maxy), (bb.minx, bb.maxy)]))
        except Exception:
            polys.append(None)

    valid_polys = [p for p in polys if p is not None]
    if not valid_polys:
        return compute_free_sides_axis_aligned(sections, tol=tol)

    u = unary_union(valid_polys)

    out: List[Dict[str, bool]] = []
    for bb in rects:
        if bb is None:
            out.append({"left": True, "right": True, "top": True, "bottom": True})
            continue
        # Robust "is this side on the OUTER boundary?" test:
        # sample multiple points along the side and compare a point slightly "inward"
        # vs "outward" relative to the union polygon.
        # If inward is inside union and outward is outside union at ANY sample -> side is free.
        #
        # This avoids brittle exact boundary intersection and fixes "cut/overflow-hidden" areas.

        # (seg endpoints), outward normal (nx, ny)
        sides: Dict[str, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = {
            "left": ((bb.minx, bb.miny), (bb.minx, bb.maxy), (-1.0, 0.0)),
            "right": ((bb.maxx, bb.miny), (bb.maxx, bb.maxy), (1.0, 0.0)),
            "top": ((bb.minx, bb.miny), (bb.maxx, bb.miny), (0.0, -1.0)),
            "bottom": ((bb.minx, bb.maxy), (bb.maxx, bb.maxy), (0.0, 1.0)),
        }

        # choose epsilon in "pixels"
        eps = max(float(tol), 1.0)

        def _is_inside(px: float, py: float) -> bool:
            try:
                return bool(u.contains(Point(px, py)))
            except Exception:
                return False

        def side_is_outer(a: Tuple[float, float], b: Tuple[float, float], n: Tuple[float, float]) -> bool:
            ax, ay = a
            bx, by = b
            nx, ny = n
            # include near-ends too (partial attachments)
            for t in (0.1, 0.3, 0.5, 0.7, 0.9):
                x = ax + (bx - ax) * t
                y = ay + (by - ay) * t
                # inward is opposite of outward normal
                outx, outy = (x + nx * eps, y + ny * eps)
                inx, iny = (x - nx * eps, y - ny * eps)
                if _is_inside(inx, iny) and (not _is_inside(outx, outy)):
                    return True
            return False

        out.append({name: side_is_outer(a, b, n) for name, (a, b, n) in sides.items()})
    return out


def compute_overhang_sides_from_footprint(
    sections: List[Dict[str, Any]],
    footprint: Any,
    *,
    tol: float = 5.0,
) -> List[Dict[str, bool]]:
    """
    For multi-floor "remaining roof" (roof_levels), we MUST avoid extending into the interior
    cutout created by subtracting upper floors.

    A side is considered free ONLY if it lies on the *outer* boundary of the floor footprint.
    We test this by sampling points along the side and checking a point slightly "inward"
    vs "outward" relative to the footprint polygon.
    """
    try:
        from shapely.geometry import Point
    except Exception:
        return compute_free_sides_axis_aligned(sections, tol=tol)

    if footprint is None or getattr(footprint, "is_empty", True):
        return compute_free_sides_axis_aligned(sections, tol=tol)

    rects: List[Optional[RectBounds]] = [rect_bounds_from_bounding_rect(s.get("bounding_rect") or []) for s in sections]
    # `tol` here is a practical pixel tolerance:
    # - sections coming from masks can be inset from the true outer wall line by a few px
    # - we still want to treat those as exterior sides and apply overhang outward
    tol = float(tol)
    buf_eps = max(1.0, tol * 0.25)
    test_dist = max(1.0, tol)

    # Buffers make the test robust to small misalignments between section rectangles and footprint.
    try:
        fp_in = footprint.buffer(buf_eps)
    except Exception:
        fp_in = footprint
    try:
        fp_out = footprint.buffer(-buf_eps)
        if fp_out is None or getattr(fp_out, "is_empty", True):
            fp_out = footprint
    except Exception:
        fp_out = footprint

    def _is_inside_in(px: float, py: float) -> bool:
        try:
            return bool(fp_in.contains(Point(px, py)))
        except Exception:
            return False

    def _is_inside_out(px: float, py: float) -> bool:
        try:
            return bool(fp_out.contains(Point(px, py)))
        except Exception:
            return False

    out: List[Dict[str, bool]] = []
    for bb in rects:
        if bb is None:
            out.append({"left": True, "right": True, "top": True, "bottom": True})
            continue

        sides: Dict[str, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = {
            "left": ((bb.minx, bb.miny), (bb.minx, bb.maxy), (-1.0, 0.0)),
            "right": ((bb.maxx, bb.miny), (bb.maxx, bb.maxy), (1.0, 0.0)),
            "top": ((bb.minx, bb.miny), (bb.maxx, bb.miny), (0.0, -1.0)),
            "bottom": ((bb.minx, bb.maxy), (bb.maxx, bb.maxy), (0.0, 1.0)),
        }

        def side_is_outer(a: Tuple[float, float], b: Tuple[float, float], n: Tuple[float, float]) -> bool:
            ax, ay = a
            bx, by = b
            nx, ny = n
            for t in (0.1, 0.3, 0.5, 0.7, 0.9):
                x = ax + (bx - ax) * t
                y = ay + (by - ay) * t
                # Test points a bit farther away from the side so we can still
                # classify sides that are slightly inset from the footprint boundary.
                outx, outy = (x + nx * test_dist, y + ny * test_dist)
                inx, iny = (x - nx * test_dist, y - ny * test_dist)
                if _is_inside_in(inx, iny) and (not _is_inside_out(outx, outy)):
                    return True
            return False

        out.append({name: side_is_outer(a, b, n) for name, (a, b, n) in sides.items()})
    return out


def sides_touching_upper_floor(
    sections: List[Dict[str, Any]],
    union_upper: Any,
    *,
    tol: float = 5.0,
) -> List[Dict[str, bool]]:
    """
    Pentru fiecare secțiune, returnează care laturi ating complet etajul superior.
    Dacă o latură atinge → ridge-ul trebuie perpendicular pe ea.
    """
    try:
        from shapely.geometry import Point
        from shapely.ops import unary_union
    except Exception:
        return []

    if union_upper is None or getattr(union_upper, "is_empty", True):
        return [{"left": False, "right": False, "top": False, "bottom": False} for _ in sections]

    try:
        if hasattr(union_upper, "__iter__") and not hasattr(union_upper, "exterior"):
            union_upper = unary_union(list(union_upper))
    except Exception:
        return []

    tol = float(tol)
    test_dist = max(2.0, tol)

    out: List[Dict[str, bool]] = []
    for sec in sections:
        bb = rect_bounds_from_bounding_rect(sec.get("bounding_rect") or [])
        if bb is None:
            out.append({"left": False, "right": False, "top": False, "bottom": False})
            continue

        sides_info: Dict[str, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = {
            "left": ((bb.minx, bb.miny), (bb.minx, bb.maxy), (-1.0, 0.0)),
            "right": ((bb.maxx, bb.miny), (bb.maxx, bb.maxy), (1.0, 0.0)),
            "top": ((bb.minx, bb.miny), (bb.maxx, bb.miny), (0.0, -1.0)),
            "bottom": ((bb.minx, bb.maxy), (bb.maxx, bb.maxy), (0.0, 1.0)),
        }

        def side_touches_upper(a: Tuple[float, float], b: Tuple[float, float], n: Tuple[float, float]) -> bool:
            hits = 0
            total = 0
            for t in (0.2, 0.4, 0.5, 0.6, 0.8):
                x = a[0] + (b[0] - a[0]) * t
                y = a[1] + (b[1] - a[1]) * t
                out_x = x + n[0] * test_dist
                out_y = y + n[1] * test_dist
                total += 1
                try:
                    if union_upper.contains(Point(out_x, out_y)):
                        hits += 1
                except Exception:
                    pass
            return total > 0 and (hits / total) >= 0.6

        out.append({name: side_touches_upper(a, b, n) for name, (a, b, n) in sides_info.items()})
    return out


def high_side_for_shed_from_upper_floor(
    sections: List[Dict[str, Any]],
    union_upper: Any,
    *,
    tol: float = 5.0,
) -> List[str]:
    """
    Pentru acoperiș într-o apă (shed): latura mai înaltă = cea care ATINGE CEL MAI MULT
    etajul superior (chiar dacă nu atinge complet). Returnează "top" | "bottom" | "left" | "right".
    """
    try:
        from shapely.geometry import Point
        from shapely.ops import unary_union
    except Exception:
        return ["top"] * len(sections)

    if union_upper is None or getattr(union_upper, "is_empty", True):
        return ["top"] * len(sections)

    try:
        if hasattr(union_upper, "__iter__") and not hasattr(union_upper, "exterior"):
            union_upper = unary_union(list(union_upper))
    except Exception:
        return ["top"] * len(sections)

    test_dist = max(2.0, float(tol))
    out: List[str] = []

    for sec in sections:
        bb = rect_bounds_from_bounding_rect(sec.get("bounding_rect") or [])
        if bb is None:
            out.append("top")
            continue

        sides_info: Dict[str, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = {
            "left": ((bb.minx, bb.miny), (bb.minx, bb.maxy), (-1.0, 0.0)),
            "right": ((bb.maxx, bb.miny), (bb.maxx, bb.maxy), (1.0, 0.0)),
            "top": ((bb.minx, bb.miny), (bb.maxx, bb.miny), (0.0, -1.0)),
            "bottom": ((bb.minx, bb.maxy), (bb.maxx, bb.maxy), (0.0, 1.0)),
        }

        def touch_ratio(a: Tuple[float, float], b: Tuple[float, float], n: Tuple[float, float]) -> float:
            hits = 0
            total = 0
            for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                x = a[0] + (b[0] - a[0]) * t
                y = a[1] + (b[1] - a[1]) * t
                out_x = x + n[0] * test_dist
                out_y = y + n[1] * test_dist
                total += 1
                try:
                    if union_upper.contains(Point(out_x, out_y)):
                        hits += 1
                except Exception:
                    pass
            return hits / total if total > 0 else 0.0

        ratios = {name: touch_ratio(a, b, n) for name, (a, b, n) in sides_info.items()}
        best = max(ratios, key=ratios.get)  # type: ignore[arg-type]
        if ratios[best] > 0:
            out.append(best)
        else:
            # Fallback: latura cu mijlocul cel mai apropiat de centroid
            try:
                c = union_upper.centroid
                upper_cx, upper_cy = float(c.x), float(c.y)
            except Exception:
                out.append("top")
                continue
            mid_left = (bb.minx, (bb.miny + bb.maxy) / 2)
            mid_right = (bb.maxx, (bb.miny + bb.maxy) / 2)
            mid_top = ((bb.minx + bb.maxx) / 2, bb.miny)
            mid_bottom = ((bb.minx + bb.maxx) / 2, bb.maxy)
            sides_dist = {
                "left": (mid_left[0] - upper_cx) ** 2 + (mid_left[1] - upper_cy) ** 2,
                "right": (mid_right[0] - upper_cx) ** 2 + (mid_right[1] - upper_cy) ** 2,
                "top": (mid_top[0] - upper_cx) ** 2 + (mid_top[1] - upper_cy) ** 2,
                "bottom": (mid_bottom[0] - upper_cx) ** 2 + (mid_bottom[1] - upper_cy) ** 2,
            }
            out.append(min(sides_dist, key=sides_dist.get))  # type: ignore[arg-type]

    return out


def adjust_ridge_for_adjacent_floor(
    roof_data: Dict[str, Any],
    union_upper: Any,
) -> Dict[str, Any]:
    """
    Dacă o latură a secțiunii atinge complet etajul superior, ridge-ul trebuie perpendicular pe ea.
    - left/right atinge → ridge horizontal (ridge paralele cu latura)
    - top/bottom atinge → ridge vertical
    """
    sections = roof_data.get("sections") or []
    if not sections or union_upper is None or getattr(union_upper, "is_empty", True):
        return roof_data

    touching = sides_touching_upper_floor(sections, union_upper)
    new_sections: List[Dict[str, Any]] = []
    for idx, sec in enumerate(sections):
        t = touching[idx] if idx < len(touching) else {}
        if not t:
            new_sections.append(sec)
            continue

        orient = str(sec.get("ridge_orientation", "horizontal"))
        # left sau right atinge → ridge trebuie horizontal (perpendicular pe verticală)
        if t.get("left") or t.get("right"):
            orient = "horizontal"
        # top sau bottom atinge → ridge trebuie vertical (perpendicular pe orizontală)
        if t.get("top") or t.get("bottom"):
            orient = "vertical"

        # Prioritate: dacă ambele tipuri ating, preferă pe cea cu latură mai lungă sau default
        left_right_touch = t.get("left") or t.get("right")
        top_bottom_touch = t.get("top") or t.get("bottom")
        if left_right_touch and top_bottom_touch:
            bb = rect_bounds_from_bounding_rect(sec.get("bounding_rect") or [])
            if bb:
                if bb.w >= bb.h:
                    orient = "horizontal"
                else:
                    orient = "vertical"

        br = sec.get("bounding_rect") or []
        ridge = sec.get("ridge_line") or []
        if len(br) >= 3:
            minx = min(p[0] for p in br)
            maxx = max(p[0] for p in br)
            miny = min(p[1] for p in br)
            maxy = max(p[1] for p in br)
            cx = (minx + maxx) / 2.0
            cy = (miny + maxy) / 2.0
            if orient == "vertical":
                ridge = [(cx, miny), (cx, maxy)]
            else:
                ridge = [(minx, cy), (maxx, cy)]

        new_sections.append({**sec, "ridge_orientation": orient, "ridge_line": ridge})
    return {**roof_data, "sections": new_sections}


def apply_overhang_to_sections(
    sections: List[Dict[str, Any]],
    *,
    overhang_px: float,
    free_sides: Optional[List[Dict[str, bool]]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns new section dicts with expanded `bounding_rect` (and adjusted `ridge_line`)
    by overhang_px for each free side.
    """
    if overhang_px <= 0:
        return list(sections)
    if free_sides is None:
        free_sides = compute_free_sides_axis_aligned(sections)

    out: List[Dict[str, Any]] = []
    for idx, sec in enumerate(sections):
        bb = rect_bounds_from_bounding_rect(sec.get("bounding_rect") or [])
        if bb is None:
            out.append(sec)
            continue
        fs = free_sides[idx] if idx < len(free_sides) else {"left": True, "right": True, "top": True, "bottom": True}
        minx = bb.minx - (overhang_px if fs.get("left", True) else 0.0)
        maxx = bb.maxx + (overhang_px if fs.get("right", True) else 0.0)
        miny = bb.miny - (overhang_px if fs.get("top", True) else 0.0)
        maxy = bb.maxy + (overhang_px if fs.get("bottom", True) else 0.0)

        rect = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]

        ridge_line = sec.get("ridge_line") or []
        if len(ridge_line) >= 2:
            rx0, ry0 = _strip_xy(ridge_line[0])
            rx1, ry1 = _strip_xy(ridge_line[1])
            ridge_mid_x = (rx0 + rx1) / 2.0
            ridge_mid_y = (ry0 + ry1) / 2.0
        else:
            ridge_mid_x = (bb.minx + bb.maxx) / 2.0
            ridge_mid_y = (bb.miny + bb.maxy) / 2.0

        orient = sec.get("ridge_orientation", "horizontal")
        if orient == "vertical":
            ridge = [(ridge_mid_x, miny + 0.0), (ridge_mid_x, maxy + 0.0)]
        else:
            ridge = [(minx + 0.0, ridge_mid_y), (maxx + 0.0, ridge_mid_y)]

        out.append({**sec, "bounding_rect": rect, "ridge_line": ridge})
    return out


def get_drip_edge_faces_3d(
    sections: List[Dict[str, Any]],
    base_z: float,
    overhang_px: float,
    roof_angle_deg: float,
    *,
    ratio: float = 0.60,
    roof_shift_dz: float = 0.0,
    roof_faces: Optional[List[Dict[str, Any]]] = None,
    free_sides: Optional[List[Dict[str, bool]]] = None,
    exclude_low_side_shed: Optional[str] = None,
    exclude_low_sides: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Drip edge: vertical în jos (90°), lipit de acoperiș.
    Pentru fiecare pantă, top-ul drip-ului urmează suprafața acoperișului (z din roof_faces).
    Doar pe laturile libere, delimitat de ridge.
    exclude_low_side_shed: pentru shed (un singur low side) - deprecated, folosiți exclude_low_sides.
    exclude_low_sides: pentru shed - exclude latura cu z minim PER SECȚIUNE (listă, una per secțiune).
    """
    if overhang_px <= 0 or not sections:
        return []
    drop_px = ratio * overhang_px
    tol = 8.0

    def _z_at_boundary(x: float, y: float) -> float:
        """z la acoperiș la (x,y). Shed: interpolat pe plan. A-frame: min z din vârfuri apropiate."""
        if roof_faces:
            if exclude_low_side_shed is not None or exclude_low_sides:
                # Shed: planul acoperișului – z exact ca suprafața efectivă
                z_plane = _z_on_roof_plane(x, y, roof_faces)
                if z_plane is not None:
                    return float(z_plane) - float(roof_shift_dz)
            else:
                # A-frame: min z din vârfuri apropiate (logica originală)
                best = None
                for f in roof_faces:
                    for v in f.get("vertices_3d") or []:
                        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
                        if abs(vx - x) <= tol and abs(vy - y) <= tol:
                            if best is None or vz < best:
                                best = vz
                if best is not None:
                    return best
        import math

        tan_angle = math.tan(math.radians(roof_angle_deg))
        return base_z - overhang_px * tan_angle - float(roof_shift_dz)

    def _z_on_roof_plane(px: float, py: float, faces: List[Dict[str, Any]]) -> Optional[float]:
        """z pe planul acoperișului la (px,py) – interpolare/extrapolare din vârfurile fețelor."""
        best_z: Optional[float] = None
        best_dist: float = float("inf")
        for f in faces:
            vs = f.get("vertices_3d") or []
            if len(vs) < 3:
                continue
            v0 = [float(vs[0][0]), float(vs[0][1]), float(vs[0][2])]
            v1 = [float(vs[1][0]), float(vs[1][1]), float(vs[1][2])]
            v2 = [float(vs[2][0]), float(vs[2][1]), float(vs[2][2])]
            # Normal N = (v1-v0) x (v2-v0)
            dx1 = v1[0] - v0[0]
            dy1 = v1[1] - v0[1]
            dz1 = v1[2] - v0[2]
            dx2 = v2[0] - v0[0]
            dy2 = v2[1] - v0[1]
            dz2 = v2[2] - v0[2]
            nx = dy1 * dz2 - dz1 * dy2
            ny = dz1 * dx2 - dx1 * dz2
            nz = dx1 * dy2 - dy1 * dx2
            if abs(nz) < 1e-12:
                continue
            # z pe plan: N·(P-P0)=0 => z = z0 - (nx*(x-x0) + ny*(y-y0))/nz
            z_at = v0[2] - (nx * (px - v0[0]) + ny * (py - v0[1])) / nz
            # Preferăm fața cel mai aproape de (px,py) în 2D
            cx = sum(float(v[0]) for v in vs) / len(vs)
            cy = sum(float(v[1]) for v in vs) / len(vs)
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            if d2 < best_dist:
                best_dist = d2
                best_z = z_at
        return best_z

    # Shed: drip pe toate laturile în afară de cea cu z minim per secțiune; ignoră free_sides
    _shed_mode = exclude_low_side_shed is not None or exclude_low_sides
    _free = free_sides
    if _shed_mode:
        _free = [{"left": True, "right": True, "top": True, "bottom": True} for _ in sections]
    _exclude_list = exclude_low_sides
    if _exclude_list is None and exclude_low_side_shed is not None:
        _exclude_list = [exclude_low_side_shed] * len(sections)
    segments = get_drip_edge_segments_2d(
        sections,
        free_sides=_free,
        tol=tol,
        exclude_eaves=not _shed_mode,
        exclude_low_sides=_exclude_list,
    )
    faces: List[Dict[str, Any]] = []
    for (x1, y1), (x2, y2) in segments:
        z1_top = _z_at_boundary(x1, y1)
        z2_top = _z_at_boundary(x2, y2)
        z1_bot = z1_top - drop_px
        z2_bot = z2_top - drop_px
        face = [
            [x1, y1, z1_top],
            [x2, y2, z2_top],
            [x2, y2, z2_bot],
            [x1, y1, z1_bot],
        ]
        faces.append({"vertices_3d": face})
    return faces


def _segment_intersect(
    a1: Tuple[float, float], a2: Tuple[float, float],
    b1: Tuple[float, float], b2: Tuple[float, float],
    tol: float = 1e-9,
) -> Optional[Tuple[float, float]]:
    """Intersecția segment a1-a2 cu b1-b2. Returnează (x,y) sau None."""
    ax1, ay1 = a1
    ax2, ay2 = a2
    bx1, by1 = b1
    bx2, by2 = b2
    dxa = ax2 - ax1
    dya = ay2 - ay1
    dxb = bx2 - bx1
    dyb = by2 - by1
    denom = dxa * dyb - dya * dxb
    if abs(denom) < tol:
        return None
    t = ((bx1 - ax1) * dyb - (by1 - ay1) * dxb) / denom
    s = ((bx1 - ax1) * dya - (by1 - ay1) * dxa) / denom
    if -tol <= t <= 1 + tol and -tol <= s <= 1 + tol:
        ix = ax1 + t * dxa
        iy = ay1 + t * dya
        return (ix, iy)
    return None


def get_drip_edge_segments_2d(
    sections: List[Dict[str, Any]],
    *,
    free_sides: Optional[List[Dict[str, bool]]] = None,
    tol: float = 8.0,
    exclude_eaves: bool = True,
    exclude_low_side: Optional[str] = None,
    exclude_low_sides: Optional[List[str]] = None,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Returnează segmentele 2D pentru drip - doar pe laturile libere, delimitate de ridge.
    Ridge-ul acoperișului divide drip-ul: segmentele se întrerup la intersecția cu ridge-ul.
    exclude_eaves=True: drip nu este plasat pe streașină (laturile perpendiculare pe ridge).
    exclude_low_side: pentru shed (un singur low) - exclude latura dată la toate secțiunile.
    exclude_low_sides: pentru shed - exclude latura cu z minim PER SECȚIUNE (listă, una per secțiune).
    """
    if not sections:
        return []
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.ops import unary_union
    except Exception:
        return []

    polys = []
    rects: List[Optional[RectBounds]] = []
    ridge_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    ridge_orientations: List[str] = []
    for s in sections:
        br = s.get("bounding_rect") or []
        if len(br) < 3:
            rects.append(None)
            ridge_orientations.append("horizontal")
            continue
        try:
            polys.append(ShapelyPolygon([(float(p[0]), float(p[1])) for p in br]))
            rects.append(rect_bounds_from_bounding_rect(br))
            ridge_orientations.append(str(s.get("ridge_orientation", "horizontal")))
            rl = s.get("ridge_line") or []
            if len(rl) >= 2:
                ridge_segments.append((_strip_xy(rl[0]), _strip_xy(rl[1])))
        except Exception:
            rects.append(None)
            ridge_orientations.append("horizontal")
    if not polys:
        return []
    try:
        u = unary_union(polys)
    except Exception:
        return []
    if u is None or getattr(u, "is_empty", True):
        return []

    def _segment_on_eaves(x1: float, y1: float, x2: float, y2: float) -> bool:
        """Segmentul e pe streașină? Streașina = latura perpendiculară pe ridge (top/bottom pt ridge oriz, left/right pt ridge vert)."""
        if not exclude_eaves:
            return False
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        seg_minx, seg_maxx = min(x1, x2), max(x1, x2)
        seg_miny, seg_maxy = min(y1, y2), max(y1, y2)
        for idx, bb in enumerate(rects):
            if bb is None:
                continue
            orient = ridge_orientations[idx] if idx < len(ridge_orientations) else "horizontal"
            if orient == "vertical":
                eaves_sides = ("left", "right")
            else:
                eaves_sides = ("top", "bottom")
            if abs(y1 - y2) < tol:
                if abs(my - bb.miny) < tol and "top" in eaves_sides and seg_maxx > bb.minx + tol and seg_minx < bb.maxx - tol:
                    return True
                if abs(my - bb.maxy) < tol and "bottom" in eaves_sides and seg_maxx > bb.minx + tol and seg_minx < bb.maxx - tol:
                    return True
            if abs(x1 - x2) < tol:
                if abs(mx - bb.minx) < tol and "left" in eaves_sides and seg_maxy > bb.miny + tol and seg_miny < bb.maxy - tol:
                    return True
                if abs(mx - bb.maxx) < tol and "right" in eaves_sides and seg_maxy > bb.miny + tol and seg_miny < bb.maxy - tol:
                    return True
        return False

    def _segment_on_section_side(
        x1: float, y1: float, x2: float, y2: float, idx: int, side: str
    ) -> bool:
        """Segmentul e pe latura dată (top/bottom/left/right) a secțiunii idx?"""
        if not side or side not in ("top", "bottom", "left", "right") or idx >= len(rects):
            return False
        bb = rects[idx]
        if bb is None:
            return False
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        seg_minx, seg_maxx = min(x1, x2), max(x1, x2)
        seg_miny, seg_maxy = min(y1, y2), max(y1, y2)
        if side == "top" and abs(my - bb.miny) < tol and seg_maxx > bb.minx + tol and seg_minx < bb.maxx - tol:
            return True
        if side == "bottom" and abs(my - bb.maxy) < tol and seg_maxx > bb.minx + tol and seg_minx < bb.maxx - tol:
            return True
        if side == "left" and abs(mx - bb.minx) < tol and seg_maxy > bb.miny + tol and seg_miny < bb.maxy - tol:
            return True
        if side == "right" and abs(mx - bb.maxx) < tol and seg_maxy > bb.miny + tol and seg_miny < bb.maxy - tol:
            return True
        return False

    def _segment_on_any_exclude_low(x1: float, y1: float, x2: float, y2: float) -> bool:
        """Segmentul e pe latura cu z minim a vreunei secțiuni? (exclude drip)"""
        if exclude_low_sides:
            for idx, low in enumerate(exclude_low_sides):
                if low and _segment_on_section_side(x1, y1, x2, y2, idx, low):
                    return True
        if exclude_low_side:
            for idx in range(len(rects)):
                if _segment_on_section_side(x1, y1, x2, y2, idx, exclude_low_side):
                    return True
        return False

    def _segment_on_free_side(x1: float, y1: float, x2: float, y2: float) -> bool:
        if not free_sides or len(free_sides) != len(rects):
            return True
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        seg_minx, seg_maxx = min(x1, x2), max(x1, x2)
        seg_miny, seg_maxy = min(y1, y2), max(y1, y2)
        for idx, bb in enumerate(rects):
            if bb is None:
                continue
            fs = free_sides[idx] if idx < len(free_sides) else {}
            if abs(y1 - y2) < tol:
                if abs(my - bb.miny) < tol and fs.get("top") and seg_maxx > bb.minx + tol and seg_minx < bb.maxx - tol:
                    return True
                if abs(my - bb.maxy) < tol and fs.get("bottom") and seg_maxx > bb.minx + tol and seg_minx < bb.maxx - tol:
                    return True
            if abs(x1 - x2) < tol:
                if abs(mx - bb.minx) < tol and fs.get("left") and seg_maxy > bb.miny + tol and seg_miny < bb.maxy - tol:
                    return True
                if abs(mx - bb.maxx) < tol and fs.get("right") and seg_maxy > bb.miny + tol and seg_miny < bb.maxy - tol:
                    return True
        return False

    def _split_at_ridges(
        x1: float, y1: float, x2: float, y2: float,
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Sparte segmentul la intersecțiile cu ridge-urile."""
        pts: List[Tuple[float, float]] = [(x1, y1), (x2, y2)]
        for (rx0, ry0), (rx1, ry1) in ridge_segments:
            hit = _segment_intersect((x1, y1), (x2, y2), (rx0, ry0), (rx1, ry1))
            if hit is None:
                continue
            hx, hy = hit
            if (abs(hx - x1) < 1e-6 and abs(hy - y1) < 1e-6) or (abs(hx - x2) < 1e-6 and abs(hy - y2) < 1e-6):
                continue
            dist = (hx - x1) ** 2 + (hy - y1) ** 2
            if dist < 1e-12 or dist > (x2 - x1) ** 2 + (y2 - y1) ** 2 - 1e-12:
                continue
            pts.append(hit)
        if len(pts) <= 2:
            return [((x1, y1), (x2, y2))]
        dx, dy = x2 - x1, y2 - y1
        pts.sort(key=lambda p: (p[0] - x1) * dx + (p[1] - y1) * dy)
        out_subs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for j in range(len(pts) - 1):
            p0, p1 = pts[j], pts[j + 1]
            if (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 > 1e-12:
                out_subs.append((p0, p1))
        return out_subs if out_subs else [((x1, y1), (x2, y2))]

    out: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    geoms = getattr(u, "geoms", [u])
    for g in geoms:
        ext = getattr(g, "exterior", None)
        if ext is None:
            continue
        coords = list(ext.coords)
        for i in range(len(coords) - 1):
            x1, y1 = float(coords[i][0]), float(coords[i][1])
            x2, y2 = float(coords[i + 1][0]), float(coords[i + 1][1])
            for (sx1, sy1), (sx2, sy2) in _split_at_ridges(x1, y1, x2, y2):
                if not _segment_on_free_side(sx1, sy1, sx2, sy2):
                    continue
                if _segment_on_eaves(sx1, sy1, sx2, sy2):
                    continue
                if _segment_on_any_exclude_low(sx1, sy1, sx2, sy2):
                    continue
                out.append(((sx1, sy1), (sx2, sy2)))
    return out


def get_gutter_segments_2d(
    sections: List[Dict[str, Any]],
    *,
    exclude_low_sides: Optional[List[str]] = None,
    exclude_low_side: Optional[str] = None,
    include_eaves_only: bool = False,
    pyramid_all_sides: bool = False,
    tol: float = 8.0,
    sections_include_mask: Optional[List[bool]] = None,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Returnează segmentele 2D pentru streașina (gutter) – laturile unde NU e drip.
    Pentru shed: exclude_low_sides (latura cu z minim).
    Pentru a-frame: include_eaves_only – doar streașina (laturile perpendiculare pe ridge).
    Pentru pyramid: pyramid_all_sides – toate laturile (streașina pe toate cele 4 fețe).
    sections_include_mask: dacă dat, doar secțiunile cu mask[i]==True intră în union (exclude interior/parter).
    """
    if not sections:
        return []
    if not include_eaves_only and not exclude_low_sides and not exclude_low_side and not pyramid_all_sides:
        return []
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.ops import unary_union
    except Exception:
        return []

    polys = []
    rects: List[Optional[RectBounds]] = []
    ridge_orientations: List[str] = []
    include_mask = sections_include_mask if sections_include_mask and len(sections_include_mask) == len(sections) else None
    for idx, s in enumerate(sections):
        if include_mask is not None and not include_mask[idx]:
            rects.append(None)
            ridge_orientations.append("horizontal")
            continue
        br = s.get("bounding_rect") or []
        if len(br) < 3:
            rects.append(None)
            ridge_orientations.append("horizontal")
            continue
        try:
            polys.append(ShapelyPolygon([(float(p[0]), float(p[1])) for p in br]))
            rects.append(rect_bounds_from_bounding_rect(br))
            ridge_orientations.append(str(s.get("ridge_orientation", "horizontal")))
        except Exception:
            rects.append(None)
            ridge_orientations.append("horizontal")
    if not polys:
        return []
    try:
        u = unary_union(polys)
    except Exception:
        return []
    if u is None or getattr(u, "is_empty", True):
        return []

    def _segment_on_eaves(x1: float, y1: float, x2: float, y2: float) -> bool:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        seg_minx, seg_maxx = min(x1, x2), max(x1, x2)
        seg_miny, seg_maxy = min(y1, y2), max(y1, y2)
        for idx, bb in enumerate(rects):
            if bb is None:
                continue
            orient = ridge_orientations[idx] if idx < len(ridge_orientations) else "horizontal"
            eaves_sides = ("left", "right") if orient == "vertical" else ("top", "bottom")
            if abs(y1 - y2) < tol:
                if abs(my - bb.miny) < tol and "top" in eaves_sides and seg_maxx > bb.minx + tol and seg_minx < bb.maxx - tol:
                    return True
                if abs(my - bb.maxy) < tol and "bottom" in eaves_sides and seg_maxx > bb.minx + tol and seg_minx < bb.maxx - tol:
                    return True
            if abs(x1 - x2) < tol:
                if abs(mx - bb.minx) < tol and "left" in eaves_sides and seg_maxy > bb.miny + tol and seg_miny < bb.maxy - tol:
                    return True
                if abs(mx - bb.maxx) < tol and "right" in eaves_sides and seg_maxy > bb.miny + tol and seg_miny < bb.maxy - tol:
                    return True
        return False

    def _segment_on_section_side(
        x1: float, y1: float, x2: float, y2: float, idx: int, side: str
    ) -> bool:
        if not side or side not in ("top", "bottom", "left", "right") or idx >= len(rects):
            return False
        bb = rects[idx]
        if bb is None:
            return False
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        seg_minx, seg_maxx = min(x1, x2), max(x1, x2)
        seg_miny, seg_maxy = min(y1, y2), max(y1, y2)
        if side == "top" and abs(my - bb.miny) < tol and seg_maxx > bb.minx + tol and seg_minx < bb.maxx - tol:
            return True
        if side == "bottom" and abs(my - bb.maxy) < tol and seg_maxx > bb.minx + tol and seg_minx < bb.maxx - tol:
            return True
        if side == "left" and abs(mx - bb.minx) < tol and seg_maxy > bb.miny + tol and seg_miny < bb.maxy - tol:
            return True
        if side == "right" and abs(mx - bb.maxx) < tol and seg_maxy > bb.miny + tol and seg_miny < bb.maxy - tol:
            return True
        return False

    def _on_gutter_side(x1: float, y1: float, x2: float, y2: float) -> bool:
        if exclude_low_sides:
            for idx, low in enumerate(exclude_low_sides):
                if low and _segment_on_section_side(x1, y1, x2, y2, idx, low):
                    return True
        if exclude_low_side:
            for idx in range(len(rects)):
                if _segment_on_section_side(x1, y1, x2, y2, idx, exclude_low_side):
                    return True
        return False

    out: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    geoms = getattr(u, "geoms", [u])
    for g in geoms:
        ext = getattr(g, "exterior", None)
        if ext is None:
            continue
        coords = list(ext.coords)
        for i in range(len(coords) - 1):
            x1, y1 = float(coords[i][0]), float(coords[i][1])
            x2, y2 = float(coords[i + 1][0]), float(coords[i + 1][1])
            if pyramid_all_sides:
                out.append(((x1, y1), (x2, y2)))
            elif include_eaves_only:
                if _segment_on_eaves(x1, y1, x2, y2):
                    out.append(((x1, y1), (x2, y2)))
            elif _on_gutter_side(x1, y1, x2, y2):
                out.append(((x1, y1), (x2, y2)))
    return out


def get_gutter_faces_3d(
    sections: List[Dict[str, Any]],
    base_z: float,
    overhang_px: float,
    roof_angle_deg: float,
    *,
    exclude_low_sides: Optional[List[str]] = None,
    exclude_low_side: Optional[str] = None,
    include_eaves_only: bool = False,
    pyramid_all_sides: bool = False,
    pyramid_extend: bool = False,
    roof_shift_dz: float = 0.0,
    roof_faces: Optional[List[Dict[str, Any]]] = None,
    radius_ratio: float = 0.24,
    n_half_circle: int = 6,
    eaves_z_lift: float = 0.0,
    sections_include_mask: Optional[List[bool]] = None,
    interior_reference_sections: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Streașină colectoare de apă: cilindru tăiat la jumătate, margine plată lipită de acoperiș.
    Shed: exclude_low_sides. A-frame: include_eaves_only. Pyramid: pyramid_all_sides.
    sections_include_mask: exclude secțiuni fără acoperiș expus (ex. parter sub etaj).
    interior_reference_sections: dacă dat (ex. etaj superior), perp2 orientat spre exterior (departe de el).
    """
    import math

    if not sections:
        return []
    if not include_eaves_only and not exclude_low_sides and not exclude_low_side and not pyramid_all_sides:
        return []
    tol = 8.0
    r = max(2.0, overhang_px * radius_ratio) * 0.70

    def _z_at(x: float, y: float) -> float:
        if roof_faces:
            z_plane = _z_on_roof_plane_for_gutter(x, y, roof_faces)
            if z_plane is not None:
                return float(z_plane) - float(roof_shift_dz)
        tan_angle = math.tan(math.radians(roof_angle_deg))
        return base_z - overhang_px * tan_angle - float(roof_shift_dz)

    def _z_on_roof_plane_for_gutter(px: float, py: float, faces: List[Dict[str, Any]]) -> Optional[float]:
        best_z: Optional[float] = None
        best_dist: float = float("inf")
        for f in faces:
            vs = f.get("vertices_3d") or []
            if len(vs) < 3:
                continue
            v0 = [float(vs[0][0]), float(vs[0][1]), float(vs[0][2])]
            v1 = [float(vs[1][0]), float(vs[1][1]), float(vs[1][2])]
            v2 = [float(vs[2][0]), float(vs[2][1]), float(vs[2][2])]
            dx1, dy1, dz1 = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
            dx2, dy2, dz2 = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
            nx = dy1 * dz2 - dz1 * dy2
            ny = dz1 * dx2 - dx1 * dz2
            nz = dx1 * dy2 - dy1 * dx2
            if abs(nz) < 1e-12:
                continue
            z_at = v0[2] - (nx * (px - v0[0]) + ny * (py - v0[1])) / nz
            cx = sum(float(v[0]) for v in vs) / len(vs)
            cy = sum(float(v[1]) for v in vs) / len(vs)
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            if d2 < best_dist:
                best_dist = d2
                best_z = z_at
        return best_z

    _excl = exclude_low_sides
    if _excl is None and exclude_low_side:
        _excl = [exclude_low_side] * len(sections)
    segments = get_gutter_segments_2d(
        sections,
        exclude_low_sides=_excl,
        exclude_low_side=exclude_low_side,
        include_eaves_only=include_eaves_only,
        pyramid_all_sides=pyramid_all_sides,
        tol=tol,
        sections_include_mask=sections_include_mask,
    )
    result: List[Dict[str, Any]] = []

    z_lift = float(eaves_z_lift)
    extend_by = r if (pyramid_all_sides and pyramid_extend) else 0.0

    for (x1, y1), (x2, y2) in segments:
        z1 = _z_at(x1, y1) + z_lift
        z2 = _z_at(x2, y2) + z_lift
        if extend_by > 0:
            dx_2d = x2 - x1
            dy_2d = y2 - y1
            L_2d = math.sqrt(dx_2d * dx_2d + dy_2d * dy_2d)
            if L_2d >= 1e-9:
                ux, uy = dx_2d / L_2d, dy_2d / L_2d
                x1, y1 = x1 - ux * extend_by, y1 - uy * extend_by
                x2, y2 = x2 + ux * extend_by, y2 + uy * extend_by
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        # Direcția axei cilindrului (de la capăt la capăt)
        ax = x2 - x1
        ay = y2 - y1
        az = z2 - z1
        L = math.sqrt(ax * ax + ay * ay + az * az)
        if L < 1e-9:
            continue
        ax, ay, az = ax / L, ay / L, az / L
        # Jgheabul atârnă vertical: marginea plată lipită de acoperiș, U-ul se deschide în jos.
        # perp2 = diametrul (orizontal), perp1 = jos. Poziționare la MARGINE: semicerc spre exterior, nu suprapus.
        perp2x, perp2y, perp2z = -ay, ax, 0.0  # axis × (0,0,1) – orizontal, perpendicular pe axă
        n2 = math.sqrt(perp2x * perp2x + perp2y * perp2y + perp2z * perp2z)
        if n2 < 1e-9:
            perp2x, perp2y, perp2z = 1.0, 0.0, 0.0
        else:
            perp2x, perp2y, perp2z = perp2x / n2, perp2y / n2, perp2z / n2
        # Orientează perp2 spre exterior (departe de interior). Când există interior_reference_sections,
        # folosim centrul lor ca referință. Verificare finală: punctul (mx,my)+perp2*r trebuie să fie
        # ÎNAFARA union-ului secțiunilor curente – altfel streașina intră sub acoperiș (ex. laturi adiacente).
        try:
            from shapely.ops import unary_union
            from shapely.geometry import Polygon as ShapelyPolygon
            from shapely.geometry import Point as ShapelyPoint
            ref_polys = []
            ref_secs = interior_reference_sections if interior_reference_sections else sections
            sec_polys = []
            include_mask = sections_include_mask if sections_include_mask and len(sections_include_mask) == len(sections) else None
            for idx, s in enumerate(sections):
                if include_mask is not None and not include_mask[idx]:
                    continue
                br = s.get("bounding_rect") or []
                if len(br) >= 3:
                    sec_polys.append(ShapelyPolygon([(float(p[0]), float(p[1])) for p in br]))
            for s in ref_secs:
                br = s.get("bounding_rect") or []
                if len(br) >= 3:
                    ref_polys.append(ShapelyPolygon([(float(p[0]), float(p[1])) for p in br]))
            if ref_polys:
                uu_ref = unary_union(ref_polys)
                cx = float(uu_ref.centroid.x)
                cy = float(uu_ref.centroid.y)
                out_x = mx - cx
                out_y = my - cy
                out_len = math.sqrt(out_x * out_x + out_y * out_y)
                if out_len > 1e-9:
                    out_x, out_y = out_x / out_len, out_y / out_len
                    dot = perp2x * out_x + perp2y * out_y
                    if dot < 0:
                        perp2x, perp2y, perp2z = -perp2x, -perp2y, -perp2z
            if sec_polys:
                uu_sec = unary_union(sec_polys)
                test_dist = max(r * 5.0, 20.0)
                pt_test = ShapelyPoint(mx + perp2x * test_dist, my + perp2y * test_dist)
                if not getattr(uu_sec, "is_empty", True) and uu_sec.contains(pt_test):
                    perp2x, perp2y, perp2z = -perp2x, -perp2y, -perp2z
        except Exception:
            pass
        perp1x = ay * perp2z - az * perp2y
        perp1y = az * perp2x - ax * perp2z
        perp1z = ax * perp2y - ay * perp2x
        if perp1z > 0:
            perp1x, perp1y, perp1z = -perp1x, -perp1y, -perp1z
        n1 = math.sqrt(perp1x * perp1x + perp1y * perp1y + perp1z * perp1z)
        if n1 >= 1e-9:
            perp1x, perp1y, perp1z = perp1x / n1, perp1y / n1, perp1z / n1
        # Offset: mută semicercul la margine – un capăt al diametrului pe acoperiș, restul spre exterior (fără suprapunere).
        # center_new = center + perp2*r → diametrul de la (roof_edge) până la (roof_edge + 2*perp2*r)
        center_off_x = perp2x * r
        center_off_y = perp2y * r
        center_off_z = perp2z * r
        # punct = center + center_off + perp2*r*cos(t) + perp1*r*sin(t), t în [0, pi]
        angs = [math.pi * t / (n_half_circle + 1) for t in range(n_half_circle + 2)]
        pts_start: List[List[float]] = []
        pts_end: List[List[float]] = []
        for ang in angs:
            dx = center_off_x + perp2x * math.cos(ang) * r + perp1x * math.sin(ang) * r
            dy = center_off_y + perp2y * math.cos(ang) * r + perp1y * math.sin(ang) * r
            dz = center_off_z + perp2z * math.cos(ang) * r + perp1z * math.sin(ang) * r
            pts_start.append([x1 + dx, y1 + dy, z1 + dz])
            pts_end.append([x2 + dx, y2 + dy, z2 + dz])
        # Fețe laterale: quad între pts_start[i], pts_start[i+1], pts_end[i+1], pts_end[i]
        for i in range(len(pts_start) - 1):
            face = [
                pts_start[i],
                pts_start[i + 1],
                pts_end[i + 1],
                pts_end[i],
            ]
            result.append({"vertices_3d": face})
        # Capete: nu aici – se adaugă în get_gutter_end_closures_3d doar la capetele fără burlan

    return result


def get_gutter_centerlines_3d(
    sections: List[Dict[str, Any]],
    base_z: float,
    overhang_px: float,
    roof_angle_deg: float,
    *,
    exclude_low_sides: Optional[List[str]] = None,
    exclude_low_side: Optional[str] = None,
    include_eaves_only: bool = False,
    pyramid_all_sides: bool = False,
    pyramid_extend: bool = False,
    roof_shift_dz: float = 0.0,
    roof_faces: Optional[List[Dict[str, Any]]] = None,
    radius_ratio: float = 0.24,
    eaves_z_lift: float = 0.0,
    sections_include_mask: Optional[List[bool]] = None,
    interior_reference_sections: Optional[List[Dict[str, Any]]] = None,
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    Returnează linia de centru a fiecărui segment de jgheab: ((cx1,cy1,cz1), (cx2,cy2,cz2)).
    Axa cilindrului trece prin centrul secțiunii semicirculare.
    """
    import math

    if not sections:
        return []
    if not include_eaves_only and not exclude_low_sides and not exclude_low_side and not pyramid_all_sides:
        return []
    tol = 8.0
    r = max(2.0, overhang_px * radius_ratio) * 0.70

    def _z_at(x: float, y: float) -> float:
        if roof_faces:
            z_plane = _z_on_roof_plane_for_gutter(x, y, roof_faces)
            if z_plane is not None:
                return float(z_plane) - float(roof_shift_dz)
        tan_angle = math.tan(math.radians(roof_angle_deg))
        return base_z - overhang_px * tan_angle - float(roof_shift_dz)

    def _z_on_roof_plane_for_gutter(px: float, py: float, faces: List[Dict[str, Any]]) -> Optional[float]:
        best_z: Optional[float] = None
        best_dist: float = float("inf")
        for f in faces:
            vs = f.get("vertices_3d") or []
            if len(vs) < 3:
                continue
            v0 = [float(vs[0][0]), float(vs[0][1]), float(vs[0][2])]
            v1 = [float(vs[1][0]), float(vs[1][1]), float(vs[1][2])]
            v2 = [float(vs[2][0]), float(vs[2][1]), float(vs[2][2])]
            dx1, dy1, dz1 = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
            dx2, dy2, dz2 = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
            nx = dy1 * dz2 - dz1 * dy2
            ny = dz1 * dx2 - dx1 * dz2
            nz = dx1 * dy2 - dy1 * dx2
            if abs(nz) < 1e-12:
                continue
            z_at = v0[2] - (nx * (px - v0[0]) + ny * (py - v0[1])) / nz
            cx = sum(float(v[0]) for v in vs) / len(vs)
            cy = sum(float(v[1]) for v in vs) / len(vs)
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            if d2 < best_dist:
                best_dist = d2
                best_z = z_at
        return best_z

    _excl = exclude_low_sides
    if _excl is None and exclude_low_side:
        _excl = [exclude_low_side] * len(sections)
    segments = get_gutter_segments_2d(
        sections,
        exclude_low_sides=_excl,
        exclude_low_side=exclude_low_side,
        include_eaves_only=include_eaves_only,
        pyramid_all_sides=pyramid_all_sides,
        tol=tol,
        sections_include_mask=sections_include_mask,
    )
    result: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
    z_lift = float(eaves_z_lift)
    extend_by = r if (pyramid_all_sides and pyramid_extend) else 0.0

    for (x1, y1), (x2, y2) in segments:
        z1 = _z_at(x1, y1) + z_lift
        z2 = _z_at(x2, y2) + z_lift
        if extend_by > 0:
            dx_2d = x2 - x1
            dy_2d = y2 - y1
            L_2d = math.sqrt(dx_2d * dx_2d + dy_2d * dy_2d)
            if L_2d >= 1e-9:
                ux, uy = dx_2d / L_2d, dy_2d / L_2d
                x1, y1 = x1 - ux * extend_by, y1 - uy * extend_by
                x2, y2 = x2 + ux * extend_by, y2 + uy * extend_by
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax = x2 - x1
        ay = y2 - y1
        az = z2 - z1
        L = math.sqrt(ax * ax + ay * ay + az * az)
        if L < 1e-9:
            continue
        ax, ay, az = ax / L, ay / L, az / L
        perp2x, perp2y, perp2z = -ay, ax, 0.0
        n2 = math.sqrt(perp2x * perp2x + perp2y * perp2y + perp2z * perp2z)
        if n2 < 1e-9:
            perp2x, perp2y, perp2z = 1.0, 0.0, 0.0
        else:
            perp2x, perp2y, perp2z = perp2x / n2, perp2y / n2, perp2z / n2
        try:
            from shapely.ops import unary_union
            from shapely.geometry import Polygon as ShapelyPolygon
            from shapely.geometry import Point as ShapelyPoint
            ref_polys = []
            ref_secs = interior_reference_sections if interior_reference_sections else sections
            sec_polys = []
            include_mask = sections_include_mask if sections_include_mask and len(sections_include_mask) == len(sections) else None
            for idx, s in enumerate(sections):
                if include_mask is not None and not include_mask[idx]:
                    continue
                br = s.get("bounding_rect") or []
                if len(br) >= 3:
                    sec_polys.append(ShapelyPolygon([(float(p[0]), float(p[1])) for p in br]))
            for s in ref_secs:
                br = s.get("bounding_rect") or []
                if len(br) >= 3:
                    ref_polys.append(ShapelyPolygon([(float(p[0]), float(p[1])) for p in br]))
            if ref_polys:
                uu_ref = unary_union(ref_polys)
                cx_ref = float(uu_ref.centroid.x)
                cy_ref = float(uu_ref.centroid.y)
                out_x = mx - cx_ref
                out_y = my - cy_ref
                out_len = math.sqrt(out_x * out_x + out_y * out_y)
                if out_len > 1e-9:
                    out_x, out_y = out_x / out_len, out_y / out_len
                    if perp2x * out_x + perp2y * out_y < 0:
                        perp2x, perp2y, perp2z = -perp2x, -perp2y, -perp2z
            if sec_polys:
                uu_sec = unary_union(sec_polys)
                if not getattr(uu_sec, "is_empty", True):
                    pt_test = ShapelyPoint(mx + perp2x * max(r * 5.0, 20.0), my + perp2y * max(r * 5.0, 20.0))
                    if uu_sec.contains(pt_test):
                        perp2x, perp2y, perp2z = -perp2x, -perp2y, -perp2z
        except Exception:
            pass
        cx1 = x1 + perp2x * r
        cy1 = y1 + perp2y * r
        cz1 = z1 + perp2z * r
        cx2 = x2 + perp2x * r
        cy2 = y2 + perp2y * r
        cz2 = z2 + perp2z * r
        result.append(((cx1, cy1, cz1), (cx2, cy2, cz2)))

    return result


def get_gutter_end_closures_3d(
    sections: List[Dict[str, Any]],
    base_z: float,
    overhang_px: float,
    roof_angle_deg: float,
    *,
    exclude_low_sides: Optional[List[str]] = None,
    exclude_low_side: Optional[str] = None,
    include_eaves_only: bool = False,
    pyramid_all_sides: bool = False,
    pyramid_extend: bool = False,
    roof_shift_dz: float = 0.0,
    roof_faces: Optional[List[Dict[str, Any]]] = None,
    radius_ratio: float = 0.24,
    n_half_circle: int = 6,
    eaves_z_lift: float = 0.0,
    tol: float = 8.0,
    downspout_endpoints: Optional[Iterable[Tuple[float, float, float]]] = None,
) -> List[Dict[str, Any]]:
    """Capace de închidere la capetele jgheaburilor – aceeași geometrie semicirculară ca jgheabul.
    Nu adaugă cap la capete unde există burlan (downspout_endpoints)."""
    import math

    if not sections:
        return []
    if not include_eaves_only and not exclude_low_sides and not exclude_low_side and not pyramid_all_sides:
        return []
    r = max(2.0, overhang_px * radius_ratio) * 0.70
    extend_by = r if (pyramid_all_sides and pyramid_extend) else 0.0

    def _z_at(x: float, y: float) -> float:
        if roof_faces:
            z_plane = _z_on_roof_plane_for_gutter(x, y, roof_faces)
            if z_plane is not None:
                return float(z_plane) - float(roof_shift_dz)
        tan_angle = math.tan(math.radians(roof_angle_deg))
        return base_z - overhang_px * tan_angle - float(roof_shift_dz)

    def _z_on_roof_plane_for_gutter(px: float, py: float, faces: List[Dict[str, Any]]) -> Optional[float]:
        best_z: Optional[float] = None
        best_dist: float = float("inf")
        for f in faces:
            vs = f.get("vertices_3d") or []
            if len(vs) < 3:
                continue
            v0 = [float(vs[0][0]), float(vs[0][1]), float(vs[0][2])]
            v1 = [float(vs[1][0]), float(vs[1][1]), float(vs[1][2])]
            v2 = [float(vs[2][0]), float(vs[2][1]), float(vs[2][2])]
            dx1, dy1, dz1 = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
            dx2, dy2, dz2 = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
            nx = dy1 * dz2 - dz1 * dy2
            ny = dz1 * dx2 - dx1 * dz2
            nz = dx1 * dy2 - dy1 * dx2
            if abs(nz) < 1e-12:
                continue
            z_at = v0[2] - (nx * (px - v0[0]) + ny * (py - v0[1])) / nz
            cx = sum(float(v[0]) for v in vs) / len(vs)
            cy = sum(float(v[1]) for v in vs) / len(vs)
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            if d2 < best_dist:
                best_dist = d2
                best_z = z_at
        return best_z

    _excl = exclude_low_sides
    if _excl is None and exclude_low_side:
        _excl = [exclude_low_side] * len(sections)
    segments = get_gutter_segments_2d(
        sections,
        exclude_low_sides=_excl,
        exclude_low_side=exclude_low_side,
        include_eaves_only=include_eaves_only,
        pyramid_all_sides=pyramid_all_sides,
        tol=tol,
    )
    z_lift = float(eaves_z_lift)
    result: List[Dict[str, Any]] = []
    downspout_set = list(downspout_endpoints) if downspout_endpoints is not None else []
    end_tol = max(tol * 2, r * 2.5) if extend_by > 0 else tol * 2
    end_tol_sq = end_tol * end_tol

    def _has_downspout(ex: float, ey: float, ez: float) -> bool:
        for (dx, dy, dz) in downspout_set:
            if (ex - dx) ** 2 + (ey - dy) ** 2 + (ez - dz) ** 2 <= end_tol_sq:
                return True
        return False

    for (x1, y1), (x2, y2) in segments:
        z1 = _z_at(x1, y1) + z_lift
        z2 = _z_at(x2, y2) + z_lift
        if extend_by > 0:
            dx_2d = x2 - x1
            dy_2d = y2 - y1
            L_2d = math.sqrt(dx_2d * dx_2d + dy_2d * dy_2d)
            if L_2d >= 1e-9:
                ux, uy = dx_2d / L_2d, dy_2d / L_2d
                x1, y1 = x1 - ux * extend_by, y1 - uy * extend_by
                x2, y2 = x2 + ux * extend_by, y2 + uy * extend_by
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax = x2 - x1
        ay = y2 - y1
        az = z2 - z1
        L = math.sqrt(ax * ax + ay * ay + az * az)
        if L < 1e-9:
            continue
        ax, ay, az = ax / L, ay / L, az / L
        perp2x, perp2y, perp2z = -ay, ax, 0.0
        n2 = math.sqrt(perp2x * perp2x + perp2y * perp2y + perp2z * perp2z)
        if n2 < 1e-9:
            perp2x, perp2y, perp2z = 1.0, 0.0, 0.0
        else:
            perp2x, perp2y, perp2z = perp2x / n2, perp2y / n2, perp2z / n2
        try:
            from shapely.ops import unary_union
            from shapely.geometry import Polygon as ShapelyPolygon
            polys = []
            for s in sections:
                br = s.get("bounding_rect") or []
                if len(br) >= 3:
                    polys.append(ShapelyPolygon([(float(p[0]), float(p[1])) for p in br]))
            if polys:
                uu = unary_union(polys)
                cx_u = float(uu.centroid.x)
                cy_u = float(uu.centroid.y)
                out_x = mx - cx_u
                out_y = my - cy_u
                out_len = math.sqrt(out_x * out_x + out_y * out_y)
                if out_len > 1e-9:
                    out_x, out_y = out_x / out_len, out_y / out_len
                    if perp2x * out_x + perp2y * out_y < 0:
                        perp2x, perp2y, perp2z = -perp2x, -perp2y, -perp2z
        except Exception:
            pass
        perp1x = ay * perp2z - az * perp2y
        perp1y = az * perp2x - ax * perp2z
        perp1z = ax * perp2y - ay * perp2x
        if perp1z > 0:
            perp1x, perp1y, perp1z = -perp1x, -perp1y, -perp1z
        n1 = math.sqrt(perp1x * perp1x + perp1y * perp1y + perp1z * perp1z)
        if n1 >= 1e-9:
            perp1x, perp1y, perp1z = perp1x / n1, perp1y / n1, perp1z / n1
        center_off_x, center_off_y, center_off_z = perp2x * r, perp2y * r, perp2z * r
        angs = [math.pi * t / (n_half_circle + 1) for t in range(n_half_circle + 2)]

        def _add_semicap(cx_pt: float, cy_pt: float, cz_pt: float) -> None:
            if _has_downspout(cx_pt, cy_pt, cz_pt):
                return
            pts: List[List[float]] = []
            for ang in angs:
                dx = center_off_x + perp2x * math.cos(ang) * r + perp1x * math.sin(ang) * r
                dy = center_off_y + perp2y * math.cos(ang) * r + perp1y * math.sin(ang) * r
                dz = center_off_z + perp2z * math.cos(ang) * r + perp1z * math.sin(ang) * r
                pts.append([cx_pt + dx, cy_pt + dy, cz_pt + dz])
            for i in range(len(pts) - 1):
                result.append({"vertices_3d": [[cx_pt, cy_pt, cz_pt], pts[i], pts[i + 1]]})

        _add_semicap(x1, y1, z1)
        _add_semicap(x2, y2, z2)

    return result


def get_pyramid_corner_hemispheres_3d(
    gutter_endpoints: List[Tuple[float, float, float]],
    gutter_segment_sections: List[List[Dict[str, Any]]],
    overhang_px: float,
    *,
    downspout_endpoints: Optional[Iterable[Tuple[float, float, float]]] = None,
    sections_for_centroid: Optional[List[Dict[str, Any]]] = None,
    interior_reference_sections: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Pentru piramidă: la colțuri fără burlan, emisferă (jumătate de sferă) poziționată
    identic cu stub-urile burlanelor – la intersecția capetelor jgheaburilor, pe bisector.
    interior_reference_sections: dacă dat (ex. etaj superior), emisfera e poziționată
    spre exterior (departe de el) – ca la streașini.
    """
    import math

    if not gutter_endpoints or len(gutter_endpoints) < 2 or not gutter_segment_sections:
        return []
    r = max(2.0, overhang_px * 0.24) * 0.70
    tol_sq = (20.0) ** 2
    downspout_set = list(downspout_endpoints) if downspout_endpoints else []
    end_tol_sq = (r * 2.5) ** 2
    z_bucket = 30.0

    def _has_downspout(ex: float, ey: float, ez: float) -> bool:
        for (dx, dy, dz) in downspout_set:
            if (ex - dx) ** 2 + (ey - dy) ** 2 + (ez - dz) ** 2 <= end_tol_sq:
                return True
        return False

    def _key(p: Tuple[float, float, float]) -> Tuple[int, int, int]:
        return (round(p[0] / 2.0), round(p[1] / 2.0), int(round(p[2] / z_bucket)))

    corner_to_segments: Dict[Tuple[int, int, int], List[Tuple[int, Tuple[float, float, float]]]] = {}
    for seg_idx in range(len(gutter_segment_sections)):
        if not gutter_segment_sections[seg_idx] or seg_idx * 2 + 1 >= len(gutter_endpoints):
            continue
        p1 = gutter_endpoints[seg_idx * 2]
        p2 = gutter_endpoints[seg_idx * 2 + 1]
        for ep in (p1, p2):
            k = _key(ep)
            if k not in corner_to_segments:
                corner_to_segments[k] = []
            corner_to_segments[k].append((seg_idx, ep))

    cent = (0.0, 0.0)
    ref_secs = interior_reference_sections if interior_reference_sections else sections_for_centroid
    if ref_secs:
        try:
            from shapely.ops import unary_union
            from shapely.geometry import Polygon as ShapelyPolygon
            polys = []
            for s in ref_secs:
                br = s.get("bounding_rect") or []
                if len(br) >= 3:
                    polys.append(ShapelyPolygon([(float(p[0]), float(p[1])) for p in br]))
            if polys:
                uu = unary_union(polys)
                cent = (float(uu.centroid.x), float(uu.centroid.y))
        except Exception:
            pass
    if cent == (0.0, 0.0):
        xs = [gutter_endpoints[i][0] for i in range(0, len(gutter_endpoints) - 1, 2)]
        ys = [gutter_endpoints[i][1] for i in range(0, len(gutter_endpoints) - 1, 2)]
        if xs and ys:
            cent = (sum(xs) / len(xs), sum(ys) / len(ys))

    result: List[Dict[str, Any]] = []
    seen_corners: set = set()

    for k, segs_at in corner_to_segments.items():
        if len(segs_at) < 2:
            continue
        segs_dedup: List[Tuple[int, Tuple[float, float, float]]] = []
        seen_seg: set = set()
        for seg_idx, ep in segs_at:
            if seg_idx in seen_seg:
                continue
            seen_seg.add(seg_idx)
            segs_dedup.append((seg_idx, ep))
        if len(segs_dedup) != 2:
            continue
        (seg_a, ep_a), (seg_b, ep_b) = segs_dedup
        corner = ep_a
        if (corner[0] - ep_b[0]) ** 2 + (corner[1] - ep_b[1]) ** 2 > tol_sq:
            continue
        gx, gy, gz = corner[0], corner[1], corner[2]
        if _has_downspout(gx, gy, gz):
            continue
        if k in seen_corners:
            continue
        seen_corners.add(k)

        p1_a, p2_a = gutter_endpoints[seg_a * 2], gutter_endpoints[seg_a * 2 + 1]
        p1_b, p2_b = gutter_endpoints[seg_b * 2], gutter_endpoints[seg_b * 2 + 1]
        ua = (p2_a[0] - p1_a[0], p2_a[1] - p1_a[1])
        ub = (p2_b[0] - p1_b[0], p2_b[1] - p1_b[1])
        la = math.sqrt(ua[0] * ua[0] + ua[1] * ua[1] + 1e-12)
        lb = math.sqrt(ub[0] * ub[0] + ub[1] * ub[1] + 1e-12)
        if la >= 1e-9 and lb >= 1e-9:
            ua = (ua[0] / la, ua[1] / la)
            ub = (ub[0] / lb, ub[1] / lb)
            if abs(ua[0] * ub[0] + ua[1] * ub[1]) > 0.98:
                continue

        perp2_list: List[Tuple[float, float, float]] = []
        tol_corner_sq = (15.0) ** 2
        for seg_idx, ep in segs_dedup:
            if seg_idx * 2 + 1 >= len(gutter_endpoints):
                continue
            p1, p2 = gutter_endpoints[seg_idx * 2], gutter_endpoints[seg_idx * 2 + 1]
            if (ep[0] - gx) ** 2 + (ep[1] - gy) ** 2 > tol_corner_sq:
                continue
            ax_s = p2[0] - p1[0]
            ay_s = p2[1] - p1[1]
            Ls = math.sqrt(ax_s * ax_s + ay_s * ay_s + 1e-12)
            if Ls < 1e-9:
                continue
            ax_s, ay_s = ax_s / Ls, ay_s / Ls
            px, py, pz = -ay_s, ax_s, 0.0
            ns = math.sqrt(px * px + py * py + 1e-12)
            if ns >= 1e-9:
                px, py, pz = px / ns, py / ns, pz / ns
                mx_s, my_s = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                out_x, out_y = mx_s - cent[0], my_s - cent[1]
                if px * out_x + py * out_y < 0:
                    px, py, pz = -px, -py, -pz
                if seg_idx < len(gutter_segment_sections) and gutter_segment_sections[seg_idx]:
                    try:
                        from shapely.ops import unary_union as _uu
                        from shapely.geometry import Polygon as ShapelyPolygon
                        from shapely.geometry import Point as ShapelyPoint
                        polys_seg = [ShapelyPolygon([(float(p[0]), float(p[1])) for p in s.get("bounding_rect") or []])
                                    for s in gutter_segment_sections[seg_idx] if len((s.get("bounding_rect") or [])) >= 3]
                        if polys_seg:
                            uu_seg = _uu(polys_seg)
                            test_d = max(r * 5.0, 20.0)
                            pt_t = ShapelyPoint(mx_s + px * test_d, my_s + py * test_d)
                            if not getattr(uu_seg, "is_empty", True) and uu_seg.contains(pt_t):
                                px, py, pz = -px, -py, -pz
                    except Exception:
                        pass
                perp2_list.append((px, py, pz))

        bx, by, bz = 0.0, 0.0, 0.0
        for px, py, pz in perp2_list:
            bx += px
            by += py
            bz += pz
        bn = math.sqrt(bx * bx + by * by + bz * bz + 1e-12)
        if bn >= 1e-9:
            bx, by, bz = bx / bn, by / bn, bz / bn
            cx_gut = gx + bx * r
            cy_gut = gy + by * r
            cz_gut = gz + bz * r
            if interior_reference_sections:
                try:
                    from shapely.ops import unary_union
                    from shapely.geometry import Polygon as ShapelyPolygon
                    from shapely.geometry import Point as ShapelyPoint
                    polys_ref = []
                    for s in interior_reference_sections:
                        br = s.get("bounding_rect") or []
                        if len(br) >= 3:
                            polys_ref.append(ShapelyPolygon([(float(p[0]), float(p[1])) for p in br]))
                    if polys_ref:
                        uu_ref = unary_union(polys_ref)
                        if not getattr(uu_ref, "is_empty", True):
                            pt = ShapelyPoint(cx_gut, cy_gut)
                            if uu_ref.contains(pt):
                                bx, by, bz = -bx, -by, -bz
                                cx_gut = gx + bx * r
                                cy_gut = gy + by * r
                                cz_gut = gz + bz * r
                except Exception:
                    pass
        else:
            dx_out = gx - cent[0]
            dy_out = gy - cent[1]
            d_out = math.sqrt(dx_out * dx_out + dy_out * dy_out)
            if d_out >= 1e-9:
                cx_gut = gx + r * (dx_out / d_out)
                cy_gut = gy + r * (dy_out / d_out)
            else:
                cx_gut, cy_gut = gx + r, gy
            cz_gut = gz

        for face in get_hemisphere_faces_3d(cx_gut, cy_gut, cz_gut, r, dome_down=True):
            result.append(face)

    return result


def get_vertical_cylinder_faces_3d(
    x: float,
    y: float,
    z_bot: float,
    z_top: float,
    radius: float,
    n_sides: int = 8,
) -> List[Dict[str, Any]]:
    """Cilindru vertical (burlan) de la (x,y,z_bot) la (x,y,z_top)."""
    import math

    if radius <= 0 or z_top <= z_bot or n_sides < 3:
        return []
    result: List[Dict[str, Any]] = []
    pts_bot: List[List[float]] = []
    pts_top: List[List[float]] = []
    for i in range(n_sides):
        ang = 2 * math.pi * i / n_sides
        dx = radius * math.cos(ang)
        dy = radius * math.sin(ang)
        pts_bot.append([x + dx, y + dy, z_bot])
        pts_top.append([x + dx, y + dy, z_top])
    for i in range(n_sides):
        i1 = (i + 1) % n_sides
        face = [pts_bot[i], pts_bot[i1], pts_top[i1], pts_top[i]]
        result.append({"vertices_3d": face})
    result.append({"vertices_3d": pts_bot})
    result.append({"vertices_3d": pts_top})
    return result


def get_hemisphere_faces_3d(
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    *,
    dome_down: bool = True,
    n_theta: int = 12,
    n_phi: int = 6,
) -> List[Dict[str, Any]]:
    """
    Emisferă (jumătate de sferă) centrată la (cx, cy, cz).
    dome_down=True: cupola spre jos, fața plată orizontală sus (z=cz).
    Poziționare ca la burlane: centrul pe fața plată.
    """
    import math

    if radius <= 0 or n_theta < 3 or n_phi < 2:
        return []
    result: List[Dict[str, Any]] = []
    if dome_down:
        phi_min, phi_max = math.pi / 2, math.pi
    else:
        phi_min, phi_max = 0.0, math.pi / 2
    for it in range(n_theta):
        t0 = 2 * math.pi * it / n_theta
        t1 = 2 * math.pi * (it + 1) / n_theta
        for ip in range(n_phi):
            p0 = phi_min + (phi_max - phi_min) * ip / n_phi
            p1 = phi_min + (phi_max - phi_min) * (ip + 1) / n_phi
            x00 = cx + radius * math.sin(p0) * math.cos(t0)
            y00 = cy + radius * math.sin(p0) * math.sin(t0)
            z00 = cz + radius * math.cos(p0)
            x01 = cx + radius * math.sin(p0) * math.cos(t1)
            y01 = cy + radius * math.sin(p0) * math.sin(t1)
            z01 = cz + radius * math.cos(p0)
            x10 = cx + radius * math.sin(p1) * math.cos(t0)
            y10 = cy + radius * math.sin(p1) * math.sin(t0)
            z10 = cz + radius * math.cos(p1)
            x11 = cx + radius * math.sin(p1) * math.cos(t1)
            y11 = cy + radius * math.sin(p1) * math.sin(t1)
            z11 = cz + radius * math.cos(p1)
            if dome_down:
                face = [[x00, y00, z00], [x01, y01, z01], [x11, y11, z11], [x10, y10, z10]]
            else:
                face = [[x00, y00, z00], [x10, y10, z10], [x11, y11, z11], [x01, y01, z01]]
            result.append({"vertices_3d": face})
    flat_pts: List[List[float]] = []
    for it in range(n_theta):
        t = 2 * math.pi * it / n_theta
        r_flat = radius * math.sin(phi_min)
        z_flat = cz + radius * math.cos(phi_min)
        flat_pts.append([cx + r_flat * math.cos(t), cy + r_flat * math.sin(t), z_flat])
    result.append({"vertices_3d": flat_pts})
    return result


def get_gutter_endpoints_3d(
    sections: List[Dict[str, Any]],
    base_z: float,
    overhang_px: float,
    roof_angle_deg: float,
    *,
    exclude_low_sides: Optional[List[str]] = None,
    exclude_low_side: Optional[str] = None,
    include_eaves_only: bool = False,
    pyramid_all_sides: bool = False,
    roof_shift_dz: float = 0.0,
    roof_faces: Optional[List[Dict[str, Any]]] = None,
    eaves_z_lift: float = 0.0,
    tol: float = 8.0,
    sections_include_mask: Optional[List[bool]] = None,
) -> List[Tuple[float, float, float]]:
    """Returnează capetele 3D ale fiecărui segment de streașină (x,y,z)."""
    import math

    if not sections:
        return []
    if not include_eaves_only and not exclude_low_sides and not exclude_low_side and not pyramid_all_sides:
        return []
    _excl = exclude_low_sides
    if _excl is None and exclude_low_side:
        _excl = [exclude_low_side] * len(sections)
    segments = get_gutter_segments_2d(
        sections,
        exclude_low_sides=_excl,
        exclude_low_side=exclude_low_side,
        include_eaves_only=include_eaves_only,
        pyramid_all_sides=pyramid_all_sides,
        tol=tol,
        sections_include_mask=sections_include_mask,
    )
    z_lift = float(eaves_z_lift)

    def _z_at(x: float, y: float) -> float:
        if roof_faces:
            z_plane = _z_on_roof_plane_for_gutter(x, y, roof_faces)
            if z_plane is not None:
                return float(z_plane) - float(roof_shift_dz)
        tan_angle = math.tan(math.radians(roof_angle_deg))
        return base_z - overhang_px * tan_angle - float(roof_shift_dz)

    def _z_on_roof_plane_for_gutter(px: float, py: float, faces: List[Dict[str, Any]]) -> Optional[float]:
        best_z: Optional[float] = None
        best_dist: float = float("inf")
        for f in faces:
            vs = f.get("vertices_3d") or []
            if len(vs) < 3:
                continue
            v0 = [float(vs[0][0]), float(vs[0][1]), float(vs[0][2])]
            v1 = [float(vs[1][0]), float(vs[1][1]), float(vs[1][2])]
            v2 = [float(vs[2][0]), float(vs[2][1]), float(vs[2][2])]
            dx1, dy1, dz1 = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
            dx2, dy2, dz2 = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
            nx = dy1 * dz2 - dz1 * dy2
            ny = dz1 * dx2 - dx1 * dz2
            nz = dx1 * dy2 - dy1 * dx2
            if abs(nz) < 1e-12:
                continue
            z_at = v0[2] - (nx * (px - v0[0]) + ny * (py - v0[1])) / nz
            cx = sum(float(v[0]) for v in vs) / len(vs)
            cy = sum(float(v[1]) for v in vs) / len(vs)
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            if d2 < best_dist:
                best_dist = d2
                best_z = z_at
        return best_z

    out: List[Tuple[float, float, float]] = []
    for (x1, y1), (x2, y2) in segments:
        z1 = _z_at(x1, y1) + z_lift
        z2 = _z_at(x2, y2) + z_lift
        out.append((float(x1), float(y1), z1))
        out.append((float(x2), float(y2), z2))
    return out


def _is_interior_corner(
    prev: Tuple[float, float],
    curr: Tuple[float, float],
    next_p: Tuple[float, float],
    poly: Any,
) -> bool:
    """Colț interior (re-entrant/concav) – return True pentru a-l exclude. Doar colțurile EXTERIOARE primesc cilindri."""
    v1x = curr[0] - prev[0]
    v1y = curr[1] - prev[1]
    v2x = next_p[0] - curr[0]
    v2y = next_p[1] - curr[1]
    cross = v1x * v2y - v1y * v2x
    try:
        is_ccw = getattr(getattr(poly, "exterior", None), "is_ccw", True)
        if is_ccw is None:
            is_ccw = True
        # Colț interior (concav): excludem, nu punem cilindru. Exterior = punem cilindru.
        interior = (cross < 0 and is_ccw) or (cross > 0 and not is_ccw)
        return interior
    except Exception:
        return False  # la incertitudine, nu excludem (punem cilindru)


def _cylinder_gutter_connection_faces(
    cx_cyl: float, cy_cyl: float, cz_cyl: float,
    gx: float, gy: float, gz: float,
    gx1: float, gy1: float, gz1: float,
    gx2: float, gy2: float, gz2: float,
    radius: float,
    n: int = 10,
) -> List[Dict[str, Any]]:
    """Suprafață de conectare: linii de la marginea burlanului la marginea jgheabului, umplute."""
    import math

    ax_cyl = gx - cx_cyl
    ay_cyl = gy - cy_cyl
    az_cyl = gz - cz_cyl
    Lc = math.sqrt(ax_cyl * ax_cyl + ay_cyl * ay_cyl + az_cyl * az_cyl)
    if Lc < 1e-9:
        return []
    ax_cyl, ay_cyl, az_cyl = ax_cyl / Lc, ay_cyl / Lc, az_cyl / Lc
    _px, _py, _pz = 1.0, 0.0, 0.0
    if abs(ax_cyl) < 0.9:
        _px, _py, _pz = 0.0, 1.0, 0.0
    px_c = _py * az_cyl - _pz * ay_cyl
    py_c = _pz * ax_cyl - _px * az_cyl
    pz_c = _px * ay_cyl - _py * ax_cyl
    nc = math.sqrt(px_c * px_c + py_c * py_c + pz_c * pz_c)
    if nc < 1e-9:
        return []
    px_c, py_c, pz_c = px_c / nc, py_c / nc, pz_c / nc
    qx_c = ay_cyl * pz_c - az_cyl * py_c
    qy_c = az_cyl * px_c - ax_cyl * pz_c
    qz_c = ax_cyl * py_c - ay_cyl * px_c

    ax_g = gx2 - gx1
    ay_g = gy2 - gy1
    az_g = gz2 - gz1
    Lg = math.sqrt(ax_g * ax_g + ay_g * ay_g + az_g * az_g)
    if Lg < 1e-9:
        return []
    ax_g, ay_g, az_g = ax_g / Lg, ay_g / Lg, az_g / Lg
    perp2x, perp2y, perp2z = -ay_g, ax_g, 0.0
    n2 = math.sqrt(perp2x * perp2x + perp2y * perp2y + perp2z * perp2z)
    if n2 < 1e-9:
        perp2x, perp2y, perp2z = 1.0, 0.0, 0.0
    else:
        perp2x, perp2y, perp2z = perp2x / n2, perp2y / n2, perp2z / n2
    perp1x = ay_g * perp2z - az_g * perp2y
    perp1y = az_g * perp2x - ax_g * perp2z
    perp1z = ax_g * perp2y - ay_g * perp2x
    if perp1z > 0:
        perp1x, perp1y, perp1z = -perp1x, -perp1y, -perp1z
    n1 = math.sqrt(perp1x * perp1x + perp1y * perp1y + perp1z * perp1z)
    if n1 >= 1e-9:
        perp1x, perp1y, perp1z = perp1x / n1, perp1y / n1, perp1z / n1
    co_x, co_y, co_z = perp2x * radius, perp2y * radius, perp2z * radius

    cyl_pts: List[List[float]] = []
    gut_pts: List[List[float]] = []
    for i in range(n + 1):
        ang = math.pi * i / n
        dx_c = radius * (px_c * math.cos(ang) + qx_c * math.sin(ang))
        dy_c = radius * (py_c * math.cos(ang) + qy_c * math.sin(ang))
        dz_c = radius * (pz_c * math.cos(ang) + qz_c * math.sin(ang))
        cyl_pts.append([gx + dx_c, gy + dy_c, gz + dz_c])
        dx_g = co_x + perp2x * math.cos(ang) * radius + perp1x * math.sin(ang) * radius
        dy_g = co_y + perp2y * math.cos(ang) * radius + perp1y * math.sin(ang) * radius
        dz_g = co_z + perp2z * math.cos(ang) * radius + perp1z * math.sin(ang) * radius
        gut_pts.append([gx + dx_g, gy + dy_g, gz + dz_g])

    result: List[Dict[str, Any]] = []
    for i in range(n):
        result.append({"vertices_3d": [cyl_pts[i], cyl_pts[i + 1], gut_pts[i + 1], gut_pts[i]]})
    return result


def _horizontal_cylinder_connector_faces(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
    radius: float,
    n_sides: int = 12,
) -> List[Dict[str, Any]]:
    """Cilindru cu capete orizontale: cercuri în plan xy la z1 și z2, centre (x1,y1) și (x2,y2).
    Punctele de pe capete se conectează perfect cu cilindrii verticali lipiți."""
    import math

    if radius <= 0 or n_sides < 3:
        return []
    pts_bot: List[List[float]] = []
    pts_top: List[List[float]] = []
    for i in range(n_sides):
        ang = 2 * math.pi * i / n_sides
        dx = radius * math.cos(ang)
        dy = radius * math.sin(ang)
        pts_bot.append([x1 + dx, y1 + dy, z1])
        pts_top.append([x2 + dx, y2 + dy, z2])
    result: List[Dict[str, Any]] = []
    for i in range(n_sides):
        i1 = (i + 1) % n_sides
        result.append({"vertices_3d": [pts_bot[i], pts_bot[i1], pts_top[i1], pts_top[i]]})
    return result


def get_tilted_cylinder_faces_3d(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
    radius: float,
    n_sides: int = 8,
) -> List[Dict[str, Any]]:
    """Cilindru înclinat de la (x1,y1,z1) la (x2,y2,z2)."""
    import math

    if radius <= 0 or n_sides < 3:
        return []
    ax = x2 - x1
    ay = y2 - y1
    az = z2 - z1
    L = math.sqrt(ax * ax + ay * ay + az * az)
    if L < 1e-9:
        return []
    ax, ay, az = ax / L, ay / L, az / L
    perp_x, perp_y, perp_z = 1.0, 0.0, 0.0
    if abs(ax) < 0.9:
        perp_x, perp_y, perp_z = 0.0, 1.0, 0.0
    px = perp_y * az - perp_z * ay
    py = perp_z * ax - perp_x * az
    pz = perp_x * ay - perp_y * ax
    n = math.sqrt(px * px + py * py + pz * pz)
    if n < 1e-9:
        return []
    px, py, pz = px / n, py / n, pz / n
    qx = ay * pz - az * py
    qy = az * px - ax * pz
    qz = ax * py - ay * px
    pts1: List[List[float]] = []
    pts2: List[List[float]] = []
    for i in range(n_sides):
        ang = 2 * math.pi * i / n_sides
        dx = radius * (px * math.cos(ang) + qx * math.sin(ang))
        dy = radius * (py * math.cos(ang) + qy * math.sin(ang))
        dz = radius * (pz * math.cos(ang) + qz * math.sin(ang))
        pts1.append([x1 + dx, y1 + dy, z1 + dz])
        pts2.append([x2 + dx, y2 + dy, z2 + dz])
    result: List[Dict[str, Any]] = []
    for i in range(n_sides):
        i1 = (i + 1) % n_sides
        result.append({"vertices_3d": [pts1[i], pts1[i1], pts2[i1], pts2[i]]})
    result.append({"vertices_3d": pts1})
    result.append({"vertices_3d": pts2})
    return result


def _tilted_cylinder_connection_faces(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
    radius: float,
    n_sides: int = 8,
) -> List[Dict[str, Any]]:
    """Linii suplimentare: conectează capetele cilindrului înclinat cu cercurile orizontale
    ale cilindrilor vecini (burlan vertical la (x1,y1,z1), cilindru mov la (x2,y2,z2))."""
    import math

    if radius <= 0 or n_sides < 3:
        return []
    ax = x2 - x1
    ay = y2 - y1
    az = z2 - z1
    L = math.sqrt(ax * ax + ay * ay + az * az)
    if L < 1e-9:
        return []
    ax, ay, az = ax / L, ay / L, az / L
    perp_x, perp_y, perp_z = 1.0, 0.0, 0.0
    if abs(ax) < 0.9:
        perp_x, perp_y, perp_z = 0.0, 1.0, 0.0
    px = perp_y * az - perp_z * ay
    py = perp_z * ax - perp_x * az
    pz = perp_x * ay - perp_y * ax
    nn = math.sqrt(px * px + py * py + pz * pz)
    if nn < 1e-9:
        return []
    px, py, pz = px / nn, py / nn, pz / nn
    qx = ay * pz - az * py
    qy = az * px - ax * pz
    qz = ax * py - ay * px
    pts_tilt1: List[List[float]] = []
    pts_tilt2: List[List[float]] = []
    pts_horiz1: List[List[float]] = []
    pts_horiz2: List[List[float]] = []
    for i in range(n_sides):
        ang = 2 * math.pi * i / n_sides
        dx_t = radius * (px * math.cos(ang) + qx * math.sin(ang))
        dy_t = radius * (py * math.cos(ang) + qy * math.sin(ang))
        dz_t = radius * (pz * math.cos(ang) + qz * math.sin(ang))
        pts_tilt1.append([x1 + dx_t, y1 + dy_t, z1 + dz_t])
        pts_tilt2.append([x2 + dx_t, y2 + dy_t, z2 + dz_t])
        dx_h = radius * math.cos(ang)
        dy_h = radius * math.sin(ang)
        pts_horiz1.append([x1 + dx_h, y1 + dy_h, z1])
        pts_horiz2.append([x2 + dx_h, y2 + dy_h, z2])
    result: List[Dict[str, Any]] = []
    for i in range(n_sides):
        i1 = (i + 1) % n_sides
        result.append({"vertices_3d": [pts_tilt1[i], pts_tilt1[i1], pts_horiz1[i1], pts_horiz1[i]]})
        result.append({"vertices_3d": [pts_tilt2[i], pts_horiz2[i], pts_horiz2[i1], pts_tilt2[i1]]})
    return result


def get_downspout_faces_for_floors(
    floors_payload: List[Tuple[Any, Any, Dict[str, Any], Tuple[float, float]]],
    wall_height: float,
    *,
    radius_ratio: float = 0.04,
    cylinder_radius: Optional[float] = None,
    terminal_height_ratio: float = 0.70,
    corner_tol: float = 15.0,
    gutter_endpoints: Optional[List[Tuple[float, float, float]]] = None,
    gutter_sections: Optional[List[Dict[str, Any]]] = None,
    gutter_segment_sections: Optional[List[List[Dict[str, Any]]]] = None,
    single_downspout: bool = False,
    return_used_endpoints: bool = False,
    return_downspout_centerlines: bool = False,
    debug_positions_out: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], List[Tuple[float, float, float]]] | Tuple[List[Dict[str, Any]], List[Tuple[float, float, float]], Dict[str, List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]]]:
    """
    Un singur burlan vertical per streașină. 4 colțuri → 2 burlane.
    - cylinder_radius: dacă dat, folosește același diametru ca la streașină.
    - Fiecare streașină primește exact un burlan (la colțul cel mai apropiat).
    - Nu conectează mai mulți burlani la aceeași streașină.
    - Dacă nu există streașini de conectat, nu adaugă niciun burlan.
    """
    import math

    if not floors_payload or wall_height <= 0:
        return []
    if not gutter_endpoints or len(gutter_endpoints) < 2:
        return []
    num_floors = len(floors_payload)
    radius = (
        float(cylinder_radius)
        if cylinder_radius is not None and cylinder_radius > 0
        else max(3.0, wall_height * radius_ratio)
    )
    result: List[Dict[str, Any]] = []
    used_endpoints: List[Tuple[float, float, float]] = []
    downspout_cl: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
    connection_cl: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []

    def _corners(poly: Any) -> List[Tuple[float, float]]:
        coords = list(poly.exterior.coords) if hasattr(poly, "exterior") else []
        if len(coords) < 2:
            return []
        return [(float(c[0]), float(c[1])) for c in (coords[:-1] if len(coords) > 1 else coords)]

    def _centroid(poly: Any) -> Tuple[float, float]:
        if hasattr(poly, "centroid"):
            c = poly.centroid
            return (float(c.x), float(c.y))
        pts = _corners(poly)
        if not pts:
            return (0.0, 0.0)
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    def _matches(c1: Tuple[float, float], c2: Tuple[float, float]) -> bool:
        return (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 <= corner_tol * corner_tol

    corners_per_floor: List[List[Tuple[float, float]]] = []
    for _p, floor_poly, _rr, _off in floors_payload:
        corners_per_floor.append(_corners(floor_poly))

    # Segment de streașină = pereche de capete consecutive
    gutter_segments: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
    for i in range(0, len(gutter_endpoints) - 1, 2):
        p1 = gutter_endpoints[i]
        p2 = gutter_endpoints[i + 1]
        gutter_segments.append((p1, p2))

    def _is_exterior_on_floor(cx: float, cy: float, floor_idx: int) -> bool:
        corners = corners_per_floor[floor_idx]
        if not corners:
            return False
        n_c = len(corners)
        for j, c in enumerate(corners):
            if not _matches((cx, cy), c):
                continue
            prev_c = corners[(j - 1) % n_c]
            next_c = corners[(j + 1) % n_c]
            return not _is_interior_corner(prev_c, c, next_c, floors_payload[floor_idx][1])
        return False

    # Colțuri exterioare eligibile: exterior pe TOATE etajele unde apar, cu suport de jos
    def _eligible_corners() -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        seen: List[Tuple[float, float]] = []
        for floor_idx, corners in enumerate(corners_per_floor):
            if not corners:
                continue
            n_c = len(corners)
            for j, (cx, cy) in enumerate(corners):
                if floor_idx > 0:
                    has_below = any(_matches((cx, cy), pc) for pc in corners_per_floor[floor_idx - 1])
                    if not has_below:
                        continue
                if any(_matches((cx, cy), s) for s in seen):
                    continue
                interior_on_any = False
                for fi in range(num_floors):
                    if not any(_matches((cx, cy), c) for c in corners_per_floor[fi]):
                        continue
                    if not _is_exterior_on_floor(cx, cy, fi):
                        interior_on_any = True
                        break
                if interior_on_any:
                    continue
                seen.append((cx, cy))
                out.append((cx, cy))
        return out

    eligible = _eligible_corners()
    if not eligible:
        return result

    # Centroid pentru orientarea perp2: identic cu get_gutter_faces_3d
    # gutter_segment_sections[seg_idx] = secțiunile care au produs acel segment (aliniere corectă)
    def _centroid_for_perp2(seg_idx: int) -> Tuple[float, float]:
        secs_to_use: Optional[List[Dict[str, Any]]] = None
        if gutter_segment_sections and 0 <= seg_idx < len(gutter_segment_sections):
            secs_to_use = gutter_segment_sections[seg_idx]
        elif gutter_sections:
            secs_to_use = gutter_sections
        if secs_to_use:
            try:
                from shapely.ops import unary_union
                from shapely.geometry import Polygon as ShapelyPolygon
                polys = []
                for s in secs_to_use:
                    br = s.get("bounding_rect") or []
                    if len(br) >= 3:
                        polys.append(ShapelyPolygon([(float(p[0]), float(p[1])) for p in br]))
                if polys:
                    uu = unary_union(polys)
                    return (float(uu.centroid.x), float(uu.centroid.y))
            except Exception:
                pass
        return _centroid(floors_payload[0][1])

    assigned_corners: List[Tuple[float, float]] = []
    gray = "#6B7280"

    for seg_idx, ((gx1, gy1, gz1), (gx2, gy2, gz2)) in enumerate(gutter_segments):
        # Găsim colțul cel mai apropiat de fiecare capăt de streașină (capătul e PE acel colț)
        def _nearest_corner(gx: float, gy: float) -> Tuple[Optional[Tuple[float, float]], float]:
            best = None
            best_d2 = float("inf")
            for (cx, cy) in eligible:
                d2 = (cx - gx) ** 2 + (cy - gy) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = (cx, cy)
            return (best, best_d2)

        c1, d1_2 = _nearest_corner(gx1, gy1)
        c2, d2_2 = _nearest_corner(gx2, gy2)
        if c1 is None and c2 is None:
            continue
        # Alegem colțul la care e capătul streașinii (capătul pe acel colț al casei), nevăzut încă
        cands: List[Tuple[float, Tuple[float, float], Tuple[float, float, float]]] = []
        if c1 is not None:
            cands.append((d1_2, c1, (gx1, gy1, gz1)))
        if c2 is not None:
            if c1 is not None and _matches(c1, c2):
                if d2_2 < d1_2:
                    cands[0] = (d2_2, c2, (gx2, gy2, gz2))
            else:
                cands.append((d2_2, c2, (gx2, gy2, gz2)))
        cands.sort(key=lambda t: t[0])
        best_corner = None
        best_endpoint = None
        for _d2, corner, endpoint in cands:
            if any(_matches(corner, ac) for ac in assigned_corners):
                continue
            best_corner = corner
            best_endpoint = endpoint
            break
        if best_corner is None or best_endpoint is None:
            continue
        assigned_corners.append(best_corner)
        cx, cy = best_corner
        gx, gy, gz = best_endpoint
        if return_used_endpoints:
            used_endpoints.append(best_endpoint)
        cent = _centroid_for_perp2(seg_idx)

        # Centrul stub-ului = centrul jgheabului tăiat (endpoint + perp2*r). Folosim perp2 al
        # SEGMENTULUI CURENT (gx1,gy1)-(gx2,gy2), nu media la colț – identic cu get_gutter_faces_3d
        ax_s = gx2 - gx1
        ay_s = gy2 - gy1
        az_s = gz2 - gz1
        Ls = math.sqrt(ax_s * ax_s + ay_s * ay_s + az_s * az_s)
        perp2x, perp2y, perp2z = 0.0, 0.0, 0.0
        cx_gut, cy_gut, cz_gut = gx, gy, gz
        if Ls >= 1e-9:
            ax_s, ay_s, az_s = ax_s / Ls, ay_s / Ls, az_s / Ls
            perp2x, perp2y, perp2z = -ay_s, ax_s, 0.0
            ns = math.sqrt(perp2x * perp2x + perp2y * perp2y + perp2z * perp2z)
            if ns >= 1e-9:
                perp2x, perp2y, perp2z = perp2x / ns, perp2y / ns, perp2z / ns
                mx_s, my_s = (gx1 + gx2) / 2, (gy1 + gy2) / 2
                out_x = mx_s - cent[0]
                out_y = my_s - cent[1]
                if perp2x * out_x + perp2y * out_y < 0:
                    perp2x, perp2y, perp2z = -perp2x, -perp2y, -perp2z
                # Nu facem flip pentru containment – centrul stub-ului trebuie să coincidă exact
                # cu centrul capătului jgheabului (cilindru tăiat), altfel se dezechilibrează.
                cx_gut = gx + perp2x * radius
                cy_gut = gy + perp2y * radius
                cz_gut = gz + perp2z * radius
        else:
            ax_g = gx2 - gx1
            ay_g = gy2 - gy1
            az_g = gz2 - gz1
            Lg = math.sqrt(ax_g * ax_g + ay_g * ay_g + az_g * az_g)
            if Lg >= 1e-9:
                ax_g, ay_g, az_g = ax_g / Lg, ay_g / Lg, az_g / Lg
                perp2x, perp2y, perp2z = -ay_g, ax_g, 0.0
                n2 = math.sqrt(perp2x * perp2x + perp2y * perp2y + perp2z * perp2z)
                if n2 >= 1e-9:
                    perp2x, perp2y, perp2z = perp2x / n2, perp2y / n2, perp2z / n2
                    mx_g, my_g = (gx1 + gx2) / 2, (gy1 + gy2) / 2
                    out_x = mx_g - cent[0]
                    out_y = my_g - cent[1]
                    if perp2x * out_x + perp2y * out_y < 0:
                        perp2x, perp2y, perp2z = -perp2x, -perp2y, -perp2z
                    # Nu facem flip pentru containment – centrul stub-ului trebuie să coincidă exact
                    # cu centrul capătului jgheabului (cilindru tăiat), altfel se dezechilibrează.
                    cx_gut = gx + perp2x * radius
                    cy_gut = gy + perp2y * radius
                    cz_gut = gz + perp2z * radius

        # Burlan vertical: poziție din colț (neschimbat)
        dx = cx - cent[0]
        dy = cy - cent[1]
        d = math.sqrt(dx * dx + dy * dy)
        if d < 1e-9:
            cx_cyl, cy_cyl = cx + radius, cy
        else:
            cx_cyl = cx + radius * (dx / d)
            cy_cyl = cy + radius * (dy / d)

        for floor_idx, corners in enumerate(corners_per_floor):
            if not any(_matches((cx, cy), c) for c in corners):
                continue
            if not _is_exterior_on_floor(cx, cy, floor_idx):
                continue
            if floor_idx > 0:
                has_below = any(_matches((cx, cy), pc) for pc in corners_per_floor[floor_idx - 1])
                if not has_below:
                    continue
            z0 = floor_idx * wall_height
            has_above = (
                floor_idx + 1 < num_floors
                and any(_matches((cx, cy), pc) for pc in corners_per_floor[floor_idx + 1])
            )
            h_ratio = 1.0 if has_above else terminal_height_ratio
            z_top = z0 + wall_height * h_ratio
            for face in get_vertical_cylinder_faces_3d(cx_cyl, cy_cyl, z0, z_top, radius):
                result.append(dict(face, color=gray))
            if return_downspout_centerlines:
                downspout_cl.append(((cx_cyl, cy_cyl, z0), (cx_cyl, cy_cyl, z_top)))
            if not has_above:
                # cx_gut, cy_gut, perp2 deja calculate sus – aliniere cu jgheabul tăiat
                stub_h = radius  # înălțime = jumătate din diametru
                z_bot = cz_gut - stub_h - 2 * stub_h  # prelungit în jos cu 2× înălțimea
                z_top_stub = cz_gut  # capul cilindrului la jgheab
                for face in get_vertical_cylinder_faces_3d(cx_gut, cy_gut, z_bot, z_top_stub, radius):
                    result.append(dict(face, color=gray))
                for face in get_tilted_cylinder_faces_3d(
                    cx_cyl, cy_cyl, z_top, cx_gut, cy_gut, z_bot, radius
                ):
                    result.append(dict(face, color=gray))
                for face in _tilted_cylinder_connection_faces(
                    cx_cyl, cy_cyl, z_top, cx_gut, cy_gut, z_bot, radius
                ):
                    result.append(dict(face, color=gray))
                if return_downspout_centerlines:
                    connection_cl.append(((cx_cyl, cy_cyl, z_top), (cx_gut, cy_gut, cz_gut)))
                if debug_positions_out is not None:
                    debug_positions_out.append({
                        "floor_idx": floor_idx,
                        "cx_cyl": cx_cyl, "cy_cyl": cy_cyl,
                        "cx_gut": cx_gut, "cy_gut": cy_gut,
                        "gx": gx, "gy": gy, "cx": cx, "cy": cy,
                        "perp2x": perp2x, "perp2y": perp2y,
                        "radius": radius, "has_above": False,
                    })
            if single_downspout:
                break
    if return_downspout_centerlines:
        return (result, used_endpoints, {"downspout": downspout_cl, "connection": connection_cl})
    if return_used_endpoints:
        return (result, used_endpoints)
    return result


def get_downspout_faces_pyramid(
    floors_payload: List[Tuple[Any, Any, Dict[str, Any], Tuple[float, float]]],
    wall_height: float,
    *,
    cylinder_radius: Optional[float] = None,
    gutter_endpoints: Optional[List[Tuple[float, float, float]]] = None,
    gutter_segment_sections: Optional[List[List[Dict[str, Any]]]] = None,
    terminal_height_ratio: float = 0.70,
    corner_tol: float = 15.0,
    return_used_gutter_endpoints: bool = False,
    return_downspout_centerlines: bool = False,
) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], List[Tuple[float, float, float]]] | Tuple[List[Dict[str, Any]], List[Tuple[float, float, float]], Dict[str, List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]]]:
    """
    Burlani pentru acoperiș piramidă: număr de coloane = număr de dreptunghiuri (secțiuni).
    Poziționare identică cu celelalte acoperișuri: colț perete (floor polygon) + offset spre exterior.
    Cilindru terminal = cel fără altul deasupra (mai scurt).
    return_used_gutter_endpoints: dacă True, returnează (result, used_endpoints) pentru capete fără closure.
    """
    import math

    gray = "#6B7280"
    if not floors_payload or wall_height <= 0:
        return []

    num_floors = len(floors_payload)
    radius = (
        float(cylinder_radius)
        if cylinder_radius is not None and cylinder_radius > 0
        else max(3.0, wall_height * 0.04)
    )

    def _corners(poly: Any) -> List[Tuple[float, float]]:
        coords = list(poly.exterior.coords) if hasattr(poly, "exterior") else []
        if len(coords) < 2:
            return []
        return [(float(c[0]), float(c[1])) for c in (coords[:-1] if len(coords) > 1 else coords)]

    def _centroid(poly: Any) -> Tuple[float, float]:
        if hasattr(poly, "centroid"):
            c = poly.centroid
            return (float(c.x), float(c.y))
        pts = _corners(poly)
        if not pts:
            return (0.0, 0.0)
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    def _matches(c1: Tuple[float, float], c2: Tuple[float, float]) -> bool:
        return (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 <= corner_tol * corner_tol

    corners_per_floor: List[List[Tuple[float, float]]] = []
    for _p, floor_poly, _rr, _off in floors_payload:
        corners_per_floor.append(_corners(floor_poly))

    def _is_exterior_on_floor(cx: float, cy: float, floor_idx: int) -> bool:
        corners = corners_per_floor[floor_idx]
        if not corners:
            return False
        n_c = len(corners)
        for j, c in enumerate(corners):
            if not _matches((cx, cy), c):
                continue
            prev_c = corners[(j - 1) % n_c]
            next_c = corners[(j + 1) % n_c]
            return not _is_interior_corner(prev_c, c, next_c, floors_payload[floor_idx][1])
        return False

    eligible: List[Tuple[float, float]] = []
    seen: List[Tuple[float, float]] = []
    for floor_idx, corners in enumerate(corners_per_floor):
        if not corners:
            continue
        for (cx, cy) in corners:
            if floor_idx > 0:
                has_below = any(_matches((cx, cy), pc) for pc in corners_per_floor[floor_idx - 1])
                if not has_below:
                    continue
            if any(_matches((cx, cy), s) for s in seen):
                continue
            interior_on_any = False
            for fi in range(num_floors):
                if not any(_matches((cx, cy), c) for c in corners_per_floor[fi]):
                    continue
                if not _is_exterior_on_floor(cx, cy, fi):
                    interior_on_any = True
                    break
            if interior_on_any:
                continue
            seen.append((cx, cy))
            eligible.append((cx, cy))

    if not eligible:
        return []

    gutter_segments: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
    if gutter_endpoints:
        for i in range(0, len(gutter_endpoints) - 1, 2):
            gutter_segments.append((gutter_endpoints[i], gutter_endpoints[i + 1]))

    def _rect_key(sec: Dict[str, Any]) -> Tuple[float, ...]:
        br = sec.get("bounding_rect") or []
        if len(br) < 3:
            return ()
        return tuple(sorted((round(float(p[0]), 1), round(float(p[1]), 1)) for p in br))

    def _centroid_for_section(sec: Dict[str, Any]) -> Tuple[float, float]:
        br = sec.get("bounding_rect") or []
        if len(br) >= 3:
            try:
                from shapely.geometry import Polygon as ShapelyPolygon
                from shapely.ops import unary_union

                poly = ShapelyPolygon([(float(p[0]), float(p[1])) for p in br])
                return (float(poly.centroid.x), float(poly.centroid.y))
            except Exception:
                pass
        pts = [(float(p[0]), float(p[1])) for p in br[:4]] if br else []
        if pts:
            return (sum(c[0] for c in pts) / len(pts), sum(c[1] for c in pts) / len(pts))
        return _centroid(floors_payload[0][1])

    seen_rect: set = set()
    unique_sections: List[Dict[str, Any]] = []
    if gutter_segment_sections:
        for secs in gutter_segment_sections:
            for sec in secs or []:
                k = _rect_key(sec)
                if k and k not in seen_rect:
                    seen_rect.add(k)
                    unique_sections.append(sec)

    assigned_corners: List[Tuple[float, float]] = []
    result: List[Dict[str, Any]] = []
    used_gutter_endpoints: List[Tuple[float, float, float]] = []
    downspout_cl: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
    connection_cl: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []

    for sec in unique_sections:
        cent = _centroid_for_section(sec)
        ref_pts_3d: List[Tuple[float, float, float]] = []
        if gutter_segment_sections and gutter_endpoints:
            for seg_idx, secs in enumerate(gutter_segment_sections):
                if not secs:
                    continue
                for s in secs:
                    if _rect_key(s) == _rect_key(sec):
                        if seg_idx * 2 + 1 < len(gutter_endpoints):
                            p1, p2 = gutter_endpoints[seg_idx * 2], gutter_endpoints[seg_idx * 2 + 1]
                            ref_pts_3d.append(p1)
                            ref_pts_3d.append(p2)
                        break
        if not ref_pts_3d:
            br = sec.get("bounding_rect") or []
            ref_pts_3d = [(float(p[0]), float(p[1]), (num_floors - 0.5) * wall_height) for p in br[:4]]

        def _nearest_eligible(gx: float, gy: float) -> Tuple[Optional[Tuple[float, float]], float]:
            best = None
            best_d2 = float("inf")
            for (cx, cy) in eligible:
                d2 = (cx - gx) ** 2 + (cy - gy) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = (cx, cy)
            return (best, best_d2)

        best_corner = None
        best_d2 = float("inf")
        best_gutter_ep: Optional[Tuple[float, float, float]] = None
        best_seg: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None
        if gutter_segment_sections and gutter_endpoints:
            for seg_idx, secs in enumerate(gutter_segment_sections):
                if not secs or seg_idx * 2 + 1 >= len(gutter_endpoints):
                    continue
                for s in secs:
                    if _rect_key(s) != _rect_key(sec):
                        continue
                    p1, p2 = gutter_endpoints[seg_idx * 2], gutter_endpoints[seg_idx * 2 + 1]
                    for ep in (p1, p2):
                        gx, gy, gz = ep[0], ep[1], ep[2]
                        c, d2 = _nearest_eligible(gx, gy)
                        if c is not None and d2 < best_d2 and not any(_matches(c, ac) for ac in assigned_corners):
                            best_d2 = d2
                            best_corner = c
                            best_gutter_ep = (gx, gy, gz)
                            best_seg = (p1, p2)
                    break
        if best_corner is None:
            for (cx, cy) in eligible:
                if any(_matches((cx, cy), ac) for ac in assigned_corners):
                    continue
                best_corner = (cx, cy)
                break
        if best_corner is None:
            continue

        cx, cy = best_corner
        assigned_corners.append(best_corner)
        dx = cx - cent[0]
        dy = cy - cent[1]
        d = math.sqrt(dx * dx + dy * dy)
        if d < 1e-9:
            cx_cyl, cy_cyl = cx + radius, cy
        else:
            cx_cyl = cx + radius * (dx / d)
            cy_cyl = cy + radius * (dy / d)

        gx, gy, gz = (0.0, 0.0, 0.0)
        if best_gutter_ep:
            gx, gy, gz = best_gutter_ep
        gx1, gy1, gz1 = gx, gy, gz
        gx2, gy2, gz2 = gx, gy, gz
        if best_seg:
            gx1, gy1, gz1 = best_seg[0][0], best_seg[0][1], best_seg[0][2]
            gx2, gy2, gz2 = best_seg[1][0], best_seg[1][1], best_seg[1][2]
        # Pentru piramidă: la intersecția capetelor (mijloace) celor două jgheaburi tăiate,
        # pe marginea acestora – bisectorul direcțiilor perp2 ale celor două segmente, la distanță radius
        perp2_list: List[Tuple[float, float, float]] = []
        if gutter_segment_sections and gutter_endpoints:
            tol_sq = (15.0) ** 2
            for seg_idx, secs in enumerate(gutter_segment_sections):
                if not secs or seg_idx * 2 + 1 >= len(gutter_endpoints):
                    continue
                p1, p2 = gutter_endpoints[seg_idx * 2], gutter_endpoints[seg_idx * 2 + 1]
                for ep in (p1, p2):
                    if (ep[0] - gx) ** 2 + (ep[1] - gy) ** 2 > tol_sq:
                        continue
                    ax_s = p2[0] - p1[0]
                    ay_s = p2[1] - p1[1]
                    Ls = math.sqrt(ax_s * ax_s + ay_s * ay_s + 1e-12)
                    if Ls < 1e-9:
                        continue
                    ax_s, ay_s = ax_s / Ls, ay_s / Ls
                    px, py, pz = -ay_s, ax_s, 0.0
                    ns = math.sqrt(px * px + py * py + 1e-12)
                    if ns >= 1e-9:
                        px, py, pz = px / ns, py / ns, pz / ns
                        mx_s, my_s = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                        out_x, out_y = mx_s - cent[0], my_s - cent[1]
                        if px * out_x + py * out_y < 0:
                            px, py, pz = -px, -py, -pz
                        perp2_list.append((px, py, pz))
                    break
        bx, by, bz = 0.0, 0.0, 0.0
        for px, py, pz in perp2_list:
            bx += px
            by += py
            bz += pz
        bn = math.sqrt(bx * bx + by * by + bz * bz + 1e-12)
        if bn >= 1e-9:
            bx, by, bz = bx / bn, by / bn, bz / bn
            cx_gut = gx + bx * radius
            cy_gut = gy + by * radius
            cz_gut = gz + bz * radius
        else:
            dx_out = gx - cent[0]
            dy_out = gy - cent[1]
            d_out = math.sqrt(dx_out * dx_out + dy_out * dy_out)
            if d_out >= 1e-9:
                cx_gut = gx + radius * (dx_out / d_out)
                cy_gut = gy + radius * (dy_out / d_out)
            else:
                cx_gut, cy_gut = gx + radius, gy
            cz_gut = gz

        for floor_idx in range(num_floors):
            if not any(_matches((cx, cy), c) for c in corners_per_floor[floor_idx]):
                continue
            if not _is_exterior_on_floor(cx, cy, floor_idx):
                continue
            if floor_idx > 0:
                has_below = any(_matches((cx, cy), pc) for pc in corners_per_floor[floor_idx - 1])
                if not has_below:
                    continue
            z0 = floor_idx * wall_height
            has_above = (
                floor_idx + 1 < num_floors
                and any(_matches((cx, cy), pc) for pc in corners_per_floor[floor_idx + 1])
            )
            h_ratio = 1.0 if has_above else terminal_height_ratio
            z_top = z0 + wall_height * h_ratio
            for face in get_vertical_cylinder_faces_3d(cx_cyl, cy_cyl, z0, z_top, radius):
                result.append(dict(face, color=gray))
            if return_downspout_centerlines:
                downspout_cl.append(((cx_cyl, cy_cyl, z0), (cx_cyl, cy_cyl, z_top)))
            if not has_above:
                stub_h = radius
                z_bot_stub = cz_gut - stub_h - 2 * stub_h
                z_top_stub = cz_gut
                for face in get_vertical_cylinder_faces_3d(cx_gut, cy_gut, z_bot_stub, z_top_stub, radius):
                    result.append(dict(face, color=gray))
                for face in get_tilted_cylinder_faces_3d(
                    cx_cyl, cy_cyl, z_top, cx_gut, cy_gut, z_bot_stub, radius
                ):
                    result.append(dict(face, color=gray))
                for face in _tilted_cylinder_connection_faces(
                    cx_cyl, cy_cyl, z_top, cx_gut, cy_gut, z_bot_stub, radius
                ):
                    result.append(dict(face, color=gray))
                if return_downspout_centerlines:
                    connection_cl.append(((cx_cyl, cy_cyl, z_top), (cx_gut, cy_gut, cz_gut)))
                if return_used_gutter_endpoints:
                    used_gutter_endpoints.append((cx_gut, cy_gut, cz_gut))

    if return_downspout_centerlines:
        return (result, used_gutter_endpoints, {"downspout": downspout_cl, "connection": connection_cl})
    if return_used_gutter_endpoints:
        return (result, used_gutter_endpoints)
    return result


def get_corner_cylinder_faces_pyramid(
    floors_payload: List[Tuple[Any, Any, Dict[str, Any], Tuple[float, float]]],
    wall_height: float,
    *,
    cylinder_radius: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Cilindri verticali la colțurile EXTERIOARE ale pereților, strict pentru acoperiș piramidă.
    - Un cilindru per colț exterior (nu colț interior/concav).
    - La etajele secundare: se generează și pentru etaj și pentru parter.
    """
    gray = "#6B7280"
    if not floors_payload or wall_height <= 0:
        return []

    num_floors = len(floors_payload)
    radius = (
        float(cylinder_radius)
        if cylinder_radius is not None and cylinder_radius > 0
        else max(3.0, wall_height * 0.04)
    )

    def _corners(poly: Any) -> List[Tuple[float, float]]:
        coords = list(poly.exterior.coords) if hasattr(poly, "exterior") else []
        if len(coords) < 4:
            return []
        return [(float(c[0]), float(c[1])) for c in coords[:-1]]

    result: List[Dict[str, Any]] = []
    corners_per_floor: List[List[Tuple[float, float]]] = []
    for _p, floor_poly, _rr, _off in floors_payload:
        corners_per_floor.append(_corners(floor_poly))

    tol = 15.0

    def _key(cx: float, cy: float, z0: float, z1: float) -> Tuple[int, int, int, int]:
        return (int(round(cx / tol)), int(round(cy / tol)), int(round(z0 / tol)), int(round(z1 / tol)))

    drawn: set = set()

    for floor_idx in range(num_floors):
        corners = corners_per_floor[floor_idx]
        if not corners:
            continue
        z0 = floor_idx * wall_height
        z1 = (floor_idx + 1) * wall_height
        floor_poly = floors_payload[floor_idx][1]
        n_c = len(corners)

        for j, (cx, cy) in enumerate(corners):
            if _is_interior_corner(
                corners[(j - 1) % n_c], (cx, cy), corners[(j + 1) % n_c], floor_poly
            ):
                continue
            k = _key(cx, cy, z0, z1)
            if k in drawn:
                continue
            drawn.add(k)
            for face in get_vertical_cylinder_faces_3d(cx, cy, z0, z1, radius):
                result.append(dict(face, color=gray))

        if floor_idx > 0:
            parter_corners = corners_per_floor[0]
            parter_poly = floors_payload[0][1]
            if parter_corners:
                z_parter_0, z_parter_1 = 0.0, wall_height
                n_p = len(parter_corners)
                for j, (cx, cy) in enumerate(parter_corners):
                    if _is_interior_corner(
                        parter_corners[(j - 1) % n_p], (cx, cy), parter_corners[(j + 1) % n_p], parter_poly
                    ):
                        continue
                    k = _key(cx, cy, z_parter_0, z_parter_1)
                    if k in drawn:
                        continue
                    drawn.add(k)
                    for face in get_vertical_cylinder_faces_3d(cx, cy, z_parter_0, z_parter_1, radius):
                        result.append(dict(face, color=gray))

    return result


def get_cylinder_positions_debug(
    floors_payload: List[Tuple[Any, Any, Dict[str, Any], Tuple[float, float]]],
    gutter_endpoints: List[Tuple[float, float, float]],
    *,
    cylinder_radius: Optional[float] = None,
    wall_height: float = 300.0,
    corner_tol: float = 15.0,
) -> List[Dict[str, Any]]:
    """
    Returnează pozițiile cilindrilor (burlan vertical + cilindru la jgheab) pentru debug 2D.
    Apelează get_downspout_faces_for_floors cu debug_positions_out.
    """
    debug_out: List[Dict[str, Any]] = []
    get_downspout_faces_for_floors(
        floors_payload,
        wall_height,
        cylinder_radius=cylinder_radius,
        corner_tol=corner_tol,
        gutter_endpoints=gutter_endpoints,
        debug_positions_out=debug_out,
    )
    return debug_out

