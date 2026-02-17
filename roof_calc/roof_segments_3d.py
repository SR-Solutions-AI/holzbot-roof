"""
Roof segments 3D – segmente identice cu house_render și extragere fețe din rețeaua de segmente.
Fețele din house_3d sunt formate din aceleași segmente ca în house_render.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from roof_calc.overhang import _is_interior_corner, ridge_intersection_corner_lines


def _round_key(p: Tuple[float, float], prec: int = 4) -> Tuple[float, ...]:
    return (round(p[0], prec), round(p[1], prec))


def _snap_to_boundary(
    cx: float,
    cy: float,
    floor_polygon: Any,
    corner_tol: float = 2.0,
    boundary_tol: float = 2.5,
) -> Tuple[float, float]:
    """Snap (cx,cy) la colț sau la cel mai apropiat punct pe conturul streașinii."""
    if floor_polygon is None or not hasattr(floor_polygon, "exterior"):
        return (cx, cy)
    coords_ex = list(getattr(floor_polygon.exterior, "coords", []))
    if len(coords_ex) < 2:
        return (cx, cy)
    for i in range(len(coords_ex) - 1):
        vx, vy = float(coords_ex[i][0]), float(coords_ex[i][1])
        if (cx - vx) ** 2 + (cy - vy) ** 2 <= corner_tol * corner_tol:
            return (vx, vy)
    best_pt = (cx, cy)
    best_d2 = float("inf")
    for i in range(len(coords_ex) - 1):
        x1, y1 = float(coords_ex[i][0]), float(coords_ex[i][1])
        x2, y2 = float(coords_ex[i + 1][0]), float(coords_ex[i + 1][1])
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        if L2 < 1e-12:
            t, nx, ny = 0.0, x1, y1
        else:
            t = max(0, min(1, ((cx - x1) * dx + (cy - y1) * dy) / L2))
            nx, ny = x1 + t * dx, y1 + t * dy
        d2 = (cx - nx) ** 2 + (cy - ny) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_pt = (nx, ny)
    return best_pt if best_d2 <= boundary_tol * boundary_tol else (cx, cy)


def get_roof_segments_3d(
    sections: List[Dict[str, Any]],
    floor_polygon: Any,
    *,
    wall_height: float,
    roof_angle_deg: float = 30.0,
    corner_lines: Optional[List[Tuple[Tuple[float, float], List[Tuple[float, float]], Any]]] = None,
    use_section_rect_eaves: bool = False,
    ridge_magenta_contour_only: bool = False,
) -> List[Tuple[List[float], List[float]]]:
    """
    Returnează segmentele 3D identice cu cele desenate în house_render:
    magenta (intersecție → colțuri), portocalii (ridge endpoints → streașină), ridge, verde (contur streașină).
    Fiecare segment = (p1, p2) cu p1,p2 = [x,y,z].
    """
    segments: List[Tuple[List[float], List[float]]] = []

    if not sections:
        return segments

    z1 = wall_height
    tan_angle = math.tan(math.radians(roof_angle_deg))

    def _ridge_z(sec: Dict[str, Any]) -> float:
        """Z al crestei: derivat din roof_angle_deg (panta din formular). Pentru 1_w (shed) ridge e la centru, span = jumătate din latură."""
        br = sec.get("bounding_rect", [])
        if len(br) < 3:
            return z1
        xs = [float(p[0]) for p in br]
        ys = [float(p[1]) for p in br]
        orient = str(sec.get("ridge_orientation", "horizontal"))
        # Pentru shed (1_w): ridge la centru → span = jumătate; ridge_z = z1 + span*tan(angle)
        span = (max(ys) - min(ys)) / 2.0 if orient == "horizontal" else (max(xs) - min(xs)) / 2.0
        return z1 + span * tan_angle

    z_ridge_common = max(_ridge_z(s) for s in sections)
    cl = corner_lines or ridge_intersection_corner_lines(sections, floor_polygon=floor_polygon)
    coords_ex = (
        list(floor_polygon.exterior.coords)
        if floor_polygon is not None and hasattr(floor_polygon, "exterior")
        else []
    )

    def snap(cx: float, cy: float) -> Tuple[float, float]:
        return _snap_to_boundary(cx, cy, floor_polygon)

    # Colțuri interioare
    interior_keys: Set[Tuple[float, float]] = set()
    for i in range(len(coords_ex)):
        curr = (float(coords_ex[i][0]), float(coords_ex[i][1]))
        prev = (
            (float(coords_ex[i - 1][0]), float(coords_ex[i - 1][1]))
            if i > 0
            else (float(coords_ex[-2][0]), float(coords_ex[-2][1]))
        )
        next_p = (
            (float(coords_ex[i + 1][0]), float(coords_ex[i + 1][1]))
            if i + 1 < len(coords_ex)
            else (float(coords_ex[1][0]), float(coords_ex[1][1]))
        )
        if _is_interior_corner(prev, curr, next_p, floor_polygon):
            interior_keys.add((round(curr[0], 1), round(curr[1], 1)))

    def is_interior(cx: float, cy: float) -> bool:
        return (round(cx, 1), round(cy, 1)) in interior_keys

    def drop_interior_and_symmetric(corners: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Dacă un segment portocaliu are vârful într-un colț interior al streașinii, îl eliminăm
        împreună cu cel simetric opus (din același capăt de ridge)."""
        drop = {
            i
            for i, c in enumerate(corners)
            if is_interior(*snap(float(c[0]), float(c[1])))
        }
        n = len(corners)
        for i in list(drop):
            drop.add(n - 1 - i)  # simetric opus din același capăt
        return [c for i, c in enumerate(corners) if i not in drop]

    # corners_by_rect
    corners_by_rect: Dict[Tuple[float, float, float, float], List[Tuple[float, float]]] = {}
    for item in cl:
        corners = item[1]
        rect_key = item[2] if len(item) >= 3 else None
        if rect_key is not None:
            rk = (
                round(rect_key[0], 2),
                round(rect_key[1], 2),
                round(rect_key[2], 2),
                round(rect_key[3], 2),
            )
            if rk not in corners_by_rect:
                corners_by_rect[rk] = []
            existing = {_round_key(c) for c in corners_by_rect[rk]}
            for c in corners:
                k = _round_key(c)
                if k not in existing:
                    existing.add(k)
                    corners_by_rect[rk].append((float(c[0]), float(c[1])))

    def add_seg(p1: List[float], p2: List[float]) -> None:
        segments.append((list(p1), list(p2)))

    # Magenta: intersecție → colțuri
    for item in cl:
        pt = item[0]
        corners = item[1]
        ix, iy = float(pt[0]), float(pt[1])
        for cx, cy in corners:
            sx, sy = snap(float(cx), float(cy))
            add_seg([ix, iy, z_ridge_common], [sx, sy, z1])

    drawn_from: Set[Tuple[int, int]] = set()

    def key(px: float, py: float) -> Tuple[int, int]:
        return (int(round(px / 5)), int(round(py / 5)))

    def already_drawn(px: float, py: float) -> bool:
        return key(px, py) in drawn_from

    def mark_drawn(px: float, py: float) -> None:
        drawn_from.add(key(px, py))

    # Portocalii (A-frame): ridge endpoints → streașină. Omise când ridge_magenta_contour_only (ex. a_faces).
    if not use_section_rect_eaves and not ridge_magenta_contour_only:
        for sec in sections:
            ridge = sec.get("ridge_line", [])
            br = sec.get("bounding_rect", [])
            if len(ridge) < 2 or len(br) < 3:
                continue
            xs = [float(p[0]) for p in br]
            ys = [float(p[1]) for p in br]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            orient = str(sec.get("ridge_orientation", "horizontal"))
            r0, r1 = ridge[0], ridge[1]
            p0 = [float(r0[0]), float(r0[1]), z_ridge_common]
            p1 = [float(r1[0]), float(r1[1]), z_ridge_common]
            rect_key = (round(minx, 2), round(miny, 2), round(maxx, 2), round(maxy, 2))
            mag_corners = corners_by_rect.get(rect_key, [])
            tl = (minx, maxy)
            tr = (maxx, maxy)
            bl = (minx, miny)
            br_pt = (maxx, miny)

            if orient == "horizontal":
                left_raw = [c for c in mag_corners if float(c[0]) < (minx + maxx) / 2]
                right_raw = [c for c in mag_corners if float(c[0]) >= (minx + maxx) / 2]
                if not mag_corners:
                    left_raw, right_raw = [tl, bl], [tr, br_pt]
                left_corners = drop_interior_and_symmetric(left_raw)
                right_corners = drop_interior_and_symmetric(right_raw)
                if float(r0[0]) <= float(r1[0]):
                    left_end, right_end = p0, p1
                else:
                    left_end, right_end = p1, p0
                if not already_drawn(float(left_end[0]), float(left_end[1])) and left_corners:
                    for c in left_corners:
                        sx, sy = snap(float(c[0]), float(c[1]))
                        add_seg(left_end, [sx, sy, z1])
                    mark_drawn(float(left_end[0]), float(left_end[1]))
                if not already_drawn(float(right_end[0]), float(right_end[1])) and right_corners:
                    for c in right_corners:
                        sx, sy = snap(float(c[0]), float(c[1]))
                        add_seg(right_end, [sx, sy, z1])
                    mark_drawn(float(right_end[0]), float(right_end[1]))
            else:
                top_raw = [c for c in mag_corners if float(c[1]) < (miny + maxy) / 2]
                bottom_raw = [c for c in mag_corners if float(c[1]) >= (miny + maxy) / 2]
                if not mag_corners:
                    top_raw, bottom_raw = [tl, tr], [bl, br_pt]
                top_corners = drop_interior_and_symmetric(top_raw)
                bottom_corners = drop_interior_and_symmetric(bottom_raw)
                if float(r0[1]) <= float(r1[1]):
                    top_end, bottom_end = p0, p1
                else:
                    top_end, bottom_end = p1, p0
                if not already_drawn(float(top_end[0]), float(top_end[1])) and top_corners:
                    for c in top_corners:
                        sx, sy = snap(float(c[0]), float(c[1]))
                        add_seg(top_end, [sx, sy, z1])
                    mark_drawn(float(top_end[0]), float(top_end[1]))
                if not already_drawn(float(bottom_end[0]), float(bottom_end[1])) and bottom_corners:
                    for c in bottom_corners:
                        sx, sy = snap(float(c[0]), float(c[1]))
                        add_seg(bottom_end, [sx, sy, z1])
                    mark_drawn(float(bottom_end[0]), float(bottom_end[1]))

    # Ridge: exact ca în a_lines – ridge din FIECARE secțiune (extinse de extend_secondary_sections_to_main_ridge)
    for sec in sections:
        ridge = sec.get("ridge_line", [])
        if len(ridge) >= 2:
            r0, r1 = ridge[0], ridge[1]
            add_seg(
                [float(r0[0]), float(r0[1]), z_ridge_common],
                [float(r1[0]), float(r1[1]), z_ridge_common],
            )

    # Verde (contur streașină la z1): doar conturul EXTERIOR al casei (ridge, magenta, contur = cele 3 tipuri)
    if use_section_rect_eaves:
        for sec in sections:
            br = sec.get("bounding_rect", [])
            if len(br) < 3:
                continue
            perim = [
                (min(p[0] for p in br), min(p[1] for p in br)),
                (max(p[0] for p in br), min(p[1] for p in br)),
                (max(p[0] for p in br), max(p[1] for p in br)),
                (min(p[0] for p in br), max(p[1] for p in br)),
                (min(p[0] for p in br), min(p[1] for p in br)),
            ]
            for i in range(len(perim) - 1):
                x1, y1 = perim[i]
                x2, y2 = perim[i + 1]
                add_seg([x1, y1, z1], [x2, y2, z1])
    else:
        for i in range(len(coords_ex) - 1):
            p1, p2 = coords_ex[i], coords_ex[i + 1]
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])
            add_seg([x1, y1, z1], [x2, y2, z1])

    return segments


def _vertex_key(p: List[float], tol: float = 1e-6) -> Tuple[float, float, float]:
    return (round(p[0] / tol) * tol, round(p[1] / tol) * tol, round(p[2] / tol) * tol)


def _deduplicate_segments(
    segments: List[Tuple[List[float], List[float]]], tol: float = 0.5
) -> List[Tuple[List[float], List[float]]]:
    """Elimină segmente duplicate (inclusiv ridge repetat)."""
    seen: Set[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = set()
    out: List[Tuple[List[float], List[float]]] = []
    for p1, p2 in segments:
        k1 = _vertex_key(p1, tol)
        k2 = _vertex_key(p2, tol)
        if k1 == k2:
            continue
        key = (k1, k2) if k1 <= k2 else (k2, k1)
        if key in seen:
            continue
        seen.add(key)
        out.append((list(p1), list(p2)))
    return out


def _angle_2d(ox: float, oy: float, px: float, py: float) -> float:
    return math.atan2(py - oy, px - ox)


def _xy_to_z_map(segments: List[Tuple[List[float], List[float]]], tol: float = 5.0) -> Dict[Tuple[float, float], float]:
    """Mapare (x,y) -> z din endpoint-urile segmentelor."""
    m: Dict[Tuple[float, float], float] = {}
    for p1, p2 in segments:
        for p in (p1, p2):
            k = _vertex_key(p, tol)
            k2 = (round(k[0], 2), round(k[1], 2))
            if k2 not in m:
                m[k2] = float(p[2])
    return m


def _lookup_z(px: float, py: float, xy_to_z: Dict[Tuple[float, float], float], tol: float = 5.0) -> float:
    """Caută z pentru (px, py) în mapare sau la cei mai apropiați vecini."""
    k2 = (round(px, 2), round(py, 2))
    if k2 in xy_to_z:
        return xy_to_z[k2]
    best_z, best_d2 = 0.0, float("inf")
    for (qx, qy), z in xy_to_z.items():
        d2 = (px - qx) ** 2 + (py - qy) ** 2
        if d2 < best_d2:
            best_d2, best_z = d2, z
    return best_z if best_d2 < (tol * 2) ** 2 else 0.0


def _round_segment_coords(
    segments: List[Tuple[List[float], List[float]]],
    decimals: int = 2,
) -> List[Tuple[List[float], List[float]]]:
    """Rotunjește (x,y) la decimals – reduce goluri numerice la polygonize, fără a uni puncte distincte."""
    if decimals < 0:
        return list(segments)
    out: List[Tuple[List[float], List[float]]] = []
    for p1, p2 in segments:
        x1, y1 = round(float(p1[0]), decimals), round(float(p1[1]), decimals)
        x2, y2 = round(float(p2[0]), decimals), round(float(p2[1]), decimals)
        if (x1 - x2) ** 2 + (y1 - y2) ** 2 < 1e-12:
            continue
        out.append(([x1, y1, float(p1[2])], [x2, y2, float(p2[2])]))
    return out


def segments_to_faces(
    segments: List[Tuple[List[float], List[float]]],
    *,
    min_area: float = 1e-6,
    max_area_ratio: float = 0.9,
    include_horizontal_faces: bool = True,
    segments_for_z: Optional[List[Tuple[List[float], List[float]]]] = None,
) -> List[List[List[float]]]:
    """
    Extrage fețe din segmente. Folosește shapely.ops.polygonize_full (GEOS) – robust pentru ridge-uri.
    Fallback: cycle tracing manual.
    segments_for_z: segmente suplimentare (ex. portocalii) pentru maparea (x,y)->z la transpunere 3D.
    """
    if not segments:
        return []

    tol = 5.0
    # Rotunjire ușoară (2 decimale) pentru polygonize – închide goluri de floating point
    segs_rounded = _round_segment_coords(segments, decimals=2)
    if not segs_rounded:
        segs_rounded = list(segments)

    all_x = [p[0] for s in segs_rounded for p in s]
    all_y = [p[1] for s in segs_rounded for p in s]
    bbox_area = (max(all_x) - min(all_x)) * (max(all_y) - min(all_y)) if all_x and all_y else 1e9
    max_area = bbox_area * max_area_ratio if bbox_area > 1e-6 else 1e9
    max_area = min(max_area, bbox_area * 0.95)
    xy_to_z = _xy_to_z_map(segs_rounded, tol)
    if segments_for_z:
        segs_z = _round_segment_coords(segments_for_z, decimals=2) or list(segments_for_z)
        xy_to_z.update(_xy_to_z_map(segs_z, tol))

    try:
        from shapely.geometry import LineString
        from shapely.ops import polygonize_full
        lines = [LineString([(float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))]) for p1, p2 in segs_rounded]
        polygons, _cuts, _dangles, _invalids = polygonize_full(lines)
        faces: List[List[List[float]]] = []
        seen_face_keys: Set[frozenset] = set()

        def _iter_polys(g):
            if g is None or getattr(g, "is_empty", True):
                return
            if hasattr(g, "geoms"):
                yield from g.geoms
            elif hasattr(g, "__iter__") and not isinstance(g, (list, tuple)):
                try:
                    yield from g
                except TypeError:
                    yield g
            else:
                yield g

        for poly in _iter_polys(polygons):
            ext = getattr(poly, "exterior", None)
            if ext is None:
                continue
            coords = list(getattr(ext, "coords", []))
            if len(coords) < 4:
                continue
            vs = []
            for i in range(len(coords) - 1):
                x, y = float(coords[i][0]), float(coords[i][1])
                z = _lookup_z(x, y, xy_to_z, tol)
                vs.append([x, y, z])
            if len(vs) < 3:
                continue
            area = 0.0
            for i in range(len(vs)):
                j = (i + 1) % len(vs)
                area += vs[i][0] * vs[j][1] - vs[j][0] * vs[i][1]
            area = abs(area) * 0.5
            if area < min_area or area > max_area:
                continue
            z_vals = [v[2] for v in vs]
            if not include_horizontal_faces and max(z_vals) - min(z_vals) < 1e-6:
                continue
            face_key = frozenset(_vertex_key(v, tol) for v in vs)
            if face_key in seen_face_keys:
                continue
            seen_face_keys.add(face_key)
            faces.append(vs)

        if faces:
            # Adaugă din cycle tracing doar fețe care NU se suprapun cu cele existente
            try:
                from shapely.geometry import Polygon as ShapelyPolygon
                existing_polys = [
                    ShapelyPolygon([(float(v[0]), float(v[1])) for v in f])
                    for f in faces
                ]
                ct_faces = _segments_to_faces_cycle_tracing(
                    segs_rounded, min_area=min_area, max_area_ratio=max_area_ratio,
                    include_horizontal_faces=include_horizontal_faces,
                )
                for vs in ct_faces:
                    fk = frozenset(_vertex_key(v, tol) for v in vs)
                    if fk in seen_face_keys:
                        continue
                    try:
                        new_poly = ShapelyPolygon([(float(v[0]), float(v[1])) for v in vs])
                        if new_poly.is_empty or not new_poly.is_valid:
                            continue
                        new_area = float(new_poly.area)
                        if new_area < min_area:
                            continue
                        overlaps = False
                        for ep in existing_polys:
                            if ep.is_empty or (hasattr(ep, "is_valid") and not ep.is_valid):
                                continue
                            inter = new_poly.intersection(ep)
                            ia = float(getattr(inter, "area", 0) or 0)
                            if ia > 0.5 * min(new_area, float(getattr(ep, "area", 0) or 1)):
                                overlaps = True
                                break
                        if overlaps:
                            continue
                        seen_face_keys.add(fk)
                        faces.append(vs)
                        existing_polys.append(new_poly)
                    except Exception:
                        pass
            except Exception:
                pass
            return faces
    except Exception:
        pass

    # Fallback: cycle tracing pe segmente rotunjite
    ct_faces = _segments_to_faces_cycle_tracing(
        segs_rounded, min_area=min_area, max_area_ratio=max_area_ratio,
        include_horizontal_faces=include_horizontal_faces,
    )
    if ct_faces:
        return ct_faces

    return _segments_to_faces_cycle_tracing(
        segments, min_area=min_area, max_area_ratio=max_area_ratio,
        include_horizontal_faces=include_horizontal_faces,
    )


def _segments_to_faces_cycle_tracing(
    segments: List[Tuple[List[float], List[float]]],
    *,
    min_area: float = 1e-6,
    max_area_ratio: float = 0.9,
    include_horizontal_faces: bool = True,
) -> List[List[List[float]]]:
    """Fallback: cycle tracing manual."""
    if not segments:
        return []
    tol = 5.0
    all_x = [p[0] for s in segments for p in s]
    all_y = [p[1] for s in segments for p in s]
    bbox_area = (max(all_x) - min(all_x)) * (max(all_y) - min(all_y)) if all_x and all_y else 1e9
    max_area = bbox_area * max_area_ratio if bbox_area > 1e-6 else 1e9
    max_area = min(max_area, bbox_area * 0.95)
    v_to_idx: Dict[Tuple[float, float, float], int] = {}
    idx_to_v: List[List[float]] = []

    def ensure_vertex(p: List[float]) -> int:
        k = _vertex_key(p, tol)
        if k not in v_to_idx:
            v_to_idx[k] = len(idx_to_v)
            idx_to_v.append([float(p[0]), float(p[1]), float(p[2])])
        return v_to_idx[k]

    edges: List[Tuple[int, int]] = []
    for p1, p2 in segments:
        i, j = ensure_vertex(p1), ensure_vertex(p2)
        if i != j:
            edges.append((i, j))
            edges.append((j, i))

    adj: Dict[int, List[Tuple[int, float, int]]] = {}
    for ei, (a, b) in enumerate(edges):
        if a not in adj:
            adj[a] = []
        px, py = idx_to_v[a][0], idx_to_v[a][1]
        qx, qy = idx_to_v[b][0], idx_to_v[b][1]
        adj[a].append((b, _angle_2d(px, py, qx, qy), ei))
    for a in adj:
        adj[a].sort(key=lambda t: t[1])

    visited: Set[int] = set()
    faces: List[List[List[float]]] = []
    seen_face_keys: Set[frozenset] = set()

    for ei in range(len(edges)):
        if ei in visited:
            continue
        a, b = edges[ei]
        cycle = [a, b]
        visited.add(ei)
        current, prev = b, a
        for _ in range(len(edges) + 10):
            if current not in adj or not adj[current]:
                break
            lst = adj[current]
            next_idx = -1
            for k, (nei, _ang, ej) in enumerate(lst):
                if nei == prev:
                    next_idx = (k + 1) % len(lst)
                    break
            if next_idx < 0:
                break
            nei, _ang, ej = lst[next_idx]
            visited.add(ej)
            cycle.append(nei)
            if nei == cycle[0]:
                break
            prev, current = current, nei

        if len(cycle) < 3 or cycle[-1] != cycle[0]:
            continue
        vs = [idx_to_v[i] for i in cycle[:-1]]
        area = 0.0
        for i in range(len(vs)):
            j = (i + 1) % len(vs)
            area += vs[i][0] * vs[j][1] - vs[j][0] * vs[i][1]
        area = abs(area) * 0.5
        if area < min_area or area > max_area:
            continue
        z_vals = [v[2] for v in vs]
        if not include_horizontal_faces and max(z_vals) - min(z_vals) < 1e-6:
            continue
        face_key = frozenset(_vertex_key(v, tol) for v in vs)
        if face_key in seen_face_keys:
            continue
        seen_face_keys.add(face_key)
        faces.append(vs)

    return faces


def _segment_intersection_3d(
    p1: List[float], p2: List[float],
    q1: List[float], q2: List[float],
    tol: float = 1e-9,
    allow_t_junction: bool = True,
) -> Optional[List[float]]:
    """Intersecție 2D (x,y) cu interpolare z.
    Returnează [x,y,z] dacă segmentele se încrucișează.
    allow_t_junction: dacă True, acceptă T-junctions (pt în interiorul segmentului A, dar la capăt al lui B)."""
    ax1, ay1 = float(p1[0]), float(p1[1])
    ax2, ay2 = float(p2[0]), float(p2[1])
    bx1, by1 = float(q1[0]), float(q1[1])
    bx2, by2 = float(q2[0]), float(q2[1])
    dxa, dya = ax2 - ax1, ay2 - ay1
    dxb, dyb = bx2 - bx1, by2 - by1
    denom = dxa * dyb - dya * dxb
    if abs(denom) < tol:
        return None
    t = ((bx1 - ax1) * dyb - (by1 - ay1) * dxb) / denom
    s = ((bx1 - ax1) * dya - (by1 - ay1) * dxa) / denom
    if t <= tol or t >= 1 - tol:
        return None
    if not allow_t_junction and (s <= tol or s >= 1 - tol):
        return None
    ix = ax1 + t * dxa
    iy = ay1 + t * dya
    z1, z2 = float(p1[2]), float(p2[2])
    qz1, qz2 = float(q1[2]), float(q2[2])
    iz = z1 + t * (z2 - z1)
    iz_alt = qz1 + s * (qz2 - qz1)
    iz = (iz + iz_alt) / 2.0
    return [ix, iy, iz]


def _point_on_segment_t(
    px: float, py: float,
    a1: List[float], a2: List[float],
    tol: float = 1e-9,
) -> Optional[float]:
    """Returnează t în (0,1) dacă (px,py) e în interiorul segmentului (a1,a2)."""
    ax1, ay1 = float(a1[0]), float(a1[1])
    ax2, ay2 = float(a2[0]), float(a2[1])
    dx, dy = ax2 - ax1, ay2 - ay1
    L2 = dx * dx + dy * dy
    if L2 < tol * tol:
        return None
    t = ((px - ax1) * dx + (py - ay1) * dy) / L2
    if t <= tol or t >= 1 - tol:
        return None
    # Verifică colinearitate
    ix, iy = ax1 + t * dx, ay1 + t * dy
    if (px - ix) ** 2 + (py - iy) ** 2 > tol * tol:
        return None
    return t


def _subdivide_segments_at_intersections(
    segments: List[Tuple[List[float], List[float]]],
    tol: float = 1e-6,
) -> List[Tuple[List[float], List[float]]]:
    """Subdivide segmente la punctele de intersecție și T-junctions (capete pe alte segmente)."""
    if len(segments) <= 1:
        return list(segments)
    out: List[Tuple[List[float], List[float]]] = []
    for i, (a1, a2) in enumerate(segments):
        cuts_a: Set[float] = {0.0, 1.0}
        for j, (b1, b2) in enumerate(segments):
            if i == j:
                continue
            pt = _segment_intersection_3d(a1, a2, b1, b2, tol=tol)
            if pt is not None:
                ax1, ay1 = float(a1[0]), float(a1[1])
                ax2, ay2 = float(a2[0]), float(a2[1])
                dx, dy = float(a2[0]) - float(a1[0]), float(a2[1]) - float(a1[1])
                L2 = dx * dx + dy * dy
                if L2 >= tol * tol:
                    t = ((pt[0] - ax1) * dx + (pt[1] - ay1) * dy) / L2
                    if tol < t < 1 - tol:
                        cuts_a.add(t)
            else:
                for ep in [b1, b2]:
                    t = _point_on_segment_t(float(ep[0]), float(ep[1]), a1, a2, tol=tol)
                    if t is not None:
                        cuts_a.add(t)
        cuts_a = sorted(cuts_a)
        for k in range(len(cuts_a) - 1):
            t0, t1 = cuts_a[k], cuts_a[k + 1]
            if t1 - t0 < tol:
                continue
            x0 = float(a1[0]) + t0 * (float(a2[0]) - float(a1[0]))
            y0 = float(a1[1]) + t0 * (float(a2[1]) - float(a1[1]))
            z0 = float(a1[2]) + t0 * (float(a2[2]) - float(a1[2]))
            x1 = float(a1[0]) + t1 * (float(a2[0]) - float(a1[0]))
            y1 = float(a1[1]) + t1 * (float(a2[1]) - float(a1[1]))
            z1 = float(a1[2]) + t1 * (float(a2[2]) - float(a1[2]))
            out.append(([x0, y0, z0], [x1, y1, z1]))
    return out if out else list(segments)


def _filter_faces_for_a_frame(faces_verts: List[List[List[float]]]) -> List[List[List[float]]]:
    """
    Elimină fețe degenerate și duplicate. Păstrează doar fețe valide, unice, înclinate (nu orizontale).
    """
    tol = 0.1

    def _vertex_key(v: List[float]) -> Tuple[float, float, float]:
        return (
            round(float(v[0]) / tol) * tol,
            round(float(v[1]) / tol) * tol,
            round(float(v[2]) / tol) * tol,
        )

    def _collapse_duplicates(vs: List[List[float]]) -> List[List[float]]:
        out: List[List[float]] = []
        for v in vs:
            if not out or _vertex_key(v) != _vertex_key(out[-1]):
                out.append(list(v))
        if len(out) >= 2 and _vertex_key(out[0]) == _vertex_key(out[-1]):
            out.pop()
        return out

    seen_keys: Set[frozenset] = set()
    result: List[List[List[float]]] = []
    for vs in faces_verts:
        if len(vs) < 3:
            continue
        vs = _collapse_duplicates(vs)
        if len(vs) < 3:
            continue
        z_vals = [float(v[2]) for v in vs]
        if max(z_vals) - min(z_vals) < 1e-6:
            continue  # excludem fețe orizontale (streașină)
        key = frozenset(_vertex_key(v) for v in vs)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        result.append(vs)
    return result


def get_faces_3d_from_segments(
    sections: List[Dict[str, Any]],
    floor_polygon: Any,
    *,
    wall_height: float,
    roof_angle_deg: float = 30.0,
    corner_lines: Optional[List[Tuple[Tuple[float, float], List[Tuple[float, float]], Any]]] = None,
    use_section_rect_eaves: bool = False,
) -> List[Dict[str, Any]]:
    """
    Returnează fețe 3D formate strict din rețeaua de segmente (ridge, magenta, contur).
    La transpunere în 3D folosește și segmentele portocalii pentru maparea (x,y)->z.
    Format: [{"vertices_3d": [[x,y,z], ...]}, ...]
    Exclude fețe orizontale, degenerate și duplicate.
    """
    seg_kw = dict(
        wall_height=wall_height,
        roof_angle_deg=roof_angle_deg,
        corner_lines=corner_lines,
        use_section_rect_eaves=use_section_rect_eaves,
    )
    segments = get_roof_segments_3d(
        sections, floor_polygon, **seg_kw, ridge_magenta_contour_only=True,
    )
    segments_full = get_roof_segments_3d(
        sections, floor_polygon, **seg_kw, ridge_magenta_contour_only=False,
    )
    segments = _deduplicate_segments(segments, tol=0.01)
    segments = _subdivide_segments_at_intersections(segments)
    segments_full = _deduplicate_segments(segments_full, tol=0.01)
    segments_full = _subdivide_segments_at_intersections(segments_full)
    faces_verts = segments_to_faces(
        segments, segments_for_z=segments_full, include_horizontal_faces=False,
    )
    faces_verts = _filter_faces_for_a_frame(faces_verts)
    return [{"vertices_3d": v} for v in faces_verts]
