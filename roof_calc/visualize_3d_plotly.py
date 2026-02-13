"""
Vizualizare 3D (piramidă) cu Plotly + Kaleido: export PNG direct (fără browser).
Folosit ca fallback când PyVista/VTK nu poate fi importat (ex. Python 3.14 + VTK wheels).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _hex_to_rgb01(hex_color: str) -> Tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def _triangulate_face(face_pts: List[List[float]]) -> List[List[List[float]]]:
    """Triangulează o față (tri sau quad) în 1-2 triunghiuri."""
    if len(face_pts) == 3:
        return [face_pts]
    if len(face_pts) == 4:
        a, b, c, d = face_pts
        return [[a, b, c], [a, c, d]]
    return []


def _triangulate_fan(pts: List[List[float]]) -> List[List[List[float]]]:
    """Triangulare fan: pentru un poligon convex, triunghiuri (pts[0], pts[i], pts[i+1])."""
    if len(pts) < 3:
        return []
    if len(pts) == 3:
        return [pts]
    return [[pts[0], pts[i], pts[i + 1]] for i in range(1, len(pts) - 1)]


def _z_roof_at(
    roof_faces: List[Dict[str, Any]],
    x: float,
    y: float,
    default_z: float,
    tol: float = 15.0,
) -> float:
    """
    Înălțimea acoperișului la (x,y). Interpolează din fețele acoperișului.
    Folosește buffer pentru puncte de pe margine; fallback: fața cea mai apropiată.
    """
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
        xs = [p[0] for p in xy]
        ys = [p[1] for p in xy]
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
            best = z
            best_dist = dist
    return float(best) if best is not None else default_z


def _z_at_point_pyramid(
    x: float, y: float,
    sec: Dict[str, Any],
    base_z: float,
    roof_angle_rad: float,
) -> float:
    """Calculează z pentru (x,y) pe o secțiune piramidă."""
    br = sec.get("bounding_rect", [])
    ridge = sec.get("ridge_line", [])
    orient = str(sec.get("ridge_orientation", "horizontal"))
    if len(br) < 3:
        return base_z
    minx = min(p[0] for p in br)
    maxx = max(p[0] for p in br)
    miny = min(p[1] for p in br)
    maxy = max(p[1] for p in br)
    if len(ridge) >= 2:
        ridge_mid_x = (ridge[0][0] + ridge[1][0]) / 2
        ridge_mid_y = (ridge[0][1] + ridge[1][1]) / 2
    else:
        ridge_mid_x = (minx + maxx) / 2
        ridge_mid_y = (miny + maxy) / 2
    span = (maxy - miny) / 2 if orient == "horizontal" else (maxx - minx) / 2
    ridge_height = base_z + span * float(np.tan(roof_angle_rad))
    tanv = float(np.tan(roof_angle_rad))
    if orient == "horizontal":
        d = abs(y - ridge_mid_y)
    else:
        d = abs(x - ridge_mid_x)
    return float(ridge_height - d * tanv)


def _clip_pyramid_faces_with_polygon(
    faces: List[List[List[float]]],
    sec: Dict[str, Any],
    base_z: float,
    roof_angle_rad: float,
    clip_poly: Any,
) -> List[List[List[float]]]:
    """Decupează fețele piramidei cu clip_poly; returnează doar triunghiurile vizibile."""
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
    except Exception:
        return faces

    if clip_poly is None or getattr(clip_poly, "is_empty", True):
        return faces

    result: List[List[List[float]]] = []
    for face in faces:
        tris = _triangulate_face(face)
        for tri in tris:
            if len(tri) != 3:
                continue
            xy = [(float(p[0]), float(p[1])) for p in tri]
            try:
                tri_poly = ShapelyPolygon(xy)
            except Exception:
                continue
            try:
                inter = tri_poly.intersection(clip_poly)
            except Exception:
                continue
            if inter is None or inter.is_empty:
                continue
            geoms = getattr(inter, "geoms", [inter])
            for g in geoms:
                if getattr(g, "is_empty", True):
                    continue
                ext = getattr(g, "exterior", None)
                if ext is None:
                    continue
                coords = list(ext.coords)[:-1]
                if len(coords) < 3:
                    continue
                # Fan triangulation from first vertex
                c0 = coords[0]
                for i in range(1, len(coords) - 1):
                    c1, c2 = coords[i], coords[i + 1]
                    x0, y0 = float(c0[0]), float(c0[1])
                    x1, y1 = float(c1[0]), float(c1[1])
                    x2, y2 = float(c2[0]), float(c2[1])
                    z0 = _z_at_point_pyramid(x0, y0, sec, base_z, roof_angle_rad)
                    z1 = _z_at_point_pyramid(x1, y1, sec, base_z, roof_angle_rad)
                    z2 = _z_at_point_pyramid(x2, y2, sec, base_z, roof_angle_rad)
                    result.append([[x0, y0, z0], [x1, y1, z1], [x2, y2, z2]])
    return result if result else faces


def _camera_eye_from_elev_azim(elev_deg: float, azim_deg: float, scale: float = 1.8) -> Dict[str, float]:
    az = np.radians(azim_deg)
    el = np.radians(elev_deg)
    return {
        # Plotly camera.eye e un vector relativ (tipic ~1–2), nu în unități de coordonate ale scenei
        "x": float(scale * np.cos(el) * np.cos(az)),
        "y": float(scale * np.cos(el) * np.sin(az)),
        "z": float(scale * np.sin(el)),
    }


def _round_v3(pt: List[float], nd: int = 3) -> Tuple[float, float, float]:
    return (round(float(pt[0]), nd), round(float(pt[1]), nd), round(float(pt[2]), nd))


def _edge_lines_from_tris(
    tris: List[List[List[float]]],
    round_ndigits: int = 3,
    boundary_only: bool = True,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Construiește un set de segmente (x,y,z cu None separators) pentru contur/wireframe,
    deduplicând muchiile comune între triunghiuri.
    """
    # Count undirected edges so we can drop internal triangulation edges.
    # Edges that appear twice are typically interior (shared by 2 triangles).
    edge_counts: Dict[Tuple[Tuple[float, float, float], Tuple[float, float, float]], int] = {}
    for tri in tris:
        if len(tri) != 3:
            continue
        a, b, c = tri[0], tri[1], tri[2]
        va = _round_v3(a, round_ndigits)
        vb = _round_v3(b, round_ndigits)
        vc = _round_v3(c, round_ndigits)
        for e0, e1 in ((va, vb), (vb, vc), (vc, va)):
            if e0 == e1:
                continue
            key = (e0, e1) if e0 < e1 else (e1, e0)
            edge_counts[key] = edge_counts.get(key, 0) + 1

    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    # Keep only boundary edges (appear once) to avoid “unnecessary” diagonals.
    for (p0, p1), cnt in edge_counts.items():
        if boundary_only and cnt != 1:
            continue
        xs.extend([p0[0], p1[0], None])  # type: ignore[arg-type]
        ys.extend([p0[1], p1[1], None])  # type: ignore[arg-type]
        zs.extend([p0[2], p1[2], None])  # type: ignore[arg-type]
    return xs, ys, zs


def _tri_normal(tri: List[List[float]]) -> Optional[Tuple[float, float, float]]:
    """Normal vector al triunghiului (normalizat)."""
    if len(tri) != 3:
        return None
    a, b, c = tri[0], tri[1], tri[2]
    ab = (float(b[0]) - float(a[0]), float(b[1]) - float(a[1]), float(b[2]) - float(a[2]))
    ac = (float(c[0]) - float(a[0]), float(c[1]) - float(a[1]), float(c[2]) - float(a[2]))
    nx = ab[1] * ac[2] - ab[2] * ac[1]
    ny = ab[2] * ac[0] - ab[0] * ac[2]
    nz = ab[0] * ac[1] - ab[1] * ac[0]
    L = (nx * nx + ny * ny + nz * nz) ** 0.5
    if L < 1e-12:
        return None
    return (nx / L, ny / L, nz / L)


def _edge_lines_from_tris_with_ridges(
    tri_by_color: Dict[str, List[List[List[float]]]],
    roof_colors: set,
    round_ndigits: int = 2,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Linii despărțitoare între fețe (ridge/valley/hip).
    Folosește: (1) muchii shared de triunghiuri cu culori diferite, SAU
    (2) muchii shared de triunghiuri cu normale diferite (pentru pyramid cu o singură culoare).
    """
    edge_to_tris: Dict[Tuple[Tuple[float, float, float], Tuple[float, float, float]], List[Tuple[str, List[List[float]]]]] = {}
    for color_hex, tris in tri_by_color.items():
        c = (color_hex or "").strip().upper()
        if c not in roof_colors:
            continue
        for tri in tris:
            if len(tri) != 3:
                continue
            va = _round_v3(tri[0], round_ndigits)
            vb = _round_v3(tri[1], round_ndigits)
            vc = _round_v3(tri[2], round_ndigits)
            for e0, e1 in ((va, vb), (vb, vc), (vc, va)):
                if e0 == e1:
                    continue
                key = (e0, e1) if e0 < e1 else (e1, e0)
                edge_to_tris.setdefault(key, []).append((c, tri))

    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for (p0, p1), entries in edge_to_tris.items():
        if len(entries) < 2:
            continue
        colors_uniq = len(set(c for c, _ in entries))
        if colors_uniq >= 2:
            xs.extend([p0[0], p1[0], None])
            ys.extend([p0[1], p1[1], None])
            zs.extend([p0[2], p1[2], None])
            continue
        norms = [_tri_normal(tri) for _, tri in entries]
        norms = [n for n in norms if n is not None]
        if len(norms) < 2:
            continue
        dot_min = 1.0
        for i in range(len(norms)):
            for j in range(i + 1, len(norms)):
                dot = abs(norms[i][0] * norms[j][0] + norms[i][1] * norms[j][1] + norms[i][2] * norms[j][2])
                dot_min = min(dot_min, dot)
        if dot_min < 0.99:
            xs.extend([p0[0], p1[0], None])
            ys.extend([p0[1], p1[1], None])
            zs.extend([p0[2], p1[2], None])
    return xs, ys, zs


# Culori pentru modul schematic (outline + segmente)
SCHEMATIC_COLORS: Dict[str, str] = {
    "roof": "#E74C3C",
    "overhang": "#3498DB",
    "drip": "#2ECC71",
    "gutter": "#9B59B6",
    "downspout": "#F39C12",
    "connection": "#E67E22",
    "wall": "#95A5A6",
}

# Map pentru etichete în wireframe debug (color hex -> label)
_COLOR_TO_LABEL: Dict[str, str] = {
    "#B0B0B0": "Pereți",
    "#6B7280": "Jgheab",
    "#8B4513": "Drip",
    "#2563EB": "Burlan vertical",
    "#FF6B6B": "Acoperiș",
    "#4ECDC4": "Acoperiș",
    "#45B7D1": "Acoperiș",
    "#E53935": "Stub 0",
    "#43A047": "Stub 1",
    "#1E88E5": "Stub 2",
    "#FB8C00": "Stub 3",
    "#8E24AA": "Stub 4",
    "#00ACC1": "Stub 5",
    "#FDD835": "Stub 6",
    "#D81B60": "Stub 7",
}


def _label_for_color(color_hex: str) -> str:
    key = (color_hex or "").strip().upper()
    return _COLOR_TO_LABEL.get(key, "Element")


def _write_wireframe_html(
    tri_by_color: Dict[str, List[List[List[float]]]],
    minx_b: float,
    maxx_b: float,
    miny_b: float,
    maxy_b: float,
    z_max: float,
    wireframe_path: str,
) -> None:
    """Salvează un HTML interactiv wireframe pentru debugging (opacitate mică + muchii + legendă)."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return

    views = [{"elev": 30, "azim": 45, "title": "Wireframe Sud-Est"}, {"elev": 20, "azim": 225, "title": "Wireframe Nord-Vest"}]
    n = max(1, len(views))
    fig = make_subplots(
        rows=1,
        cols=n,
        specs=[[{"type": "scene"} for _ in range(n)]],
        subplot_titles=[v.get("title", f"View {i+1}") for i, v in enumerate(views[:n])],
    )

    for col_idx in range(n):
        for color_hex, tris in tri_by_color.items():
            xs: List[float] = []
            ys: List[float] = []
            zs: List[float] = []
            ii: List[int] = []
            jj: List[int] = []
            kk: List[int] = []
            for tri in tris:
                base = len(xs)
                for vx, vy, vz in tri:
                    xs.append(float(vx))
                    ys.append(float(vy))
                    zs.append(float(vz))
                ii.append(base + 0)
                jj.append(base + 1)
                kk.append(base + 2)
            r01, g01, b01 = _hex_to_rgb01(color_hex)
            rgb = f"rgb({int(r01*255)},{int(g01*255)},{int(b01*255)})"
            label = _label_for_color(color_hex)

            # Mesh semi-transparent (opacitate 0.12)
            fig.add_trace(
                go.Mesh3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    i=ii,
                    j=jj,
                    k=kk,
                    color=rgb,
                    opacity=0.12,
                    flatshading=True,
                    name=label,
                    legendgroup=label,
                ),
                row=1,
                col=col_idx + 1,
            )

            # Muchii wireframe (toate muchiile, nu doar contur)
            lx, ly, lz = _edge_lines_from_tris(tris, round_ndigits=3, boundary_only=False)
            if len(lx) > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=lx,
                        y=ly,
                        z=lz,
                        mode="lines",
                        line=dict(color=rgb, width=1.5),
                        name=label,
                        legendgroup=label,
                        showlegend=False,
                    ),
                    row=1,
                    col=col_idx + 1,
                )

        v = views[col_idx] if col_idx < len(views) else {"elev": 30, "azim": 45}
        eye = _camera_eye_from_elev_azim(float(v.get("elev", 30)), float(v.get("azim", 45)), scale=1.8)
        scene_name = "scene" if col_idx == 0 else f"scene{col_idx+1}"
        fig.update_layout(
            **{
                scene_name: dict(
                    xaxis=dict(range=[minx_b, maxx_b], backgroundcolor="white", gridcolor="lightgrey"),
                    yaxis=dict(range=[miny_b, maxy_b], backgroundcolor="white", gridcolor="lightgrey"),
                    zaxis=dict(range=[0, z_max], backgroundcolor="white", gridcolor="lightgrey"),
                    aspectmode="data",
                    camera=dict(
                        eye=eye,
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        projection=dict(type="perspective"),
                    ),
                )
            }
        )

    fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="top", y=1.02), margin=dict(l=10, r=10, t=60, b=10))
    Path(wireframe_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(wireframe_path, include_plotlyjs="cdn", full_html=True)
    logger.info("Saved wireframe debug HTML to %s", wireframe_path)


def _segment_to_plotly_line(p1: List[float], p2: List[float]) -> Tuple[List[float], List[float], List[float]]:
    """Convert segment (p1,p2) to Plotly line format (x,y,z with None separators)."""
    return [p1[0], p2[0], None], [p1[1], p2[1], None], [p1[2], p2[2], None]


def _extract_downspout_segments_from_tris(
    tris: List[List[List[float]]], tol_xy: float = 80.0
) -> List[Tuple[float, float, float, float]]:
    """Din triunghiuri (fețe cilindru vertical), extrage câte un segment (cx, cy, z_bot, z_top) per burlan.
    tol_xy mare (ex. 80) pentru ca toate triunghiurile unui cilindru să intre în același cluster."""
    clusters: Dict[Tuple[int, int], List[List[float]]] = {}
    for tri in tris:
        if len(tri) != 3:
            continue
        vs = tri
        cx = sum(float(v[0]) for v in vs) / 3
        cy = sum(float(v[1]) for v in vs) / 3
        zs = [float(v[2]) for v in vs]
        z_min, z_max = min(zs), max(zs)
        key = (int(round(cx / tol_xy)), int(round(cy / tol_xy)))
        clusters.setdefault(key, []).append([cx, cy, z_min, z_max])
    out: List[Tuple[float, float, float, float]] = []
    for pts in clusters.values():
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        z_min = min(p[2] for p in pts)
        z_max = max(p[3] for p in pts)
        if z_max - z_min > 80.0:
            out.append((cx, cy, z_min, z_max))
    return out


def _write_schematic_3d(
    tri_by_color: Dict[str, List[List[List[float]]]],
    gutter_endpoints: List[Tuple[float, float, float]],
    minx_b: float,
    maxx_b: float,
    miny_b: float,
    maxy_b: float,
    z_max: float,
    output_path: str,
    html_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    gutter_centerlines: Optional[List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]] = None,
    downspout_centerlines: Optional[Dict[str, List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]]] = None,
    roof_base_tris: Optional[List[List[List[float]]]] = None,
    floors_payload: Optional[List[Tuple[Any, Any, Dict[str, Any], Tuple[float, float]]]] = None,
    wall_height: float = 300.0,
) -> bool:
    """
    Randare schematică: acoperiș/drip doar outline (gol), tinichigerie doar segmente.
    Dacă floors_payload și wall_height sunt date, desenează și pereții întregi (outline).
    Culori: acoperiș, overhang, drip, streașină, burlan, pereți.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return False

    config = config or {}
    views = config.get("views", [{"elev": 30, "azim": 45, "title": "Schematic"}, {"elev": 20, "azim": 225, "title": "Nord-Vest"}])
    n = max(1, len(views))
    config_colors = config.get("colors", ["#FF6B6B", "#4ECDC4", "#45B7D1"])
    fig = make_subplots(
        rows=1,
        cols=n,
        specs=[[{"type": "scene"} for _ in range(n)]],
        subplot_titles=[v.get("title", f"View {i+1}") for i, v in enumerate(views[:n])],
    )

    roof_colors_upper = {str(c).strip().upper() for c in config_colors}
    drip_color_upper = "#8B4513"
    gray_upper = "#B0B0B0"

    roof_tris: List[List[List[float]]] = []
    drip_tris: List[List[List[float]]] = []
    metal_tris: List[List[List[float]]] = []

    for color_hex, tris in tri_by_color.items():
        c = (color_hex or "").strip().upper()
        if c == gray_upper:
            continue
        if c == drip_color_upper:
            drip_tris.extend(tris)
        elif c in roof_colors_upper:
            roof_tris.extend(tris)
        else:
            metal_tris.extend(tris)

    def add_edge_trace(lx: List[float], ly: List[float], lz: List[float], color: str, name: str) -> None:
        if len(lx) == 0:
            return
        for col_idx in range(n):
            fig.add_trace(
                go.Scatter3d(
                    x=lx, y=ly, z=lz,
                    mode="lines",
                    line=dict(color=color, width=3),
                    name=name,
                    legendgroup=name,
                ),
                row=1,
                col=col_idx + 1,
            )

    lx_r, ly_r, lz_r = _edge_lines_from_tris(roof_tris, boundary_only=True)
    add_edge_trace(lx_r, ly_r, lz_r, SCHEMATIC_COLORS["roof"], "Acoperiș (cu overhang)")

    lx_ridge, ly_ridge, lz_ridge = _edge_lines_from_tris_with_ridges(tri_by_color, roof_colors_upper)
    add_edge_trace(lx_ridge, ly_ridge, lz_ridge, SCHEMATIC_COLORS["roof"], "Linii despărțitoare fețe")

    if roof_base_tris:
        lx_rb, ly_rb, lz_rb = _edge_lines_from_tris(roof_base_tris, boundary_only=True)
        lx_rb_ridge, ly_rb_ridge, lz_rb_ridge = _edge_lines_from_tris_with_ridges(
            {"#BASE": roof_base_tris}, {"#BASE"}
        )
        lx_rb.extend(lx_rb_ridge)
        ly_rb.extend(ly_rb_ridge)
        lz_rb.extend(lz_rb_ridge)
        add_edge_trace(lx_rb, ly_rb, lz_rb, SCHEMATIC_COLORS["overhang"], "Acoperiș (fără overhang)")

    lx_d, ly_d, lz_d = _edge_lines_from_tris(drip_tris, boundary_only=True)
    add_edge_trace(lx_d, ly_d, lz_d, SCHEMATIC_COLORS["drip"], "Drip")

    lx_g, ly_g, lz_g = [], [], []
    if gutter_centerlines:
        for (p1, p2) in gutter_centerlines:
            lx, ly, lz = _segment_to_plotly_line(list(p1), list(p2))
            lx_g.extend(lx)
            ly_g.extend(ly)
            lz_g.extend(lz)
    else:
        for i in range(0, len(gutter_endpoints) - 1, 2):
            p1 = list(gutter_endpoints[i])
            p2 = list(gutter_endpoints[i + 1])
            lx, ly, lz = _segment_to_plotly_line(p1, p2)
            lx_g.extend(lx)
            ly_g.extend(ly)
            lz_g.extend(lz)
    add_edge_trace(lx_g, ly_g, lz_g, SCHEMATIC_COLORS["gutter"], "Streașină")

    lx_dsp, ly_dsp, lz_dsp = [], [], []
    lx_conn, ly_conn, lz_conn = [], [], []
    if downspout_centerlines:
        for (p1, p2) in downspout_centerlines.get("downspout", []):
            lx, ly, lz = _segment_to_plotly_line(list(p1), list(p2))
            lx_dsp.extend(lx)
            ly_dsp.extend(ly)
            lz_dsp.extend(lz)
        for (p1, p2) in downspout_centerlines.get("connection", []):
            lx, ly, lz = _segment_to_plotly_line(list(p1), list(p2))
            lx_conn.extend(lx)
            ly_conn.extend(ly)
            lz_conn.extend(lz)
    else:
        downspout_segments = _extract_downspout_segments_from_tris(metal_tris)
        for cx, cy, z_bot, z_top in downspout_segments:
            lx, ly, lz = _segment_to_plotly_line([cx, cy, z_bot], [cx, cy, z_top])
            lx_dsp.extend(lx)
            ly_dsp.extend(ly)
            lz_dsp.extend(lz)
    add_edge_trace(lx_dsp, ly_dsp, lz_dsp, SCHEMATIC_COLORS["downspout"], "Burlan")
    add_edge_trace(lx_conn, ly_conn, lz_conn, SCHEMATIC_COLORS["connection"], "Conexiune burlan–streașină")

    # Pereți întregi (outline) – când floors_payload e furnizat
    if floors_payload and wall_height > 0:
        lx_w, ly_w, lz_w = [], [], []
        for floor_idx, (_path, floor_poly, _rr, _offset) in enumerate(floors_payload):
            z0 = floor_idx * wall_height
            z1 = (floor_idx + 1) * wall_height
            coords = list(floor_poly.exterior.coords) if hasattr(floor_poly, "exterior") else []
            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i + 1]
                x1, y1 = float(p1[0]), float(p1[1])
                x2, y2 = float(p2[0]), float(p2[1])
                # Bordură inferioară, superioară și cele două muchii verticale
                for (a1, b1, z_a), (a2, b2, z_b) in [
                    ((x1, y1, z0), (x2, y2, z0)),
                    ((x1, y1, z1), (x2, y2, z1)),
                    ((x1, y1, z0), (x1, y1, z1)),
                    ((x2, y2, z0), (x2, y2, z1)),
                ]:
                    lx, ly, lz = _segment_to_plotly_line([a1, b1, z_a], [a2, b2, z_b])
                    lx_w.extend(lx)
                    ly_w.extend(ly)
                    lz_w.extend(lz)
        add_edge_trace(lx_w, ly_w, lz_w, SCHEMATIC_COLORS["wall"], "Pereți")

    for col_idx in range(n):
        v = views[col_idx] if col_idx < len(views) else {"elev": 30, "azim": 45}
        eye = _camera_eye_from_elev_azim(float(v.get("elev", 30)), float(v.get("azim", 45)), scale=1.8)
        scene_name = "scene" if col_idx == 0 else f"scene{col_idx+1}"
        fig.update_layout(
            **{
                scene_name: dict(
                    xaxis=dict(range=[minx_b, maxx_b], backgroundcolor="white", gridcolor="lightgrey"),
                    yaxis=dict(range=[miny_b, maxy_b], backgroundcolor="white", gridcolor="lightgrey"),
                    zaxis=dict(range=[0, z_max], backgroundcolor="white", gridcolor="lightgrey"),
                    aspectmode="data",
                    camera=dict(eye=eye, up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), projection=dict(type="perspective")),
                )
            }
        )

    fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="top", y=1.02), margin=dict(l=10, r=10, t=60, b=10))

    ok = False
    if html_path:
        try:
            Path(html_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
            ok = True
        except Exception as e:
            logger.warning("Schematic HTML failed: %s", e)
    if output_path:
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            import threading
            _result, _exc = [None], [None]

            def _write():
                try:
                    fig.write_image(output_path, width=1200 * n, height=900, scale=2)
                    _result[0] = True
                except Exception as e:
                    _exc[0] = e

            t = threading.Thread(target=_write, daemon=True)
            t.start()
            t.join(timeout=90)
            if t.is_alive():
                logger.warning("Schematic PNG timed out (90s), skipping")
            elif _exc[0]:
                logger.warning("Schematic PNG failed: %s", _exc[0])
            elif _result[0]:
                ok = True
        except Exception as e:
            logger.warning("Schematic PNG failed: %s", e)
    return ok


def visualize_3d_standard_plotly(
    output_path: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    all_floor_paths: Optional[List[str]] = None,
    floor_roof_results: Optional[List[Dict[str, Any]]] = None,
    html_output_path: Optional[str] = None,
    roof_levels: Optional[List[Tuple[float, Dict[str, Any], float, float, int]]] = None,
    lower_floor_roof_mode: str = "standard",
) -> bool:
    """
    Randare 3D standard (gable) cu Plotly + Kaleido, salvează PNG la output_path.
    Folosește pipeline-ul multi-etaj bazat pe `rectangles_floor` (similar cu Matplotlib),
    dar cu export robust (fără probleme de z-order).
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        logger.debug("Plotly not available: %s", e)
        return False

    import cv2
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union
    from shapely import affinity as shapely_affinity

    from roof_calc.flood_fill import flood_fill_interior, get_house_shape_mask
    from roof_calc.geometry import extract_polygon
    import roof_calc.roof_faces_3d as rf
    from roof_calc.overhang import (
        apply_overhang_to_sections,
        clip_roof_faces_to_polygon,
        compute_overhang_sides_from_footprint,
        compute_overhang_sides_from_union_boundary,
        extend_sections_to_connect,
        extend_secondary_sections_to_main_ridge,
        get_faces_3d_aframe_with_magenta,
        ridge_intersection_corner_lines,
        get_drip_edge_faces_3d,
        get_downspout_faces_for_floors,
        get_downspout_faces_pyramid,
        get_gutter_centerlines_3d,
        get_gutter_end_closures_3d,
        get_gutter_endpoints_3d,
        get_gutter_faces_3d,
    )

    config = config or {}
    roof_angle_deg = float(config.get("roof_angle", 30.0))
    wall_height = float(config.get("wall_height", 300.0))
    colors = config.get("colors", ["#FF6B6B", "#4ECDC4", "#45B7D1"])
    show_outlines = bool(config.get("show_outlines", True))
    outline_color = str(config.get("outline_color", "rgba(0,0,0,0.65)"))
    outline_width = int(config.get("outline_width", 3))
    overhang_px = float(config.get("overhang_px", 0.0))
    overhang_keep_height = bool(config.get("overhang_keep_height", True))
    overhang_drop_down = bool(config.get("overhang_drop_down", True))
    overhang_shift_whole_roof_down = bool(config.get("overhang_shift_whole_roof_down", False))
    views = config.get(
        "views",
        [{"elev": 30, "azim": 45, "title": "Sud-Est"}, {"elev": 20, "azim": 225, "title": "Nord-Vest"}],
    )

    if not all_floor_paths or not floor_roof_results or len(all_floor_paths) != len(floor_roof_results):
        return False

    def _polygon_from_path(path: str) -> Optional[Any]:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        filled0 = flood_fill_interior(im)
        house0 = get_house_shape_mask(filled0)
        poly0 = extract_polygon(house0)
        if poly0 is None or poly0.is_empty:
            return None
        return poly0

    def _largest_section_bbox(rr: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
        best = None
        best_area = -1
        for sec in rr.get("sections") or []:
            rect = sec.get("bounding_rect", [])
            if len(rect) < 3:
                continue
            xs = [p[0] for p in rect]
            ys = [p[1] for p in rect]
            minx, maxx = int(min(xs)), int(max(xs))
            miny, maxy = int(min(ys)), int(max(ys))
            area = (maxx - minx) * (maxy - miny)
            if area > best_area:
                best_area = area
                best = (minx, miny, maxx, maxy)
        return best

    def _bbox_center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def _compute_offsets(paths: List[str], results: List[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
        """Etaje cu aceeași mărime și formă se pun unul peste altul: aliniem centrele (centroid poligon)."""
        polys = [_polygon_from_path(p) for p in paths]
        best_idx = 0
        best_area = -1.0
        for i, poly in enumerate(polys):
            if poly is not None and not getattr(poly, "is_empty", True):
                a = float(getattr(poly, "area", 0) or 0)
                if a > best_area:
                    best_area = a
                    best_idx = i
        ref_poly = polys[best_idx] if best_idx < len(polys) else None
        if ref_poly is None or getattr(ref_poly, "is_empty", True):
            cx_ref, cy_ref = 0.0, 0.0
        else:
            c = ref_poly.centroid
            cx_ref, cy_ref = float(c.x), float(c.y)
        out: Dict[str, Tuple[int, int]] = {}
        for p, poly in zip(paths, polys):
            if poly is None or getattr(poly, "is_empty", True):
                out[p] = (int(round(-cx_ref)), int(round(-cy_ref)))
            else:
                c = poly.centroid
                ox = int(round(cx_ref - c.x))
                oy = int(round(cy_ref - c.y))
                out[p] = (ox, oy)
        return out

    def _translate_sections(secs: List[Dict[str, Any]], dx: float, dy: float) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for sec in secs:
            rect = sec.get("bounding_rect", [])
            ridge = sec.get("ridge_line", [])
            out.append(
                {
                    **sec,
                    "bounding_rect": [(float(p[0]) + dx, float(p[1]) + dy) for p in rect] if rect else [],
                    "ridge_line": [(float(p[0]) + dx, float(p[1]) + dy) for p in ridge]
                    if ridge and len(ridge) >= 2
                    else [],
                }
            )
        return out

    def _filter_connections_by_sections(conns: List[Dict[str, Any]], keep_ids: set[int]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for c in conns or []:
            ids = c.get("section_ids") or c.get("section_id") or []
            try:
                if isinstance(ids, (list, tuple)):
                    if all(int(x) in keep_ids for x in ids):
                        out.append(c)
                else:
                    if int(ids) in keep_ids:
                        out.append(c)
            except Exception:
                continue
        return out

    def _covered_by_upper(sec_poly: Any, union_above: Any, area_thresh: float = 500.0) -> bool:
        if union_above is None or getattr(union_above, "is_empty", True):
            return False
        try:
            rem = sec_poly.difference(union_above)
            return float(getattr(rem, "area", 0.0) or 0.0) < area_thresh
        except Exception:
            return False

    offsets = _compute_offsets(all_floor_paths, floor_roof_results)
    floors_payload: List[Tuple[str, Any, Dict[str, Any], Tuple[int, int]]] = []
    for p, rr in zip(all_floor_paths, floor_roof_results):
        poly0 = _polygon_from_path(p)
        if poly0 is None:
            continue
        ox, oy = offsets.get(p, (0, 0))
        poly_t = shapely_affinity.translate(poly0, xoff=ox, yoff=oy)
        floors_payload.append((p, poly_t, rr, (ox, oy)))
    if not floors_payload:
        return False

    floors_payload.sort(key=lambda t: float(getattr(t[1], "area", 0.0) or 0.0), reverse=True)
    num_floors = len(floors_payload)
    z_max = num_floors * wall_height + 500

    # Pre-compute roof faces for lower floors (roof_levels) pentru prelungirea pereților
    roof_faces_by_floor: Dict[int, List[Dict[str, Any]]] = {}
    roof_faces_base_by_floor: Dict[int, List[Dict[str, Any]]] = {}
    use_shed_lower = lower_floor_roof_mode == "shed"
    if roof_levels:
        try:
            if use_shed_lower:
                from roof_calc.visualize_3d_pyvista import _roof_section_faces_shed  # type: ignore
                from roof_calc.overhang import high_side_for_shed_from_upper_floor
            for z_base, roof_data, dx, dy, _fl in roof_levels:
                fl = int(_fl) if _fl is not None else 0
                footprint_fl = floors_payload[fl][1] if 0 <= fl < len(floors_payload) else None
                secs0 = roof_data.get("sections") or []
                conns0 = roof_data.get("connections") or []
                secs_t = _translate_sections(secs0, float(dx), float(dy))
                secs_t = extend_sections_to_connect(secs_t, conns0) if (secs_t and conns0) else secs_t
                if secs_t and not use_shed_lower:
                    secs_t = extend_secondary_sections_to_main_ridge(secs_t)
                # Base (fără overhang) pentru cliparea pereților
                if use_shed_lower:
                    union_upper = None
                    if 0 <= fl < len(floors_payload):
                        polys_above = [floors_payload[i][1] for i in range(fl + 1, len(floors_payload))]
                        if polys_above:
                            union_upper = unary_union(polys_above)
                    high_sides = high_side_for_shed_from_upper_floor(secs_t, union_upper)
                    faces_base_fl = []
                    roof_angle_rad = np.radians(roof_angle_deg)
                    for s_idx, sec in enumerate(secs_t):
                        hs = high_sides[s_idx] if s_idx < len(high_sides) else "top"
                        for face in _roof_section_faces_shed(sec, float(z_base), roof_angle_rad, hs):
                            faces_base_fl.append({"vertices_3d": face})
                else:
                    cl_base = ridge_intersection_corner_lines(secs_t)
                    faces_base_fl = get_faces_3d_aframe_with_magenta(
                        secs_t, conns0, roof_angle_deg=roof_angle_deg, wall_height=float(z_base),
                        corner_lines=cl_base, floor_polygon=footprint_fl,
                    )
                if footprint_fl is not None:
                    faces_base_fl = clip_roof_faces_to_polygon(faces_base_fl, footprint_fl)
                roof_faces_base_by_floor[fl] = faces_base_fl
                if overhang_px > 0 and secs_t:
                    footprint = floors_payload[fl][1] if 0 <= fl < len(floors_payload) else None
                    free = (
                        compute_overhang_sides_from_footprint(secs_t, footprint)
                        if footprint is not None
                        else compute_overhang_sides_from_union_boundary(secs_t)
                    )
                    secs_t = apply_overhang_to_sections(secs_t, overhang_px=overhang_px, free_sides=free)
                if use_shed_lower:
                    union_upper = None
                    if 0 <= fl < len(floors_payload):
                        polys_above = [floors_payload[i][1] for i in range(fl + 1, len(floors_payload))]
                        if polys_above:
                            union_upper = unary_union(polys_above)
                    high_sides = high_side_for_shed_from_upper_floor(secs_t, union_upper)
                    faces = []
                    roof_angle_rad = np.radians(roof_angle_deg)
                    for s_idx, sec in enumerate(secs_t):
                        hs = high_sides[s_idx] if s_idx < len(high_sides) else "top"
                        for face in _roof_section_faces_shed(sec, float(z_base), roof_angle_rad, hs):
                            faces.append({"vertices_3d": face})
                else:
                    faces = rf.get_faces_3d_standard(secs_t, conns0, roof_angle_deg=roof_angle_deg, wall_height=float(z_base))
                if overhang_px > 0 and overhang_shift_whole_roof_down and faces:
                    import math
                    dz = float(math.tan(math.radians(roof_angle_deg)) * float(overhang_px))
                    for f in faces:
                        vs = f.get("vertices_3d") or []
                        f["vertices_3d"] = [[float(x), float(y), float(z) - dz] for x, y, z in vs]
                if faces and footprint_fl is not None:
                    faces = clip_roof_faces_to_polygon(faces, footprint_fl)
                roof_faces_by_floor[fl] = faces
        except Exception:
            pass

    tri_by_color: Dict[str, List[List[List[float]]]] = {}

    def add_face(face: List[List[float]], color_hex: str) -> None:
        for tri in _triangulate_face(face):
            tri_by_color.setdefault(color_hex, []).append(tri)

    gray = "#B0B0B0"
    xs_all: List[float] = []
    ys_all: List[float] = []
    gutter_endpoints: List[Tuple[float, float, float]] = []
    gutter_centerlines: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
    gutter_segment_sections: List[List[Dict[str, Any]]] = []
    gutter_closure_calls: List[Tuple[tuple, dict]] = []
    roof_base_tris: List[List[List[float]]] = []

    for floor_idx, (_p, floor_poly, rr, (ox, oy)) in enumerate(floors_payload):
        z0 = floor_idx * wall_height
        z1 = (floor_idx + 1) * wall_height

        floor_coords = list(floor_poly.exterior.coords) if hasattr(floor_poly, "exterior") else []
        xs_all.extend([c[0] for c in floor_coords])
        ys_all.extend([c[1] for c in floor_coords])

        union_above = None
        if floor_idx + 1 < len(floors_payload):
            union_above = unary_union([fp[1] for fp in floors_payload[floor_idx + 1 :]])

        secs = rr.get("sections") or []
        conns = rr.get("connections") or []

        kept: List[Dict[str, Any]] = []
        keep_ids: set[int] = set()
        for sec in secs:
            br = sec.get("bounding_rect", [])
            if len(br) < 3:
                continue
            try:
                sp = ShapelyPolygon(br)
                sp = shapely_affinity.translate(sp, xoff=ox, yoff=oy)
            except Exception:
                continue
            if _covered_by_upper(sp, union_above, area_thresh=500.0):
                continue
            sec_t = _translate_sections([sec], float(ox), float(oy))[0]
            kept.append(sec_t)
            try:
                keep_ids.add(int(sec_t.get("section_id")))
            except Exception:
                pass

        conns_kept = _filter_connections_by_sections(conns, keep_ids)
        # Top-floor roof only; lower floors come from `roof_levels` (remaining areas)
        draw_roof = True
        kept_extended = extend_sections_to_connect(kept, conns_kept) if (draw_roof and kept and conns_kept) else kept
        if draw_roof and kept_extended:
            kept_extended = extend_secondary_sections_to_main_ridge(kept_extended)
        corner_lines_base = ridge_intersection_corner_lines(kept_extended, floor_polygon=floor_poly) if draw_roof and kept_extended else []
        roof_faces_base = (
            get_faces_3d_aframe_with_magenta(
                kept_extended, conns_kept, roof_angle_deg=roof_angle_deg, wall_height=z1,
                corner_lines=corner_lines_base, floor_polygon=floor_poly,
            )
            if draw_roof and kept_extended
            else []
        )

        kept_use = kept_extended
        if draw_roof and overhang_px > 0 and kept_extended:
            free = (
                compute_overhang_sides_from_footprint(kept_extended, floor_poly)
                if floor_poly is not None and not getattr(floor_poly, "is_empty", True)
                else compute_overhang_sides_from_union_boundary(kept_extended)
            )
            kept_use = apply_overhang_to_sections(kept_extended, overhang_px=overhang_px, free_sides=free)
        corner_lines_use = ridge_intersection_corner_lines(kept_use, floor_polygon=floor_poly) if draw_roof and kept_use else []
        roof_faces = (
            get_faces_3d_aframe_with_magenta(
                kept_use, conns_kept, roof_angle_deg=roof_angle_deg, wall_height=z1,
                corner_lines=corner_lines_use, floor_polygon=floor_poly,
            )
            if draw_roof and kept_use
            else []
        )
        if roof_faces and floor_poly is not None:
            roof_faces = clip_roof_faces_to_polygon(roof_faces, floor_poly)
        if roof_faces_base and floor_poly is not None:
            roof_faces_base = clip_roof_faces_to_polygon(roof_faces_base, floor_poly)

        if overhang_px > 0 and overhang_shift_whole_roof_down and roof_faces:
            try:
                import math

                dz = float(math.tan(math.radians(roof_angle_deg)) * float(overhang_px))
                for f in roof_faces:
                    vs = f.get("vertices_3d") or []
                    if not vs:
                        continue
                    f["vertices_3d"] = [[float(x), float(y), float(z) - dz] for x, y, z in vs]
            except Exception:
                pass

        if overhang_px > 0 and overhang_keep_height and roof_faces_base and roof_faces:
            try:
                base_z = float(z1)
                maxz_base = max(float(v[2]) for f in roof_faces_base for v in (f.get("vertices_3d") or []))
                maxz_new = max(float(v[2]) for f in roof_faces for v in (f.get("vertices_3d") or []))
                d_base = maxz_base - base_z
                d_new = maxz_new - base_z
                if d_base > 1e-6 and d_new > 1e-6:
                    s = d_base / d_new
                    for f in roof_faces:
                        vs = f.get("vertices_3d") or []
                        f["vertices_3d"] = [[float(x), float(y), base_z + (float(z) - base_z) * s] for x, y, z in vs]
            except Exception:
                pass

        # Drop ONLY the overhang part below wall-top (z1), keeping the original roof unchanged.
        if overhang_px > 0 and overhang_drop_down and roof_faces and kept:
            try:
                import math, re

                tanv = float(math.tan(math.radians(roof_angle_deg)))
                peaks: Dict[int, float] = {}
                for f in roof_faces:
                    lbl = str(f.get("label") or "")
                    m = re.search(r"sec(\\d+)_", lbl)
                    if not m:
                        continue
                    si = int(m.group(1))
                    vs = f.get("vertices_3d") or []
                    if not vs:
                        continue
                    peaks[si] = max(peaks.get(si, float("-inf")), max(float(v[2]) for v in vs))

                for f in roof_faces:
                    lbl = str(f.get("label") or "")
                    m = re.search(r"sec(\\d+)_", lbl)
                    if not m:
                        continue
                    si = int(m.group(1))
                    if si < 0 or si >= len(kept):
                        continue
                    peak_z = float(peaks.get(si, z1))
                    base_sec = kept[si]
                    br0 = base_sec.get("bounding_rect") or []
                    if len(br0) < 3:
                        continue
                    xs0 = [float(p[0]) for p in br0]
                    ys0 = [float(p[1]) for p in br0]
                    minx0, maxx0 = min(xs0), max(xs0)
                    miny0, maxy0 = min(ys0), max(ys0)
                    orient = str(base_sec.get("ridge_orientation", "horizontal"))
                    ridge = base_sec.get("ridge_line") or []
                    if len(ridge) >= 2:
                        rx0, ry0 = float(ridge[0][0]), float(ridge[0][1])
                        rx1, ry1 = float(ridge[1][0]), float(ridge[1][1])
                        ridge_x = (rx0 + rx1) / 2.0
                        ridge_y = (ry0 + ry1) / 2.0
                    else:
                        ridge_x = (minx0 + maxx0) / 2.0
                        ridge_y = (miny0 + maxy0) / 2.0

                    if orient == "vertical":
                        try:
                            from shapely.geometry import box as _box

                            big = 1e6
                            band = _box(-big, miny0, big, maxy0)
                            fp = floor_poly
                            inter = fp.intersection(band) if fp is not None else None
                            if inter is not None and (not getattr(inter, "is_empty", True)):
                                minx_fp, _miny_fp, maxx_fp, _maxy_fp = inter.bounds
                                minx0, maxx0 = float(minx_fp), float(maxx_fp)
                        except Exception:
                            pass
                        d0 = max(abs(minx0 - ridge_x), abs(maxx0 - ridge_x))
                        def d_perp(x: float, y: float) -> float:  # noqa: ARG001
                            return abs(x - ridge_x)
                    else:
                        try:
                            from shapely.geometry import box as _box

                            big = 1e6
                            band = _box(minx0, -big, maxx0, big)
                            fp = floor_poly
                            inter = fp.intersection(band) if fp is not None else None
                            if inter is not None and (not getattr(inter, "is_empty", True)):
                                _minx_fp, miny_fp, _maxx_fp, maxy_fp = inter.bounds
                                miny0, maxy0 = float(miny_fp), float(maxy_fp)
                        except Exception:
                            pass
                        d0 = max(abs(miny0 - ridge_y), abs(maxy0 - ridge_y))
                        def d_perp(x: float, y: float) -> float:  # noqa: ARG001
                            return abs(y - ridge_y)

                    new_vs = []
                    for x, y, z in (f.get("vertices_3d") or []):
                        x = float(x); y = float(y); z = float(z)
                        d1 = float(d_perp(x, y))
                        if d1 > d0 + 1e-6:
                            z = peak_z - tanv * d1
                        new_vs.append([x, y, z])
                    f["vertices_3d"] = new_vs
            except Exception:
                pass

        # Pereți: clipăm la outline-ul acoperișului FĂRĂ overhang – nu depășim niciodată
        faces_for_wall_z = (
            roof_faces_base if (draw_roof and roof_faces_base) else roof_faces_base_by_floor.get(floor_idx, [])
        )
        if not faces_for_wall_z and (draw_roof and roof_faces):
            faces_for_wall_z = roof_faces
        if not faces_for_wall_z:
            faces_for_wall_z = roof_faces_by_floor.get(floor_idx, [])
        for i in range(len(floor_coords) - 1):
            p1, p2 = floor_coords[i], floor_coords[i + 1]
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])
            if faces_for_wall_z:
                # Pereți: min(z1, zt) – taiem orice bucată care iese peste outline-ul acoperișului
                n_pts = 13
                pts_top = []
                for k in range(n_pts):
                    t = k / (n_pts - 1) if n_pts > 1 else 1.0
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    zt = _z_roof_at(faces_for_wall_z, x, y, z1, tol=40.0)
                    pts_top.append([x, y, min(z1, zt)])
                # Poligon: jos (p1,p2), sus invers (de la p2 la p1 pe conturul acoperișului)
                face_pts = [[x1, y1, z0], [x2, y2, z0]] + [list(p) for p in reversed(pts_top)]
                for tri in _triangulate_fan(face_pts):
                    add_face(tri, gray)
            else:
                add_face(
                    [[x1, y1, z0], [x2, y2, z0], [x2, y2, z1], [x1, y1, z1]],
                    gray,
                )

        for idx, f in enumerate(roof_faces or []):
            verts = f.get("vertices_3d")
            if not verts:
                continue
            for vx, vy, _vz in verts:
                xs_all.append(float(vx))
                ys_all.append(float(vy))
            color_hex = colors[idx % len(colors)]
            add_face(verts, color_hex)

        if draw_roof and overhang_px > 0 and roof_faces_base:
            for f in roof_faces_base:
                verts = f.get("vertices_3d")
                if verts:
                    for tri in _triangulate_face(verts):
                        roof_base_tris.append(tri)

        # Drip edge a_frame: lipit de acoperiș, doar pe laturile libere (deschideri)
        if draw_roof and overhang_px > 0 and kept_use:
            roof_shift_dz = 0.0
            if overhang_shift_whole_roof_down:
                import math as _m
                roof_shift_dz = float(_m.tan(_m.radians(roof_angle_deg)) * float(overhang_px))
            free_drip = compute_overhang_sides_from_union_boundary(kept_use)
            drip_faces = get_drip_edge_faces_3d(
                kept_use, float(z1), overhang_px, roof_angle_deg,
                roof_shift_dz=roof_shift_dz, roof_faces=roof_faces, free_sides=free_drip
            )
            drip_color = "#8B4513"
            for df in drip_faces:
                v = df.get("vertices_3d")
                if v:
                    for vx, vy, _ in v:
                        xs_all.append(float(vx))
                        ys_all.append(float(vy))
                    add_face(v, drip_color)
            # Streașină colectoare pe streașină (eaves) – a-frame
            for gf in get_gutter_faces_3d(
                kept_use, float(z1), overhang_px, roof_angle_deg,
                roof_shift_dz=roof_shift_dz, roof_faces=roof_faces,
                include_eaves_only=True,
                eaves_z_lift=overhang_px * 0.60,
            ):
                v = gf.get("vertices_3d")
                if v:
                    for vx, vy, _ in v:
                        xs_all.append(float(vx))
                        ys_all.append(float(vy))
                    add_face(v, "#6B7280")
            gutter_closure_calls.append((
                (kept_use, float(z1), overhang_px, roof_angle_deg),
                {"roof_shift_dz": roof_shift_dz, "roof_faces": roof_faces, "include_eaves_only": True, "eaves_z_lift": overhang_px * 0.60},
            ))
            ep = get_gutter_endpoints_3d(
                kept_use, float(z1), overhang_px, roof_angle_deg,
                roof_shift_dz=roof_shift_dz, roof_faces=roof_faces,
                include_eaves_only=True,
                eaves_z_lift=overhang_px * 0.60,
            )
            gutter_endpoints.extend(ep)
            _kw = {"roof_shift_dz": roof_shift_dz, "roof_faces": roof_faces, "include_eaves_only": True, "eaves_z_lift": overhang_px * 0.60}
            for seg in get_gutter_centerlines_3d(kept_use, float(z1), overhang_px, roof_angle_deg, **_kw):
                gutter_centerlines.append(seg)
            for _ in range(len(ep) // 2):
                gutter_segment_sections.append(kept_use)

    # Add lower-floor roofs from `roof_levels` (remaining areas only)
    use_shed_lower = lower_floor_roof_mode == "shed"
    if use_shed_lower:
        from roof_calc.visualize_3d_pyvista import _roof_section_faces_shed  # type: ignore
        from roof_calc.overhang import high_side_for_shed_from_upper_floor

    if roof_levels:
        try:
            for z_base, roof_data, dx, dy, _fl in roof_levels:
                fl_idx = int(_fl) if _fl is not None else 0
                footprint_fl = floors_payload[fl_idx][1] if 0 <= fl_idx < len(floors_payload) else None
                secs0 = roof_data.get("sections") or []
                conns0 = roof_data.get("connections") or []
                secs_t = _translate_sections(secs0, float(dx), float(dy))
                secs_t_no_oh = secs_t
                if overhang_px > 0 and secs_t:
                    footprint = None
                    try:
                        if _fl is not None and 0 <= int(_fl) < len(floors_payload):
                            footprint = floors_payload[int(_fl)][1]
                    except Exception:
                        footprint = None
                    free = (
                        compute_overhang_sides_from_footprint(secs_t, footprint)
                        if footprint is not None
                        else compute_overhang_sides_from_union_boundary(secs_t)
                    )
                    secs_t = apply_overhang_to_sections(secs_t, overhang_px=overhang_px, free_sides=free)

                if use_shed_lower:
                    union_upper = None
                    if _fl is not None and 0 <= int(_fl) < len(floors_payload):
                        try:
                            polys_above = [floors_payload[i][1] for i in range(int(_fl) + 1, len(floors_payload))]
                            if polys_above:
                                union_upper = unary_union(polys_above)
                        except Exception:
                            pass
                    high_sides = high_side_for_shed_from_upper_floor(secs_t, union_upper)
                    faces = []
                    roof_angle_rad = np.radians(roof_angle_deg)
                    for s_idx, sec in enumerate(secs_t):
                        hs = high_sides[s_idx] if s_idx < len(high_sides) else "top"
                        for face in _roof_section_faces_shed(sec, float(z_base), roof_angle_rad, hs):
                            faces.append({"vertices_3d": face})
                    if overhang_px > 0 and secs_t_no_oh:
                        for s_idx, sec in enumerate(secs_t_no_oh):
                            hs = high_sides[s_idx] if s_idx < len(high_sides) else "top"
                            for face in _roof_section_faces_shed(sec, float(z_base), roof_angle_rad, hs):
                                for tri in _triangulate_face(face):
                                    roof_base_tris.append(tri)
                    # Drip edge pentru shed: peste tot în afară de latura cu z minim (per secțiune)
                    if overhang_px > 0 and secs_t:
                        import math as _m
                        _low = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
                        exclude_low_sides = [
                            _low.get(high_sides[i] if i < len(high_sides) else "top", "bottom")
                            for i in range(len(secs_t))
                        ]
                        roof_shift_dz = float(_m.tan(_m.radians(roof_angle_deg)) * float(overhang_px)) if overhang_shift_whole_roof_down else 0.0
                        free_drip = free if overhang_px > 0 else compute_overhang_sides_from_union_boundary(secs_t)
                        for df in get_drip_edge_faces_3d(
                            secs_t, float(z_base), overhang_px, roof_angle_deg,
                            roof_shift_dz=roof_shift_dz, roof_faces=faces, free_sides=free_drip,
                            exclude_low_sides=exclude_low_sides,
                        ):
                            v = df.get("vertices_3d")
                            if v:
                                for vx, vy, _ in v:
                                    xs_all.append(float(vx))
                                    ys_all.append(float(vy))
                                add_face(v, "#8B4513")
                        # Streașină colectoare (cilindru tăiat la jumătate) pe latura fără drip
                        for gf in get_gutter_faces_3d(
                            secs_t, float(z_base), overhang_px, roof_angle_deg,
                            roof_shift_dz=roof_shift_dz, roof_faces=faces,
                            exclude_low_sides=exclude_low_sides,
                        ):
                            v = gf.get("vertices_3d")
                            if v:
                                for vx, vy, _ in v:
                                    xs_all.append(float(vx))
                                    ys_all.append(float(vy))
                                add_face(v, "#6B7280")
                        gutter_closure_calls.append((
                            (secs_t, float(z_base), overhang_px, roof_angle_deg),
                            {"roof_shift_dz": roof_shift_dz, "roof_faces": faces, "exclude_low_sides": exclude_low_sides, "eaves_z_lift": overhang_px * 0.60},
                        ))
                        ep = get_gutter_endpoints_3d(
                            secs_t, float(z_base), overhang_px, roof_angle_deg,
                            roof_shift_dz=roof_shift_dz, roof_faces=faces,
                            exclude_low_sides=exclude_low_sides,
                        )
                        gutter_endpoints.extend(ep)
                        _kw_s = {"roof_shift_dz": roof_shift_dz, "roof_faces": faces, "exclude_low_sides": exclude_low_sides}
                        for seg in get_gutter_centerlines_3d(secs_t, float(z_base), overhang_px, roof_angle_deg, **_kw_s):
                            gutter_centerlines.append(seg)
                        for _ in range(len(ep) // 2):
                            gutter_segment_sections.append(secs_t)
                else:
                    cl_af = ridge_intersection_corner_lines(secs_t)
                    faces = get_faces_3d_aframe_with_magenta(
                        secs_t, conns0, roof_angle_deg=roof_angle_deg, wall_height=float(z_base),
                        corner_lines=cl_af, floor_polygon=footprint_fl,
                    )
                    if overhang_px > 0 and secs_t_no_oh:
                        cl_af_base = ridge_intersection_corner_lines(secs_t_no_oh)
                        faces_base_af = get_faces_3d_aframe_with_magenta(
                            secs_t_no_oh, conns0, roof_angle_deg=roof_angle_deg, wall_height=float(z_base),
                            corner_lines=cl_af_base, floor_polygon=footprint_fl,
                        )
                        for f in faces_base_af:
                            verts = f.get("vertices_3d")
                            if verts:
                                for tri in _triangulate_face(verts):
                                    roof_base_tris.append(tri)

                if overhang_px > 0 and overhang_shift_whole_roof_down and faces:
                    try:
                        import math

                        dz = float(math.tan(math.radians(roof_angle_deg)) * float(overhang_px))
                        for f in faces:
                            vs = f.get("vertices_3d") or []
                            if not vs:
                                continue
                            f["vertices_3d"] = [[float(x), float(y), float(z) - dz] for x, y, z in vs]
                    except Exception:
                        pass
                # drop only the overhang part below this level's wall-top (gable logic; skip for shed)
                if not use_shed_lower and overhang_px > 0 and overhang_drop_down and faces and secs0:
                    try:
                        import math, re

                        tanv = float(math.tan(math.radians(roof_angle_deg)))
                        base_secs_local = _translate_sections(secs0, float(dx), float(dy))
                        peaks: Dict[int, float] = {}
                        for f in faces:
                            lbl = str(f.get("label") or "")
                            m = re.search(r"sec(\\d+)_", lbl)
                            if not m:
                                continue
                            si = int(m.group(1))
                            vs = f.get("vertices_3d") or []
                            if not vs:
                                continue
                            peaks[si] = max(peaks.get(si, float("-inf")), max(float(v[2]) for v in vs))
                        for f in faces:
                            lbl = str(f.get("label") or "")
                            m = re.search(r"sec(\\d+)_", lbl)
                            if not m:
                                continue
                            si = int(m.group(1))
                            if si < 0 or si >= len(base_secs_local):
                                continue
                            peak_z = float(peaks.get(si, float(z_base)))
                            base_sec = base_secs_local[si]
                            br0 = base_sec.get("bounding_rect") or []
                            if len(br0) < 3:
                                continue
                            xs0 = [float(p[0]) for p in br0]
                            ys0 = [float(p[1]) for p in br0]
                            minx0, maxx0 = min(xs0), max(xs0)
                            miny0, maxy0 = min(ys0), max(ys0)
                            orient = str(base_sec.get("ridge_orientation", "horizontal"))
                            ridge = base_sec.get("ridge_line") or []
                            if len(ridge) >= 2:
                                rx0, ry0 = float(ridge[0][0]), float(ridge[0][1])
                                rx1, ry1 = float(ridge[1][0]), float(ridge[1][1])
                                ridge_x = (rx0 + rx1) / 2.0
                                ridge_y = (ry0 + ry1) / 2.0
                            else:
                                ridge_x = (minx0 + maxx0) / 2.0
                                ridge_y = (miny0 + maxy0) / 2.0

                            if orient == "vertical":
                                d0 = max(abs(minx0 - ridge_x), abs(maxx0 - ridge_x))
                                def d_perp(x: float, y: float) -> float:  # noqa: ARG001
                                    return abs(x - ridge_x)
                            else:
                                d0 = max(abs(miny0 - ridge_y), abs(maxy0 - ridge_y))
                                def d_perp(x: float, y: float) -> float:  # noqa: ARG001
                                    return abs(y - ridge_y)

                            new_vs = []
                            for x, y, z in (f.get("vertices_3d") or []):
                                x = float(x); y = float(y); z = float(z)
                                d1 = float(d_perp(x, y))
                                if d1 > d0 + 1e-6:
                                    z = peak_z - tanv * d1
                                new_vs.append([x, y, z])
                            f["vertices_3d"] = new_vs
                    except Exception:
                        pass
                for idx, f in enumerate(faces or []):
                    verts = f.get("vertices_3d")
                    if not verts:
                        continue
                    for vx, vy, _vz in verts:
                        xs_all.append(float(vx))
                        ys_all.append(float(vy))
                    color_hex = colors[idx % len(colors)]
                    add_face(verts, color_hex)

                # Drip edge a_frame la roof_levels (lipit de acoperiș, doar laturi libere)
                if not use_shed_lower and overhang_px > 0 and secs_t:
                    roof_shift_dz = 0.0
                    if overhang_shift_whole_roof_down:
                        import math as _m
                        roof_shift_dz = float(_m.tan(_m.radians(roof_angle_deg)) * float(overhang_px))
                    free_drip = free if overhang_px > 0 else compute_overhang_sides_from_union_boundary(secs_t)
                    drip_faces = get_drip_edge_faces_3d(
                        secs_t, float(z_base), overhang_px, roof_angle_deg,
                        roof_shift_dz=roof_shift_dz, roof_faces=faces, free_sides=free_drip
                    )
                    drip_color = "#8B4513"
                    for df in drip_faces:
                        v = df.get("vertices_3d")
                        if v:
                            for vx, vy, _ in v:
                                xs_all.append(float(vx))
                                ys_all.append(float(vy))
                            add_face(v, drip_color)
                    for gf in get_gutter_faces_3d(
                        secs_t, float(z_base), overhang_px, roof_angle_deg,
                        roof_shift_dz=roof_shift_dz, roof_faces=faces,
                        include_eaves_only=True,
                        eaves_z_lift=overhang_px * 0.60,
                    ):
                        v = gf.get("vertices_3d")
                        if v:
                            for vx, vy, _ in v:
                                xs_all.append(float(vx))
                                ys_all.append(float(vy))
                            add_face(v, "#6B7280")
                    gutter_closure_calls.append((
                        (secs_t, float(z_base), overhang_px, roof_angle_deg),
                        {"roof_shift_dz": roof_shift_dz, "roof_faces": faces, "include_eaves_only": True, "eaves_z_lift": overhang_px * 0.60},
                    ))
                    ep = get_gutter_endpoints_3d(
                        secs_t, float(z_base), overhang_px, roof_angle_deg,
                        roof_shift_dz=roof_shift_dz, roof_faces=faces,
                        include_eaves_only=True,
                        eaves_z_lift=overhang_px * 0.60,
                    )
                    gutter_endpoints.extend(ep)
                    _kw_r = {"roof_shift_dz": roof_shift_dz, "roof_faces": faces, "include_eaves_only": True, "eaves_z_lift": overhang_px * 0.60}
                    for seg in get_gutter_centerlines_3d(secs_t, float(z_base), overhang_px, roof_angle_deg, **_kw_r):
                        gutter_centerlines.append(seg)
                    for _ in range(len(ep) // 2):
                        gutter_segment_sections.append(secs_t)
        except Exception:
            pass

    # Burlani (cilindri verticali + legături la streașină) la fiecare colț exterior
    gutter_radius = max(2.0, overhang_px * 0.24) * 0.70 if overhang_px > 0 else None
    downspout_result = get_downspout_faces_for_floors(
        floors_payload,
        wall_height,
        cylinder_radius=gutter_radius,
        gutter_endpoints=gutter_endpoints,
        gutter_segment_sections=gutter_segment_sections if gutter_segment_sections else None,
        return_used_endpoints=True,
        return_downspout_centerlines=True,
    )
    if isinstance(downspout_result, tuple) and len(downspout_result) >= 3:
        downspout_faces, used_endpoints, downspout_centerlines_data = downspout_result[0], downspout_result[1], downspout_result[2]
    elif isinstance(downspout_result, tuple):
        downspout_faces, used_endpoints = downspout_result
        downspout_centerlines_data = {"downspout": [], "connection": []}
    else:
        downspout_faces, used_endpoints, downspout_centerlines_data = downspout_result, [], {"downspout": [], "connection": []}
    for df in downspout_faces:
        v = df.get("vertices_3d")
        if v:
            for vx, vy, _ in v:
                xs_all.append(float(vx))
                ys_all.append(float(vy))
            add_face(v, df.get("color", "#6B7280"))
    for args, kwargs in gutter_closure_calls:
        kwargs = dict(kwargs, downspout_endpoints=used_endpoints)
        for gf in get_gutter_end_closures_3d(*args, **kwargs):
            v = gf.get("vertices_3d")
            if v:
                for vx, vy, _ in v:
                    xs_all.append(float(vx))
                    ys_all.append(float(vy))
                add_face(v, "#6B7280")

    # bounds helper from roof rects too
    for _fp, _poly, rr, (ox, oy) in floors_payload:
        for sec in rr.get("sections") or []:
            br = sec.get("bounding_rect", [])
            if br:
                xs_all.extend([float(p[0]) + ox for p in br])
                ys_all.extend([float(p[1]) + oy for p in br])

    if not xs_all or not ys_all:
        return False
    # Bounds (after overhang + lower roofs) + padding to avoid clipping.
    minx_b, maxx_b = float(min(xs_all)), float(max(xs_all))
    miny_b, maxy_b = float(min(ys_all)), float(max(ys_all))
    pad_ratio = float(config.get("bounds_pad_ratio", 0.08))
    pad_min = float(config.get("bounds_pad_px", 60.0))
    span = max(maxx_b - minx_b, maxy_b - miny_b, 1.0)
    pad = max(pad_min, pad_ratio * span)
    minx_b, maxx_b = (minx_b - pad, maxx_b + pad)
    miny_b, maxy_b = (miny_b - pad, maxy_b + pad)

    n = max(1, len(views))
    fig = make_subplots(
        rows=1,
        cols=n,
        specs=[[{"type": "scene"} for _ in range(n)]],
        subplot_titles=[v.get("title", f"View {i+1}") for i, v in enumerate(views[:n])],
    )

    for col_idx in range(n):
        for color_hex, tris in tri_by_color.items():
            xs: List[float] = []
            ys: List[float] = []
            zs: List[float] = []
            ii: List[int] = []
            jj: List[int] = []
            kk: List[int] = []
            for tri in tris:
                base = len(xs)
                for vx, vy, vz in tri:
                    xs.append(float(vx))
                    ys.append(float(vy))
                    zs.append(float(vz))
                ii.append(base + 0)
                jj.append(base + 1)
                kk.append(base + 2)
            r01, g01, b01 = _hex_to_rgb01(color_hex)
            fig.add_trace(
                go.Mesh3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    i=ii,
                    j=jj,
                    k=kk,
                    color=f"rgb({int(r01*255)},{int(g01*255)},{int(b01*255)})",
                    opacity=1.0,
                    flatshading=True,
                    lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.9, specular=0.1),
                ),
                row=1,
                col=col_idx + 1,
            )

        if show_outlines:
            tris_all: List[List[List[float]]] = []
            for _c, tris in tri_by_color.items():
                tris_all.extend(tris)
            lx, ly, lz = _edge_lines_from_tris(tris_all, round_ndigits=3)
            if len(lx) > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=lx,
                        y=ly,
                        z=lz,
                        mode="lines",
                        line=dict(color=outline_color, width=outline_width),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=1,
                    col=col_idx + 1,
                )

        v = views[col_idx] if col_idx < len(views) else {"elev": 30, "azim": 45, "title": f"View {col_idx+1}"}
        eye = _camera_eye_from_elev_azim(float(v.get("elev", 30)), float(v.get("azim", 45)), scale=1.8)
        scene_name = "scene" if col_idx == 0 else f"scene{col_idx+1}"
        fig.update_layout(
            **{
                scene_name: dict(
                    xaxis=dict(range=[minx_b, maxx_b], backgroundcolor="white", gridcolor="lightgrey"),
                    yaxis=dict(range=[miny_b, maxy_b], backgroundcolor="white", gridcolor="lightgrey"),
                    zaxis=dict(range=[0, z_max], backgroundcolor="white", gridcolor="lightgrey"),
                    aspectmode="data",
                    camera=dict(
                        eye=eye,
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        projection=dict(type="perspective"),
                    ),
                )
            }
        )

    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    ok_png = False
    ok_html = False

    # Always try HTML (interactive in browser). Does NOT require Kaleido/Chrome.
    if html_output_path:
        try:
            Path(html_output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(html_output_path, include_plotlyjs="cdn", full_html=True)
            logger.info("Saved Plotly 3D standard HTML to %s", html_output_path)
            ok_html = True
        except Exception as e:  # pragma: no cover
            logger.warning("Plotly write_html failed (standard): %s", e)

    wireframe_html_path = config.get("wireframe_html_path")
    if wireframe_html_path and tri_by_color:
        try:
            _write_wireframe_html(tri_by_color, minx_b, maxx_b, miny_b, maxy_b, z_max, wireframe_html_path)
        except Exception as e:  # pragma: no cover
            logger.warning("Wireframe HTML failed (standard): %s", e)

    schematic_output_path = config.get("schematic_output_path")
    if schematic_output_path and tri_by_color and gutter_endpoints:
        try:
            _write_schematic_3d(
                tri_by_color,
                gutter_endpoints,
                minx_b, maxx_b, miny_b, maxy_b, z_max,
                output_path=str(schematic_output_path),
                html_path=str(config.get("schematic_html_path", "")) or None,
                config=config,
                gutter_centerlines=gutter_centerlines if gutter_centerlines else None,
                downspout_centerlines=downspout_centerlines_data,
                roof_base_tris=roof_base_tris if roof_base_tris else None,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("Schematic 3D failed (standard): %s", e)

    schematic_with_walls_path = config.get("schematic_with_walls_output_path")
    if schematic_with_walls_path and tri_by_color and gutter_endpoints and floors_payload:
        try:
            _write_schematic_3d(
                tri_by_color,
                gutter_endpoints,
                minx_b, maxx_b, miny_b, maxy_b, z_max,
                output_path=str(schematic_with_walls_path),
                html_path=str(config.get("schematic_with_walls_html_path", "")) or None,
                config=config,
                gutter_centerlines=gutter_centerlines if gutter_centerlines else None,
                downspout_centerlines=downspout_centerlines_data,
                roof_base_tris=roof_base_tris if roof_base_tris else None,
                floors_payload=floors_payload,
                wall_height=wall_height,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("Schematic 3D with walls failed (standard): %s", e)

    # Try PNG export (may fail if Kaleido/Chrome issues)
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(output_path, width=1200 * n, height=900, scale=2)
        logger.info("Saved Plotly 3D standard to %s", output_path)
        ok_png = True
    except Exception as e:  # pragma: no cover
        logger.warning("Plotly write_image failed (standard): %s", e)

    # Return True only if PNG was written (so caller can fallback to Matplotlib for PNG)
    return ok_png if output_path else ok_html


def visualize_3d_shed_plotly(
    output_path: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    all_floor_paths: Optional[List[str]] = None,
    floor_roof_results: Optional[List[Dict[str, Any]]] = None,
    html_output_path: Optional[str] = None,
    roof_levels: Optional[List[Tuple[float, Dict[str, Any], float, float, int]]] = None,
) -> bool:
    """
    Randare 3D acoperiș într-o apă (shed): ultimul etaj = a_frame, restul = o singură pantă.
    Creasta pe latura cea mai apropiată de etajul superior.
    """
    return visualize_3d_standard_plotly(
        output_path,
        config=config,
        all_floor_paths=all_floor_paths,
        floor_roof_results=floor_roof_results,
        html_output_path=html_output_path,
        roof_levels=roof_levels,
        lower_floor_roof_mode="shed",
    )


def visualize_3d_pyramid_plotly(
    wall_mask: Any,
    roof_data: Dict[str, Any],
    output_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    all_floor_paths: Optional[List[str]] = None,
    floor_roof_results: Optional[List[Dict[str, Any]]] = None,
    roof_levels: Optional[List[Tuple[float, Dict[str, Any], float, float]]] = None,
    html_output_path: Optional[str] = None,
) -> bool:
    """
    Randare 3D piramidă cu Plotly + Kaleido, salvează PNG la output_path.
    Returnează True dacă a reușit.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        logger.debug("Plotly not available: %s", e)
        return False

    from roof_calc.visualize import _ends_adjacent_to_upper_floor, _free_roof_ends
    # Refolosim generatorul de fețe din modulul PyVista (nu importă pyvista la import-time)
    from roof_calc.visualize_3d_pyvista import _roof_section_faces_pyramid  # type: ignore
    from roof_calc.overhang import (
        apply_overhang_to_sections,
        compute_overhang_sides_from_footprint,
        compute_overhang_sides_from_free_ends,
        get_downspout_faces_pyramid,
        get_gutter_centerlines_3d,
        get_gutter_end_closures_3d,
        get_gutter_endpoints_3d,
        get_gutter_faces_3d,
        get_pyramid_corner_hemispheres_3d,
    )

    import cv2
    from roof_calc.flood_fill import flood_fill_interior, get_house_shape_mask
    from roof_calc.geometry import extract_polygon
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union
    from shapely import affinity as shapely_affinity

    config = config or {}
    roof_angle_deg = config.get("roof_angle", 30.0)
    wall_height = config.get("wall_height", 300)
    colors = config.get("colors", ["#FF6B6B", "#4ECDC4", "#45B7D1"])
    show_outlines = bool(config.get("show_outlines", True))
    outline_color = str(config.get("outline_color", "rgba(0,0,0,0.65)"))
    outline_width = int(config.get("outline_width", 3))
    overhang_px = float(config.get("overhang_px", 0.0))
    overhang_keep_height = bool(config.get("overhang_keep_height", True))
    overhang_drop_down = bool(config.get("overhang_drop_down", True))
    overhang_shift_whole_roof_down = bool(config.get("overhang_shift_whole_roof_down", False))
    views = config.get(
        "views",
        [{"elev": 30, "azim": 45, "title": "Sud-Est"}, {"elev": 20, "azim": 225, "title": "Nord-Vest"}],
    )
    roof_angle_rad = np.radians(roof_angle_deg)

    wall_mask_path = wall_mask if isinstance(wall_mask, str) else None

    def _polygon_from_path(path: str) -> Optional[Any]:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        filled0 = flood_fill_interior(im)
        house0 = get_house_shape_mask(filled0)
        poly0 = extract_polygon(house0)
        if poly0 is None or poly0.is_empty:
            return None
        return poly0

    def _largest_section_bbox(rr: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
        best = None
        best_area = -1
        for sec in rr.get("sections") or []:
            rect = sec.get("bounding_rect", [])
            if len(rect) < 3:
                continue
            xs = [p[0] for p in rect]
            ys = [p[1] for p in rect]
            minx, maxx = int(min(xs)), int(max(xs))
            miny, maxy = int(min(ys)), int(max(ys))
            area = (maxx - minx) * (maxy - miny)
            if area > best_area:
                best_area = area
                best = (minx, miny, maxx, maxy)
        return best

    def _bbox_center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def _compute_offsets(paths: List[str], results: List[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
        """Etaje cu aceeași mărime și formă se pun unul peste altul: aliniem centrele (centroid poligon)."""
        polys = [_polygon_from_path(p) for p in paths]
        best_idx = 0
        best_area = -1.0
        for i, poly in enumerate(polys):
            if poly is not None and not getattr(poly, "is_empty", True):
                a = float(getattr(poly, "area", 0) or 0)
                if a > best_area:
                    best_area = a
                    best_idx = i
        ref_poly = polys[best_idx] if best_idx < len(polys) else None
        if ref_poly is None or getattr(ref_poly, "is_empty", True):
            cx_ref, cy_ref = 0.0, 0.0
        else:
            c = ref_poly.centroid
            cx_ref, cy_ref = float(c.x), float(c.y)
        out: Dict[str, Tuple[int, int]] = {}
        for p, poly in zip(paths, polys):
            if poly is None or getattr(poly, "is_empty", True):
                out[p] = (int(round(-cx_ref)), int(round(-cy_ref)))
            else:
                c = poly.centroid
                ox = int(round(cx_ref - c.x))
                oy = int(round(cy_ref - c.y))
                out[p] = (ox, oy)
        return out

    def _translate_sections(secs: List[Dict[str, Any]], dx: float, dy: float) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for sec in secs:
            rect = sec.get("bounding_rect", [])
            ridge = sec.get("ridge_line", [])
            out.append(
                {
                    "bounding_rect": [(float(p[0]) + dx, float(p[1]) + dy) for p in rect] if rect else [],
                    "ridge_line": [(float(p[0]) + dx, float(p[1]) + dy) for p in ridge] if ridge and len(ridge) >= 2 else [],
                    "ridge_orientation": sec.get("ridge_orientation", "horizontal"),
                    "is_main": sec.get("is_main", False),
                }
            )
        return out

    # New multi-floor path: use per-floor roof_results (rectangles_floor) + overlay offsets
    floors_payload: List[Tuple[str, Any, Dict[str, Any], Tuple[int, int]]] = []
    if (
        wall_mask_path
        and all_floor_paths
        and floor_roof_results
        and len(all_floor_paths) == len(floor_roof_results)
    ):
        offsets = _compute_offsets(all_floor_paths, floor_roof_results)
        for p, rr in zip(all_floor_paths, floor_roof_results):
            poly0 = _polygon_from_path(p)
            if poly0 is None:
                continue
            ox, oy = offsets.get(p, (0, 0))
            poly_t = shapely_affinity.translate(poly0, xoff=ox, yoff=oy)
            floors_payload.append((p, poly_t, rr, (ox, oy)))

    # Fallback single-floor polygon
    if not floors_payload:
        if isinstance(wall_mask, str):
            img = cv2.imread(wall_mask, cv2.IMREAD_GRAYSCALE)
        else:
            img = np.asarray(wall_mask, dtype=np.uint8)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is None:
            return False
        filled = flood_fill_interior(img)
        house_mask = get_house_shape_mask(filled)
        polygon = extract_polygon(house_mask)
        if polygon is None or polygon.is_empty:
            return False
        floors_payload = [(wall_mask_path or "floor0", polygon, roof_data, (0, 0))]

    # Sort floors bottom->top by footprint area (largest first)
    floors_payload.sort(key=lambda t: float(getattr(t[1], "area", 0.0) or 0.0), reverse=True)
    num_floors = len(floors_payload)
    z_max = num_floors * wall_height + 500

    # Scene bounds (from all polygons + all section rects)
    xs_all: List[float] = []
    ys_all: List[float] = []
    gutter_endpoints_pyr: List[Tuple[float, float, float]] = []
    gutter_centerlines_pyr: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
    gutter_segment_sections_pyr: List[List[Dict[str, Any]]] = []
    pyramid_gutter_closure_calls: List[tuple] = []
    for _p, poly, rr, (ox, oy) in floors_payload:
        coords = list(poly.exterior.coords) if hasattr(poly, "exterior") else []
        xs_all.extend([c[0] for c in coords])
        ys_all.extend([c[1] for c in coords])
        for sec in rr.get("sections") or []:
            rect = sec.get("bounding_rect", [])
            xs_all.extend([float(p[0]) + ox for p in rect] if rect else [])
            ys_all.extend([float(p[1]) + oy for p in rect] if rect else [])
    if not xs_all or not ys_all:
        return False
    minx_b, maxx_b = float(min(xs_all)), float(max(xs_all))
    miny_b, maxy_b = float(min(ys_all)), float(max(ys_all))

    # Construim fețe (triunghiuri) grupate pe culoare
    tri_by_color: Dict[str, List[List[List[float]]]] = {}

    def add_face(face: List[List[float]], color_hex: str) -> None:
        for tri in _triangulate_face(face):
            tri_by_color.setdefault(color_hex, []).append(tri)

    gray = "#B0B0B0"
    roof_base_tris_pyr: List[List[List[float]]] = []

    secs_overlay_pyr: List[Dict[str, Any]] = []
    # Pereți + acoperișuri per etaj, bazat pe `rectangles_floor` (floor_roof_results)
    for floor_idx, (_p, floor_poly, rr, (ox, oy)) in enumerate(floors_payload):
        z0 = floor_idx * wall_height
        z1 = (floor_idx + 1) * wall_height
        coords = list(floor_poly.exterior.coords) if hasattr(floor_poly, "exterior") else []

        # union of upper floors footprint (overlay coords)
        union_above = None
        if floor_idx + 1 < len(floors_payload):
            union_above = unary_union([fp[1] for fp in floors_payload[floor_idx + 1 :]])

        # upper floor sections translated into this floor coords (for end suppression)
        upper_secs_all: List[Dict[str, Any]] = []
        for _pp, _poly_u, rr_u, (ox_u, oy_u) in floors_payload[floor_idx + 1 :]:
            dxu = float(ox_u - ox)
            dyu = float(oy_u - oy)
            upper_secs_all.extend(_translate_sections(rr_u.get("sections") or [], dxu, dyu))

        # Top-floor roof: draw directly. Lower floors: via roof_levels (same as standard)
        draw_roof = True

        secs_base = rr.get("sections") or []
        conns = rr.get("connections") or []
        free_ends = _free_roof_ends(secs_base, conns)

        secs = secs_base
        if draw_roof and overhang_px > 0 and secs_base:
            free = compute_overhang_sides_from_free_ends(secs_base, free_ends)
            secs = apply_overhang_to_sections(secs_base, overhang_px=overhang_px, free_sides=free)

        sections_to_draw: List[Tuple[int, Any]] = [(i, sec) for i, sec in enumerate(secs)] if draw_roof else []

        section_face_data: List[Tuple[str, Any]] = []

        for s_idx, sec in sections_to_draw:
            br = sec.get("bounding_rect", [])
            if len(br) < 3:
                continue
            try:
                sp = ShapelyPolygon(br)
                sp = shapely_affinity.translate(sp, xoff=ox, yoff=oy)
            except Exception:
                continue
            if union_above is not None and not getattr(union_above, "is_empty", True):
                try:
                    if float(sp.difference(union_above).area) < 500.0:
                        continue
                except Exception:
                    pass

            # draw with translated section coords
            sec_t = {
                **sec,
                "bounding_rect": [(float(p[0]) + ox, float(p[1]) + oy) for p in br],
                "ridge_line": [
                    (float(p[0]) + ox, float(p[1]) + oy) for p in (sec.get("ridge_line") or [])
                ],
            }
            color_hex = colors[s_idx % len(colors)]
            fe = free_ends[s_idx] if s_idx < len(free_ends) else {}
            base_sec_for_ridge = None
            if overhang_px > 0 and s_idx < len(secs_base):
                sec0 = secs_base[s_idx]
                br0 = sec0.get("bounding_rect", [])
                base_sec_for_ridge = {
                    **sec0,
                    "bounding_rect": [(float(p[0]) + ox, float(p[1]) + oy) for p in br0],
                    "ridge_line": [(float(p[0]) + ox, float(p[1]) + oy) for p in (sec0.get("ridge_line") or [])],
                }
            faces_oh = _roof_section_faces_pyramid(
                sec_t, z1, roof_angle_rad, fe, upper_secs_all, base_sec_for_ridge=base_sec_for_ridge
            )

            # Ca la gable: shift întreg acoperișul în jos, apoi keep_height restaurează vârful – partea de jos rămâne mai jos
            if overhang_px > 0 and overhang_shift_whole_roof_down and faces_oh:
                try:
                    import math

                    dz = float(math.tan(roof_angle_rad) * float(overhang_px))
                    faces_oh = [
                        [[float(x), float(y), float(z) - dz] for x, y, z in face] for face in faces_oh
                    ]
                except Exception:
                    pass

            # Baseline faces for same section (no overhang) to keep peak height
            if overhang_px > 0:
                try:
                    sec0 = secs_base[s_idx]
                    br0 = sec0.get("bounding_rect", [])
                    sec0_t = {
                        **sec0,
                        "bounding_rect": [(float(p[0]) + ox, float(p[1]) + oy) for p in br0],
                        "ridge_line": [
                            (float(p[0]) + ox, float(p[1]) + oy) for p in (sec0.get("ridge_line") or [])
                        ],
                    }
                    faces_base = _roof_section_faces_pyramid(sec0_t, z1, roof_angle_rad, fe, upper_secs_all)
                    for face in (faces_base or []):
                        for tri in _triangulate_face(face):
                            roof_base_tris_pyr.append(tri)
                except Exception:
                    pass
            if overhang_px > 0 and overhang_keep_height:
                try:
                    sec0 = secs_base[s_idx]
                    br0 = sec0.get("bounding_rect", [])
                    sec0_t = {
                        **sec0,
                        "bounding_rect": [(float(p[0]) + ox, float(p[1]) + oy) for p in br0],
                        "ridge_line": [
                            (float(p[0]) + ox, float(p[1]) + oy) for p in (sec0.get("ridge_line") or [])
                        ],
                    }
                    faces_base = _roof_section_faces_pyramid(sec0_t, z1, roof_angle_rad, fe, upper_secs_all)
                    base_z = float(z1)
                    maxz_base = max(float(p[2]) for face in faces_base for p in face) if faces_base else None
                    maxz_new = max(float(p[2]) for face in faces_oh for p in face) if faces_oh else None
                    if maxz_base is not None and maxz_new is not None:
                        d_base = float(maxz_base) - base_z
                        d_new = float(maxz_new) - base_z
                        if d_base > 1e-6 and d_new > 1e-6:
                            s = d_base / d_new
                            faces_oh = [
                                [[float(x), float(y), base_z + (float(z) - base_z) * s] for x, y, z in face]
                                for face in faces_oh
                            ]
                except Exception:
                    pass

            # Drop ONLY the overhang part below wall-top (z1), keeping the original roof unchanged.
            # Pentru piramidă: folosim distanța 2D de la vârf (centru), nu creasta – toate 4 laturile sunt streașini.
            if overhang_px > 0 and overhang_drop_down and faces_oh:
                try:
                    import math

                    tanv = float(math.tan(roof_angle_rad))
                    base_sec_t = None
                    try:
                        sec0 = secs_base[s_idx]
                        br0 = sec0.get("bounding_rect", [])
                        base_sec_t = {
                            **sec0,
                            "bounding_rect": [(float(p[0]) + ox, float(p[1]) + oy) for p in br0],
                            "ridge_line": [(float(p[0]) + ox, float(p[1]) + oy) for p in (sec0.get("ridge_line") or [])],
                        }
                    except Exception:
                        base_sec_t = sec_t

                    br0t = (base_sec_t.get("bounding_rect") or []) if base_sec_t else []
                    if len(br0t) >= 3:
                        xs0 = [float(p[0]) for p in br0t]
                        ys0 = [float(p[1]) for p in br0t]
                        minx0, maxx0 = min(xs0), max(xs0)
                        miny0, maxy0 = min(ys0), max(ys0)
                        ridge = (base_sec_t or {}).get("ridge_line") or []
                        if len(ridge) >= 2:
                            ridge_x = (float(ridge[0][0]) + float(ridge[1][0])) / 2.0
                            ridge_y = (float(ridge[0][1]) + float(ridge[1][1])) / 2.0
                        else:
                            ridge_x = (minx0 + maxx0) / 2.0
                            ridge_y = (miny0 + maxy0) / 2.0

                        peak_z = max(float(p[2]) for face in faces_oh for p in face)
                        tol = 1e-6
                        for face in faces_oh:
                            for p in face:
                                x, y = float(p[0]), float(p[1])
                                in_overhang = x < minx0 - tol or x > maxx0 + tol or y < miny0 - tol or y > maxy0 + tol
                                if in_overhang:
                                    d_peak = math.sqrt((x - ridge_x) ** 2 + (y - ridge_y) ** 2)
                                    p[2] = float(peak_z - tanv * d_peak)
                except Exception:
                    pass

            # Nu aplicăm shift la etaj (top floor) - îl coboară prea mult. Parter: în roof_levels.

            # include final face vertices in bounds
            for face in faces_oh:
                for vx, vy, _vz in face:
                    xs_all.append(float(vx))
                    ys_all.append(float(vy))
            section_face_data.append((color_hex, faces_oh))

        # Pereți: clipăm la outline-ul acoperișului FĂRĂ overhang (piramidă)
        pyramid_faces_base: List[Dict[str, Any]] = []
        if draw_roof and secs_base:
            for s_idx, sec0 in enumerate(secs_base):
                br0 = sec0.get("bounding_rect") or []
                if len(br0) < 3:
                    continue
                sec0_t = {
                    **sec0,
                    "bounding_rect": [(float(p[0]) + ox, float(p[1]) + oy) for p in br0],
                    "ridge_line": [(float(p[0]) + ox, float(p[1]) + oy) for p in (sec0.get("ridge_line") or [])],
                }
                fe = free_ends[s_idx] if s_idx < len(free_ends) else {}
                faces_base_pyr = _roof_section_faces_pyramid(sec0_t, z1, roof_angle_rad, fe, upper_secs_all)
                for f in (faces_base_pyr or []):
                    pyramid_faces_base.append({"vertices_3d": f})
        faces_for_wall_z = pyramid_faces_base if pyramid_faces_base else [{"vertices_3d": f} for faces in [d[1] for d in section_face_data] for f in faces]
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])
            if draw_roof and faces_for_wall_z:
                n_pts = 13
                pts_top = []
                for k in range(n_pts):
                    t = k / (n_pts - 1) if n_pts > 1 else 1.0
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    zt = _z_roof_at(faces_for_wall_z, x, y, z1, tol=40.0)
                    pts_top.append([x, y, min(z1, zt)])
                face_pts = [[x1, y1, z0], [x2, y2, z0]] + [list(p) for p in reversed(pts_top)]
                for tri in _triangulate_fan(face_pts):
                    add_face(tri, gray)
            else:
                add_face(
                    [[x1, y1, z0], [x2, y2, z0], [x2, y2, z1], [x1, y1, z1]],
                    gray,
                )

        for _color, faces in section_face_data:
            for f in faces:
                add_face(f, _color)

        # Streașină colectoare pe eaves – pyramid main roof
        if draw_roof and overhang_px > 0 and secs and section_face_data:
            secs_overlay_pyr = [
                {
                    **sec,
                    "bounding_rect": [(float(p[0]) + ox, float(p[1]) + oy) for p in (sec.get("bounding_rect") or [])],
                    "ridge_line": [(float(p[0]) + ox, float(p[1]) + oy) for p in (sec.get("ridge_line") or [])],
                }
                for sec in secs
            ]
            all_faces = [{"vertices_3d": f} for _c, faces in section_face_data for f in faces]
            parter_secs_for_main: List[Dict[str, Any]] = []
            if roof_levels:
                for _zb, rd, _dx, _dy, _fl in roof_levels:
                    parter_secs_for_main.extend(
                        _translate_sections(rd.get("sections") or [], float(_dx), float(_dy))
                    )
            import math as _m_pyr2
            roof_shift_dz = float(_m_pyr2.tan(_m_pyr2.radians(roof_angle_deg)) * float(overhang_px)) if overhang_shift_whole_roof_down else 0.0
            for gf in get_gutter_faces_3d(
                secs_overlay_pyr, float(z1), overhang_px, roof_angle_deg,
                roof_shift_dz=roof_shift_dz, roof_faces=all_faces,
                pyramid_all_sides=True,
                pyramid_extend=True,
                eaves_z_lift=overhang_px * 0.60,
                interior_reference_sections=parter_secs_for_main if parter_secs_for_main else None,
            ):
                v = gf.get("vertices_3d")
                if v:
                    for vx, vy, _ in v:
                        xs_all.append(float(vx))
                        ys_all.append(float(vy))
                    add_face(v, "#6B7280")
            ep = get_gutter_endpoints_3d(
                secs_overlay_pyr, float(z1), overhang_px, roof_angle_deg,
                roof_shift_dz=roof_shift_dz, roof_faces=all_faces,
                pyramid_all_sides=True,
                eaves_z_lift=overhang_px * 0.60,
            )
            gutter_endpoints_pyr.extend(ep)
            for seg in get_gutter_centerlines_3d(
                secs_overlay_pyr, float(z1), overhang_px, roof_angle_deg,
                roof_shift_dz=roof_shift_dz, roof_faces=all_faces,
                pyramid_all_sides=True, pyramid_extend=True,
                eaves_z_lift=overhang_px * 0.60,
                interior_reference_sections=parter_secs_for_main if parter_secs_for_main else None,
            ):
                gutter_centerlines_pyr.append(seg)
            for _ in range(len(ep) // 2):
                gutter_segment_sections_pyr.append(secs_overlay_pyr)
            pyramid_gutter_closure_calls.append(
                (secs_overlay_pyr, float(z1), roof_shift_dz, all_faces)
            )

    interior_refs_pyr: List[Dict[str, Any]] = []
    # Add lower-floor pyramid roofs from roof_levels (diagonale identice cu pyramid_lines.png)
    if roof_levels:
        try:
            for z_base, rd, dx, dy, _fl in roof_levels:
                secs0 = rd.get("sections") or []
                conns0 = rd.get("connections") or []
                secs_t = _translate_sections(secs0, float(dx), float(dy))
                secs_t_no_oh = secs_t
                free_ends0 = _free_roof_ends(secs_t, conns0)
                if overhang_px > 0 and secs_t:
                    footprint = None
                    try:
                        if _fl is not None and 0 <= int(_fl) < len(floors_payload):
                            footprint = floors_payload[int(_fl)][1]
                    except Exception:
                        footprint = None
                    free = (
                        compute_overhang_sides_from_footprint(secs_t, footprint)
                        if footprint is not None
                        else compute_overhang_sides_from_free_ends(secs_t, free_ends0)
                    )
                    secs_t = apply_overhang_to_sections(secs_t, overhang_px=overhang_px, free_sides=free)
                free_ends1 = _free_roof_ends(secs_t, conns0)
                # Secțiuni etaj superior în overlay (ca pyramid_lines) - fără diagonală pe laturile lipite
                upper_secs_parter: List[Dict[str, Any]] = []
                if _fl is not None and 0 <= int(_fl) < len(floors_payload):
                    for _pp, _poly_u, rr_u, (ox_u, oy_u) in floors_payload[int(_fl) + 1 :]:
                        upper_secs_parter.extend(
                            _translate_sections(rr_u.get("sections") or [], float(ox_u), float(oy_u))
                        )
                if upper_secs_parter:
                    interior_refs_pyr.extend(upper_secs_parter)
                all_faces_floor: List[Dict[str, Any]] = []
                if overhang_px > 0 and secs_t_no_oh:
                    free_ends_no_oh = _free_roof_ends(secs_t_no_oh, conns0)
                    for s_idx, sec in enumerate(secs_t_no_oh):
                        fe = free_ends_no_oh[s_idx] if s_idx < len(free_ends_no_oh) else {}
                        faces_base_pyr = _roof_section_faces_pyramid(
                            sec, float(z_base), roof_angle_rad, fe, upper_secs_parter if upper_secs_parter else None
                        )
                        for face in (faces_base_pyr or []):
                            for tri in _triangulate_face(face):
                                roof_base_tris_pyr.append(tri)
                for s_idx, sec in enumerate(secs_t):
                    fe = free_ends1[s_idx] if s_idx < len(free_ends1) else {}
                    base_sec_ridge = (
                        secs_t_no_oh[s_idx] if overhang_px > 0 and s_idx < len(secs_t_no_oh) else None
                    )
                    faces = _roof_section_faces_pyramid(
                        sec,
                        float(z_base),
                        roof_angle_rad,
                        fe,
                        upper_secs_parter if upper_secs_parter else None,
                        base_sec_for_ridge=base_sec_ridge,
                    )
                    # Ca la gable: shift întreg acoperișul în jos, apoi keep_height restaurează vârful
                    if overhang_px > 0 and overhang_shift_whole_roof_down and faces:
                        try:
                            import math

                            dz = float(math.tan(roof_angle_rad) * float(overhang_px))
                            faces = [
                                [[float(x), float(y), float(z) - dz] for x, y, z in face]
                                for face in faces
                            ]
                        except Exception:
                            pass
                    if overhang_px > 0 and overhang_keep_height and faces and secs_t_no_oh and s_idx < len(secs_t_no_oh):
                        try:
                            sec0_lo = secs_t_no_oh[s_idx]
                            fe0 = free_ends_no_oh[s_idx] if s_idx < len(free_ends_no_oh) else {}
                            faces_base_lo = _roof_section_faces_pyramid(
                                sec0_lo, float(z_base), roof_angle_rad, fe0,
                                upper_secs_parter if upper_secs_parter else None,
                            )
                            if faces_base_lo:
                                base_z = float(z_base)
                                maxz_base = max(float(p[2]) for face in faces_base_lo for p in face)
                                maxz_new = max(float(p[2]) for face in faces for p in face)
                                d_base = maxz_base - base_z
                                d_new = maxz_new - base_z
                                if d_base > 1e-6 and d_new > 1e-6:
                                    s = d_base / d_new
                                    faces = [
                                        [[float(x), float(y), base_z + (float(z) - base_z) * s] for x, y, z in face]
                                        for face in faces
                                    ]
                        except Exception:
                            pass
                    if overhang_px > 0 and overhang_drop_down and faces:
                        try:
                            import math

                            sec0_lo = secs_t_no_oh[s_idx] if s_idx < len(secs_t_no_oh) else None
                            br0 = (sec0_lo or {}).get("bounding_rect") or []
                            if len(br0) >= 3:
                                xs0 = [float(p[0]) for p in br0]
                                ys0 = [float(p[1]) for p in br0]
                                minx0, maxx0 = min(xs0), max(xs0)
                                miny0, maxy0 = min(ys0), max(ys0)
                                ridge = (sec0_lo or {}).get("ridge_line") or []
                                if len(ridge) >= 2:
                                    ridge_x = (float(ridge[0][0]) + float(ridge[1][0])) / 2.0
                                    ridge_y = (float(ridge[0][1]) + float(ridge[1][1])) / 2.0
                                else:
                                    ridge_x = (minx0 + maxx0) / 2.0
                                    ridge_y = (miny0 + maxy0) / 2.0
                                tanv = float(math.tan(roof_angle_rad))
                                peak_z = max(float(p[2]) for face in faces for p in face)
                                tol = 1e-6
                                for face in faces:
                                    for p in face:
                                        x, y = float(p[0]), float(p[1])
                                        in_overhang = x < minx0 - tol or x > maxx0 + tol or y < miny0 - tol or y > maxy0 + tol
                                        if in_overhang:
                                            d_peak = math.sqrt((x - ridge_x) ** 2 + (y - ridge_y) ** 2)
                                            p[2] = float(peak_z - tanv * d_peak)
                        except Exception:
                            pass
                    for f in faces:
                        all_faces_floor.append({"vertices_3d": f})
                        for vx, vy, _vz in f:
                            xs_all.append(float(vx))
                            ys_all.append(float(vy))
                        add_face(f, colors[s_idx % len(colors)])
                # Streașină colectoare pe eaves – pyramid (exclude secțiuni fără acoperiș expus, ex. parter sub etaj)
                if overhang_px > 0 and secs_t and all_faces_floor:
                    pyr_include_mask = [
                        any(
                            (free_ends1[idx] if idx < len(free_ends1) else {}).get(s, True)
                            and s not in _ends_adjacent_to_upper_floor(sec, upper_secs_parter or [])
                            for s in ("top", "bottom", "left", "right")
                        )
                        for idx, sec in enumerate(secs_t)
                    ]
                    if any(pyr_include_mask):
                        import math as _m_pyr
                        roof_shift_dz = float(_m_pyr.tan(_m_pyr.radians(roof_angle_deg)) * float(overhang_px)) if overhang_shift_whole_roof_down else 0.0
                        for gf in get_gutter_faces_3d(
                            secs_t, float(z_base), overhang_px, roof_angle_deg,
                            roof_shift_dz=roof_shift_dz, roof_faces=all_faces_floor,
                            pyramid_all_sides=True,
                            pyramid_extend=True,
                            eaves_z_lift=overhang_px * 0.60,
                            sections_include_mask=pyr_include_mask,
                            interior_reference_sections=upper_secs_parter if upper_secs_parter else None,
                        ):
                            v = gf.get("vertices_3d")
                            if v:
                                for vx, vy, _ in v:
                                    xs_all.append(float(vx))
                                    ys_all.append(float(vy))
                                add_face(v, "#6B7280")
                        ep = get_gutter_endpoints_3d(
                            secs_t, float(z_base), overhang_px, roof_angle_deg,
                            roof_shift_dz=roof_shift_dz, roof_faces=all_faces_floor,
                            pyramid_all_sides=True,
                            eaves_z_lift=overhang_px * 0.60,
                            sections_include_mask=pyr_include_mask,
                        )
                        gutter_endpoints_pyr.extend(ep)
                        for seg in get_gutter_centerlines_3d(
                            secs_t, float(z_base), overhang_px, roof_angle_deg,
                            roof_shift_dz=roof_shift_dz, roof_faces=all_faces_floor,
                            pyramid_all_sides=True, pyramid_extend=True,
                            eaves_z_lift=overhang_px * 0.60,
                            sections_include_mask=pyr_include_mask,
                            interior_reference_sections=upper_secs_parter if upper_secs_parter else None,
                        ):
                            gutter_centerlines_pyr.append(seg)
                        for _ in range(len(ep) // 2):
                            gutter_segment_sections_pyr.append(secs_t)
                        pyramid_gutter_closure_calls.append(
                            (secs_t, float(z_base), roof_shift_dz, all_faces_floor)
                        )
        except Exception:
            pass

    gutter_radius_pyr = max(2.0, overhang_px * 0.24) * 0.70 if overhang_px > 0 else None
    downspout_result_pyr = get_downspout_faces_pyramid(
        floors_payload,
        wall_height,
        cylinder_radius=gutter_radius_pyr,
        gutter_endpoints=gutter_endpoints_pyr,
        gutter_segment_sections=gutter_segment_sections_pyr if gutter_segment_sections_pyr else None,
        return_used_gutter_endpoints=True,
        return_downspout_centerlines=True,
    )
    if isinstance(downspout_result_pyr, tuple) and len(downspout_result_pyr) >= 3:
        downspout_faces_pyr = downspout_result_pyr[0]
        used_gutter_endpoints_pyr = downspout_result_pyr[1]
        downspout_centerlines_data_pyr = downspout_result_pyr[2]
    elif isinstance(downspout_result_pyr, tuple):
        downspout_faces_pyr = downspout_result_pyr[0]
        used_gutter_endpoints_pyr = downspout_result_pyr[1]
        downspout_centerlines_data_pyr = {"downspout": [], "connection": []}
    else:
        downspout_faces_pyr = downspout_result_pyr
        used_gutter_endpoints_pyr = []
        downspout_centerlines_data_pyr = {"downspout": [], "connection": []}
    for df in downspout_faces_pyr:
        v = df.get("vertices_3d")
        if v:
            for vx, vy, _ in v:
                xs_all.append(float(vx))
                ys_all.append(float(vy))
            add_face(v, df.get("color", "#6B7280"))

    for cf in get_pyramid_corner_hemispheres_3d(
        gutter_endpoints_pyr,
        gutter_segment_sections_pyr,
        overhang_px,
        downspout_endpoints=used_gutter_endpoints_pyr,
        sections_for_centroid=pyramid_gutter_closure_calls[0][0] if pyramid_gutter_closure_calls else None,
        interior_reference_sections=interior_refs_pyr if interior_refs_pyr else None,
    ):
        v = cf.get("vertices_3d")
        if v:
            for vx, vy, _ in v:
                xs_all.append(float(vx))
                ys_all.append(float(vy))
            add_face(v, "#6B7280")

    # Finalize bounds (now that we included roof vertices) + add padding.
    minx_b, maxx_b = float(min(xs_all)), float(max(xs_all))
    miny_b, maxy_b = float(min(ys_all)), float(max(ys_all))
    pad_ratio = float(config.get("bounds_pad_ratio", 0.08))
    pad_min = float(config.get("bounds_pad_px", 60.0))
    span = max(maxx_b - minx_b, maxy_b - miny_b, 1.0)
    pad = max(pad_min, pad_ratio * span)
    minx_b, maxx_b = (minx_b - pad, maxx_b + pad)
    miny_b, maxy_b = (miny_b - pad, maxy_b + pad)

    # Build subplot fig
    n = max(1, len(views))
    fig = make_subplots(
        rows=1,
        cols=n,
        specs=[[{"type": "scene"} for _ in range(n)]],
        subplot_titles=[v.get("title", f"View {i+1}") for i, v in enumerate(views[:n])],
    )

    # Add traces to each scene
    for col_idx in range(n):
        for color_hex, tris in tri_by_color.items():
            xs: List[float] = []
            ys: List[float] = []
            zs: List[float] = []
            ii: List[int] = []
            jj: List[int] = []
            kk: List[int] = []
            for tri in tris:
                base = len(xs)
                for vx, vy, vz in tri:
                    xs.append(float(vx))
                    ys.append(float(vy))
                    zs.append(float(vz))
                ii.append(base + 0)
                jj.append(base + 1)
                kk.append(base + 2)
            r01, g01, b01 = _hex_to_rgb01(color_hex)
            fig.add_trace(
                go.Mesh3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    i=ii,
                    j=jj,
                    k=kk,
                    color=f"rgb({int(r01*255)},{int(g01*255)},{int(b01*255)})",
                    opacity=1.0,
                    flatshading=True,
                    lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.9, specular=0.1),
                ),
                row=1,
                col=col_idx + 1,
            )

        if show_outlines:
            tris_all: List[List[List[float]]] = []
            for _c, tris in tri_by_color.items():
                tris_all.extend(tris)
            lx, ly, lz = _edge_lines_from_tris(tris_all, round_ndigits=3)
            if len(lx) > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=lx,
                        y=ly,
                        z=lz,
                        mode="lines",
                        line=dict(color=outline_color, width=outline_width),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=1,
                    col=col_idx + 1,
                )

        v = views[col_idx] if col_idx < len(views) else {"elev": 30, "azim": 45, "title": f"View {col_idx+1}"}
        eye = _camera_eye_from_elev_azim(float(v.get("elev", 30)), float(v.get("azim", 45)), scale=1.8)
        scene_name = "scene" if col_idx == 0 else f"scene{col_idx+1}"
        fig.update_layout(
            **{
                scene_name: dict(
                    xaxis=dict(range=[minx_b, maxx_b], backgroundcolor="white", gridcolor="lightgrey"),
                    yaxis=dict(range=[miny_b, maxy_b], backgroundcolor="white", gridcolor="lightgrey"),
                    zaxis=dict(range=[0, z_max], backgroundcolor="white", gridcolor="lightgrey"),
                    aspectmode="data",
                    camera=dict(
                        eye=eye,
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        projection=dict(type="perspective"),
                    ),
                )
            }
        )

    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10))

    ok_png = False
    ok_html = False

    if html_output_path:
        try:
            Path(html_output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(html_output_path, include_plotlyjs="cdn", full_html=True)
            logger.info("Saved Plotly 3D pyramid HTML to %s", html_output_path)
            ok_html = True
        except Exception as e:  # pragma: no cover
            logger.warning("Plotly write_html failed (pyramid): %s", e)

    wireframe_html_path = config.get("wireframe_html_path")
    if wireframe_html_path and tri_by_color:
        try:
            _write_wireframe_html(tri_by_color, minx_b, maxx_b, miny_b, maxy_b, z_max, wireframe_html_path)
        except Exception as e:  # pragma: no cover
            logger.warning("Wireframe HTML failed (pyramid): %s", e)

    schematic_output_path = config.get("schematic_output_path")
    if schematic_output_path and tri_by_color and gutter_endpoints_pyr:
        try:
            _write_schematic_3d(
                tri_by_color,
                gutter_endpoints_pyr,
                minx_b, maxx_b, miny_b, maxy_b, z_max,
                output_path=str(schematic_output_path),
                html_path=str(config.get("schematic_html_path", "")) or None,
                config=config,
                gutter_centerlines=gutter_centerlines_pyr if gutter_centerlines_pyr else None,
                downspout_centerlines=downspout_centerlines_data_pyr,
                roof_base_tris=roof_base_tris_pyr if roof_base_tris_pyr else None,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("Schematic 3D failed (pyramid): %s", e)

    schematic_with_walls_path = config.get("schematic_with_walls_output_path")
    if schematic_with_walls_path and tri_by_color and gutter_endpoints_pyr and floors_payload:
        try:
            _write_schematic_3d(
                tri_by_color,
                gutter_endpoints_pyr,
                minx_b, maxx_b, miny_b, maxy_b, z_max,
                output_path=str(schematic_with_walls_path),
                html_path=str(config.get("schematic_with_walls_html_path", "")) or None,
                config=config,
                gutter_centerlines=gutter_centerlines_pyr if gutter_centerlines_pyr else None,
                downspout_centerlines=downspout_centerlines_data_pyr,
                roof_base_tris=roof_base_tris_pyr if roof_base_tris_pyr else None,
                floors_payload=floors_payload,
                wall_height=wall_height,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("Schematic 3D with walls failed (pyramid): %s", e)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(output_path, width=1200 * n, height=900, scale=2)
            logger.info("Saved Plotly 3D pyramid to %s", output_path)
            ok_png = True
        except Exception as e:  # pragma: no cover
            logger.warning("Plotly write_image failed (pyramid): %s", e)

    # Return True only if PNG was written (so caller can fallback to Matplotlib for PNG)
    return ok_png if output_path else ok_html


def visualize_3d_house_render_plotly(
    output_path: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    all_floor_paths: Optional[List[str]] = None,
    floor_roof_results: Optional[List[Dict[str, Any]]] = None,
    html_output_path: Optional[str] = None,
    roof_levels: Optional[List[Tuple[float, Dict[str, Any], float, float, int]]] = None,
    final_mode: bool = False,
    extend_segments_mode: bool = False,
) -> bool:
    """
    Randare 3D house_render: pereți + linii 3D (magenta, ridge, verde contur) + fețe acoperiș.
    final_mode: acoperiș caramiziu, drip, burlane.
    extend_segments_mode: segmentele acoperișului sunt prelungite la capete în direcția lor (altă culoare).
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return False

    import math
    import cv2
    from shapely.geometry import Point as ShapelyPoint
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union
    from shapely import affinity as shapely_affinity

    from roof_calc.flood_fill import flood_fill_interior, get_house_shape_mask
    from roof_calc.geometry import extract_polygon
    from roof_calc.overhang import (
        apply_overhang_to_sections,
        clip_roof_faces_to_polygon,
        compute_overhang_px_from_roof_results,
        compute_overhang_sides_from_footprint,
        compute_overhang_sides_from_union_boundary,
        extend_sections_to_connect,
        extend_secondary_sections_to_main_ridge,
        get_drip_edge_faces_3d,
        get_downspout_faces_for_floors,
        get_faces_3d_aframe_with_magenta,
        get_gutter_centerlines_3d,
        get_gutter_end_closures_3d,
        get_gutter_endpoints_3d,
        get_gutter_faces_3d,
        ridge_intersection_corner_lines,
        _is_interior_corner,
    )
    from roof_calc.roof_segments_3d import get_faces_3d_from_segments, get_roof_segments_3d

    if not all_floor_paths or not floor_roof_results or len(all_floor_paths) != len(floor_roof_results):
        return False

    def _polygon_from_path(path: str) -> Optional[Any]:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        filled0 = flood_fill_interior(im)
        house0 = get_house_shape_mask(filled0)
        poly0 = extract_polygon(house0)
        if poly0 is None or poly0.is_empty:
            return None
        return poly0

    def _section_rect_center(sec: Dict[str, Any]) -> Tuple[float, float]:
        br = sec.get("bounding_rect") or []
        if len(br) < 3:
            return (0.0, 0.0)
        xs = [float(p[0]) for p in br]
        ys = [float(p[1]) for p in br]
        return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)

    def _section_area(sec: Dict[str, Any]) -> float:
        br = sec.get("bounding_rect") or []
        if len(br) < 3:
            return 0.0
        xs = [float(p[0]) for p in br]
        ys = [float(p[1]) for p in br]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    def _compute_offsets(paths: List[str], results: List[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
        polys = [_polygon_from_path(p) for p in paths]
        best_idx = 0
        best_area = -1.0
        for i, poly in enumerate(polys):
            if poly is not None and not getattr(poly, "is_empty", True):
                a = float(getattr(poly, "area", 0) or 0)
                if a > best_area:
                    best_area = a
                    best_idx = i
        cx_ref, cy_ref = 0.0, 0.0
        ref_path = paths[best_idx] if best_idx < len(paths) else paths[0]
        if len(paths) >= 2 and len(results) >= 2:
            rr_ref = results[best_idx]
            secs_ref = rr_ref.get("sections") or []
            for other_idx, (path_oth, rr_oth) in enumerate(zip(paths, results)):
                if other_idx == best_idx:
                    continue
                secs_oth = rr_oth.get("sections") or []
                for s_ref in secs_ref:
                    a_ref = _section_area(s_ref)
                    if a_ref <= 0:
                        continue
                    for s_oth in secs_oth:
                        a_oth = _section_area(s_oth)
                        if a_oth <= 0:
                            continue
                        if min(a_ref, a_oth) / max(a_ref, a_oth) >= 0.9:
                            cx_ref, cy_ref = _section_rect_center(s_ref)
                            break
                    if cx_ref != 0 or cy_ref != 0:
                        break
                if cx_ref != 0 or cy_ref != 0:
                    break
        if cx_ref == 0 and cy_ref == 0:
            ref_poly = polys[best_idx] if best_idx < len(polys) else None
            if ref_poly and not getattr(ref_poly, "is_empty", True):
                c = ref_poly.centroid
                cx_ref, cy_ref = float(c.x), float(c.y)
        out: Dict[str, Tuple[int, int]] = {}
        for p, poly in zip(paths, polys):
            if p == ref_path:
                out[p] = (0, 0)
                continue
            found = False
            if p in paths:
                pi = paths.index(p)
                if pi < len(results):
                    rr_p = results[pi]
                    rr_ref = results[best_idx] if best_idx < len(results) else None
                    if rr_ref:
                        for s_p in rr_p.get("sections") or []:
                            a_p = _section_area(s_p)
                            if a_p <= 0:
                                continue
                            for s_ref in rr_ref.get("sections") or []:
                                a_ref = _section_area(s_ref)
                                if a_ref <= 0:
                                    continue
                                if min(a_p, a_ref) / max(a_p, a_ref) >= 0.9:
                                    cxi, cyi = _section_rect_center(s_p)
                                    out[p] = (int(round(cx_ref - cxi)), int(round(cy_ref - cyi)))
                                    found = True
                                    break
                            if found:
                                break
            if not found:
                if poly is None or getattr(poly, "is_empty", True):
                    out[p] = (int(round(-cx_ref)), int(round(-cy_ref)))
                else:
                    c = poly.centroid
                    out[p] = (int(round(cx_ref - c.x)), int(round(cy_ref - c.y)))
        return out

    def _translate_sections(secs: List[Dict[str, Any]], dx: float, dy: float) -> List[Dict[str, Any]]:
        out = []
        for sec in secs:
            rect = sec.get("bounding_rect", [])
            ridge = sec.get("ridge_line", [])
            out.append({
                **sec,
                "bounding_rect": [(float(p[0]) + dx, float(p[1]) + dy) for p in rect] if rect else [],
                "ridge_line": [(float(p[0]) + dx, float(p[1]) + dy) for p in ridge] if ridge and len(ridge) >= 2 else [],
            })
        return out

    def _filter_connections_by_sections(conns: List[Dict[str, Any]], keep_ids: set[int]) -> List[Dict[str, Any]]:
        out = []
        for c in conns or []:
            ids = c.get("section_ids") or c.get("section_id") or []
            try:
                lst = ids if isinstance(ids, (list, tuple)) else [ids]
                if all(int(x) in keep_ids for x in lst):
                    out.append(c)
            except Exception:
                pass
        return out

    def _covered_by_upper(sec_poly: Any, union_above: Any, area_thresh: float = 500.0) -> bool:
        if union_above is None or getattr(union_above, "is_empty", True):
            return False
        try:
            rem = sec_poly.difference(union_above)
            return float(getattr(rem, "area", 0.0) or 0.0) < area_thresh
        except Exception:
            return False

    config = config or {}
    roof_angle_deg = float(config.get("roof_angle", 30.0))
    wall_height = float(config.get("wall_height", 300.0))
    views = config.get("views", [{"elev": 30, "azim": 45, "title": "Sud-Est"}, {"elev": 20, "azim": 225, "title": "Nord-Vest"}])
    overhang_px = float(config.get("overhang_px", 0.0))
    if (final_mode or extend_segments_mode) and overhang_px <= 0:
        overhang_px = compute_overhang_px_from_roof_results(floor_roof_results or [], ratio=0.10)

    # Pereții din imaginile pereților (all_floor_paths) – acoperișurile se pun peste ei
    offsets = _compute_offsets(all_floor_paths, floor_roof_results)
    floors_payload: List[Tuple[str, Any, Dict[str, Any], Tuple[int, int], int]] = []
    for idx, (p, rr) in enumerate(zip(all_floor_paths, floor_roof_results)):
        poly0 = _polygon_from_path(p)
        if poly0 is None:
            continue
        ox, oy = offsets.get(p, (0, 0))
        poly_t = shapely_affinity.translate(poly0, xoff=ox, yoff=oy)
        floors_payload.append((p, poly_t, rr, (ox, oy), idx))
    if not floors_payload:
        return False

    # Sortare: aria descendentă, apoi index original ascendent – la arii egale, ultimul etaj (index mare) rămâne ultimul
    floors_payload.sort(key=lambda t: (-(float(getattr(t[1], "area", 0.0) or 0.0)), t[4]))
    num_floors = len(floors_payload)
    z_max = num_floors * wall_height + 500
    tan_angle = math.tan(math.radians(roof_angle_deg))

    # Pentru final_mode: date top floor pentru drip/gutter
    top_floor_for_final: Optional[Any] = None

    # Încarcă fețe din a_faces – pentru fiecare etaj, exact ce e în a_faces.png (a_faces_faces.json)
    a_faces_by_etaj: Dict[int, List[Dict[str, Any]]] = {}
    a_lines_segments_by_etaj: Dict[int, Dict[str, Any]] = {}
    try:
        import json
        import random
        out_dir = Path(output_path).parent
        for etaj_dir in sorted(out_dir.glob("etaj_*")):
            if not etaj_dir.is_dir():
                continue
            try:
                etaj_idx = int(etaj_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            faces_path = etaj_dir / "a_faces_faces.json"
            if faces_path.exists():
                data = json.loads(faces_path.read_text(encoding="utf-8"))
                faces = data.get("faces", [])
                if faces:
                    a_faces_by_etaj[etaj_idx] = faces
            seg_path = etaj_dir / "a_lines_segments.json"
            if seg_path.exists():
                try:
                    seg_data = json.loads(seg_path.read_text(encoding="utf-8"))
                    if seg_data:
                        a_lines_segments_by_etaj[etaj_idx] = seg_data
                except Exception:
                    pass
    except Exception:
        pass

    def _ridge_z(sec: Dict[str, Any], base_z: float) -> float:
        br = sec.get("bounding_rect", [])
        if len(br) < 3:
            return base_z
        xs = [float(p[0]) for p in br]
        ys = [float(p[1]) for p in br]
        orient = str(sec.get("ridge_orientation", "horizontal"))
        span = (max(ys) - min(ys)) / 2.0 if orient == "horizontal" else (max(xs) - min(xs)) / 2.0
        return base_z + span * tan_angle

    lx_magenta, ly_magenta, lz_magenta = [], [], []
    lx_ridge, ly_ridge, lz_ridge = [], [], []
    lx_aframe, ly_aframe, lz_aframe = [], [], []
    lx_green, ly_green, lz_green = [], [], []
    lx_ext, ly_ext, lz_ext = [], [], []
    ext_outer_points: List[List[float]] = []
    xs_all: List[float] = []
    ys_all: List[float] = []
    roof_faces_with_colors: List[Tuple[List[List[float]], str]] = []

    for floor_idx, (_p, floor_poly, rr, (ox, oy), _orig_idx) in enumerate(floors_payload):
        z0 = floor_idx * wall_height
        z1 = (floor_idx + 1) * wall_height
        coords = list(floor_poly.exterior.coords) if hasattr(floor_poly, "exterior") else []
        for c in coords:
            xs_all.append(float(c[0]))
            ys_all.append(float(c[1]))

        draw_roof = True
        union_above = None
        if floor_idx + 1 < len(floors_payload):
            union_above = unary_union([fp[1] for fp in floors_payload[floor_idx + 1:]])
        secs = rr.get("sections") or []
        conns = rr.get("connections") or []
        kept: List[Dict[str, Any]] = []
        keep_ids: set[int] = set()
        for sec in secs:
            br = sec.get("bounding_rect", [])
            if len(br) < 3:
                continue
            sp = ShapelyPolygon(br)
            sp = shapely_affinity.translate(sp, xoff=ox, yoff=oy)
            if _covered_by_upper(sp, union_above, area_thresh=500.0):
                continue
            sec_t = _translate_sections([sec], float(ox), float(oy))[0]
            kept.append(sec_t)
            try:
                keep_ids.add(int(sec_t.get("section_id")))
            except Exception:
                pass
        conns_kept = _filter_connections_by_sections(conns, keep_ids)
        kept_extended = extend_sections_to_connect(kept, conns_kept) if (draw_roof and kept and conns_kept) else kept
        if draw_roof and kept_extended:
            kept_extended = extend_secondary_sections_to_main_ridge(kept_extended)

        if draw_roof and kept_extended:
            z_ridge_common = max(_ridge_z(sec, z1) for sec in kept_extended)
            # Linii ridge/magenta/verde doar pentru etajul cu acoperiș (ultimul în ordinea ariei)
            draw_lines_this_floor = (floor_idx == num_floors - 1)
            a_lines_seg = a_lines_segments_by_etaj.get(_orig_idx) if (_orig_idx is not None and draw_lines_this_floor) else None
            if a_lines_seg:
                for seg in a_lines_seg.get("ridge") or []:
                    if len(seg) >= 2:
                        p1 = [float(seg[0][0]) + ox, float(seg[0][1]) + oy, z_ridge_common]
                        p2 = [float(seg[1][0]) + ox, float(seg[1][1]) + oy, z_ridge_common]
                        lx, ly, lz = _segment_to_plotly_line(p1, p2)
                        lx_ridge.extend(lx)
                        ly_ridge.extend(ly)
                        lz_ridge.extend(lz)
                for seg in a_lines_seg.get("magenta") or []:
                    if len(seg) >= 2:
                        p1 = [float(seg[0][0]) + ox, float(seg[0][1]) + oy, z_ridge_common]
                        p2 = [float(seg[1][0]) + ox, float(seg[1][1]) + oy, z1]
                        lx, ly, lz = _segment_to_plotly_line(p1, p2)
                        lx_magenta.extend(lx)
                        ly_magenta.extend(ly)
                        lz_magenta.extend(lz)
                for seg in a_lines_seg.get("contour") or []:
                    if len(seg) >= 2:
                        p1 = [float(seg[0][0]) + ox, float(seg[0][1]) + oy, z1]
                        p2 = [float(seg[1][0]) + ox, float(seg[1][1]) + oy, z1]
                        lx, ly, lz = _segment_to_plotly_line(p1, p2)
                        lx_green.extend(lx)
                        ly_green.extend(ly)
                        lz_green.extend(lz)
            else:
                corner_lines = ridge_intersection_corner_lines(kept_extended, floor_polygon=floor_poly)
                coords_ex = list(floor_poly.exterior.coords) if hasattr(floor_poly, "exterior") else []
                corner_tol = 2.0
                boundary_tol = 2.5

                def _snap_to_boundary(cx: float, cy: float) -> Tuple[float, float]:
                    """Întâi: dacă e colț la 1–2 px, mergi la colț. Altfel: cel mai apropiat punct pe contur."""
                    if len(coords_ex) < 2:
                        return (cx, cy)
                    for i in range(len(coords_ex) - 1):
                        vx, vy = float(coords_ex[i][0]), float(coords_ex[i][1])
                        d2 = (cx - vx) ** 2 + (cy - vy) ** 2
                        if d2 <= corner_tol * corner_tol:
                            return (vx, vy)
                    best_pt = (cx, cy)
                    best_d2 = float("inf")
                    for i in range(len(coords_ex) - 1):
                        x1, y1 = float(coords_ex[i][0]), float(coords_ex[i][1])
                        x2, y2 = float(coords_ex[i + 1][0]), float(coords_ex[i + 1][1])
                        dx, dy = x2 - x1, y2 - y1
                        L2 = dx * dx + dy * dy
                        if L2 < 1e-12:
                            t = 0.0
                            nx, ny = x1, y1
                        else:
                            t = max(0, min(1, ((cx - x1) * dx + (cy - y1) * dy) / L2))
                            nx, ny = x1 + t * dx, y1 + t * dy
                        d2 = (cx - nx) ** 2 + (cy - ny) ** 2
                        if d2 < best_d2:
                            best_d2 = d2
                            best_pt = (nx, ny)
                    if best_d2 <= boundary_tol * boundary_tol:
                        return best_pt
                    return (cx, cy)

                for item in corner_lines:
                    pt = item[0]
                    corners = item[1]
                    ix, iy = float(pt[0]), float(pt[1])
                    for cx, cy in corners:
                        sx, sy = _snap_to_boundary(float(cx), float(cy))
                        lx, ly, lz = _segment_to_plotly_line([ix, iy, z_ridge_common], [sx, sy, z1])
                        lx_magenta.extend(lx)
                        ly_magenta.extend(ly)
                        lz_magenta.extend(lz)
                drawn_from: set = set()

                def _key(px: float, py: float) -> Tuple[int, int]:
                    return (int(round(px / 5)), int(round(py / 5)))

                def _already_drawn_from(px: float, py: float) -> bool:
                    return _key(px, py) in drawn_from

                def _mark_drawn(px: float, py: float) -> None:
                    drawn_from.add(_key(px, py))

                # Colțuri interioare = doar vârfuri concave ale poligonului streașină
                interior_corner_pts: List[Tuple[float, float]] = []
                for i in range(len(coords_ex)):
                    curr = (float(coords_ex[i][0]), float(coords_ex[i][1]))
                    prev = (float(coords_ex[i - 1][0]), float(coords_ex[i - 1][1])) if i > 0 else (float(coords_ex[-2][0]), float(coords_ex[-2][1]))
                    next_p = (float(coords_ex[i + 1][0]), float(coords_ex[i + 1][1])) if i + 1 < len(coords_ex) else (float(coords_ex[1][0]), float(coords_ex[1][1]))
                    if _is_interior_corner(prev, curr, next_p, floor_poly):
                        interior_corner_pts.append(curr)
                interior_keys = {(round(ix, 1), round(iy, 1)) for ix, iy in interior_corner_pts}

                def _is_interior(cx: float, cy: float) -> bool:
                    """Colț interior (concav) – excludem linii portocalii către el."""
                    return (round(cx, 1), round(cy, 1)) in interior_keys

                corners_by_rect: Dict[Tuple[float, float, float, float], List[Tuple[float, float]]] = {}
                for item in corner_lines:
                    corners = item[1]
                    rect_key = item[2] if len(item) >= 3 else None
                    if rect_key is not None:
                        rk = (round(rect_key[0], 2), round(rect_key[1], 2), round(rect_key[2], 2), round(rect_key[3], 2))
                        if rk not in corners_by_rect:
                            corners_by_rect[rk] = []
                        existing = {(round(x, 4), round(y, 4)) for x, y in corners_by_rect[rk]}
                        for c in corners:
                            k = (round(float(c[0]), 4), round(float(c[1]), 4))
                            if k not in existing:
                                existing.add(k)
                                corners_by_rect[rk].append((float(c[0]), float(c[1])))

                for sec in kept_extended:
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

                    def _drop_interior_and_symmetric_opposite(corners: list) -> list:
                        drop = {i for i, c in enumerate(corners) if _is_interior(*_snap_to_boundary(float(c[0]), float(c[1])))}
                        n = len(corners)
                        for i in list(drop):
                            drop.add(n - 1 - i)
                        return [c for i, c in enumerate(corners) if i not in drop]

                    if orient == "horizontal":
                        if float(r0[0]) <= float(r1[0]):
                            left_end, right_end = p0, p1
                        else:
                            left_end, right_end = p1, p0
                        if mag_corners:
                            left_raw = [c for c in mag_corners if float(c[0]) < (minx + maxx) / 2]
                            right_raw = [c for c in mag_corners if float(c[0]) >= (minx + maxx) / 2]
                        else:
                            left_raw, right_raw = [tl, bl], [tr, br_pt]
                        left_corners = _drop_interior_and_symmetric_opposite(left_raw)
                        right_corners = _drop_interior_and_symmetric_opposite(right_raw)
                        if not _already_drawn_from(float(left_end[0]), float(left_end[1])) and left_corners and right_corners:
                            for c in left_corners:
                                sx, sy = _snap_to_boundary(float(c[0]), float(c[1]))
                                lx, ly, lz = _segment_to_plotly_line(left_end, [sx, sy, z1])
                                lx_aframe.extend(lx)
                                ly_aframe.extend(ly)
                                lz_aframe.extend(lz)
                            _mark_drawn(float(left_end[0]), float(left_end[1]))
                        if not _already_drawn_from(float(right_end[0]), float(right_end[1])) and left_corners and right_corners:
                            for c in right_corners:
                                sx, sy = _snap_to_boundary(float(c[0]), float(c[1]))
                                lx, ly, lz = _segment_to_plotly_line(right_end, [sx, sy, z1])
                                lx_aframe.extend(lx)
                                ly_aframe.extend(ly)
                                lz_aframe.extend(lz)
                            _mark_drawn(float(right_end[0]), float(right_end[1]))
                    else:
                        if float(r0[1]) <= float(r1[1]):
                            top_end, bottom_end = p0, p1
                        else:
                            top_end, bottom_end = p1, p0
                        if mag_corners:
                            top_raw = [c for c in mag_corners if float(c[1]) < (miny + maxy) / 2]
                            bottom_raw = [c for c in mag_corners if float(c[1]) >= (miny + maxy) / 2]
                        else:
                            top_raw, bottom_raw = [tl, tr], [bl, br_pt]
                        top_corners = _drop_interior_and_symmetric_opposite(top_raw)
                        bottom_corners = _drop_interior_and_symmetric_opposite(bottom_raw)
                        if not _already_drawn_from(float(top_end[0]), float(top_end[1])) and top_corners and bottom_corners:
                            for c in top_corners:
                                sx, sy = _snap_to_boundary(float(c[0]), float(c[1]))
                                lx, ly, lz = _segment_to_plotly_line(top_end, [sx, sy, z1])
                                lx_aframe.extend(lx)
                                ly_aframe.extend(ly)
                                lz_aframe.extend(lz)
                            _mark_drawn(float(top_end[0]), float(top_end[1]))
                        if not _already_drawn_from(float(bottom_end[0]), float(bottom_end[1])) and top_corners and bottom_corners:
                            for c in bottom_corners:
                                sx, sy = _snap_to_boundary(float(c[0]), float(c[1]))
                                lx, ly, lz = _segment_to_plotly_line(bottom_end, [sx, sy, z1])
                                lx_aframe.extend(lx)
                                ly_aframe.extend(ly)
                                lz_aframe.extend(lz)
                            _mark_drawn(float(bottom_end[0]), float(bottom_end[1]))
                for sec in kept_extended:
                    ridge = sec.get("ridge_line", [])
                    if len(ridge) >= 2:
                        r0, r1 = ridge[0], ridge[1]
                        lx, ly, lz = _segment_to_plotly_line(
                            [float(r0[0]), float(r0[1]), z_ridge_common],
                            [float(r1[0]), float(r1[1]), z_ridge_common],
                        )
                        lx_ridge.extend(lx)
                        ly_ridge.extend(ly)
                        lz_ridge.extend(lz)
                coords_ex = list(floor_poly.exterior.coords) if hasattr(floor_poly, "exterior") else []
                for i in range(len(coords_ex) - 1):
                    p1, p2 = coords_ex[i], coords_ex[i + 1]
                    x1, y1 = float(p1[0]), float(p1[1])
                    x2, y2 = float(p2[0]), float(p2[1])
                    lx, ly, lz = _segment_to_plotly_line([x1, y1, z1], [x2, y2, z1])
                    lx_green.extend(lx)
                    ly_green.extend(ly)
                    lz_green.extend(lz)

            # extend_segments_mode: prelungiri la capete în direcția segmentului (altă culoare)
            if extend_segments_mode and overhang_px > 0:
                segs = get_roof_segments_3d(
                    kept_extended, floor_poly, wall_height=float(z1), roof_angle_deg=roof_angle_deg,
                )
                floor_geom = floor_poly if (floor_poly is not None and not getattr(floor_poly, "is_empty", True)) else None
                for _p1, _p2 in segs:
                    px1, py1, pz1 = float(_p1[0]), float(_p1[1]), float(_p1[2])
                    px2, py2, pz2 = float(_p2[0]), float(_p2[1]), float(_p2[2])
                    dx, dy, dz = px2 - px1, py2 - py1, pz2 - pz1
                    L = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if L < 1e-9:
                        continue
                    dx, dy, dz = dx / L, dy / L, dz / L
                    ext = float(overhang_px)
                    ext_p1 = [px1 - dx * ext, py1 - dy * ext, pz1 - dz * ext]
                    ext_p2 = [px2 + dx * ext, py2 + dy * ext, pz2 + dz * ext]

                    def _keep_ext(a: List[float]) -> bool:
                        ax, ay, az = float(a[0]), float(a[1]), float(a[2])
                        if az > z_ridge_common + 1e-6:
                            return False
                        if floor_geom is not None:
                            try:
                                pt = ShapelyPoint(ax, ay)
                                if floor_geom.contains(pt):
                                    return False
                            except Exception:
                                pass
                        return True

                    for a1, a2, outer in [
                        ([ext_p1[0], ext_p1[1], ext_p1[2]], [px1, py1, pz1], ext_p1),
                        ([px2, py2, pz2], [ext_p2[0], ext_p2[1], ext_p2[2]], ext_p2),
                    ]:
                        if not _keep_ext(outer):
                            continue
                        lx0, ly0, lz0 = _segment_to_plotly_line(
                            [float(a1[0]), float(a1[1]), float(a1[2])],
                            [float(a2[0]), float(a2[1]), float(a2[2])],
                        )
                        lx_ext.extend(lx0)
                        ly_ext.extend(ly0)
                        lz_ext.extend(lz0)
                        ext_outer_points.append([float(outer[0]), float(outer[1]), float(outer[2])])

            # Fețe strict din a_faces.png (a_faces_faces.json) – niciodată fallback
            raw_faces: List[List[List[float]]] = []
            if draw_roof:
                raw_faces_from_etaj = a_faces_by_etaj.get(_orig_idx) if _orig_idx is not None else None
                if raw_faces_from_etaj:
                    for f in raw_faces_from_etaj:
                        vs = f.get("vertices_3d") or []
                        if len(vs) >= 3:
                            raw_faces.append([[float(v[0]) + ox, float(v[1]) + oy, float(v[2]) + z0] for v in vs])
            if raw_faces:
                if final_mode and draw_roof and overhang_px > 0 and kept_extended:
                    free = (
                        compute_overhang_sides_from_footprint(kept_extended, floor_poly)
                        if floor_poly is not None and not getattr(floor_poly, "is_empty", True)
                        else compute_overhang_sides_from_union_boundary(kept_extended)
                    )
                    kept_oh = apply_overhang_to_sections(kept_extended, overhang_px=overhang_px, free_sides=free)
                    corner_lines_oh = ridge_intersection_corner_lines(kept_oh, floor_polygon=floor_poly)
                    roof_faces_oh = get_faces_3d_aframe_with_magenta(
                        kept_oh, conns_kept, roof_angle_deg=roof_angle_deg, wall_height=float(z1),
                        corner_lines=corner_lines_oh, floor_polygon=floor_poly,
                    )
                    roof_faces_oh_dicts = [{"vertices_3d": f.get("vertices_3d", [])} for f in roof_faces_oh if f.get("vertices_3d")]
                    top_floor_for_final = (
                        kept_oh, float(z1), overhang_px, roof_angle_deg,
                        roof_faces_oh_dicts, free,
                    )
                display_faces = raw_faces
                rng = random.Random(42)
                faces_sorted = sorted(
                    [(sum(float(v[1]) for v in vs) / len(vs), vs) for vs in display_faces if len(vs) >= 3],
                    key=lambda t: t[0], reverse=True,
                )
                brick_rgba = "rgba(181,82,51,0.75)" if final_mode else None
                for _cy, vs in faces_sorted:
                    if final_mode:
                        rgba = brick_rgba
                    else:
                        r, g, b = rng.random(), rng.random(), rng.random()
                        rgba = f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.75)"
                    roof_faces_with_colors.append((vs, rgba))

    # extend_segments_mode: contur care înconjoară capetele segmentelor albastre
    lx_contour, ly_contour, lz_contour = [], [], []
    if extend_segments_mode and len(ext_outer_points) >= 3:
        seen: Set[Tuple[float, float, float]] = set()
        unique: List[List[float]] = []
        for p in ext_outer_points:
            k = (round(p[0], 2), round(p[1], 2), round(p[2], 2))
            if k not in seen:
                seen.add(k)
                unique.append(p)
        if len(unique) >= 3:
            cx = sum(q[0] for q in unique) / len(unique)
            cy = sum(q[1] for q in unique) / len(unique)
            ordered = sorted(unique, key=lambda q: math.atan2(float(q[1]) - cy, float(q[0]) - cx))
            ordered.append(ordered[0])
            for i in range(len(ordered) - 1):
                a, b = ordered[i], ordered[i + 1]
                lx0, ly0, lz0 = _segment_to_plotly_line(
                    [float(a[0]), float(a[1]), float(a[2])],
                    [float(b[0]), float(b[1]), float(b[2])],
                )
                lx_contour.extend(lx0)
                ly_contour.extend(ly0)
                lz_contour.extend(lz0)

    if not xs_all and not ys_all:
        xs_all = [0.0]
        ys_all = [0.0]
    minx_b = float(min(xs_all))
    maxx_b = float(max(xs_all))
    miny_b = float(min(ys_all))
    maxy_b = float(max(ys_all))
    pad = max(60.0, 0.08 * max(maxx_b - minx_b, maxy_b - miny_b, 1.0))
    minx_b, maxx_b = minx_b - pad, maxx_b + pad
    miny_b, maxy_b = miny_b - pad, maxy_b + pad

    n = max(1, len(views))
    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{"type": "scene"} for _ in range(n)]],
        subplot_titles=[v.get("title", f"View {i+1}") for i, v in enumerate(views[:n])],
    )

    def add_line_trace(lx: List[float], ly: List[float], lz: List[float], color: str, name: str) -> None:
        if len(lx) == 0:
            return
        for col_idx in range(n):
            fig.add_trace(
                go.Scatter3d(
                    x=lx, y=ly, z=lz,
                    mode="lines",
                    line=dict(color=color, width=3),
                    name=name,
                    legendgroup=name,
                ),
                row=1, col=col_idx + 1,
            )

    # Fețe transparente din a_faces (desenate în spatele liniilor)
    def _face_to_mesh_triangles(vs: List[List[float]]) -> Tuple[List[float], List[float], List[float], List[int], List[int], List[int]]:
        """Triangulare fan: (0,1,2), (0,2,3), ..."""
        xs = [float(v[0]) for v in vs]
        ys = [float(v[1]) for v in vs]
        zs = [float(v[2]) for v in vs]
        ni = len(vs)
        ii, jj, kk = [], [], []
        for i in range(1, ni - 1):
            ii.append(0)
            jj.append(i)
            kk.append(i + 1)
        return xs, ys, zs, ii, jj, kk

    if roof_faces_with_colors:
        first_face_name = "Fețe acoperiș (caramiză)" if final_mode else "Fețe acoperiș (a_faces)"
        for idx, (vs, rgba) in enumerate(roof_faces_with_colors):
            xs, ys, zs, ii, jj, kk = _face_to_mesh_triangles(vs)
            if not ii:
                continue
            for col_idx in range(n):
                fig.add_trace(
                    go.Mesh3d(
                        x=xs, y=ys, z=zs,
                        i=ii, j=jj, k=kk,
                        color=rgba,
                        opacity=0.75,
                        flatshading=True,
                        name=first_face_name if idx == 0 else None,
                        showlegend=(idx == 0),
                        legendgroup=first_face_name,
                    ),
                    row=1, col=col_idx + 1,
                )

    # final_mode: drip edge, burlane (jgheaburi), streșini 3D – același pipeline ca visualize_3d_standard_plotly
    if final_mode and overhang_px > 0 and top_floor_for_final:
        try:
            kept_oh, z1_f, oh_px, r_angle, roof_faces_list, free_drip = top_floor_for_final
            _kw = {"roof_faces": roof_faces_list, "include_eaves_only": True, "eaves_z_lift": oh_px * 0.60}
            drip_faces = get_drip_edge_faces_3d(
                kept_oh, z1_f, oh_px, r_angle, roof_faces=roof_faces_list, free_sides=free_drip,
            )
            for idx_df, df in enumerate(drip_faces):
                v = df.get("vertices_3d")
                if v and len(v) >= 3:
                    xs, ys, zs, ii, jj, kk = _face_to_mesh_triangles(v)
                    if ii:
                        for col_idx in range(n):
                            fig.add_trace(
                                go.Mesh3d(
                                    x=xs, y=ys, z=zs, i=ii, j=jj, k=kk,
                                    color="rgba(139,69,19,0.9)", opacity=0.9,
                                    flatshading=True, name="Drip" if idx_df == 0 else None,
                                    showlegend=(idx_df == 0), legendgroup="Drip",
                                ),
                                row=1, col=col_idx + 1,
                            )
            gutter_faces_list = get_gutter_faces_3d(kept_oh, z1_f, oh_px, r_angle, **_kw)
            for idx_gf, gf in enumerate(gutter_faces_list):
                v = gf.get("vertices_3d")
                if v and len(v) >= 3:
                    xs, ys, zs, ii, jj, kk = _face_to_mesh_triangles(v)
                    if ii:
                        for col_idx in range(n):
                            fig.add_trace(
                                go.Mesh3d(
                                    x=xs, y=ys, z=zs, i=ii, j=jj, k=kk,
                                    color="rgba(107,114,128,0.95)", opacity=0.95,
                                    flatshading=True, name="Burlane" if idx_gf == 0 else None,
                                    showlegend=(idx_gf == 0), legendgroup="Burlane",
                                ),
                                row=1, col=col_idx + 1,
                            )
            ep = list(get_gutter_endpoints_3d(kept_oh, z1_f, oh_px, r_angle, **_kw))
            gutter_radius = max(2.0, oh_px * 0.24) * 0.70 if oh_px > 0 else 2.0
            seg_sections = [kept_oh] * max(1, len(ep) // 2) if ep else []
            downspout_result = get_downspout_faces_for_floors(
                [(fp[0], fp[1], fp[2], fp[3]) for fp in floors_payload], wall_height,
                cylinder_radius=gutter_radius, gutter_endpoints=ep,
                gutter_segment_sections=seg_sections if seg_sections else None,
                return_used_endpoints=True,
            )
            used_ep = downspout_result[1] if isinstance(downspout_result, tuple) and len(downspout_result) > 1 else []
            ds_faces = downspout_result[0] if isinstance(downspout_result, tuple) else downspout_result
            for df in (ds_faces if isinstance(ds_faces, list) else [ds_faces]):
                v = (df.get("vertices_3d") if isinstance(df, dict) else []) or []
                if v and len(v) >= 3:
                    xs, ys, zs, ii, jj, kk = _face_to_mesh_triangles(v)
                    if ii:
                        for col_idx in range(n):
                            fig.add_trace(
                                go.Mesh3d(x=xs, y=ys, z=zs, i=ii, j=jj, k=kk,
                                    color="rgba(107,114,128,0.95)", opacity=0.95, flatshading=True, showlegend=False),
                                row=1, col=col_idx + 1,
                            )
            closure_kw = dict(_kw, downspout_endpoints=used_ep)
            for gf in get_gutter_end_closures_3d(kept_oh, z1_f, oh_px, r_angle, **closure_kw):
                v = gf.get("vertices_3d")
                if v and len(v) >= 3:
                    xs, ys, zs, ii, jj, kk = _face_to_mesh_triangles(v)
                    if ii:
                        for col_idx in range(n):
                            fig.add_trace(
                                go.Mesh3d(x=xs, y=ys, z=zs, i=ii, j=jj, k=kk,
                                    color="rgba(107,114,128,0.95)", opacity=0.95, flatshading=True, showlegend=False),
                                row=1, col=col_idx + 1,
                            )
        except Exception:
            pass

    lx_wall, ly_wall, lz_wall = [], [], []
    for floor_idx, item in enumerate(floors_payload):
        _path, floor_poly, _rr, _offset = item[0], item[1], item[2], item[3]
        z0 = floor_idx * wall_height
        z1 = (floor_idx + 1) * wall_height
        coords = list(floor_poly.exterior.coords) if hasattr(floor_poly, "exterior") else []
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])
            for (a1, b1, za), (a2, b2, zb) in [
                ((x1, y1, z0), (x2, y2, z0)),
                ((x1, y1, z1), (x2, y2, z1)),
                ((x1, y1, z0), (x1, y1, z1)),
                ((x2, y2, z0), (x2, y2, z1)),
            ]:
                lx, ly, lz = _segment_to_plotly_line([a1, b1, za], [a2, b2, zb])
                lx_wall.extend(lx)
                ly_wall.extend(ly)
                lz_wall.extend(lz)
    add_line_trace(lx_wall, ly_wall, lz_wall, "#95A5A6", "Pereți")
    add_line_trace(lx_ext, ly_ext, lz_ext, "#00BFFF", "Prelungiri segmente acoperiș")
    add_line_trace(lx_contour, ly_contour, lz_contour, "#0066CC", "Contur prelungiri")
    add_line_trace(lx_aframe, ly_aframe, lz_aframe, "#FF8C00", "Deschideri A-frame (ridge→streașină)")
    add_line_trace(lx_magenta, ly_magenta, lz_magenta, "#CC00FF", "Linii intersecție → colțuri")
    add_line_trace(lx_ridge, ly_ridge, lz_ridge, "#8B0000", "Ridge")
    add_line_trace(lx_green, ly_green, lz_green, "#2ECC71", "Contur casă (streașină)")

    for col_idx in range(n):
        v = views[col_idx] if col_idx < len(views) else {"elev": 30, "azim": 45}
        eye = _camera_eye_from_elev_azim(float(v.get("elev", 30)), float(v.get("azim", 45)), scale=1.8)
        scene_name = "scene" if col_idx == 0 else f"scene{col_idx+1}"
        fig.update_layout(**{
            scene_name: dict(
                xaxis=dict(range=[minx_b, maxx_b], backgroundcolor="white", gridcolor="lightgrey"),
                yaxis=dict(range=[miny_b, maxy_b], backgroundcolor="white", gridcolor="lightgrey"),
                zaxis=dict(range=[0, z_max], backgroundcolor="white", gridcolor="lightgrey"),
                aspectmode="data",
                camera=dict(eye=eye, up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), projection=dict(type="perspective")),
            )
        })

    fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="top", y=1.02), margin=dict(l=10, r=10, t=60, b=10))

    ok_png = False
    ok_html = False
    if html_output_path:
        try:
            Path(html_output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(html_output_path, include_plotlyjs="cdn", full_html=True)
            ok_html = True
        except Exception:
            pass
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(output_path, width=1200 * n, height=900, scale=2)
            ok_png = True
        except Exception:
            pass
    return ok_png if output_path else ok_html


def visualize_3d_a_frame_plotly(
    output_path: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    all_floor_paths: Optional[List[str]] = None,
    floor_roof_results: Optional[List[Dict[str, Any]]] = None,
    html_output_path: Optional[str] = None,
    roof_levels: Optional[List[Tuple[float, Dict[str, Any], float, float, int]]] = None,
    use_a_lines_structure: bool = True,
) -> bool:
    """
    Randare 3D A-frame: pereți + segmente din get_roof_segments_3d (identice cu house_render).
    Salvează house_a_frame.png și house_a_frame.html.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return False

    import cv2
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union
    from shapely import affinity as shapely_affinity

    from roof_calc.flood_fill import flood_fill_interior, get_house_shape_mask
    from roof_calc.geometry import extract_polygon
    from roof_calc.overhang import (
        extend_sections_to_connect,
        extend_secondary_sections_to_main_ridge,
        ridge_intersection_corner_lines,
    )
    from roof_calc.roof_segments_3d import get_faces_3d_from_segments

    if not all_floor_paths or not floor_roof_results or len(all_floor_paths) != len(floor_roof_results):
        return False

    def _polygon_from_path(path: str) -> Optional[Any]:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        filled0 = flood_fill_interior(im)
        house0 = get_house_shape_mask(filled0)
        poly0 = extract_polygon(house0)
        if poly0 is None or poly0.is_empty:
            return None
        return poly0

    def _compute_offsets(paths: List[str], results: List[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
        polys = [_polygon_from_path(p) for p in paths]
        best_idx = 0
        best_area = -1.0
        for i, poly in enumerate(polys):
            if poly is not None and not getattr(poly, "is_empty", True):
                a = float(getattr(poly, "area", 0) or 0)
                if a > best_area:
                    best_area = a
                    best_idx = i
        ref_poly = polys[best_idx] if best_idx < len(polys) else None
        cx_ref = float(ref_poly.centroid.x) if ref_poly and not getattr(ref_poly, "is_empty", True) else 0.0
        cy_ref = float(ref_poly.centroid.y) if ref_poly and not getattr(ref_poly, "is_empty", True) else 0.0
        out: Dict[str, Tuple[int, int]] = {}
        for p, poly in zip(paths, polys):
            if poly is None or getattr(poly, "is_empty", True):
                out[p] = (int(round(-cx_ref)), int(round(-cy_ref)))
            else:
                c = poly.centroid
                out[p] = (int(round(cx_ref - c.x)), int(round(cy_ref - c.y)))
        return out

    def _translate_sections(secs: List[Dict[str, Any]], dx: float, dy: float) -> List[Dict[str, Any]]:
        out = []
        for sec in secs:
            rect = sec.get("bounding_rect", [])
            ridge = sec.get("ridge_line", [])
            out.append({
                **sec,
                "bounding_rect": [(float(p[0]) + dx, float(p[1]) + dy) for p in rect] if rect else [],
                "ridge_line": [(float(p[0]) + dx, float(p[1]) + dy) for p in ridge] if ridge and len(ridge) >= 2 else [],
            })
        return out

    def _filter_connections_by_sections(conns: List[Dict[str, Any]], keep_ids: set[int]) -> List[Dict[str, Any]]:
        out = []
        for c in conns or []:
            ids = c.get("section_ids") or c.get("section_id") or []
            try:
                lst = ids if isinstance(ids, (list, tuple)) else [ids]
                if all(int(x) in keep_ids for x in lst):
                    out.append(c)
            except Exception:
                pass
        return out

    def _covered_by_upper(sec_poly: Any, union_above: Any, area_thresh: float = 500.0) -> bool:
        if union_above is None or getattr(union_above, "is_empty", True):
            return False
        try:
            rem = sec_poly.difference(union_above)
            return float(getattr(rem, "area", 0.0) or 0.0) < area_thresh
        except Exception:
            return False

    config = config or {}
    roof_angle_deg = float(config.get("roof_angle", 30.0))
    wall_height = float(config.get("wall_height", 300.0))
    views = config.get("views", [{"elev": 30, "azim": 45, "title": "Sud-Est"}, {"elev": 20, "azim": 225, "title": "Nord-Vest"}])

    # Încarcă fețe din a_faces – strict ce e în a_faces.png
    a_faces_by_etaj = {}
    try:
        import json
        out_dir = Path(output_path).parent
        for etaj_dir in sorted(out_dir.glob("etaj_*")):
            if not etaj_dir.is_dir():
                continue
            try:
                etaj_idx = int(etaj_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            faces_path = etaj_dir / "a_faces_faces.json"
            if not faces_path.exists():
                continue
            data = json.loads(faces_path.read_text(encoding="utf-8"))
            faces = data.get("faces", [])
            if faces:
                a_faces_by_etaj[etaj_idx] = faces
    except Exception:
        pass

    # Pereții din imaginile pereților – acoperișurile se pun corect peste ei
    offsets = _compute_offsets(all_floor_paths, floor_roof_results)
    floors_payload = []
    for idx, (p, rr) in enumerate(zip(all_floor_paths, floor_roof_results)):
        poly0 = _polygon_from_path(p)
        if poly0 is None:
            continue
        ox, oy = offsets.get(p, (0, 0))
        poly_t = shapely_affinity.translate(poly0, xoff=ox, yoff=oy)
        floors_payload.append((p, poly_t, rr, (ox, oy), idx))
    if not floors_payload:
        return False
    floors_payload.sort(key=lambda t: (-(float(getattr(t[1], "area", 0.0) or 0.0)), t[4]))
    num_floors = len(floors_payload)
    z_max = num_floors * wall_height + 500

    tri_by_color: Dict[str, List[List[List[float]]]] = {}
    RED = "#FF0000"
    GRAY = "#95A5A6"
    roof_face_edges: List[Tuple[List[float], List[float]]] = []

    def add_face(face: List[List[float]], color_hex: str) -> None:
        tris = _triangulate_face(face)
        if not tris:
            tris = _triangulate_fan(face)
        for tri in tris:
            tri_by_color.setdefault(color_hex, []).append(tri)

    xs_all: List[float] = []
    ys_all: List[float] = []

    for floor_idx, payload_item in enumerate(floors_payload):
        _p, floor_poly, rr, (ox, oy) = payload_item[0], payload_item[1], payload_item[2], payload_item[3]
        _orig_idx = payload_item[4] if len(payload_item) > 4 else floor_idx
        z0 = floor_idx * wall_height
        z1 = (floor_idx + 1) * wall_height
        coords = list(floor_poly.exterior.coords) if hasattr(floor_poly, "exterior") else []
        for c in coords:
            xs_all.append(float(c[0]))
            ys_all.append(float(c[1]))

        for i in range(len(coords) - 1):
            x1, y1 = float(coords[i][0]), float(coords[i][1])
            x2, y2 = float(coords[i + 1][0]), float(coords[i + 1][1])
            add_face([[x1, y1, z0], [x2, y2, z0], [x2, y2, z1], [x1, y1, z1]], GRAY)

        draw_roof = True
        union_above = None
        if floor_idx + 1 < len(floors_payload):
            union_above = unary_union([fp[1] for fp in floors_payload[floor_idx + 1:]])
        secs = rr.get("sections") or []
        conns = rr.get("connections") or []
        kept: List[Dict[str, Any]] = []
        keep_ids: set[int] = set()
        for sec in secs:
            br = sec.get("bounding_rect", [])
            if len(br) < 3:
                continue
            sp = ShapelyPolygon(br)
            sp = shapely_affinity.translate(sp, xoff=ox, yoff=oy)
            if _covered_by_upper(sp, union_above, area_thresh=500.0):
                continue
            sec_t = _translate_sections([sec], float(ox), float(oy))[0]
            kept.append(sec_t)
            try:
                keep_ids.add(int(sec_t.get("section_id")))
            except Exception:
                pass
        conns_kept = _filter_connections_by_sections(conns, keep_ids)
        kept_extended = extend_sections_to_connect(kept, conns_kept) if (draw_roof and kept and conns_kept) else kept
        if draw_roof and kept_extended:
            kept_extended = extend_secondary_sections_to_main_ridge(kept_extended)

        if draw_roof and kept_extended:
            # Fețe strict din a_faces.png – fără fallback
            roof_faces_data: List[Dict[str, Any]] = []
            raw_faces_from_etaj = a_faces_by_etaj.get(_orig_idx) if _orig_idx is not None else None
            if raw_faces_from_etaj:
                for f in raw_faces_from_etaj:
                    vs = f.get("vertices_3d") or []
                    if vs:
                        roof_faces_data.append({
                            "vertices_3d": [[float(v[0]) + ox, float(v[1]) + oy, float(v[2]) + z0] for v in vs],
                        })
            if roof_faces_data:
                seen_edge: set = set()
                tol = 1e-4

                def _edge_key(a: List[float], b: List[float]) -> tuple:
                    ka = (round(a[0] / tol) * tol, round(a[1] / tol) * tol, round(a[2] / tol) * tol)
                    kb = (round(b[0] / tol) * tol, round(b[1] / tol) * tol, round(b[2] / tol) * tol)
                    return (ka, kb) if ka <= kb else (kb, ka)

                for f in roof_faces_data:
                    vs = f.get("vertices_3d") or []
                    if vs:
                        add_face(vs, RED)
                        for i in range(len(vs)):
                            p1, p2 = list(vs[i]), list(vs[(i + 1) % len(vs)])
                            key = _edge_key(p1, p2)
                            if key not in seen_edge:
                                seen_edge.add(key)
                                roof_face_edges.append((p1, p2))

    if not xs_all and not ys_all:
        xs_all = [0.0]
        ys_all = [0.0]
    minx_b = float(min(xs_all))
    maxx_b = float(max(xs_all))
    miny_b = float(min(ys_all))
    maxy_b = float(max(ys_all))
    pad = max(60.0, 0.08 * max(maxx_b - minx_b, maxy_b - miny_b, 1.0))
    minx_b, maxx_b = minx_b - pad, maxx_b + pad
    miny_b, maxy_b = miny_b - pad, maxy_b + pad

    n = max(1, len(views))
    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{"type": "scene"} for _ in range(n)]],
        subplot_titles=[v.get("title", f"View {i+1}") for i, v in enumerate(views[:n])],
    )

    tris_all: List[List[List[float]]] = []
    for color_hex, tris in tri_by_color.items():
        tris_all.extend(tris)

    for col_idx in range(n):
        shadow_tris = [[[vx, vy, 0.0] for vx, vy, vz in tri] for tri in tris_all]
        if shadow_tris:
            xs_s, ys_s, zs_s = [], [], []
            ii_s, jj_s, kk_s = [], [], []
            for tri in shadow_tris:
                base = len(xs_s)
                for vx, vy, vz in tri:
                    xs_s.append(float(vx))
                    ys_s.append(float(vy))
                    zs_s.append(float(vz))
                ii_s.append(base + 0)
                jj_s.append(base + 1)
                kk_s.append(base + 2)
            fig.add_trace(
                go.Mesh3d(
                    x=xs_s, y=ys_s, z=zs_s, i=ii_s, j=jj_s, k=kk_s,
                    color="rgba(0,0,0,0.25)",
                    opacity=0.25,
                    flatshading=True,
                    name="Umbră",
                    legendgroup="Umbră",
                ),
                row=1, col=col_idx + 1,
            )

        for color_hex, tris in tri_by_color.items():
            if not tris:
                continue
            xs: List[float] = []
            ys: List[float] = []
            zs: List[float] = []
            ii: List[int] = []
            jj: List[int] = []
            kk: List[int] = []
            for tri in tris:
                base = len(xs)
                for vx, vy, vz in tri:
                    xs.append(float(vx))
                    ys.append(float(vy))
                    zs.append(float(vz))
                ii.append(base + 0)
                jj.append(base + 1)
                kk.append(base + 2)
            r01, g01, b01 = _hex_to_rgb01(color_hex)
            name = "Acoperiș" if color_hex == RED else "Pereți"
            fig.add_trace(
                go.Mesh3d(
                    x=xs, y=ys, z=zs, i=ii, j=jj, k=kk,
                    color=f"rgb({int(r01*255)},{int(g01*255)},{int(b01*255)})",
                    opacity=1.0,
                    flatshading=True,
                    name=name,
                    legendgroup=name,
                ),
                row=1, col=col_idx + 1,
            )

        lx_outline, ly_outline, lz_outline = [], [], []
        for p1, p2 in roof_face_edges:
            lx, ly, lz = _segment_to_plotly_line(list(p1), list(p2))
            lx_outline.extend(lx)
            ly_outline.extend(ly)
            lz_outline.extend(lz)
        for floor_idx, (_path, floor_poly, _rr, _offset) in enumerate(floors_payload):
            z0 = floor_idx * wall_height
            z1 = (floor_idx + 1) * wall_height
            coords = list(floor_poly.exterior.coords) if hasattr(floor_poly, "exterior") else []
            for i in range(len(coords) - 1):
                x1, y1 = float(coords[i][0]), float(coords[i][1])
                x2, y2 = float(coords[i + 1][0]), float(coords[i + 1][1])
                for (a1, b1, za), (a2, b2, zb) in [
                    ((x1, y1, z0), (x2, y2, z0)),
                    ((x1, y1, z1), (x2, y2, z1)),
                    ((x1, y1, z0), (x1, y1, z1)),
                    ((x2, y2, z0), (x2, y2, z1)),
                ]:
                    lx, ly, lz = _segment_to_plotly_line([a1, b1, za], [a2, b2, zb])
                    lx_outline.extend(lx)
                    ly_outline.extend(ly)
                    lz_outline.extend(lz)
        if lx_outline:
            fig.add_trace(
                go.Scatter3d(
                    x=lx_outline, y=ly_outline, z=lz_outline,
                    mode="lines",
                    line=dict(color="black", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1, col=col_idx + 1,
            )

    for col_idx in range(n):
        v = views[col_idx] if col_idx < len(views) else {"elev": 30, "azim": 45}
        eye = _camera_eye_from_elev_azim(float(v.get("elev", 30)), float(v.get("azim", 45)), scale=1.8)
        scene_name = "scene" if col_idx == 0 else f"scene{col_idx+1}"
        fig.update_layout(**{
            scene_name: dict(
                xaxis=dict(range=[minx_b, maxx_b], backgroundcolor="white", gridcolor="lightgrey"),
                yaxis=dict(range=[miny_b, maxy_b], backgroundcolor="white", gridcolor="lightgrey"),
                zaxis=dict(range=[0, z_max], backgroundcolor="white", gridcolor="lightgrey"),
                aspectmode="data",
                camera=dict(eye=eye, up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), projection=dict(type="perspective")),
            )
        })

    fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="top", y=1.02), margin=dict(l=10, r=10, t=60, b=10))

    ok_png = False
    ok_html = False
    if html_output_path:
        try:
            Path(html_output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(html_output_path, include_plotlyjs="cdn", full_html=True)
            ok_html = True
        except Exception:
            pass
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(output_path, width=1200 * n, height=900, scale=2)
            ok_png = True
        except Exception:
            pass
    return ok_png if output_path else ok_html

