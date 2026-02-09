"""
Vizualizare 3D (piramidă) cu Plotly + Kaleido: export PNG direct (fără browser).
Folosit ca fallback când PyVista/VTK nu poate fi importat (ex. Python 3.14 + VTK wheels).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
            inter = tri_poly.intersection(clip_poly)
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
            fig.write_image(output_path, width=1200 * n, height=900, scale=2)
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
        compute_overhang_sides_from_footprint,
        compute_overhang_sides_from_union_boundary,
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
        """Pixel-perfect: referință și centre întregi (aliniere cu floors_overlay)."""
        bboxes: List[Tuple[int, int, int, int]] = []
        for rr in results:
            bb = _largest_section_bbox(rr) or (0, 0, 0, 0)
            bboxes.append(bb)
        widths = [b[2] - b[0] for b in bboxes]
        heights = [b[3] - b[1] for b in bboxes]
        max_w = max(widths) if widths else 0
        max_h = max(heights) if heights else 0
        padding = 50
        ref_cx = int(round((max_w + 2 * padding) / 2))
        ref_cy = int(round((max_h + 2 * padding) / 2))
        out: Dict[str, Tuple[int, int]] = {}
        for p, bb in zip(paths, bboxes):
            cx = (bb[0] + bb[2]) // 2
            cy = (bb[1] + bb[3]) // 2
            ox = ref_cx - cx
            oy = ref_cy - cy
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
    use_shed_lower = lower_floor_roof_mode == "shed"
    if roof_levels:
        try:
            if use_shed_lower:
                from roof_calc.visualize_3d_pyvista import _roof_section_faces_shed  # type: ignore
                from roof_calc.overhang import high_side_for_shed_from_upper_floor
            for z_base, roof_data, dx, dy, _fl in roof_levels:
                fl = int(_fl) if _fl is not None else 0
                secs0 = roof_data.get("sections") or []
                conns0 = roof_data.get("connections") or []
                secs_t = _translate_sections(secs0, float(dx), float(dy))
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
        draw_roof = floor_idx == (num_floors - 1)
        roof_faces_base = (
            rf.get_faces_3d_standard(kept, conns_kept, roof_angle_deg=roof_angle_deg, wall_height=z1) if draw_roof else []
        )

        kept_use = kept
        if draw_roof and overhang_px > 0 and kept:
            free = compute_overhang_sides_from_union_boundary(kept)
            kept_use = apply_overhang_to_sections(kept, overhang_px=overhang_px, free_sides=free)
        roof_faces = (
            rf.get_faces_3d_standard(kept_use, conns_kept, roof_angle_deg=roof_angle_deg, wall_height=z1) if draw_roof else []
        )

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

        # Pereți: prelungim până la acoperiș (etaj superior sau roof_levels)
        # Formăm forma geometrică: segment perete + contur acoperiș de-a lungul peretelui
        faces_for_wall_z = (
            roof_faces if (draw_roof and roof_faces) else roof_faces_by_floor.get(floor_idx, [])
        )
        for i in range(len(floor_coords) - 1):
            p1, p2 = floor_coords[i], floor_coords[i + 1]
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])
            if faces_for_wall_z:
                # Eșantionăm 5 puncte de-a lungul peretelui pentru a urmări panta acoperișului
                pts_top = []
                for t in (0.0, 0.25, 0.5, 0.75, 1.0):
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    zt = _z_roof_at(faces_for_wall_z, x, y, z1)
                    pts_top.append([x, y, max(z1, zt)])
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
                    faces = rf.get_faces_3d_standard(secs_t, conns0, roof_angle_deg=roof_angle_deg, wall_height=float(z_base))
                    if overhang_px > 0 and secs_t_no_oh:
                        faces_base_af = rf.get_faces_3d_standard(secs_t_no_oh, conns0, roof_angle_deg=roof_angle_deg, wall_height=float(z_base))
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
        """Pixel-perfect: referință și centre întregi (aliniere cu floors_overlay)."""
        bboxes: List[Tuple[int, int, int, int]] = []
        for rr in results:
            bb = _largest_section_bbox(rr) or (0, 0, 0, 0)
            bboxes.append(bb)
        widths = [b[2] - b[0] for b in bboxes]
        heights = [b[3] - b[1] for b in bboxes]
        max_w = max(widths) if widths else 0
        max_h = max(heights) if heights else 0
        padding = 50
        ref_cx = int(round((max_w + 2 * padding) / 2))
        ref_cy = int(round((max_h + 2 * padding) / 2))
        out: Dict[str, Tuple[int, int]] = {}
        for p, bb in zip(paths, bboxes):
            cx = (bb[0] + bb[2]) // 2
            cy = (bb[1] + bb[3]) // 2
            ox = ref_cx - cx
            oy = ref_cy - cy
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
        draw_roof = floor_idx == (num_floors - 1)

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

        # Pereți prelungiți până la acoperiș (piramidă)
        faces_for_wall_z = [{"vertices_3d": f} for faces in [d[1] for d in section_face_data] for f in faces]
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            if draw_roof and faces_for_wall_z:
                z1_top = _z_roof_at(faces_for_wall_z, float(p1[0]), float(p1[1]), z1)
                z2_top = _z_roof_at(faces_for_wall_z, float(p2[0]), float(p2[1]), z1)
                z1_top = max(z1, z1_top)
                z2_top = max(z1, z2_top)
            else:
                z1_top = z2_top = z1
            add_face(
                [[p1[0], p1[1], z0], [p2[0], p2[1], z0], [p2[0], p2[1], z2_top], [p1[0], p1[1], z1_top]],
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

