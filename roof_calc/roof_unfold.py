"""
Unfold roof faces to 2D masks with dimensions matching the plan (pixel scale).

Generates per-face masks for standard (gable), pyramid, and shed roof types.
Output: roof_unfolded/{type}/etaj_{i}/fara_overhang/ și cu_overhang/ cu măști separate.
  - fata_N_*.png: mască crop (dimensiuni unfolded)
  - fata_N_*_overlay.png: mască la dimensiunea planului (plan_h x plan_w), pentru overlay
  - toate_fetele_overlay.png: toate fețele combinate, dimensiune plan
  - lungimi_streasure_burlane.json: lungimile streașinilor și burlanelor în pixeli
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# Culori distincte per față (pentru randare 3D)
_FACE_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
    "#DDA0DD", "#F0E68C", "#87CEEB", "#CD853F", "#9370DB",
    "#20B2AA", "#FF69B4", "#778899", "#32CD32", "#FF4500",
]


def _polygon_perimeter_2d(poly_2d: List[Tuple[float, float]]) -> float:
    """Perimetrul poligonului 2D (suma lungimilor laturilor)."""
    if len(poly_2d) < 2:
        return 0.0
    p = 0.0
    for i in range(len(poly_2d)):
        j = (i + 1) % len(poly_2d)
        dx = poly_2d[j][0] - poly_2d[i][0]
        dy = poly_2d[j][1] - poly_2d[i][1]
        p += math.sqrt(dx * dx + dy * dy)
    return p


def _unfold_face_to_2d(face_3d: List[List[float]]) -> Optional[Tuple[List[Tuple[float, float]], float, float]]:
    """
    Proiectează o față plană 3D în 2D (unfold).
    Returnează (polygon_2d, width, height) unde dimensiunile sunt în pixeli (aceeași scală ca planul).
    """
    if len(face_3d) < 3:
        return None
    pts = [[float(p[0]), float(p[1]), float(p[2])] for p in face_3d]
    # Alegem origine = primul punct, u = direcția primei muchii, v = normală în plan
    p0 = pts[0]
    p1 = pts[1]
    u_vec = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
    u_len = math.sqrt(u_vec[0] ** 2 + u_vec[1] ** 2 + u_vec[2] ** 2)
    if u_len < 1e-9:
        return None
    u_vec = [u_vec[0] / u_len, u_vec[1] / u_len, u_vec[2] / u_len]

    # v = perpendicular pe u, în planul feței (folosim a doua muchie)
    p2 = pts[2] if len(pts) > 2 else pts[0]
    edge2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]]
    # v = edge2 - (edge2·u)u
    dot = edge2[0] * u_vec[0] + edge2[1] * u_vec[1] + edge2[2] * u_vec[2]
    v_vec = [
        edge2[0] - dot * u_vec[0],
        edge2[1] - dot * u_vec[1],
        edge2[2] - dot * u_vec[2],
    ]
    v_len = math.sqrt(v_vec[0] ** 2 + v_vec[1] ** 2 + v_vec[2] ** 2)
    if v_len < 1e-9:
        v_vec = [0.0, 0.0, 0.0]
    else:
        v_vec = [v_vec[0] / v_len, v_vec[1] / v_len, v_vec[2] / v_len]

    poly_2d: List[Tuple[float, float]] = []
    for p in pts:
        r = [p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]]
        u_val = r[0] * u_vec[0] + r[1] * u_vec[1] + r[2] * u_vec[2]
        v_val = r[0] * v_vec[0] + r[1] * v_vec[1] + r[2] * v_vec[2]
        poly_2d.append((u_val, v_val))

    xs = [pt[0] for pt in poly_2d]
    ys = [pt[1] for pt in poly_2d]
    min_u, max_u = min(xs), max(xs)
    min_v, max_v = min(ys), max(ys)
    width = max_u - min_u
    height = max_v - min_v
    if width < 1e-6:
        width = 1e-6
    if height < 1e-6:
        height = 1e-6
    return (poly_2d, width, height)


def _rasterize_polygon_to_mask(
    poly_2d: List[Tuple[float, float]],
    width: float,
    height: float,
) -> Tuple[np.ndarray, int, int]:
    """
    Rasterizează poligonul 2D într-o mască.
    Dimensiunile măștii = ceil(width) x ceil(height) px – 1 px = 1 unitate din plan.
    """
    w_px = max(1, int(math.ceil(width)))
    h_px = max(1, int(math.ceil(height)))
    mask = np.zeros((h_px, w_px), dtype=np.uint8)
    # Translatăm poligonul astfel încât (min_u, min_v) -> (0, 0)
    xs = [p[0] for p in poly_2d]
    ys = [p[1] for p in poly_2d]
    min_u, min_v = min(xs), min(ys)
    coords = np.array(
        [[(p[0] - min_u), (p[1] - min_v)] for p in poly_2d],
        dtype=np.float32,
    )
    coords[:, 0] *= (w_px - 1) / max(width, 1e-6)
    coords[:, 1] *= (h_px - 1) / max(height, 1e-6)
    pts_int = np.array(coords, dtype=np.int32)
    if pts_int.size >= 6:
        cv2.fillPoly(mask, [pts_int], 255)
    return mask, w_px, h_px


def _rasterize_face_footprint_on_plan(
    verts: List[List[float]],
    plan_h: int,
    plan_w: int,
) -> np.ndarray:
    """
    Proiectează fața 3D pe planul z=0 și rasterizează într-o mască de dimensiune plan (plan_h x plan_w).
    Coordonate: (x,y) din verts = (col, row) în imagine.
    """
    if len(verts) < 3:
        return np.zeros((plan_h, plan_w), dtype=np.uint8)
    pts = np.array(
        [[float(p[0]), float(p[1])] for p in verts],
        dtype=np.float32,
    )
    pts_int = np.array(pts, dtype=np.int32)
    mask = np.zeros((plan_h, plan_w), dtype=np.uint8)
    if pts_int.size >= 6:
        cv2.fillPoly(mask, [pts_int], 255)
    return mask


def _face_centroid(verts: List[List[float]]) -> Tuple[float, float, float]:
    """Centroidul unei fețe 3D."""
    n = len(verts)
    if n == 0:
        return (0.0, 0.0, 0.0)
    cx = sum(float(p[0]) for p in verts) / n
    cy = sum(float(p[1]) for p in verts) / n
    cz = sum(float(p[2]) for p in verts) / n
    return (cx, cy, cz)


def _face_to_triangles(verts: List[List[float]]) -> List[List[List[float]]]:
    """Triangulează o față (quad sau triunghi) în triunghiuri."""
    if len(verts) == 3:
        return [verts]
    if len(verts) == 4:
        return [[verts[0], verts[1], verts[2]], [verts[0], verts[2], verts[3]]]
    if len(verts) > 4:
        tris = []
        for i in range(1, len(verts) - 1):
            tris.append([verts[0], verts[i], verts[i + 1]])
        return tris
    return []


def _render_roof_3d_numbered(
    faces: List[Tuple[int, str, List[List[float]]]],
    output_dir: Path,
) -> bool:
    """Randare 3D cu fețe pline și numere pe fiecare față.
    faces: listă de (face_id, label, verts) – face_id = numărul afișat pe față.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return False
    if not faces:
        return False

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    minx = miny = float("inf")
    maxx = maxy = maxz = float("-inf")
    for _, _, verts in faces:
        for p in verts:
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            minx, maxx = min(minx, x), max(maxx, x)
            miny, maxy = min(miny, y), max(maxy, y)
            maxz = max(maxz, z)
    pad = max(50, (maxx - minx + maxy - miny) * 0.1)
    minx_b, maxx_b = minx - pad, maxx + pad
    miny_b, maxy_b = miny - pad, maxy + pad
    z_max = maxz + 100

    fig = go.Figure()
    centroids_x, centroids_y, centroids_z = [], [], []
    labels_text = []

    for face_id, label, verts in faces:
        tris = _face_to_triangles(verts)
        if not tris:
            continue
        xs, ys, zs = [], [], []
        ii, jj, kk = [], [], []
        idx = 0
        for tri in tris:
            for v in tri:
                xs.append(float(v[0]))
                ys.append(float(v[1]))
                zs.append(float(v[2]))
            ii.append(idx)
            jj.append(idx + 1)
            kk.append(idx + 2)
            idx += 3
        color_hex = _FACE_COLORS[(face_id - 1) % len(_FACE_COLORS)]
        r = int(color_hex[1:3], 16) / 255.0
        g = int(color_hex[3:5], 16) / 255.0
        b = int(color_hex[5:7], 16) / 255.0
        fig.add_trace(
            go.Mesh3d(
                x=xs, y=ys, z=zs, i=ii, j=jj, k=kk,
                color=f"rgb({int(r*255)},{int(g*255)},{int(b*255)})",
                opacity=1.0, flatshading=True,
            )
        )
        cx, cy, cz = _face_centroid(verts)
        centroids_x.append(cx)
        centroids_y.append(cy)
        centroids_z.append(cz)
        labels_text.append(str(face_id))

    if centroids_x:
        fig.add_trace(
            go.Scatter3d(
                x=centroids_x, y=centroids_y, z=centroids_z,
                mode="text",
                text=labels_text,
                textfont=dict(size=18, color="black", family="Arial Black"),
                hoverinfo="text",
                hovertext=[f"Față {fid}" for fid, _, _ in faces],
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[minx_b, maxx_b], backgroundcolor="white"),
            yaxis=dict(range=[miny_b, maxy_b], backgroundcolor="white"),
            zaxis=dict(range=[0, z_max], backgroundcolor="white"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )

    html_path = output_dir / "roof_3d_numbered.html"
    png_path = output_dir / "roof_3d_numbered.png"
    try:
        fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    except Exception:
        pass
    try:
        fig.write_image(str(png_path), width=900, height=700, scale=2)
    except Exception:
        pass
    return True


def _get_faces_standard(
    sections: List[Dict[str, Any]],
    connections: List[Dict[str, Any]],
    roof_angle_deg: float,
) -> List[Tuple[str, List[List[float]]]]:
    """Fețe pentru acoperiș standard (gable)."""
    try:
        import roof_calc.roof_faces_3d as rf

        base_z = 0.0
        fs = rf.get_faces_3d_standard(
            sections, connections, roof_angle_deg=roof_angle_deg, wall_height=base_z
        )
        faces_out: List[Tuple[str, List[List[float]]]] = []
        for i, f in enumerate(fs):
            verts = f.get("vertices_3d")
            if verts and len(verts) >= 3:
                label = str(f.get("label", f"fata{i + 1}"))
                faces_out.append((label, verts))
        return faces_out
    except Exception:
        return []


def _get_faces_pyramid(
    sections: List[Dict[str, Any]],
    roof_angle_deg: float,
) -> List[Tuple[str, List[List[float]]]]:
    """Fețe pentru acoperiș piramidă."""
    try:
        from roof_calc.visualize import _free_roof_ends
        from roof_calc.visualize_3d_pyvista import _roof_section_faces_pyramid

        roof_angle_rad = math.radians(roof_angle_deg)
        conns = []
        free_ends = _free_roof_ends(sections, conns)
        faces_out: List[Tuple[str, List[List[float]]]] = []
        for s_idx, sec in enumerate(sections):
            fe = free_ends[s_idx] if s_idx < len(free_ends) else {}
            faces = _roof_section_faces_pyramid(sec, 0.0, roof_angle_rad, fe, None)
            for i, face in enumerate(faces):
                if face and len(face) >= 3:
                    label = f"sec{s_idx}_piramida_{i + 1}"
                    faces_out.append((label, face))
        return faces_out
    except Exception:
        return []


def _get_faces_shed(
    sections: List[Dict[str, Any]],
    roof_angle_deg: float,
    union_upper: Optional[Any] = None,
) -> List[Tuple[str, List[List[float]]]]:
    """Fețe pentru acoperiș shed."""
    try:
        from roof_calc.overhang import high_side_for_shed_from_upper_floor
        from roof_calc.visualize_3d_pyvista import _roof_section_faces_shed

        roof_angle_rad = math.radians(roof_angle_deg)
        high_sides = high_side_for_shed_from_upper_floor(sections, union_upper or [])
        faces_out: List[Tuple[str, List[List[float]]]] = []
        for s_idx, sec in enumerate(sections):
            hs = high_sides[s_idx] if s_idx < len(high_sides) else "top"
            faces = _roof_section_faces_shed(sec, 0.0, roof_angle_rad, hs)
            for i, face in enumerate(faces):
                if face and len(face) >= 3:
                    label = f"sec{s_idx}_shed_{i + 1}"
                    faces_out.append((label, face))
        return faces_out
    except Exception:
        return []


def generate_unfolded_faces(
    roof_type: str,
    sections: List[Dict[str, Any]],
    connections: List[Dict[str, Any]],
    shape: Tuple[int, ...],
    config: Dict[str, Any],
    output_dir: Path,
) -> int:
    """
    Generează măști unfold pentru toate fețele acoperișului.
    roof_type: "standard" | "pyramid" | "shed"
    shape: (H, W) din plan – pentru validare; dimensiunile fețelor sunt în aceeași scală.
    Returnează numărul de fețe generate.
    """
    if cv2 is None:
        return 0
    roof_angle = float(config.get("roof_angle", 30.0))

    if roof_type == "standard":
        faces = _get_faces_standard(sections, connections, roof_angle)
    elif roof_type == "pyramid":
        faces = _get_faces_pyramid(sections, roof_angle)
    elif roof_type == "shed":
        faces = _get_faces_shed(sections, roof_angle, None)
    else:
        return 0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_h, plan_w = (int(shape[0]), int(shape[1])) if len(shape) >= 2 else (0, 0)
    count = 0
    written_faces: List[Tuple[int, str, str]] = []
    faces_for_3d: List[Tuple[str, List[List[float]]]] = []
    for face_id, (label, verts) in enumerate(faces, start=1):
        res = _unfold_face_to_2d(verts)
        if res is None:
            continue
        poly_2d, width, height = res
        mask, w_px, h_px = _rasterize_polygon_to_mask(poly_2d, width, height)
        safe_label = label.replace(" ", "_").replace("/", "_")
        base_name = f"fata_{face_id}_{safe_label}"
        png_path = output_dir / f"{base_name}.png"
        meta_path = output_dir / f"{base_name}_meta.json"
        cv2.imwrite(str(png_path), mask)
        area = float(np.count_nonzero(mask))
        meta = {
            "face_id": face_id,
            "label": label,
            "width_units": round(width, 2),
            "height_units": round(height, 2),
            "area_units_sq": round(area, 2),
            "width_px": w_px,
            "height_px": h_px,
            "plan_h": plan_h,
            "plan_w": plan_w,
            "units": "pixels (same as input plan)",
            "scale": "1 px in image = 1 unit in 3D (true unfolded dimensions)",
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        written_faces.append((face_id, label, f"{base_name}.png"))
        faces_for_3d.append((face_id, label, verts))

        # Mască overlay: aceeași dimensiune ca planul (plan_h x plan_w), față la poziția corectă
        if plan_h > 0 and plan_w > 0:
            overlay = _rasterize_face_footprint_on_plan(verts, plan_h, plan_w)
            overlay_path = output_dir / f"{base_name}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)
        count += 1

    if count > 0:
        _render_roof_3d_numbered(faces_for_3d, output_dir)
        legend_lines = [
            "# Număr | Etichetă | Fișier mască crop | *_overlay.png (dimensiune plan)",
            "# ----- | -------- | ----------------- | ------------------------------",
        ]
        for fid, lbl, fname in written_faces:
            ov = fname.replace(".png", "_overlay.png") if plan_h > 0 and plan_w > 0 else "-"
            legend_lines.append(f"{fid} | {lbl} | {fname} | {ov}")
        (output_dir / "numerotare.txt").write_text("\n".join(legend_lines), encoding="utf-8")
        # Mască combinată: toate fețele pe plan (plan_h x plan_w)
        if plan_h > 0 and plan_w > 0 and faces_for_3d:
            combined = np.zeros((plan_h, plan_w), dtype=np.uint8)
            for _, _, verts in faces_for_3d:
                ov = _rasterize_face_footprint_on_plan(verts, plan_h, plan_w)
                np.maximum(combined, ov, out=combined)
            cv2.imwrite(str(output_dir / "toate_fetele_overlay.png"), combined)
    return count


def _segment_length_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _segment_length_2d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def _compute_streasure_burlane_lengths(
    roof_type: str,
    sections: List[Dict[str, Any]],
    connections: List[Dict[str, Any]],
    roof_angle_deg: float,
    wall_height: float,
    overhang_px: float,
    union_upper: Optional[Any],
    floors_payload: Optional[List[Tuple[Any, Any, Dict[str, Any], Tuple[float, float]]]],
) -> Dict[str, Any]:
    """
    Calculează lungimile exacte: streașini, burlane, conexiuni streașini-burlane.
    sections: secțiuni CU overhang.
    """
    try:
        from roof_calc.overhang import (
            get_gutter_segments_2d,
            get_gutter_endpoints_3d,
            get_downspout_faces_for_floors,
            get_downspout_faces_pyramid,
            high_side_for_shed_from_upper_floor,
        )
    except Exception:
        return {"streașini": {}, "burlane": {}, "conexiuni": {}, "eroare": "import"}

    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    exclude_low_sides: Optional[List[str]] = None
    if roof_type == "standard":
        segments = get_gutter_segments_2d(sections, include_eaves_only=True)
    elif roof_type == "pyramid":
        segments = get_gutter_segments_2d(sections, pyramid_all_sides=True)
    elif roof_type == "shed":
        high_sides = high_side_for_shed_from_upper_floor(sections, union_upper or [])
        exclude_low_sides = []
        for hs in high_sides:
            low = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}.get(hs, "bottom")
            exclude_low_sides.append(low)
        segments = get_gutter_segments_2d(sections, exclude_low_sides=exclude_low_sides)
    else:
        return {"streașini": {}, "burlane": {}, "conexiuni": {}}

    # Streașini – lungimi exacte din segmente 2D
    streașini_list: List[Dict[str, Any]] = []
    total_streasure_px = 0.0
    for i, ((x1, y1), (x2, y2)) in enumerate(segments):
        length_px = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_streasure_px += length_px
        streașini_list.append({
            "segment_id": i + 1,
            "length_px": round(length_px, 2),
            "p1": [round(x1, 2), round(y1, 2)],
            "p2": [round(x2, 2), round(y2, 2)],
        })

    burlane_list: List[Dict[str, Any]] = []
    conexiuni_list: List[Dict[str, Any]] = []
    total_burlane_px = 0.0
    total_conexiuni_px = 0.0

    if floors_payload and wall_height > 0:
        z_base = wall_height
        gutter_endpoints: List[Tuple[float, float, float]] = []
        gutter_segment_sections: List[List[Dict[str, Any]]] = []
        gutter_radius = max(2.0, overhang_px * 0.24) * 0.70 if overhang_px > 0 else None

        try:
            if roof_type == "standard":
                ep = get_gutter_endpoints_3d(
                    sections, float(z_base), overhang_px, roof_angle_deg,
                    include_eaves_only=True,
                    eaves_z_lift=overhang_px * 0.60,
                )
            elif roof_type == "pyramid":
                ep = get_gutter_endpoints_3d(
                    sections, float(z_base), overhang_px, roof_angle_deg,
                    pyramid_all_sides=True,
                    eaves_z_lift=overhang_px * 0.60,
                )
            else:
                ep = get_gutter_endpoints_3d(
                    sections, float(z_base), overhang_px, roof_angle_deg,
                    exclude_low_sides=exclude_low_sides or [],
                    eaves_z_lift=overhang_px * 0.60,
                )
            gutter_endpoints = list(ep)
            for _ in range(max(0, len(gutter_endpoints) // 2)):
                gutter_segment_sections.append(sections)
        except Exception:
            gutter_endpoints = []
            gutter_segment_sections = []

        if gutter_endpoints and len(gutter_endpoints) >= 2:
            try:
                if roof_type == "pyramid":
                    res = get_downspout_faces_pyramid(
                        floors_payload,
                        wall_height,
                        cylinder_radius=gutter_radius,
                        gutter_endpoints=gutter_endpoints,
                        gutter_segment_sections=gutter_segment_sections or None,
                        return_downspout_centerlines=True,
                    )
                else:
                    res = get_downspout_faces_for_floors(
                        floors_payload,
                        wall_height,
                        cylinder_radius=gutter_radius,
                        gutter_endpoints=gutter_endpoints,
                        gutter_segment_sections=gutter_segment_sections or None,
                        return_downspout_centerlines=True,
                    )
                if isinstance(res, tuple) and len(res) >= 3:
                    _faces, _used, cl_data = res[0], res[1], res[2]
                    downspout_cl = cl_data.get("downspout", [])
                    connection_cl = cl_data.get("connection", [])
                    for j, (p1, p2) in enumerate(downspout_cl):
                        len_3d = _segment_length_3d(p1, p2)
                        len_vertical = abs(p2[2] - p1[2])
                        total_burlane_px += len_vertical
                        burlane_list.append({
                            "burlan_id": j + 1,
                            "length_px": round(len_vertical, 2),
                            "length_3d_px": round(len_3d, 2),
                            "p1": [round(p1[0], 2), round(p1[1], 2), round(p1[2], 2)],
                            "p2": [round(p2[0], 2), round(p2[1], 2), round(p2[2], 2)],
                        })
                    for k, (p1, p2) in enumerate(connection_cl):
                        len_3d = _segment_length_3d(p1, p2)
                        len_2d = _segment_length_2d(p1, p2)
                        total_conexiuni_px += len_3d
                        conexiuni_list.append({
                            "conexiune_id": k + 1,
                            "length_px": round(len_3d, 2),
                            "length_2d_px": round(len_2d, 2),
                            "p1": [round(p1[0], 2), round(p1[1], 2), round(p1[2], 2)],
                            "p2": [round(p2[0], 2), round(p2[1], 2), round(p2[2], 2)],
                        })
            except Exception:
                pass

    z_gutter = 0.0
    if wall_height > 0 and overhang_px >= 0:
        tan_a = math.tan(math.radians(roof_angle_deg))
        z_gutter = wall_height - overhang_px * tan_a + overhang_px * 0.60

    return {
        "streașini": {
            "segments": streașini_list,
            "total_length_px": round(total_streasure_px, 2),
            "unitate": "pixeli (raportate la planul din care se face acoperișul)",
        },
        "burlane": {
            "segments": burlane_list,
            "total_length_px": round(total_burlane_px, 2),
            "count": len(burlane_list),
            "unitate": "pixeli (lungime verticală)",
        },
        "conexiuni_streasure_burlane": {
            "segments": conexiuni_list,
            "total_length_px": round(total_conexiuni_px, 2),
            "count": len(conexiuni_list),
            "unitate": "pixeli (lungime 3D)",
        },
        "render_meta": {
            "z_gutter": round(z_gutter, 2),
            "roof_angle_deg": roof_angle_deg,
            "overhang_px": overhang_px,
            "wall_height": wall_height,
        },
    }


def generate_roof_unfolded_all_types(
    floor_path: str,
    roof_result: Dict[str, Any],
    shape: Tuple[int, ...],
    config: Dict[str, Any],
    base_output_dir: Path,
    floor_level: int,
    *,
    overhang_px: float = 0.0,
    footprint: Optional[Any] = None,
    wall_height: float = 0.0,
    union_upper: Optional[Any] = None,
    floors_payload: Optional[List[Tuple[Any, Any, Dict[str, Any], Tuple[float, float]]]] = None,
) -> Dict[str, int]:
    """
    Generează unfold pentru toate tipurile: standard, pyramid, shed.
    Măști separate: fara_overhang/ (fețe fără overhang) și cu_overhang/ (fețe cu overhang).
    Generează lungimi_streasure_burlane.json cu lungimile în pixeli.
    """
    try:
        from roof_calc.overhang import (
            apply_overhang_to_sections,
            compute_overhang_sides_from_footprint,
            compute_overhang_sides_from_union_boundary,
        )
    except Exception:
        apply_overhang_to_sections = None
        compute_overhang_sides_from_footprint = None
        compute_overhang_sides_from_union_boundary = None

    sections = roof_result.get("sections") or []
    connections = roof_result.get("connections") or []
    if not sections:
        return {}

    sections_base = sections
    if overhang_px > 0 and apply_overhang_to_sections and compute_overhang_sides_from_union_boundary:
        if footprint is not None and compute_overhang_sides_from_footprint:
            free_sides = compute_overhang_sides_from_footprint(sections, footprint)
        else:
            free_sides = compute_overhang_sides_from_union_boundary(sections)
        sections_oh = apply_overhang_to_sections(sections, overhang_px=overhang_px, free_sides=free_sides)
    else:
        sections_oh = sections

    roof_angle = float(config.get("roof_angle", 30.0))
    wh = float(config.get("wall_height", 0.0) or wall_height or 0.0)

    fp = floors_payload
    if fp is None and footprint is not None:
        fp = [(floor_path, footprint, roof_result, (0.0, 0.0))]

    counts: Dict[str, int] = {}
    lengths_by_type: Dict[str, Dict[str, Any]] = {}

    for roof_type in ("standard", "pyramid", "shed"):
        base_out = base_output_dir / roof_type / f"etaj_{floor_level}"

        out_fara = base_out / "fara_overhang"
        n_fara = generate_unfolded_faces(
            roof_type, sections_base, connections, shape, config, out_fara
        )

        out_cu = base_out / "cu_overhang"
        n_cu = generate_unfolded_faces(
            roof_type, sections_oh, connections, shape, config, out_cu
        )

        counts[roof_type] = max(n_fara, n_cu)

        lengths_by_type[roof_type] = _compute_streasure_burlane_lengths(
            roof_type,
            sections_oh,
            connections,
            roof_angle,
            wh,
            overhang_px,
            union_upper,
            fp,
        )

    for roof_type in ("standard", "pyramid", "shed"):
        base_out = base_output_dir / roof_type / f"etaj_{floor_level}"
        lengths_path = base_out / "lungimi_streasure_burlane.json"
        data = lengths_by_type[roof_type]
        lengths_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        html_path = base_out / "lungimi_3d_interactive.html"
        render_lengths_3d_interactive(data, html_path, roof_type=roof_type)

    return counts


def render_lengths_3d_interactive(
    lengths_data: Dict[str, Any],
    output_path: Path,
    *,
    roof_type: str = "",
) -> bool:
    """
    Generează o randare 3D interactivă (HTML) cu segmente streașini, burlane, conexiuni.
    La hover pe fiecare segment se afișează lungimea în pixeli.
    Poate fi deschis online într-un browser.
    """
    if lengths_data.get("eroare"):
        return False
    try:
        import plotly.graph_objects as go
    except ImportError:
        return False

    meta = lengths_data.get("render_meta") or {}
    z_gutter = float(meta.get("z_gutter", 0.0))
    if z_gutter == 0:
        for seg in (lengths_data.get("conexiuni_streasure_burlane") or {}).get("segments") or []:
            p2 = seg.get("p2", [])
            if len(p2) >= 3:
                z_gutter = max(z_gutter, float(p2[2]))
        if z_gutter == 0:
            for seg in (lengths_data.get("burlane") or {}).get("segments") or []:
                p2 = seg.get("p2", [])
                if len(p2) >= 3:
                    z_gutter = max(z_gutter, float(p2[2]))

    fig = go.Figure()
    all_x, all_y, all_z = [], [], []

    # Streașini – segmente 2D la z=z_gutter
    streașini = lengths_data.get("streașini") or {}
    for seg in streașini.get("segments") or []:
        p1, p2 = seg.get("p1", []), seg.get("p2", [])
        if len(p1) >= 2 and len(p2) >= 2:
            length = seg.get("length_px", 0)
            x = [float(p1[0]), float(p2[0])]
            y = [float(p1[1]), float(p2[1])]
            z = [z_gutter, z_gutter]
            all_x.extend(x)
            all_y.extend(y)
            all_z.extend(z)
            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode="lines",
                    line=dict(color="#2196F3", width=8),
                    name=f"Streașină {seg.get('segment_id', 0)}",
                    hovertext=f"Streașină segment {seg.get('segment_id', 0)}: <b>{length} px</b>",
                    hoverinfo="text",
                )
            )

    # Burlane – segmente 3D
    burlane = lengths_data.get("burlane") or {}
    for seg in burlane.get("segments") or []:
        p1, p2 = seg.get("p1", []), seg.get("p2", [])
        if len(p1) >= 3 and len(p2) >= 3:
            length = seg.get("length_px", 0)
            x = [float(p1[0]), float(p2[0])]
            y = [float(p1[1]), float(p2[1])]
            z = [float(p1[2]), float(p2[2])]
            all_x.extend(x)
            all_y.extend(y)
            all_z.extend(z)
            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode="lines",
                    line=dict(color="#4CAF50", width=8),
                    name=f"Burlan {seg.get('burlan_id', 0)}",
                    hovertext=f"Burlan {seg.get('burlan_id', 0)}: <b>{length} px</b>",
                    hoverinfo="text",
                )
            )

    # Conexiuni streașini–burlane
    conexiuni = lengths_data.get("conexiuni_streasure_burlane") or {}
    for seg in conexiuni.get("segments") or []:
        p1, p2 = seg.get("p1", []), seg.get("p2", [])
        if len(p1) >= 3 and len(p2) >= 3:
            length = seg.get("length_px", 0)
            x = [float(p1[0]), float(p2[0])]
            y = [float(p1[1]), float(p2[1])]
            z = [float(p1[2]), float(p2[2])]
            all_x.extend(x)
            all_y.extend(y)
            all_z.extend(z)
            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode="lines",
                    line=dict(color="#FF9800", width=6),
                    name=f"Conexiune {seg.get('conexiune_id', 0)}",
                    hovertext=f"Conexiune {seg.get('conexiune_id', 0)}: <b>{length} px</b>",
                    hoverinfo="text",
                )
            )

    if not all_x or not all_y or not all_z:
        return False

    pad = max(80, (max(all_x) - min(all_x) + max(all_y) - min(all_y)) * 0.15)
    fig.update_layout(
        title=f"Lungimi streașini, burlane, conexiuni (px) – {roof_type or 'acoperiș'}",
        scene=dict(
            xaxis=dict(range=[min(all_x) - pad, max(all_x) + pad], title="x"),
            yaxis=dict(range=[min(all_y) - pad, max(all_y) + pad], title="y"),
            zaxis=dict(range=[max(0, min(all_z) - 50), max(all_z) + 50], title="z"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.2)),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    try:
        fig.write_html(
            str(output_path),
            include_plotlyjs="cdn",
            full_html=True,
            config=dict(displayModeBar=True, responsive=True),
        )
        return True
    except Exception:
        return False
