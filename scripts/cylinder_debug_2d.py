#!/usr/bin/env python3
"""
Generează fișiere 2D de debug pentru fiecare tip de acoperiș (a_frame, shed, pyramid),
afișând cum sunt generate și poziționați cilindrii (burlane + cilindru la jgheab).

Utilizare:
  python scripts/cylinder_debug_2d.py [output_test]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
from shapely import affinity as shapely_affinity

from roof_calc import calculate_roof_from_walls
from roof_calc.overhang import (
    apply_overhang_to_sections,
    compute_overhang_px_from_roof_results,
    compute_overhang_sides_from_footprint,
    compute_overhang_sides_from_union_boundary,
    compute_overhang_sides_from_free_ends,
    get_cylinder_positions_debug,
    get_gutter_endpoints_3d,
    high_side_for_shed_from_upper_floor,
)
from roof_calc.visualize import _free_roof_ends
from roof_calc.visualize_3d_matplotlib import (
    _compute_offsets,
    _covered_by_upper,
    _polygon_from_path,
    _translate_sections,
)


def _build_floors_payload(
    all_floor_paths: List[str],
    floor_roof_results: List[Dict[str, Any]],
) -> List[Tuple[str, Any, Dict[str, Any], Tuple[int, int]]]:
    offsets = _compute_offsets(all_floor_paths, floor_roof_results)
    floors_payload: List[Tuple[str, Any, Dict[str, Any], Tuple[int, int]]] = []
    for p, rr in zip(all_floor_paths, floor_roof_results):
        poly0 = _polygon_from_path(p)
        if poly0 is None:
            continue
        ox, oy = offsets.get(p, (0, 0))
        poly_t = shapely_affinity.translate(poly0, xoff=ox, yoff=oy)
        floors_payload.append((p, poly_t, rr, (ox, oy)))
    if floors_payload:
        floors_payload.sort(key=lambda t: float(getattr(t[1], "area", 0.0) or 0.0), reverse=True)
    return floors_payload


def _build_gutter_endpoints_a_frame(
    floors_payload: List[Tuple[str, Any, Dict[str, Any], Tuple[int, int]]],
    wall_height: float,
    overhang_px: float,
    roof_angle_deg: float,
    roof_shift_dz: float = 0.0,
) -> List[Tuple[float, float, float]]:
    gutter_endpoints: List[Tuple[float, float, float]] = []
    num_floors = len(floors_payload)
    for floor_idx, (_p, poly, rr, (ox, oy)) in enumerate(floors_payload):
        z1 = (floor_idx + 1) * wall_height
        union_above = (
            unary_union([fp[1] for fp in floors_payload[floor_idx + 1 :]])
            if floor_idx + 1 < num_floors
            else None
        )
        secs = rr.get("sections") or []
        conns = rr.get("connections") or []
        kept: List[Dict[str, Any]] = []
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
        draw_roof = floor_idx == num_floors - 1
        if not draw_roof or not kept:
            continue
        free = compute_overhang_sides_from_union_boundary(kept)
        kept_use = apply_overhang_to_sections(kept, overhang_px=overhang_px, free_sides=free)
        roof_faces = None  # nu e necesar pentru capete
        gutter_endpoints.extend(
            get_gutter_endpoints_3d(
                kept_use,
                float(z1),
                overhang_px,
                roof_angle_deg,
                roof_shift_dz=roof_shift_dz,
                roof_faces=roof_faces,
                include_eaves_only=True,
                eaves_z_lift=overhang_px * 0.60,
            )
        )
    return gutter_endpoints


def _build_gutter_endpoints_shed(
    floors_payload: List[Tuple[str, Any, Dict[str, Any], Tuple[int, int]]],
    roof_levels: Optional[List[Tuple]],
    wall_height: float,
    overhang_px: float,
    roof_angle_deg: float,
    roof_shift_dz: float = 0.0,
) -> List[Tuple[float, float, float]]:
    from roof_calc.visualize_3d_pyvista import _roof_section_faces_shed

    gutter_endpoints: List[Tuple[float, float, float]] = []
    num_floors = len(floors_payload)
    roof_angle_rad = np.radians(roof_angle_deg)

    # Top floor: a_frame style (include_eaves_only)
    for floor_idx, (_p, poly, rr, (ox, oy)) in enumerate(floors_payload):
        if floor_idx != num_floors - 1:
            continue
        z1 = (floor_idx + 1) * wall_height
        union_above = None
        secs = rr.get("sections") or []
        kept: List[Dict[str, Any]] = []
        for sec in secs:
            br = sec.get("bounding_rect", [])
            if len(br) < 3:
                continue
            sp = ShapelyPolygon(br)
            sp = shapely_affinity.translate(sp, xoff=ox, yoff=oy)
            if _covered_by_upper(sp, union_above or ShapelyPolygon(), area_thresh=500.0):
                continue
            sec_t = _translate_sections([sec], float(ox), float(oy))[0]
            kept.append(sec_t)
        if not kept:
            continue
        free = compute_overhang_sides_from_union_boundary(kept)
        kept_use = apply_overhang_to_sections(kept, overhang_px=overhang_px, free_sides=free)
        gutter_endpoints.extend(
            get_gutter_endpoints_3d(
                kept_use,
                float(z1),
                overhang_px,
                roof_angle_deg,
                roof_shift_dz=roof_shift_dz,
                include_eaves_only=True,
                eaves_z_lift=overhang_px * 0.60,
            )
        )

    # Lower floors: shed (exclude_low_sides)
    if roof_levels:
        for item in roof_levels:
            z_base, roof_data, dx, dy, fl = item[0], item[1], item[2], item[3], item[4]
            secs0 = roof_data.get("sections") or []
            secs_t = _translate_sections(secs0, float(dx), float(dy))
            footprint = floors_payload[fl][1] if 0 <= fl < len(floors_payload) else None
            free = (
                compute_overhang_sides_from_footprint(secs_t, footprint)
                if footprint
                else compute_overhang_sides_from_union_boundary(secs_t)
            )
            secs_t = apply_overhang_to_sections(secs_t, overhang_px=overhang_px, free_sides=free)
            union_upper = (
                unary_union([floors_payload[i][1] for i in range(fl + 1, len(floors_payload))])
                if 0 <= fl < len(floors_payload)
                else None
            )
            high_sides = high_side_for_shed_from_upper_floor(secs_t, union_upper)
            _low = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
            exclude_low_sides = [
                _low.get(high_sides[i] if i < len(high_sides) else "top", "bottom")
                for i in range(len(secs_t))
            ]
            faces = []
            for s_idx, sec in enumerate(secs_t):
                hs = high_sides[s_idx] if s_idx < len(high_sides) else "top"
                for face in _roof_section_faces_shed(sec, float(z_base), roof_angle_rad, hs):
                    faces.append({"vertices_3d": face})
            gutter_endpoints.extend(
                get_gutter_endpoints_3d(
                    secs_t,
                    float(z_base),
                    overhang_px,
                    roof_angle_deg,
                    roof_shift_dz=roof_shift_dz,
                    roof_faces=faces,
                    exclude_low_sides=exclude_low_sides,
                )
            )
    return gutter_endpoints


def _build_gutter_endpoints_pyramid(
    floors_payload: List[Tuple[str, Any, Dict[str, Any], Tuple[int, int]]],
    wall_height: float,
    overhang_px: float,
    roof_angle_deg: float,
    roof_shift_dz: float = 0.0,
) -> List[Tuple[float, float, float]]:
    gutter_endpoints: List[Tuple[float, float, float]] = []
    num_floors = len(floors_payload)
    for floor_idx, (_p, poly, rr, (ox, oy)) in enumerate(floors_payload):
        z1 = (floor_idx + 1) * wall_height
        union_above = (
            unary_union([fp[1] for fp in floors_payload[floor_idx + 1 :]])
            if floor_idx + 1 < num_floors
            else None
        )
        secs_base = _translate_sections(rr.get("sections") or [], float(ox), float(oy))
        conns = rr.get("connections") or []
        free_ends = _free_roof_ends(secs_base, conns)
        draw_roof = floor_idx == num_floors - 1
        secs = secs_base
        if draw_roof and overhang_px > 0 and secs_base:
            free = compute_overhang_sides_from_free_ends(secs_base, free_ends)
            secs = apply_overhang_to_sections(secs_base, overhang_px=overhang_px, free_sides=free)
        roof_faces: List[Dict[str, Any]] = []
        for s_idx, sec in enumerate(secs):
            br = sec.get("bounding_rect", [])
            if len(br) < 3:
                continue
            sp = ShapelyPolygon(br)
            if draw_roof and _covered_by_upper(sp, union_above or ShapelyPolygon(), area_thresh=500.0):
                continue
            from roof_calc.visualize_3d_pyvista import _roof_section_faces_pyramid

            upper_secs_all: List[Dict[str, Any]] = []
            for _pp, _poly_u, rr_u, (ox_u, oy_u) in floors_payload[floor_idx + 1 :]:
                upper_secs_all.extend(
                    _translate_sections(rr_u.get("sections") or [], float(ox_u - ox), float(oy_u - oy))
                )
            fe = free_ends[s_idx] if s_idx < len(free_ends) else {}
            roof_faces.extend(
                _roof_section_faces_pyramid(
                    sec, z1, np.radians(roof_angle_deg), fe, upper_secs_all
                )
            )
        all_faces = [{"vertices_3d": f} for f in roof_faces]
        if not draw_roof or not secs or not all_faces:
            continue
        gutter_endpoints.extend(
            get_gutter_endpoints_3d(
                secs,
                float(z1),
                overhang_px,
                roof_angle_deg,
                roof_shift_dz=roof_shift_dz,
                roof_faces=all_faces,
                pyramid_all_sides=True,
                eaves_z_lift=overhang_px * 0.60,
            )
        )
    return gutter_endpoints


def _draw_cylinder_debug_2d(
    floors_payload: List[Tuple[str, Any, Dict[str, Any], Tuple[int, int]]],
    gutter_endpoints: List[Tuple[float, float, float]],
    debug_positions: List[Dict[str, Any]],
    roof_type: str,
    output_path: Path,
    wall_height: float = 300.0,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    # Footprint-uri (contur clădire)
    for _p, poly, _rr, _off in floors_payload:
        if hasattr(poly, "exterior"):
            coords = list(poly.exterior.coords)
            xs = [float(c[0]) for c in coords]
            ys = [float(c[1]) for c in coords]
            ax.plot(xs, ys, "k-", linewidth=2, label="Clădire" if _p == floors_payload[0][0] else None)
            ax.fill(xs, ys, alpha=0.15, color="gray")

    # Segmente jgheab (streașină)
    for i in range(0, len(gutter_endpoints) - 1, 2):
        p1, p2 = gutter_endpoints[i], gutter_endpoints[i + 1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "b-", linewidth=2, alpha=0.8)
    ax.plot([], [], "b-", linewidth=2, label="Jgheab (streașină)")

    # Capete jgheab
    for i in range(0, len(gutter_endpoints) - 1, 2):
        for j in (i, i + 1):
            gx, gy, _ = gutter_endpoints[j]
            ax.plot(gx, gy, "ko", markersize=6)
    ax.plot([], [], "ko", markersize=6, label="Capete jgheab")

    # Poziții cilindri (burlan vertical + cilindru la jgheab)
    for d in debug_positions:
        cx_cyl = d.get("cx_cyl")
        cy_cyl = d.get("cy_cyl")
        cx_gut = d.get("cx_gut")
        cy_gut = d.get("cy_gut")
        gx = d.get("gx")
        gy = d.get("gy")
        cx = d.get("cx")
        cy = d.get("cy")
        perp2x = d.get("perp2x", 0)
        perp2y = d.get("perp2y", 0)
        r = d.get("radius", 10)

        if cx_cyl is not None and cy_cyl is not None:
            circle = plt.Circle((cx_cyl, cy_cyl), r, fill=False, color="green", linewidth=2, linestyle="-")
            ax.add_patch(circle)
            ax.plot(cx_cyl, cy_cyl, "g+", markersize=12)
        if cx_gut is not None and cy_gut is not None:
            circle2 = plt.Circle((cx_gut, cy_gut), r, fill=False, color="purple", linewidth=1.5, linestyle="--")
            ax.add_patch(circle2)
            ax.plot(cx_gut, cy_gut, "m+", markersize=10)
        if gx is not None and gy is not None:
            ax.plot(gx, gy, "rs", markersize=8)
        if cx is not None and cy is not None:
            ax.plot(cx, cy, "c^", markersize=8)
        if abs(perp2x) > 1e-6 or abs(perp2y) > 1e-6:
            scale = r * 2.5
            ax.arrow(gx, gy, perp2x * scale, perp2y * scale, head_width=r * 0.5, head_length=r * 0.3, fc="orange", ec="orange")

    ax.plot([], [], "g-", linewidth=2, label="Burlan vertical (centru)")
    ax.plot([], [], "m--", linewidth=1.5, label="Cilindru la jgheab (centru)")
    ax.plot([], [], "rs", markersize=8, label="Endpoint jgheab")
    ax.plot([], [], "c^", markersize=8, label="Colț clădire")
    ax.plot([], [], color="orange", marker=r"$\rightarrow$", linestyle="None", markersize=12, label="perp2 (spre exterior)")

    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Debug cilindri – {roof_type}\nVerde=burlan vertical, Violet=cilindru la jgheab, Săgeată=perp2")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"   Salvat: {output_path}")


def generate_cylinder_debug_2d(out_dir: Path) -> bool:
    """
    Generează imaginile 2D de debug pentru cilindri (a_frame, shed, pyramid).
    Returnează True la succes.
    """
    if not out_dir.exists():
        print(f"Directorul {out_dir} nu există. Rulează mai întâi complete_workflow.")
        return False

    # Detectează etajele: din output_test/etaj_0, etaj_1 (sau input folder cu măști)
    etaj0 = out_dir / "etaj_0"
    etaj1 = out_dir / "etaj_1"
    all_floor_paths: List[str] = []
    # Preferăm remaining_mask pentru etaj inferior (zona vizibilă), roof_mask_binary pentru superior
    if etaj0.exists():
        for m in ["remaining_mask.png", "roof_mask_binary.png"]:
            p = etaj0 / m
            if p.exists():
                all_floor_paths.append(str(p))
                break
    if etaj1.exists():
        for m in ["roof_mask_binary.png"]:
            p = etaj1 / m
            if p.exists():
                all_floor_paths.append(str(p))
                break
    if not all_floor_paths and etaj0.exists():
        for p in sorted(etaj0.glob("roof_mask*.png")):
            all_floor_paths.append(str(p))
            break
    if not all_floor_paths:
        for p in sorted(out_dir.glob("**/roof_mask_binary.png")):
            all_floor_paths.append(str(p))
    if not all_floor_paths:
        print("Nu s-au găsit măști de etaj. Rulează complete_workflow pe un folder cu blueprint-uri.")
        return False

    # Sortează după arie (mai mare = parter)
    def _area(p: str) -> float:
        try:
            import cv2
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            return (im.shape[0] * im.shape[1]) if im is not None else 0
        except Exception:
            return 0

    all_floor_paths.sort(key=_area)

    print("Calcul acoperiș per etaj...")
    floor_roof_results: List[Dict[str, Any]] = []
    for fp in all_floor_paths:
        rr = calculate_roof_from_walls(fp, roof_angle=30, overhang_px=2.0)
        floor_roof_results.append(rr)
    overhang_px = compute_overhang_px_from_roof_results(floor_roof_results, ratio=0.05)
    roof_angle_deg = 30.0
    wall_height = 300.0
    import math
    roof_shift_dz = float(math.tan(math.radians(roof_angle_deg)) * overhang_px)

    floors_payload = _build_floors_payload(all_floor_paths, floor_roof_results)
    if not floors_payload:
        print("Nu s-au putut construi floors_payload.")
        return False

    # roof_levels pentru etaje inferioare (shed) – opțional; dacă lipsește, shed folosește doar etajul superior
    roof_levels: Optional[List[Tuple]] = None

    debug_dir = out_dir / "cylinders_debug_2d"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # A-Frame
    print("Generez cylinders_debug_a_frame.png...")
    gutter_af = _build_gutter_endpoints_a_frame(
        floors_payload, wall_height, overhang_px, roof_angle_deg, roof_shift_dz
    )
    debug_af = get_cylinder_positions_debug(
        floors_payload, gutter_af, wall_height=wall_height, cylinder_radius=None
    )
    _draw_cylinder_debug_2d(
        floors_payload, gutter_af, debug_af, "a_frame",
        debug_dir / "cylinders_debug_a_frame.png", wall_height
    )

    # Shed
    print("Generez cylinders_debug_shed.png...")
    gutter_shed = _build_gutter_endpoints_shed(
        floors_payload, roof_levels, wall_height, overhang_px, roof_angle_deg, roof_shift_dz
    )
    debug_shed = get_cylinder_positions_debug(
        floors_payload, gutter_shed, wall_height=wall_height, cylinder_radius=None
    )
    _draw_cylinder_debug_2d(
        floors_payload, gutter_shed, debug_shed, "shed",
        debug_dir / "cylinders_debug_shed.png", wall_height
    )

    # Pyramid
    print("Generez cylinders_debug_pyramid.png...")
    gutter_pyr = _build_gutter_endpoints_pyramid(
        floors_payload, wall_height, overhang_px, roof_angle_deg, roof_shift_dz
    )
    debug_pyr = get_cylinder_positions_debug(
        floors_payload, gutter_pyr, wall_height=wall_height, cylinder_radius=None
    )
    _draw_cylinder_debug_2d(
        floors_payload, gutter_pyr, debug_pyr, "pyramid",
        debug_dir / "cylinders_debug_pyramid.png", wall_height
    )

    print(f"\nFișiere 2D de debug generate în: {debug_dir}")
    return True


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output_test"
    generate_cylinder_debug_2d(out_dir)


if __name__ == "__main__":
    main()
