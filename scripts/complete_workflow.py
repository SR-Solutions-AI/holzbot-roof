#!/usr/bin/env python3
"""
Workflow complet pentru analiza acoperișurilor: de la blueprint la preț final.
"""

import sys
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Any

# Add project root so "roof_calc" is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Ensure Matplotlib/Fontconfig caches are writable (important in sandboxed runs).
_mpl = ROOT / ".mplconfig"
_cache = ROOT / ".cache"
_mpl.mkdir(exist_ok=True)
_cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache))
os.environ.setdefault("MPLBACKEND", "Agg")

from roof_calc import calculate_roof_from_walls
from roof_calc.pricing import calculate_materials_and_price
from roof_calc.visualize import (
    visualize_rectangles,
    visualize_roof_lines,
    visualize_individual_rectangles,
    visualize_pyramid_lines,
    visualize_a_frame_lines,
    _ordered_floor_polygons,
    _get_largest_rect_bbox,
)
from roof_calc.visualize_3d_matplotlib import (
    visualize_3d_standard_matplotlib,
    visualize_3d_pyramid_matplotlib,
)
from roof_calc.visualize_3d_plotly import (
    visualize_3d_standard_plotly,
    visualize_3d_pyramid_plotly,
)
from roof_calc.masks import (
    generate_binary_mask,
    generate_sections_mask,
    generate_gradient_mask,
    generate_texture_mask,
    generate_comparison,
    generate_roof_texture_standard,
    generate_roof_texture_pyramid,
    generate_roof_faces_unfolded_standard,
    generate_roof_faces_unfolded_pyramid,
    save_roof_overview_numbered,
)

from shapely.ops import unary_union
from shapely import affinity as shapely_affinity
import cv2
import numpy as np


MIN_REMAINING_AREA_PX = 500


def _rasterize_polygon_to_mask(geom: Any, padding: int = 10) -> Optional[np.ndarray]:
    """Rasterizează un poligon Shapely (sau MultiPolygon) într-o mască binară (alb = interior)."""
    try:
        if geom.is_empty:
            return None
        minx, miny, maxx, maxy = geom.bounds
        w = int(maxx - minx) + 2 * padding
        h = int(maxy - miny) + 2 * padding
        if w <= 0 or h <= 0:
            return None
        translated = shapely_affinity.translate(geom, -minx + padding, -miny + padding)
        mask = np.zeros((h, w), dtype=np.uint8)
        geoms = getattr(translated, "geoms", [translated])
        for g in geoms:
            if g.is_empty:
                continue
            ext = getattr(g, "exterior", None)
            if ext is None:
                continue
            coords = np.array(ext.coords[:-1], dtype=np.int32)
            if coords.size < 6:
                continue
            cv2.fillPoly(mask, [coords], 255)
        if np.count_nonzero(mask) < MIN_REMAINING_AREA_PX:
            return None
        return mask
    except Exception:
        return None


def _compute_lower_floor_roofs(
    all_floor_paths: List[str],
    floor_roof_results: List[dict],
    roof_floor_path: Optional[str] = None,
    wall_height: float = 300.0,
    output_base: Optional[Path] = None,
    main_roof_orientation: Optional[str] = None,
) -> List[Tuple[float, dict, float, float, int]]:
    """
    Pentru etaje suprapuse: calculează zona rămasă pe fiecare etaj inferior (nevăzută de etajul de sus)
    și generează acoperiș pentru acea zonă. Returnează [(z_base, roof_data, offset_x, offset_y, floor_level), ...].
    offset_* traduce roof_data din spațiul măștii în spațiul aliniat al etajelor.
    """
    if len(all_floor_paths) < 2 or len(floor_roof_results) != len(all_floor_paths):
        return []
    floors_ordered = _ordered_floor_polygons(all_floor_paths, roof_floor_path, floor_roof_results)
    if len(floors_ordered) < 2:
        return []

    path_to_result = {all_floor_paths[i]: floor_roof_results[i] for i in range(len(all_floor_paths))}

    # Același centru de referință ca la alinierea etajelor (bazat pe etajul acoperișului)
    cx_ref = 0.0
    cy_ref = 0.0
    try:
        rr = path_to_result.get(roof_floor_path or "")
        if rr is None and floor_roof_results:
            rr = floor_roof_results[-1]
        bbox = _get_largest_rect_bbox(rr) if rr is not None else None
        if bbox:
            cx_ref = (bbox[0] + bbox[2]) / 2.0
            cy_ref = (bbox[1] + bbox[3]) / 2.0
    except Exception:
        cx_ref = 0.0
        cy_ref = 0.0

    result: List[Tuple[float, dict, float, float, int]] = []
    for i in range(len(floors_ordered) - 1):
        _path_i, poly_i = floors_ordered[i]
        floor_level = i
        union_above = unary_union([p for _path, p in floors_ordered[i + 1 :]])
        try:
            remaining = poly_i.difference(union_above)
        except Exception:
            continue
        if remaining is None or remaining.is_empty:
            continue
        area = getattr(remaining, "area", 0) or 0
        if area < MIN_REMAINING_AREA_PX:
            continue

        minx, miny, _mx, _my = remaining.bounds
        mask = _rasterize_polygon_to_mask(remaining, padding=10)
        if mask is None:
            continue

        # Dreptunghiuri etaje de deasupra (rectangles_floor) în coordonatele măștii (padding=10).
        adjacent_floor_rects: List[Any] = []
        for up_path, _up_poly in floors_ordered[i + 1 :]:
            rr_up = path_to_result.get(up_path)
            if rr_up is None:
                continue
            for sec in rr_up.get("sections", []) or []:
                br = sec.get("bounding_rect", [])
                if not br or len(br) < 3:
                    continue
                try:
                    from shapely.geometry import Polygon as ShapelyPolygon

                    rpoly = ShapelyPolygon(br)
                    # aliniază ca floors_ordered (același -cx_ref, -cy_ref)
                    rpoly = shapely_affinity.translate(rpoly, xoff=-cx_ref, yoff=-cy_ref)
                    # translate în coordonatele măștii remaining (padding=10)
                    rpoly = shapely_affinity.translate(rpoly, xoff=-minx + 10, yoff=-miny + 10)
                    adjacent_floor_rects.append(rpoly)
                except Exception:
                    continue

        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                cv2.imwrite(f.name, mask)
                roof_data = calculate_roof_from_walls(
                    f.name,
                    roof_angle=30,
                    overhang_px=5,
                    adjacent_floor_rects=adjacent_floor_rects,
                    main_roof_orientation=main_roof_orientation,
                )
            Path(f.name).unlink(missing_ok=True)
        except Exception:
            continue

        if not roof_data.get("sections"):
            continue

        z_base = (i + 1) * wall_height
        if output_base is not None:
            etaj_dir = output_base / f"etaj_{floor_level}"
            etaj_dir.mkdir(parents=True, exist_ok=True)
            mask_path = etaj_dir / "remaining_mask.png"
            cv2.imwrite(str(mask_path), mask)

        result.append((z_base, roof_data, float(minx - 10), float(miny - 10), floor_level))

    return result


def _transform_roof_sections_to_mask_coords(
    sections: List[dict],
    offset_x: float,
    offset_y: float,
    source_center_x: Optional[float] = None,
    source_center_y: Optional[float] = None,
) -> List[dict]:
    cx = source_center_x if source_center_x is not None else 0.0
    cy = source_center_y if source_center_y is not None else 0.0
    tx = offset_x
    ty = offset_y
    out = []
    for sec in sections:
        rect = sec.get("bounding_rect", [])
        ridge = sec.get("ridge_line", [])
        new_rect = [(float(p[0]) - cx - tx, float(p[1]) - cy - ty) for p in rect] if rect else []
        new_ridge = [(float(p[0]) - cx - tx, float(p[1]) - cy - ty) for p in ridge] if len(ridge) >= 2 else []
        out.append(
            {
                "bounding_rect": new_rect,
                "ridge_line": new_ridge,
                "ridge_orientation": sec.get("ridge_orientation", "horizontal"),
                "is_main": sec.get("is_main", False),
            }
        )
    return out


def _generate_roof_outputs(
    wall_mask_path: str,
    roof_result: dict,
    out_dir: Path,
    config: dict,
    upper_floor_roof_sections: Optional[List[dict]] = None,
    upper_floor_footprint: Optional[List[List[Tuple[float, float]]]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # `rectangles_floor/` (din imaginea etajului) e sursa de adevăr.
    # Nu generăm / nu folosim `rectangles_remaining`.

    visualize_rectangles(wall_mask_path, roof_result, output_path=str(out_dir / "rectangles.png"))
    visualize_roof_lines(wall_mask_path, roof_result, output_path=str(out_dir / "roof_lines.png"))

    try:
        visualize_pyramid_lines(
            wall_mask_path,
            roof_result,
            output_path=str(out_dir / "pyramid_lines.png"),
            roof_angle_deg=config.get("roof_angle", 30.0),
            wall_height=300.0,
            upper_floor_roof_sections=upper_floor_roof_sections,
            upper_floor_footprint=upper_floor_footprint,
        )
    except Exception:
        pass

    try:
        visualize_a_frame_lines(
            wall_mask_path,
            roof_result,
            output_path=str(out_dir / "a_lines.png"),
            roof_angle_deg=config.get("roof_angle", 30.0),
            wall_height=300.0,
            upper_floor_roof_sections=upper_floor_roof_sections,
        )
    except Exception:
        pass

    # Nu scriem dreptunghiuri derivate din `remaining_mask` (nu au relevanță).

    wall_mask_img = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
    if wall_mask_img is None:
        return
    shape = wall_mask_img.shape

    sections = roof_result.get("sections", [])
    connections = roof_result.get("connections", [])

    rectangles = []
    ridge_lines = []
    for sec in sections:
        bounding_rect = sec.get("bounding_rect", [])
        if len(bounding_rect) >= 3:
            from shapely.geometry import Polygon as ShapelyPolygon

            rect_poly = ShapelyPolygon(bounding_rect)
            rectangles.append(rect_poly)
            ridge_line = sec.get("ridge_line", [])
            if len(ridge_line) >= 2:
                ridge_lines.append(
                    {
                        "start": ridge_line[0],
                        "end": ridge_line[1],
                        "is_main": sec.get("is_main", False),
                        "orientation": sec.get("ridge_orientation", "horizontal"),
                    }
                )

    if not rectangles or not ridge_lines:
        return

    binary_mask = generate_binary_mask(rectangles, shape)
    cv2.imwrite(str(out_dir / "roof_mask_binary.png"), binary_mask)
    sections_mask = generate_sections_mask(rectangles, shape)
    cv2.imwrite(str(out_dir / "roof_mask_sections.png"), sections_mask)
    gradient_mask = generate_gradient_mask(rectangles, ridge_lines, shape)
    cv2.imwrite(str(out_dir / "roof_mask_gradient.png"), gradient_mask)

    generate_texture_mask(rectangles, ridge_lines, connections, shape, output_path=str(out_dir / "roof_texture.png"))

    tex_standard, labels_standard = generate_roof_texture_standard(sections, connections, shape)
    cv2.imwrite(str(out_dir / "roof_texture_standard.png"), tex_standard)
    (out_dir / "roof_texture_standard_labels.txt").write_text(
        "# ID_pixel = nume_față\n" + "\n".join(f"{fid}\t{name}" for fid, name in labels_standard),
        encoding="utf-8",
    )

    tex_pyramid, labels_pyramid = generate_roof_texture_pyramid(sections, connections, shape)
    cv2.imwrite(str(out_dir / "roof_texture_pyramid.png"), tex_pyramid)
    (out_dir / "roof_texture_pyramid_labels.txt").write_text(
        "# ID_pixel = nume_față\n" + "\n".join(f"{fid}\t{name}" for fid, name in labels_pyramid),
        encoding="utf-8",
    )

    try:
        generate_roof_faces_unfolded_standard(sections, connections, config, out_dir / "roof_faces_standard")
        generate_roof_faces_unfolded_pyramid(sections, connections, config, out_dir / "roof_faces_pyramid")
        save_roof_overview_numbered(tex_standard, out_dir / "roof_faces_standard" / "acoperis_numerotat.png")
        save_roof_overview_numbered(tex_pyramid, out_dir / "roof_faces_pyramid" / "acoperis_numerotat.png")
    except Exception:
        pass

    generate_comparison(
        wall_mask_img,
        rectangles,
        ridge_lines,
        connections,
        output_path=str(out_dir / "roof_comparison.png"),
    )


def _largest_section_bbox(roof_data: dict) -> Optional[Tuple[int, int, int, int]]:
    """BBox-ul celui mai mare dreptunghi (după arie) din roof_data['sections']."""
    best = None
    best_area = -1
    for sec in roof_data.get("sections") or []:
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


def _compute_overlay_offsets(floor_paths: List[str], roof_results: List[dict]) -> dict:
    """
    Reproduce alinierea din `floors_overlay.png`: aliniază centrele celui mai mare dreptunghi per etaj.
    Returnează dict path -> (ox, oy) astfel încât (x + ox, y + oy) să fie în coordonatele overlay.
    """
    bboxes: List[Tuple[int, int, int, int]] = []
    for rr in roof_results:
        bb = _largest_section_bbox(rr)
        if bb is None:
            bb = _get_largest_rect_bbox(rr)
        if bb is None:
            bb = (0, 0, 0, 0)
        bboxes.append(bb)

    widths = [b[2] - b[0] for b in bboxes]
    heights = [b[3] - b[1] for b in bboxes]
    max_w = max(widths) if widths else 0
    max_h = max(heights) if heights else 0
    padding = 50
    ref_cx = (max_w + 2 * padding) / 2.0
    ref_cy = (max_h + 2 * padding) / 2.0

    out = {}
    for p, bb in zip(floor_paths, bboxes):
        cx, cy = _bbox_center(bb)
        ox = int(round(ref_cx - cx))
        oy = int(round(ref_cy - cy))
        out[p] = (ox, oy)
    return out


def _translate_sections(sections: List[dict], dx: float, dy: float) -> List[dict]:
    """Translatează bounding_rect + ridge_line cu (dx, dy)."""
    out: List[dict] = []
    for sec in sections:
        rect = sec.get("bounding_rect", [])
        ridge = sec.get("ridge_line", [])
        new_rect = [(float(p[0]) + dx, float(p[1]) + dy) for p in rect] if rect else []
        new_ridge = [(float(p[0]) + dx, float(p[1]) + dy) for p in ridge] if ridge and len(ridge) >= 2 else []
        out.append(
            {
                "bounding_rect": new_rect,
                "ridge_line": new_ridge,
                "ridge_orientation": sec.get("ridge_orientation", "horizontal"),
                "is_main": sec.get("is_main", False),
            }
        )
    return out


def detect_floors_from_folder(folder_path: str) -> Tuple[List[str], Optional[str]]:
    folder = Path(folder_path)
    if not folder.is_dir():
        return [], None
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    image_paths = [str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in image_extensions]
    if len(image_paths) == 0:
        return [], None
    if len(image_paths) == 1:
        return image_paths, image_paths[0]

    floor_sizes = []
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape[:2]
        area = h * w
        floor_sizes.append((img_path, area))
    if len(floor_sizes) == 0:
        return [], None

    floor_sizes.sort(key=lambda x: x[1])
    roof_floor = floor_sizes[0][0]
    all_paths = [x[0] for x in floor_sizes]
    return all_paths, roof_floor


def complete_workflow(wall_mask_path: str, output_dir: str = "output"):
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    print("=" * 60)
    print("🏠 ROOF ANALYSIS SYSTEM - Complete Workflow")
    print("=" * 60)

    input_path = Path(wall_mask_path)
    all_floor_paths: List[str] = []
    roof_floor_path: str = wall_mask_path

    if input_path.is_dir():
        print(f"\n📁 Detectat folder cu etaje: {wall_mask_path}")
        all_floor_paths, roof_floor_path = detect_floors_from_folder(wall_mask_path)
        if roof_floor_path is None:
            print("   ⚠ Nu s-au găsit imagini în folder!")
            return None
        print(f"   ✓ {len(all_floor_paths)} etaje detectate")
        print(f"   ✓ Etaj pentru acoperiș: {Path(roof_floor_path).name}")
    else:
        if not input_path.exists():
            print(f"   ⚠ Fișier nu există: {wall_mask_path}")
            return None
        all_floor_paths = [wall_mask_path]
        roof_floor_path = wall_mask_path

    print("\n[1/4] Analiză blueprint...")
    roof_result = calculate_roof_from_walls(roof_floor_path, roof_angle=30, overhang_px=10)
    sections = roof_result.get("sections", [])
    print(f"   ✓ {len(sections)} secțiuni detectate")
    print(f"   ✓ Suprafață totală: {roof_result.get('total_roof_area', 0):.2f} px²")

    print("\n[2/4] Calculul materialelor și prețurilor...")
    config = {
        "roof_angle": 30.0,
        "wall_height": 3.0,
        "meters_per_pixel": 0.01,
        "prices": {
            "tile_per_m2": 15.0,
            "ridge_tile_per_m": 8.0,
            "valley_sheet_per_m": 12.0,
            "labor_per_m2": 20.0,
            "waste_factor": 1.10,
        },
    }
    pricing = calculate_materials_and_price(roof_result, config)
    print(f"\n   📊 REZULTATE:")
    print(f"   ├─ Suprafață acoperiș: {pricing['materials']['roof_area_m2']:.2f} m²")
    print(f"   ├─ Lungime creste: {pricing['materials']['ridge_length_m']:.2f} m")
    print(f"   ├─ Lungime vale: {pricing['materials']['valley_length_m']:.2f} m")
    print(f"   └─ Cost total: {pricing['total']:.2f} EUR")

    print("\n[3/4] Generare vizualizări...")

    floor_roof_results: List[dict] = []
    if input_path.is_dir() and len(all_floor_paths) >= 1:
        for floor_path in all_floor_paths:
            try:
                floor_roof_results.append(calculate_roof_from_walls(floor_path, roof_angle=30, overhang_px=10))
            except Exception as e:
                print(f"   ⚠ ({Path(floor_path).name}): {e}")
                floor_roof_results = []
                break
        if len(floor_roof_results) != len(all_floor_paths):
            floor_roof_results = []
    else:
        # Single-floor input: still provide `floor_roof_results` so 3D renderers can run.
        floor_roof_results = [roof_result]

    floors_ordered: List[Tuple[str, Any]] = []
    if input_path.is_dir() and floor_roof_results:
        floors_ordered = _ordered_floor_polygons(all_floor_paths, roof_floor_path, floor_roof_results)
    if not floors_ordered and all_floor_paths:
        floors_ordered = [(p, None) for p in all_floor_paths]

    num_floors = len(floors_ordered)
    for floor_level in range(num_floors):
        (output_path / f"etaj_{floor_level}").mkdir(parents=True, exist_ok=True)
        (output_path / f"etaj_{floor_level}" / "rectangles_floor").mkdir(parents=True, exist_ok=True)
        # nu mai folosim `rectangles/`/`rectangles_remaining` ca input; rămâne doar `rectangles_floor/`

    if floor_roof_results and floors_ordered:
        path_to_result = {all_floor_paths[i]: floor_roof_results[i] for i in range(len(all_floor_paths))}
        for floor_level in range(len(floors_ordered)):
            path = floors_ordered[floor_level][0]
            res = path_to_result.get(path)
            if res is None:
                continue
            rect_dir = output_path / f"etaj_{floor_level}" / "rectangles_floor"
            etaj_paths = visualize_individual_rectangles(path, res, output_dir=str(rect_dir))
            print(f"   ✓ Etaj {floor_level}: {len(etaj_paths)} imagini în etaj_{floor_level}/rectangles_floor/")

    # Offsets identice cu `floors_overlay.png` (pentru a putea transpune etajele între ele)
    offsets_by_path = {}
    if input_path.is_dir() and floor_roof_results and len(floor_roof_results) == len(all_floor_paths):
        offsets_by_path = _compute_overlay_offsets(all_floor_paths, floor_roof_results)

    main_roof_orientation: Optional[str] = None
    for sec in roof_result.get("sections", []):
        if sec.get("is_main"):
            main_roof_orientation = sec.get("ridge_orientation", "horizontal")
            break

    roof_levels: List[Tuple[float, dict, float, float, int]] = []
    if input_path.is_dir() and len(all_floor_paths) >= 2 and floor_roof_results:
        roof_levels = _compute_lower_floor_roofs(
            all_floor_paths,
            floor_roof_results,
            roof_floor_path=roof_floor_path,
            wall_height=300.0,
            output_base=output_path,
            main_roof_orientation=main_roof_orientation,
        )

    # Generează liniile (a_lines/pyramid_lines) pe baza `rectangles_floor` pentru fiecare etaj.
    # Asta înseamnă: folosim `floor_roof_results[floor]` ca roof_result pentru etaj,
    # iar etajele superioare sunt transpuse în coordonatele etajului curent folosind offset-urile overlay.
    if input_path.is_dir() and floor_roof_results and floors_ordered:
        path_to_result = {all_floor_paths[i]: floor_roof_results[i] for i in range(len(all_floor_paths))}
        for floor_level in range(len(floors_ordered)):
            floor_path = floors_ordered[floor_level][0]
            rr = path_to_result.get(floor_path)
            if rr is None:
                continue

            # Upper floors (translated into current floor coords)
            upper_sections: List[dict] = []
            upper_footprint: List[List[Tuple[float, float]]] = []
            ox_cur, oy_cur = offsets_by_path.get(floor_path, (0, 0))

            for up_level in range(floor_level + 1, len(floors_ordered)):
                up_path = floors_ordered[up_level][0]
                rr_up = path_to_result.get(up_path)
                if rr_up is None:
                    continue
                ox_up, oy_up = offsets_by_path.get(up_path, (0, 0))
                dx = float(ox_up - ox_cur)
                dy = float(oy_up - oy_cur)
                # translate sections from upper floor into current floor coords
                upper_sections.extend(_translate_sections(rr_up.get("sections", []) or [], dx, dy))

                # translate footprint polygon too (for drawing)
                img_up = cv2.imread(up_path, cv2.IMREAD_GRAYSCALE)
                if img_up is not None:
                    from roof_calc.flood_fill import flood_fill_interior, get_house_shape_mask
                    from roof_calc.geometry import extract_polygon

                    filled_up = flood_fill_interior(img_up)
                    mask_up = get_house_shape_mask(filled_up)
                    poly_up = extract_polygon(mask_up)
                    if poly_up is not None and not poly_up.is_empty and hasattr(poly_up, "exterior"):
                        coords = [(float(x) + dx, float(y) + dy) for (x, y) in list(poly_up.exterior.coords)]
                        upper_footprint.append(coords)

            etaj_dir = output_path / f"etaj_{floor_level}"
            _generate_roof_outputs(
                floor_path,
                rr,
                etaj_dir,
                config,
                upper_floor_roof_sections=upper_sections if upper_sections else None,
                upper_floor_footprint=upper_footprint if upper_footprint else None,
            )
        roof_floor_level = next((k for k in range(len(floors_ordered)) if floors_ordered[k][0] == roof_floor_path), 0)
        print(f"   ✓ Linii acoperiș per etaj (din rectangles_floor): etaj_0..etaj_{len(floors_ordered)-1}/ (pyramid_lines, a_lines)")
    else:
        roof_floor_level = next((k for k in range(len(floors_ordered)) if floors_ordered[k][0] == roof_floor_path), 0) if floors_ordered else 0
        etaj_roof_dir = output_path / f"etaj_{roof_floor_level}"
        _generate_roof_outputs(roof_floor_path, roof_result, etaj_roof_dir, config)
        print(f"   ✓ Acoperiș principal: etaj_{roof_floor_level}/ (rectangles, roof_lines, pyramid_lines, a_lines, măști, texturi, fețe desfășurate)")

    all_floors_for_3d = all_floor_paths
    _cfg_3d = {
        "roof_angle": 30.0,
        "wall_height": 300,
        "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1"],
        "views": [{"elev": 30, "azim": 45, "title": "Vedere Sud-Est"}, {"elev": 20, "azim": 225, "title": "Vedere Nord-Vest"}],
        # Plotly-only nice defaults
        "show_outlines": True,
        "outline_color": "rgba(0,0,0,0.65)",
        "outline_width": 3,
    }
    # Prefer Plotly (export PNG robust, z-order corect). Matplotlib rămâne fallback.
    ok_std = False
    try:
        ok_std = visualize_3d_standard_plotly(
            str(output_path / "house_3d.png"),
            config=_cfg_3d,
            all_floor_paths=all_floors_for_3d,
            floor_roof_results=floor_roof_results if floor_roof_results else None,
        )
    except Exception as e:
        print(f"   ⚠ Plotly 3D (standard) a eșuat: {e}")
        ok_std = False
    if ok_std:
        print("   ✓ Vizualizare 3D (Plotly): house_3d.png")
    else:
        try:
            ok_std = visualize_3d_standard_matplotlib(
                str(output_path / "house_3d.png"),
                config=_cfg_3d,
                all_floor_paths=all_floors_for_3d,
                floor_roof_results=floor_roof_results if floor_roof_results else None,
            )
            if ok_std:
                print("   ✓ Vizualizare 3D (Matplotlib): house_3d.png")
        except Exception as e:
            print(f"   ⚠ Matplotlib 3D (standard) a eșuat: {e}")

    ok_pyr = False
    try:
        ok_pyr = visualize_3d_pyramid_plotly(
            roof_floor_path,  # wall_mask (path) - doar pentru fallback single-floor
            roof_result,
            output_path=str(output_path / "house_3d_pyramid.png"),
            config=_cfg_3d,
            all_floor_paths=all_floors_for_3d,
            floor_roof_results=floor_roof_results if floor_roof_results else None,
        )
    except Exception as e:
        print(f"   ⚠ Plotly 3D (piramidă) a eșuat: {e}")
        ok_pyr = False
    if ok_pyr:
        print("   ✓ Vizualizare 3D piramidă (Plotly): house_3d_pyramid.png")
    else:
        try:
            ok_pyr = visualize_3d_pyramid_matplotlib(
                str(output_path / "house_3d_pyramid.png"),
                config=_cfg_3d,
                all_floor_paths=all_floors_for_3d,
                floor_roof_results=floor_roof_results if floor_roof_results else None,
            )
            if ok_pyr:
                print("   ✓ Vizualizare 3D piramidă (Matplotlib): house_3d_pyramid.png")
        except Exception as e:
            print(f"   ⚠ Matplotlib 3D (piramidă) a eșuat: {e}")

    if input_path.is_dir() and len(all_floor_paths) >= 1:
        _scripts_dir = str(ROOT / "scripts")
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        try:
            from overlay_floors_2d import overlay_floors_2d

            rectangle_folders = [
                str(output_path / f"etaj_{next((k for k in range(len(floors_ordered)) if floors_ordered[k][0] == p), 0)}" / "rectangles_floor")
                for p in all_floor_paths
            ]
            overlay_floors_2d(
                all_floor_paths,
                str(output_path / "floors_overlay.png"),
                alignment="center",
                roof_results=floor_roof_results if floor_roof_results else None,
                rectangle_folders=rectangle_folders if floor_roof_results else None,
            )
            print("   ✓ Suprapunere etaje 2D (din rectangle_Sx): floors_overlay.png")
        except Exception as e:
            print(f"   ⚠ Suprapunere etaje 2D: {e}")
        try:
            from overlay_floors_2d_debug import generate_overlay_debug_image

            generate_overlay_debug_image(
                all_floor_paths,
                str(output_path / "floors_overlay_debug.png"),
                roof_results=floor_roof_results if floor_roof_results else None,
            )
            print("   ✓ Debug suprapunere (dreptunghiuri): floors_overlay_debug.png")
        except Exception as e:
            print(f"   ⚠ Debug suprapunere: {e}")

    print("\n[4/4] Generare raport...")
    report_path = output_path / "report.txt"
    report_path.write_text("OK\n", encoding="utf-8")
    print(f"   ✓ Raport salvat: {report_path}")
    print("\n" + "=" * 60)
    print("✅ Workflow COMPLET!")
    print("=" * 60)

    return {"result": roof_result, "pricing": pricing, "output_dir": output_path}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Utilizare: python complete_workflow.py <imagine.png SAU folder/> [output_dir]")
        sys.exit(1)
    wall_mask_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    complete_workflow(wall_mask_path, output_dir)
    print("\n🎉 Gata! Verifică folder-ul de output pentru rezultate.")

