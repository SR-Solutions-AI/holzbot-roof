"""Debug: afișează segmentele și ridge-urile folosite pentru a_faces."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_calc.flood_fill import flood_fill_interior, get_house_shape_mask
from roof_calc.geometry import extract_polygon
from roof_calc.overhang import extend_secondary_sections_to_main_ridge, ridge_intersection_corner_lines
from roof_calc.roof_segments_3d import (
    get_roof_segments_3d,
    _deduplicate_segments,
    _subdivide_segments_at_intersections,
    segments_to_faces,
)


def main():
    import cv2
    from roof_calc import calculate_roof_from_walls

    mask_path = ROOT / "test3" / "01_walls_from_coords.png"
    if not mask_path.exists():
        mask_path = ROOT / "output" / "etaj_0" / "roof_mask_binary.png"
    if not mask_path.exists():
        print("Nu există mască. Rulează workflow cu test3/ sau output/.")
        return

    roof_result = calculate_roof_from_walls(str(mask_path), roof_angle=30, overhang_px=10)
    sections = roof_result.get("sections") or []
    sections = extend_secondary_sections_to_main_ridge(sections)

    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    filled = flood_fill_interior(img)
    house_mask = get_house_shape_mask(filled)
    floor_poly = extract_polygon(house_mask)
    if floor_poly is None or getattr(floor_poly, "is_empty", True):
        print("Nu există floor_polygon")
        return

    cl = ridge_intersection_corner_lines(sections, floor_polygon=floor_poly)

    print(f"\n=== SECȚIUNI ({len(sections)}) ===")
    for i, sec in enumerate(sections):
        ridge = sec.get("ridge_line") or []
        orient = sec.get("ridge_orientation", "?")
        is_main = sec.get("is_main", False)
        print(f"  sec {i}: orient={orient} is_main={is_main} ridge_pts={len(ridge)}")
        if len(ridge) >= 2:
            print(f"    ridge: ({ridge[0][0]:.1f},{ridge[0][1]:.1f}) -> ({ridge[1][0]:.1f},{ridge[1][1]:.1f})")

    raw = get_roof_segments_3d(
        sections, floor_poly, wall_height=300, roof_angle_deg=30,
        corner_lines=cl, ridge_magenta_contour_only=True,
    )
    z1 = 300
    ridge_segs = [(p1, p2) for p1, p2 in raw if float(p1[2]) > z1 and float(p2[2]) > z1]
    print(f"\n=== SEGMENTE RAW: {len(raw)} total, ~{len(ridge_segs)} ridge (z>300) ===")

    dedup = _deduplicate_segments(raw, tol=0.01)
    ridge_after_dedup = [(p1, p2) for p1, p2 in dedup if float(p1[2]) > z1 and float(p2[2]) > z1]
    print(f"=== DUPĂ DEDUP: {len(dedup)} total, ~{len(ridge_after_dedup)} ridge ===")

    subdiv = _subdivide_segments_at_intersections(dedup)
    ridge_after_subdiv = [(p1, p2) for p1, p2 in subdiv if float(p1[2]) > z1 and float(p2[2]) > z1]
    print(f"=== DUPĂ SUBDIVIZARE: {len(subdiv)} total, ~{len(ridge_after_subdiv)} ridge ===")

    faces = segments_to_faces(subdiv)
    print(f"\n=== FEȚE: {len(faces)} ===")


if __name__ == "__main__":
    main()
