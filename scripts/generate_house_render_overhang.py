#!/usr/bin/env python3
"""
Generează doar house_render_overhang.png și house_render_overhang.html.
Rulează: python scripts/generate_house_render_overhang.py test3/ output/
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/generate_house_render_overhang.py <input_dir> <output_dir>")
        print("Example: python scripts/generate_house_render_overhang.py test3/ output/")
        sys.exit(1)
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    from roof_calc.visualize_3d_plotly import visualize_3d_house_render_plotly
    from roof_calc.algorithm import calculate_roof_from_walls
    import cv2

    mask_paths = sorted(input_dir.glob("*.png"))
    if not mask_paths:
        print("No PNG files in", input_dir)
        sys.exit(1)

    all_floors = [str(p) for p in mask_paths]
    floor_results = []
    for p in all_floors:
        rr = calculate_roof_from_walls(p)
        floor_results.append(rr)

    cfg = {"roof_angle": 30.0, "wall_height": 300.0, "overhang_px": 0}
    try:
        from roof_calc.overhang import compute_overhang_px_from_roof_results
        cfg["overhang_px"] = compute_overhang_px_from_roof_results(floor_results, ratio=0.10)
    except Exception:
        cfg["overhang_px"] = 40.0

    ok = visualize_3d_house_render_plotly(
        str(output_dir / "house_render_overhang.png"),
        config=cfg,
        all_floor_paths=all_floors,
        floor_roof_results=floor_results,
        html_output_path=str(output_dir / "house_render_overhang.html"),
        extend_segments_mode=True,
    )
    if ok:
        print("✓ Generated house_render_overhang.png and house_render_overhang.html")
    else:
        print("✗ Failed to generate house_render_overhang")
        sys.exit(1)

if __name__ == "__main__":
    main()
