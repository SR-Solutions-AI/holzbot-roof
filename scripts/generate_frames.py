#!/usr/bin/env python3
"""
Generează frame.html pentru folderele de tip acoperiș existente (1_w, 2_w, 4_w, 4.5_w).
Util pentru output-uri create înainte de adăugarea acestei funcționalități.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from roof_calc.roof_types_workflow import generate_frames_for_roof_types_dir

if __name__ == "__main__":
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output_clean3" / "roof_types" / "floor_0"
    wh = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    print(f"Generând frame.html în {base} (wall_height={wh})...")
    generate_frames_for_roof_types_dir(base, wall_height=wh)
    for name in ("1_w", "2_w", "4_w", "4.5_w"):
        subdir = base / name
        if (subdir / "frame.html").exists():
            print(f"  ✓ {subdir.relative_to(ROOT)}/frame.html")
