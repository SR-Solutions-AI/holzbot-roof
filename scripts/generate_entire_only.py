#!/usr/bin/env python3
"""Generează doar entire/{1_w,2_w,4_w,4.5_w}/frame.html și filled.html din roof_types existent."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from roof_calc.roof_types_workflow import generate_entire_frame_html

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "output_clean2"
    roof_types_dir = Path(out) / "roof_types"
    if not roof_types_dir.exists():
        print(f"Lipsește: {roof_types_dir}")
        sys.exit(1)
    for rt in ("1_w", "2_w", "4_w", "4.5_w"):
        try:
            generate_entire_frame_html(roof_types_dir, Path(out), wall_height=300.0, roof_type=rt)
            print(f"✓ entire/{rt}/frame.html, filled.html")
        except Exception as e:
            print(f"✗ entire/{rt}/: {e}")
            import traceback
            traceback.print_exc()
