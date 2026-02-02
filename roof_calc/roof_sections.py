"""
Loader shim.

Restores original implementation from `roof_calc/_orig_pyc/roof_sections.<cache_tag>.pyc`.
"""

from __future__ import annotations

import sys
from importlib.machinery import SourcelessFileLoader
from pathlib import Path

_base = Path(__file__).with_name("_orig_pyc")
_stem = Path(__file__).stem
_pyc = _base / f"{_stem}.{sys.implementation.cache_tag}.pyc"
if not _pyc.exists():
    matches = sorted(_base.glob(f"{_stem}.*.pyc"))
    if not matches:
        raise FileNotFoundError(f"Missing cached bytecode for {_stem} in {_base}")
    _pyc = matches[0]

SourcelessFileLoader(__name__, str(_pyc)).exec_module(sys.modules[__name__])

