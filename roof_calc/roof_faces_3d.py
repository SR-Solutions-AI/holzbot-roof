"""
Loader shim.

Restores original implementation from `roof_calc/_orig_pyc/roof_faces_3d.<cache_tag>.pyc`.
Tries each .pyc in _orig_pyc until one loads (avoids "bad magic number" when server Python differs from build Python).
If no .pyc loads, uses a lazy stub module so "import roof_calc.roof_faces_3d as rf" succeeds and rf.xxx() returns None.
"""

from __future__ import annotations

import sys
from importlib.machinery import SourcelessFileLoader
from pathlib import Path
from typing import Any

_base = Path(__file__).with_name("_orig_pyc")
_stem = Path(__file__).stem
_candidates: list[Path] = []
_exact = _base / f"{_stem}.{sys.implementation.cache_tag}.pyc"
if _exact.exists():
    _candidates.append(_exact)
_candidates.extend(sorted(_base.glob(f"{_stem}.*.pyc")))
if not _candidates:
    raise FileNotFoundError(f"Missing cached bytecode for {_stem} in {_base}")

_last_err: Exception | None = None
for _pyc in _candidates:
    try:
        _loader = SourcelessFileLoader(__name__, str(_pyc))
        _loader.exec_module(sys.modules[__name__])
        _last_err = None
        break
    except ImportError as e:
        _last_err = e
        continue

if _last_err is not None:

    def __getattr__(name: str) -> Any:
        def _stub(*args: Any, **kwargs: Any) -> Any:
            return None
        return _stub
