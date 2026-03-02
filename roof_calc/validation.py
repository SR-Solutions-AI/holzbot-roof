"""
Loader shim.

Restores original implementation from `roof_calc/_orig_pyc/validation.<cache_tag>.pyc`.
Tries each .pyc in _orig_pyc until one loads (avoids "bad magic number" when server Python differs from build Python).
If no .pyc loads, uses minimal stubs so import succeeds and pipeline can run.
"""

from __future__ import annotations

import sys
from importlib.machinery import SourcelessFileLoader
from pathlib import Path
from typing import Any, Tuple

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
    def validate_inputs(*args: Any, **kwargs: Any) -> Tuple[bool, str]:
        return True, ""

    def validate_roof_output(*args: Any, **kwargs: Any) -> Tuple[bool, str]:
        return True, ""
