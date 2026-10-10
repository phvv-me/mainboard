"""Imported by every interpreter a gate starts: the Python guard, then the offline catalog."""

import importlib
import os

try:
    importlib.import_module("guard").Guard.from_environment().install()
except Exception as error:
    # Python prints and ignores a failing sitecustomize, which would run an unguarded worker.
    raise SystemExit(f"M00 bootstrap refused: {error}") from error

if "MB_GATE_CATALOG" in os.environ:
    importlib.import_module("catalog_double")
