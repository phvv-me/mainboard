"""The constants of `analysis` other modules share."""

LANGUAGES = {
    "PY": "python",
    "RS": "rust",
    "TS": "typescript",
    "C": "c",
    "CPP": "cpp",
    "CU": "cuda",
    "SW": "swift",
}

# The SQL naming what a finding declares, for each rule that calls a name dead, a unit redundant
# or a constant a private file detail: the name a consumer would import (a module's stem for the
# rules in MODULES).
# A finding about a route (an `__all__` entry, an unread import) is used only by an import from
# that very module (EXPORTS); a definition is also used through any package re-exporting it.
DECLARED = {
    "ALL-REAC0001": "regexp_extract(qualname, '[^.]+$')",
    "ALL-REAC0002": "regexp_extract(qualname, '[^.]+$')",
    "PY-MODU0005": "public_export_name",
    "PY-CONS0001": "name",
    "PY-DEAD0001": "name",
    "PY-RUFF0001": "declared",
    "PY-RUFF0002": "declared",
    "ALL-FUNC0002": "name",
    "ALL-FUNC0003": "name",
    "ALL-FUNC0006": "name",
    "ALL-ARCH0005": "regexp_extract(path, '([^/]+?)(/__init__)?\\.pyi?$', 1)",
}

# The applicabilities a fix run applies, safe ones alone or those marked for review too.
FIXABLE = {"safe": ("safe",), "review": ("safe", "review")}
