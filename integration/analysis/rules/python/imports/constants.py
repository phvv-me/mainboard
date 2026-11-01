"""The packages PY-IMPO0002 is tried on, as they stand and as its fix leaves them. `CASE` stands
for the name of the folder holding a case, so that a case can spell its package in full."""

from .builders import Case, code

IMPORTED = code("from .a import _hidden", "", "V = _hidden")
IMPORTED_FIXED = code("from .a import hidden", "", "V = hidden")
TOOL = code("def tool():", "    return 1")
RENAMES = {
    "a-name-read-by-an-importer": Case(
        {
            "a.py": code("_hidden = 1", "", "", "def use():", "    return _hidden"),
            "b.py": IMPORTED,
        },
        {
            "a.py": code("hidden = 1", "", "", "def use():", "    return hidden"),
            "b.py": IMPORTED_FIXED,
        },
    ),
    "a-name-in-all-imported-by-two-files-and-read-through-the-module": Case(
        {
            "a.py": code("_hidden = 1", "", '__all__ = ["_hidden"]'),
            "b.py": IMPORTED,
            "c.py": code("from .a import _hidden as h", "", "W = h"),
            "d.py": code("from . import a", "", "X = a._hidden"),
            "e.py": code("import pkg.CASE.a", "", "Y = pkg.CASE.a._hidden"),
        },
        {
            "a.py": code("hidden = 1", "", '__all__ = ["hidden"]'),
            "b.py": IMPORTED_FIXED,
            "c.py": code("from .a import hidden as h", "", "W = h"),
            "d.py": code("from . import a", "", "X = a.hidden"),
            "e.py": code("import pkg.CASE.a", "", "Y = pkg.CASE.a.hidden"),
        },
    ),
    "a-name-of-a-package-initializer": Case(
        {
            "__init__.py": code("_hidden = 1"),
            "b.py": code("from . import _hidden", "", "V = _hidden"),
        },
        {
            "__init__.py": code("hidden = 1"),
            "b.py": code("from . import hidden", "", "V = hidden"),
        },
    ),
    "a-class": Case(
        {
            "a.py": code("class _Thing:", "    pass"),
            "b.py": code("from .a import _Thing", "", "V = _Thing()"),
        },
        {
            "a.py": code("class Thing:", "    pass"),
            "b.py": code("from .a import Thing", "", "V = Thing()"),
        },
    ),
    "a-name-of-a-nonpublic-module": Case(
        {
            "_tools.py": code("def _helper():", "    return 1"),
            "b.py": code("from ._tools import _helper", "", "V = _helper()"),
        },
        {
            "tools.py": code("def helper():", "    return 1"),
            "b.py": code("from .tools import helper", "", "V = helper()"),
        },
    ),
}
MOVES = {
    "a-module-imported-by-name": Case(
        {"_tools.py": TOOL, "b.py": code("from ._tools import tool", "", "V = tool()")},
        {"tools.py": TOOL, "b.py": code("from .tools import tool", "", "V = tool()")},
    ),
    "a-module-with-a-stub-read-by-a-plain-import": Case(
        {
            "_tools.py": TOOL,
            "_tools.pyi": code("def tool() -> int: ..."),
            "b.py": code("import pkg.CASE._tools", "", "V = pkg.CASE._tools.tool()"),
        },
        {
            "tools.py": TOOL,
            "tools.pyi": code("def tool() -> int: ..."),
            "b.py": code("import pkg.CASE.tools", "", "V = pkg.CASE.tools.tool()"),
        },
    ),
    "a-module-imported-as-a-name-of-its-package": Case(
        {"_tools.py": TOOL, "b.py": code("from . import _tools", "", "V = _tools.tool()")},
        {"tools.py": TOOL, "b.py": code("from . import tools as _tools", "", "V = _tools.tool()")},
    ),
}
REFUSALS = {
    "a-new-name-the-importer-binds": {
        "a.py": code("_hidden = 1"),
        "b.py": code("from .a import _hidden", "", "hidden = _hidden"),
    },
    "a-new-name-the-definer-binds": {
        "a.py": code("_hidden = 1", "hidden = 2"),
        "b.py": IMPORTED,
    },
    "a-new-name-that-is-a-keyword": {
        "a.py": code("_class = 1"),
        "b.py": code("from .a import _class", "", "V = _class"),
    },
    "a-string-spelling-the-name": {
        "a.py": code("_hidden = 1", 'NAMES = ["_hidden"]'),
        "b.py": IMPORTED,
    },
    "a-string-spelling-the-name-in-a-file-that-imports-the-module": {
        "a.py": code("_hidden = 1"),
        "b.py": IMPORTED,
        "c.py": code("from . import a", "", 'V = getattr(a, "_hidden")'),
    },
    "a-string-path-ending-in-the-name-in-a-file-that-never-imports-the-module": {
        "a.py": code("_hidden = 1"),
        "b.py": IMPORTED,
        "test_b.py": code("from unittest import mock", "", 'mock.patch("pkg.CASE.a._hidden", 2)'),
    },
    "an-f-string-field-printing-the-name": {
        "a.py": code("_hidden = 1"),
        "b.py": code("from .a import _hidden", "", 'V = f"{_hidden=}"'),
    },
    "a-star-import-of-the-module": {
        "a.py": code("_hidden = 1"),
        "b.py": IMPORTED,
        "c.py": code("from .a import *"),
    },
    "a-file-that-is-no-python-naming-the-name": {
        "a.py": code("_hidden = 1"),
        "b.py": IMPORTED,
        "notes.md": code("The entry point is pkg.CASE.a:_hidden."),
    },
    "a-module-no-file-defines": {"b.py": code("from .missing import _hidden", "", "V = _hidden")},
    "a-name-the-module-does-not-define": {
        "a.py": code("OTHER = 1"),
        "b.py": code("from .a import _missing"),
    },
    "a-module-two-files-define": {
        "a.py": code("_hidden = 1"),
        "a/__init__.py": code("_hidden = 2"),
        "b.py": IMPORTED,
    },
    "a-stub-beside-the-definer": {
        "a.py": code("_hidden = 1"),
        "a.pyi": code("_hidden: int"),
        "b.py": IMPORTED,
    },
    "a-module-where-the-new-one-would-stand": {
        "_tools.py": TOOL,
        "tools.py": code("OTHER = 1"),
        "b.py": code("from ._tools import tool", "", "V = tool()"),
    },
    "a-file-that-is-no-python-naming-the-module": {
        "_tools.py": TOOL,
        "b.py": code("from ._tools import tool", "", "V = tool()"),
        "notes.md": code("See pkg.CASE._tools for the tool."),
    },
    "a-nonpublic-package": {
        "_inner/__init__.py": code("Z = 2"),
        "_inner/x.py": code("Y = 1"),
        "b.py": code("from ._inner.x import Y", "from ._inner import Z", "", "V = Y + Z"),
    },
    "a-name-the-defining-module-re-exports": {
        "z.py": code("_hidden = 1"),
        "a.py": code("from .z import _hidden", "", '__all__ = ["_hidden"]'),
        "b.py": IMPORTED,
    },
}
