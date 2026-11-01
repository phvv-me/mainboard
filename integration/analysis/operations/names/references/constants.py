from ..builders import Case, code

DEFINED = "_hidden = 1\n"
RENAMED = "hidden = 1\n"
IMPORT = "from .a import _hidden"
RENAMES = {
    "a-variable-and-the-functions-reading-it": Case(
        {"a.py": code("_hidden = 1", "", "", "def use():", "    return _hidden")},
        {"a.py": code("hidden = 1", "", "", "def use():", "    return hidden")},
    ),
    "a-function": Case(
        {"a.py": code("def _hidden():", "    return 1", "", "", "V = _hidden()")},
        {"a.py": code("def hidden():", "    return 1", "", "", "V = hidden()")},
    ),
    "a-class": Case(
        {"a.py": code("class _hidden:", "    pass", "", "", "V = _hidden()")},
        {"a.py": code("class hidden:", "    pass", "", "", "V = hidden()")},
    ),
    "a-decorated-async-function": Case(
        {"a.py": code("@decorator", "async def  _hidden():", "    return 1")},
        {"a.py": code("@decorator", "async def  hidden():", "    return 1")},
    ),
    "a-definition-under-a-try": Case(
        {"a.py": code("try:", "    _hidden = 1", "except ImportError:", "    _hidden = 2")},
        {"a.py": code("try:", "    hidden = 1", "except ImportError:", "    hidden = 2")},
    ),
    "the-strings-of-all": Case(
        {
            "a.py": code(
                "_hidden = 1", "__all__ = ['_hidden', \"other\"]", '__all__ += ("_hidden",)'
            )
        },
        {"a.py": code("hidden = 1", "__all__ = ['hidden', \"other\"]", '__all__ += ("hidden",)')},
    ),
    "comments-and-strings-that-only-mention-it": Case(
        {"a.py": code("_hidden = 1  # the _hidden one", 'NOTE = "about _hidden"')},
        {"a.py": code("hidden = 1  # the _hidden one", 'NOTE = "about _hidden"')},
    ),
    "an-f-string-and-a-comprehension-reading-it": Case(
        {"a.py": code("_hidden = 3", 'V = f"{_hidden}" + str([_hidden for _ in ()])')},
        {"a.py": code("hidden = 3", 'V = f"{hidden}" + str([hidden for _ in ()])')},
    ),
    "an-importer-and-its-reads": Case(
        {"a.py": DEFINED, "b.py": code(IMPORT, "", "V = _hidden + 1")},
        {"a.py": RENAMED, "b.py": code("from .a import hidden", "", "V = hidden + 1")},
    ),
    "an-importer-among-other-names-over-lines": Case(
        {
            "a.py": code("_hidden = 1", "other = 2"),
            "b.py": code("from .a import (", "    other,", "    _hidden,", ")", "", "V = _hidden"),
        },
        {
            "a.py": code("hidden = 1", "other = 2"),
            "b.py": code("from .a import (", "    other,", "    hidden,", ")", "", "V = hidden"),
        },
    ),
    "an-importer-under-an-alias": Case(
        {"a.py": DEFINED, "b.py": code("from .a import _hidden as h", "", "V = h")},
        {"a.py": RENAMED, "b.py": code("from .a import hidden as h", "", "V = h")},
    ),
    "an-importer-re-exporting-it": Case(
        {"a.py": DEFINED, "b.py": code(IMPORT, "", '__all__ = ["_hidden"]')},
        {"a.py": RENAMED, "b.py": code("from .a import hidden", "", '__all__ = ["hidden"]')},
    ),
    "a-local-of-the-same-name-in-an-importer": Case(
        {
            "a.py": DEFINED,
            "b.py": code(IMPORT, "", "", "def f():", "    _hidden = 3", "    return _hidden")
            + code("", "", "V = _hidden"),
        },
        {
            "a.py": RENAMED,
            "b.py": code("from .a import hidden", "", "", "def f():", "    _hidden = 3")
            + code("    return _hidden", "", "", "V = hidden"),
        },
    ),
    "a-global-read-in-an-importer-function": Case(
        {"a.py": DEFINED, "b.py": code(IMPORT, "", "", "def g():", "    return _hidden")},
        {
            "a.py": RENAMED,
            "b.py": code("from .a import hidden", "", "", "def g():", "    return hidden"),
        },
    ),
    "an-attribute-of-another-object": Case(
        {
            "a.py": code("_hidden = 1", "V = obj._hidden"),
            "b.py": code(IMPORT, "V = obj._hidden + _hidden"),
        },
        {
            "a.py": code("hidden = 1", "V = obj._hidden"),
            "b.py": code("from .a import hidden", "V = obj._hidden + hidden"),
        },
    ),
    "a-read-through-the-module": Case(
        {"a.py": DEFINED, "b.py": code("from . import a", "", "V = a._hidden")},
        {"a.py": RENAMED, "b.py": code("from . import a", "", "V = a.hidden")},
    ),
    "a-read-spelling-the-module-in-full": Case(
        {"a.py": DEFINED, "b.py": code("import pkg.CASE.a", "", "V = pkg.CASE.a._hidden")},
        {"a.py": RENAMED, "b.py": code("import pkg.CASE.a", "", "V = pkg.CASE.a.hidden")},
    ),
    "an-importer-in-a-subpackage": Case(
        {
            "a.py": DEFINED,
            "sub/__init__.py": "",
            "sub/b.py": code("from ..a import _hidden", "", "V = _hidden"),
        },
        {"a.py": RENAMED, "sub/b.py": code("from ..a import hidden", "", "V = hidden")},
    ),
    "a-top-level-module-of-the-same-name": Case(
        {"a.py": DEFINED, "b.py": code("import a", "", "V = a._hidden")}, {"a.py": RENAMED}
    ),
    "every-kind-of-reader-at-once": Case(
        {
            "a.py": code("_hidden = 1", "__all__ = ['_hidden']"),
            "b.py": code(IMPORT, "", "V = _hidden"),
            "c.py": code("from . import a", "", "V = a._hidden"),
            "d.py": code("import pkg.CASE.a", "", "V = pkg.CASE.a._hidden"),
        },
        {
            "a.py": code("hidden = 1", "__all__ = ['hidden']"),
            "b.py": code("from .a import hidden", "", "V = hidden"),
            "c.py": code("from . import a", "", "V = a.hidden"),
            "d.py": code("import pkg.CASE.a", "", "V = pkg.CASE.a.hidden"),
        },
    ),
}
REFUSALS = {
    "a-new-name-bound-in-the-module": Case(
        {"a.py": code("_hidden = 1", "hidden = 2")}, {}, "already a name"
    ),
    "a-new-name-read-as-a-global-or-builtin": Case(
        {"a.py": code("_hidden = 1", "", "", "def f():", "    return hidden()")},
        {},
        "already a name",
    ),
    "a-new-name-bound-by-an-importer": Case(
        {"a.py": DEFINED, "b.py": code(IMPORT, "hidden = 3")}, {}, "already a name in `src/pkg"
    ),
    "a-new-name-taken-in-an-importer-function": Case(
        {
            "a.py": DEFINED,
            "b.py": code(IMPORT, "", "", "def f():", "    hidden = 1", "    return hidden"),
        },
        {},
        "already a name in `src/pkg",
    ),
    "an-alias-spelling-the-old-name": Case(
        {"a.py": DEFINED, "b.py": code("from .a import _hidden as _hidden", "", "V = _hidden")},
        {},
        "spells `_hidden`",
    ),
    "a-star-import-of-the-module": Case(
        {"a.py": DEFINED, "b.py": "from .a import *\n"}, {}, "or is starred"
    ),
    "a-string-spelling-the-name": Case(
        {"a.py": code("_hidden = 1", 'V = getattr(x, "_hidden")')}, {}, "a token spells `_hidden`"
    ),
    "a-string-path-ending-in-the-name": Case(
        {"a.py": DEFINED, "b.py": code(IMPORT, 'PATCH = "pkg.CASE.a._hidden"')},
        {},
        "spells `_hidden`",
    ),
    "an-f-string-field-printing-the-name-where-it-is-defined": Case(
        {"a.py": code("_hidden = 1", 'V = f"{_hidden!r} {_hidden=}"')},
        {},
        "a token spells `_hidden`",
    ),
    "an-f-string-field-printing-the-name-in-an-importer": Case(
        {"a.py": DEFINED, "b.py": code(IMPORT, 'V = f"{_hidden + 1=}"')},
        {},
        "spells `_hidden`",
    ),
    "an-import-inside-a-function": Case(
        {
            "a.py": DEFINED,
            "b.py": code("def f():", "    from .a import _hidden", "    return _hidden"),
        },
        {},
        "inside a function or class",
    ),
    "a-global-statement": Case(
        {"a.py": code("_hidden = 1", "", "", "def f():", "    global _hidden", "    _hidden = 2")},
        {},
        "a token spells `_hidden`",
    ),
    "a-class-binding-the-name-too": Case(
        {"a.py": code("_hidden = 1", "", "", "class C:", "    _hidden = 2")},
        {},
        "no plain module-level binding",
    ),
    "a-name-the-module-never-binds": Case(
        {"a.py": "other = 1\n", "b.py": code(IMPORT)}, {}, "no plain module-level binding"
    ),
    "a-module-reading-its-names-by-spelling": Case(
        {"a.py": code("_hidden = 1", 'V = eval("1")')}, {}, "by their spelling"
    ),
    "an-importer-binding-the-old-name-in-a-class": Case(
        {"a.py": DEFINED, "b.py": code(IMPORT, "", "", "class C:", "    _hidden = 1")},
        {},
        "cannot tell apart",
    ),
    "an-importer-spelling-the-name-as-an-except-target": Case(
        {
            "a.py": DEFINED,
            "b.py": code(IMPORT, "", "", "def f():", "    try:", "        pass")
            + code("    except OSError as _hidden:", "        pass"),
        },
        {},
        "spells `_hidden`",
    ),
}
SOURCES = {
    f"src/pkg/{case.replace('-', '_')}/{name}": text.replace("CASE", case.replace("-", "_"))
    for case, kept in {**RENAMES, **REFUSALS}.items()
    for name, text in {"__init__.py": "", **kept.files}.items()
}
