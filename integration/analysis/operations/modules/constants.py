# The packages of a generated source tree (each holds an `__init__.py`) and its possible modules.
PACKAGES = ("pkg", "pkg.a", "pkg.b", "pkg.a.c")
MODULES = ("pkg.m", "pkg.a.m", "pkg.a.n", "pkg.b.m", "pkg.b.n", "pkg.a.c.m")

# A package whose `rules/rules.py` is to become `rules/core.py`, and the move that does it.
TREE = {
    "src/pkg/__init__.py": "",
    "src/pkg/rules/__init__.py": "",
    "src/pkg/rules/rules.py": "X = 1\n",
}
MOVES = {"pkg.rules.rules": "pkg.rules.core"}
RULES = "src/pkg/rules/rules.py"
CORE = "src/pkg/rules/core.py"
