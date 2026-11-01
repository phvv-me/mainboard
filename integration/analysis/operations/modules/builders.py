from collections.abc import Mapping, Sequence

from hypothesis import strategies as st

from mainboard.analysis.operations.python.imports import Place

from .constants import MODULES, PACKAGES


def path_of(module: str, *, package: bool = False) -> str:
    """The file of a dotted `module` under `src/`: its `__init__.py` when it is a `package`."""
    stem = f"src/{module.replace('.', '/')}"
    return f"{stem}/__init__.py" if package else f"{stem}.py"


def module_of(path: str) -> str:
    """The dotted module of the Python file at `path` under `src/`."""
    return (
        path.removeprefix("src/").removesuffix(".py").removesuffix("/__init__").replace("/", ".")
    )


def records(sources: Mapping[str, str]) -> list[dict[str, str]]:
    """The `files` records (`path`, `module`, `content`) of the Python files of `sources`."""
    return [
        {"path": path, "module": module_of(path), "content": content}
        for path, content in sorted(sources.items())
    ]


def _spelled(here: Place, source: str, *, level: int) -> str:
    """The module `source` spelled from the package `here` starts from, with `level` dots."""
    start = here.package[: len(here.package) - level + 1]
    return "." * level + ".".join(source.split(".")[len(start) :])


@st.composite
def _statement(draw: st.DrawFn, here: Place, names: Sequence[str]) -> str:
    """One `from ... import` of a name in `names` (a module, or a name inside it) from `here`.

    It is absolute or reaches the target's package by any number of dots that still do; a module
    is imported as a member of its package, a name as a member of the module, either aliased.
    """
    target = draw(st.sampled_from(names))
    member = draw(st.booleans()) and "." in target
    source, _, leaf = target.rpartition(".")
    source, name = (source, leaf) if member else (target, "NAME")
    parts = source.split(".")
    levels = [
        level
        for level in range(1, len(here.package) + 1)
        if parts[: len(here.package) - level + 1] == here.package[: len(here.package) - level + 1]
    ]
    level = draw(st.sampled_from([0, *levels]))
    spelled = _spelled(here, source, level=level) if level else source
    suffix = f" as {name.lower()}_alias" if draw(st.booleans()) else ""
    return f"from {spelled} import {name}{suffix}"


@st.composite
def trees(draw: st.DrawFn) -> tuple[dict[str, str], str, str]:
    """A source tree, a module in it, and the name the module is to take in a package of it.

    Every file imports a few of the others, absolutely or relatively, a module as a member of its
    package or a name from inside it, aliased or not.
    """
    modules = draw(st.lists(st.sampled_from(MODULES), min_size=1, unique=True))
    moving = draw(st.sampled_from(modules))
    new = f"{draw(st.sampled_from(PACKAGES))}.moved"
    places = [Place(path_of(each, package=True), each) for each in PACKAGES]
    places += [Place(path_of(each), each) for each in modules]
    sources = {}
    for place in places:
        statements = [
            draw(_statement(place, [*PACKAGES, *modules])) for _ in range(draw(st.integers(0, 3)))
        ]
        sources[place.path] = "".join(f"{each}\n" for each in statements)
    return sources, moving, new


type Row = dict[str, str | list[str] | list[dict[str, str]]]


def moving(sources: Mapping[str, str], *moves: tuple[str, str], **columns: list[str]) -> Row:
    """A `move-modules` fix row moving each `(path, target)`, over the Python files of `sources`.

    `columns` add the row's other columns (`held`, `inits`, `mentions`, `removed`).
    """
    files = records({path: text for path, text in sources.items() if path.endswith(".py")})
    return {
        "operation": "move-modules",
        "moves": [{"path": path, "target": target} for path, target in moves],
        "files": files,
        **columns,
    }
