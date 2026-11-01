from collections.abc import Sequence

_COLLECTIONS = "collections.abc"
# The capability each concrete type stands in for, as PY-INTE0001 reads it.
_ABSTRACT = {
    "dict": (_COLLECTIONS, "Mapping"),
    "float": ("numbers", "Real"),
    "int": ("numbers", "Integral"),
    "list": (_COLLECTIONS, "Sequence"),
    "set": (_COLLECTIONS, "Set"),
    "tuple": (_COLLECTIONS, "Sequence"),
}
_CHECK = """\
def first(items):
    if isinstance(items, {types}):
        return len(items)
    return items
"""
_SEQUENCE = """\
from collections.abc import Sequence


"""
_PRESENT = """\
from collections.abc import Sequence


def first(items: Sequence):
    if isinstance(items, (list, tuple)):
        return items[0]
    return items
"""
_SHARED = """\
def first(a, b):
    if isinstance(a, list):
        return a[0]
    if isinstance(b, (dict, list)):
        return len(b)
    return None
"""
_SHARED_FIXED = """\
from collections.abc import Mapping, Sequence


def first(a, b):
    if isinstance(a, Sequence):
        return a[0]
    if isinstance(b, (Mapping, Sequence)):
        return len(b)
    return None
"""
_TAKEN = """\
def first(items):
    Sequence = 3
    if isinstance(items, (list, tuple)):
        return items[0] + Sequence
    return items
"""
_ATTRIBUTE = """\
import numpy as np


def first(items):
    if isinstance(items, (list, np.ndarray)):
        return items[0]
    return items
"""
_ATTRIBUTE_FIXED = """\
from collections.abc import Sequence

import numpy as np


def first(items):
    if isinstance(items, (Sequence, np.ndarray)):
        return items[0]
    return items
"""
_ALIASED = "from collections.abc import Sequence as Seq\n\n\n" + _CHECK.format(
    types="(list, tuple)"
)
_UNKNOWN = _CHECK.format(types="(list, make())")
_ONLY_STR = _CHECK.format(types="(str, bytes)")


def checks() -> dict[str, tuple[str, str]]:
    """The files the PY-INTE0001 fix reads, by project path, each with the text it must leave.

    A file the fix must refuse is its own expectation.
    """
    return {
        "carriage_returns.py": (
            _CHECK.format(types="(list, tuple)").replace("\n", "\r\n"),
            (_SEQUENCE + _CHECK.format(types="Sequence")).replace("\n", "\r\n"),
        ),
        "present.py": (_PRESENT, _PRESENT.replace("(list, tuple)", "Sequence")),
        "twice.py": (_SHARED, _SHARED_FIXED),
        "attribute_member.py": (_ATTRIBUTE, _ATTRIBUTE_FIXED),
        "taken.py": (_TAKEN, _TAKEN),
        "aliased.py": (_ALIASED, _ALIASED),
        "unknown.py": (_UNKNOWN, _UNKNOWN),
        "only_str.py": (_ONLY_STR, _ONLY_STR),
    }


def concrete_types() -> list[str]:
    """The types a check may name: those with a capability, and two with none."""
    return [*_ABSTRACT, "str", "Row"]


def stands_in(types: Sequence[str]) -> bool:
    """Whether some of `types` has a capability to stand in for."""
    return any(kind in _ABSTRACT for kind in types)


def checked(types: Sequence[str]) -> str:
    """A function guarding a `len` by an `isinstance` check against `types`."""
    shown = types[0] if len(types) == 1 else f"({', '.join(types)})"
    return _CHECK.format(types=shown)


def abstracted(types: Sequence[str]) -> str:
    """The function `checked` becomes: each type its capability, named once, each imported."""
    names = list(
        dict.fromkeys(_ABSTRACT[kind][1] if kind in _ABSTRACT else kind for kind in types)
    )
    homes = sorted({_ABSTRACT[kind] for kind in types if kind in _ABSTRACT})
    modules = sorted({home for home, _ in homes})
    imports = "".join(
        f"from {home} import {', '.join(sorted(name for owner, name in homes if owner == home))}\n"
        for home in modules
    )
    return imports + "\n\n" + checked(names)


_POINT = """\
from pydantic import BaseModel


class Point(BaseModel):
    x: int
    y: int
    z: int
    w: int
    tags: list[str] = []
    note: str | None = None
"""
_SPOT = """\
from pydantic import BaseModel


class Spot(BaseModel):
    x: int
    y: int
    z: int
    w: int

    def as_dict(self):
        return {"w": self.w, "z": self.z, "y": self.y, "x": self.x}
"""
_SPOT_DUMPED = _SPOT.replace(
    '{"w": self.w, "z": self.z, "y": self.y, "x": self.x}',
    'self.model_dump(include={"w", "z", "y", "x"})',
)
_ANNOTATED = """\
from shapes import Point


def dump(p: Point):
    return {"x": p.x, "y": p.y, "tags": p.tags, "note": p.note}
"""
_ANNOTATED_DUMPED = _ANNOTATED.replace(
    '{"x": p.x, "y": p.y, "tags": p.tags, "note": p.note}',
    'p.model_dump(include={"x", "y", "tags", "note"})',
)
_QUARTET = '{"x": p.x, "y": p.y, "z": p.z, "w": p.w}'
_DUMP = f"from shapes import Point\n\n\ndef dump(p: Point):\n    return {_QUARTET}\n"
_PLAIN = "def dump(r):\n    return " + _QUARTET.replace("p.", "r.") + "\n"
_DASHED = _DUMP.replace('"z"', '"z-0"')
_SPREAD = _DUMP.replace("}\n", ", **EXTRA}\n")
_UNION = _DUMP.replace("p: Point", "p: Point | None")
_COMMENTED = """\
from shapes import Point


def dump(p: Point):
    return {
        "x": p.x,  # first
        "y": p.y,
        "z": p.z,
        "w": p.w,
    }
"""
_KEYWORDS = _DUMP.replace(_QUARTET, "dict(x=p.x, y=p.y, z=p.z, w=p.w)")


def projections() -> dict[str, tuple[str, str]]:
    """The files the PY-MODE0004 fix reads, by project path, each with the text it must leave.

    `shapes.py` holds the model the files annotate their parameters with. A file the fix must
    refuse is its own expectation.
    """
    return {
        "shapes.py": (_POINT, _POINT),
        "spot.py": (_SPOT, _SPOT_DUMPED),
        "annotated.py": (_ANNOTATED, _ANNOTATED_DUMPED),
        "carriage_returns.py": (
            _DUMP.replace("\n", "\r\n"),
            _DUMP.replace(_QUARTET, 'p.model_dump(include={"x", "y", "z", "w"})').replace(
                "\n", "\r\n"
            ),
        ),
        "plain.py": (_PLAIN, _PLAIN),
        "dashed.py": (_DASHED, _DASHED),
        "spread.py": (_SPREAD, _SPREAD),
        "union.py": (_UNION, _UNION),
        "commented.py": (_COMMENTED, _COMMENTED),
        "keywords.py": (_KEYWORDS, _KEYWORDS),
    }


def modeled(fields: Sequence[str]) -> str:
    """A model of five fields whose method projects `fields`, in that order, by hand."""
    members = "".join(f"    {name}: int\n" for name in "abcde")
    entries = ", ".join(f'"{name}": self.{name}' for name in fields)
    return (
        "from pydantic import BaseModel\n\n\nclass Row(BaseModel):\n"
        f"{members}\n    def as_dict(self):\n        return {{{entries}}}\n"
    )


def dumped(fields: Sequence[str]) -> str:
    """The model `modeled` becomes: the projection a `model_dump` over the same fields."""
    names = ", ".join(f'"{name}"' for name in fields)
    return (
        modeled(fields).split("        return")[0]
        + f"        return self.model_dump(include={{{names}}})\n"
    )
