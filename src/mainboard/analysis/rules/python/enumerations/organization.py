"""Rules over where enums live (mcmr's PY-ENUM organization family)."""

from ....catalog import Example, Rule

_VOCABULARY = """from enum import StrEnum


class Mode(StrEnum):
    FAST = "fast"
    SLOW = "slow"


class Level(StrEnum):
    LOW = "low"
    HIGH = "high"


class Phase(StrEnum):
    START = "start"
    STOP = "stop"


def describe(mode: Mode) -> str:
    return mode.value
"""
_MOVED = """from .enums import Mode


def describe(mode: Mode) -> str:
    return mode.value
"""
_ONE_MOVED = """from enum import StrEnum

from .enums import Mode


class Level(StrEnum):
    LOW = "low"
    HIGH = "high"


class Phase(StrEnum):
    START = "start"
    STOP = "stop"


def describe(mode: Mode) -> str:
    return mode.value
"""
_MODE = 'from enum import StrEnum\n\n\nclass Mode(StrEnum):\n    FAST = "fast"\n'
_STAYING = """from enum import IntEnum, StrEnum

LIMIT = 3


class Mode(IntEnum):
    FAST = LIMIT
    SLOW = 1


class Level(StrEnum):
    LOW = "low"
    HIGH = "high"


class Phase(StrEnum):
    START = "start"
    STOP = "stop"
"""
_HOME = "palette/vocabulary.py"
_USERS = {
    "palette/__init__.py": "",
    "palette/first.py": "from .vocabulary import Level, Mode\n",
    "palette/second.py": "from .vocabulary import Phase\n",
    "palette/third.py": "from .vocabulary import Mode\n",
}
_SHOP_HOME = "shop/billing/status.py"
_SHOP_USERS = {
    "shop/__init__.py": "",
    "shop/billing/__init__.py": "",
    "shop/orders/__init__.py": "",
    "shop/orders/cart.py": "from ..billing.status import Level, Mode\n",
    "shop/orders/view.py": "from ..billing.status import Mode\n",
    "shop/orders/pay.py": "from ..billing.status import Phase\n",
}
_NESTED_HOME = f"atelier/{_HOME}"
_NESTED_USERS = {
    "atelier/__init__.py": "",
    **{f"atelier/{name}": text for name, text in _USERS.items()},
}
_SUPPORT = {
    "atelier/checks/test_palette.py": """from .builders import colors


def test_the_palette_has_colors():
    assert colors()
""",
    "atelier/checks/builders.py": """from ..palette.vocabulary import Level, Mode, Phase


def colors():
    return [Level.LOW, Mode.FAST, Phase.START]
""",
}


class SharedEnumsModuleCandidate(Rule, id="PY-ENUM0003", applicability="review"):
    """Enum classes imported by other modules, outside the `enums.py` of the lowest folder holding
    the enum and every one of its importers, so that an `enums.py` is imported only from within
    its own folder."""

    preferred_module: str = "enums"

    # An enum sits right in an `enums/` package, whose modules each name one subject, or in an
    # `enums.py` whose folder is the lowest one holding every importer. Any other enum with an
    # importer, tests apart from source, belongs in `<lowest shared folder>/enums.py`, unless that
    # folder holds an `enums/` package, since a module and a package cannot share the name.
    query = r"""
        WITH homes AS (
          SELECT r.class_id, r.directory, r.destination, d.path, d.name,
                 regexp_replace(d.path, '/?[^/]*$', '') AS folder,
                 regexp_replace(regexp_extract(d.path, '[^/]*$'), '\.pyi?$', '')
                   = $preferred_module AS is_module
          FROM enum_reuse r JOIN enum_definitions d ON d.class_id = r.class_id
        )
        SELECT h.path, min(c.start_line) AS start_line,
               arg_min(c.start_column, c.start_line) AS start_column, h.destination,
               plural(count(*), 'reused enum') AS enums, list(h.name ORDER BY h.name) AS names
        FROM homes h JOIN class_defs c ON c.class_id = h.class_id
        WHERE NOT list_contains(string_split(h.folder, '/'), $preferred_module)
          AND NOT (h.is_module AND h.directory = h.folder)
          AND NOT EXISTS (SELECT 1 FROM class_files f
                          WHERE starts_with(f.path, regexp_replace(h.destination, '\.py$', '/')))
        GROUP BY h.path, h.destination
    """
    message = "`{destination}` would hold the {enums} defined here"
    # One move per finding (`move-definitions`): every enum the module defines for that
    # destination goes there, each importing statement respelled. Moves into a destination
    # that does not exist yet take it in turns (the operation claims it, so the first creates it
    # with its imports and the others follow a pass later). Left alone, with the reason, what the
    # operation refuses (an enum reading a name of its module that stays, a destination defining
    # the name otherwise, an importer binding the module as `import a as b`, a header comment
    # naming an enum that moves and a definition that stays).
    fix = """
        SELECT f.finding_key, 'move-definitions' AS operation, r.*
        FROM finding f, move_row(f.path, f.names, f.destination) r
    """
    examples = (
        Example(bad=_VOCABULARY, good=_MOVED, path=_HOME, context=_USERS),
        Example(
            bad=_VOCABULARY,
            good=_MOVED,
            path=_NESTED_HOME,
            context={**_NESTED_USERS, **_SUPPORT},
        ),
        Example(
            bad=_VOCABULARY,
            good=_MOVED.replace(".enums", "..enums"),
            path=_SHOP_HOME,
            context=_SHOP_USERS,
        ),
        Example(bad=_STAYING, good=_STAYING, path=_HOME, context=_USERS),
        Example(
            bad=_VOCABULARY,
            good=_ONE_MOVED,
            path=_HOME,
            context={
                "palette/__init__.py": "",
                "palette/first.py": "from .vocabulary import Mode\n",
            },
        ),
        Example(
            bad=_MODE,
            good="",
            path="palette/deep/enums.py",
            context={
                "palette/__init__.py": "",
                "palette/deep/__init__.py": "",
                "palette/first.py": "from .deep.enums import Mode\n",
            },
        ),
    )


_ENUMS_PACKAGE = {"pkg/__init__.py": "", "pkg/enums/__init__.py": ""}
_MIXED = """from enum import Enum


class Color(Enum):
    RED = 1


class Shape(Enum):
    SQUARE = 1
"""
_MIXED_THREE = """from enum import Enum


class Color(Enum):
    RED = 1


class Shape(Enum):
    SQUARE = 1


class Level(Enum):
    LOW = 1
"""
_MIXED_WITH_CLASS = """from enum import Enum


class Color(Enum):
    RED = 1


class Palette:
    shades = 3
"""
_MIXED_READING = """from enum import Enum

LIMIT = 4


class Color(Enum):
    RED = 1


class Shape(Enum):
    SQUARE = LIMIT
"""
_SHAPE_USER = "from .enums.mix import Shape\n\n\ndef corners(shape: Shape) -> int:\n    return 4\n"
_BOTH_USER = """from .enums.mix import Level, Shape


def both() -> tuple:
    return Level, Shape
"""
_PALETTE_USER = (
    "from .enums.mix import Palette\n\n\ndef pick() -> Palette:\n    return Palette()\n"
)


class SharedEnumFileShape(Rule, id="PY-ENUM0004", applicability="review"):
    """Shared `enums` implementation files that do not hold exactly one enum class."""

    # A file with an enum states each of its other top-level classes as a finding of its own, so
    # that moving one is progress the fixer sees; a file with none is one finding.
    query = """
        WITH tops AS (
          SELECT f.path, f.top_level_class_count, c.name, c.start_line, c.start_column,
                 e.class_id IS NOT NULL AS is_enum,
                 min(c.start_line) FILTER (WHERE e.class_id IS NOT NULL)
                   OVER (PARTITION BY f.path) AS first_enum
          FROM enum_files f
          JOIN class_defs c ON c.path = f.path
          JOIN class_nodes s ON s.fact_id = c.fact_id AND s.node_id = c.statement_node
                            AND s.depth = 1
          LEFT JOIN enum_classes e ON e.class_id = c.class_id
          WHERE NOT f.is_package_initializer
        )
        SELECT path, start_line, start_column, name, is_enum,
               '`' || name || '` shares an enums file that holds exactly one enum class'
                 AS message
        FROM tops WHERE top_level_class_count > 1 AND start_line <> first_enum
        UNION ALL
        SELECT path, 1, 0, NULL, NULL, '`' || path || '` is an enums file with no enum class'
        FROM enum_files WHERE NOT is_package_initializer AND enum_class_count = 0
    """
    # Each class beyond the first enum moves into a module of its own (`move-definitions`):
    # another enum beside the file, a class that is no enum up to the folder holding the first
    # `enums` directory, since the file's own directory would flag it again. A destination that
    # exists stays untouched. Left alone, with the reason, what the operation refuses (a class
    # reading a name of its module that stays, an importer binding the module as `import a as b`).
    fix = r"""
        WITH moved AS (
          SELECT finding_key, path, name,
                 module_path(if(is_enum, regexp_replace(path, '/?[^/]*$', ''),
                                regexp_replace(path, '(^|/)enums(/.*)?$', '')), name)
                   AS destination
          FROM finding WHERE name IS NOT NULL
        )
        SELECT m.finding_key, 'move-definitions' AS operation, r.*
        FROM moved m, move_row(m.path, [m.name], m.destination) r
        WHERE NOT EXISTS (SELECT 1 FROM core_file c WHERE c.path = m.destination)
    """
    examples = (
        Example(
            bad=_SHAPE_USER,
            good=_SHAPE_USER.replace("enums.mix", "enums.shape"),
            path="pkg/use.py",
            context={**_ENUMS_PACKAGE, "pkg/enums/mix.py": _MIXED},
        ),
        Example(
            bad=_BOTH_USER,
            good=_BOTH_USER.replace(
                "from .enums.mix import Level, Shape",
                "from .enums.level import Level\nfrom .enums.shape import Shape",
            ),
            path="pkg/use.py",
            context={**_ENUMS_PACKAGE, "pkg/enums/mix.py": _MIXED_THREE},
        ),
        Example(
            bad=_PALETTE_USER,
            good=_PALETTE_USER.replace("enums.mix", "palette"),
            path="pkg/use.py",
            context={**_ENUMS_PACKAGE, "pkg/enums/mix.py": _MIXED_WITH_CLASS},
        ),
        Example(
            bad=_SHAPE_USER,
            good=_SHAPE_USER,
            path="pkg/use.py",
            context={**_ENUMS_PACKAGE, "pkg/enums/mix.py": _MIXED_READING},
        ),
    )
