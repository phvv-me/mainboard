"""The language-neutral core every rule reads: SQL relations over `ast_nodes`, built by
sitting_duck's semantic types so one query serves Python, TypeScript, C/C++ and Rust.

Each relation is a table named `core_<relation>` (narrowed to a language by the cursor's
`search_path` like any family), keyed by the integer `file_id` of `core_file` and built once per
run, only the ones the rules read. A rule keeps its own shapes in its own SQL; the core holds
what several rules would spell again: which files parse, what a definition, a call, a parameter,
an import and a name use are, and which module an import reaches.
"""

import re

from ...catalog import Family
from .annotations import ANNOTATION
from .calls import CALL, CALL_ARG
from .definitions import DECOR, DEF, PARAM
from .edges import EDGE
from .files import FILE, NODE
from .imports import IMPORT
from .references import REF
from .text import TEXT


class Core(Family):
    """The core as one family: `file` first, so every relation narrows through its `language`."""

    family = "Core"
    relations = {
        "file": FILE,
        "node": NODE,
        "import": IMPORT,
        "def": DEF,
        "param": PARAM,
        "decor": DECOR,
        "annotation": ANNOTATION,
        "call": CALL,
        "call_arg": CALL_ARG,
        "ref": REF,
        "edge": EDGE,
        "text": TEXT,
    }

    @classmethod
    def reading(cls, views: set[str]) -> dict[str, str]:
        """The relations `views` name and, transitively, those they read, in dependency order."""
        wanted = {name for name in cls.relations if f"core_{name}" in views}
        pending = list(wanted)
        while pending:
            for read in re.findall(r"\bcore_(\w+)", cls.relations[pending.pop()]):
                if read in cls.relations and read not in wanted:
                    wanted.add(read)
                    pending.append(read)
        return {name: sql for name, sql in cls.relations.items() if name in wanted}
