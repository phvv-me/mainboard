"""The core's relations as one family, and the closure of those a set of views reads."""

import re
from collections.abc import Set

from ...catalog import Family
from .constants import annotations, calls, files
from .constants.bindings import definitions, imports, references
from .constants.text import TEXT


class Core(Family):
    """The core as one family: `file` first, so every relation narrows through its `language`."""

    family = "Core"
    verified = False
    relations = {
        "file": files.FILE,
        "node": files.NODE,
        "import": imports.IMPORT,
        "def": definitions.DEF,
        "param": definitions.PARAM,
        "decor": definitions.DECOR,
        "annotation": annotations.ANNOTATION,
        "call": calls.CALL,
        "call_arg": calls.CALL_ARG,
        "ref": references.REF,
        "text": TEXT,
    }

    @classmethod
    def reading(cls, views: Set[str]) -> dict[str, str]:
        """The relations `views` name and, transitively, those they read, in dependency order."""
        wanted = {name for name in cls.relations if f"core_{name}" in views}
        pending = list(wanted)
        while pending:
            for read in re.findall(r"\bcore_(\w+)", cls.relations[pending.pop()]):
                if read in cls.relations and read not in wanted:
                    wanted.add(read)
                    pending.append(read)
        return {name: sql for name, sql in cls.relations.items() if name in wanted}
