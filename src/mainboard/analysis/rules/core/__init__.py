"""The language-neutral core every rule reads: SQL relations over `ast_nodes`, built by
sitting_duck's semantic types so one query serves Python, TypeScript, C/C++ and Rust.

Each relation is a table named `core_<relation>` (narrowed to a language by the cursor's
`search_path` like any family), keyed by the integer `file_id` of `core_file` and built once per
run, only the ones the rules read. A rule keeps its own shapes in its own SQL; the core holds
what several rules would spell again: which files parse, what a definition, a call, a parameter,
an import and a name use are, and which module an import reaches.
"""

from . import text as text
from .family import Core as Core
