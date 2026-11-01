"""What a rule is: a named SQL relation over the fact views whose rows are findings, with its
settings as typed fields and its fix as SQL over those findings.

```python
class PackageDepth(Rule, id="ALL-FILE0002"):
    \"\"\"Directory levels between a directory and the source root above it.\"\"\"

    maximum: int = 5
    framework_roots: tuple[str, ...] = ()

    query = \"\"\"
        SELECT *, source_depth AS value FROM directory_facts WHERE source_depth > $maximum
    \"\"\"
    message = "package depth is {value} for `{fact_id}`"
```

Fields are settings: a project overrides them, pydantic validates them, and the query reads them
as `$name`. `query` yields one row per finding with `path`, `start_line`, `start_column` (and
`end_line`, `end_column` when the fix edits the finding's own span), plus the columns `message`
templates, or a `message` column itself.

`fix` is SQL over `finding`, the query's rows, yielding one row per edit (`Edit`): a span and the
`text` replacing it (empty text deletes, an empty span inserts), an `import` the text needs, or a
file `operation`. A fix the facts cannot compute in SQL is a method `repair(self, hit)` returning
edits instead. `Relation`s are the shared building blocks: `class PlacedSymbols(Relation)` is the
table macro `placed_symbols()`, whose views resolve in the calling rule's language and whose
fields are its parameters.

A `Bundle` is one rule whose findings are those of its `Part`s, each a rule written as above but
with no id: `class Ceiling(Bundle, id="ALL-CEIL0001")` holds one `Part` field per measure, so a
project configures a part as `[tool.mcmr.rules."ALL-CEIL0001".module_line_count]`.
"""

from .relations import Evidenced, Family, Function, Relation, define, needed
from .rules import Applicability, Bundle, Example, Part, Repair, Rule, RuleProvenance

__all__ = [
    "Applicability",
    "Bundle",
    "Evidenced",
    "Example",
    "Family",
    "Function",
    "Part",
    "Relation",
    "Repair",
    "Rule",
    "RuleProvenance",
    "define",
    "needed",
]
