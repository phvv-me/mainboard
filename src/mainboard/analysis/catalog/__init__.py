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
"""

from .relations import Evidenced, Family, Function, Relation, define, install, needed
from .rules import Applicability, Example, Repair, Rule, RuleProvenance

__all__ = [
    "Applicability",
    "Evidenced",
    "Example",
    "Family",
    "Function",
    "Relation",
    "Repair",
    "Rule",
    "RuleProvenance",
    "define",
    "install",
    "needed",
]
