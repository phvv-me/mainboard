"""mcmr's DependencyFact: the release evidence of a project's direct dependencies.

No syntax tree holds that evidence. mcmr's refresher asks the package index and the source host
for each dependency's resolved and latest compatible release day and standardized states, and its
stored baselines ran without one, so their ALL-DEPE rules never executed. `ReleaseEvidence` is
that refresher: it loads `release_runs`, `release_records` and `release_failures` (cached on
disk), and without a network answer it loads them empty, leaving this family without a fact.
"""

from pathlib import Path

from ..catalog import Family
from ..facts import Facts
from .releases import ReleaseEvidence

_FACTS = """
    SELECT 0::UBIGINT AS fact_order, 'dependencies' AS fact_id, 'pyproject.toml' AS path,
           1::UBIGINT AS start_line, 0::UBIGINT AS start_column, 1::UBIGINT AS end_line,
           0::UBIGINT AS end_column, NULL::VARCHAR AS language,
           (SELECT count(*) FROM release_records) AS dependencies_length,
           true AS dependencies_present,
           (SELECT count(*) FROM release_failures) AS evidence_length, true AS evidence_present
    FROM release_runs
"""

_DEPENDENCIES = """
    SELECT 0::UBIGINT AS fact_order, 'dependencies' AS fact_id, 'dependencies' AS parent_id,
           'dependencies/dependencies:' || ordinal AS record_id, ordinal, name, is_development,
           resolved_release_day, latest_compatible_release_day, latest_compatible_version,
           project_state, repository_state, resolved_release_state
    FROM release_records
"""

_EVIDENCE = """
    SELECT 0::UBIGINT AS fact_order, 'dependencies' AS fact_id, 'dependencies' AS parent_id,
           'dependencies/evidence:' || ordinal AS record_id, ordinal, 1.0 AS confidence, detail,
           signal, source
    FROM release_failures
"""


class Dependency(Family):
    family = "DependencyFact"
    relations = {"facts": _FACTS, "dependencies": _DEPENDENCIES, "evidence": _EVIDENCE}

    @classmethod
    def provide(cls, facts: Facts, root: Path) -> None:
        """Load the release evidence the relations read."""
        ReleaseEvidence(root=root).register(facts)
