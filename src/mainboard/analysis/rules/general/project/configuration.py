"""Rules over project configuration (mcmr's ALL-CONF family)."""

from ....catalog import Example, Rule
from ....engine.facts import Setting

_PATTERN_SYNTAX = r"[\^$|()+]|\\[^\w]|\\[dDwWsSbBAZ]"


def _named(*words: str) -> str:
    """Regex matching a lowercase name that holds one of `words` as a whole word."""
    return rf"(?:^|[^a-z])(?:{'|'.join(words)})(?:$|[^a-z])"


class HardcodedPathPolicyCount(Rule, id="ALL-CONF0001"):
    """Files holding collections that hardcode which paths are discovered."""

    minimum_paths: int = 3
    policy_names: tuple[str, ...] = (
        "directories",
        "directory",
        "exclude",
        "excluded",
        "ignore",
        "ignored",
        "suffixes",
        "suffix",
    )

    query = """
        WITH assignments AS (
          SELECT fact_id, record_id, lower(name) AS name FROM project_configuration_assignments
          WHERE collection_kind <> 'other' AND values_length >= $minimum_paths
            AND NOT is_typed_configuration_field
        ), shapes AS (
          SELECT fact_id, parent_id,
            bool_and((starts_with(string_value, '.') OR contains(string_value, '*'))
                     AND NOT is_pattern) AS all_suffixes,
            bool_or((contains(string_value, '/') OR contains(string_value, '\\')
                     OR contains(string_value, '*') OR starts_with(string_value, '.'))
                    AND NOT is_pattern) AS any_path_syntax,
            bool_and(is_pattern) AS all_patterns
          FROM (SELECT *, regexp_matches(string_value, $pattern_syntax) AS is_pattern
                FROM project_configuration_values WHERE relation = 'assignments.values')
          GROUP BY fact_id, parent_id
        )
        SELECT f.*, value FROM project_configuration_facts f JOIN (
          SELECT a.fact_id, count(*) AS value
          FROM assignments a JOIN shapes s ON s.fact_id = a.fact_id AND s.parent_id = a.record_id
          WHERE regexp_matches(name, $policy)
            AND CASE WHEN regexp_matches(name, $suffix) THEN all_suffixes
                     WHEN regexp_matches(name, $directory) THEN NOT all_patterns
                     ELSE any_path_syntax END
          GROUP BY a.fact_id) USING (fact_id)
    """
    message = "hardcoded path policy count is {value} for `{fact_id}`"

    def parameters(self) -> dict[str, Setting]:
        pattern_syntax = _PATTERN_SYNTAX
        policy = _named(*self.policy_names)
        suffix = _named("suffix", "suffixes")
        directory = _named("directory", "directories")
        return super().parameters() | {
            "pattern_syntax": pattern_syntax,
            "policy": policy,
            "suffix": suffix,
            "directory": directory,
        }


class UnmatchedRuleExclude(Rule, id="ALL-CONF0002", applicability="safe"):
    """Rule excludes that match no file of the project.

    A waiver names the files it exempts by path, so one whose file moved or went away exempts
    nothing: the move silently drops what it was written for, and the stale line hides that it
    is spent.
    """

    query = """
        SELECT f.fact_order, f.fact_id, e.path, e.line AS start_line, 0 AS start_column,
               e.line AS end_line, 0 AS end_column, f.language, e.rule, e.pattern
        FROM project_configuration_facts f, manifest_rule_excludes e
    """
    message = "`{rule}` excludes `{pattern}`, which matches no file of the project"
    # The stale pattern leaves its array (`drop-exclude`): with its line when it stands alone
    # there, cut from a one-line array otherwise. Safe: a pattern matching no file exempts
    # nothing, so dropping it changes no finding.
    fix = """
        SELECT f.finding_key, 'drop-exclude' AS operation, f.path, f.start_line, f.pattern,
               s.content
        FROM finding f JOIN source_files s ON s.path = f.path
    """
    examples = (
        Example(
            bad=(
                """[tool.mcmr.rules."ALL-FUNC0001"]
exclude = [
  "src/kept.py",
  "src/gone.py",
]
"""
            ),
            good='[tool.mcmr.rules."ALL-FUNC0001"]\nexclude = [\n  "src/kept.py",\n]\n',
            path="pyproject.toml",
            context={"src/kept.py": "VALUE = 1\n"},
        ),
    )
