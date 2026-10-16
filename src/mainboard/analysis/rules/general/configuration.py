"""Rules over project configuration (mcmr's ALL-CONF family)."""

from ...catalog import Rule
from ...facts import Setting

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
