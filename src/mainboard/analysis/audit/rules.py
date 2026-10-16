"""Rules over the state of declared dependencies (mcmr's ALL-DEPE state family)."""

from ..catalog import Rule


class DependencyTechnicalLag(Rule, id="ALL-DEPE0001"):
    """Files whose resolved dependencies lag their latest compatible release too often."""

    maximum: float = 5
    maximum_release_lag_days: int = 180
    include_development: bool = False

    query = """
        SELECT f.*, lagging / measurable * 100.0 AS value FROM dependency_facts f JOIN (
          SELECT fact_id, count(*) AS measurable,
                 count(*) FILTER (WHERE latest_compatible_release_day - resolved_release_day
                                        > $maximum_release_lag_days) AS lagging
          FROM dependency_dependencies
          WHERE ($include_development OR NOT is_development)
            AND resolved_release_day IS NOT NULL AND latest_compatible_release_day IS NOT NULL
          GROUP BY fact_id) USING (fact_id)
        WHERE value > $maximum
    """
    message = "dependency technical lag is {value} percent for `{fact_id}`"


class ExplicitDependencyStateCount(Rule, id="ALL-DEPE0002"):
    """Files holding dependencies with an explicit adverse upstream or release state."""

    include_yanked: bool = True

    query = """
        SELECT f.*, value FROM dependency_facts f JOIN (
          SELECT fact_id, count(*) AS value FROM dependency_dependencies
          WHERE project_state IN ('archived', 'deprecated', 'quarantined')
             OR repository_state = 'archived'
             OR ($include_yanked AND resolved_release_state = 'yanked')
          GROUP BY fact_id) USING (fact_id)
    """
    message = "explicit dependency state count is {value} for `{fact_id}`"


class DependencyEvidenceGapPercentage(Rule, id="ALL-DEPE0004"):
    """Files whose dependencies too often miss the facts offline version checks need."""

    maximum: float = 5

    query = """
        SELECT f.*, CASE WHEN dependencies_length > 0
                         THEN coalesce(incomplete, 0) / dependencies_length * 100.0
                         ELSE 0.0 END AS share
        FROM dependency_facts f LEFT JOIN (
          SELECT fact_id, count(*) AS incomplete FROM dependency_dependencies
          WHERE resolved_release_day IS NULL OR latest_compatible_version IS NULL
             OR latest_compatible_release_day IS NULL
          GROUP BY fact_id) USING (fact_id)
        WHERE share > $maximum
    """
    message = "dependency evidence gap percentage is {share} for `{fact_id}`"
