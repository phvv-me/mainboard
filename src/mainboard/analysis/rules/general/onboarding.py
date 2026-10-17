"""Rules over how quickly a newcomer can work (mcmr's ALL-ONBO0001)."""

from ...catalog import Rule
from ...facts import Setting


class OnboardingReadiness(Rule, id="ALL-ONBO0001"):
    """Required onboarding capabilities without one documented, repository-owned, scripted task."""

    minimum: float = 95
    required_capabilities: tuple[str, ...] = (
        "setup",
        "test",
        "architecture",
        "debug",
        "contribute",
    )

    query = """
        SELECT f.*, CASE WHEN $required = 0 THEN 100.0
                         ELSE coalesce(v.verified, 0) / $required * 100.0 END AS value
        FROM automation_task_facts f LEFT JOIN (
          SELECT fact_id, count(DISTINCT capability) AS verified FROM automation_task_tasks
          WHERE list_contains($required_capabilities::VARCHAR[], capability) AND commands_length = 1
            AND guidance_locations_length > 0 AND is_repository_owned AND is_noninteractive
          GROUP BY fact_id) v USING (fact_id)
        WHERE value < $minimum
    """
    message = "onboarding readiness is {value} for `{fact_id}`"

    def parameters(self) -> dict[str, Setting]:
        required = len(set(self.required_capabilities))
        return super().parameters() | {"required": required}
