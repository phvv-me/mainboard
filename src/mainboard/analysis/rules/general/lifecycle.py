"""Rules over repeatable project work and temporary controls (mcmr's ALL-LIFE family)."""

from ...catalog import Rule
from ...facts import Setting

_SPACE = " \t\n\r\x0b\x0c"


class ProjectAutomation(Rule, id="ALL-LIFE0001"):
    """Required actions lacking one repository-owned, unattended command."""

    required_tasks: tuple[str, ...] = ("setup", "lint", "typecheck", "test", "build")

    query = """
        SELECT f.* FROM automation_task_facts f
        LEFT JOIN (
          SELECT fact_id, count(DISTINCT capability) AS canonical FROM automation_task_tasks
          WHERE commands_length = 1 AND is_repository_owned AND is_noninteractive
            AND list_contains($required_tasks::VARCHAR[], capability)
          GROUP BY fact_id) c USING (fact_id)
        WHERE coalesce(c.canonical, 0) < $required
    """
    message = "`{fact_id}` satisfies project automation"

    def parameters(self) -> dict[str, Setting]:
        required = len(set(self.required_tasks))
        return super().parameters() | {"required": required}
