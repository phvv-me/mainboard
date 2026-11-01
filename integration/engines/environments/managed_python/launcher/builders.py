from pathlib import Path

from mainboard.context.plan import environment_prefix
from mainboard.dispatch.agent.program import Request


def survey(root: str) -> Request:
    """The first thing a mirror asks: describe `root`, here with nothing in scope."""
    return {"survey": {"root": root, "state": ".state", "scopes": [], "named": []}}


def environment(home: Path) -> Path:
    """Where the workspace environment's interpreter sits under `home`."""
    return Path(environment_prefix(str(home / ".mb-jobs"), env="default")) / "bin" / "python"
