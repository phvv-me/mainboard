# Running a project's proof on demand: its own command over a copy of the project as it stands.

from pathlib import Path

from ....log import logger
from ...engine.fixing.proof import Proof
from ...project.config import ProjectConfig


def proof(root: Path = Path()) -> int:
    """Run ROOT's proof (`[tool.mcmr] proof`) over a copy of the project as it stands.

    The copy keeps the run safe from edits made meanwhile, and the proof is told of the project
    (`MB_FIX_ROOT`) as `refactor fix` tells it. The exit is the proof's verdict: 0 when it passes.

    Args:
        root: the project whose proof runs.
    """
    root = root.resolve()
    command = ProjectConfig.of(root).proof
    if not command:
        logger.result(f"{root} declares no proof (`[tool.mcmr] proof`)")
        return 2
    judge = Proof(root, command)
    passed = judge.passes([], proven=root)
    logger.result(f"proof {'passed' if passed else 'failed'}: {judge.said}")
    return 0 if passed else 1
