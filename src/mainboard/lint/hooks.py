import json
from pathlib import Path

from patos import FrozenOpenModel

from ..core.errors import MissionError
from ..core.project import Project
from .git import git, printed

# What an agent sees of a failing pass. Claude Code carries hook context into the conversation,
# and a type checker's full dump would crowd out the edit it is about.
_CONTEXT = 4000

# Git runs a hook through its own POSIX shell on every platform, Git for Windows included, so
# one line of `sh` is the whole script and everything it does lives in Python.
_SCRIPT = f"""#!/bin/sh
# Written by `{Project().name} lint install-hook`: lint what this commit records.
exec {Project().name} lint commit
"""


class GitHook:
    """The pre-commit hook that makes every `git commit` in the workspace run `lint commit`.

    root: the workspace root, inside the git work tree whose hook is written.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def install(self) -> Path:
        """Write the hook where git reads it, `core.hooksPath` and worktrees honored."""
        answer = git(self.root, "rev-parse", "--git-path", "hooks/pre-commit")
        if answer.returncode:
            raise MissionError(f"{self.root} is not in a git work tree to install a hook into")
        path = self.root / printed(answer.stdout).strip()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_SCRIPT, encoding="utf-8", newline="\n")
        path.chmod(0o755)
        return path


class _ToolInput(FrozenOpenModel):
    file_path: str = ""
    notebook_path: str = ""


class EditHook(FrozenOpenModel):
    """The payload Claude Code hands a PostToolUse hook, read for the one file just written.

    cwd: the session's working directory, which a relative path is read against.
    tool_input: the arguments of the tool that wrote the file.
    """

    cwd: Path = Path()
    tool_input: _ToolInput = _ToolInput()

    @property
    def edited(self) -> Path | None:
        """The file the tool wrote, None when the tool named none or it is already gone."""
        name = self.tool_input.file_path or self.tool_input.notebook_path
        if not name:
            return None
        path = (self.cwd / name).resolve()
        return path if path.is_file() else None

    @staticmethod
    def context(findings: str) -> str:
        """The hook answer that hands `findings` to the agent as context for its next step."""
        return json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": findings[:_CONTEXT],
                }
            }
        )
