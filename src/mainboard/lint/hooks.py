import json
from pathlib import Path

from patos import FrozenOpenModel

from ..core.errors import MissionError
from ..core.project import Project
from .git import git, printed

# What an agent sees of a failing pass. Claude Code carries hook context into the conversation,
# and a type checker's full dump would crowd out the edit it is about.
_CONTEXT = 4000


class GitHook:
    """A git hook that runs one of this tool's verbs, and nothing else, before git goes on.

    Git runs a hook through its own POSIX shell on every platform, Git for Windows included, so
    one line of `sh` is the whole script and everything it does lives in Python.

    root: a directory inside the git work tree whose hook is written.
    hook: the hook's name, `pre-commit` or `pre-push`.
    command: the verb and its arguments the hook runs, `lint commit` say.
    """

    def __init__(self, root: Path, *, hook: str, command: str) -> None:
        self.root = root
        self.hook = hook
        self.command = command

    @property
    def script(self) -> str:
        """The hook's whole text."""
        tool = Project().name
        return (
            f"#!/bin/sh\n# Written by `{tool} {self.command.split()[0]} install-hook`.\n"
            f"exec {tool} {self.command}\n"
        )

    def install(self) -> Path:
        """Write the hook where git reads it, `core.hooksPath` and worktrees honored."""
        answer = git(self.root, "rev-parse", "--git-path", f"hooks/{self.hook}")
        if answer.returncode:
            raise MissionError(f"{self.root} is not in a git work tree to install a hook into")
        path = self.root / printed(answer.stdout).strip()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.script, encoding="utf-8", newline="\n")
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
