import subprocess
import sys
from pathlib import Path
from shlex import quote

import pytest

from mainboard import Project, load
from mainboard.lint import Linter

# One stand-in for every formatter and linter: a script whose first word picks what it does, so
# a test manifest spells real command lines and the pass runs real processes.
_TOOL = """
import os, subprocess, sys, time
mode, *rest = sys.argv[1:]
if mode == "append":
    word, *files = rest
    for name in files:
        with open(name, "a", encoding="utf-8") as handle:
            handle.write(word + "\\n")
elif mode in ("flag", "lacks"):
    word, *files = rest
    flagged = [
        name for name in files if (word in open(name, encoding="utf-8").read()) == (mode == "flag")
    ]
    print("\\n".join(flagged))
    sys.exit(1 if flagged else 0)
elif mode == "where":
    print(os.getcwd(), *rest)
    sys.exit(1)
elif mode == "hang":
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    time.sleep(60)
"""

_HEADER = '[workspace]\nname = "lint"\n'


class Repository:
    """A throwaway git work tree holding a workspace manifest and the stand-in tool.

    root: the work tree, which is also the workspace root.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q", "-b", "main")

    def git(self, *arguments: str) -> str:
        """Run git in the work tree, as an author whose identity no global config supplies."""
        return subprocess.run(
            [
                "git",
                "-C",
                str(self.root),
                "-c",
                "user.name=lint",
                "-c",
                "user.email=lint@example.com",
                *arguments,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def write(self, name: str, content: str | bytes) -> Path:
        """Write `content` at the workspace-relative `name`, parents made, bytes kept exact."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)
        return path

    def commit(self) -> None:
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "state")

    def manifest(self, lint: str = "") -> None:
        """Write the manifest with `lint` as its `[lint]` tables, the stand-in tool beside it."""
        self.write("tool.py", _TOOL)
        self.write(Project().manifest, f"{_HEADER}\n{lint}")

    def linter(self) -> Linter:
        return Linter(self.root, load(self.root / Project().manifest))


def tool(arguments: str) -> str:
    """The command line that runs the stand-in tool in `arguments`' mode."""
    return f"{quote(Path(sys.executable).as_posix())} {{root}}/tool.py {arguments}"


@pytest.fixture(autouse=True)
def hermetic_git(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the machine's own git configuration, a global hooks path above all, out of reach."""
    empty = tmp_path_factory.mktemp("git") / "config"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def repository(tmp_path: Path) -> Repository:
    """An empty repository with one commit, so HEAD exists for every diff a test asks for."""
    made = Repository(tmp_path / "work")
    made.manifest()
    made.commit()
    return made
