# Commands in the workspace's scripts that mean different things on different machines.
#
# The center may be macOS, Linux or Windows and the agents on it run whichever shell they run, so
# a script that leans on GNU or BSD behavior works on one center and breaks on the next, usually
# without an error: `sed -i` takes a suffix argument on macOS and none on Linux, `stat -f` is a
# format on BSD and a filesystem query on GNU, and `timeout` does not exist on Windows at all.
# Every such command is found here in the tracked scripts, tasks and agent hooks, and named with
# the portable replacement the default environment carries or the verb this tool provides.

import re
from fnmatch import fnmatch
from typing import TYPE_CHECKING

from pathspec import GitIgnoreSpec
from patos import FrozenModel

from ..core.section import Section, Verdict
from ..workstation import abbreviated

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ..git.repo import Repo

# The tracked files whose lines are commands: scripts of every shell, task tables and hooks.
_SCRIPTS = (
    "*.sh",
    "*.bash",
    "*.zsh",
    "*.ps1",
    "*.bat",
    "*.cmd",
    "Makefile",
    "*.mk",
    "maskfile.md",
    "mainboard.toml",
    "settings.json",
    ".github/workflows/*.yml",
)

# Files past this size are data that happens to match a script's name, not a script.
_LARGEST = 1 << 20


class Divergence(FrozenModel):
    """One platform-divergent command form and what replaces it everywhere.

    name: the form, as a reader would search for it.
    pattern: the regular expression finding it in a command line.
    replacement: what to write instead, portable on every center.
    """

    name: str
    pattern: str
    replacement: str


RULES = tuple(
    Divergence(name=name, pattern=pattern, replacement=replacement)
    for name, pattern, replacement in (
        (
            "sed -i",
            r"\bsed\s+(?:-\w+\s+)*-\w*i\b",
            "sd (in-place edits with one syntax on every system)",
        ),
        ("find -printf", r"\bfind\b[^|;&\n]*\s-printf\b", "fd --format, or rg --files"),
        ("grep -P", r"\bgrep\s+(?:-\w+\s+)*-\w*P", "rg -P"),
        (
            "timeout",
            r"(?:^|[;&|(]|\s)timeout\s+-?\w",
            "mainboard proc timeout <seconds> -- <command>",
        ),
        (
            "xargs -r",
            r"\bxargs\s+(?:-\w+\s+)*-\w*r\b",
            "fd --exec, or a loop in the task's own language",
        ),
        (
            "readlink -f",
            r"\breadlink\s+-\w*f\b",
            "realpath from the environment's uutils coreutils",
        ),
        (
            "stat -c/-f",
            r"\bstat\s+(?:-\w+\s+)*-[cf]\b",
            "the environment's uutils stat -c (GNU flags), or Python's os.stat",
        ),
        ("date -d", r"\bdate\s+(?:-\w+\s+)*-d\b", "the environment's uutils date -d (GNU flags)"),
        (
            "pkill/killall",
            r"\b(?:pkill|killall)\b",
            "mainboard proc kill <pid> (the whole process tree)",
        ),
        ("ps aux", r"\bps\s+(?:aux|-ef)\b", "procs"),
        ("jq", r"(?:^|[|;&(]|\s)jq\s", "yq -p json (go-yq)"),
        (
            "flock",
            r"(?:^|[;&|(]|\s)flock\s",
            "a lock taken in Python (filelock), which Windows also honors",
        ),
        ("nc -z", r"\bnc\s+(?:-\w+\s+)*-\w*z", "mainboard proc wait --port <host:port>"),
        (
            "sleep loop",
            r"\bwhile\b[^\n]*;\s*do\s+sleep\b",
            "mainboard proc wait --file <path> or --port <host:port>",
        ),
    )
)


class Portability:
    """Every platform-divergent command in the scripts the owned repositories track.

    repos: the owned repositories, each read through its own `git ls-files`.
    exclude: gitignore-style workspace patterns never read, the `[lint]` ones: vendored and
        frozen sources are somebody else's scripts.
    """

    def __init__(self, repos: Sequence[Repo], exclude: Sequence[str] = ()) -> None:
        self.repos = repos
        self.excluded = GitIgnoreSpec.from_lines(exclude)

    def sections(self) -> list[Section]:
        """One row per divergent form found, naming where; one passing row when there is none."""
        found = self.found()
        rows = [
            Section(
                section=f"portable: {rule.name}",
                verdict=Verdict.WARN,
                detail=f"{len(places)} uses: {abbreviated(places)}",
                fix=rule.replacement,
            )
            for rule in RULES
            if (places := found.get(rule.name))
        ]
        return rows or [
            Section(
                section="portable",
                verdict=Verdict.PASS,
                detail="no platform-divergent commands in the tracked scripts",
            )
        ]

    def found(self) -> dict[str, list[str]]:
        """Every place each form appears, as workspace-relative `file:line`, by the form's name."""
        compiled = [(rule.name, re.compile(rule.pattern)) for rule in RULES]
        places: dict[str, list[str]] = {}
        for where, path in self.scripts():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError, OSError:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if line.lstrip().startswith(("#", "//", "REM ", "::")):
                    continue
                for name, pattern in compiled:
                    if pattern.search(line):
                        places.setdefault(name, []).append(f"{where}:{number}")
        return places

    def scripts(self) -> list[tuple[str, Path]]:
        """Every tracked script file small enough to be one, by its workspace-relative name."""
        found: list[tuple[str, Path]] = []
        for repo in self.repos:
            prefix = "" if repo.name == "." else f"{repo.name}/"
            found += [
                (prefix + relative, path)
                for relative in repo.git.run("ls-files", "-z").stdout.split("\0")
                if relative
                and not self.excluded.match_file(prefix + relative)
                and any(
                    fnmatch(relative, pattern) or fnmatch(relative.rsplit("/", 1)[-1], pattern)
                    for pattern in _SCRIPTS
                )
                and (path := repo.path / relative).is_file()
                and not path.is_symlink()
                and path.stat().st_size <= _LARGEST
            ]
        return found
