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

# How many places a row names before it only counts the rest.
_NAMED = 3


class Divergence(FrozenModel):
    """One platform-divergent command form and what replaces it everywhere.

    name: the form, as a reader would search for it.
    pattern: the regular expression finding it in a command line.
    replacement: what to write instead, portable on every center.
    """

    name: str
    pattern: str
    replacement: str


RULES = (
    Divergence(
        name="sed -i",
        pattern=r"\bsed\s+(?:-\w+\s+)*-\w*i\b",
        replacement="sd (in-place edits with one syntax on every system)",
    ),
    Divergence(
        name="find -printf",
        pattern=r"\bfind\b[^|;&\n]*\s-printf\b",
        replacement="fd --format, or rg --files",
    ),
    Divergence(
        name="grep -P",
        pattern=r"\bgrep\s+(?:-\w+\s+)*-\w*P",
        replacement="rg -P",
    ),
    Divergence(
        name="timeout",
        pattern=r"(?:^|[;&|(]|\s)timeout\s+-?\w",
        replacement="mainboard proc timeout <seconds> -- <command>",
    ),
    Divergence(
        name="xargs -r",
        pattern=r"\bxargs\s+(?:-\w+\s+)*-\w*r\b",
        replacement="fd --exec, or a loop in the task's own language",
    ),
    Divergence(
        name="readlink -f",
        pattern=r"\breadlink\s+-\w*f\b",
        replacement="realpath from the environment's uutils coreutils",
    ),
    Divergence(
        name="stat -c/-f",
        pattern=r"\bstat\s+(?:-\w+\s+)*-[cf]\b",
        replacement="the environment's uutils stat -c (GNU flags), or Python's os.stat",
    ),
    Divergence(
        name="date -d",
        pattern=r"\bdate\s+(?:-\w+\s+)*-d\b",
        replacement="the environment's uutils date -d (GNU flags)",
    ),
    Divergence(
        name="pkill/killall",
        pattern=r"\b(?:pkill|killall)\b",
        replacement="mainboard proc kill <pid> (the whole process tree)",
    ),
    Divergence(
        name="ps aux",
        pattern=r"\bps\s+(?:aux|-ef)\b",
        replacement="procs",
    ),
    Divergence(
        name="jq",
        pattern=r"(?:^|[|;&(]|\s)jq\s",
        replacement="yq -p json (go-yq)",
    ),
    Divergence(
        name="flock",
        pattern=r"(?:^|[;&|(]|\s)flock\s",
        replacement="a lock taken in Python (filelock), which Windows also honors",
    ),
    Divergence(
        name="nc -z",
        pattern=r"\bnc\s+(?:-\w+\s+)*-\w*z",
        replacement="mainboard proc wait --port <host:port>",
    ),
    Divergence(
        name="sleep loop",
        pattern=r"\bwhile\b[^\n]*;\s*do\s+sleep\b",
        replacement="mainboard proc wait --file <path> or --port <host:port>",
    ),
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
                detail=f"{len(places)} uses: {_listed(places)}",
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
        """Every place each form appears, as `file:line`, keyed by the form's name."""
        compiled = [(rule.name, re.compile(rule.pattern)) for rule in RULES]
        places: dict[str, list[str]] = {}
        for repo, path in self.scripts():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError, OSError:
                continue
            where = path.relative_to(repo.path).as_posix()
            prefix = "" if repo.name == "." else f"{repo.name}/"
            for number, line in enumerate(text.splitlines(), start=1):
                if line.lstrip().startswith(("#", "//", "REM ", "::")):
                    continue
                for name, pattern in compiled:
                    if pattern.search(line):
                        places.setdefault(name, []).append(f"{prefix}{where}:{number}")
        return places

    def scripts(self) -> list[tuple[Repo, Path]]:
        """Every tracked script file of every owned repository, small enough to be one."""
        found: list[tuple[Repo, Path]] = []
        for repo in self.repos:
            listed = repo.git.run("ls-files", "-z").stdout.split("\0")
            prefix = "" if repo.name == "." else f"{repo.name}/"
            for relative in listed:
                if (
                    not relative
                    or self.excluded.match_file(prefix + relative)
                    or not any(
                        fnmatch(relative, pattern) or fnmatch(relative.rsplit("/", 1)[-1], pattern)
                        for pattern in _SCRIPTS
                    )
                ):
                    continue
                path = repo.path / relative
                if path.is_file() and not path.is_symlink() and path.stat().st_size <= _LARGEST:
                    found.append((repo, path))
        return found


def _listed(places: Sequence[str]) -> str:
    """The first few places, and how many more there are."""
    rest = len(places) - _NAMED
    return ", ".join(places[:_NAMED]) + (f" and {rest} more" if rest > 0 else "")
