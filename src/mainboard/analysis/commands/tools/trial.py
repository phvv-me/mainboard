# Trying the fixes on a scratch copy: what they would change, and whether the project still holds
# together afterwards, with the project itself untouched.

import re
import shutil
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs git, ruff and the interpreter with fixed argv since=2026-10-25
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import TracebackType

from ....log import logger
from ...constants import FIXABLE
from ...engine.fixing import Fixpoint
from ...engine.lint.verbs import Selection
from ...engine.mirror import Mirror

# What a module import is given before it counts as hung, and how many run at once.
_IMPORT_SECONDS = 120
_IMPORTERS = 8
# Every rule, at the default budget.
_EVERY = Selection()
# A line of Ruff's concise output, its position dropped since the fixes move lines.
_CONCISE = re.compile(r"(?P<path>[^:]+):\d+:\d+: (?P<finding>.+)")


class Trial:
    """A git copy of one project the fixes run on, and the checks before and after them.

    The copy is a mirror (mirror.py), so the Ruff configuration the project extends stands where
    the project's own configuration finds it, and every check reads what it reads in place.
    """

    @staticmethod
    def _imported(copy: Path, name: str) -> str:
        """Whether the module at `name` imports in a process of its own, else its last error."""
        module = name.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
        probe = (
            "import importlib, sys; sys.path[:0] = [sys.argv[1] + '/src', sys.argv[1]]; "
            "importlib.import_module(sys.argv[2]); print('IMPORTED')"
        )
        try:
            ran = subprocess.run(
                [sys.executable, "-B", "-c", probe, str(copy), module.removeprefix("src.")],
                capture_output=True,
                text=True,
                timeout=_IMPORT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "hung"
        if "IMPORTED" in ran.stdout:
            return "ok"
        said = [line for line in ran.stderr.splitlines() if line.strip()]
        return said[-1][:160] if said else f"exit {ran.returncode}"

    def __enter__(self) -> Trial:
        return self

    def __exit__(self, *_: type[BaseException] | BaseException | TracebackType | None) -> None:
        self.mirror.close()

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.mirror = Mirror(self.root)
        self.copy = self.mirror.path
        self._git("init", "-q")
        self._git("add", "-A")
        self._git("-c", "user.name=trial", "-c", "user.email=trial@mb", "commit", "-qm", "base")

    def _git(self, *arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.copy), *arguments], capture_output=True, text=True, check=True
        ).stdout

    def changed(self) -> list[str]:
        """The Python files the fixes changed or created."""
        names = (
            self._git("diff", "--name-only").split()
            + self._git("ls-files", "--others", "--exclude-standard").split()
        )
        return sorted(name for name in names if name.endswith(".py"))

    def imports(self, names: list[str]) -> dict[str, str]:
        """Each of `names` that imported before the fixes and does not after, with why.

        The state before is a checkout of the copy's first commit beside it.
        """
        base = self.mirror.top / ".mb-trial-base"
        self._git("worktree", "add", "-q", "--detach", str(base), "HEAD")
        with ThreadPoolExecutor(_IMPORTERS) as pool:
            after = list(pool.map(lambda name: Trial._imported(self.copy, name), names))
            before = list(pool.map(lambda name: Trial._imported(base, name), names))
        return {
            name: now
            for name, was, now in zip(names, before, after, strict=True)
            if was == "ok" and now != "ok"
        }

    def patch(self) -> str:
        """The fixes as a patch `git apply` takes in the project."""
        self._git("add", "-A")
        return self._git("diff", "--cached")

    def pyflakes(self) -> Counter[str]:
        """Ruff's pyflakes and syntax findings in the copy, by file, code and message.

        The project's configuration applies (its exclusions, per-file ignores, target version).
        """
        ruff = shutil.which("ruff") or "ruff"
        ran = subprocess.run(
            [ruff, "check", "--select", "F,E9", "--output-format", "concise", "."],
            cwd=self.copy,
            capture_output=True,
            text=True,
            check=False,
        )
        return Counter(
            f"{found['path']}: {found['finding']}"
            for found in map(_CONCISE.fullmatch, ran.stdout.splitlines())
            if found
        )

    def run(self, *, selection: Selection, unsafe: bool, passes: int) -> list[str]:
        """The fixer's report over the copy, its fixes written there.

        The fixer settles the imports a fix leaves unused or out of order and formats what it
        wrote, so only what the fixes break counts against them.
        """
        applicability = FIXABLE["review" if unsafe else "safe"]
        with selection.lint(self.copy) as lint:
            return list(Fixpoint(lint, self.copy, applicability, limit=passes).run(write=True))


def trial(
    root: Path = Path(),
    *,
    selection: Selection = _EVERY,
    unsafe: bool = False,
    passes: int = 100,
    patch: Path | None = None,
) -> int:
    """Fix a scratch copy of ROOT and report what the fixes change, ROOT itself untouched.

    The report holds the fixer's own lines, the files changed, the pyflakes and syntax findings
    the fixes add or remove, and every changed module that imported before and no longer does
    (each imported in a process of its own). The exit is 1 when the fixes add a pyflakes finding
    or break an import.

    Args:
        root: the project to try the fixes on.
        unsafe: include fixes marked for review, not only safe ones.
        passes: the most passes before stopping.
        patch: write the fixes as a patch here, to apply with `git apply`.
    """
    with Trial(root) as attempt:
        before = attempt.pyflakes()
        report = attempt.run(selection=selection, unsafe=unsafe, passes=passes)
        changed, after = attempt.changed(), attempt.pyflakes()
        broken = attempt.imports(changed)
        if patch is not None:
            patch.write_text(attempt.patch(), encoding="utf-8")
    added, removed = after - before, before - after
    logger.result(
        "\n".join(
            [
                *report,
                f"{len(changed)} Python files changed",
                f"pyflakes: {sum(added.values())} added, {sum(removed.values())} removed",
                *(f"  added {finding}" for finding in sorted(added)),
                *(f"  no longer imports: {name}: {why}" for name, why in sorted(broken.items())),
                *([f"patch written to {patch}"] if patch is not None else []),
            ]
        )
    )
    return int(bool(added or broken))
