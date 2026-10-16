"""`gate`: the engine as a lint step, safe fixes and findings for the files one pass reads.

`mainboard lint` runs it once per owner, from the owner's directory, with the pass's files. The
whole owner is analyzed (a rule reads every file's facts), but only the named files are repaired
and reported, so a commit is red for what it touched alone. Fixes are the engine's `safe` ones: a
review fix renames or deletes public API, so `fix --unsafe` stays a person's call. The reach rules
run too: the names the workspace's other repositories import count as used (consumers.py), and a
fix that would remove one is refused.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from ...core.errors import MissionError
from ...log import logger
from ..catalog import Rule
from ..fixing import Fixpoint
from ..lint import Lint, app, load_catalog

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    from ..catalog import Repair
    from ..facts import Facts

_SAFE = ("safe",)


class Pass(Lint):
    """The engine over one owner, repairing only the files a lint pass reads.

    files: the pass's files, relative to the owner's root.
    """

    def __init__(
        self,
        root: Path,
        rules: Iterable[type[Rule]],
        files: Collection[str],
        *,
        memory: str,
        fixing: bool,
    ) -> None:
        super().__init__(root, rules, memory=memory, fixing=fixing)
        self.files = frozenset(files)
        self.refused: list[str] = []
        self.changes: list[str] = []

    def repairs(self, db: Facts, applicability: tuple[str, ...]) -> dict[str, list[Repair]]:
        """The engine's repairs that edit nothing outside the pass's files."""
        return {
            rule: [group for group in groups if all(edit.file() in self.files for edit in group)]
            for rule, groups in super().repairs(db, applicability).items()
        }

    def repair(self, root: Path) -> Facts:
        """Fix the pass's files to the engine's fixpoint and read the facts again; the fixes it
        refused (they would remove a name another repository imports) are kept in `refused`, and
        a line for each fix it kept (`path:line RULE change`) in `changes`."""
        fixer = Fixpoint(self, root, _SAFE)
        for line in fixer.run(write=True):
            logger.debug(line)
        self.refused = list(fixer.refusals)
        self.changes = fixer.changes
        return self.facts(root)

    def reported(self, db: Facts) -> list[str]:
        """`path:line:column: RULE message` for each finding in the pass's files, in order."""
        found = sorted(
            (path, line or 0, column or 0, f"{rule} {message}")
            for rule, rows in self.findings(db).items()
            for path, line, column, message in rows
            if path in self.files
        )
        return [f"{path}:{line}:{column}: {text}" for path, line, column, text in found]


def _selected(skip: Collection[str]) -> list[type[Rule]]:
    load_catalog()
    unknown = sorted(set(skip) - set(Rule.defined))
    if unknown:
        raise MissionError(f"no rule {unknown[0]!r} to skip; see `mainboard analyze catalog`")
    return [rule for name, rule in Rule.defined.items() if name not in skip]


def _named(root: Path, files: Collection[Path]) -> set[str]:
    """The pass's files as paths relative to `root`, refusing one outside it."""
    names = set()
    for file in files:
        try:
            names.add(Path(os.path.normpath(root / file)).relative_to(root).as_posix())
        except ValueError:
            raise MissionError(f"{file} is outside the project at {root}") from None
    return names


@app.command
def gate(
    *files: Path,
    root: Path = Path(),
    fix: bool = False,
    skip: tuple[str, ...] = (),
    memory: str = "2GB",
) -> int:
    """Report the findings in FILES, repairing their safe ones first with `--fix`.

    The exit is 0 when nothing remains in FILES and 1 otherwise, so the step is red exactly for
    the files the pass reads. Each finding prints as `path:line:column: RULE message`; a fix
    refused because another repository imports what it removes is told after them. With `--fix`,
    a line `path:line RULE change` for each fix kept comes first, so whoever reads the output
    knows what the engine did.

    Args:
        files: the pass's files, relative to ROOT.
        root: the project to analyze, the working directory by default (`mainboard lint` runs
            the step inside each owner).
        fix: apply the engine's safe fixes to FILES; a review fix is never applied.
        skip: rule ids left out of the pass.
        memory: a DuckDB memory limit such as `4GB`.
    """
    root = root.resolve()
    names = _named(root, files)
    with Pass(root, _selected(skip), names, memory=memory, fixing=fix) as lint:
        db = lint.facts(root)
        if fix and any(lint.repairs(db, _SAFE).values()):
            db = lint.repair(root)
        found = lint.reported(db)
        refused = lint.refused
        changes = lint.changes
    if found or changes:
        print("\n".join([*changes, *found, *(f"fix refused, {reason}" for reason in refused)]))
    return int(bool(found))
