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
from ..engine.fixing import Fixpoint
from ..engine.lint import Lint, load_catalog, selected
from ..findings import Report, Seen

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence

    from ..catalog import Repair
    from ..engine.facts import Facts

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

    def repair(self, root: Path) -> Facts:
        """Fix the pass's files to the engine's fixpoint and read the facts again.

        The fixes it refused (they would remove a name another repository imports) are kept in
        `refused`, and a line for each fix it kept (`path:line RULE change`) in `changes`.
        """
        fixer = Fixpoint(self, root, _SAFE)
        for line in fixer.run(write=True):
            logger.debug(line)
        self.refused = list(fixer.refusals)
        self.changes = fixer.changes
        return self.facts(root)

    def repairs(self, db: Facts, applicability: Sequence[str]) -> dict[str, list[Repair]]:
        """The engine's repairs that edit nothing outside the pass's files."""
        return {
            rule: [group for group in groups if all(edit.file() in self.files for edit in group)]
            for rule, groups in super().repairs(db, applicability).items()
        }

    def reported(self, db: Facts) -> Report:
        """The findings in the pass's files."""
        return Report(
            found={
                rule: [
                    (path, line or 0, column or 0, message)
                    for path, line, column, message in rows
                    if path in self.files
                ]
                for rule, rows in self.findings(db).items()
            }
        )


def _selected(skip: Collection[str], rule: tuple[str, ...] = ()) -> list[type[Rule]]:
    load_catalog()
    unknown = sorted(set(skip) - set(Rule.defined))
    if unknown:
        raise MissionError(f"no rule {unknown[0]!r} to skip; see `mainboard refactor catalog`")
    unknown = sorted(set(rule) - set(Rule.defined))
    if unknown:
        raise MissionError(f"no rule {unknown[0]!r} to select; see `mainboard refactor catalog`")
    return [entry for entry in selected(rule) if entry.id not in skip]


def _named(root: Path, files: Collection[Path]) -> set[str]:
    """The pass's files as paths relative to `root`, refusing one outside it."""
    names = set()
    for file in files:
        try:
            names.add(Path(os.path.normpath(root / file)).relative_to(root).as_posix())
        except ValueError:
            raise MissionError(f"{file} is outside the project at {root}") from None
    return names


def gate(
    *files: Path,
    root: Path = Path(),
    fix: bool = False,
    skip: tuple[str, ...] = (),
    rule: tuple[str, ...] = (),
    memory: str = "2GB",
    new: bool = False,
) -> int:
    """Report the findings in FILES, repairing their safe ones first with `--fix`.

    The exit is 0 when nothing remains in FILES and 1 otherwise, so the step is red exactly for
    the files the pass reads. Findings print grouped, as `mainboard refactor check` prints them; a
    fix refused because another repository imports what it removes is told after them. With
    `--fix`, a line `path:line RULE change` for each fix kept comes first, so whoever reads the
    output knows what the engine did.

    Args:
        files: the pass's files, relative to ROOT.
        root: the project to analyze, the working directory by default (`mainboard lint` runs
            the step inside each owner).
        fix: apply the engine's safe fixes to FILES; a review fix is never applied.
        skip: rule ids left out of the pass.
        rule: only these rule ids; the whole catalog when empty.
        memory: a DuckDB memory limit such as `4GB`.
        new: report only the findings FILES gained since they were last checked; every pass
            records what it found either way.
    """
    root = root.resolve()
    names = _named(root, files)
    with Pass(root, _selected(skip, rule), names, memory=memory, fixing=fix) as lint:
        facts = lint.facts(root)
        if fix and any(lint.repairs(facts, _SAFE).values()):
            facts = lint.repair(root)
        found = lint.reported(facts)
        fresh = Seen(root).fresh(found, names)
        found = fresh if new else found
        refused = lint.refused
        changes = lint.changes
    shown = [found.grouped] if found.total else []
    if shown or changes:
        logger.result(
            "\n".join([*changes, *shown, *(f"fix refused, {reason}" for reason in refused)])
        )
    return int(bool(found.total))
