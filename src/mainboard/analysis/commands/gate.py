"""`gate`: the engine as a lint step, safe fixes and findings for the files one pass reads.

`mb code lint` runs it once per owner, from the owner's directory, with the pass's files. The
whole owner is analyzed (a rule reads every file's facts), but only the named files are repaired
and reported, so a commit is red for what it touched alone. Fixes are the engine's `safe` ones: a
review fix renames or deletes public API, so `fix --unsafe` stays a person's call. The reach rules
run too: the names the workspace's other repositories import count as used (consumers.py), and a
fix that would remove one is refused.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ConfigDict

from ...core.errors import MissionError
from ...log import logger
from ..catalog import Rule
from ..engine.fixing import Fixpoint
from ..engine.lint import Lint, load_catalog, selected
from ..engine.lint.seen import Seen
from ..findings import Report

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
        return self.whole(db).within(self.files)

    def whole(self, db: Facts) -> Report:
        """Every finding of the owner, which `Seen` nets moves over."""
        return Report(
            found={
                rule: [
                    (path, line or 0, column or 0, message) for path, line, column, message in rows
                ]
                for rule, rows in self.findings(db).items()
            }
        )


def _named(root: Path, files: Collection[Path]) -> set[str]:
    """The pass's files as paths relative to `root`, refusing one outside it."""
    names = set()
    for file in files:
        try:
            names.add(Path(os.path.normpath(root / file)).relative_to(root).as_posix())
        except ValueError:
            raise MissionError(f"{file} is outside the project at {root}") from None
    return names


@Parameter(name="*")
class Options(FrozenModel):
    """How one pass runs."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    fix: bool = False
    """apply the engine's safe fixes to FILES; a review fix is never applied."""
    skip: tuple[str, ...] = ()
    """rule ids left out of the pass."""
    rule: tuple[str, ...] = ()
    """only these rule ids; the whole catalog when empty."""
    memory: str = "2GB"
    """a DuckDB memory limit such as `4GB`."""
    new: bool = False
    """report only the findings FILES gained since they were last checked; every pass records
    what it found either way."""

    def lint(self, root: Path, files: Collection[str]) -> Pass:
        """The pass over `files` of the project at `root`, with these rules and budget."""
        return Pass(root, self.selected(), files, memory=self.memory, fixing=self.fix)

    def selected(self) -> list[type[Rule]]:
        """The rules of `rule` (the whole catalog when empty) without those of `skip`."""
        load_catalog()
        unknown = sorted(set(self.skip) - set(Rule.defined))
        if unknown:
            raise MissionError(
                f"no rule {unknown[0]!r} to skip; `mb code rules explain` lists them"
            )
        return [entry for entry in selected(self.rule) if entry.id not in self.skip]


_PLAIN = Options()


def gate(*files: Path, root: Path = Path(), options: Options = _PLAIN) -> int:
    """Report the findings in FILES, repairing their safe ones first with `--fix`.

    The exit is 0 when nothing remains in FILES and 1 otherwise, so the step is red exactly for
    the files the pass reads. Findings print grouped, as `mb code rules check` prints them; a
    fix refused because another repository imports what it removes is told after them. With
    `--fix`, a line `path:line RULE change` for each fix kept comes first, so whoever reads the
    output knows what the engine did.

    Args:
        files: the pass's files, relative to ROOT.
        root: the project to analyze, the working directory by default (`mb code lint` runs
            the step inside each owner).
        options: how the pass runs.
    """
    root = root.resolve()
    names = _named(root, files)
    with options.lint(root, names) as lint:
        facts = lint.facts(root)
        if options.fix and any(lint.repairs(facts, _SAFE).values()):
            facts = lint.repair(root)
        found = lint.whole(facts)
        fresh = Seen(root).fresh(found, names)
        found = fresh if options.new else found.within(names)
        refused = lint.refused
        changes = lint.changes
    shown = [found.grouped] if found.total else []
    if shown or changes:
        logger.result(
            "\n".join([*changes, *shown, *(f"fix refused, {reason}" for reason in refused)])
        )
    return int(bool(found.total))
