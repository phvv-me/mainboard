"""Fixing a project to its fixpoint: every fixable finding worked on, pass after pass, keeping only
proven changes.

The work happens in a mirror of the project's files. Each pass plans every fix that overlaps no
other (the rest wait a pass), leaving out the fixes of a file the pass would leave unparsable
(checks.py), then writes the plan into the mirror and extracts the facts again, with ty's type
diagnostics as findings of their own. Files one fix changes together stay together: they form a
component, and every component is judged on its own after each pass. A component is proven when
none of its files holds more findings of a rule (a type error included) than at its last proven
state and some of those findings are gone; it is then kept. A component may hold more findings on
the way (deleting dead code strands what only it used, for the next pass to delete), so it waits
for later passes. When the passes stop short of proving it (nothing is left to fix, or a pass fixed
nothing), the component returns to its proven state and its files' fixes are withheld from then
on, which bounds the run. A new type error in a file outside the pending components holds back
the components of the files it imports until it is gone. The project itself is written once, from
the kept changes. Each kept fix is told in a line of its own (`path:line RULE change`) that an
agent can read to pick up from it.
"""

import time
from collections import Counter
from collections.abc import Iterator, Sequence, Set
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

from ..apply import Plan, Spot
from ..checks import Types, imported
from ..mirror import Mirror, is_under
from .components import Applied, Components
from .planning import Planner
from .proof import Proof
from .tallies import by_file, cleared, grown, holding, progressed, split, typed_growth

if TYPE_CHECKING:
    from ...catalog import Repair
    from ..facts import Facts
    from ..lint import Lint


class Fixpoint:
    """Fixes one project with `lint`'s rules until nothing fixable is left."""

    applied: list[Applied]
    carried: Counter[Spot]
    current: Counter[Spot]
    facts: Facts
    held: set[str]
    kept: list[set[str]]
    pending: Components
    proven: Mirror
    proven_fixes: list[Applied]
    start: Counter[Spot]
    work: Mirror

    @property
    def changes(self) -> list[str]:
        """One line per kept fix, `path:line RULE change`, in file order.

        A line is where the finding stood when its pass began.
        """
        return [
            str(fix)
            for fix in sorted(self.proven_fixes, key=lambda fix: (fix.path, fix.line, fix.rule))
        ]

    @property
    def refusals(self) -> list[str]:
        """Why each fix removing a name another repository imports was refused, once run."""
        return list(self.planner.refusals)

    def __init__(
        self,
        lint: Lint,
        root: Path,
        applicability: tuple[str, ...],
        *,
        limit: int = 100,
        proof: Sequence[str] = (),
    ) -> None:
        """proof: the project's proof command (`ProjectConfig.proof`), none for no proof run."""
        self.lint, self.root, self.applicability, self.limit = lint, root, applicability, limit
        self.proof = Proof(root, proof)
        self.types = Types()

    def _abandon(self, files: set[str]) -> str:
        """Return the components holding `files` to their proven state.

        Their fixes are withheld from then on.
        """
        files = self.pending.around(files) or files
        growth = grown(files, carried=by_file(self.carried), current=by_file(self.current))
        self.applied = [fix for fix in self.applied if not fix.files & files]
        self.work.adopt(self.proven.path, files)
        self.held |= files
        self._extract()
        self._restart(files)
        why = "; ".join(f"{path}: {', '.join(rules)}" for path, rules in sorted(growth.items()))
        return f"returned {len(files)} files to their proven state, fixes withheld" + (
            f"; more findings of a rule in {why}" if why else ""
        )

    def _applied(self, plan: Plan, groups: Sequence[tuple[str, Repair]]) -> Counter[Spot]:
        """Write `plan` into the work mirror and note its fixes; answer the findings before it."""
        before = self.current
        changed = self.work.apply(plan)
        self._extract()
        self.carried = plan.relocate(self.carried)
        for group in groups:
            self._join(plan, group, changed)
        return before

    def _audit(self) -> Iterator[str]:
        """Undo kept components until no file holds more type errors than at the start.

        A component kept beside pending ones may have leaned on one that was then undone.
        """
        while growth := typed_growth(by_file(self.start), current=by_file(self.current)):
            pointed = growth | {name for path in growth for name in imported(self.work.path, path)}
            culprits = set().union(*(files for files in self.kept if files & pointed))
            if not culprits:
                yield f"audit: {len(growth)} files hold new type errors no kept fix explains"
                return
            self.kept = [files for files in self.kept if not files & culprits]
            self.proven_fixes = [fix for fix in self.proven_fixes if not fix.files & culprits]
            self.proven.adopt(self.root, culprits)
            self.work.adopt(self.root, culprits)
            self.held |= culprits
            self._extract()
            yield (
                f"audit: undid the fixes of {len(culprits)} files behind new type errors in "
                f"{len(growth)} files ({', '.join(sorted(growth)[:3])})"
            )

    def _blocked(
        self,
        carried: dict[str, Counter[tuple[int, str]]],
        *,
        current: dict[str, Counter[tuple[int, str]]],
        touched: set[str],
    ) -> set[str]:
        """The pending files a new type error points at.

        Those are the files its file imports (a fix there may have moved or renamed what it
        imports), else, for a file no pending fix changed, every file this pass touched.
        """
        pending, blocked = set(self.pending.parent), set()
        for path in typed_growth(carried, current=current):
            suspects = imported(self.work.path, path) & pending
            blocked |= suspects or (set() if path in pending else touched)
        return blocked

    def _closing(self, first: tuple[int, int], *, write: bool) -> Iterator[str]:
        """The kept fixes and the counts they moved; with `write`, the changed files written."""
        if self.proven_fixes:
            yield f"{len(self.proven_fixes)} fixes kept:"
            yield from self.changes
        last, changed = split(self.current), set().union(*self.kept)
        yield (
            f"findings {first[0]} -> {last[0]}, type diagnostics {first[1]} -> {last[1]}; "
            f"{len(changed)} files changed; fixes withheld from {len(self.held)} files"
        )
        if write and changed:
            self.proven.publish(self.root, changed)
            yield f"wrote {len(changed)} files"

    def _extract(self) -> None:
        """The work mirror's facts, findings and type diagnostics."""
        self.facts = self.lint.facts(self.work.path)
        self.current = Counter(
            (path, line or 1, column or 0, rule)
            for path, line, column, rule, *_ in self.lint.flat(self.facts)
        ) + self.types.diagnostics(self.work.path)

    def _fixed(self) -> Iterator[str]:
        """Every pass, until one finds nothing left to fix or the limit is reached."""
        for number in range(1, self.limit + 1):
            line, done = self._pass(number)
            yield line
            if done:
                return
        if waiting := set(self.pending.parent):
            yield f"stopped at {self.limit} passes; {self._abandon(waiting)}"

    def _join(self, plan: Plan, group: tuple[str, Repair], changed: Set[str]) -> None:
        """Join the files one fix changed and note the fix until they are proven or returned.

        A renamed or removed directory counts by its files.
        """
        rule, repair = group
        names = {edit.file() for edit in repair}
        names |= {plan.moved(name) for name in names}
        files = names | {name for name in changed if is_under(name, names)}
        self.pending.join(files)
        self.applied.append(
            Applied(
                rule=rule,
                path=repair.path,
                line=repair.line,
                change=repair.change,
                files=frozenset(files),
            )
        )

    def _judged(
        self, plan: Plan, groups: Sequence[tuple[str, Repair]], *, head: str, started: float
    ) -> str:
        """Apply one pass's fixes, keep what they proved, and say what came of them."""
        before = self._applied(plan, groups)
        rules = ", ".join(f"{rule} {n}" for rule, n in Counter(rule for rule, _ in groups).items())
        head += f": {len(groups)} fixes ({rules}), {plan.deferred} deferred"
        touched = {plan.moved(path) for path in plan.touched()}
        if not progressed(plan, before, after=self.current):
            return f"{head}; no finding left the files it touched; {self._abandon(touched)}"
        fixed, spent = self._settle(touched), time.perf_counter() - started
        pending = len(self.pending.parent)
        return f"{head}; {fixed} findings fixed and kept, {pending} files pending ({spent:.1f} s)"

    def _pass(self, number: int) -> tuple[str, bool]:
        """Plan, apply and judge one pass: what it did, and whether nothing is left to fix."""
        started = time.perf_counter()
        plan, groups, broken = self.planner.plan(self.facts, work=self.work.path)
        head = f"pass {number}" + (
            f" (withheld {len(broken)} files a fix would break: {broken[0]})" if broken else ""
        )
        if groups:
            return self._judged(plan, groups, head=head, started=started), False
        if not self.pending.parent:
            return f"{head}: nothing left to fix", True
        return f"{head}: nothing left to fix; {self._abandon(set(self.pending.parent))}", False

    def _prove(self) -> Iterator[str]:
        """Run the project's proof over the kept changes, undoing each component that fails it.

        A proof that fails on the project as it stands judges no fix, so nothing is undone.
        """
        proof, proven = self.proof, self.proven.path
        if not proof.command or not self.kept or proof.passes(self.kept, proven=proven):
            yield from (["proof: passed"] if proof.command and self.kept else [])
            return
        if not proof.passes([], proven=proven):
            yield f"proof: fails on the project as it stands, so it judges no fix: {proof.said}"
            return
        while self.kept and not proof.passes(self.kept, proven=proven):
            culprit = proof.culprit(self.kept, proven=proven)
            self.kept.remove(culprit)
            self.proven_fixes = [fix for fix in self.proven_fixes if not fix.files & culprit]
            self.proven.adopt(self.root, culprit)
            self.held |= culprit
            yield (
                f"proof: undid the fixes of {len(culprit)} files that fail it "
                f"({', '.join(sorted(culprit)[:3])}): {proof.said}"
            )
        yield "proof: passed"

    def _restart(self, files: set[str]) -> None:
        """Take the present findings of `files` and the directories holding them as proven."""
        counted = holding(files)
        self.carried = Counter(
            {spot: n for spot, n in self.carried.items() if spot[0] not in counted}
        ) + Counter({spot: n for spot, n in self.current.items() if spot[0] in counted})
        self.pending.drop(files)

    def _settle(self, touched: set[str]) -> int:
        """Keep every pending component the passes have proven; return the findings it fixed."""
        carried, current = by_file(self.carried), by_file(self.current)
        blocked = self._blocked(carried, current=current, touched=touched)
        proven: set[str] = set()
        fixed = 0
        for files in self.pending.groups():
            if not files & blocked and (gone := cleared(holding(files), carried, current=current)):
                self.kept.append(files)
                proven |= files
                fixed += gone
        self.proven.adopt(self.work.path, proven)
        self.proven_fixes += [fix for fix in self.applied if fix.files & proven]
        self.applied = [fix for fix in self.applied if not fix.files & proven]
        self._restart(proven)
        return fixed

    def run(self, *, write: bool) -> Iterator[str]:
        """Fix pass after pass, describing each; with `write`, write the kept changes."""
        with (
            closing(Mirror(self.root)) as self.proven,
            closing(Mirror(self.proven.path)) as self.work,
        ):
            self.pending, self.held, self.kept = Components(), set(), []
            self.planner = Planner(self.lint, self.applicability, held=self.held)
            self.applied, self.proven_fixes = [], []
            self._extract()
            self.carried = self.start = self.current
            first = split(self.current)
            yield from self._fixed()
            yield from self._audit()
            yield from self._prove()
            yield from (f"refused: {reason}" for reason in self.planner.refusals)
            yield from self._closing(first, write=write)
