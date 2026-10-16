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
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

from .apply import Mirror, Plan, Spot, under
from .checks import Syntax, Types, imported

if TYPE_CHECKING:
    from .catalog import Repair
    from .lint import Lint


class Applied(FrozenModel):
    """A fix applied in the mirror: where its finding stood in the file as that pass began, the
    rule's words for what it changed, and the files it changed."""

    rule: str
    path: str
    line: int
    change: str
    files: frozenset[str]

    def __str__(self) -> str:
        return f"{self.path}:{self.line} {self.rule} {self.change}"


class Components:
    """Files changed since their last proven state, joined when one fix changed several."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def join(self, names: Iterable[str]) -> None:
        first, *rest = names
        self.parent.setdefault(first, first)
        for name in rest:
            self.parent.setdefault(name, name)
            self.parent[self._root(name)] = self._root(first)

    def groups(self) -> list[set[str]]:
        found: dict[str, set[str]] = {}
        for name in self.parent:
            found.setdefault(self._root(name), set()).add(name)
        return list(found.values())

    def around(self, names: set[str]) -> set[str]:
        """Every file in a component holding one of `names`."""
        roots = {self._root(name) for name in names if name in self.parent}
        return {name for name in self.parent if self._root(name) in roots}

    def drop(self, names: set[str]) -> None:
        """Forget whole components."""
        self.parent = {name: root for name, root in self.parent.items() if name not in names}

    def _root(self, name: str) -> str:
        while self.parent[name] != name:
            self.parent[name] = self.parent[self.parent[name]]
            name = self.parent[name]
        return name


class Fixpoint:
    """Fixes one project with `lint`'s rules until nothing fixable is left."""

    def __init__(
        self, lint: Lint, root: Path, applicability: tuple[str, ...], *, limit: int = 100
    ) -> None:
        self.lint, self.root, self.applicability, self.limit = lint, root, applicability, limit
        self.syntax, self.types = Syntax(), Types()
        self.refusals: dict[str, None] = {}

    def run(self, *, write: bool) -> Iterator[str]:
        """Fix pass after pass, describing each; with `write`, write the kept changes."""
        self.proven = Mirror(self.root)
        self.work = Mirror(self.proven.path)
        self.pending = Components()
        self.held: set[str] = set()
        self.kept: list[set[str]] = []
        self.applied: list[Applied] = []
        self.proven_fixes: list[Applied] = []
        try:
            self._extract()
            self.carried = self.start = self.current
            first = _split(self.current)
            for number in range(1, self.limit + 1):
                started = time.perf_counter()
                plan, groups, broken = self._plan()
                head = f"pass {number}"
                if broken:
                    head += f" (withheld {len(broken)} files a fix would break: {broken[0]})"
                if not groups:
                    if not self.pending.parent:
                        yield f"{head}: nothing left to fix"
                        break
                    waiting = set(self.pending.parent)
                    yield f"{head}: nothing left to fix; {self._abandon(waiting)}"
                    continue
                before = self.current
                changed = self.work.apply(plan)
                self._extract()
                self.carried = plan.relocate(self.carried)
                for group in groups:
                    self._join(plan, group, changed)
                head += f": {len(groups)} fixes ({_by_rule(groups)}), {plan.deferred} deferred"
                if not _progress(plan, before, self.current):
                    touched = {plan.moved(path) for path in plan.touched()}
                    yield f"{head}; no finding left the files it touched; {self._abandon(touched)}"
                    continue
                fixed = self._settle({plan.moved(path) for path in plan.touched()})
                yield (
                    f"{head}; {fixed} findings fixed and kept, {len(self.pending.parent)} files "
                    f"pending ({time.perf_counter() - started:.1f} s)"
                )
            else:
                if waiting := set(self.pending.parent):
                    yield f"stopped at {self.limit} passes; {self._abandon(waiting)}"
            yield from self._audit()
            yield from (f"refused: {reason}" for reason in self.refusals)
            if self.proven_fixes:
                yield f"{len(self.proven_fixes)} fixes kept:"
                yield from self.changes
            last, changed = _split(self.current), set().union(*self.kept)
            yield (
                f"findings {first[0]} -> {last[0]}, type diagnostics {first[1]} -> {last[1]}; "
                f"{len(changed)} files changed; fixes withheld from {len(self.held)} files"
            )
            if write and changed:
                self.proven.publish(self.root, changed)
                yield f"wrote {len(changed)} files"
        finally:
            self.work.close()
            self.proven.close()

    @property
    def changes(self) -> list[str]:
        """One line per kept fix, `path:line RULE change`, in file order. A line is where the
        finding stood when its pass began."""
        return [
            str(fix)
            for fix in sorted(self.proven_fixes, key=lambda fix: (fix.path, fix.line, fix.rule))
        ]

    def _extract(self) -> None:
        """The work mirror's facts, findings and type diagnostics."""
        self.db = self.lint.facts(self.work.path)
        self.current = Counter(
            (path, line or 1, column or 0, rule)
            for path, line, column, rule, *_ in self.lint.flat(self.db)
        ) + self.types.diagnostics(self.work.path)

    def _plan(self) -> tuple[Plan, list[tuple[str, Repair]], list[str]]:
        """One pass of every fix that overlaps no other and touches no withheld file, with the
        rule of each fix taken; a file the pass would leave unparsable is withheld and the pass
        planned again, as is a fix that removes a name another repository imports (consumers.py),
        which only that fix is refused."""
        repairs = self.lint.repairs(self.db, self.applicability)
        broken: list[str] = []
        refused: set[int] = set()
        while True:
            plan, groups = Plan(self.work.path), []
            for rule, candidates in repairs.items():
                for group in candidates:
                    if (
                        id(group) not in refused
                        and not any(under(edit.file(), self.held) for edit in group)
                        and plan.add(group)
                    ):
                        groups.append((rule, group))
            if newly := self.syntax.broken(plan):
                broken += newly
                self.held |= {line.partition(":")[0] for line in newly}
            elif lost := self.lint.consumers.broken(plan):
                refused |= self._refuse([group for _, group in groups], lost)
            else:
                return plan, groups, broken

    def _refuse(self, groups: list[Repair], lost: list[str]) -> set[int]:
        """The fixes among `groups` that cause the `lost` names: each one alone, else every fix
        in the files they name. The reasons are kept in `refusals`."""
        named = {line.partition(":")[0] for line in lost}
        near = [group for group in groups if named & {edit.file() for edit in group}]
        alone = []
        for group in near:
            trial = Plan(self.work.path)
            trial.add(group)
            if self.lint.consumers.broken(trial):
                alone.append(group)
        self.refusals.update(dict.fromkeys(lost))
        return {id(group) for group in alone or near}

    def _join(self, plan: Plan, group: tuple[str, Repair], changed: set[str]) -> None:
        """Join the files one fix changed (a renamed or removed directory by its files), and note
        the fix until its files are proven or returned."""
        rule, repair = group
        names = {edit.file() for edit in repair}
        names |= {plan.moved(name) for name in names}
        files = names | {name for name in changed if under(name, names)}
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

    def _settle(self, touched: set[str]) -> int:
        """Keep every pending component the passes have proven; return the findings it fixed."""
        carried, current = _by_file(self.carried), _by_file(self.current)
        blocked = self._blocked(carried, current, touched)
        proven: set[str] = set()
        fixed = 0
        for files in self.pending.groups():
            if not files & blocked and (gone := _proven(files, carried, current)):
                self.kept.append(files)
                proven |= files
                fixed += gone
        self.proven.adopt(self.work.path, proven)
        self.proven_fixes += [fix for fix in self.applied if fix.files & proven]
        self.applied = [fix for fix in self.applied if not fix.files & proven]
        self._restart(proven)
        return fixed

    def _blocked(
        self,
        carried: dict[str, Counter[tuple[int, str]]],
        current: dict[str, Counter[tuple[int, str]]],
        touched: set[str],
    ) -> set[str]:
        """Pending files a new type error points at: those its file imports (a fix there may have
        moved or renamed what it imports), else, for a file no pending fix changed, every file
        this pass touched."""
        pending, blocked = set(self.pending.parent), set()
        for path in _typed_growth(carried, current):
            suspects = imported(self.work.path, path) & pending
            blocked |= suspects or (set() if path in pending else touched)
        return blocked

    def _audit(self) -> Iterator[str]:
        """Undo kept components until no file holds more type errors than at the start: a
        component kept beside pending ones may have leaned on one that was then undone."""
        while grown := _typed_growth(_by_file(self.start), _by_file(self.current)):
            pointed = grown | {name for path in grown for name in imported(self.work.path, path)}
            culprits = set().union(*(files for files in self.kept if files & pointed))
            if not culprits:
                yield f"audit: {len(grown)} files hold new type errors no kept fix explains"
                return
            self.kept = [files for files in self.kept if not files & culprits]
            self.proven_fixes = [fix for fix in self.proven_fixes if not fix.files & culprits]
            self.proven.adopt(self.root, culprits)
            self.work.adopt(self.root, culprits)
            self.held |= culprits
            self._extract()
            yield (
                f"audit: undid the fixes of {len(culprits)} files behind new type errors in "
                f"{len(grown)} files ({', '.join(sorted(grown)[:3])})"
            )

    def _abandon(self, files: set[str]) -> str:
        """Return the components holding `files` to their proven state, withholding their fixes
        from then on."""
        files = self.pending.around(files) or files
        self.applied = [fix for fix in self.applied if not fix.files & files]
        self.work.adopt(self.proven.path, files)
        self.held |= files
        self._extract()
        self._restart(files)
        sample = ", ".join(sorted(files)[:3])
        return f"returned {len(files)} files to their proven state, fixes withheld ({sample})"

    def _restart(self, files: set[str]) -> None:
        """Count `files` from their present findings, as a proven state."""
        self.carried = Counter(
            {spot: n for spot, n in self.carried.items() if spot[0] not in files}
        ) + Counter({spot: n for spot, n in self.current.items() if spot[0] in files})
        self.pending.drop(files)


def _proven(
    files: set[str],
    carried: dict[str, Counter[tuple[int, str]]],
    current: dict[str, Counter[tuple[int, str]]],
) -> int:
    """How many findings of `files` left since their proven state, when none of them holds more
    findings of a rule than then; 0 otherwise."""
    gone = 0
    for path in files:
        was, now = carried.get(path, Counter()), current.get(path, Counter())
        had = Counter({rule: 0 for _, rule in was}) + _rules(was)
        if any(count > had[rule] for rule, count in _rules(now).items()):
            return 0
        gone += (was - now).total()
    return gone


def _typed_growth(
    carried: dict[str, Counter[tuple[int, str]]], current: dict[str, Counter[tuple[int, str]]]
) -> set[str]:
    """The files holding more type errors of some kind than they did."""
    grown = set()
    for path, places in current.items():
        had = _rules(carried.get(path, Counter()))
        if any(r.startswith("ty:") and n > had[r] for r, n in _rules(places).items()):
            grown.add(path)
    return grown


def _split(spots: Counter[Spot]) -> tuple[int, int]:
    """Lint findings and type diagnostics."""
    types = sum(count for (*_, rule), count in spots.items() if rule.startswith("ty:"))
    return spots.total() - types, types


def _rules(places: Counter[tuple[int, str]]) -> Counter[str]:
    totals: Counter[str] = Counter()
    for (_, rule), count in places.items():
        totals[rule] += count
    return totals


def _by_file(spots: Counter[Spot]) -> dict[str, Counter[tuple[int, str]]]:
    """Each file's findings by line and rule."""
    files: dict[str, Counter[tuple[int, str]]] = {}
    for (path, line, _, rule), count in spots.items():
        files.setdefault(path, Counter())[line, rule] += count
    return files


def _progress(plan: Plan, before: Counter[Spot], after: Counter[Spot]) -> bool:
    """Whether some finding left the files `plan` touched, from where its edits moved it."""
    touched = {plan.moved(path) for path in plan.touched()}
    return bool(_places(plan.relocate(before), touched) - _places(after, touched))


def _by_rule(groups: list[tuple[str, Repair]]) -> str:
    return ", ".join(f"{rule} {n}" for rule, n in Counter(rule for rule, _ in groups).items())


def _places(spots: Counter[Spot], files: set[str]) -> Counter[tuple[str, int, str]]:
    places: Counter[tuple[str, int, str]] = Counter()
    for (path, line, _, rule), count in spots.items():
        if under(path, files):
            places[path, line, rule] += count
    return places
