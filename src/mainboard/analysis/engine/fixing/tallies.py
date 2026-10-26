# Counting findings: the fixer judges a fix by how a file's findings of each rule moved.

from collections import Counter
from collections.abc import Mapping, Set
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from ..mirror import is_under

if TYPE_CHECKING:
    from ..apply import Plan, Spot


def typed_growth(
    carried: Mapping[str, Counter[tuple[int, str]]],
    *,
    current: Mapping[str, Counter[tuple[int, str]]],
) -> set[str]:
    """The files holding more type errors of some kind than they did."""
    found = set()
    for path, counts in current.items():
        had = _by_rule(carried.get(path, Counter()))
        if any(r.startswith("ty:") and n > had[r] for r, n in _by_rule(counts).items()):
            found.add(path)
    return found


def split(spots: Counter[Spot]) -> tuple[int, int]:
    """Lint findings and type diagnostics."""
    types = sum(count for (*_, rule), count in spots.items() if rule.startswith("ty:"))
    return spots.total() - types, types


def _by_rule(places: Counter[tuple[int, str]]) -> Counter[str]:
    """Each rule's findings, summed over lines."""
    totals: Counter[str] = Counter()
    for (_, rule), count in places.items():
        totals[rule] += count
    return totals


def by_file(spots: Counter[Spot]) -> dict[str, Counter[tuple[int, str]]]:
    """Each file's findings by line and rule."""
    files: dict[str, Counter[tuple[int, str]]] = {}
    for (path, line, _, rule), count in spots.items():
        files.setdefault(path, Counter())[line, rule] += count
    return files


def holding(files: Set[str]) -> set[str]:
    """`files` and the directories holding them.

    A finding about a directory's contents (ALL-FILE0003) stands on the directory.
    """
    return {
        *files,
        *(parent.as_posix() for path in files for parent in PurePosixPath(path).parents),
    }


def _places(spots: Counter[Spot], files: set[str]) -> Counter[tuple[str, int, str]]:
    """The findings in `files` or on the directories holding them, by path, line and rule."""
    held = holding(files)
    found: Counter[tuple[str, int, str]] = Counter()
    for (path, line, _, rule), count in spots.items():
        if is_under(path, files) or path in held:
            found[path, line, rule] += count
    return found


def grown(
    files: Set[str],
    *,
    carried: Mapping[str, Counter[tuple[int, str]]],
    current: Mapping[str, Counter[tuple[int, str]]],
) -> dict[str, list[str]]:
    """The rules holding more findings in each of `files` than at its proven state."""
    found = {
        path: sorted(
            rule
            for rule, count in _by_rule(current.get(path, Counter())).items()
            if count > _by_rule(carried.get(path, Counter()))[rule]
        )
        for path in files
    }
    return {path: rules for path, rules in found.items() if rules}


def progressed(plan: Plan, before: Counter[Spot], *, after: Counter[Spot]) -> bool:
    """Whether some finding left the files `plan` touched, from where its edits moved it."""
    touched = {plan.moved(path) for path in plan.touched()}
    return bool(_places(plan.relocate(before), touched) - _places(after, touched))


def cleared(
    files: Set[str],
    carried: Mapping[str, Counter[tuple[int, str]]],
    *,
    current: Mapping[str, Counter[tuple[int, str]]],
) -> int:
    """How many findings of `files` left since their proven state, else 0.

    It counts only when none of the files holds more findings of a rule than then.
    """
    left = 0
    for path in files:
        was, now = carried.get(path, Counter()), current.get(path, Counter())
        had = Counter({rule: 0 for _, rule in was}) + _by_rule(was)
        if any(count > had[rule] for rule, count in _by_rule(now).items()):
            return 0
        left += (was - now).total()
    return left
