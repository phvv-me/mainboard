# The proofs a fix set passes to be kept: the project's own command, run over a copy of the project
# with some kept changes laid in (`Proof`), and the test ids the kept fixes leave collected
# (`Collection`).

import os
import sys
import tempfile
from collections.abc import Sequence, Set
from contextlib import closing
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

from patos import FrozenModel

from ....proc import Processes
from ..mirror import Mirror
from .components import Applied, Remap

# How long one run of a project's proof may take before it counts as failed, and how many of
# its last output lines a failure of a proof or of a listing is told with.
_PROOF_SECONDS = 3600
_SAID = 5

# How long one listing of test ids may take, and the statuses pytest ends a listing with (ok, some
# tests failed to import, interrupted by an import error, no tests).
_LISTING_SECONDS = 600
_LISTED = frozenset({0, 1, 2, 5})

# pytest's default names for the files it collects tests from, and how many ids a reason shows.
_PATTERNS = ("test_*.py", "*_test.py")
_SHOWN = 3


class Proof:
    """The project's proof command and what its last run said.

    root: the project as it stands, which every run starts from and the proof is told of.
    command: the proof command (`ProjectConfig.proof`), `{root}` standing for the copy; empty
        for a project without one.
    """

    def __init__(self, root: Path, command: Sequence[str]) -> None:
        self.root, self.command, self.said = root.resolve(), tuple(command), ""

    def culprit(self, components: Sequence[set[str]], *, proven: Path) -> set[str]:
        """The first of `components` whose changes, with those before it, fail the proof.

        The proof passes with none of them and fails with all of them, so a binary search over
        the prefixes finds it in a logarithm of their count.
        """
        passing, failing = 0, len(components)
        while failing - passing > 1:
            middle = (passing + failing) // 2
            if self.passes(components[:middle], proven=proven):
                passing = middle
            else:
                failing = middle
        return components[failing - 1]

    def passes(self, components: Sequence[set[str]], *, proven: Path) -> bool:
        """Whether the proof passes on the project with the changes of `components` alone."""
        with closing(Mirror(self.root)) as state:
            state.adopt(proven, set().union(*components))
            command = [part.replace("{root}", str(state.path)) for part in self.command]
            log = state.top / "proof.log"
            with log.open("wb") as output:
                failed = Processes().timeout(
                    _PROOF_SECONDS,
                    command,
                    cwd=state.path,
                    env={
                        **os.environ,
                        "MB_FIX_MIRROR": str(state.path),
                        "MB_FIX_ROOT": str(self.root),
                    },
                    output=output,
                )
            self.said = " | ".join(log.read_text(errors="replace").strip().splitlines()[-_SAID:])
            return not failed


class Listing(FrozenModel):
    """The test ids pytest lists from some files of a tree, and the files it could not import.

    ids: node ids as pytest spells them, relative to the tree.
    broken: files whose import failed.
    fault: why pytest listed nothing, empty when it listed.
    """

    ids: frozenset[str] = frozenset()
    broken: frozenset[str] = frozenset()
    fault: str = ""

    @classmethod
    def of(cls, tree: Path, files: Sequence[str], *, root: Path) -> Listing:
        """What `pytest --collect-only` lists from `files` of `tree`, the fault when it does not.

        The project's own pytest settings apply, its source under `src` is imported from the tree,
        and the run is told of the project as the proof is (`MB_FIX_MIRROR`, `MB_FIX_ROOT`).
        """
        if not files:
            return cls()
        # A tree under a linked folder (macOS /var) would give ids with no path to pytest.
        status, lines = Listing._run(tree.resolve(), files, root=root)
        if status not in _LISTED:
            return cls(fault=f"pytest ended with {status}: {' | '.join(lines[-_SAID:])}")
        named = set(files)
        listed = (line for line in lines if "::" in line)
        errors = (
            line.removeprefix("ERROR ").partition(" - ")[0].strip()
            for line in lines
            if line.startswith("ERROR ")
        )
        return cls(
            ids=frozenset(line for line in listed if line.partition("::")[0] in named),
            broken=frozenset(error for error in errors if error in named),
        )

    @staticmethod
    def _environment(tree: Path, *, root: Path) -> dict[str, str]:
        """The environment of a listing: the tree's `src` leads the import path if there is one.

        No bytecode is written, so listing the project itself leaves nothing in it.
        """
        env = {**os.environ, "MB_FIX_MIRROR": str(tree), "MB_FIX_ROOT": str(root)}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if (source := tree / "src").is_dir():
            paths = filter(None, (str(source), os.environ.get("PYTHONPATH")))
            env["PYTHONPATH"] = os.pathsep.join(paths)
        return env

    @staticmethod
    def _run(tree: Path, files: Sequence[str], *, root: Path) -> tuple[int, list[str]]:
        """The status and the output lines of pytest listing `files`.

        `--verbosity=0` then `-q` makes the lines ids, whatever verbosity the addopts set.
        """
        command = [sys.executable, "-m", "pytest", "--collect-only", "--verbosity=0", "-q"]
        command += ["--continue-on-collection-errors", "-p", "no:cacheprovider"]
        command += ["--rootdir", str(tree), *files]
        env = Listing._environment(tree, root=root)
        with tempfile.TemporaryFile() as log:
            status = Processes().timeout(_LISTING_SECONDS, command, cwd=tree, env=env, output=log)
            log.seek(0)
            return status, log.read().decode(errors="replace").splitlines()


class Collection:
    """Whether the test ids the kept fixes leave are the ones their change lines account for.

    For the test files a kept component changed, pytest lists the ids in the project as it stands
    and in the proven mirror. The tests the component's fixes account for (`Remap`: a merge keeps
    each test as a case, a deletion removes the tests it names) must be the mirror's, no more and
    no fewer: the ids a change line names exactly, every other test by its function, whatever
    cases it has. A fix that changes a test file and says nothing about ids leaves every function
    as it was.

    root: the project as it stands.
    judged: how many components the last judgment found test files in.
    listed: how many test ids pytest listed from the mirror in it.
    said: why the last judgment listed nothing and so judged no fix, empty when it did.
    """

    @staticmethod
    def _accounted(ids: Set[str], remaps: Sequence[Remap]) -> tuple[set[str], set[str]]:
        """The ids `ids` become under the change lines of `remaps`, and the ids they gave."""
        expected, promised = set(ids), set[str]()
        for remap in remaps:
            promised = remap.apply(promised) | remap.promised(expected)
            expected = remap.apply(expected)
        return expected, promised

    @staticmethod
    def _existing(tree: Path, names: Set[str]) -> list[str]:
        """The files among `names` that exist in `tree`."""
        return sorted(name for name in names if (tree / name).is_file())

    @staticmethod
    def _tests(files: Set[str], remaps: Sequence[Remap]) -> set[str]:
        """The files among `files` that hold tests: named as pytest collects, or named by a fix."""
        collected = {
            name
            for name in files
            if any(fnmatch(PurePosixPath(name).name, pattern) for pattern in _PATTERNS)
        }
        return collected | (set().union(*(remap.files for remap in remaps)) & files)

    @staticmethod
    def _why(tests: Set[str], remaps: Sequence[Remap], *, before: Listing, after: Listing) -> str:
        """How the ids of `tests` after the fixes differ from the ones the fixes account for.

        An id a change line gave to a test must be listed exactly. Every other id counts by its
        test function, so a loop that became parametrized, a case id restyled and a test run per
        backend (`[asyncio]`) lose and gain no test.

        Empty when nothing differs, and when the project already could not import one of the files.
        """
        if before.broken & tests:
            return ""
        if broken := after.broken & tests:
            return f"no longer import: {', '.join(sorted(broken))}"
        was = {node for node in before.ids if node.partition("::")[0] in tests}
        expected, promised = Collection._accounted(was, remaps)
        found = {node for node in after.ids if node.partition("::")[0] in tests}
        kept = _functions(found)
        lost = (promised - found) | (_functions(expected - promised) - kept)
        told = (_told("lose", lost), _told("gain", kept - _functions(expected)))
        return "; ".join(part for part in told if part)

    def __init__(self, root: Path) -> None:
        self.root, self.judged, self.listed, self.said = root.resolve(), 0, 0, ""

    def _listed(self, wanted: Set[str], proven: Path) -> tuple[Listing, Listing]:
        """The ids pytest lists from the `wanted` files in the project and in `proven`."""
        before = Listing.of(self.root, Collection._existing(self.root, wanted), root=self.root)
        return before, Listing.of(proven, Collection._existing(proven, wanted), root=self.root)

    def broken(
        self, components: Sequence[Set[str]], *, fixes: Sequence[Applied], proven: Path
    ) -> list[tuple[Set[str], str]]:
        """Each of `components` whose test ids are not the ones its fixes account for, with how.

        Two listings serve every component: the test files of all of them, in the project and in
        `proven`, the mirror holding their changes. When pytest cannot list, `said` says why and
        no component is judged.
        """
        remaps = [
            [Remap.of(fix) for fix in fixes if not fix.files.isdisjoint(files)]
            for files in components
        ]
        tests = [Collection._tests(f, each) for f, each in zip(components, remaps, strict=True)]
        before, after = self._listed(set().union(*tests), proven)
        if fault := before.fault or after.fault:
            self.judged, self.listed, self.said = 0, 0, fault
            return []
        self.judged, self.listed, self.said = sum(map(bool, tests)), len(after.ids), ""
        judged = zip(components, tests, remaps, strict=True)
        return [
            (files, why)
            for files, held, each in judged
            if held and (why := Collection._why(held, each, before=before, after=after))
        ]

    def told(self, broken: Sequence[tuple[Set[str], str]]) -> list[str]:
        """The report lines of the last judgment, whose refused components are `broken`."""
        if self.said:
            return [f"collection: lists no test ids, so it judges no fix: {self.said}"]
        counted = (
            f"collection: {self.listed} test ids listed from the mirror, "
            f"{self.judged} components judged, {len(broken)} undone"
        )
        return ([counted] if self.judged else []) + [
            f"collection: undid the fixes of {len(files)} files "
            f"({', '.join(sorted(files)[:_SHOWN])}): {why}"
            for files, why in broken
        ]


def _functions(nodes: Set[str]) -> set[str]:
    """The test functions of the ids `nodes`, one for all the cases of a parametrized test."""
    return {Remap.function(node) for node in nodes}


def _told(verb: str, nodes: Set[str]) -> str:
    """`verb` and how many of the ids `nodes` there are, with the first ones."""
    shown = ", ".join(sorted(nodes)[:_SHOWN]) + (", ..." if len(nodes) > _SHOWN else "")
    return f"{verb} {len(nodes)} test ids ({shown})" if nodes else ""
