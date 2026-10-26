"""`check`, `compare`, `fix` and `test`: the engine's verbs over one project, and over every
rule's examples."""

import difflib
import json
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs the frozen mcmr CLI with fixed argv since=2026-10-13
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ConfigDict

from ....log import logger
from ...catalog import Example, Rule
from ...constants import FIXABLE
from ...findings import Report, Seen
from ..facts import listed, vendored
from ..fixing import Fixpoint
from .core import Lint, selected


@Parameter(name="*")
class Selection(FrozenModel):
    """Which rules run, and the fact base's budget."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    rule: tuple[str, ...] = ()
    """rule ids, every SQL rule (every fixable one, for a fix) when empty."""
    memory: str = "2GB"
    """a DuckDB memory limit such as `4GB`."""
    family: tuple[str, ...] = ()
    """mcmr families to derive in SQL though not yet verified (`CallFact`)."""

    def lint(self, root: Path, *, fixing: bool = True) -> Lint:
        """These rules, configured for the project at `root`."""
        return Lint(
            root, selected(self.rule), memory=self.memory, families=self.family, fixing=fixing
        )


# Every rule, at the default budget.
_EVERY = Selection()


def check(
    root: Path, *, selection: Selection = _EVERY, counts: bool = False, new: bool = False
) -> None:
    """Every finding of the selected rules (all when none), grouped by rule, and the stage times.

    Each rule's message prints once with `_` holes, then each file's lines, a line's values
    filling its holes in order: `path 12[name|3]` reads as the message with `name` and `3` in it.

    Args:
        root: the project to check.
        counts: one line per rule with its number of findings instead, most first.
        new: only the findings each file gained since it was last checked; every check records
            what it found either way.
    """
    start = time.perf_counter()
    with selection.lint(root, fixing=False) as lint:
        base = lint.facts(root)
        extracted = time.perf_counter()
        report = Report(found=lint.findings(base))
    fresh = Seen(root).fresh(report, listed(root))
    report = fresh if new else report
    ran = time.perf_counter()
    logger.result(report.counts if counts else report.grouped)
    logger.result(
        f"{report.total} findings; facts {extracted - start:.2f} s, rules {ran - extracted:.3f} s"
    )


def compare(root: Path, *, selection: Selection = _EVERY, baseline: Path | None = None) -> None:
    """Each selected rule's findings against mcmr's (frozen): equal, missing, extra.

    Args:
        root: the project both engines analyze.
        baseline: a stored full `mcmr check --format json` report; without it mcmr runs per rule.
    """
    report = json.loads(baseline.read_text()) if baseline else None
    with selection.lint(root, fixing=False) as lint:
        found = lint.findings(lint.facts(root))
    for spec in lint.rules:
        old = _baseline_findings(report, spec.id) if report else _old_findings(root, spec.id)
        skipped = vendored(root, {finding[0] for finding in old})
        old = {finding for finding in old if finding[0] not in skipped}
        new = set(found.get(spec.id, []))
        logger.result(
            f"{spec.id}\t{old == new}\t{len(old)}\t{len(new)}\tmissing {len(old - new)}\t"
            f"extra {len(new - old)}"
        )
        for sample in sorted(old ^ new, key=str)[:3]:
            logger.result(f"    {'missing' if sample in old else 'extra'} {sample}")


def fix(
    root: Path,
    *,
    selection: Selection = _EVERY,
    write: bool = False,
    unsafe: bool = False,
    passes: int = 100,
) -> None:
    """Fix every fixable finding, pass after pass, until nothing fixable is left.

    Each state is proven on a mirror and only the last proven one is written (with `--write`).

    Args:
        root: the project to fix.
        write: write the last proven state into the project; without it the run only reports.
        unsafe: include fixes marked for review, not only safe ones.
        passes: the most passes before stopping.
    """
    applicability = FIXABLE["review" if unsafe else "safe"]
    with selection.lint(root) as lint:
        fixer = Fixpoint(lint, root, applicability, limit=passes, proof=lint.project.proof)
        for report in fixer.run(write=write):
            logger.result(report)


def _baseline_findings(report: Mapping, rule: str) -> set[tuple]:
    """A rule's findings from a stored full mcmr report."""
    return {
        (f["span"]["path"], f["span"]["start_line"], f["span"]["start_column"], f["message"])
        for failure in report["failures"]
        if failure["rule"] == rule
        for f in failure["findings"]
    }


def _old_findings(root: Path, rule: str) -> set[tuple]:
    """mcmr's findings for one rule, run live.

    A rule mcmr's deterministic lanes do not run (an external one) has none.
    """
    run = subprocess.run(
        [
            "mcmr",
            "check",
            str(root),
            "--select",
            rule,
            "--format",
            "json",
            "--limit",
            "1000000",
            "--deterministic",
            "--no-contextual",
            "--no-external",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return _baseline_findings(json.loads(run.stdout), rule)
    except json.JSONDecodeError:
        return set()


_SKIPPED = "skipped: it reads facts no SQL family derives"


def test(*, selection: Selection = _EVERY) -> None:
    """Run the selected rules' examples, every rule's when none is selected.

    Each `bad` must be flagged, the rule's fix must turn it into `good`, and `good` must not be
    flagged.
    """
    failed = 0
    for spec in selected(selection.rule):
        for number, example in enumerate(spec.examples, 1):
            verdict = _trial(spec, example, selection)
            failed += verdict not in {"pass", _SKIPPED}
            logger.result(f"{spec.id} example {number}: {verdict}")
    logger.result(f"{failed} examples failed")
    if failed:
        raise SystemExit(1)


def _cleaned(spec: type[Rule], example: Example, lint: Lint, root: Path, *, path: Path) -> str:
    """Whether `good`, written where `bad` was, goes unflagged.

    A `None` good is the file's absence, which a rule over the file inventory must not flag.
    """
    if example.good is None:
        path.unlink()
    else:
        path.write_text(example.good, encoding="utf-8")
    return "good is flagged" if lint.findings(lint.facts(root)).get(spec.id) else "pass"


def _example_path(language: str | None, text: str) -> str:
    if text.lstrip().startswith("["):
        return "pyproject.toml"
    if language in (None, "python"):
        return "test_module.py" if "def test_" in text else "module.py"
    return {
        "rust": "lib.rs",
        "cuda": "kernel.cu",
        "typescript": "module.ts",
        "swift": "View.swift",
    }.get(language, f"module.{language}")


def _fixed(example: Example, lint: Lint, root: Path, *, path: Path) -> str:
    """What the rule's fix made of `bad`: empty when it gives `good`, else the verdict.

    A `None` good is the file's removal, which ends the trial there.
    """
    report = "\n".join(Fixpoint(lint, root, FIXABLE["review"]).run(write=True))
    if example.good is None:
        return "pass" if not path.exists() else f"the fix keeps the file\n{report}"
    fixed_text = path.read_text(encoding="utf-8")
    if fixed_text == example.good:
        return ""
    diff = difflib.unified_diff(
        example.good.splitlines(keepends=True),
        fixed_text.splitlines(keepends=True),
        "good",
        "fixed",
    )
    return f"the fix does not give good\n{''.join(diff)}{report}"


def _trial(spec: type[Rule], example: Example, selection: Selection) -> str:
    """One example's verdict: `pass`, or what went wrong.

    The snippet lives in `example.path`, by default a TOML snippet in `pyproject.toml` and code in
    a module of the rule's language (a test module when it defines a test). A Python rule's project
    always holds a module, so it holds Python whatever the snippet is.
    """
    with tempfile.TemporaryDirectory(prefix="mb-trial-") as scratch:
        root = Path(scratch)
        language = spec().language
        path = root / (example.path or _example_path(language, example.bad))
        for name, text in {**example.context, str(path.relative_to(root)): example.bad}.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(text)
        if language == "python" and not any(root.rglob("*.py")):
            (root / "anchor.py").write_text("")
        with Lint(root, [spec], memory=selection.memory, families=selection.family) as lint:
            return _verdict(spec, example, lint, root, path=path)


def _verdict(spec: type[Rule], example: Example, lint: Lint, root: Path, *, path: Path) -> str:
    """Whether `bad` is flagged, the fix turns it into `good`, and `good` is not flagged."""
    base = lint.facts(root)
    if not lint.rules:
        return _SKIPPED
    if not lint.findings(base).get(spec.id):
        return "bad is not flagged"
    if example.good == "":
        return "pass"
    refused = _fixed(example, lint, root, path=path) if spec.fixes() != "none" else ""
    return refused or _cleaned(spec, example, lint, root, path=path)
