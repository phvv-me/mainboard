"""The selected Xcode, used through xcrun with bounded tool output, never launching the IDE."""

import json
import os
import plistlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

from plumbum import local
from pydantic import ValidationError

from ..core.errors import MissionError
from ..core.execution.tool import Runner, ToolRunner
from ..engines.compile.generated import GeneratedFiles
from .bundle.core import AppleBundle, BundleChecks, BundleReport
from .bundle.launch import Launch
from .evidence import BuildReport, BuildSummary, Summary, TestReport, TestTree
from .fonts import FontRenderer
from .xcode import Xcodebuild
from .xcrun import simulators, xcrun

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .evidence import TestNode
    from .fonts import FontPreview, Preview
    from .xcode import XcodeRun

# What `xcode-select -p` reads, without a process per command.
_XCODE_SELECT = "/var/db/xcode_select_link"
_COMMAND_LINE_TOOLS = "/Library/Developer/CommandLineTools"
_APPLICATIONS = Path("/Applications")
# How many of the slowest test cases a report keeps.
_SLOWEST = 10


def _bundle_version(app: Path) -> tuple[int, ...]:
    """An app's `CFBundleShortVersionString` as numbers (`27.0.1` is (27, 0, 1)); () unreadable."""
    try:
        info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    except OSError, plistlib.InvalidFileException:
        return ()
    version = str(info.get("CFBundleShortVersionString", ""))
    return tuple(int(part) for part in version.split(".") if part.isdigit())


def _selector(text: str) -> str:
    """A test selector without its slashes at either end or a trailing `()`."""
    return text.strip("/").removesuffix("()")


class Apple:
    """Use xcrun's selected Xcode (including DEVELOPER_DIR), with bounded tool output."""

    @staticmethod
    def _bundle(path: Path) -> Path:
        """The exact result bundle `path` names, or the newest one directly inside it."""
        path = path.expanduser().resolve()
        if not path.is_dir():
            raise MissionError(f"no result bundle or test-log directory at {path}")
        if path.suffix == ".xcresult":
            return path
        candidates = [item for item in path.glob("*.xcresult") if item.is_dir()]
        if not candidates:
            raise MissionError(f"no .xcresult bundles directly inside {path}")
        return max(candidates, key=lambda item: (item.stat().st_mtime_ns, item.name))

    @staticmethod
    def _missing(
        selectors: Iterable[str], *, skipped: Iterable[str], identifiers: Sequence[str]
    ) -> list[str]:
        """The exact selectors no test case answered, unless a skip covers them.

        A selector short of `target/class/case` or holding a wildcard selects no one case.
        """
        skips = [_selector(skip) for skip in skipped]
        return [
            selector
            for selector in selectors
            if len(_selector(selector).split("/")) >= 3
            and not any(char in selector for char in "*?[")
            and not any(
                _selector(selector) == skip or _selector(selector).startswith(skip + "/")
                for skip in skips
            )
            and not any(
                found == _selector(selector) or found.endswith("/" + _selector(selector))
                for found in identifiers
            )
        ]

    @staticmethod
    def retain(report: BuildReport | TestReport) -> Path:
        """Keep an atomic JSON sidecar beside its exact bundle; never replace old evidence."""
        suffix = ".tests.json" if isinstance(report, TestReport) else ".build.json"
        target = report.bundle.with_suffix(suffix)
        with GeneratedFiles(directory=target.parent).locked() as writer:
            if target.exists() or target.is_symlink():
                raise MissionError(f"refusing to overwrite Apple evidence at {target}")
            writer.write(target, json.dumps(report.document(), indent=2) + "\n")
        return target

    @staticmethod
    def select_xcode() -> None:
        """Point `DEVELOPER_DIR` at the newest Xcode.app when Command Line Tools are chosen.

        It holds for this process and every one it starts. The Command Line Tools lack the
        SwiftParser host libraries the Swift rules read and `xcodebuild` itself, which made 444
        commands in a week prefix `DEVELOPER_DIR` by hand (2026-10-23). A `DEVELOPER_DIR` the
        caller set, another platform, or a choice of Xcode is left alone.
        """
        if "DEVELOPER_DIR" in os.environ or sys.platform != "darwin":
            return
        try:
            selected = os.readlink(_XCODE_SELECT)
        except OSError:
            return
        if selected.startswith(_COMMAND_LINE_TOOLS) and (
            installed := sorted(_APPLICATIONS.glob("Xcode*.app"), key=_bundle_version)
        ):
            # plumbum, which launches `mb run`'s commands, copied the environment at import.
            os.environ["DEVELOPER_DIR"] = local.env["DEVELOPER_DIR"] = str(
                installed[-1] / "Contents" / "Developer"
            )

    def __init__(self, runner: Runner | None = None) -> None:
        self.runner = runner if runner is not None else ToolRunner(timeout=30)

    def _result(self, section: str, bundle: Path) -> str:
        """One section of a bundle's test results, as xcresulttool prints it."""
        return xcrun(
            self.runner,
            *("xcresulttool", "get", "test-results", section, "--path", str(bundle), "--compact"),
        )

    def bundle(self, path: Path, checks: BundleChecks | None = None) -> BundleReport:
        """The built bundle at `path` read and judged against `checks`, its code never run."""
        return AppleBundle(self.runner).inspect(path, checks or BundleChecks())

    def diagnostics(self, path: Path) -> BuildReport:
        """Read the build action, including bundles that failed before tests could start.

        A successful build action says nothing about subsequent test results. Incremental
        builds report only the diagnostics Xcode retained for that action.
        """
        bundle = Apple._bundle(path)
        payload = xcrun(
            self.runner, "xcresulttool", "get", "build-results", "--path", str(bundle), "--compact"
        )
        try:
            summary = BuildSummary.model_validate_json(payload)
        except ValidationError as fault:
            raise MissionError(
                f"invalid xcresulttool build evidence for {bundle}: {fault}"
            ) from fault
        return BuildReport(bundle=bundle, summary=summary)

    def execute(self, project: Path, run: XcodeRun, bundle: Path | None = None) -> int:
        """Build or test `project` as `run` asks with the selected Xcode; its exit status."""
        return Xcodebuild(self.runner).run(project, run, bundle)

    def font_previews(
        self, paths: Iterable[Path], preview: Preview, output: Path
    ) -> list[FontPreview]:
        """Render the first face of each font with Core Text into the fresh `output`."""
        return FontRenderer(self.runner).render(paths, preview, output)

    def launch(
        self, path: Path, *, simulator: str, os_version: str = "", timeout: float = 180
    ) -> dict[str, str]:
        """Install a built app and open it in one explicitly selected simulator."""
        launched = Launch(self.runner, timeout=timeout)
        return launched.run(path, simulator=simulator, os_version=os_version)

    def results(
        self, path: Path, *, only_testing: Iterable[str] = (), skip_testing: Iterable[str] = ()
    ) -> TestReport:
        """Read an exact bundle, or the newest direct child of a Logs/Test directory.

        Never fall back to an older successful bundle when the newest one is incomplete.
        Timings are Xcode's test-case durations, not sums of parallel parameterized runs.
        """
        bundle = Apple._bundle(path)
        summary_text, tests_text = self._result("summary", bundle), self._result("tests", bundle)
        try:
            summary, tree = (
                Summary.model_validate_json(summary_text),
                TestTree.model_validate_json(tests_text),
            )
        except ValidationError as fault:
            raise MissionError(
                f"invalid xcresulttool test evidence for {bundle}: {fault}"
            ) from fault
        cases: list[TestNode] = [case for node in tree.nodes for case in node.cases()]
        identifiers = [_selector(unquote(urlsplit(case.identifier).path)) for case in cases]
        timed = sorted(
            (case for case in cases if case.seconds is not None),
            key=lambda case: case.seconds or 0,
            reverse=True,
        )
        return TestReport(
            bundle=bundle,
            summary=summary,
            missing_selectors=Apple._missing(
                only_testing, skipped=skip_testing, identifiers=identifiers
            ),
            slowest=timed[:_SLOWEST],
        )

    def simulators(self) -> list[dict[str, str]]:
        """List available devices without booting, creating or erasing a simulator."""
        return simulators(self.runner)

    def toolchain(self) -> dict[str, str]:
        """The selected Xcode and Swift versions, and where swift and xcresulttool live."""
        return {
            "xcode": xcrun(self.runner, "xcodebuild", "-version").strip(),
            "swift": xcrun(self.runner, "swift", "--version").strip(),
            "swift_path": xcrun(self.runner, "--find", "swift").strip(),
            "xcresulttool_path": xcrun(self.runner, "--find", "xcresulttool").strip(),
        }
