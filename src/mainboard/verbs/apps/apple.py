from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from ...apple import Apple, BuildReport, BundleChecks, Preview, TestReport, XcodeRun
from ...core.errors import MissionError
from ...log import logger
from ..base import Verbs
from ..options import Output

if TYPE_CHECKING:
    from collections.abc import Callable

# Where each run's fresh result bundle and preview directory are kept unless asked otherwise.
_RESULTS = Path(".build/Results")

# The verbs' defaults: compact output, a bundle checked against its own shape alone.
_COMPACT = Output()
_UNCHECKED = BundleChecks()


class AppleVerbs(Verbs):
    """Xcode, simulators, built bundles and the evidence a build or test leaves."""

    @staticmethod
    def _evidence(apple: Apple, bundle: Path, run: XcodeRun) -> bool:
        """Print what a run's bundle holds, its build and its tests, answering whether all passed.

        Each report is also kept as a JSON sidecar beside the bundle. A report that cannot be
        read is said and counts as a failure.
        """
        readers = [("build", apple.diagnostics)] + (
            [("tests", lambda path: AppleVerbs._tests(apple, path, run))]
            if run.action == "test"
            else []
        )
        return all([AppleVerbs._kept(apple, bundle, kind, read) for kind, read in readers])

    @staticmethod
    def _kept(apple: Apple, bundle: Path, kind: str, read: _Reader) -> bool:
        """Read one report from `bundle`, keep and print it; whether it passed."""
        try:
            report = read(bundle)
        except (MissionError, OSError) as fault:
            return AppleVerbs._unavailable(kind, fault)
        _COMPACT.print_record(
            {"report": str(apple.retain(report))}, title=f"Apple {kind} evidence"
        )
        _COMPACT.print_record(report.document(), title=f"Apple {kind}")
        return report.successful

    @staticmethod
    def _tests(apple: Apple, path: Path, run: XcodeRun) -> TestReport:
        """The test results in `path`, judged against the selectors `run` named."""
        return apple.results(path, only_testing=run.only_testing, skip_testing=run.skip_testing)

    @staticmethod
    def _unavailable(kind: str, fault: Exception) -> bool:
        """Say a run's report could not be read, which counts as a failure."""
        logger.info(f"Apple {kind} evidence unavailable: {fault}")
        return False

    def apple_bundle(
        self, path: Path, *, checks: BundleChecks = _UNCHECKED, output: Output = _COMPACT
    ) -> int:
        """Inspect a built app's metadata, resources and every packaged Mach-O architecture.

        No app code is executed. nm/otool inspect all architectures and their complete
        output stays in the report. Missing/truncated native evidence fails. Inspection
        is bounded to 10,000 entries, 128 Mach-O files and 16 slices per fat header.
        Internal symlinks are deduplicated;
        links escaping the bundle fail. This does not prove source provenance, optimized-out
        static dependencies, signing, installation or rendering.

        Args:
            path: a built app bundle, including its embedded products.
        """
        report = Apple().bundle(path, checks)
        output.print_record(report.document(), title="Apple bundle")
        return 0 if report.successful else 1

    def apple_diagnostics(self, path: Path, *, output: Output = _COMPACT) -> int:
        """Read compiler errors, warnings, source locations and the build action's status.

        Exit 1 for failed or unknown builds; malformed or incomplete evidence is an error.
        Warnings remain visible without failing a successful build. Works when compilation
        failed before tests started. Build success does not establish test success; use
        apple results for that. Incremental builds are not a whole-project warning audit.

        Args:
            path: an exact .xcresult bundle or a directory (newest direct bundle).
        """
        report = Apple().diagnostics(path)
        output.print_record(report.document(), title="Apple build diagnostics")
        return 0 if report.summary.successful else 1

    def apple_execute(self, project: Path, *, run: XcodeRun, results: Path = _RESULTS) -> int:
        """Run Xcode on a project/workspace/package; retain fresh evidence and its verdict.

        Native tool output is streamed; typed JSON sidecars retain readable results even
        when native execution fails. No shell, signing changes,
        simulator erasure or global Xcode selection is performed. Committed Swift
        package versions are required. Tests must report actual passing cases.

        Args:
            project: an .xcworkspace, .xcodeproj or Swift package directory.
            results: parent for a new, unique .xcresult bundle on every invocation.
        """
        apple = Apple()
        if run.resolving:
            return apple.execute(project, run)
        bundle = results / f"{run.action}-{uuid4().hex}.xcresult"
        _COMPACT.print_record({"bundle": str(bundle.resolve())}, title="Apple execution")
        code = apple.execute(project, run, bundle)
        # The evidence is kept whatever Xcode answered, so a failed run still says why.
        passed = AppleVerbs._evidence(apple, bundle, run)
        return code or (0 if passed else 1)

    def apple_font_previews(
        self,
        paths: tuple[Path, ...],
        *,
        preview: Preview,
        results: Path = _RESULTS,
        output: Output = _COMPACT,
    ) -> None:
        """Render transparent PNG previews with installed Apple text APIs, without an app.

        Each source can be a font file or a directory of fonts. Collections use the first
        face. Every glyph must exist in that face; system font fallback is rejected.
        A fresh output directory is retained on each run. No global fonts are installed.

        Args:
            paths: font files or directories.
            results: parent for a fresh font-preview directory.
        """
        directory = results.expanduser().resolve() / ("fonts-" + uuid4().hex)
        rendered = Apple().font_previews(paths, preview, directory)
        output.print_rows([item.document() for item in rendered], title="Font previews")

    def apple_launch(
        self,
        path: Path,
        *,
        simulator: str,
        os_version: str = "",
        timeout: float = 180,
        output: Output = _COMPACT,
    ) -> None:
        """Install a built app and show its selected simulator in Xcode 27 Device Hub.

        Uses Device Hub from the selected Xcode. No hidden build, device creation or data erasure.
        Simulator is an exact available name or UDID; OS version can disambiguate names.
        Timeout bounds the complete boot/install/launch/window sequence in seconds.
        """
        launched = Apple().launch(
            path, simulator=simulator, os_version=os_version, timeout=timeout
        )
        output.print_record(launched, title="Launched app")

    def apple_results(self, path: Path, *, output: Output = _COMPACT) -> int:
        """Read xcresult test counts, failures, devices and the ten slowest test cases.

        Exit 1 for failed, unknown or empty runs. Skips and expected failures stay visible.
        Build failures without test evidence are errors, never a successful zero-test run.
        Supports Swift Testing and XCTest through the installed xcresulttool.

        Args:
            path: an exact .xcresult bundle or a Logs/Test directory (newest direct bundle).
        """
        report = Apple().results(path)
        output.print_record(report.document(), title="Apple tests")
        return 0 if report.summary.successful else 1

    def apple_simulators(self, *, output: Output = _COMPACT) -> None:
        """List available simulator names, runtime identifiers, UDIDs and current states.

        Uses the selected Xcode without changing device state. A device's availability
        does not establish eligibility for a particular project's scheme or deployment target.
        """
        output.print_rows(Apple().simulators(), title="Apple simulators")

    def apple_toolchain(self, *, output: Output = _COMPACT) -> None:
        """Report Xcode and Swift selected by DEVELOPER_DIR or xcode-select; change neither."""
        output.print_record(Apple().toolchain(), title="Apple toolchain")


# How a run's evidence is read from its bundle: its build, or its tests.
type _Reader = Callable[[Path], BuildReport | TestReport]
