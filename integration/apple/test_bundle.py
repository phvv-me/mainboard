"""Bundle policy consumes complete native evidence and never runs the inspected executable."""

import hashlib
import json
import plistlib
from collections.abc import Iterator, Sequence
from itertools import repeat
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.apple import Apple, BundleChecks
from mainboard.cli import build
from mainboard.core import Runner, ToolOutput


class BundleTool(Runner):
    def __init__(self, tool: str = "", result: ToolOutput | None = None) -> None:
        self.tool = tool
        self.result = result
        self.calls: list[tuple[str, ...]] = []

    def run(self, tool: str, args: Sequence[str], *, cwd: Path | None = None) -> ToolOutput:
        assert tool == "/usr/bin/xcrun" and cwd is None
        self.calls.append(tuple(args))
        if args[0] == self.tool and self.result is not None:
            return self.result
        if args[0] == "plutil":
            assert args[1:5] == ("-convert", "json", "-o", "-")
            return ToolOutput(stdout=json.dumps(plistlib.loads(Path(args[5]).read_bytes())))
        assert args[:4] in [("nm", "-arch", "all", "-u"), ("otool", "-arch", "all", "-L")]
        return ToolOutput(stdout=f"{args[4]} (arm64 and x86_64): libSystem U _swift_release\n")


@pytest.fixture(params=[False, True], ids=["iOS", "macOS"])
def app_bundle(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    bundle = tmp_path / "日本 ' $(literal).app"
    contents = bundle / "Contents" if request.param else bundle
    resources = contents / "Resources" if request.param else bundle
    executable = contents / "MacOS/Main" if request.param else bundle / "Main"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(bytes.fromhex("cffaedfe") + b"fixture")
    (bundle / "Alias").symlink_to(executable)
    embedded = contents / "PlugIns/Review.appex/Review"
    embedded.parent.mkdir(parents=True)
    embedded.write_bytes(bytes.fromhex("cafebabe00000002") + b"fixture")
    # A Java version number must not be mistaken for the number of Mach-O fat slices.
    (bundle / "Example.class").write_bytes(bytes.fromhex("cafebabe0000003d") + b"java")
    (contents / "Info.plist").write_bytes(
        plistlib.dumps(
            {
                "CFBundleExecutable": "Main",
                "CFBundleDisplayName": "Example",
                "CFBundleIcons": {"CFBundlePrimaryIcon": {"CFBundleIconName": "ExampleIcon"}},
                "NSMicrophoneUsageDescription": "Recognize a confirmed answer.",
            }
        )
    )
    for locale, name in [("en", "Example"), ("ja", "例")]:
        localized = resources / f"{locale}.lproj/InfoPlist.strings"
        localized.parent.mkdir(parents=True)
        localized.write_bytes(
            plistlib.dumps({"CFBundleDisplayName": name}, fmt=plistlib.FMT_BINARY)
        )
    return bundle


def test_bundle_inspects_all_binaries_preserves_evidence_and_deduplicates_internal_links(
    app_bundle: Path,
) -> None:
    tool = BundleTool()
    report = Apple(tool).bundle(
        app_bundle,
        BundleChecks(
            expect_info=(
                "CFBundleDisplayName=Example",
                "CFBundleIcons.CFBundlePrimaryIcon.CFBundleIconName=ExampleIcon",
            ),
            require_info=("NSMicrophoneUsageDescription",),
            expect_name=("en=Example", "ja=例"),
        ),
    )
    assert report.successful and report.violations == []
    assert report.localized_names == {"en": "Example", "ja": "例"}
    assert len(report.binaries) == 2 and report.file_count == 6
    assert sum(call[0] in {"nm", "otool"} for call in tool.calls) == 4
    for binary in report.binaries:
        assert binary.sha256 == hashlib.sha256((app_bundle / binary.path).read_bytes()).hexdigest()
        assert binary.symbols.stdout.endswith("U _swift_release\n")
        assert binary.libraries.stdout
    assert report.document()["successful"] is True


@pytest.mark.parametrize("relative", ["study.store", "nested/study.store", "PlugIns/Debug.xctest"])
def test_forbidden_resources_include_root_nested_files_and_directories(
    app_bundle: Path,
    relative: str,
) -> None:
    target = app_bundle / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix == ".xctest":
        target.mkdir()
    else:
        target.write_bytes(b"fixture")
    report = Apple(BundleTool()).bundle(
        app_bundle, BundleChecks(forbid_resource=("**/study.store", "**/*.xctest"))
    )
    assert not report.successful
    assert any(relative in finding for finding in report.violations)


def test_symbols_stderr_metadata_and_localized_policy_all_affect_the_verdict(
    app_bundle: Path,
) -> None:
    report = Apple(BundleTool("nm", ToolOutput(stderr="U _FMDatabase"))).bundle(
        app_bundle,
        BundleChecks(
            forbid_symbol=("fmdatabase",),
            expect_info=("CFBundleDisplayName=Wrong",),
            require_info=("Missing", "CFBundleIcons"),
            expect_name=("ja=Wrong",),
        ),
    )
    assert not report.successful
    assert len(report.violations) == 6
    assert report.binaries[0].symbols.stderr == "U _FMDatabase"


@pytest.mark.parametrize(
    ("tool", "result"),
    [
        ("nm", ToolOutput(returncode=1, stderr="unreadable")),
        ("otool", ToolOutput(truncated=True)),
        ("otool", ToolOutput()),
        ("plutil", ToolOutput(stdout="not JSON")),
        ("plutil", ToolOutput(stdout="[]")),
    ],
)
def test_incomplete_native_inspection_never_passes(
    app_bundle: Path, tool: str, result: ToolOutput
) -> None:
    with pytest.raises(MissionError):
        Apple(BundleTool(tool, result)).bundle(app_bundle)


@pytest.mark.parametrize(
    "policy",
    [
        {"forbid_symbol": ("[",)},
        {"forbid_symbol": ("",)},
        {"forbid_resource": ("/absolute",)},
        {"forbid_resource": ("../outside",)},
        {"expect_info": ("Name",)},
        {"expect_info": ("Name=A", "Name=B")},
        {"expect_name": ("../en=Name",)},
        {"require_info": ("bad..path",)},
    ],
)
def test_invalid_policy_stops_before_native_execution(
    app_bundle: Path,
    policy: dict[str, tuple[str, ...]],
) -> None:
    tool = BundleTool()
    with pytest.raises(MissionError):
        Apple(tool).bundle(app_bundle, BundleChecks(**policy))
    assert tool.calls == []


def test_external_symlinks_are_rejected_before_native_execution(
    app_bundle: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.write_bytes(b"untouched")
    (app_bundle / "Escape").symlink_to(outside)
    tool = BundleTool()
    with pytest.raises(MissionError, match="escapes"):
        Apple(tool).bundle(app_bundle)
    assert tool.calls == [] and outside.read_bytes() == b"untouched"


def test_a_bundle_without_its_declared_executable_cannot_pass(tmp_path: Path) -> None:
    (tmp_path / "Info.plist").write_bytes(plistlib.dumps({"CFBundleExecutable": "Missing"}))
    report = Apple(BundleTool()).bundle(tmp_path)
    assert not report.successful and report.binaries == []
    assert "CFBundleExecutable" in report.violations[0]


@pytest.mark.parametrize(
    "header",
    [
        "feedface00000000",
        "feedfacf00000000",
        "cefaedfe00000000",
        "cffaedfe00000000",
        "cafebabe00000002",
        "cafebabf00000002",
        "bebafeca02000000",
        "bfbafeca02000000",
    ],
)
def test_thin_and_fat_headers_delegate_every_architecture_to_native_tools(
    app_bundle: Path,
    header: str,
) -> None:
    main = (
        app_bundle / "Contents/MacOS/Main"
        if (app_bundle / "Contents").exists()
        else app_bundle / "Main"
    )
    main.write_bytes(bytes.fromhex(header) + b"fixture")
    tool = BundleTool()
    assert Apple(tool).bundle(app_bundle).successful
    assert sum(call[0] in {"nm", "otool"} for call in tool.calls) == 4


@pytest.mark.parametrize("header", ["cafebabe", "cafebabe00000000", "cafebabe00000011"])
def test_malformed_or_unsupported_fat_headers_cannot_disappear_from_inspection(
    app_bundle: Path,
    header: str,
) -> None:
    (app_bundle / "Broken").write_bytes(bytes.fromhex(header))
    with pytest.raises(MissionError, match="fat Mach-O"):
        Apple(BundleTool()).bundle(app_bundle)


def test_entry_limit_is_enforced_while_enumerating(
    app_bundle: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def entries(path: Path, pattern: str) -> Iterator[Path]:
        assert path == app_bundle and pattern == "*"
        return repeat(app_bundle / "Example.class", 10001)

    monkeypatch.setattr(Path, "rglob", entries)
    with pytest.raises(MissionError, match="10,000"):
        Apple(BundleTool()).bundle(app_bundle)


def test_binary_limit_stops_native_work(app_bundle: Path) -> None:
    for index in range(129):
        (app_bundle / f"Binary{index}").write_bytes(bytes.fromhex("feedfacf") + b"fixture")
    tool = BundleTool()
    with pytest.raises(MissionError, match="128"):
        Apple(tool).bundle(app_bundle)
    assert sum(call[0] in {"nm", "otool"} for call in tool.calls) == 256


@pytest.mark.parametrize("forbidden", ["unmatched", "swift_release"])
def test_cli_json_and_exit_status_preserve_bundle_verdict(
    app_bundle: Path,
    forbidden: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("mainboard.verbs.apps.apple.Apple", lambda: Apple(BundleTool()))
    with pytest.raises(SystemExit) as stopped:
        build()(["apple", "bundle", str(app_bundle), "--forbid-symbol", forbidden, "--json"])
    assert stopped.value.code == (1 if forbidden == "swift_release" else 0)
    assert json.loads(capsys.readouterr().out)["successful"] == (forbidden == "unmatched")
