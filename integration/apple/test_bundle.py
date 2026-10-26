import hashlib
import json
import plistlib
import string
from collections.abc import Iterator, Mapping, Sequence
from itertools import repeat
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from pydantic import JsonValue

from mainboard import MissionError
from mainboard.apple import Apple, BundleChecks
from mainboard.core import Runner, ToolOutput

from .constants import XCRUN

_THIN = bytes.fromhex("cffaedfe") + b"fixture"
_FAT = bytes.fromhex("cafebabe00000002") + b"fixture"
_JAVA_CLASS = bytes.fromhex("cafebabe0000003d") + b"java"
_LAYOUTS = {
    "iOS": (".", ".", "."),
    "macOS": ("Contents", "Contents/MacOS", "Contents/Resources"),
}
_INFO = plistlib.dumps(
    {
        "CFBundleExecutable": "Main",
        "CFBundleDisplayName": "Example",
        "CFBundleIcons": {"CFBundlePrimaryIcon": {"CFBundleIconName": "ExampleIcon"}},
        "NSMicrophoneUsageDescription": "Recognize a confirmed answer.",
    }
)
_NO_EXECUTABLE = "CFBundleExecutable does not identify an inspected Mach-O file"
_SEGMENT = st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=4)


class BundleTool(Runner):
    """Answers nm, otool and plutil for the bundle's files, or a tool from `replies`."""

    @property
    def native(self) -> int:
        """How many nm and otool runs were asked."""
        return sum(call[0] in {"nm", "otool"} for call in self.calls)

    def __init__(self, replies: Mapping[str, ToolOutput] | None = None) -> None:
        self.replies = replies or {}
        self.calls: list[Sequence[str]] = []

    def run(self, tool: str, args: Sequence[str], *, cwd: Path | None = None) -> ToolOutput:
        assert tool == XCRUN and cwd is None
        self.calls.append(args)
        if args[0] in self.replies:
            return self.replies[args[0]]
        if args[0] == "plutil":
            assert args[1:5] == ("-convert", "json", "-o", "-")
            return ToolOutput(stdout=json.dumps(plistlib.loads(Path(args[5]).read_bytes())))
        assert args[:4] in [("nm", "-arch", "all", "-u"), ("otool", "-arch", "all", "-L")]
        return ToolOutput(stdout=f"{args[4]} (arm64 and x86_64): libSystem U _swift_release\n")


def _strings(name: str) -> bytes:
    """A binary InfoPlist.strings declaring the display name `name`."""
    return plistlib.dumps({"CFBundleDisplayName": name}, fmt=plistlib.FMT_BINARY)


def _add(bundle: Path, files: Mapping[str, bytes], links: Mapping[str, str]) -> None:
    """Add files and symlinks, given by name and content or target, to `bundle`."""
    for name, data in files.items():
        (bundle / name).write_bytes(data)
    for name, target in links.items():
        (bundle / name).symlink_to(target)


@pytest.fixture(params=["iOS", "macOS"])
def app_bundle(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    """An app bundle in the flat iOS or the Contents macOS layout, holding two Mach-O files."""
    bundle = tmp_path / "日本 ' $(literal).app"
    contents, executables, resources = (bundle / part for part in _LAYOUTS[request.param])
    files = {
        executables / "Main": _THIN,
        contents / "PlugIns/Review.appex/Review": _FAT,
        # A Java version number must not be mistaken for the number of Mach-O fat slices.
        bundle / "Example.class": _JAVA_CLASS,
        contents / "Info.plist": _INFO,
        resources / "en.lproj/InfoPlist.strings": _strings("Example"),
        resources / "ja.lproj/InfoPlist.strings": _strings("例"),
        resources / "de.lproj/InfoPlist.strings": plistlib.dumps({}, fmt=plistlib.FMT_BINARY),
    }
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (bundle / "Alias").symlink_to(executables / "Main")
    return bundle


def test_bundle_inspects_all_binaries_preserves_evidence_and_deduplicates_internal_links(
    app_bundle: Path,
) -> None:
    tool = BundleTool()
    checks = BundleChecks(
        expect_info=(
            "CFBundleDisplayName=Example",
            "CFBundleIcons.CFBundlePrimaryIcon.CFBundleIconName=ExampleIcon",
        ),
        require_info=("NSMicrophoneUsageDescription",),
        expect_name=("en=Example", "ja=例"),
    )
    report = Apple(tool).bundle(app_bundle, checks)
    assert report.successful and report.violations == [] and report.document()["successful"]
    assert report.localized_names == {"en": "Example", "ja": "例"}
    assert (len(report.binaries), report.file_count, tool.native) == (2, 7, 4)
    assert [binary.sha256 for binary in report.binaries] == [
        hashlib.sha256((app_bundle / binary.path).read_bytes()).hexdigest()
        for binary in report.binaries
    ]
    assert all(
        binary.symbols.stdout.endswith("U _swift_release\n") and binary.libraries.stdout
        for binary in report.binaries
    )


@pytest.mark.parametrize(
    ("relative", "kind", "pattern"),
    [
        pytest.param("study.store", "file", "**/study.store", id="a-file-at-the-root"),
        pytest.param("nested/study.store", "file", "**/study.store", id="a-nested-file"),
        pytest.param("PlugIns/Debug.xctest", "dir", "**/*.xctest", id="a-directory"),
    ],
)
def test_forbidden_resources_include_root_nested_files_and_directories(
    app_bundle: Path, relative: str, kind: str, pattern: str
) -> None:
    target = app_bundle / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    {"file": Path.touch, "dir": Path.mkdir}[kind](target)
    checks = BundleChecks(forbid_resource=("**/study.store", "**/*.xctest"))
    report = Apple(BundleTool()).bundle(app_bundle, checks)
    assert not report.successful
    assert report.violations == [f"forbidden resource: {relative} ({pattern})"]


def test_symbols_stderr_metadata_and_localized_policy_all_affect_the_verdict(
    app_bundle: Path,
) -> None:
    checks = BundleChecks(
        forbid_symbol=("fmdatabase",),
        expect_info=("CFBundleDisplayName=Wrong",),
        require_info=("Missing", "CFBundleIcons"),
        expect_name=("ja=Wrong", "de=Anything"),
    )
    report = Apple(BundleTool({"nm": ToolOutput(stderr="U _FMDatabase")})).bundle(
        app_bundle, checks
    )
    assert not report.successful
    assert report.violations == [
        "Info.plist CFBundleDisplayName does not equal 'Wrong'",
        "Info.plist Missing is not nonempty text",
        "Info.plist CFBundleIcons is not nonempty text",
        "ja display name does not equal 'Wrong'",
        "de display name does not equal 'Anything'",
        *[f"{binary.path}: nm matched 'FMDatabase'" for binary in report.binaries],
    ]
    assert report.binaries[0].symbols.stderr == "U _FMDatabase"


@pytest.mark.parametrize(
    ("replies", "files", "links", "match"),
    [
        pytest.param(
            {"nm": ToolOutput(returncode=1, stderr="unreadable")},
            {},
            {},
            r"inspection nm failed \(exit 1, truncated=False\): unreadable",
            id="nm-fails",
        ),
        pytest.param(
            {"otool": ToolOutput(truncated=True)},
            {},
            {},
            r"inspection otool failed \(exit 0, truncated=True\)",
            id="otool-output-is-cut-short",
        ),
        pytest.param(
            {"otool": ToolOutput()},
            {},
            {},
            "otool returned no inspection",
            id="otool-prints-nothing",
        ),
        pytest.param(
            {"plutil": ToolOutput(stdout="not JSON")},
            {},
            {},
            "invalid bundle property list",
            id="a-property-list-that-is-not-json",
        ),
        pytest.param(
            {"plutil": ToolOutput(stdout="[]")},
            {},
            {},
            "invalid bundle property list",
            id="a-property-list-that-is-not-a-dictionary",
        ),
        *(
            pytest.param(
                {},
                {"Broken": bytes.fromhex(header)},
                {},
                "fat Mach-O",
                id=f"fat-header-{header}",
            )
            for header in ["cafebabe", "cafebabe00000000", "cafebabe00000011", "bebafeca11000000"]
        ),
        pytest.param(
            {},
            {"Broken.class": bytes.fromhex("cafebabe0000002c")},
            {},
            "fat Mach-O",
            id="a-class-file-version-below-the-java-range",
        ),
    ],
)
def test_incomplete_native_inspection_never_passes(
    app_bundle: Path, replies, files, links, match: str
) -> None:
    _add(app_bundle, files, links)
    tool = BundleTool(replies)
    with pytest.raises(MissionError, match=match):
        Apple(tool).bundle(app_bundle)
    assert {call[0] for call in tool.calls} <= {"nm", "otool", "plutil"}


@pytest.mark.parametrize(
    ("header", "name", "inspected"),
    [
        pytest.param("feedface00000000", "Extra", 1, id="thin-32-bit"),
        pytest.param("feedfacf00000000", "Extra", 1, id="thin-64-bit"),
        pytest.param("cefaedfe00000000", "Extra", 1, id="thin-32-bit-byte-swapped"),
        pytest.param("cffaedfe00000000", "Extra", 1, id="thin-64-bit-byte-swapped"),
        pytest.param("cafebabe00000002", "Extra", 1, id="fat-32-bit"),
        pytest.param("cafebabf00000002", "Extra", 1, id="fat-64-bit"),
        pytest.param("bebafeca02000000", "Extra", 1, id="fat-32-bit-little-endian"),
        pytest.param("bfbafeca02000000", "Extra", 1, id="fat-64-bit-little-endian"),
        pytest.param("cafebabe00000010", "Extra", 1, id="fat-with-sixteen-slices"),
        pytest.param("bebafeca10000000", "Extra", 1, id="little-endian-fat-with-sixteen-slices"),
        pytest.param("cafebabe0000003d", "Extra.class", 0, id="a-java-class-file"),
        pytest.param("deadbeef00000000", "Extra", 0, id="no-mach-o-magic"),
    ],
)
def test_thin_and_fat_headers_delegate_every_architecture_to_native_tools(
    app_bundle: Path, header: str, name: str, inspected: int
) -> None:
    (app_bundle / name).write_bytes(bytes.fromhex(header))
    tool = BundleTool()
    assert Apple(tool).bundle(app_bundle).successful
    assert tool.native == 4 + 2 * inspected


def test_the_entry_limit_is_enforced_while_enumerating(
    app_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def entries(path: Path, pattern: str) -> Iterator[Path]:
        assert path == app_bundle and pattern == "*"
        return repeat(app_bundle / "Example.class", 10001)

    monkeypatch.setattr(Path, "rglob", entries)
    tool = BundleTool()
    with pytest.raises(MissionError, match="10,000"):
        Apple(tool).bundle(app_bundle)
    assert tool.calls == []


@pytest.mark.parametrize(
    ("files", "links", "match", "asked"),
    [
        pytest.param(
            {f"Binary{index}": _THIN for index in range(129)},
            {},
            "128 Mach-O files",
            1 + 2 * 128,
            id="more-than-128-mach-o-files",
        ),
        pytest.param(
            {},
            {"Escape": "../outside"},
            "bundle symlink escapes",
            0,
            id="a-symlink-leaving-the-bundle",
        ),
    ],
)
def test_unbounded_or_escaping_content_stops_native_work(
    app_bundle: Path, files, links, match: str, asked: int
) -> None:
    outside = app_bundle.parent / "outside"
    outside.write_bytes(b"untouched")
    _add(app_bundle, files, links)
    tool = BundleTool()
    with pytest.raises(MissionError, match=match):
        Apple(tool).bundle(app_bundle)
    assert len(tool.calls) == asked and outside.read_bytes() == b"untouched"


@pytest.mark.parametrize(
    ("info", "files"),
    [
        pytest.param({"CFBundleExecutable": "Missing"}, {}, id="an-executable-that-is-missing"),
        pytest.param({}, {}, id="no-executable-declared"),
        pytest.param({"CFBundleExecutable": 5}, {}, id="an-executable-that-is-no-name"),
        pytest.param(
            {"CFBundleExecutable": "Main"},
            {"Main": b"#!/bin/sh"},
            id="an-executable-that-is-no-mach-o",
        ),
    ],
)
def test_a_bundle_without_a_declared_mach_o_executable_cannot_pass(
    tmp_path: Path, info: dict[str, JsonValue], files: dict[str, bytes]
) -> None:
    (tmp_path / "Info.plist").write_bytes(plistlib.dumps(info))
    _add(tmp_path, files, {})
    report = Apple(BundleTool()).bundle(tmp_path)
    assert not report.successful and report.binaries == []
    assert report.violations == [_NO_EXECUTABLE]


def test_a_plain_file_is_no_bundle(tmp_path: Path) -> None:
    (tmp_path / "Main").write_bytes(_THIN)
    tool = BundleTool()
    with pytest.raises(MissionError, match="no Apple bundle directory"):
        Apple(tool).bundle(tmp_path / "Main")
    assert tool.calls == []


def test_a_path_that_does_not_exist_is_no_bundle(tmp_path: Path) -> None:
    tool = BundleTool()
    with pytest.raises(MissionError, match="no Apple bundle directory"):
        Apple(tool).bundle(tmp_path / "Missing.app")
    assert tool.calls == []


@pytest.mark.parametrize(
    ("policy", "match"),
    [
        pytest.param(
            {"forbid_symbol": ("[",)},
            "invalid forbidden-symbol pattern",
            id="a-symbol-regex-that-does-not-compile",
        ),
        pytest.param({"forbid_symbol": ("",)}, "must not be empty", id="an-empty-symbol-pattern"),
        pytest.param({"forbid_resource": (" ",)}, "must not be empty", id="a-blank-resource-glob"),
        pytest.param(
            {"forbid_resource": ("/absolute",)}, "relative to the bundle", id="an-absolute-glob"
        ),
        pytest.param(
            {"forbid_resource": ("../outside",)},
            "relative to the bundle",
            id="a-glob-leaving-the-bundle",
        ),
        pytest.param(
            {"expect_info": ("Name",)}, "unique KEY=VALUE", id="an-expectation-without-a-value"
        ),
        pytest.param(
            {"expect_info": ("Name=A", "Name=B")}, "unique KEY=VALUE", id="a-repeated-expectation"
        ),
        pytest.param(
            {"expect_name": ("../en=Name",)}, "unique KEY=VALUE", id="a-dotted-locale-key"
        ),
        pytest.param(
            {"expect_name": ("en/US=Name",)},
            "one locale directory name",
            id="a-locale-that-is-no-directory-name",
        ),
        pytest.param(
            {"require_info": ("bad..path",)},
            "nonempty dotted paths",
            id="a-required-key-with-an-empty-segment",
        ),
    ],
)
def test_an_unsound_bundle_policy_cannot_be_built(policy, match: str) -> None:
    with pytest.raises(MissionError, match=match):
        BundleChecks(**policy)


@settings(max_examples=200)
@given(parts=st.lists(_SEGMENT, min_size=1, max_size=3), value=st.text(max_size=6))
@example(parts=["CFBundleIcons", "CFBundlePrimaryIcon"], value="a=b")
@example(parts=["Name"], value=" ")
def test_info_expectations_follow_the_dotted_path_and_compare_the_text_after_the_first_equals(
    parts: list[str], value: str
) -> None:
    key = ".".join(parts)
    info: dict[str, JsonValue] = {parts[-1]: value}
    for part in reversed(parts[:-1]):
        info = {part: info}
    exact = BundleChecks(expect_info=(f"{key}={value}",), require_info=(key,))
    wrong = BundleChecks(expect_info=(f"{key}={value}x",), require_info=(f"{key}.below",))
    assert exact.misread(info) == (
        [] if value.strip() else [f"Info.plist {key} is not nonempty text"]
    )
    assert wrong.misread(info) == [
        f"Info.plist {key} does not equal {value + 'x'!r}",
        f"Info.plist {key}.below is not nonempty text",
    ]


@pytest.mark.parametrize(
    ("forbidden", "code"),
    [
        pytest.param("unmatched", 0, id="a-symbol-nothing-prints-passes"),
        pytest.param("swift_release", 1, id="a-printed-symbol-fails"),
    ],
)
def test_the_bundle_verb_prints_json_and_exits_with_the_bundle_verdict(
    cli, app_bundle: Path, forbidden: str, code: int
) -> None:
    exit_code, out = cli(
        Apple(BundleTool()), "bundle", str(app_bundle), "--forbid-symbol", forbidden, "--json"
    )
    assert exit_code == code
    assert json.loads(out)["successful"] is (code == 0)
