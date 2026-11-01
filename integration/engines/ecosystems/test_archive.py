"""Pinned release archives and the resolver that finds a tool and judges its version."""

import tarfile
from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.core import Verdict
from mainboard.engines.compile.ecosystems import Binaries
from mainboard.engines.compile.ecosystems.archive import Archive, Format, accepts
from mainboard.enums import Source

from .builders import executable, installed, link, tarball, zipped
from .constants import CONTENT, URL

_VERSION = st.tuples(*[st.integers(0, 999)] * 3).map(lambda parts: ".".join(map(str, parts)))
_PIN = {
    "version": "0.7.3",
    "url": URL,
    "sha256": "b" * 64,
    "member": "duckdb-otlp",
}


@pytest.fixture
def shelf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Binaries:
    """Resolver over an empty install directory and environment, PATH holding only `path/`."""
    monkeypatch.setenv("PATH", str(tmp_path / "path"))
    return Binaries(tmp_path / "binary", environment=tmp_path / "prefix" / "bin")


@given(version=_VERSION)
@example(version="0.7.3")
def test_a_pin_meets_only_itself_and_a_star_meets_everything(version: str) -> None:
    assert accepts(version, version=version)
    assert accepts("*", version=version)
    assert accepts("*", version="not a version")
    assert not accepts(version, version="not a version")


@pytest.mark.parametrize(
    ("wanted", "found", "meets"),
    [
        pytest.param(">=0.46, <0.47", "0.46.2", True, id="inside-a-conda-range"),
        pytest.param(">=0.46, <0.47", "0.47.0", False, id="above-a-conda-range"),
        pytest.param("0.46.*", "0.46.9", True, id="under-a-wildcard-pin"),
        pytest.param("0.7.3", "0.7.4", False, id="a-newer-patch-than-the-pin"),
        pytest.param("1.124.0", "", False, id="a-tool-that-printed-no-version"),
    ],
)
def test_a_manifest_version_is_read_as_a_pin_or_a_range(
    wanted: str, found: str, meets: bool
) -> None:
    assert accepts(wanted, version=found) is meets


@pytest.mark.parametrize(
    "name", ["tool", "./tool"], ids=["a-plain-entry", "an-entry-under-dot-slash"]
)
def test_a_tarball_member_is_read_whole_and_nothing_else_is_extracted(
    tmp_path: Path, name: str
) -> None:
    packed = tarball(tmp_path / "t.tar.gz", files={name: CONTENT, "other": b"unrelated"})
    assert Format.TAR.read(packed, "tool") == CONTENT
    assert sorted(path.name for path in tmp_path.iterdir()) == ["t.tar.gz"]


def test_a_zip_member_is_read_whole_beside_a_traversal_entry(tmp_path: Path) -> None:
    packed = zipped(tmp_path / "z.zip", "tool", CONTENT)
    assert Format.ZIP.read(packed, "tool") == CONTENT
    assert sorted(path.name for path in tmp_path.iterdir()) == ["z.zip"]


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        pytest.param((link("tool"),), "regular executable", id="a-symlink-named-as-the-member"),
        pytest.param(
            (link("tool", kind=tarfile.LNKTYPE),), "regular executable", id="a-hard-link"
        ),
        pytest.param(
            (link("tool", kind=tarfile.DIRTYPE),), "regular executable", id="a-directory"
        ),
        pytest.param((link("other"),), "declared member", id="no-entry-of-that-name"),
        pytest.param((link("../tool"),), "declared member", id="only-an-entry-climbing-out"),
    ],
)
def test_a_tarball_refuses_a_member_that_is_not_one_regular_file(
    tmp_path: Path, entries: tuple[tarfile.TarInfo, ...], message: str
) -> None:
    packed = tarball(tmp_path / "t.tar.gz", *entries)
    with pytest.raises(MissionError, match=message):
        Format.TAR.read(packed, "tool")


@pytest.mark.parametrize(
    ("format_", "message"),
    [
        pytest.param(Format.TAR, "TAR", id="not-a-gzip-tar"),
        pytest.param(Format.ZIP, "ZIP", id="not-a-zip"),
    ],
)
def test_a_damaged_archive_is_a_mission_error_naming_its_format(
    tmp_path: Path, format_: Format, message: str
) -> None:
    broken = tmp_path / "broken"
    broken.write_bytes(b"not an archive")
    with pytest.raises(MissionError, match=f"cannot read binary {message}"):
        format_.read(broken, "tool")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("url", "http://releases.example/t.tar.gz", id="a-plain-http-url"),
        pytest.param("url", "https://u:p@releases.example/t.tar.gz", id="credentials-in-the-url"),
        pytest.param("url", "https://releases.example/tool.rar", id="an-archive-no-reader-opens"),
        pytest.param("sha256", "B" * 64, id="an-uppercase-digest"),
        pytest.param("member", "../tool", id="a-member-climbing-out"),
        pytest.param("member", "/tool", id="an-absolute-member"),
        pytest.param("version", ">=0.7", id="a-range-in-place-of-a-pin"),
    ],
)
def test_an_archive_coordinate_must_be_exact_before_anything_is_fetched(
    field: str, value: str
) -> None:
    with pytest.raises(ValueError, match=r"binary requires|no reader"):
        Archive.model_validate({**_PIN, field: value})


def test_an_installed_pin_is_found_at_the_version_its_receipt_names(
    shelf: Binaries, tmp_path: Path
) -> None:
    installed(tmp_path / "binary", "duckdb-otlp", version="0.7.3")
    found = shelf.check("duckdb-otlp", wanted="0.7.3")
    assert (found.source, found.version) == (Source.INSTALLED, "0.7.3")
    assert found.path == tmp_path / "binary" / "bin" / "duckdb-otlp"


def test_an_installed_executable_altered_after_its_install_is_refused(
    shelf: Binaries, tmp_path: Path
) -> None:
    script = installed(tmp_path / "binary", "duckdb-otlp", version="0.7.3")
    script.write_bytes(CONTENT + b"# edited")
    with pytest.raises(MissionError, match="differs from the receipt"):
        shelf.find("duckdb-otlp")


def test_an_installed_pin_off_the_version_the_manifest_wants_is_refused(
    shelf: Binaries, tmp_path: Path
) -> None:
    installed(tmp_path / "binary", "duckdb-otlp", version="0.7.3")
    with pytest.raises(MissionError, match="wanted 0.8.0"):
        shelf.check("duckdb-otlp", wanted="0.8.0")


def test_a_linked_executable_in_the_install_directory_is_not_a_pin(
    shelf: Binaries, tmp_path: Path
) -> None:
    script = installed(tmp_path / "binary", "jj", version="0.46.0")
    real = script.with_name("real")
    script.rename(real)
    script.symlink_to(real)
    assert shelf.find("jj") is None


def test_the_environment_bin_answers_before_path_and_its_version_is_probed(
    shelf: Binaries, tmp_path: Path
) -> None:
    executable(tmp_path / "prefix" / "bin", "jj", said="jj 0.46.1")
    executable(tmp_path / "path", "jj", said="jj 0.40.0")
    found = shelf.check("jj", wanted=">=0.46, <0.47")
    assert (found.source, found.version) == (Source.ENVIRONMENT, "0.46.1")


def test_a_tool_only_on_path_is_found_and_judged_by_what_it_prints(
    shelf: Binaries, tmp_path: Path
) -> None:
    executable(tmp_path / "path", "process-compose", said="Process Compose v1.124.0")
    found = shelf.check("process-compose", wanted=">=1.124, <1.125")
    assert (found.source, found.version) == (Source.PATH, "1.124.0")


def test_a_tool_knowing_only_a_version_subcommand_is_judged_by_what_that_prints(
    shelf: Binaries, tmp_path: Path
) -> None:
    script = tmp_path / "path" / "process-compose"
    script.parent.mkdir()
    script.write_text(
        '#!/bin/sh\n[ "$1" = version ] && echo "Version: v1.124.0" || echo "unknown flag"\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    assert shelf.check("process-compose", wanted=">=1.124, <2").version == "1.124.0"


def test_a_tool_outside_the_wanted_range_is_refused_with_both_versions(
    shelf: Binaries, tmp_path: Path
) -> None:
    executable(tmp_path / "prefix" / "bin", "jj", said="jj 0.47.0")
    with pytest.raises(MissionError, match=r"jj 0\.47\.0 at .*, wanted >=0\.46, <0\.47"):
        shelf.check("jj", wanted=">=0.46, <0.47")


def test_a_missing_tool_is_not_installed_and_its_row_names_the_repair(shelf: Binaries) -> None:
    (row,) = shelf.rows({"jj": "*"}, fix="mb binary sync")
    assert (row.verdict, row.fix) == (Verdict.FAIL, "mb binary sync")
    assert row.detail.startswith("jj is pinned but not installed from its archive")
    with pytest.raises(MissionError, match="jj is not installed"):
        shelf.check("jj")


@pytest.mark.parametrize("where", ["path", "prefix/bin"], ids=["on-path", "in-the-environment"])
def test_a_same_version_twin_of_a_pin_meets_no_pin_and_fails_its_row(
    shelf: Binaries, tmp_path: Path, where: str
) -> None:
    executable(tmp_path / where, "duckdb-otlp", said="duckdb-otlp 0.7.3")
    assert shelf.check("duckdb-otlp", wanted="0.7.3").source is not Source.INSTALLED
    with pytest.raises(MissionError, match="pinned but not installed from its archive"):
        shelf.pinned("duckdb-otlp")
    row = shelf.section({"duckdb-otlp": "0.7.3"}, fix="mb binary sync")
    assert row is not None and (row.verdict, row.fix) == (Verdict.FAIL, "mb binary sync")


def test_the_doctor_row_fails_on_the_broken_tools_only_and_passes_when_all_fit(
    shelf: Binaries, tmp_path: Path
) -> None:
    installed(tmp_path / "binary", "duckdb-otlp", version="0.7.3")
    assert shelf.section({}, fix="mb binary sync") is None
    fit = shelf.section({"duckdb-otlp": "0.7.3"}, fix="mb binary sync")
    broken = shelf.section({"duckdb-otlp": "0.7.3", "jj": "*"}, fix="mb binary sync")
    assert fit is not None and fit.verdict is Verdict.PASS and fit.fix == ""
    assert broken is not None and broken.verdict is Verdict.FAIL
    assert broken.detail.startswith("jj: jj is pinned but not installed from its archive")
    assert broken.fix == "mb binary sync"
