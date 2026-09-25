import json
import os
from typing import TYPE_CHECKING

import pytest

from mainboard.engines.compile.backend import EnvironmentAudit
from mainboard.engines.compile.backend.repair import recorded_extensions

if TYPE_CHECKING:
    from pathlib import Path

    from ..support import Record

_EXTENSION = "core.cpython-314-x86_64-linux-gnu.so"


@pytest.fixture
def site_packages(tmp_path: Path) -> Path:
    """The empty site-packages tree of a provisioned environment prefix."""
    tree = tmp_path / "prefix" / "lib" / "python3.14" / "site-packages"
    tree.mkdir(parents=True)
    return tree


@pytest.fixture
def audit(site_packages: Path) -> EnvironmentAudit:
    return EnvironmentAudit(site_packages.parent.parent.parent)


def aged(path: Path, moment: int) -> None:
    """Stamp `path` with a modification time in nanoseconds."""
    os.utime(path, ns=(moment, moment))


def linked(prefix: Path, name: str, *files: str, present: bool = True) -> None:
    """Record a conda package linking `files` into `prefix`, writing them when `present`."""
    record = {"name": name, "version": "1.0", "files": list(files), "paths_data": {"paths": []}}
    (prefix / "conda-meta").mkdir(exist_ok=True)
    (prefix / "conda-meta" / f"{name}-1.0-0.json").write_text(json.dumps(record))
    for file in files if present else ():
        (prefix / file).parent.mkdir(parents=True, exist_ok=True)
        (prefix / file).write_text("")


def test_a_conda_package_missing_a_file_it_linked_is_damaged_and_an_intact_one_is_not(
    audit: EnvironmentAudit,
) -> None:
    """dust ships a `.crates.toml` beside its binary, and cargo reading it deleted the binary."""
    linked(audit.prefix, "dust", ".crates.toml", ".crates2.json", "bin/dust")
    linked(audit.prefix, "ripgrep", "bin/rg")
    linked(audit.prefix, "_openmp_mutex")
    (audit.prefix / "bin" / "dust").unlink()

    assert audit.damaged() == ("dust",)
    assert audit.suspect() == ("dust",)


def test_a_wheel_that_lost_every_import_root_is_damaged_and_named_in_a_stable_order(
    audit: EnvironmentAudit, record: Record, site_packages: Path
) -> None:
    for name in ("Zeta", "alpha", "Beta"):
        record(site_packages, name, roots=f"{name.lower()}\n")

    assert audit.damaged() == ("alpha", "Beta", "Zeta")
    assert audit.suspect() == ("alpha", "Beta", "Zeta")


@pytest.mark.parametrize(
    "surviving",
    [
        pytest.param("demo", id="a-package-directory"),
        pytest.param("demo.py", id="a-plain-module"),
        pytest.param("demo.abi3.so", id="a-compiled-extension"),
    ],
)
def test_a_wheel_keeping_one_import_root_is_left_alone(
    surviving: str, audit: EnvironmentAudit, record: Record, site_packages: Path
) -> None:
    record(site_packages, "demo", roots="demo\ndemo_compat\n")
    if surviving.endswith((".py", ".so")):
        site_packages.joinpath(surviving).write_text("")
    else:
        site_packages.joinpath(surviving).mkdir()

    assert audit.damaged() == ()


@pytest.mark.parametrize(
    ("installer", "roots"),
    [
        pytest.param("uv-pixi", "", id="a-distribution-declaring-no-import-root"),
        pytest.param("conda", "conda_owned\n", id="a-record-another-manager-owns"),
    ],
)
def test_a_record_claiming_nothing_pixi_installed_is_never_reinstalled(
    installer: str, roots: str, audit: EnvironmentAudit, record: Record, site_packages: Path
) -> None:
    record(site_packages, "claimless", installer=installer, roots=roots)

    assert audit.suspect() == ()


def test_an_editable_is_judged_by_its_clock_and_not_by_its_import_roots(
    audit: EnvironmentAudit, record: Record, site_packages: Path, tmp_path: Path
) -> None:
    source = tmp_path / "editable-demo"
    source.mkdir()
    record(
        site_packages,
        "editable-demo",
        roots="editable_demo\n",
        url=source.as_uri(),
        editable=True,
    )

    assert audit.damaged() == ()
    assert audit.suspect() == ()


def _native_editable(
    record: Record, site_packages: Path, source: Path, *, built: bool = True
) -> Path:
    """Record `source` as an editable carrying one extension, written when `built`."""
    module = source.name.replace("-", "_")
    source.mkdir(parents=True, exist_ok=True)
    record(
        site_packages,
        source.name,
        url=source.as_uri(),
        editable=True,
        files=[f"{module}/{_EXTENSION}"],
    )
    artifact = site_packages / module / _EXTENSION
    if built:
        artifact.parent.mkdir()
        artifact.write_bytes(b"compiled")
    return artifact


def test_a_native_editable_is_dated_by_its_compiled_sources_alone(
    audit: EnvironmentAudit, record: Record, site_packages: Path, tmp_path: Path
) -> None:
    """A `.py` edit is already live in an editable, while a `.cpp` edit needs the rebuild."""
    source = tmp_path / "native-demo"
    artifact = _native_editable(record, site_packages, source)
    source.joinpath("core.cpp").write_text("void changed() {}\n")
    source.joinpath("wrapper.py").write_text("from .core import run\n")
    aged(source / "core.cpp", 1_000_000_000)
    aged(artifact, 2_000_000_000)
    aged(source / "wrapper.py", 3_000_000_000)

    assert audit.suspect() == ()

    aged(source / "core.cpp", 4_000_000_000)

    assert audit.suspect() == ("native-demo",)
    assert audit.damaged() == ()


def test_a_native_editable_whose_extension_vanished_is_rebuilt(
    audit: EnvironmentAudit, record: Record, site_packages: Path, tmp_path: Path
) -> None:
    """Asked apart from any clock, so it holds with no source file newer than anything."""
    source = tmp_path / "native-demo"
    _native_editable(record, site_packages, source, built=False)
    source.joinpath("Cargo.toml").write_text("[package]\n")

    assert audit.suspect() == ("native-demo",)


@pytest.mark.parametrize(
    "noise",
    [
        pytest.param("pyproject.toml", id="packaging-configuration-rewritten-in-place"),
        pytest.param("CMakeLists.txt", id="build-configuration-rewritten-in-place"),
        pytest.param(".venv/include/vendored.h", id="a-header-under-a-dot-directory"),
        pytest.param("target/debug/generated.c", id="a-source-under-build-output"),
    ],
)
def test_a_newer_file_no_compile_reads_never_dates_an_extension(
    noise: str, audit: EnvironmentAudit, record: Record, site_packages: Path, tmp_path: Path
) -> None:
    """cutoken read stale on a `pyproject.toml` two days ahead of its `.cpp` (2026-09-05)."""
    source = tmp_path / "config-demo"
    artifact = _native_editable(record, site_packages, source)
    source.joinpath("core.cpp").write_text("void run() {}\n")
    source.joinpath(noise).parent.mkdir(parents=True, exist_ok=True)
    source.joinpath(noise).write_text("touched, not compiled\n")
    aged(source / "core.cpp", 1_000_000_000)
    aged(artifact, 2_000_000_000)
    aged(source / noise, 9_000_000_000)

    assert audit.suspect() == ()


def test_a_build_that_wrote_several_extensions_is_dated_by_the_one_it_finished_with(
    audit: EnvironmentAudit, record: Record, site_packages: Path, tmp_path: Path
) -> None:
    """The first artifact would call a multi-extension build stale for its own sources."""
    source = tmp_path / "many-demo"
    source.mkdir()
    files = [f"many_demo/first{_EXTENSION}", f"many_demo/second{_EXTENSION}"]
    record(site_packages, "many-demo", url=source.as_uri(), editable=True, files=files)
    site_packages.joinpath("many_demo").mkdir()
    for at, name in enumerate(files):
        artifact = site_packages / name
        artifact.write_bytes(b"compiled")
        aged(artifact, 1_000_000_000 + at * 2_000_000_000)
    source.joinpath("core.cu").write_text("__global__ void run() {}\n")
    aged(source / "core.cu", 2_000_000_000)

    assert audit.suspect() == ()

    aged(source / "core.cu", 4_000_000_000)

    assert audit.suspect() == ("many-demo",)


@pytest.mark.parametrize(
    ("files", "marker", "origin"),
    [
        pytest.param(
            ["demo/__init__.py"],
            "pyproject.toml",
            None,
            id="a-pure-python-editable-compiled-nothing-to-go-stale",
        ),
        pytest.param(
            None,
            "meson.build",
            None,
            id="an-install-claiming-no-files-claims-no-extension-either",
        ),
        pytest.param(
            [f"demo/{_EXTENSION}"],
            "Cargo.toml",
            "https://example.invalid/demo.zip",
            id="an-editable-installed-from-somewhere-other-than-a-directory",
        ),
        pytest.param(
            [f"demo/{_EXTENSION}"],
            "core.cpp",
            "",
            id="a-wheel-carries-no-source-tree-to-be-newer-than-it",
        ),
    ],
)
def test_an_install_with_nothing_to_rebuild_is_left_alone(
    files: list[str] | None,
    marker: str,
    origin: str | None,
    audit: EnvironmentAudit,
    record: Record,
    site_packages: Path,
    tmp_path: Path,
) -> None:
    source = tmp_path / "demo-source"
    source.mkdir()
    source.joinpath(marker).write_text("newer than anything installed\n")
    site_packages.joinpath("demo").mkdir()
    record(
        site_packages,
        "demo",
        roots="demo\n",
        url=source.as_uri() if origin is None else origin,
        editable=True,
        files=files,
    )

    assert audit.suspect() == ()


@pytest.mark.parametrize(
    ("distribution", "files", "asked", "found"),
    [
        pytest.param(
            "single-demo",
            ["single.py", "native/core.so", "README", "single_demo-1.0.dist-info/RECORD"],
            "single",
            ["native/core.so"],
            id="a-top-level-module-is-an-import-root-of-its-own",
        ),
        pytest.param(
            "Native",
            ["impl/core.so", "../../../bin/native"],
            "native",
            ["impl/core.so"],
            id="a-distribution-named-like-the-import-answers-when-no-root-claims-it",
        ),
        pytest.param("native", ["impl/core.so"], "absent", [], id="an-unclaimed-import"),
    ],
)
def test_the_extensions_an_import_resolves_through_are_read_off_the_claiming_record(
    distribution: str,
    files: list[str],
    asked: str,
    found: list[str],
    record: Record,
    site_packages: Path,
) -> None:
    record(site_packages, distribution, files=files)
    prefix = site_packages.parent.parent.parent

    assert recorded_extensions(asked, prefix=prefix) == tuple(
        site_packages / path for path in found
    )
