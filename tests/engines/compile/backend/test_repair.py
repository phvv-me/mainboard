import os
from typing import TYPE_CHECKING

import pytest

from mainboard.engines.compile.backend import EnvironmentAudit

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

type Record = Callable[..., Path]

_EXTENSION = "core.cpython-314-x86_64-linux-gnu.so"


@pytest.fixture
def site_packages(tmp_path: Path) -> Path:
    """The site-packages tree of a provisioned environment prefix, empty to start with."""
    tree = tmp_path / "prefix" / "lib" / "python3.14" / "site-packages"
    tree.mkdir(parents=True)
    return tree


@pytest.fixture
def audit(site_packages: Path) -> EnvironmentAudit:
    """An audit of the prefix owning the site-packages tree each test populates."""
    return EnvironmentAudit(site_packages.parent.parent.parent)


@pytest.fixture
def record(site_packages: Path) -> Record:
    """A factory writing one `dist-info` the way an installer leaves it behind.

    name: the distribution name, which is also what a repair reinstalls it by.
    installer: the manager claiming the record, `uv-pixi` for everything pixi installs.
    roots: the `top_level.txt` import roots, the file left out entirely when empty.
    url: where the install came from, its PEP 610 record left out entirely when empty.
    editable: whether that PEP 610 record marks the install as editable.
    files: the `RECORD` paths relative to site-packages, the file left out when `None`.
    """

    def write(
        name: str,
        *,
        installer: str = "uv-pixi",
        roots: str = "",
        url: str = "",
        editable: bool = False,
        files: list[str] | None = None,
    ) -> Path:
        metadata = site_packages / f"{name}-1.0.dist-info"
        metadata.mkdir()
        metadata.joinpath("METADATA").write_text(f"Name: {name}\nVersion: 1.0\n")
        metadata.joinpath("INSTALLER").write_text(installer)
        if roots:
            metadata.joinpath("top_level.txt").write_text(roots)
        if url:
            editability = str(editable).lower()
            metadata.joinpath("direct_url.json").write_text(
                f'{{"url": "{url}", "dir_info": {{"editable": {editability}}}}}'
            )
        if files is not None:
            metadata.joinpath("RECORD").write_text("".join(f"{path},,\n" for path in files))
        return metadata

    return write


def aged(path: Path, moment: int) -> None:
    """Stamp ``path`` with a modification time the test chose, in nanoseconds."""
    os.utime(path, ns=(moment, moment))


def test_a_wheel_that_lost_every_import_root_is_damaged(
    audit: EnvironmentAudit, record: Record
) -> None:
    """A `dist-info` outliving its own files is what a swapped provider leaves behind."""
    record("cupy-cuda13x", roots="cupy\ncupyx\n")

    assert audit.damaged() == ("cupy-cuda13x",)
    assert audit.suspect() == ("cupy-cuda13x",)


@pytest.mark.parametrize("kind", ["package", "module", "extension"])
def test_a_wheel_keeping_one_import_root_is_left_alone(
    kind: str, audit: EnvironmentAudit, record: Record, site_packages: Path
) -> None:
    """One surviving root is enough, in whichever shape Python can import it from."""
    record("demo", roots="demo\ndemo_compat\n")
    if kind == "package":
        site_packages.joinpath("demo").mkdir()
    else:
        site_packages.joinpath(f"demo{'.py' if kind == 'module' else '.abi3.so'}").write_text("")

    assert audit.damaged() == ()


def test_a_distribution_declaring_no_import_root_is_never_damaged(
    audit: EnvironmentAudit, record: Record
) -> None:
    """Metadata-only distributions and namespace shims claim nothing that could go missing."""
    record("metadata-only")

    assert audit.damaged() == ()


def test_records_another_manager_owns_are_left_to_it(
    audit: EnvironmentAudit, record: Record
) -> None:
    """Only what uv installed for pixi is pixi's to reinstall, however broken a conda one looks."""
    record("conda-owned", installer="conda", roots="conda_owned\n")

    assert audit.suspect() == ()


def test_reported_names_are_distinct_and_ordered_case_insensitively(
    audit: EnvironmentAudit, record: Record
) -> None:
    """The argv a repair builds is stable, so the same damage always reinstalls the same way."""
    for name in ("Zeta", "alpha", "Beta"):
        record(name, roots=f"{name.lower()}\n")

    assert audit.damaged() == ("alpha", "Beta", "Zeta")


def test_an_editable_is_judged_by_its_clock_and_not_by_its_import_roots(
    audit: EnvironmentAudit, record: Record, tmp_path: Path
) -> None:
    """An editable imports through a path hook, so absent site-packages roots mean nothing."""
    source = tmp_path / "editable-demo"
    source.mkdir()
    record("editable-demo", roots="editable_demo\n", url=source.as_uri(), editable=True)

    assert audit.damaged() == ()
    assert audit.suspect() == ()


def test_a_native_editable_is_dated_by_its_compiled_sources_alone(
    audit: EnvironmentAudit, record: Record, site_packages: Path, tmp_path: Path
) -> None:
    """A `.py` edit is already live in an editable, while a `.cpp` edit needs the rebuild."""
    source = tmp_path / "native-demo"
    source.joinpath("src").mkdir(parents=True)
    record("native-demo", url=source.as_uri(), editable=True, files=[f"native_demo/{_EXTENSION}"])
    artifact = site_packages / "native_demo" / _EXTENSION
    artifact.parent.mkdir()
    artifact.write_bytes(b"compiled")
    source.joinpath("src", "core.cpp").write_text("void changed() {}\n")
    source.joinpath("src", "wrapper.py").write_text("from .core import run\n")
    aged(source / "src" / "core.cpp", 1_000_000_000)
    aged(artifact, 2_000_000_000)
    aged(source / "src" / "wrapper.py", 3_000_000_000)

    assert audit.suspect() == ()

    aged(source / "src" / "core.cpp", 4_000_000_000)

    assert audit.suspect() == ("native-demo",)
    assert audit.damaged() == ()


def test_a_native_editable_whose_extension_vanished_is_rebuilt(
    audit: EnvironmentAudit, record: Record, tmp_path: Path
) -> None:
    """A recorded extension nobody can find dates to nothing, so any source outranks it."""
    source = tmp_path / "native-demo"
    source.mkdir()
    record("native-demo", url=source.as_uri(), editable=True, files=[f"native_demo/{_EXTENSION}"])
    source.joinpath("Cargo.toml").write_text("[package]\n")

    assert audit.suspect() == ("native-demo",)


def test_a_pure_python_editable_is_never_rebuilt(
    audit: EnvironmentAudit, record: Record, tmp_path: Path
) -> None:
    """Nothing was compiled, so its sources are already the ones Python imports."""
    source = tmp_path / "pure-demo"
    source.mkdir()
    record("pure-demo", url=source.as_uri(), editable=True, files=["pure_demo/__init__.py"])
    source.joinpath("pyproject.toml").write_text("[project]\n")

    assert audit.suspect() == ()


def test_an_editable_without_a_record_of_its_files_is_never_rebuilt(
    audit: EnvironmentAudit, record: Record, tmp_path: Path
) -> None:
    """An install claiming no files claims no extension either."""
    source = tmp_path / "unrecorded-demo"
    source.mkdir()
    record("unrecorded-demo", url=source.as_uri(), editable=True)
    source.joinpath("meson.build").write_text("project('demo')\n")

    assert audit.suspect() == ()


def test_an_editable_installed_from_somewhere_other_than_a_directory_is_left_alone(
    audit: EnvironmentAudit, record: Record
) -> None:
    """With no local tree there is no clock to read, and an editable is never read as a wheel."""
    record(
        "remote-demo",
        roots="remote_demo\n",
        url="https://example.invalid/remote-demo.zip",
        editable=True,
        files=[f"remote_demo/{_EXTENSION}"],
    )

    assert audit.suspect() == ()


def test_a_wheel_that_shipped_an_extension_is_not_rebuilt_by_its_sources(
    audit: EnvironmentAudit, record: Record, site_packages: Path
) -> None:
    """A wheel carries no source tree to be newer than it, however native its contents are."""
    record("binary-wheel", roots="binary_wheel\n", files=[f"binary_wheel/{_EXTENSION}"])
    artifact = site_packages / "binary_wheel" / _EXTENSION
    artifact.parent.mkdir()
    artifact.write_bytes(b"compiled")

    assert audit.suspect() == ()


def test_build_output_inside_a_source_tree_never_dates_a_rebuild(
    audit: EnvironmentAudit, record: Record, site_packages: Path, tmp_path: Path
) -> None:
    """A vendored `.venv` or `target/` is full of newer headers nothing here compiles from."""
    source = tmp_path / "vendored-demo"
    for vendored in (".venv/include", "target/debug", "src"):
        source.joinpath(vendored).mkdir(parents=True)
    record(
        "vendored-demo",
        url=source.as_uri(),
        editable=True,
        files=[f"vendored_demo/{_EXTENSION}"],
    )
    artifact = site_packages / "vendored_demo" / _EXTENSION
    artifact.parent.mkdir()
    artifact.write_bytes(b"compiled")
    source.joinpath("src", "core.rs").write_text("fn main() {}\n")
    source.joinpath(".venv", "include", "vendored.h").write_text("#define VENDORED 1\n")
    source.joinpath("target", "debug", "generated.c").write_text("int generated(void);\n")
    aged(source / "src" / "core.rs", 1_000_000_000)
    aged(artifact, 2_000_000_000)
    aged(source / ".venv" / "include" / "vendored.h", 3_000_000_000)
    aged(source / "target" / "debug" / "generated.c", 3_000_000_000)

    assert audit.suspect() == ()
