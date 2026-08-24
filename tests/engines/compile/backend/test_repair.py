import os
from typing import TYPE_CHECKING

import pytest

from mainboard.engines.compile.backend import EnvironmentAudit

if TYPE_CHECKING:
    from pathlib import Path

    from ..conftest import Record

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


def aged(path: Path, moment: int) -> None:
    """Stamp ``path`` with a modification time the test chose, in nanoseconds."""
    os.utime(path, ns=(moment, moment))


def test_a_wheel_that_lost_every_import_root_is_damaged_and_named_in_a_stable_order(
    audit: EnvironmentAudit, record: Record, site_packages: Path
) -> None:
    """A `dist-info` outliving its own files is what a swapped provider leaves behind, and the
    argv a repair builds is ordered case-insensitively so the same damage reinstalls the same."""
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
    """One surviving root is enough, in whichever shape Python can import it from."""
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
    """Metadata-only distributions claim nothing that could go missing, and only what uv
    installed for pixi is pixi's to reinstall, however broken a conda one looks."""
    record(site_packages, "claimless", installer=installer, roots=roots)

    assert audit.suspect() == ()


def test_an_editable_is_judged_by_its_clock_and_not_by_its_import_roots(
    audit: EnvironmentAudit, record: Record, site_packages: Path, tmp_path: Path
) -> None:
    """An editable imports through a path hook, so absent site-packages roots mean nothing."""
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


def test_a_native_editable_is_dated_by_its_compiled_sources_alone(
    audit: EnvironmentAudit, record: Record, site_packages: Path, tmp_path: Path
) -> None:
    """A `.py` edit is already live in an editable, while a `.cpp` edit needs the rebuild."""
    source = tmp_path / "native-demo"
    source.joinpath("src").mkdir(parents=True)
    record(
        site_packages,
        "native-demo",
        url=source.as_uri(),
        editable=True,
        files=[f"native_demo/{_EXTENSION}"],
    )
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
    audit: EnvironmentAudit, record: Record, site_packages: Path, tmp_path: Path
) -> None:
    """A recorded extension nobody can find dates to nothing, so any source outranks it."""
    source = tmp_path / "native-demo"
    source.mkdir()
    record(
        site_packages,
        "native-demo",
        url=source.as_uri(),
        editable=True,
        files=[f"native_demo/{_EXTENSION}"],
    )
    source.joinpath("Cargo.toml").write_text("[package]\n")

    assert audit.suspect() == ("native-demo",)


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
    """With no local tree there is no clock to read, and nothing compiled can go out of date."""
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


def test_build_output_inside_a_source_tree_never_dates_a_rebuild(
    audit: EnvironmentAudit, record: Record, site_packages: Path, tmp_path: Path
) -> None:
    """A vendored `.venv` or `target/` is full of newer headers nothing here compiles from."""
    source = tmp_path / "vendored-demo"
    for vendored in (".venv/include", "target/debug", "src"):
        source.joinpath(vendored).mkdir(parents=True)
    record(
        site_packages,
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
