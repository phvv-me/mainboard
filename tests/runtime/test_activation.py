import os
import runpy
import shlex
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.runtime import activation
from mainboard.runtime.activation import (
    BuildPaths,
    CompileCaches,
    Runtime,
    WheelLibraries,
    defaulted,
    main,
    prepended,
)

# Directory names a path list can carry, never the separator itself.
_ENTRIES = st.lists(
    st.text(alphabet="abcdefghij/_-.", min_size=1, max_size=8), max_size=5, unique=True
)


@given(current=_ENTRIES, leading=_ENTRIES)
def test_a_path_list_leads_with_what_it_lacked_and_keeps_what_it_had(
    current: list[str], leading: list[str]
) -> None:
    environ = {"LIST": os.pathsep.join(current)} if current else {}
    prepended(environ, "LIST", [Path(entry) for entry in leading])
    once = dict(environ)
    prepended(environ, "LIST", [Path(entry) for entry in leading])
    assert environ == once
    entries = environ.get("LIST", "").split(os.pathsep) if environ.get("LIST") else []
    added = [str(Path(entry)) for entry in leading if str(Path(entry)) not in current]
    assert entries[: len(added)] == added
    assert set(current) <= set(entries)


def test_a_default_fills_a_missing_or_empty_variable_and_never_replaces_a_set_one() -> None:
    environ = {"SET": "mine", "EMPTY": ""}
    for name in ("SET", "EMPTY", "MISSING"):
        defaulted(environ, name, "default")
    assert environ == {"SET": "mine", "EMPTY": "default", "MISSING": "default"}


def test_pip_cuda_wheel_libraries_lead_the_loader_path(tmp_path: Path) -> None:
    """A binary linking a wheel's library outside torch's own RPATH otherwise cannot load it."""
    site = tmp_path / "lib" / "python3.14" / "site-packages"
    for library in ("nvidia/cublas/lib", "nvidia/nccl/lib", "libcudf/lib64"):
        (site / library).mkdir(parents=True)
    (site / "nvidia" / "stub").mkdir()
    (site / "nvidia" / "stub" / "lib").write_text("not a directory", encoding="utf-8")
    environ = {"LD_LIBRARY_PATH": "/opt/host"}
    WheelLibraries().apply(environ, tmp_path)
    assert environ["LD_LIBRARY_PATH"].split(os.pathsep) == [
        str(site / "nvidia/cublas/lib"),
        str(site / "nvidia/nccl/lib"),
        str(site / "libcudf/lib64"),
        "/opt/host",
    ]
    bare: dict[str, str] = {}
    WheelLibraries().apply(bare, tmp_path / "elsewhere")
    assert bare == {}


def test_a_prefixs_build_search_paths_are_exported_only_where_they_exist(tmp_path: Path) -> None:
    (tmp_path / "lib" / "pkgconfig").mkdir(parents=True)
    (tmp_path / "lib" / "libclang.so.18").write_text("", encoding="utf-8")
    environ: dict[str, str] = {}
    BuildPaths().apply(environ, tmp_path)
    assert environ == {
        "PKG_CONFIG_PATH": str(tmp_path / "lib" / "pkgconfig"),
        "LIBCLANG_PATH": str(tmp_path / "lib"),
    }
    nothing: dict[str, str] = {}
    BuildPaths().apply(nothing, tmp_path / "empty")
    assert nothing == {}


@pytest.mark.parametrize(
    ("scheduled", "mounted", "moved"),
    [({"PBS_JOBID": "7.opbs"}, False, True), ({}, True, True), ({}, False, False)],
    ids=["a scheduled job", "a node with a local mount", "a laptop"],
)
def test_compile_caches_move_to_node_local_scratch_only_on_a_node(
    tmp_path: Path, scheduled: dict[str, str], mounted: bool, moved: bool
) -> None:
    """A laptop's temporary directory is not node-local scratch, however writable it is."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    mount = tmp_path / "local"
    if mounted:
        mount.mkdir()
    environ = {"LOCALDIR": "", "TMPDIR": str(scratch), **scheduled}
    CompileCaches(mount).apply(environ, tmp_path)
    assert ("TRITON_CACHE_DIR" in environ) is moved
    if moved:
        assert environ["TORCHINDUCTOR_CACHE_DIR"] == str(scratch / "torchinductor")
        assert (scratch / "triton").is_dir() and (scratch / "torchinductor").is_dir()


def test_a_scheduled_node_with_no_writable_scratch_keeps_its_caches(tmp_path: Path) -> None:
    environ = {"PBS_JOBID": "7", "LOCALDIR": str(tmp_path / "missing")}
    CompileCaches(tmp_path / "local").apply(environ, tmp_path)
    assert "TRITON_CACHE_DIR" not in environ


def test_a_cache_the_caller_placed_is_kept_even_where_it_cannot_be_made(tmp_path: Path) -> None:
    """Making the directory is best effort; the compiler reports a bad one when it writes."""
    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")
    environ = {
        "LOCALDIR": str(tmp_path),
        "TRITON_CACHE_DIR": str(blocker / "triton"),
    }
    CompileCaches(tmp_path / "local").apply(environ, tmp_path)
    assert environ["TRITON_CACHE_DIR"] == str(blocker / "triton")
    assert environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "torchinductor")


def test_the_runtime_reports_only_what_it_changes_as_lines_a_shell_evaluates(
    tmp_path: Path,
) -> None:
    (tmp_path / "lib" / "pkgconfig").mkdir(parents=True)
    environ = {"PKG_CONFIG_PATH": "/opt/it's here", "UNRELATED": "x"}
    runtime = Runtime(tmp_path)
    changed = runtime.changes(environ)
    assert changed == {
        "PKG_CONFIG_PATH": os.pathsep.join([str(tmp_path / "lib" / "pkgconfig"), "/opt/it's here"])
    }
    assert runtime.shell(environ) == (
        f"export PKG_CONFIG_PATH={shlex.quote(changed['PKG_CONFIG_PATH'])}\n"
    )
    assert environ == {"PKG_CONFIG_PATH": "/opt/it's here", "UNRELATED": "x"}


def test_the_module_prints_for_the_prefix_the_shell_entered_and_nothing_without_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "lib" / "pkgconfig").mkdir(parents=True)
    monkeypatch.delenv("PKG_CONFIG_PATH", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    main()
    assert capsys.readouterr().out.startswith("export PKG_CONFIG_PATH=")
    monkeypatch.delenv("CONDA_PREFIX")
    runpy.run_path(activation.__file__, run_name="__main__")
    assert capsys.readouterr().out == ""
