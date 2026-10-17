"""An environment whose CUDA libraries come from the host is found before a kernel fails on it.

pedro-cvlab, 2026-10-17: torch's cuDNN 9.20 wheel ships eight of the ten libraries, so the loader
took `libcudnn_engines_tensor_ir` from the apt copy (9.27) and every convolution raised
`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`. The trees below are that machine in miniature: real
ELF headers, no CUDA.
"""

import json
from pathlib import Path
from struct import pack

from mainboard.board import Board
from mainboard.core.project import Project
from mainboard.core.section import Verdict
from mainboard.libs import Kind, LibraryAudit, Raise, Relink
from mainboard.libs.elf import Elf
from mainboard.libs.loader import Loader

_NEEDED, _RPATH, _RUNPATH = 1, 15, 29
_SITE = Path("lib/python3.14/site-packages")
_CUDNN = _SITE / "nvidia/cudnn/lib"
_ENGINES = ("libcudnn_engines_precompiled", "libcudnn_engines_tensor_ir")


def library(
    path: Path,
    *,
    needed: tuple[str, ...] = (),
    runpath: str = "",
    rpath: str = "",
    opens: tuple[str, ...] = (),
) -> Path:
    """Write a 64-bit ELF holding only a dynamic section, whose bytes also spell `opens`."""
    strings, entries = bytearray(b"\0"), []
    for tag, value in [
        *((_NEEDED, name) for name in needed),
        (_RUNPATH, runpath),
        (_RPATH, rpath),
    ]:
        if value:
            entries.append((tag, len(strings)))
            strings += value.encode() + b"\0"
    dynamic = b"".join(pack("<qQ", tag, value) for tag, value in [*entries, (0, 0)])
    body = bytes(strings) + dynamic + b"".join(name.encode() + b"\0" for name in opens)

    def section(kind: int, offset: int, size: int, link: int) -> bytes:
        return pack("<IIQQQQIIQQ", 0, kind, 0, 0, offset, size, link, 0, 0, 0)

    headers = (
        section(0, 0, 0, 0)
        + section(3, 64, len(strings), 0)
        + section(6, 64 + len(strings), len(dynamic), 1)
    )
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    header[0x28:0x30] = pack("<Q", 64 + len(body))
    header[0x3A:0x3E] = pack("<HH", 64, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + body + headers)
    return path


def wheel(prefix: Path, name: str, version: str, files: list[Path]) -> None:
    """Record `files` (under site-packages) as the installed wheel `name`."""
    site = prefix / _SITE
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    (info / "RECORD").write_text("".join(f"{file.relative_to(site)},,\n" for file in files))


def conda(prefix: Path, name: str, version: str, files: list[Path]) -> None:
    record = prefix / "conda-meta" / f"{name}-{version}-0.json"
    record.parent.mkdir(exist_ok=True)
    listed = [str(file.relative_to(prefix)) for file in files]
    record.write_text(json.dumps({"name": name, "version": version, "files": listed}))


def cudnn(prefix: Path, version: str, *, omit: tuple[str, ...] = ()) -> Path:
    """The cuDNN wheel of `version`: the dispatcher opens the graph library and that the engines,
    each carried as a bare `libcudnn_graph.so` with the version appended when it is opened."""
    directory = prefix / _CUDNN
    opens = {
        "libcudnn": ("libcudnn_graph.so",),
        "libcudnn_graph": tuple(f"{name}.so" for name in _ENGINES),
    }
    names = ["libcudnn", "libcudnn_graph", *_ENGINES]
    made = [
        library(directory / f"{name}.so.9", runpath="$ORIGIN", opens=opens.get(name, ()))
        for name in names
        if name not in omit
    ]
    wheel(prefix, "nvidia-cudnn-cu13", version, made)
    return directory


def audit(prefix: Path, host: Path) -> LibraryAudit:
    """`prefix` as a job enters it, on a host whose cache lists every library under `host`."""
    cache = {file.name: file for file in sorted(host.glob("lib*.so*"))}
    return LibraryAudit(prefix, library_path=[prefix / _CUDNN], cache=cache, trusted=())


def test_a_family_the_wheel_leaves_incomplete_is_found_in_the_host_in_a_mixed_release(
    tmp_path: Path,
) -> None:
    prefix, host = tmp_path / "env", tmp_path / "usr-lib"
    cudnn(prefix, "9.20.0.48", omit=("libcudnn_engines_tensor_ir",))
    library(host / "libcudnn_engines_tensor_ir.so.9.27.0")
    (host / "libcudnn_engines_tensor_ir.so.9").symlink_to("libcudnn_engines_tensor_ir.so.9.27.0")

    found = {finding.kind: finding for finding in audit(prefix, host).findings()}
    section = audit(prefix, host).section()

    outside, split = found[Kind.OUTSIDE], found[Kind.SPLIT]
    assert outside.remedy == Raise(package="nvidia-cudnn-cu13", ecosystem="python")
    assert "libcudnn_engines_tensor_ir.so.9, asked by libcudnn_graph.so.9" in outside.detail
    assert "ships no copy" in outside.detail
    assert split.verdict is Verdict.FAIL
    assert split.detail == "releases mixed in one process: nvidia-cudnn-cu13 9.20.0.48, the host"
    assert section.verdict is Verdict.FAIL
    assert section.fix == f"{Project().name} add nvidia-cudnn-cu13 --lang python"


def test_siblings_opened_by_their_full_version_are_linked_inside_the_environment(
    workspace: Path,
) -> None:
    prefix, host = workspace / "env", workspace / "usr-lib"
    directory = cudnn(prefix, "9.27.0.42")
    for name in ("libcudnn_graph", *_ENGINES):
        library(host / f"{name}.so.9.27.0")

    findings = audit(prefix, host).findings()

    assert {(finding.kind, finding.verdict) for finding in findings} == {
        (Kind.OUTSIDE, Verdict.WARN)
    }
    links = audit(prefix, host).remedies(Relink)
    assert sorted(link.name for link in links) == [
        "libcudnn_engines_precompiled.so.9.27.0",
        "libcudnn_engines_tensor_ir.so.9.27.0",
        "libcudnn_graph.so.9.27.0",
    ]
    assert all(link.directory == directory for link in links)
    said = [link.apply(Board(workspace)) for link in links]
    assert all(row.detail.startswith("linked ") for row in said)
    assert links[0].apply(Board(workspace)).detail.endswith("already exists")
    assert audit(prefix, host).section().verdict is Verdict.PASS


def test_two_packages_holding_one_family_in_two_releases_are_a_split(tmp_path: Path) -> None:
    prefix = tmp_path / "env"
    nccl = library(prefix / _SITE / "nvidia/nccl/lib/libnccl.so.2")
    cublas = library(prefix / _SITE / "nvidia/cublas/lib/libcublas.so.12")
    wheel(prefix, "nvidia-nccl-cu13", "2.29.7", [nccl])
    wheel(prefix, "nvidia-cublas-cu12", "12.9.1.4", [cublas])
    conda(prefix, "nccl", "2.32.3", [library(prefix / "lib/libnccl.so.2.32.3")])

    found = {finding.family: finding for finding in audit(prefix, tmp_path / "none").findings()}

    assert found["NCCL"].verdict is Verdict.WARN
    assert found["NCCL"].remedy == Raise(package="nvidia-nccl-cu13", ecosystem="python")
    assert "nccl 2.32.3" in found["NCCL"].detail
    assert found["CUDA"].verdict is Verdict.FAIL
    assert "nvidia-cublas-cu12, nvidia-nccl-cu13" in found["CUDA"].detail


def test_the_loader_takes_a_name_from_the_first_rule_that_has_it(tmp_path: Path) -> None:
    places = {rule: tmp_path / rule for rule in ("rpath", "path", "runpath", "cache", "trusted")}
    for place in places.values():
        library(place / "libx.so.1")
    cached = {"libx.so.1": places["cache"] / "libx.so.1"}
    modern = Elf.read(
        library(tmp_path / "a.so", rpath=str(places["rpath"]), runpath=str(places["runpath"]))
    )
    legacy = Elf.read(library(tmp_path / "b.so", rpath=str(places["rpath"])))
    nothing = Elf(path=tmp_path)
    trusted = [places["trusted"]]

    def via(loader: Loader, asker: Elf) -> str | None:
        found = loader.find("libx.so.1", asker)
        return found.via if found else None

    assert via(Loader([places["path"]], cached, trusted), modern) == "LD_LIBRARY_PATH"
    assert via(Loader([places["path"]], cached, trusted), legacy) == "RPATH"
    assert via(Loader([], cached, trusted), modern) == "RUNPATH"
    assert via(Loader([], cached, trusted), nothing) == "ld.so.cache"
    assert via(Loader([], {}, trusted), nothing) == "default directories"
    assert via(Loader([], {}, []), nothing) is None


def test_an_elf_reads_its_dynamic_section_and_anything_else_reads_as_empty(tmp_path: Path) -> None:
    graph = library(
        tmp_path / "lib/libcudnn_graph.so.9",
        needed=("libcudart.so.13",),
        runpath="$ORIGIN:$ORIGIN/../other",
        opens=("libcudnn_ops.so.9.27.0", "libcudnn_ops.so", "libcudnn_graph.so.9"),
    )
    script = tmp_path / "libc.so"
    script.write_text("GROUP ( /lib/libc.so.6 )")

    read = Elf.read(graph)

    assert read.needed == ("libcudart.so.13",)
    assert read.runpath == (tmp_path / "lib", tmp_path / "lib/../other")
    assert read.mentions("libcudnn") == (
        "libcudnn_graph.so.9",
        "libcudnn_ops.so",
        "libcudnn_ops.so.9.27.0",
    )
    assert Elf.read(script) == Elf(path=script)
    assert Elf.read(tmp_path / "absent.so") == Elf(path=tmp_path / "absent.so")


def test_an_environment_whose_libraries_all_load_from_itself_passes(tmp_path: Path) -> None:
    prefix = tmp_path / "env"
    cudnn(prefix, "9.27.0.42")

    section = audit(prefix, tmp_path / "none").section()

    assert section.verdict is Verdict.PASS
    assert section.fix == ""
    assert section.detail.startswith("default: 1 CUDA families")


def test_a_linked_library_found_nowhere_fails_and_has_no_package_to_raise(tmp_path: Path) -> None:
    prefix = tmp_path / "env"
    library(prefix / _SITE / "torch/lib/libtorch_cuda.so", needed=("libnccl.so.2",))

    (finding,) = audit(prefix, tmp_path / "none").findings()

    assert (finding.kind, finding.verdict, finding.remedy) == (Kind.MISSING, Verdict.FAIL, None)
    assert finding.detail.startswith("libnccl.so.2, asked by libtorch_cuda.so, is found nowhere")
