import tomllib
from pathlib import Path
from shutil import copytree

import pytest

from mainboard import Manifest, MissionError
from mainboard.engines.compile import Provisioner, digest_of
from mainboard.engines.compile.compiler import Compiler
from mainboard.engines.compile.generated import GeneratedFiles
from mainboard.engines.compile.pixi_manifest import anchored, self_installed
from mainboard.engines.compile.provisioner import environment_shard
from mainboard.engines.compile.vendor import outside, vendor_root

# A workspace that installs itself and depends on a house package two directories above its
# root, the shape every unpublished house package arrives in. It resolves on the workstation,
# where the workspace sits inside the monorepo holding that package, and names nothing at all on
# a host, where the mirror is a sibling of the monorepo's mirror rather than a child of it.
_MANIFEST = """
[workspace]
name = "lab"

[python.deps]
lab = { path = ".", editable = true }
paleta-tsukuba = { path = "../../packages/paleta", editable = true }
"""

_PYPROJECT = '[project]\nname = "paleta-tsukuba"\nversion = "0.1.0"\n'

# Where the compiled artifact of the default environment spells the vendored dependency: inside
# the workspace, three parents up from the shard it is written into, on every machine.
_VENDORED = "../../../.mainboard/vendor/paleta-tsukuba"

# A lock as pixi writes one for an editable path dependency: the location, relative to the
# manifest it was solved from, and no hash of anything under it.
_LOCK = f"version: 7\npackages:\n- pypi: {_VENDORED}\n  name: paleta-tsukuba\n"


def workstation(base: Path, *, manifest: str = _MANIFEST) -> Path:
    """The workspace as it stands here: inside a monorepo, below the package it depends on."""
    root = base / "mono" / "research" / "repro"
    (root / "src").mkdir(parents=True)
    (root / "mainboard.toml").write_text(manifest, encoding="utf-8")
    source = base / "mono" / "packages" / "paleta"
    (source / "src" / "paleta").mkdir(parents=True)
    (source / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
    (source / "src" / "paleta" / "__init__.py").write_text("SHADE = 'ai'\n", encoding="utf-8")
    return root


def mirrored(base: Path, sent_from: Path, *, manifest: str = _MANIFEST) -> Path:
    """The same workspace as its mirror on a host, beside the monorepo's mirror rather than in it.

    `../../packages/paleta` names `<work>/packages/paleta` there, which nothing ever creates.
    What the host has instead is the transfer's dereferenced copy of the vendored tree, which is
    the one directory the compiled manifest and the lock name.
    """
    work = base / "work"
    (work / "projects").mkdir(parents=True)
    root = work / "repro"
    (root / "src").mkdir(parents=True)
    (root / "mainboard.toml").write_text(manifest, encoding="utf-8")
    copytree(sent_from / vendor_root(), root / vendor_root(), symlinks=False)
    return root


def compile_at(root: Path, environment: str = "default") -> Compiler:
    """Compile one of `root`'s environments and hand back the compiler that wrote it."""
    manifest = Manifest.model_validate(
        tomllib.loads((root / "mainboard.toml").read_text(encoding="utf-8"))
    )
    compiler = Provisioner(root, manifest).compiler_for(environment)
    with GeneratedFiles(directory=compiler.out).locked() as files:
        compiler.write(files)
    return compiler


@pytest.mark.parametrize(
    ("path", "leaves"),
    [
        (".", False),
        ("packages/paleta", False),
        ("packages/../paleta", False),
        ("..", True),
        ("../../packages/paleta", True),
        ("packages/../../paleta", True),
        ("/opt/paleta", True),
        ("C:/paleta", True),
    ],
)
def test_whether_a_declared_path_leaves_the_root_is_arithmetic_and_not_a_stat(
    path: str, leaves: bool
) -> None:
    """A host answers this the same way as the workstation, where neither location exists."""
    assert outside(path) is leaves


def test_a_dependency_that_leaves_the_root_is_compiled_at_a_location_inside_it(
    tmp_path: Path,
) -> None:
    """The declared path is the one thing in a manifest that cannot travel.

    It resolves from the workspace root, and what stands above that root is not the same tree on
    a host. So the compiled artifact never spells it: what it records is a location under the
    root, which a mirror carries like any other, which anchors into the machine's own workspace
    on the way into a prefix, and which reads back as one of this workspace's own editable
    packages when a dispatch asks what a job must import.
    """
    root = workstation(tmp_path)
    compiled = compile_at(root).pixi.manifest.read_text(encoding="utf-8")

    assert _VENDORED in compiled
    assert "packages/paleta" not in compiled
    anchor = anchored(compiled, root=root, generated_dir=environment_shard("default"))
    assert f'path = "{root}/.mainboard/vendor/paleta-tsukuba"' in anchor
    assert self_installed(compiled, generated_dir=environment_shard("default")) == [
        "",
        ".mainboard/vendor/paleta-tsukuba",
    ]


def test_a_workstation_and_a_hosts_mirror_reach_one_environment_and_one_resolution(
    tmp_path: Path,
) -> None:
    """The two numbers a dispatch turns on must not depend on which machine computed them.

    A prefix is addressed by the compiled manifest and the lock beside it, and a lock is vouched
    for by a digest over that manifest and every local project's own metadata. Both now read one
    location that exists on both machines, so a host installs from the lock the workstation
    solved instead of refusing it, and the address the dispatch pinned is the address the host
    builds.
    """
    here = workstation(tmp_path)
    solved = compile_at(here)
    solved.pixi.lock.write_text(_LOCK, encoding="utf-8")
    there = mirrored(tmp_path, here)
    landed = compile_at(there)
    landed.pixi.lock.write_text(_LOCK, encoding="utf-8")

    assert digest_of(solved.out) == digest_of(landed.out)
    assert solved.resolution_digest() == landed.resolution_digest()
    # And the host left the copy the mirror carried exactly as it found it, since there is no
    # tree above its root to link into and nothing better to say about it.
    copy = there / vendor_root() / "paleta-tsukuba"
    assert copy.is_dir() and not (copy / "src").is_symlink()
    assert (copy / "src" / "paleta" / "__init__.py").read_text() == "SHADE = 'ai'\n"


def test_the_vendored_copy_is_a_real_directory_whose_entries_track_the_source(
    tmp_path: Path,
) -> None:
    """Real so nothing resolves it elsewhere, linked so an edit needs no re-vendoring.

    A resolver handed a symlinked project root is free to record where it really went, and one
    canonicalised path would put this machine's own tree back into the lock. The entries under
    it are links, which is the whole of the editable semantics the manifest used to get by
    naming the source directly: an edit is seen by the next import, with nothing rerun.
    """
    root = workstation(tmp_path)
    source = tmp_path / "mono" / "packages" / "paleta"
    compile_at(root)
    vendored = root / vendor_root() / "paleta-tsukuba"

    assert vendored.is_dir() and not vendored.is_symlink()
    assert (vendored / "src").is_symlink()
    assert (vendored / "src" / "paleta" / "__init__.py").read_text() == "SHADE = 'ai'\n"

    (source / "src" / "paleta" / "__init__.py").write_text("SHADE = 'kon'\n", encoding="utf-8")
    assert (vendored / "src" / "paleta" / "__init__.py").read_text() == "SHADE = 'kon'\n"

    # A file added or removed at the source's own root changes what the package is, and reaches
    # the vendored directory on the next compile.
    (source / "README.md").write_text("paleta\n", encoding="utf-8")
    (source / "pyproject.toml").unlink()
    compile_at(root)
    assert (vendored / "README.md").read_text() == "paleta\n"
    assert not (vendored / "pyproject.toml").exists()


def test_a_vendored_sources_code_never_moves_an_address_but_its_metadata_does(
    tmp_path: Path,
) -> None:
    """An editable install contributes dependency metadata to a prefix, and no code.

    So a queued wave cannot be stranded by an edit under the package it imports, which is why
    that source is pinned in the tree a job runs from rather than in the environment. The one
    file a solve does read is the package's own `pyproject.toml`, and moving that has to
    invalidate the lock, because it is exactly what the lock answered to.
    """
    root = workstation(tmp_path)
    compiler = compile_at(root)
    compiler.pixi.lock.write_text(_LOCK, encoding="utf-8")
    source = tmp_path / "mono" / "packages" / "paleta"
    before = (digest_of(compiler.out), compiler.resolution_digest())

    (source / "src" / "paleta" / "__init__.py").write_text("SHADE = 'kon'\n", encoding="utf-8")
    assert (digest_of(compiler.out), compiler.resolution_digest()) == before

    (source / "pyproject.toml").write_text(_PYPROJECT.replace("0.1.0", "0.2.0"), encoding="utf-8")
    assert compiler.resolution_digest() != before[1]
    assert digest_of(compiler.out) == before[0]


def test_a_source_that_is_neither_here_nor_vendored_is_refused_by_name(tmp_path: Path) -> None:
    """Named where it was declared, rather than several layers down in pixi's own report.

    pixi answers a location that is not there with `does not appear to be a Python project`,
    about a path it had already rewritten, which says nothing about the manifest line that is
    wrong nor about the mirror that never carried the tree.
    """
    root = tmp_path / "mono" / "research" / "repro"
    (root / "src").mkdir(parents=True)
    (root / "mainboard.toml").write_text(_MANIFEST, encoding="utf-8")

    with pytest.raises(MissionError, match=r"\.\./\.\./packages/paleta"):
        compile_at(root)


def test_a_distribution_the_manifest_stopped_declaring_leaves_the_vendored_tree(
    tmp_path: Path,
) -> None:
    """The tree holds what the manifest declares and nothing else, so no mirror ships a stray."""
    root = workstation(tmp_path)
    compile_at(root)
    assert (root / vendor_root() / "paleta-tsukuba").is_dir()

    (root / "mainboard.toml").write_text(
        '[workspace]\nname = "lab"\n\n[python.deps]\nlab = { path = ".", editable = true }\n',
        encoding="utf-8",
    )
    compile_at(root)
    assert not (root / vendor_root() / "paleta-tsukuba").exists()


def test_one_environments_compile_keeps_what_another_environment_declares(
    tmp_path: Path,
) -> None:
    """Two shards write into one vendored tree, and neither may sweep the other's package.

    Which is why the roster is read off the whole manifest and never off one environment's
    projection of it, the only thing every other file in a shard is compiled from.
    """
    root = workstation(
        tmp_path,
        manifest=(
            f"{_MANIFEST}\n[envs.serving]\nno-default = true\n"
            '[envs.serving.python.deps]\nvllm = ">=0.11"\n'
        ),
    )
    compile_at(root)
    compile_at(root, "serving")

    assert (root / vendor_root() / "paleta-tsukuba" / "pyproject.toml").is_file()
