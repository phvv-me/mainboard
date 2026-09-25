import tomllib
from pathlib import Path, PurePosixPath
from shutil import copytree

import pytest

from mainboard import Manifest, MissionError
from mainboard.engines.compile import Provisioner, digest_of
from mainboard.engines.compile.compiler import Compiler
from mainboard.engines.compile.generated import GeneratedFiles
from mainboard.engines.compile.pixi_manifest import anchored, self_installed
from mainboard.engines.compile.provisioner import environment_shard
from mainboard.engines.compile.state import SyncState
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
sample-lib = { path = "../../packages/sample_lib", editable = true }
"""

_PYPROJECT = '[project]\nname = "sample-lib"\nversion = "0.1.0"\n'

# Where the compiled artifact of the default environment spells the vendored dependency: inside
# the workspace, three parents up from the shard it is written into, on every machine.
_VENDORED = "../../../.mainboard/vendor/sample-lib"

# A lock as pixi writes one for an editable path dependency: the location, relative to the
# manifest it was solved from, and no hash of anything under it.
_LOCK = f"version: 7\npackages:\n- pypi: {_VENDORED}\n  name: sample-lib\n"


def workstation(base: Path, *, manifest: str = _MANIFEST) -> Path:
    """The workspace as it stands here: inside a monorepo, below the package it depends on."""
    root = base / "mono" / "research" / "repro"
    (root / "src").mkdir(parents=True)
    (root / "mainboard.toml").write_text(manifest, encoding="utf-8")
    source = base / "mono" / "packages" / "sample_lib"
    (source / "src" / "sample_lib").mkdir(parents=True)
    (source / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
    (source / "src" / "sample_lib" / "__init__.py").write_text("SHADE = 'ai'\n", encoding="utf-8")
    return root


def mirrored(base: Path, sent_from: Path, *, manifest: str = _MANIFEST) -> Path:
    """The same workspace as its mirror on a host, beside the monorepo's mirror rather than in it.

    `../../packages/sample_lib` names a sibling package there, which nothing ever creates.
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
        ("packages/sample_lib", False),
        ("packages/../sample_lib", False),
        ("..", True),
        ("../../packages/sample_lib", True),
        ("packages/../../sample_lib", True),
        ("/opt/sample_lib", True),
        ("C:/sample_lib", True),
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
    assert "packages/sample_lib" not in compiled
    anchor = anchored(compiled, root=root, generated_dir=environment_shard("default"))
    assert f'path = "{root}/.mainboard/vendor/sample-lib"' in anchor
    assert self_installed(compiled, generated_dir=environment_shard("default")) == [
        "",
        ".mainboard/vendor/sample-lib",
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
    copy = there / vendor_root() / "sample-lib"
    assert copy.is_dir() and not (copy / "src").is_symlink()
    assert (copy / "src" / "sample_lib" / "__init__.py").read_text() == "SHADE = 'ai'\n"


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
    source = tmp_path / "mono" / "packages" / "sample_lib"
    compile_at(root)
    vendored = root / vendor_root() / "sample-lib"

    assert vendored.is_dir() and not vendored.is_symlink()
    assert (vendored / "src").is_symlink()
    assert (vendored / "src" / "sample_lib" / "__init__.py").read_text() == "SHADE = 'ai'\n"

    (source / "src" / "sample_lib" / "__init__.py").write_text("SHADE = 'kon'\n", encoding="utf-8")
    assert (vendored / "src" / "sample_lib" / "__init__.py").read_text() == "SHADE = 'kon'\n"

    # A file added or removed at the source's own root changes what the package is, and reaches
    # the vendored directory on the next compile.
    (source / "README.md").write_text("sample_lib\n", encoding="utf-8")
    (source / "pyproject.toml").unlink()
    compile_at(root)
    assert (vendored / "README.md").read_text() == "sample_lib\n"
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
    source = tmp_path / "mono" / "packages" / "sample_lib"
    before = (digest_of(compiler.out), compiler.resolution_digest())

    (source / "src" / "sample_lib" / "__init__.py").write_text("SHADE = 'kon'\n", encoding="utf-8")
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

    with pytest.raises(MissionError, match=r"\.\./\.\./packages/sample_lib"):
        compile_at(root)


def test_a_distribution_the_manifest_stopped_declaring_leaves_the_vendored_tree(
    tmp_path: Path,
) -> None:
    """The tree holds what the manifest declares and nothing else, so no mirror ships a stray."""
    root = workstation(tmp_path)
    compile_at(root)
    assert (root / vendor_root() / "sample-lib").is_dir()

    (root / "mainboard.toml").write_text(
        '[workspace]\nname = "lab"\n\n[python.deps]\nlab = { path = ".", editable = true }\n',
        encoding="utf-8",
    )
    compile_at(root)
    assert not (root / vendor_root() / "sample-lib").exists()


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

    assert (root / vendor_root() / "sample-lib" / "pyproject.toml").is_file()


# Verbatim from research/reproducibility/.mainboard/envs/default/pixi.lock, solved by pixi
# 0.79.0 on 2026-09-06 from a manifest spelling `../../../.mainboard/vendor/atpx`. pixi kept the
# workspace's own root as the three parents it was handed and collapsed the vendored pair into
# two, which is one directory written two ways and was read as two.
_SOLVED_LOCK = """version: 7
environments:
  default:
    packages:
      linux-64:
      - conda: https://conda.anaconda.org/conda-forge/noarch/zipp-4.1.0-pyhcf101f3_0.conda
      - pypi: ../../..
      - pypi: ../../vendor/atpx
      - pypi: ../../vendor/sample-lib
packages:
- pypi: ../../..
  name: reproducibility
  requires_python: '>=3.14'
- pypi: ../../vendor/atpx
  name: atpx
  requires_dist:
  - patos>=0.0.8
- pypi: ../../vendor/sample-lib
  name: sample-lib
  requires_dist:
  - cycler>=0.12
"""

# The same lock as pixi would have written it had it kept the spelling it was handed. One
# artifact, two texts, and until 2026-09-06 two addresses.
_HANDED_LOCK = _SOLVED_LOCK.replace("../../vendor/", "../../../.mainboard/vendor/")


def test_a_prefix_resolves_the_spelling_pixi_chose_and_not_only_the_one_it_was_handed() -> None:
    """The lock pixi actually wrote, taken into a prefix on a host, must name real directories.

    `anchored` matched the exact three parents `rerooted` writes, which held for as long as the
    only local source was the workspace root itself. pixi 0.79 normalised the vendored pair to
    two parents, that token rode into the prefix untouched, and it resolved against the prefix
    rather than the workspace: `error extracting extension from
    /work/xg25g007/x10537/reproducibility/.mainboard/prefixes/default/e04076233b9a4bb3/../../vendor/atpx`,
    `Failed to update PyPI packages for environment 'default'`, and Miyabi job 3299884 died with
    no torch in the environment at all.
    """
    host = PurePosixPath("/work/xg25g007/x10537/reproducibility")
    resolved = anchored(
        _SOLVED_LOCK, root=host, generated_dir=environment_shard("default")
    ).splitlines()

    assert f"- pypi: {host}/.mainboard/vendor/atpx" in resolved
    assert f"- pypi: {host}/.mainboard/vendor/sample-lib" in resolved
    assert f"- pypi: {host}" in resolved
    assert "../.." not in "\n".join(resolved)


def test_two_spellings_of_one_vendored_directory_are_one_environment_address(
    tmp_path: Path,
) -> None:
    """A rewrite that moves no package, version or hash may not move an address.

    Which is the whole reason a digest is taken over a canonical lock rather than over the bytes
    pixi last happened to write. A location it respells is that same rewrite: two texts, one
    dependency, and a workstation and a host that pinned different environments over it.
    """
    shards = []
    for side, lock in (("solved", _SOLVED_LOCK), ("handed", _HANDED_LOCK)):
        shard = tmp_path / side / ".mainboard" / "envs" / "default"
        shard.mkdir(parents=True)
        (shard / "pixi.toml").write_text('[workspace]\nname = "lab"\n', encoding="utf-8")
        (shard / "pixi.lock").write_text(lock, encoding="utf-8")
        shards.append(shard)

    assert digest_of(shards[0]) == digest_of(shards[1])


@pytest.mark.parametrize(
    "untouched",
    [
        "a run that took 3 s ... and then stopped",
        "source /opt/site/lib/../share/env.sh",
        "- pypi: ../../../../packages/sample_lib",
        "- pypi: https://files.pythonhosted.org/packages/04/4b/h11-0.16.0-py3-none-any.whl",
    ],
)
def test_only_a_token_that_reaches_inside_the_workspace_is_ever_rewritten(
    untouched: str,
) -> None:
    """Prose, a path that merely contains a step, and one that climbs past the root all stand.

    The last is the point: a location above the workspace names something no mirror carries, and
    inventing a place for it on a host would hide exactly the fault vendoring exists to end.
    """
    assert (
        anchored(
            untouched, root=PurePosixPath("/work/lab"), generated_dir=environment_shard("default")
        )
        == untouched
    )


# Verbatim from /home/pedro/projects/.mainboard/envs/default/pixi.toml, the monorepo's own
# compiled artifact. `{{ config_root }}` renders to the manifest's directory at load time, on
# the promise the manifest states in as many words, that a host mirroring the repository
# elsewhere still gets its own root. So this one line is the machine's rather than the
# workspace's, and every other byte of the file is the same everywhere.
_ROOTED_MANIFEST = """[workspace]
name = "life"
version = "0.1.0"
channels = ["rapidsai", "conda-forge", "nvidia"]

[activation]
scripts = ["dotenv.sh", "unset.sh", "../../../scripts/activate.sh"]

[activation.env]
PYTHONPATH = "{root}:{root}/research:{root}/research/liereadout/src"
LOG_LEVEL = "INFO"

[pypi-dependencies.sample-lib]
path = "../../../packages/sample_lib"
editable = true
"""

# And from the lock beside it, which carries no machine path at all: every local source it
# records is already relative, which is why the two machines' locks are byte-identical and only
# the manifest split them.
_ROOTED_LOCK = """version: 7
platforms:
- name: linux-64-system
  subdir: linux-64
packages:
- pypi: ../../../packages/sample_lib
  name: sample-lib
"""


def test_one_workspace_compiled_on_two_machines_is_one_environment(tmp_path: Path) -> None:
    """A compile is machine-independent in everything but the root it renders, and that is fatal.

    A prefix is addressed by the content of the artifact, so the one value that is this
    machine's rather than this workspace's became an address of its own: the workstation pinned
    a4c06131efc5808c, the host recompiled its own mirror and read fc4975ef2096b9ac, and Miyabi
    jobs 3300221, 3300226, 3300241 and 3300249 all died at environment prime with no prefix ever
    built under the monorepo mirror. The reproducibility workspace, whose compiled artifact
    carries no machine path at all, went on dispatching throughout.
    """
    digests = []
    for root in (tmp_path / "home/pedro/projects", tmp_path / "work/xg25g007/x10537/projects"):
        shard = root / ".mainboard" / "envs" / "default"
        shard.mkdir(parents=True)
        (shard / "pixi.toml").write_text(
            _ROOTED_MANIFEST.format(root=root.as_posix()), encoding="utf-8"
        )
        (shard / "pixi.lock").write_text(_ROOTED_LOCK, encoding="utf-8")
        SyncState.path(shard).write_text(
            SyncState(environment="default", compiled_at=str(root)).render(), encoding="utf-8"
        )
        digests.append(digest_of(shard))

    assert digests[0] == digests[1]


def test_an_export_the_workspace_really_changed_still_moves_the_address(tmp_path: Path) -> None:
    """Writing the machine's root out is not the same as ignoring what a workspace exports.

    Stripping the activation table would have been the cheaper answer and the wrong one: a
    workspace that genuinely changes what it exports would then keep being served a prefix built
    before the change, whose own activation script was generated from the older text.
    """
    digests = []
    for level in ("INFO", "DEBUG"):
        shard = tmp_path / level / ".mainboard" / "envs" / "default"
        shard.mkdir(parents=True)
        (shard / "pixi.toml").write_text(
            _ROOTED_MANIFEST.format(root=(tmp_path / level).as_posix()).replace(
                'LOG_LEVEL = "INFO"', f'LOG_LEVEL = "{level}"'
            ),
            encoding="utf-8",
        )
        (shard / "pixi.lock").write_text(_ROOTED_LOCK, encoding="utf-8")
        SyncState.path(shard).write_text(
            SyncState(environment="default", compiled_at=str(tmp_path / level)).render(),
            encoding="utf-8",
        )
        digests.append(digest_of(shard))

    assert digests[0] != digests[1]


def test_a_host_reading_a_pinned_snapshot_addresses_what_the_mirror_compiled(
    tmp_path: Path,
) -> None:
    """A job's artifact is read out of a snapshot under the mirror, never where it was written.

    So the root it was compiled FOR cannot be the directory it is standing in: the snapshot's
    own root is `<mirror>/.mainboard/dispatch/sources/<key>`, which matches nothing in the text
    and leaves the machine's root counting as content. That is how a host still read
    fc4975ef2096b9ac against a pinned 136d5ed03c0f20a5 after the roots themselves had been
    reconciled.
    """
    mirror = tmp_path / "work/xg25g007/x10537/projects"
    for shard in (
        mirror / ".mainboard/envs/default",
        mirror / ".mainboard/dispatch/sources/7e145df6-dirty/.mainboard/envs/default",
    ):
        shard.mkdir(parents=True)
        (shard / "pixi.toml").write_text(
            _ROOTED_MANIFEST.format(root=mirror.as_posix()), encoding="utf-8"
        )
        (shard / "pixi.lock").write_text(_ROOTED_LOCK, encoding="utf-8")
        SyncState.path(shard).write_text(
            SyncState(environment="default", compiled_at=str(mirror)).render(), encoding="utf-8"
        )

    compiled, pinned = (
        digest_of(mirror / ".mainboard/envs/default"),
        digest_of(mirror / ".mainboard/dispatch/sources/7e145df6-dirty/.mainboard/envs/default"),
    )
    assert compiled == pinned
