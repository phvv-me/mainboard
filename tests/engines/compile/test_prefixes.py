import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mainboard import MissionError
from mainboard.engines.compile.backend import Pixi
from mainboard.engines.compile.ecosystems import SecondStage
from mainboard.engines.compile.prefixes import STAMP, Prefixes, digest_of, prefix_path
from mainboard.engines.compile.provisioner import Provisioner
from mainboard.engines.compile.state import SyncState

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pytest_subprocess import FakeProcess

    from mainboard import Manifest

_WORKSPACE = '[workspace]\nname = "w"\n'


@pytest.fixture
def artifact(tmp_path: Path, manifest_from: Callable[[str], Manifest]) -> Callable[[str], Path]:
    """A factory writing a compiled artifact whose lock pins `text`, returning its directory."""

    def make(text: str) -> Path:
        source = tmp_path / "artifacts" / text
        source.mkdir(parents=True, exist_ok=True)
        (source / "pixi.toml").write_text('[workspace]\nname = "w"\n', encoding="utf-8")
        (source / "pixi.lock").write_text(f"version: 7\n# {text}\n", encoding="utf-8")
        stage = SecondStage(tmp_path, manifest_from(_WORKSPACE), source, Pixi(source))
        SyncState.path(source).write_text(
            SyncState(runtime_from=stage.digest()).render(), encoding="utf-8"
        )
        return source

    return make


@pytest.fixture
def self_installing(artifact: Callable[[str], Path]) -> Path:
    """A compiled artifact for a workspace that installs its own root as an editable package.

    Which is how a research repository ships its own `src/`, and the one shape whose every
    declared location is written relative to the shard it was compiled into: the workspace root
    itself, the same root in the lock beside it, and `.env` in the generated dotenv loader.
    """
    source = artifact("self")
    (source / "pixi.toml").write_text(
        f'{_WORKSPACE}\n[pypi-dependencies]\nw = {{ path = "../../..", editable = true }}\n',
        encoding="utf-8",
    )
    (source / "pixi.lock").write_text(
        "version: 7\npackages:\n- pypi: ../../..\n  name: w\n", encoding="utf-8"
    )
    (source / "dotenv.sh").write_text('. "../../../.env"\n', encoding="utf-8")
    return source


@pytest.fixture
def prefixes(tmp_path: Path, manifest_from: Callable[[str], Manifest]) -> Prefixes:
    """The addressed environments of a workspace rooted in `tmp_path`."""
    return Prefixes(tmp_path, manifest_from(_WORKSPACE))


def test_an_environment_is_addressed_by_the_artifact_it_would_be_built_from(
    artifact: Callable[[str], Path], prefixes: Prefixes
) -> None:
    """The manifest and the lock decide every package that lands, so they are the identity.

    Path arithmetic only, and the same arithmetic on both sides of an ssh connection: the
    dispatcher pins a digest into a snapshot on a host it has asked nothing of yet, and the host
    builds into the very directory the job was told to activate.
    """
    one, other = artifact("one"), artifact("other")

    assert digest_of(one) != digest_of(other)
    assert digest_of(artifact("one")) == digest_of(one)
    assert prefixes.path(digest_of(one)) == Path(
        prefix_path(str(prefixes.root), "default", digest_of(one))
    )
    assert str(prefixes.path(digest_of(one))).endswith(
        f".mainboard/prefixes/default/{digest_of(one)}"
    )
    # And nothing was built by asking where it would go.
    assert not prefixes.built(digest_of(one))
    assert not prefixes.base.exists()


@pytest.mark.parametrize(
    "modules",
    [
        {"cuda": "12.9", "gcc": "14"},
        {"gcc": "14", "cuda": "12.8"},
        {"cuda": "12.8"},
    ],
)
def test_module_versions_and_load_order_change_the_environment_address(
    artifact: Callable[[str], Path], modules: Mapping[str, str]
) -> None:
    source = artifact("modules")
    assert digest_of(source, modules={"cuda": "12.8", "gcc": "14"}) != digest_of(
        source, modules=modules
    )


@pytest.mark.parametrize("name", ["package.json", "package-lock.json", "dotenv.sh", "unset.sh"])
def test_generated_install_and_activation_inputs_are_part_of_the_address(
    artifact: Callable[[str], Path], name: str
) -> None:
    source = artifact("generated")
    before = digest_of(source)
    (source / name).write_text("first\n", encoding="utf-8")
    first = digest_of(source)
    (source / name).write_text("second\n", encoding="utf-8")
    assert before != first != digest_of(source)


@pytest.mark.parametrize(
    "tasks",
    [
        '[tasks]\ncheck = "python -m hooks.cli"\n',
        '[feature.serving.tasks]\ncheck = "python -m hooks.cli"\n',
        '[target.linux-64.tasks]\ncheck = "python -m hooks.cli"\n',
    ],
)
def test_tasks_share_a_prefix_but_activation_still_changes_its_identity(
    artifact: Callable[[str], Path], tasks: str
) -> None:
    """Source snapshots own tasks; changing exported values still needs a new prefix."""
    source = artifact("tasks")
    base = (
        _WORKSPACE
        + '[feature.serving.dependencies]\nripgrep = "*"\n'
        + '[target.linux-64.dependencies]\npython = "*"\n'
    )
    (source / "pixi.toml").write_text(base, encoding="utf-8")
    before = digest_of(source)
    (source / "pixi.toml").write_text(base + tasks, encoding="utf-8")
    assert digest_of(source) == before
    (source / "pixi.toml").write_text(
        base + tasks + '[activation.env]\nACTIVE = "new"\n', encoding="utf-8"
    )
    assert digest_of(source) != before


def test_mutable_activation_and_compile_bookkeeping_do_not_change_identity(
    artifact: Callable[[str], Path],
) -> None:
    source = artifact("bookkeeping")
    before = digest_of(source)
    (source / "activate.sh").write_text("local hook\n", encoding="utf-8")
    (source / ".mainboard-synced").write_text("stamp\n", encoding="utf-8")
    state = SyncState.load(source)
    SyncState.path(source).write_text(
        state.model_copy(update={"compiled_from": "another-root", "solved_by": "0.99"}).render(),
        encoding="utf-8",
    )
    assert digest_of(source) == before


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ('[rust.deps]\nrg = "1"', '[rust.deps]\nrg = "2"'),
        (
            '[rust.deps]\nrg = {git = "https://example.org/tool", rev = "abc"}',
            '[rust.deps]\nrg = {git = "https://example.org/tool", rev = "def"}',
        ),
        (
            '[rust.deps]\nrg = {path = "packages/one", locked = true}',
            '[rust.deps]\nrg = {path = "packages/two", locked = false}',
        ),
        ('[go.deps]\n"example.org/tool" = "1.0"', '[go.deps]\n"example.org/tool" = "2.0"'),
        ('[nodejs.deps]\nprettier = "3"', '[nodejs.deps]\nprettier = "4"'),
        (
            '[nodejs]\nmanager = "npm"\n[nodejs.deps]\nprettier = "3"',
            '[nodejs]\nmanager = "pnpm"\n[nodejs.deps]\nprettier = "3"',
        ),
        (
            '[nodejs]\napp = false\n[nodejs.deps]\nprettier = "3"',
            '[nodejs]\napp = true\n[nodejs.deps]\nprettier = "3"',
        ),
        (
            '[nodejs.package]\nprivate = false\n[nodejs.deps]\nprettier = "3"',
            '[nodejs.package]\nprivate = true\n[nodejs.deps]\nprettier = "3"',
        ),
        ('[nodejs.dev]\neslint = "8"', '[nodejs.dev]\neslint = "9"'),
        ('[on.linux-64.rust.deps]\nrg = "1"', '[on.linux-64.rust.deps]\nrg = "2"'),
        ('[on.win-64.rust.deps]\nrg = "1"', '[on.win-64.rust.deps]\nrg = "2"'),
        ('[dev.rust.deps]\nrg = "1"', '[dev.rust.deps]\nrg = "2"'),
        (
            '[go.deps]\n"one.org/tool" = "1"\n"two.org/tool" = "2"',
            '[go.deps]\n"two.org/tool" = "2"\n"one.org/tool" = "1"',
        ),
    ],
)
def test_second_stage_changes_move_the_address_even_with_the_same_pixi_lock(
    tmp_path: Path,
    manifest_from: Callable[[str], Manifest],
    before: str,
    after: str,
) -> None:
    identities = []
    for declaration in (before, after):
        provisioner = Provisioner(tmp_path, manifest_from(f"{_WORKSPACE}\n{declaration}\n"))
        provisioner.recompiled()
        source = provisioner.environment_dir()
        (source / "pixi.lock").write_text("version: 7\n", encoding="utf-8")
        assert SyncState.load(source).runtime_from == provisioner.stage.digest()
        identities.append(digest_of(source))
    assert identities[0] != identities[1]


def test_a_selected_second_stage_ignores_unrelated_tasks_hosts_and_environments(
    tmp_path: Path, manifest_from: Callable[[str], Manifest]
) -> None:
    identities = []
    for version in ("1", "2"):
        manifest = manifest_from(
            f'{_WORKSPACE}\n[rust.deps]\nrg = "1"\n[tasks]\ncheck = "echo {version}"\n'
            f'[envs.other.rust.deps]\nrg = "{version}"\n'
            f'[hosts.remote.modules]\ncuda = "{version}"\n'
        )
        identities.append(Provisioner(tmp_path, manifest).stage.digest())
    assert identities[0] == identities[1]


def test_relocated_second_stage_paths_and_generated_inputs_keep_one_identity(
    tmp_path: Path, manifest_from: Callable[[str], Manifest]
) -> None:
    identities = []
    runtime_identities = []
    for name in ("workstation", "host"):
        root = tmp_path / name
        root.mkdir()
        manifest = manifest_from(
            f'{_WORKSPACE}\n[rust.deps]\nrg = {{path = "{root}/packages/tool"}}\n'
            f'[env]\nPYTHONPATH = "{root}/src"\n'
        )
        provisioner = Provisioner(root, manifest)
        provisioner.recompiled()
        source = provisioner.environment_dir()
        (source / "pixi.lock").write_text("version: 7\n", encoding="utf-8")
        identities.append(digest_of(source, modules={"cuda": "12.8"}))
        runtime_identities.append(SyncState.load(source).runtime_from)
    assert identities[0] == identities[1]
    assert runtime_identities[0] == runtime_identities[1]


def test_half_an_artifact_names_no_environment_and_says_which_command_makes_one(
    artifact: Callable[[str], Path], prefixes: Prefixes
) -> None:
    """Building from a manifest whose lock is missing is how a prefix ends up describing one
    lock and containing another."""
    source = artifact("one")
    (source / "pixi.lock").unlink()

    with pytest.raises(MissionError, match=r"pixi.lock is missing.*install --resolve"):
        digest_of(source)


def test_a_second_lock_builds_beside_the_first_and_never_into_it(
    fp: FakeProcess,
    artifact: Callable[[str], Path],
    prefixes: Prefixes,
    tool_paths: Mapping[str, str],
) -> None:
    """The whole point: a queued job's environment survives the next solve.

    A wave dispatched against one lock kept running while a second lock was installed over the
    one shared prefix, and died importing sqlite3 against a half-reconciled environment. Here
    the second lock is a second directory, and the first is untouched down to its bytes.
    """
    fp.register([tool_paths["pixi"], "shell-hook", fp.any()], stdout="export ONE=1\n")
    fp.register([fp.any()], stdout="environment ready\n", occurrences=8)
    first = prefixes.materialize(artifact("one"))
    before = {path.name: path.read_bytes() for path in first.iterdir() if path.is_file()}

    second = prefixes.materialize(artifact("other"))

    assert first != second
    assert first.parent == second.parent
    assert {path.name: path.read_bytes() for path in first.iterdir() if path.is_file()} == before
    assert (first / STAMP).read_text(encoding="utf-8").strip() == first.name
    # Each prefix keeps the artifact it was built from, so neither can be read as the other's.
    assert (first / "pixi.lock").read_text(encoding="utf-8") != (second / "pixi.lock").read_text(
        encoding="utf-8"
    )
    # And each carries the activation a job sources, naming itself rather than the mirror's
    # mutable environment.
    assert "export ONE=1" in (first / "activate.sh").read_text(encoding="utf-8")


def test_a_workspace_that_installs_itself_is_built_against_its_root_and_not_the_prefix(
    fp: FakeProcess,
    prefixes: Prefixes,
    self_installing: Path,
    tmp_path: Path,
    tool_paths: Mapping[str, str],
) -> None:
    """A compiled artifact names every location relative to the shard it was compiled into.

    A prefix is not that shard, so a verbatim copy carried `path = "../../.."` one directory
    deeper and pixi read the workspace as its own generated tree: `Failed to build
    reproducibility @ .../.mainboard`, `does not appear to be a Python project`, four Miyabi jobs
    dead on 2026-09-05, and every workspace with a self-install with them. So the copy is
    anchored where it lands, in the manifest, in the lock that records the same local source,
    and in the generated shell the activation sources by name and could not find at all.
    """
    fp.register([tool_paths["pixi"], "shell-hook", fp.any()], stdout="export ONE=1\n")
    fp.register([fp.any()], stdout="environment ready\n", occurrences=8)
    digest = digest_of(self_installing)

    built = prefixes.materialize(self_installing)

    root = tmp_path.as_posix()
    assert f'path = "{root}"' in (built / "pixi.toml").read_text(encoding="utf-8")
    assert f"- pypi: {root}\n" in (built / "pixi.lock").read_text(encoding="utf-8")
    assert (built / "dotenv.sh").read_text(encoding="utf-8") == f'. "{root}/.env"\n'
    # Nothing relative survives anywhere in the copy, whatever the artifact declared it for.
    assert not [path for path in built.iterdir() if "../.." in path.read_text(encoding="utf-8")]
    # And the environment is still addressed by what it was built from, not by where: the source
    # keeps its bytes, so the digest a dispatch pinned is the one the host arrives at.
    assert digest_of(self_installing) == digest == built.name


def test_one_artifact_is_one_environment_at_every_root_that_builds_it(
    fp: FakeProcess,
    manifest_from: Callable[[str], Manifest],
    self_installing: Path,
    tmp_path: Path,
    tool_paths: Mapping[str, str],
) -> None:
    """Two machines hold one lock at two paths, and a blessing has to travel between them.

    So the identity stays over the artifact's own bytes and the absolute root enters only the
    copy each workspace builds for itself, which is also what lets a job activate an environment
    built from a pinned tree that no longer stands.
    """
    fp.register([tool_paths["pixi"], "shell-hook", fp.any()], stdout="export ONE=1\n")
    fp.register([fp.any()], stdout="environment ready\n", occurrences=16)
    here, there = (
        Prefixes(tmp_path / name, manifest_from(_WORKSPACE)) for name in ("here", "there")
    )

    built_here, built_there = here.materialize(self_installing), there.materialize(self_installing)

    assert built_here.name == built_there.name == digest_of(self_installing)
    for prefix, root in ((built_here, here.root), (built_there, there.root)):
        assert f'path = "{root.as_posix()}"' in (prefix / "pixi.toml").read_text(encoding="utf-8")


def test_an_environment_already_built_is_answered_and_never_built_again(
    fp: FakeProcess,
    artifact: Callable[[str], Path],
    prefixes: Prefixes,
    tool_paths: Mapping[str, str],
) -> None:
    """Which is what lets every job of a wave call this on the way in without a race."""
    fp.register([tool_paths["pixi"], "shell-hook", fp.any()], stdout="export ONE=1\n")
    fp.register([fp.any()], stdout="environment ready\n", occurrences=8)
    source = artifact("one")
    built = prefixes.materialize(source)
    spent = len(fp.calls)

    assert prefixes.materialize(source) == built
    assert len(fp.calls) == spent
    assert prefixes.built(digest_of(source))


def test_module_stacks_build_separate_activations_without_rewriting_the_old_prefix(
    fp: FakeProcess,
    artifact: Callable[[str], Path],
    prefixes: Prefixes,
    tool_paths: Mapping[str, str],
) -> None:
    fp.register(
        [tool_paths["pixi"], "shell-hook", fp.any()], stdout="export ONE=1\n", occurrences=2
    )
    fp.register([fp.any()], stdout="environment ready\n", occurrences=16)
    source = artifact("modules")
    first = prefixes.materialize(source, modules={"cuda": "12.8"})
    activation = (first / "activate.sh").read_bytes()
    second = prefixes.materialize(source, modules={"cuda": "12.9"})
    assert first != second
    assert first.name == digest_of(source, modules={"cuda": "12.8"})
    assert second.name == digest_of(source, modules={"cuda": "12.9"})
    assert (first / "activate.sh").read_bytes() == activation
    assert b"cuda/12.8" in activation
    assert "cuda/12.9" in (second / "activate.sh").read_text(encoding="utf-8")


@pytest.mark.parametrize("missing", [STAMP, "activate.sh", "pixi.toml", "pixi.lock"])
def test_a_stamp_cannot_vouch_for_a_prefix_missing_a_required_file(
    prefixes: Prefixes, missing: str
) -> None:
    target = prefixes.path("incomplete")
    target.mkdir(parents=True)
    for name in (STAMP, "activate.sh", "pixi.toml", "pixi.lock"):
        if name != missing:
            (target / name).write_text("incomplete\n", encoding="utf-8")
    assert not prefixes.built("incomplete")


def test_materialize_refuses_a_runtime_not_recorded_by_the_compiled_artifact(
    artifact: Callable[[str], Path], prefixes: Prefixes
) -> None:
    source = artifact("unrecorded")
    SyncState.path(source).write_text(SyncState().render(), encoding="utf-8")
    with pytest.raises(MissionError, match="selected second-stage runtime"):
        prefixes.materialize(source)
    assert not prefixes.base.exists()


def test_materialize_refuses_second_stage_declarations_changed_after_compile(
    artifact: Callable[[str], Path],
    prefixes: Prefixes,
    manifest_from: Callable[[str], Manifest],
) -> None:
    source = artifact("changed-runtime")
    changed = Prefixes(prefixes.root, manifest_from(f'{_WORKSPACE}\n[rust.deps]\nrg = "2"\n'))
    with pytest.raises(MissionError, match="selected second-stage runtime"):
        changed.materialize(source)
    assert not prefixes.base.exists()


def test_an_interrupted_build_is_never_mistaken_for_a_finished_one(
    artifact: Callable[[str], Path], prefixes: Prefixes
) -> None:
    """A prefix is safe to activate when it is stamped, not when its directory exists."""
    digest = digest_of(artifact("one"))
    (prefixes.path(digest) / ".pixi").mkdir(parents=True)

    assert not prefixes.built(digest)


def test_prune_keeps_every_environment_a_pinned_tree_still_names(
    prefixes: Prefixes, tmp_path: Path
) -> None:
    """A queued job's tree is what proves its environment is in use, whatever its age.

    So the sweep that drops the source trees nothing is owed from drops the prefixes with them,
    and never the one a wave that has not started yet will activate. The newest `KEEP` stand
    unconditionally, since the wave running now and the wave just dispatched are both live
    before anything has pinned them.
    """
    for age, name in enumerate(("kept", "stale", "recent", "newest")):
        (prefixes.base / name).mkdir(parents=True)
        os.utime(prefixes.base / name, (1_000_000 + age, 1_000_000 + age))
    sources = tmp_path / ".mainboard" / "dispatch" / "sources"
    pinned = sources / "abc" / ".mainboard" / "envs" / "default"
    pinned.mkdir(parents=True)
    (pinned / ".pixi").symlink_to(prefixes.path("kept") / ".pixi")

    live = prefixes.referenced(sources)
    dropped = prefixes.prune(live=live)

    assert live == {"kept"}
    # The oldest environment on the host survives because a tree still activates it, and the one
    # a moment younger, which nothing names, does not.
    assert dropped == ["stale"]
    assert sorted(entry.name for entry in prefixes.base.iterdir()) == [
        "kept",
        "newest",
        "recent",
    ]


def test_prune_on_a_host_that_has_never_built_anything_is_not_an_error(
    prefixes: Prefixes,
) -> None:
    """It runs from the same sweep as the snapshot prune, against every mirrored host."""
    assert prefixes.prune(live=()) == []
