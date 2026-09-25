import json
import os
import shlex
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from plumbum import local

from mainboard import MissionError
from mainboard.engines.compile import Provisioner, SecondStage, task_line
from mainboard.engines.compile.backend import CommandResult, Pixi
from mainboard.engines.compile.compiler import Compiler
from mainboard.engines.compile.generated import GeneratedFiles
from mainboard.engines.compile.state import SyncState

if TYPE_CHECKING:
    from pytest_subprocess import FakeProcess

    from mainboard.manifest import Manifest

_BARE = '[workspace]\nname = "w"\n'
_NODE = '[workspace]\nname = "w"\n[nodejs.deps]\nprettier = ">=3"\n'
_PINNED = '[workspace]\nname = "w"\nplatforms = ["linux-64"]\n'
_WRAPPED = "pixi run --manifest-path .mainboard/envs/{env}/pixi.toml --frozen"


def _solvable(provisioner: Provisioner, environment: str = "default") -> None:
    """Seed the lock a real `pixi install --resolve` would leave behind as it solves.

    The fake process writes nothing, so the recursive locked re-verify that follows a solve
    would find no lock to check.
    """
    provisioner.environment_dir(environment).mkdir(parents=True, exist_ok=True)
    provisioner.pixi_for(environment).lock.write_text("version: 7\n")


def test_provision_compiles_and_installs_under_one_lock(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    fp: FakeProcess,
    solver_version: str,
) -> None:
    """A second provision recompiles nothing.

    The writer is a no-op once the generated file already matches.
    """
    provisioner = Provisioner(tmp_path, manifest_from(_PINNED))
    assert provisioner.out == tmp_path / ".mainboard"
    assert provisioner.pixi.manifest == provisioner.out / "envs" / "default" / "pixi.toml"
    assert provisioner.stage is provisioner.compiler.stage
    assert provisioner.artifact == provisioner.artifact_for("default")
    _solvable(provisioner)
    for _ in range(3):
        fp.register([fp.any()], stdout="environment ready\n")

    provisioner.provision(resolve=True)
    compiled = provisioner.pixi.manifest.read_text()
    provisioner.provision()

    assert provisioner.pixi.manifest.read_text() == compiled
    state = SyncState.load(provisioner.environment_dir())
    assert state.environment == "default"
    assert state.compiled_from == provisioner.compiler.digest()


def test_run_and_capture_recompile_a_stale_environment_before_delegating(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both execution paths repair generated state before the backend can observe it."""
    observed: list[tuple[str, tuple[str, ...], str, float | None]] = []

    def run(pixi: Pixi, command: Sequence[str], env: str = "default") -> int:
        observed.append(("run", tuple(command), env, None))
        return 17

    def capture(
        pixi: Pixi,
        command: Sequence[str],
        env: str = "default",
        *,
        timeout: float | None = None,
    ) -> CommandResult:
        observed.append(("capture", tuple(command), env, timeout))
        return CommandResult(19, "out", "err")

    monkeypatch.setattr(Pixi, "run", run)
    monkeypatch.setattr(Pixi, "capture", capture)
    provisioner = Provisioner(tmp_path, manifest_from(_BARE))
    pixi = provisioner.pixi
    pixi.manifest.parent.mkdir(parents=True)
    pixi.manifest.write_text("stale")

    assert provisioner.run(("tool", "run")) == 17
    assert provisioner.run(("tool", "fresh")) == 17

    edited = Provisioner(tmp_path, manifest_from(f'{_BARE}[deps]\nripgrep = "*"\n'))
    assert edited.capture(("tool", "capture"), timeout=3.0) == CommandResult(19, "out", "err")
    assert edited.capture(("tool", "fresh-capture")) == CommandResult(19, "out", "err")
    assert observed == [
        ("run", ("tool", "run"), "default", None),
        ("run", ("tool", "fresh"), "default", None),
        ("capture", ("tool", "capture"), "default", 3.0),
        ("capture", ("tool", "fresh-capture"), "default", None),
    ]


def test_a_local_run_normalizes_the_working_directory_to_the_workspace_root(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`mainboard run` promises the repo root, whichever subdirectory it was typed in.

    An ad-hoc command takes the cwd it is started in, so a command typed from a package
    directory used to run there while a declared task still ran from the compiled root: the same
    verb, two working directories, and the manifest's own contract broken for exactly the half
    nobody had a task for.
    """
    seen: list[Path] = []
    monkeypatch.setattr(Pixi, "run", lambda self, command, env="default": _cwd(seen))
    monkeypatch.setattr(
        Pixi,
        "capture",
        lambda self, command, env="default", *, timeout=None: CommandResult(_cwd(seen), "", ""),
    )
    provisioner = Provisioner(tmp_path, manifest_from(_BARE))
    inside = tmp_path / "packages" / "mainboard"
    inside.mkdir(parents=True)
    with local.cwd(str(inside)):
        provisioner.run(("python", "-c", "pass"))
        provisioner.capture(("python", "-c", "pass"))
        assert Path.cwd() == inside
    assert seen == [tmp_path, tmp_path]


def _cwd(seen: list[Path]) -> int:
    """Record the directory a pixi child would have inherited, as its own exit code stand-in."""
    seen.append(Path.cwd())
    return 0


@pytest.mark.parametrize(
    ("edit", "installs"),
    [
        pytest.param('[tasks]\ncheck = "python -m hooks.cli"\n', 1, id="task-only"),
        pytest.param('[env]\nACTIVE = "new"\n', 1, id="activation-only"),
        pytest.param('[deps]\nripgrep = "*"\n', 2, id="dependency"),
    ],
)
def test_entering_an_environment_brings_it_in_line_with_its_lock_once(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    edit: str,
    installs: int,
) -> None:
    """pixi updates a prefix on the way into every command, which is a race for a whole wave.

    Nine jobs starting together out of one pinned tree share the prefix, so each decides for
    itself that it needs updating and the losers meet it mid-write (miyabi-g, 2026-09-05). The
    update is taken here instead, inside the lock that already serializes the compile, and
    stamped with the lock and manifest it was taken against so nothing repeats it until one of
    them moves.
    """
    synced: list[str] = []
    activated: list[str] = []
    monkeypatch.setattr(Pixi, "run", lambda self, command, env="default": 0)
    monkeypatch.setattr(
        Pixi,
        "capture",
        lambda self, command, env="default", *, timeout=None: CommandResult(0, "", ""),
    )
    monkeypatch.setattr(Pixi, "ready", lambda self, env: True)
    monkeypatch.setattr(Pixi, "sync", lambda self, env: synced.append(env))
    monkeypatch.setattr(Pixi, "cache_windows_activation", lambda self, env: activated.append(env))
    provisioner = Provisioner(tmp_path, manifest_from(_BARE))
    provisioner.recompiled()
    provisioner.pixi.lock.write_text("version: 7\n", encoding="utf-8")

    provisioner.run(("python", "-c", "pass"))
    provisioner.run(("python", "-c", "pass"))
    provisioner.capture(("python", "-c", "pass"))

    assert synced == ["default"]
    previously_compiled = provisioner.pixi.manifest.read_text()
    edited = Provisioner(tmp_path, manifest_from(_BARE + edit))
    edited.run(("python", "-c", "pass"))
    assert synced == ["default"] * installs
    assert activated == ["default"]
    assert edited.pixi.manifest.read_text() != previously_compiled
    assert not edited.compiler.stale()
    # Lock changes still reconcile even when a task or activation edit did not.
    edited.pixi.lock.write_text("version: 8\n", encoding="utf-8")
    edited.run(("python", "-c", "pass"))
    assert synced == ["default"] * (installs + 1)


def test_local_resolver_metadata_refreshes_an_unchanged_manifest_prefix(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest alone cannot see an editable project's changed requirements."""
    synced: list[str] = []
    monkeypatch.setattr(Pixi, "run", lambda self, command, env="default": 0)
    monkeypatch.setattr(Pixi, "ready", lambda self, env: True)
    monkeypatch.setattr(Pixi, "sync", lambda self, env: synced.append(env))
    metadata = tmp_path / "packages" / "local" / "pyproject.toml"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('[project]\nname = "local"\n', encoding="utf-8")
    manifest = manifest_from(
        _BARE + '[python.deps]\nlocal = {path = "packages/local", editable = true}\n'
    )
    provisioner = Provisioner(tmp_path, manifest)
    provisioner.recompiled()
    provisioner.pixi.lock.write_text("version: 7\n", encoding="utf-8")
    provisioner.run(("python", "-c", "pass"))
    metadata.write_text('[project]\nname = "local"\ndependencies = ["numpy"]\n', encoding="utf-8")
    provisioner.run(("python", "-c", "pass"))
    provisioner.run(("python", "-c", "pass"))
    assert synced == ["default", "default"]


def test_an_environment_nothing_installed_is_never_synced_on_the_way_in(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A command there is refused by the activation, which names the install to run.

    Syncing would answer that question with pixi's words instead, and a workspace with no lock
    has nothing to be brought in line with at all.
    """
    synced: list[str] = []
    monkeypatch.setattr(Pixi, "run", lambda self, command, env="default": 0)
    monkeypatch.setattr(Pixi, "sync", lambda self, env: synced.append(env))
    monkeypatch.setattr(Pixi, "ready", lambda self, env: False)
    provisioner = Provisioner(tmp_path, manifest_from(_BARE))
    provisioner.pixi.manifest.parent.mkdir(parents=True)
    provisioner.pixi.lock.write_text("version: 7\n", encoding="utf-8")
    provisioner.run(("python", "-c", "pass"))
    assert synced == []

    monkeypatch.setattr(Pixi, "ready", lambda self, env: True)
    provisioner.pixi.lock.unlink()
    provisioner.run(("python", "-c", "pass"))
    assert synced == []


def test_running_after_a_manifest_edit_retakes_the_activation_the_recompile_invalidated(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recompile makes the generated manifest newer than the cached Windows activation.

    That cache is read as stale the moment it is older than the manifest, so an edit followed by
    an ordinary `run` refused every command until someone reinstalled the whole environment. The
    cache is Pixi's own answer about a prefix that is already installed, so it is retaken beside
    the recompile, and only a prefix that is genuinely not installed still refuses.
    """
    observed: list[str] = []
    monkeypatch.setattr(Pixi, "run", lambda self, command, env="default": 0)
    monkeypatch.setattr(
        Pixi, "cache_windows_activation", lambda self, env: observed.append("activation")
    )
    monkeypatch.setattr(Pixi, "ready", lambda self, env: True)
    provisioner = Provisioner(tmp_path, manifest_from(_BARE))
    provisioner.pixi.manifest.parent.mkdir(parents=True)
    provisioner.pixi.manifest.write_text("stale")

    provisioner.run(("python", "-c", "pass"))
    assert observed == ["activation"]
    # Nothing was stale the second time, so nothing was recompiled and nothing was retaken.
    provisioner.run(("python", "-c", "pass"))
    assert observed == ["activation"]

    monkeypatch.setattr(Pixi, "ready", lambda self, env: False)
    edited = Provisioner(tmp_path, manifest_from(f'{_BARE}[deps]\nripgrep = "*"\n'))
    edited.run(("python", "-c", "pass"))
    assert observed == ["activation"]


def test_a_ready_environment_caches_its_windows_activation_after_provisioning(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache is made only after every installer has completed successfully."""
    observed: list[str] = []
    monkeypatch.setattr(Compiler, "write", lambda self, files: observed.append("compile"))
    monkeypatch.setattr(
        Compiler,
        "install_locked",
        lambda self, files, *, resolve: observed.append("pixi"),
    )
    monkeypatch.setattr(
        SecondStage, "install", lambda self, env, *, resolve=False: observed.append("second-stage")
    )
    monkeypatch.setattr(Pixi, "ready", lambda self, env: True)
    monkeypatch.setattr(
        Pixi, "cache_windows_activation", lambda self, env: observed.append("activation")
    )

    Provisioner(tmp_path, manifest_from(_BARE)).provision()

    assert observed == ["compile", "pixi", "second-stage", "activation"]


def test_each_environment_compiles_into_an_independent_selected_manifest_shard(
    manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    """Default, inherited and isolated environments carry exactly their active scopes."""
    provisioner = Provisioner(
        tmp_path,
        manifest_from(
            """
            [workspace]
            name = "w"
            [deps]
            root = "*"
            [python.deps]
            local-root = { path = "packages/root", editable = true }
            [dev.deps]
            devtool = "*"
            [tasks]
            check = "python -m pytest"
            [envs.serving.deps]
            server = "*"
            [envs.isolated]
            no-default = true
            [envs.isolated.deps]
            kernel = "*"
            """
        ),
    )

    with GeneratedFiles(directory=provisioner.out).locked() as files:
        for environment in ("default", "serving", "isolated"):
            provisioner.compiler_for(environment).write(files)

    documents = {
        environment: tomllib.loads(
            provisioner.pixi_for(environment).manifest.read_text(encoding="utf-8")
        )
        for environment in ("default", "serving", "isolated")
    }
    assert set(documents["default"].get("dependencies", {})) == {"root"}
    assert set(documents["default"]["feature"]) == {"dev"}
    assert "serving" not in documents["default"].get("feature", {})
    assert set(documents["serving"].get("dependencies", {})) == {"root"}
    assert set(documents["serving"]["feature"]) == {"serving"}
    assert documents["serving"]["environments"] == {"serving": {"features": ["serving"]}}
    assert "dependencies" not in documents["isolated"]
    assert set(documents["isolated"]["feature"]) == {"isolated"}
    assert documents["isolated"]["environments"] == {
        "isolated": {"features": ["isolated"], "no-default-feature": True}
    }
    assert documents["default"]["pypi-dependencies"]["local-root"]["path"] == (
        "../../../packages/root"
    )
    assert documents["default"]["tasks"]["check"]["cwd"] == "../../.."
    assert provisioner.pixi_for("serving").env_prefix("serving") == (
        provisioner.out / "envs" / "serving" / ".pixi" / "envs" / "serving"
    )
    assert provisioner.artifact_for("serving")[:3] == (
        ".mainboard/envs/serving/pixi.toml",
        ".mainboard/envs/serving/pixi.lock",
        ".mainboard/envs/serving/state.toml",
    )


@pytest.mark.parametrize(
    "environment",
    [
        "../escape",
        "a/b",
        r"a\b",
        "a:b",
        ".hidden",
        "trailing.",
        "CON",
        "con.txt",
        "LPT9",
    ],
)
def test_an_environment_name_cannot_escape_or_alias_its_portable_shard_directory(
    environment: str, manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    """One conservative path-segment contract is enforced before any path is constructed."""
    manifest = manifest_from(f'[workspace]\nname = "w"\n[envs.{json.dumps(environment)}]\n')
    with pytest.raises(MissionError, match="cannot name a generated directory"):
        Provisioner(tmp_path, manifest).environment_dir(environment)
    assert not (tmp_path / ".mainboard").exists()


def test_environment_names_that_are_portable_segments_keep_their_logical_spelling(
    manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    """Dots, underscores and hyphens remain available for ordinary logical names."""
    provisioner = Provisioner(
        tmp_path, manifest_from('[workspace]\nname = "w"\n[envs."py3.14_cuda-13"]\n')
    )
    assert provisioner.environment_dir("py3.14_cuda-13") == (
        tmp_path / ".mainboard" / "envs" / "py3.14_cuda-13"
    )


def test_task_wrapping_validates_the_environment_before_interpolating_its_manifest_path(
    manifest_from: Callable[[str], Manifest],
) -> None:
    """Remote command staging cannot manufacture an escaped shard path either."""
    manifest = manifest_from('[workspace]\nname = "w"\n[tasks]\ncheck = "pytest"\n')
    with pytest.raises(MissionError, match="cannot name a generated directory"):
        task_line(manifest, "check", env="../escape")


def test_environment_names_cannot_alias_on_a_case_insensitive_filesystem(
    manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    """The same manifest keeps one shard per environment on all three operating systems."""
    manifest = manifest_from('[workspace]\nname = "w"\n[envs.Train]\n[envs.train]\n')
    with pytest.raises(MissionError, match="case-insensitive filesystem"):
        Provisioner(tmp_path, manifest)


def test_activated_never_compiles_a_workspace_that_was_never_provisioned(
    manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    """First-time compilation is `provision`'s job, `activated` only catches up a stale one."""
    provisioner = Provisioner(tmp_path, manifest_from(_BARE))

    with provisioner.activated():
        pass

    assert not provisioner.pixi.manifest.exists()


def test_activated_recompiles_a_provisioned_env_that_has_gone_stale(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    fp: FakeProcess,
    solver_version: str,
) -> None:
    """`activated` only compiles under its lock, it never touches `pixi install`."""
    provisioner = Provisioner(tmp_path, manifest_from(_BARE))
    _solvable(provisioner)
    for _ in range(2):
        fp.register([fp.any()], stdout="environment ready\n")
    provisioner.provision(resolve=True)
    settled = len(fp.calls)

    edited = Provisioner(tmp_path, manifest_from(f'{_BARE}[deps]\nripgrep = "*"\n'))
    with edited.activated():
        pass

    assert len(fp.calls) == settled  # no further pixi invocation happened during `activated`
    assert "ripgrep" in edited.pixi.manifest.read_text()


def test_activated_puts_a_second_stage_toolchains_binaries_ahead_of_the_env(
    manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    """A tool npm installed is reachable by name exactly like a conda one."""
    provisioner = Provisioner(tmp_path, manifest_from(_NODE))
    linked = provisioner.environment_dir() / "node_modules" / ".bin"
    linked.mkdir(parents=True)
    env_bin = provisioner.pixi.env_prefix("default") / ("Scripts" if os.name == "nt" else "bin")
    env_bin.mkdir(parents=True)

    with provisioner.activated():
        assert local.env["PATH"].startswith(
            os.pathsep.join(
                (str(linked), str(provisioner.pixi.env_prefix("default")), str(env_bin))
            )
            if os.name == "nt"
            else os.pathsep.join((str(linked), str(env_bin), ""))
        )


def test_activated_leaves_out_a_directory_nothing_has_installed_into(
    manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    """An environment provisioned without a `[nodejs]` table exports no dead PATH entry."""
    provisioner = Provisioner(tmp_path, manifest_from(_BARE))
    before = local.env["PATH"]

    with provisioner.activated():
        assert local.env["PATH"] == before


@pytest.mark.parametrize(
    ("modules", "loaded"),
    [
        pytest.param({}, False, id="no-modules-leaves-the-surrounding-stack-alone"),
        pytest.param(
            {"singularity": "4.2.1"}, True, id="a-per-host-map-is-loaded-by-name-and-version"
        ),
    ],
)
def test_activate_writes_the_script_a_bare_shell_gets_the_whole_runtime_from(
    modules: dict[str, str],
    *,
    loaded: bool,
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    fp: FakeProcess,
) -> None:
    """A remote job sourcing `activate.sh` gets the same PATH `activated` builds in process."""
    provisioner = Provisioner(tmp_path, manifest_from(_NODE))
    linked = provisioner.environment_dir() / "node_modules" / ".bin"
    linked.mkdir(parents=True)
    fp.register([fp.any()], stdout="export PATH=/env/bin:$PATH\n")

    path = provisioner.activate(modules=modules)

    assert path == provisioner.out / "activate.sh"
    text = path.read_text()
    assert ("module load singularity/4.2.1" in text) is loaded
    assert "export PATH=/env/bin:$PATH" in text
    assert str(linked) in text


def test_the_generated_activation_is_bash_wherever_it_was_written(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    fp: FakeProcess,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`activate.sh` is sourced by bash whatever machine wrote it, so its PATH is colon-joined.

    Joining with the writing machine's own separator put a Windows `;` into a script only bash
    reads, which is a PATH of one unusable entry.
    """
    monkeypatch.setattr(os, "pathsep", ";")
    provisioner = Provisioner(tmp_path, manifest_from(_NODE))
    linked = provisioner.environment_dir() / "node_modules" / ".bin"
    linked.mkdir(parents=True)
    extra = provisioner.environment_dir() / "bin"
    extra.mkdir(parents=True)
    monkeypatch.setattr(Provisioner, "binaries", lambda self, env: [linked, extra])
    fp.register([fp.any()], stdout="export PATH=/env/bin:$PATH\n")

    text = provisioner.activate().read_text(encoding="utf-8")

    [exported] = [line for line in text.splitlines() if str(linked) in line]
    assert exported == f'export PATH={shlex.quote(str(linked))}:{shlex.quote(str(extra))}:"$PATH"'


def test_activate_gives_a_named_environment_its_own_script(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, fp: FakeProcess
) -> None:
    """Installing one environment must not overwrite the activation another one is sourced by."""
    provisioner = Provisioner(tmp_path, manifest_from(f"{_BARE}[envs.serving]\n"))
    fp.register([fp.any()], stdout="export PATH=/env/serving/bin:$PATH\n")

    path = provisioner.activate("serving")

    assert path == provisioner.out / "activate-serving.sh"
    assert not (provisioner.out / "activate.sh").exists()


def test_provision_installs_the_second_stage_after_pixi(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    fp: FakeProcess,
    stub_binary: Callable[[str], str],
    solver_version: str,
) -> None:
    """Every second-stage manager ships as a conda package, so pixi has to land first."""
    npm = stub_binary("npm")
    provisioner = Provisioner(tmp_path, manifest_from(f'{_BARE}[nodejs.deps]\nprettier = ">=3"\n'))
    _solvable(provisioner)
    for _ in range(3):
        fp.register([fp.any()], stdout="environment ready\n")

    provisioner.provision(resolve=True)

    assert "prettier" in (provisioner.environment_dir() / "package.json").read_text()
    assert [next(iter(call)) for call in fp.calls][-1] == npm


def test_artifact_ships_the_same_generated_inputs_that_name_the_prefix(
    manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    provisioner = Provisioner(tmp_path, manifest_from(_NODE))
    provisioner.recompiled()
    source = provisioner.environment_dir()
    with pytest.raises(MissionError, match="no npm lock"):
        _ = provisioner.artifact
    (source / "package-lock.json").write_text('{"lockfileVersion":3}\n')
    (source / "activate.sh").write_text("host-specific bookkeeping\n")
    (source / ".mainboard-synced").write_text("mutable stamp\n")
    expected = {
        *GeneratedFiles(directory=source).inputs,
        source / "pixi.lock",
        SyncState.path(source),
    }
    assert {tmp_path / name for name in provisioner.artifact} == expected
    assert source / "package-lock.json" in expected
    assert source / "package.json" in expected


def test_the_provisioner_says_which_pixi_solves_here(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, solver_version: str
) -> None:
    """A blessing records it, and `doctor` compares it against what the fleet is pinned to."""
    assert Provisioner(tmp_path, manifest_from(_BARE)).solver_version() == solver_version


def test_a_refresh_asks_the_indexes_before_installing_and_blesses_the_result(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    fp: FakeProcess,
    solver_version: str,
) -> None:
    """Satisfying the manifest and being current differ, so `update` runs before the install."""
    provisioner = Provisioner(tmp_path, manifest_from(_PINNED))
    _solvable(provisioner)
    for _ in range(3):
        fp.register([fp.any()], stdout="lock updated\n")

    provisioner.provision(refresh=True)

    assert "update" in fp.calls[0]
    blessing = SyncState.load(provisioner.environment_dir())
    assert blessing.solved_from
    assert blessing.solved_by == solver_version


@pytest.mark.parametrize(
    ("command", "env", "line"),
    [
        pytest.param(
            "lint --fix",
            "default",
            f"{_WRAPPED.format(env='default')} -e default lint --fix",
            id="a-workspace-task-goes-to-pixi-with-its-arguments",
        ),
        pytest.param(
            "serve",
            "serving",
            f"{_WRAPPED.format(env='serving')} -e serving serve",
            id="an-env-declares-its-own-tasks",
        ),
        pytest.param("serve", "default", "serve", id="another-envs-task-is-not-a-task-here"),
        pytest.param(
            "python -c 'print(1)'",
            "default",
            "python -c 'print(1)'",
            id="a-command-line-is-not-a-task-name",
        ),
        pytest.param(
            "lint",
            "undeclared",
            f"{_WRAPPED.format(env='undeclared')} -e undeclared lint",
            id="an-env-nobody-declared-still-resolves-the-workspace-tasks",
        ),
    ],
)
def test_task_line_hands_only_a_declared_task_to_pixi(
    command: str, env: str, line: str, manifest_from: Callable[[str], Manifest]
) -> None:
    """The manifest path is relative because a wrapped command already changed into the root."""
    manifest = manifest_from(
        f'{_BARE}[tasks]\nlint = "ruff check"\n[envs.serving.tasks]\nserve = "vllm serve"\n'
    )
    assert task_line(manifest, command, env=env) == line


# A workspace as a dispatch finds one: it installs itself and it declares tasks, which is the
# table an afternoon's edit touches most often and the one a compile carries into the artifact.
_DISPATCHED = """
[workspace]
name = "life"

[python.deps]
life = {{ path = ".", editable = true }}

[tasks]
tok-paper = "tectonic paper.tex"
{extra}
"""


def test_a_task_row_added_between_dispatches_refreshes_source_but_reuses_the_prefix(
    tmp_path: Path,
) -> None:
    """Dispatch still compiles fresh task definitions, but tasks cannot change installed deps."""
    from mainboard import Manifest
    from mainboard.engines.compile import digest_of

    root = tmp_path / "life"
    (root / "src").mkdir(parents=True)

    def compile_with(extra: str) -> None:
        (root / "mainboard.toml").write_text(_DISPATCHED.format(extra=extra), encoding="utf-8")
        manifest = Manifest.model_validate(
            tomllib.loads((root / "mainboard.toml").read_text(encoding="utf-8"))
        )
        Provisioner(root, manifest).recompiled("default")

    shard = root / ".mainboard" / "envs" / "default"
    compile_with("")
    (shard / "pixi.lock").write_text("version: 7\n", encoding="utf-8")
    before = digest_of(shard)
    compile_with('head-paper = "tectonic head.tex"')
    after = digest_of(shard)

    assert "head-paper" in (shard / "pixi.toml").read_text()
    assert after == before


def test_a_resolve_for_a_platform_this_machine_is_not_solves_the_lock_and_installs_nothing(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    fp: FakeProcess,
    solver_version: str,
) -> None:
    """A lock solved for a card elsewhere ships with `setup`, which installs it where it runs.

    The Windows card's environment is solved from the Linux workstation, and neither pixi nor
    the second stage is asked to install a prefix this machine could never execute.
    """
    foreign = '[workspace]\nname = "w"\nplatforms = ["linux-ppc64le"]\n'
    provisioner = Provisioner(tmp_path, manifest_from(foreign))
    fp.register([fp.any()], stdout="lock solved\n")

    provisioner.provision(resolve=True)

    assert not provisioner.runs_here()
    assert [call[1] for call in fp.calls if call[1] != "--version"] == ["lock"]
    assert SyncState.load(provisioner.environment_dir()).solved_by == solver_version


def test_a_local_run_hands_its_exports_to_pixi(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, str] | None] = []

    def run(
        pixi: Pixi,
        command: Sequence[str],
        env: str = "default",
        *,
        exports: dict[str, str] | None = None,
    ) -> int:
        seen.append(exports)
        return 0

    monkeypatch.setattr(Pixi, "run", run)
    provisioner = Provisioner(tmp_path, manifest_from(_BARE))

    assert provisioner.run(("python", "probe.py"), exports={"CELL": "a"}) == 0
    assert seen == [{"CELL": "a"}]
