import base64
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.dispatch import Facts
from mainboard.dispatch import onboard as onboard_module
from mainboard.dispatch.onboard import Bootstrap, Onboarding, installers
from mainboard.dispatch.shells import (
    POWERSHELL,
    PosixShell,
    WindowsShell,
    dialect_for,
    encoded,
    is_windows,
    open_shell,
    plain_errors,
    quoted,
)
from mainboard.dispatch.tarball import Tarball
from mainboard.engines.compile.backend import PIXI_VERSION, WINDOWS_INSTALLER
from mainboard.manifest import HostProfile

from .support import RecordingTransport, cache, machine_with, plan
from .test_onboard import FakeDispatcher

_ROOT = "C:/Users/me/mainboard-managed"
_FACTS_JSON = '{"schema_version": 1, "hostname": "homelab", "cpu_logical_cores": 16}'
_WINDOWS_FACTS = Facts(
    name="homelab", root="C:/Users/me/projects", platform="Windows AMD64", uv="C:/uv.exe"
)


def windows_plan(**overrides: str) -> object:
    """A plan for a Windows host with a declared root and one export."""
    profile = HostProfile(kind="ssh", root=_ROOT, platform="win-64", sync={"include": ["src"]})
    return plan(host="homelab", profile=profile, exports={"HF_HUB_OFFLINE": "1"}, **overrides)


def decoded(argv: list[str]) -> str:
    return base64.b64decode(argv[-1]).decode("utf-16-le")


def test_the_dialect_follows_the_profiles_platform_family() -> None:
    assert is_windows(HostProfile(platform="win-64"))
    assert not is_windows(HostProfile(platform="linux-aarch64"))
    assert not is_windows(HostProfile())
    assert type(dialect_for(HostProfile(platform="win-64"))).__name__ == "Windows"
    assert type(dialect_for(HostProfile(platform="osx-arm64"))).__name__ == "Posix"


def test_a_powershell_script_rides_encoded_so_nothing_is_quoted_for_cmd_exe() -> None:
    script = "Write-Output 'it''s \"quoted\" & piped | fine'"
    assert base64.b64decode(encoded(script)).decode("utf-16-le") == script
    assert quoted("C:/a b's") == "'C:/a b''s'"
    assert POWERSHELL[-1] == "-EncodedCommand"


def test_the_windows_stage_sets_location_and_path_and_hands_activation_to_the_hosts_tool() -> None:
    execution = windows_plan()
    shell = WindowsShell(execution, _ROOT, ssh=RecordingTransport())
    bare = shell.stage("uv --version", activate=False)
    assert bare.startswith("$ProgressPreference = 'SilentlyContinue'; $LASTEXITCODE = 0; ")
    assert f"; Set-Location -LiteralPath '{_ROOT}' -ErrorAction Stop; " in bare
    assert (
        '$env:Path = "$HOME\\.local\\bin;$HOME\\.pixi\\bin;$HOME\\.cargo\\bin;" + $env:Path'
        in bare
    )
    assert bare.endswith("; uv --version; exit $LASTEXITCODE")
    assert "mainboard run" not in bare
    activated = shell.stage("mainboard facts --json", activate=True)
    assert (
        "$env:HF_HUB_OFFLINE = '1'; mainboard run --env default -- mainboard facts --json"
        in activated
    )


def test_the_windows_shell_runs_each_script_as_its_own_encoded_ssh_one_shot() -> None:
    transport = RecordingTransport(rules=[("broken", 1, "boom")])
    shell = WindowsShell(windows_plan(), _ROOT, ssh=transport)
    assert shell.run("uv --version") == ""
    argv = transport.calls[-1]
    assert argv[:3] == ["ssh", "-o", "BatchMode=yes"]
    assert argv[3] == "homelab"
    assert argv[4:-1] == list(POWERSHELL)
    assert transport.scripts[-1].endswith("uv --version; exit $LASTEXITCODE")
    assert shell.ok(shell.dialect.has("uv"))
    assert (
        "if (Get-Command uv -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
        in (transport.scripts[-1])
    )
    assert not shell.ok("broken")
    with pytest.raises(MissionError, match="`broken` failed on 'homelab'"):
        shell.run("broken")
    assert shell.proof == f"{_ROOT}/.mainboard/envs/default/.pixi/envs/default"
    assert shell.provisioned.startswith("if (Test-Path -LiteralPath '")
    assert shell.activation_record == ""


def test_the_windows_routes_chain_with_semicolons_and_fetch_uv_through_powershell() -> None:
    shell = WindowsShell(windows_plan(), _ROOT, ssh=RecordingTransport())
    routes = installers(shell, "packages/tool")
    assert routes.names == ["uv", "uv-bootstrap", "pip"]
    assert routes.select("uv").probe.startswith("if (Get-Command uv")
    assert routes.select("uv-bootstrap").probe == "exit 0"
    assert routes.select("uv-bootstrap").command == (
        "irm https://astral.sh/uv/install.ps1 | iex; "
        "uv tool install --force --editable packages/tool"
    )
    assert routes.select("pip").command.startswith("python -m pip install --user")
    assert " && " not in routes.select("uv-bootstrap").command
    assert shell.dialect.pixi_installer == f"{WINDOWS_INSTALLER}; $LASTEXITCODE = 0"
    assert f"PIXI_VERSION='{PIXI_VERSION}'" in WINDOWS_INSTALLER


def test_clixml_error_records_are_unwrapped_to_the_text_a_person_would_read() -> None:
    wrapped = (
        '#< CLIXML\n<Objs Version="1.1.0.1"><Obj S="progress" RefId="0"><TN RefId="0"></TN></Obj>'
        '<S S="Error">uv : The term &#39;uv&#39; is not recognized_x000D__x000A_</S>'
        '<S S="Error">At line:1 char:1_x000D__x000A_</S></Objs>'
    )
    assert plain_errors(wrapped) == "uv : The term 'uv' is not recognized\nAt line:1 char:1"
    assert plain_errors("plain text") == "plain text"


def test_the_posix_shell_keeps_its_bash_lines_and_closes_its_connection() -> None:
    host = machine_with(rules=[("broken", 1, "")])
    with PosixShell(host, plan(), "/repo") as shell:
        assert shell.ok("command -v uv")
        assert host.lines[-1].startswith("cd /repo && export PATH=")
        assert host.calls[-1][:2] == ["bash", "-lc"]
        assert shell.proof == "/repo/.mainboard/activate.sh"
        assert shell.activation_record == "/repo/.mainboard/activate.sh"
    assert shell.dialect.session("gold", "cd /x && mainboard shell")[3] == (
        "bash -lc 'cd /x && mainboard shell'"
    )


def test_open_shell_picks_the_family_off_the_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    host = machine_with()
    monkeypatch.setattr("mainboard.dispatch.shells.connection", lambda alias, ssh=None: host)
    assert isinstance(open_shell(plan(), "/repo"), PosixShell)
    assert isinstance(open_shell(windows_plan(), _ROOT), WindowsShell)


def test_a_windows_host_is_onboarded_through_powershell_without_a_queue_daemon(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Probe in PowerShell, mirror by tar, install with uv, provision with the host's own tool."""
    transport = RecordingTransport(
        rules=[
            ("pueue status", 1, ""),
            ("facts --json", 0, f"chatter\n{_FACTS_JSON}\n"),
            ("pixi --version", 0, f"pixi {PIXI_VERSION}\n"),
            ("--version", 0, "0.1.0\n"),
        ]
    )
    monkeypatch.setattr(
        onboard_module, "probe_capabilities", lambda alias, ssh=None: _WINDOWS_FACTS
    )
    monkeypatch.setattr(
        onboard_module,
        "open_shell",
        lambda execution, root, ssh=None: WindowsShell(execution, root, ssh=transport),
    )
    dispatcher = FakeDispatcher(cache())
    undeclared = plan(host="homelab", profile=HostProfile(kind="ssh", sync={"include": ["src"]}))
    setup = Onboarding(dispatcher, undeclared, root=_ROOT, digest="d1")
    with caplog.at_level("WARNING", logger="mainboard.dispatch"):
        report = setup.run()
    assert dispatcher.mirrored == [("homelab", _ROOT)]
    assert transport.ran("uv tool install --force --editable packages/mainboard")
    assert transport.ran("mainboard install default --profile homelab")
    assert transport.ran(
        f"Test-Path -LiteralPath '{_ROOT}/.mainboard/envs/default/.pixi/envs/default'"
    )
    assert transport.ran("mainboard run --env default -- mainboard facts --json")
    assert not transport.ran("pueued -d")
    assert any("answers no pueue" in message for message in caplog.messages)
    assert (report.installer, report.activate, report.tool) == ("uv", "", "0.1.0")
    assert report.capabilities is not None and report.capabilities.pixi_platform == "win-64"
    assert report.hardware is not None and report.hardware.hostname == "homelab"
    assert setup.plan.profile.platform == "win-64"
    assert dispatcher.cache.host("homelab").root == _ROOT


def test_a_windows_host_that_left_no_prefix_behind_is_refused_by_name() -> None:
    transport = RecordingTransport(rules=[("Test-Path", 1, "")])
    shell = WindowsShell(windows_plan(), _ROOT, ssh=transport)
    with pytest.raises(MissionError, match="has no C:/Users/me/mainboard-managed/.mainboard/envs"):
        Bootstrap(shell).environment()


def test_the_tarball_lists_what_rsync_would_send_and_ships_only_what_the_host_lacks(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "keep.py").write_text("print(1)\n")
    (workspace / "src" / "scratch.pyc").write_bytes(b"\x00")
    (workspace / "src" / "__pycache__").mkdir()
    (workspace / "src" / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    (workspace / ".gitignore").write_text("*.pyc\n")
    tarball = Tarball(workspace, RecordingTransport())
    from mainboard.dispatch.sync import Rsync

    files = tarball.listing(
        ["src"],
        flags=Rsync.RECURSIVE | Rsync.LINKS | Rsync.RELATIVE,
        filters=["merge,- .gitignore", ":- .gitignore"],
        exclude=["__pycache__/"],
    )
    assert files == ["src/keep.py"]
    stat = (workspace / "src" / "keep.py").stat()
    same = {"src/keep.py": (stat.st_size, int(stat.st_mtime))}
    assert tarball.pending(files, same) == []
    assert tarball.pending(files, {"src/keep.py": (stat.st_size, 0)}) == ["src/keep.py"]
    assert tarball.pending(files, {}) == ["src/keep.py"]
    assert tarball.pending([*files, "src/gone.py"], same) == []


def test_the_remote_listing_reads_every_top_directory_deep_and_the_generated_ones_flat() -> None:
    transport = RecordingTransport(
        rules=[
            (
                "Get-ChildItem",
                0,
                "src/a.py\t3\t1700000000\nnoise\n.mainboard/envs/default/pixi.lock\t9\t5\n",
            )
        ]
    )
    shell = WindowsShell(windows_plan(), _ROOT, ssh=transport)
    held = Tarball(Path("."), transport).held(
        shell, ["src/a.py", "packages/tool/x.py", ".mainboard/envs/default/pixi.lock"]
    )
    script = transport.scripts[-1]
    assert "foreach ($top in @('packages', 'src'))" in script
    assert "foreach ($flat in @('.mainboard/envs/default'))" in script
    assert held == {"src/a.py": (3, 1700000000), ".mainboard/envs/default/pixi.lock": (9, 5)}
