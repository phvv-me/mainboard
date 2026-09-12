from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.dispatch import rentals as rentals_module
from mainboard.dispatch.rentals import (
    LANDING_SECONDS,
    LAUNCH,
    handoff,
    identity,
    reachable,
    seeded,
    waiting,
)
from mainboard.dispatch.transport import Endpoint, HostUnreachable

from .support import Naps, keypair


def test_the_key_a_rental_is_opened_with_is_one_this_machine_holds_both_halves_of(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider hands out a machine, not an account, so the public half goes with the create.

    The declared key wins over the standard pair, since a workspace that names one is saying
    which account the rental should trust.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    standard = keypair(tmp_path)
    declared = keypair(tmp_path, "work_key")
    assert identity().private == str(standard)
    assert identity().public == "ssh-ed25519 AAAA me@here"
    assert identity(str(declared)).private == str(declared)


def test_a_machine_with_no_key_pair_refuses_naming_the_command_that_makes_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal has to arrive before the rental, since a box nobody can log into still bills."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with pytest.raises(MissionError, match="ssh-keygen -t ed25519"):
        identity()
    with pytest.raises(MissionError, match="nowhere/key"):
        identity("nowhere/key")


def test_the_rented_entrypoint_waits_for_the_landing_and_gives_up_on_the_meter() -> None:
    """A container comes up minutes before the workspace reaches it, so it holds still first.

    The wait is bounded because it is billed: a workstation that dies mid-landing leaves a
    machine waiting for a script nobody will write, and the deadline is what turns that into a
    status the ordinary settle-and-cancel path can act on.
    """
    script = waiting()
    assert f"while [ ! -f {LAUNCH} ]" in script
    assert f"bash {LAUNCH}" in script
    assert "status=$?" in script
    assert f'"$mb_waited" -ge {LANDING_SECONDS}' in script
    assert "status=75" in script and "no dispatch landed" in script


def test_the_launch_script_is_moved_into_place_rather_than_written_where_it_is_watched() -> None:
    """The entrypoint is polling for that exact name, so a half-written file is half a job."""
    handed = handoff()
    assert handed.startswith("cat > ") and "mv " in handed
    assert handed.endswith(LAUNCH)
    assert LAUNCH not in handed.split("&&")[0]


def test_a_rental_is_only_reachable_once_ssh_actually_answers_on_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Knocking is the only honest test, since a published address answers nothing for a while."""
    endpoint = Endpoint(address="ssh5.vast.ai", port=41022, user="root")
    knocks: list[str] = []

    def warm(self: object, host: str) -> None:
        knocks.append(host)
        if len(knocks) < 3:
            raise HostUnreachable("connection refused")

    monkeypatch.setattr(rentals_module.SshTransport, "warm", warm)
    naps = Naps()
    assert reachable(endpoint, sleeper=naps) is endpoint
    assert knocks == ["root@ssh5.vast.ai"] * 3
    assert naps.waited == [5.0, 5.0]


def test_a_machine_that_never_answers_ssh_is_refused_rather_than_landed_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal here still holds the handle, so the rental can be ended before it is wasted."""
    monkeypatch.setattr(
        rentals_module.SshTransport,
        "warm",
        lambda self, host: (_ for _ in ()).throw(HostUnreachable("connection refused")),
    )
    naps = Naps()
    with pytest.raises(MissionError, match="ssh never answered at root@box"):
        reachable(Endpoint(address="box", user="root"), sleeper=naps, attempts=3)
    assert naps.waited == [5.0, 5.0, 5.0]


def test_seeding_writes_one_key_once_with_the_modes_sshd_accepts() -> None:
    lines = seeded("ssh-ed25519 AAAA me@here\n").splitlines()
    assert lines[0] == "mkdir -p /root/.ssh && chmod 700 /root/.ssh"
    assert lines[1].startswith("grep -qxF 'ssh-ed25519 AAAA me@here' /root/.ssh/authorized_keys")
    assert lines[1].endswith("|| echo 'ssh-ed25519 AAAA me@here' >> /root/.ssh/authorized_keys")
    assert lines[2].startswith("chmod 600 /root/.ssh/authorized_keys && chown root:root")
