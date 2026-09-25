from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

import mainboard.trials.provenance as provenance
from mainboard.dispatch.shared import CLOSURE_VAR, DIGEST_VAR
from mainboard.trials import (
    Admissibility,
    Cell,
    Flag,
    LaneStatus,
    Preflight,
    Probed,
    Stage,
    Stance,
    Vocabulary,
    Word,
    card_of,
    digest_of,
    digested,
    held,
    installed,
    moved,
    reading,
    source,
)

from .support import Card, Machine


def knob(name: str, held_value: str) -> tuple[Flag, dict[str, str]]:
    state = {name: held_value}
    return (
        Flag(name=name, read=lambda: state[name], write=lambda value: state.update({name: value})),
        state,
    )


def dispatched(monkeypatch: pytest.MonkeyPatch, root: Path, closure: str, digest: str) -> None:
    """Pose as a dispatched job whose captured closure listing is `closure` under `digest`."""
    copied = root / ".mainboard-closure"
    copied.write_bytes(Path(closure).read_bytes())
    monkeypatch.setenv(CLOSURE_VAR, str(copied))
    monkeypatch.setenv(DIGEST_VAR, digest)


def test_a_vocabulary_answers_only_for_the_words_its_consumer_declared() -> None:
    """A word prints its declared letter or its initial; an undeclared word would write a
    receipt no report can group, so it refuses."""
    words = Vocabulary(
        words=(
            Word(name="validated", letter="V", stance=Stance.CONFIRMS),
            Word(name="refuted", stance=Stance.REFUTES),
            Word(name="undecided"),
        )
    )
    assert words.names == ("validated", "refuted", "undecided")
    assert [words[name].mark for name in words.names] == ["V", "R", "U"]
    assert "known" not in words and "undecided" in words
    assert words.stanced(Stance.NEITHER) == ("undecided",)
    assert words.stanced(Stance.CONFIRMS) == ("validated",)
    assert Vocabulary.of("held", "broke").stanced(Stance.NEITHER) == ("held", "broke")
    with pytest.raises(KeyError, match="not a declared settle word"):
        words["ranked"]


def test_a_held_flag_comes_back_and_an_asserted_one_is_never_written() -> None:
    """A knob read only at process start cannot honestly be moved back, so it has no `write`."""
    writable, state = knob("policy", "pinned")
    watched = {"env": "unset"}
    asserted = Flag(name="env", read=lambda: watched["env"])
    assert reading((writable, asserted)) == {"policy": "pinned", "env": "unset"}

    with held(writable, asserted) as baseline:
        state["policy"] = "default"
        watched["env"] = "moved"
        assert baseline == {"policy": "pinned", "env": "unset"}
    assert state["policy"] == "pinned"
    assert watched["env"] == "moved"

    with pytest.raises(ZeroDivisionError), held(writable):
        state["policy"] = "default"
        raise ZeroDivisionError
    assert state["policy"] == "pinned"

    assert moved((writable, asserted), {"policy": "pinned", "env": "unset"}) == {"env": "moved"}
    assert not moved((writable,), {"policy": "pinned"})


@pytest.mark.parametrize(
    ("values", "probing", "named"),
    [
        ({"card": "GPU-1", "model": "qwen"}, {"card": "found", "model": "unasked"}, "GPU-1, qwen"),
        ({"card": "", "model": ""}, {"card": "absent", "model": "unasked"}, "card absent"),
        ({"card": "", "model": ""}, {"card": "failed", "model": "unasked"}, "card failed"),
        ({"card": "", "model": ""}, {"card": "unasked", "model": "unasked"}, ""),
    ],
    ids=[
        "a found axis names its value",
        "a host with no device says so",
        "a broken probe says something else",
        "an axis nobody asked about stays quiet",
    ],
)
def test_a_cell_never_lets_four_different_empties_read_as_one(
    values: Mapping[str, str], probing: Mapping[str, str], named: str
) -> None:
    found = Cell(values=dict(values), probing={axis: Probed(why) for axis, why in probing.items()})
    assert found.named == named
    assert found.filters == {**values, "card_probed": probing["card"], "model_probed": "unasked"}
    assert found.key == tuple(sorted(found.filters.items()))


@pytest.mark.parametrize(
    ("want", "have", "missing", "state"),
    [
        (2, 2, (), "complete"),
        (2, 1, ("b",), "partial"),
        (2, 0, ("a", "b"), "missing"),
        (0, 0, (), "missing"),
    ],
    ids=[
        "every sample is on file",
        "some are",
        "none are",
        "a lane with no grid is not complete",
    ],
)
def test_a_lane_status_states_what_it_still_owes(
    want: int, have: int, missing: tuple[str, ...], state: str
) -> None:
    status = LaneStatus(
        lane="alpha/test_law.py::test_holds",
        want=want,
        have=have,
        missing=missing,
        run="run-1" if have else "",
        cell=Cell(values={"card": "GPU-1"}, probing={"card": Probed.FOUND}),
    )
    assert status.state == state
    line = status.line()
    assert state in line and "on GPU-1" in line
    assert ("from run-1" in line) is (state == "complete")


def test_a_status_line_truncates_a_long_missing_list() -> None:
    status = LaneStatus(lane="l", want=5, have=0, missing=("a", "b", "c", "d"))
    assert "missing 4: a, b, c..." in status.line()


def test_a_local_source_is_captured_and_a_dispatched_source_is_verified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(CLOSURE_VAR, raising=False)
    monkeypatch.setenv("PATH", "")
    lane = tmp_path / "test_law.py"
    lane.write_text("pass")
    captured = source(tmp_path)
    assert captured == source(tmp_path)
    assert len(captured.digest) == 64 and not captured.mirrored
    dispatched(monkeypatch, tmp_path, captured.closure, captured.digest)
    mirrored = source(tmp_path)
    assert mirrored.mirrored and mirrored.digest == captured.digest
    assert mirrored.admissibility is Admissibility.ADMISSIBLE
    lane.write_text("changed")
    with pytest.raises(RuntimeError, match="bytes differ"):
        source(tmp_path)


def test_a_declared_listing_requires_an_authentic_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dispatched(monkeypatch, tmp_path, source(tmp_path).closure, "wrong")
    with pytest.raises(RuntimeError, match="content digest"):
        source(tmp_path)


def test_nested_trial_projects_verify_the_dispatch_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "research" / "example"
    experiments = project / "experiments"
    experiments.mkdir(parents=True)
    lane = experiments / "test_law.py"
    lane.write_text("pass")
    (tmp_path / "conftest.py").write_text("pass")
    captured = source(tmp_path)
    dispatched(monkeypatch, tmp_path, captured.closure, captured.digest)
    taken = Preflight(experiments, project, machine=Machine())
    assert taken.source.root == tmp_path.resolve()
    assert taken.digest == captured.digest
    assert taken.admissibility is Admissibility.ADMISSIBLE
    assert taken.admits(lane) is Admissibility.ADMISSIBLE
    with pytest.raises(RuntimeError, match="outside the captured"):
        source(tmp_path.parent)
    (tmp_path / "conftest.py").write_text("changed outside the nested project")
    with pytest.raises(RuntimeError, match="bytes differ"):
        source(project)


def test_a_probe_that_broke_is_never_mistaken_for_a_host_with_no_device() -> None:
    found = card_of(Machine((Card(),)))
    assert found.id == "GPU-1111" and found.name == "Test Card"
    # The driver is the HOST driver and the runtime version rides beside it, never inside it.
    assert (found.driver, found.runtime) == ("580.65.06", "13.1")
    assert found.probed is Probed.FOUND

    unnamed = card_of(Machine((Card(uuid="", driver="", runtime=None),)))
    assert unnamed.id == "Test Card" and not unnamed.driver and not unnamed.runtime

    assert card_of(Machine()).probed is Probed.ABSENT
    broken = card_of(Machine(breaks="nvml is not loaded"))
    assert broken.probed is Probed.FAILED and broken.detail == "nvml is not loaded"
    assert not broken.id


def test_a_preflight_derives_every_field_a_receipt_would_otherwise_retype(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(CLOSURE_VAR, raising=False)
    (tmp_path / "test_law.py").write_text("def test_holds(trial): ...\n")
    taken = Preflight(
        tmp_path,
        tmp_path,
        probed=("polars", "no-such-distribution"),
        machine=Machine((Card(),)),
    )
    stamped = taken.stamp
    assert stamped["card"] == "GPU-1111" and stamped["card_name"] == "Test Card"
    assert stamped["card_probed"] == "found" and stamped["capability"] == "sm_89"
    assert "commit" not in stamped and "worktree_dirty" not in stamped
    assert stamped["source_digest"] == taken.source.digest
    assert stamped["mirrored"] is False
    assert stamped["versions"] == {
        "polars": installed("polars"),
        "no-such-distribution": "absent",
    }
    assert installed("polars") != "absent"
    assert taken.admits(tmp_path / "test_law.py") is Admissibility.ADMISSIBLE
    assert taken.admits(tmp_path / "test_scratch.py") is Admissibility.UNRECORDED

    registered = tmp_path / "alpha" / "baselines"
    registered.mkdir(parents=True)
    (registered / "cells.json").write_text('{"law_low": 0.9}')
    assert taken.baselines("alpha") == digest_of(registered)
    assert taken.baselines("beta") == ""


def test_an_import_name_finds_a_platform_specific_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def provider_version(name: str) -> str:
        if name == "triton-windows":
            return "3.7.1"
        raise PackageNotFoundError(name)

    monkeypatch.setattr(provenance, "version", provider_version)
    monkeypatch.setattr(
        provenance,
        "packages_distributions",
        lambda: {"triton": ["triton-ghost", "triton-windows"]},
    )

    assert provenance.installed("triton") == "3.7.1"


def test_preflight_checks_captured_membership_and_detects_later_changes(tmp_path: Path) -> None:
    lane = tmp_path / "test_law.py"
    lane.write_text("pass")
    taken = Preflight(tmp_path, tmp_path, machine=Machine())
    assert taken.admits(lane) is Admissibility.ADMISSIBLE
    lane.write_text("changed")
    assert taken.admits(lane) is Admissibility.UNRECORDED
    new = tmp_path / "new.py"
    new.write_text("pass")
    assert taken.admits(new) is Admissibility.UNRECORDED
    refreshed = Preflight(tmp_path, tmp_path, machine=Machine())
    assert refreshed.digest != taken.digest
    assert refreshed.admits(new) is Admissibility.ADMISSIBLE


def test_a_digest_pins_the_bytes_on_disk_and_a_registration_row_pins_its_own_values(
    tmp_path: Path,
) -> None:
    """Untracked files count, a moved file changes the digest like a changed one, and build
    output is skipped so one tree digests one way on every machine."""
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "one.py").write_text("x = 1\n")
    first = digest_of(tmp_path, "*.py")
    assert first and first == digest_of(tmp_path, "*.py")

    (tmp_path / "alpha" / "scratch.py").write_text("y = 2\n")
    moved_in = digest_of(tmp_path, "*.py")
    assert moved_in != first

    (tmp_path / "alpha" / "scratch.py").rename(tmp_path / "scratch.py")
    assert digest_of(tmp_path, "*.py") != moved_in

    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "one.py").write_text("compiled\n")
    assert digest_of(tmp_path, "*.py") != moved_in
    assert digest_of(tmp_path / "nowhere") == ""
    assert digest_of(tmp_path / "alpha", "*.json") == ""

    row = {"label": "qwen", "law_low": 0.9, "law_high": 1.1}
    assert digested(row) == digested(dict(reversed(list(row.items()))))
    assert digested(row) != digested({**row, "law_high": 1.2})


def test_a_stage_holds_one_claim_and_refuses_a_release_that_did_not_come_back() -> None:
    """The refusal answers a run that lost 68 trials to a card eleven claims had filled, each
    claim correct on its own; a flat universe's claim is named as the root."""
    made: list[str] = []
    stage = Stage("alpha")
    first = stage.kept("qwen", lambda: made.append("qwen") or "bundle")
    assert first == stage.kept("qwen", lambda: made.append("qwen") or "bundle")
    assert made == ["qwen"]
    stage.drop()
    assert not stage.held

    resident = {"bytes": 100}
    watched = Stage("beta", resident=lambda: resident["bytes"])
    watched.kept("weights", lambda: "loaded")
    watched.drop()

    resident["bytes"] = 400
    leaking = Stage("gamma", resident=lambda: resident["bytes"])
    resident["bytes"] = 900
    with pytest.raises(RuntimeError, match="gamma did not release"):
        leaking.drop()
    flat = iter([0, 10])
    with pytest.raises(RuntimeError, match="the universe root did not release"):
        Stage("", resident=lambda: next(flat)).drop()
