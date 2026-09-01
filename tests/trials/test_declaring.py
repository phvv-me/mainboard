import subprocess
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

import mainboard.trials.provenance as provenance
from mainboard.trials import (
    SOURCE_VAR,
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
    """One writable flag over a dictionary, plus the dictionary it moves."""
    state = {name: held_value}
    return (
        Flag(name=name, read=lambda: state[name], write=lambda value: state.update({name: value})),
        state,
    )


def test_a_vocabulary_answers_only_for_the_words_its_consumer_declared() -> None:
    """A word prints its declared letter or its own initial, and an undeclared one refuses.

    The letter fallback matters because a consumer declaring five words should not have to type
    five letters to get a readable progress line, and the refusal matters because settling a word
    nobody declared writes a receipt no report can group.
    """
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
    """`held` restores what it recorded even when the block dies, and skips what has no write.

    The asserted half is the whole reason `write` is optional: a knob the library only reads at
    process start cannot honestly be moved back, so pretending to would put a value in a column
    that the machine never had.
    """
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
    """An empty coordinate is four facts, and the outcome column is what tells them apart."""
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
    """The three states, and a line that names the run when there is nothing left to take."""
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
    """Three names and an ellipsis, since a reader fixing a lane wants the count not the list."""
    status = LaneStatus(lane="l", want=5, have=0, missing=("a", "b", "c", "d"))
    assert "missing 4: a, b, c..." in status.line()


def reads(head: str = "abc1234", status: str = "", listed: str = "") -> Callable[..., str]:
    """A git stand-in answering the three questions a preflight asks, and nothing else."""

    def read(*args: str) -> str:
        if "rev-parse" in args:
            return f"{head}-tree" if "HEAD^{tree}" in args else head
        return listed if "ls-files" in args else status

    return read


def test_a_source_is_probed_where_there_is_a_repository_and_declared_where_there_is_not(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A mirror has the source and not the history, so a declared commit says it is declared.

    A dispatched job gets the working tree rsynced onto a host and not the repository, so refusing
    a receipt there would say no remote card may ever take a reading. The commit is the FULL one
    and the committed tree rides beside it, because a program whose every claimed row named the
    same seven characters and `dirty` could reconstruct none of them.
    """
    monkeypatch.delenv(SOURCE_VAR, raising=False)
    clean = source(tmp_path, read=reads())
    assert clean == source(tmp_path, read=reads())
    assert clean.commit == "abc1234" and clean.tree == "abc1234-tree"
    assert not clean.dirty and not clean.mirrored
    assert clean.admissibility is Admissibility.ADMISSIBLE

    dirty = source(tmp_path, read=reads(status=" M x"))
    assert dirty.dirty and not dirty.mirrored
    assert dirty.admissibility is Admissibility.DIRTY

    with pytest.raises(RuntimeError, match=SOURCE_VAR):
        source(tmp_path, read=lambda *args: "")
    monkeypatch.setenv(SOURCE_VAR, "deadbee-dirty")
    mirrored = source(tmp_path, read=lambda *args: "")
    assert mirrored.commit == "deadbee" and mirrored.dirty and mirrored.mirrored
    assert mirrored.admissibility is Admissibility.DIRTY


def test_the_real_git_reader_answers_about_a_directory_that_is_not_a_repository(
    tmp_path: Path,
) -> None:
    """The default reader is the local `git`, and a non-repository is an empty answer not a raise.

    Asserted against the tool itself rather than a stand-in, because the whole point of the
    fallback is that a real `git` outside a work tree returns nothing on stdout.
    """
    with pytest.raises(RuntimeError, match="not a git working tree"):
        source(tmp_path / "nowhere")


def test_a_probe_that_broke_is_never_mistaken_for_a_host_with_no_device() -> None:
    """Found, absent and failed are three answers, and only the middle one is a clean machine."""
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
    """One probe, one shape, and an installed distribution beside one this environment lacks."""
    monkeypatch.delenv(SOURCE_VAR, raising=False)
    (tmp_path / "test_law.py").write_text("def test_holds(trial): ...\n")
    taken = Preflight(
        tmp_path,
        tmp_path,
        probed=("polars", "no-such-distribution"),
        machine=Machine((Card(),)),
        read=reads(listed=f"{(tmp_path / 'test_law.py').name}\0"),
    )
    stamped = taken.stamp
    assert stamped["card"] == "GPU-1111" and stamped["card_name"] == "Test Card"
    assert stamped["card_probed"] == "found" and stamped["capability"] == "sm_89"
    assert stamped["commit"] == "abc1234" and stamped["tree"] == "abc1234-tree"
    assert stamped["source_digest"] == digest_of(tmp_path, "*.py")
    assert stamped["mirrored"] is False and stamped["worktree_dirty"] is False
    assert stamped["versions"] == {
        "polars": installed("polars"),
        "no-such-distribution": "absent",
    }
    assert installed("polars") != "absent"
    assert taken.admits(tmp_path / "test_law.py") is Admissibility.ADMISSIBLE
    assert taken.admits(tmp_path / "test_scratch.py") is Admissibility.UNTRACKED

    registered = tmp_path / "alpha" / "baselines"
    registered.mkdir(parents=True)
    (registered / "cells.json").write_text('{"law_low": 0.9}')
    assert taken.baselines("alpha") == digest_of(registered)
    assert taken.baselines("beta") == ""


def test_an_import_name_finds_a_platform_specific_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A logical receipt key survives a differently named platform distribution."""

    def provider_version(name: str) -> str:
        if name == "triton-windows":
            return "3.7.1"
        raise PackageNotFoundError(name)

    monkeypatch.setattr(provenance, "version", provider_version)
    monkeypatch.setattr(
        provenance, "packages_distributions", lambda: {"triton": ["triton-windows"]}
    )

    assert provenance.installed("triton") == "3.7.1"


def test_a_dirty_tree_is_inadmissible_rather_than_refused_and_an_untracked_lane_with_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Scratch work still runs, it just stops counting, which is the whole of the rule.

    A run on a moving tree measures, prints and writes; what it may not do is satisfy a claim,
    because two dirty trees at one commit are one string and a reader cannot tell them apart.
    A mirror is not asked about trackedness at all, since the repository it was copied from is
    not here and an unanswerable question must not read as the answer `nothing is tracked`.
    """
    monkeypatch.delenv(SOURCE_VAR, raising=False)
    moving = Preflight(tmp_path, tmp_path, machine=Machine(), read=reads(status=" M src/x.py"))
    assert moving.admissibility is Admissibility.DIRTY
    assert moving.admits(tmp_path / "anything.py") is Admissibility.DIRTY

    monkeypatch.setenv(SOURCE_VAR, "deadbee")
    mirrored = Preflight(tmp_path, tmp_path, machine=Machine(), read=lambda *args: "")
    assert mirrored.tracked is None
    assert mirrored.admits(tmp_path / "anything.py") is Admissibility.ADMISSIBLE
    assert mirrored.stamp["source_digest"] == digest_of(tmp_path, "*.py")


def test_a_digest_pins_the_bytes_on_disk_and_a_registration_row_pins_its_own_values(
    tmp_path: Path,
) -> None:
    """Two dirty trees at one commit differ in files git is not carrying, so untracked ones count.

    The relative path is folded in beside the bytes, so a file that MOVED changes the digest as
    surely as a file that changed, and build output is skipped so one tree does not digest two
    ways on two machines.
    """
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
    """Made once, dropped on leaving, and checked against the floor when a probe was declared.

    The refusal is the answer to a run that lost 68 trials to a card eleven claims had filled,
    where every claim was correct on its own and the session was what was wrong.
    """
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


def test_a_flat_universe_names_its_stage_by_the_root_it_leaked_from() -> None:
    """The refusal has to name something, and a flat universe's claim is the root itself."""
    resident = iter([0, 10])
    stage = Stage("", resident=lambda: next(resident))
    with pytest.raises(RuntimeError, match="the universe root did not release"):
        stage.drop()


def test_a_real_repository_answers_with_a_full_commit_a_tree_and_its_own_tracked_lanes(
    tmp_path: Path,
) -> None:
    """The whole preflight against a repository it builds, rather than against a stand-in.

    Everything the receipt now stands on is git answering for itself: the full forty-character
    commit, the tree it resolves to, which lanes are in that tree, and the moment the tree stops
    being the one the commit names.
    """
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "test_law.py").write_text("def test_holds(trial): ...\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.email=a@b",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            "x",
        ],
        check=True,
    )
    taken = source(tmp_path)
    assert len(taken.commit) == 40 and len(taken.tree) == 40 and not taken.mirrored
    assert not taken.dirty

    clean = Preflight(tmp_path, tmp_path, machine=Machine())
    assert clean.admissibility is Admissibility.ADMISSIBLE
    assert clean.admits(tmp_path / "test_law.py") is Admissibility.ADMISSIBLE
    assert clean.admits(tmp_path / "test_scratch.py") is Admissibility.UNTRACKED

    (tmp_path / "test_scratch.py").write_text("def test_tries(trial): ...\n")
    assert source(tmp_path).dirty
    assert Preflight(tmp_path, tmp_path, machine=Machine()).digest != clean.digest
