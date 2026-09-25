from collections.abc import Callable

import pytest

from mainboard.git import Outcome, Step

from .conftest import FOREIGN, OWNED, Workspace


def _advance(workspace: Workspace) -> Workspace:
    """Another machine moves the library and the reference forward and records both pointers."""
    colleague = workspace.colleague()
    colleague.forge.publish(colleague.lib, {"lib.txt": "theirs\n"})
    upstream = workspace.forge.publish(workspace.forge.seed(FOREIGN, "ref"), {"ref.txt": "up\n"})
    colleague.git(colleague.ref, "fetch", "-q", "origin")
    colleague.git(colleague.ref, "checkout", "-q", "--detach", upstream)
    colleague.forge.publish(colleague.path, {})
    return colleague


def _advance_on_main(workspace: Workspace) -> Workspace:
    """Another machine moves the library forward while this one sits on its `main`."""
    colleague = workspace.colleague()
    colleague.forge.publish(colleague.lib, {"lib.txt": "theirs\n"})
    workspace.git(workspace.lib, "switch", "-q", "main")
    return colleague


def test_pull_moves_the_root_then_carries_every_checkout_to_its_new_pointer(
    workspace: Workspace,
) -> None:
    """The library follows and lands on its trunk; the foreign reference is moved, not pulled."""
    colleague = _advance(workspace)

    steps = workspace.tree().pull()

    assert [(step.repo, step.outcome) for step in steps] == [
        (".", Outcome.DONE),
        ("packages/lib", Outcome.DONE),
        ("references/ref", Outcome.DONE),
    ]
    assert steps[0].detail == "fast-forwarded 1 from origin/main"
    assert "attached to main" in steps[1].detail
    for mine, theirs in [
        (workspace.path, colleague.path),
        (workspace.lib, colleague.lib),
        (workspace.ref, colleague.ref),
    ]:
        assert workspace.head(mine) == colleague.head(theirs)
    assert workspace.git(workspace.lib, "symbolic-ref", "--short", "HEAD") == "main"
    again = workspace.tree().pull()
    assert [step.outcome for step in again] == [Outcome.CURRENT, Outcome.CURRENT]
    assert again[1].detail == "main level with origin/main"


def test_a_submodule_never_checked_out_is_cloned_at_its_pointer(workspace: Workspace) -> None:
    workspace.git(workspace.path, "submodule", "deinit", "-q", "-f", "--all")

    steps = workspace.tree().pull()

    assert [(step.repo, step.outcome) for step in steps] == [
        (".", Outcome.CURRENT),
        ("packages/lib", Outcome.DONE),
        ("packages/lib", Outcome.DONE),
        ("packages/lib/vendor/dep", Outcome.DONE),
        ("references/ref", Outcome.DONE),
    ]
    assert steps[1].detail.startswith("cloned at")
    assert (workspace.dep / "dep.txt").is_file()


def test_a_foreign_checkout_moved_by_hand_stays_where_it_was_put(workspace: Workspace) -> None:
    _advance(workspace)
    mine = workspace.forge.commit(workspace.ref, "my own look", {"mine.txt": "mine\n"})

    step = workspace.tree().pull()[-1]

    assert step.outcome is Outcome.HELD
    assert step.detail.startswith(f"left at {mine[:7]}")
    assert workspace.head(workspace.ref) == mine


def test_a_diverged_branch_is_held_and_the_rest_still_pulled(workspace: Workspace) -> None:
    _advance(workspace)
    workspace.forge.commit(workspace.path, "mine", {"mine.txt": "mine\n"})

    steps = workspace.tree().pull()

    assert steps[0] == Step(
        repo=".", outcome=Outcome.HELD, detail="diverged from origin/main: 1 ahead, 1 behind"
    )
    assert steps[1].repo == "packages/lib"
    assert steps[1].detail == "attached to main; fast-forwarded 1 from origin/main"


def test_a_remote_that_does_not_answer_fails_its_repository(workspace: Workspace) -> None:
    gone = workspace.forge.bare(OWNED, "gone")
    workspace.git(workspace.lib, "remote", "set-url", "origin", str(gone))

    lib = workspace.tree().pull()[1]

    assert lib.outcome is Outcome.FAILED


@pytest.mark.parametrize(
    "advance",
    [
        pytest.param(_advance, id="following the parent's new pointer"),
        pytest.param(_advance_on_main, id="fast-forwarding the branch"),
    ],
)
def test_a_move_that_would_overwrite_local_changes_is_held(
    workspace: Workspace, advance: Callable[[Workspace], Workspace]
) -> None:
    advance(workspace)
    (workspace.lib / "lib.txt").write_text("my uncommitted edit\n", encoding="utf-8")

    lib = workspace.tree().pull()[1]

    assert lib.outcome is Outcome.HELD
    assert (workspace.lib / "lib.txt").read_text(encoding="utf-8") == "my uncommitted edit\n"


def test_a_detached_head_behind_its_local_trunk_moves_forward_onto_it(
    workspace: Workspace,
) -> None:
    workspace.git(workspace.lib, "switch", "-q", "main")
    workspace.forge.commit(workspace.lib, "local only", {"local.txt": "local\n"})
    workspace.git(workspace.lib, "switch", "-q", "--detach", "HEAD~1")

    lib = workspace.tree().pull()[1]

    assert lib == Step(repo="packages/lib", outcome=Outcome.DONE, detail="attached to main")


def test_a_detached_head_off_its_trunk_is_held(workspace: Workspace) -> None:
    lib = workspace.lib
    workspace.git(lib, "switch", "-q", "-c", "side")
    workspace.forge.commit(lib, "side", {"side.txt": "side\n"})
    workspace.git(lib, "switch", "-q", "main")
    workspace.forge.commit(lib, "main", {"main.txt": "main\n"})
    workspace.git(lib, "switch", "-q", "--detach", "side")

    step = workspace.tree().pull()[1]

    assert step.outcome is Outcome.HELD
    assert step.detail.endswith("off the line of main")


def test_a_branch_tracking_nothing_is_left_alone(workspace: Workspace) -> None:
    workspace.git(workspace.lib, "switch", "-q", "-c", "feature")

    lib = workspace.tree().pull()[1]

    assert lib == Step(
        repo="packages/lib", outcome=Outcome.CURRENT, detail="feature tracking no upstream"
    )


def test_a_branch_sitting_on_the_old_pointer_fast_forwards_to_the_new_one(
    workspace: Workspace,
) -> None:
    colleague = _advance(workspace)
    workspace.git(workspace.lib, "switch", "-q", "main")

    lib = workspace.tree().pull()[1]

    assert lib.outcome is Outcome.DONE
    assert lib.detail.startswith("followed the parent to")
    assert workspace.head(workspace.lib) == colleague.head(colleague.lib)


def test_a_pointer_its_remote_cannot_serve_fails_the_checkout(workspace: Workspace) -> None:
    colleague = workspace.colleague()
    colleague.forge.commit(colleague.ref, "never pushed", {"x.txt": "x\n"})
    colleague.forge.publish(colleague.path, {})

    ref = workspace.tree().pull()[-1]

    assert ref.repo == "references/ref"
    assert ref.outcome is Outcome.FAILED
