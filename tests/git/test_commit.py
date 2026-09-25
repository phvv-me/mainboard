from mainboard.git import Outcome, Step

from .conftest import Workspace

# The fixture ceiling is 0.001 MB, 1048 bytes, so a file one byte past it is oversized.
_HEAVY = b"x" * 1049


def _outcomes(steps: list[Step]) -> dict[str, Outcome]:
    return {step.repo: step.outcome for step in steps}


def test_commit_goes_bottom_up_and_leaves_withheld_content_in_the_working_tree(
    workspace: Workspace,
) -> None:
    """The library commits on its attached trunk, then the root records the new pointer.

    The evidence artifact never enters, the oversized file stays unstaged and is named, the
    oversized LFS file goes in because what history takes for it is a pointer, and a file
    staged by hand under `never-commit` is taken back out of the index.
    """
    lib = workspace.lib
    (lib / "lib.txt").write_text("edited\n", encoding="utf-8")
    (lib / ".gitattributes").write_text("*.lfs filter=lfs\n", encoding="utf-8")
    (lib / "model.lfs").write_bytes(_HEAVY)
    (lib / "big.dat").write_bytes(_HEAVY)
    artifact = lib / "run" / "evidence" / "artifacts" / "out.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"artifact")
    forced = lib / "evidence" / "artifacts" / "forced.bin"
    forced.parent.mkdir(parents=True)
    forced.write_bytes(b"forced")
    workspace.git(lib, "add", "--", "evidence/artifacts/forced.bin")

    steps = workspace.tree(ceiling_mb=0.001).commit("Record the work")

    assert [step.repo for step in steps] == ["packages/lib", "."]
    assert _outcomes(steps) == {"packages/lib": Outcome.DONE, ".": Outcome.DONE}
    assert "withheld big.dat, evidence/artifacts/forced.bin" in steps[0].detail
    assert workspace.git(lib, "symbolic-ref", "--short", "HEAD") == "main"
    assert workspace.git(lib, "rev-parse", "--abbrev-ref", "@{u}") == "origin/main"
    committed = workspace.git(lib, "show", "--name-only", "--format=", "HEAD").split()
    assert sorted(committed) == [".gitattributes", "lib.txt", "model.lfs"]
    untracked = workspace.git(lib, "status", "--porcelain", "-uall").splitlines()
    assert sorted(untracked) == [
        "?? big.dat",
        "?? evidence/artifacts/forced.bin",
        "?? run/evidence/artifacts/out.bin",
    ]
    recorded = workspace.git(workspace.path, "ls-tree", "HEAD", "packages/lib").split()[2]
    assert recorded == workspace.head(lib)
    assert workspace.git(workspace.path, "log", "-1", "--format=%s") == "Record the work"

    again = workspace.tree(ceiling_mb=0.001).commit("Nothing new")
    assert _outcomes(again) == {"packages/lib": Outcome.CURRENT, ".": Outcome.CURRENT}
    assert again[0].detail == "withheld big.dat"
    assert again[1].detail == "clean"


def test_an_empty_never_commit_list_withholds_only_by_size(workspace: Workspace) -> None:
    artifact = workspace.lib / "evidence" / "artifacts" / "out.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"artifact")
    steps = workspace.tree(never_commit=[]).commit("Keep the artifact")
    assert steps[0].outcome is Outcome.DONE
    assert "evidence/artifacts/out.bin" in workspace.git(workspace.lib, "ls-files")


def test_a_repository_behind_its_upstream_holds_itself_and_every_parent(
    workspace: Workspace,
) -> None:
    colleague = workspace.colleague()
    colleague.git(colleague.lib, "switch", "-q", "main")
    colleague.forge.commit(colleague.lib, "theirs", {"theirs.txt": "theirs\n"})
    colleague.git(colleague.lib, "push", "-q", "origin", "main")
    workspace.git(workspace.lib, "fetch", "-q", "origin")
    (workspace.lib / "mine.txt").write_text("mine\n", encoding="utf-8")
    (workspace.path / "root.txt").write_text("root\n", encoding="utf-8")

    steps = workspace.tree().commit("Too early")

    assert steps == [
        Step(repo="packages/lib", outcome=Outcome.HELD, detail="1 behind origin/main; pull first"),
        Step(repo=".", outcome=Outcome.HELD, detail="packages/lib did not commit"),
    ]


def test_a_detached_head_off_its_trunk_is_held_rather_than_committed_nowhere(
    workspace: Workspace,
) -> None:
    lib = workspace.lib
    workspace.git(lib, "switch", "-q", "-c", "side")
    workspace.forge.commit(lib, "side work", {"side.txt": "side\n"})
    workspace.git(lib, "switch", "-q", "main")
    workspace.forge.commit(lib, "main work", {"main.txt": "main\n"})
    workspace.git(lib, "switch", "-q", "--detach", "side")
    (lib / "more.txt").write_text("more\n", encoding="utf-8")

    step = workspace.tree().commit("Nowhere")[0]

    assert step.outcome is Outcome.HELD
    assert step.detail.endswith("off the line of main")


def test_a_detached_head_diverged_from_the_remote_trunk_is_held(workspace: Workspace) -> None:
    lib = workspace.lib
    colleague = workspace.colleague()
    colleague.git(colleague.lib, "switch", "-q", "main")
    colleague.forge.commit(colleague.lib, "theirs", {"theirs.txt": "theirs\n"})
    colleague.git(colleague.lib, "push", "-q", "origin", "main")
    workspace.git(lib, "branch", "-q", "-D", "main")
    workspace.forge.commit(lib, "mine", {"mine.txt": "mine\n"})
    workspace.git(lib, "fetch", "-q", "origin")
    (lib / "more.txt").write_text("more\n", encoding="utf-8")

    assert workspace.tree().commit("Diverged")[0].outcome is Outcome.HELD


def test_a_detached_head_with_no_trunk_anywhere_gets_one_made_for_it(
    workspace: Workspace,
) -> None:
    lib = workspace.lib
    workspace.git(lib, "branch", "-q", "-D", "main")
    workspace.git(lib, "update-ref", "-d", "refs/remotes/origin/main")
    (lib / "more.txt").write_text("more\n", encoding="utf-8")

    step = workspace.tree().commit("Fresh trunk")[0]

    assert step.outcome is Outcome.DONE
    assert step.detail.endswith("on main")


def test_a_hook_that_refuses_the_commit_fails_the_repository(workspace: Workspace) -> None:
    hooks = workspace.path / ".git" / "hooks"
    hook = hooks / "pre-commit"
    hook.write_text("#!/bin/sh\necho 'lint says no' >&2\nexit 1\n", encoding="utf-8", newline="\n")
    hook.chmod(0o755)
    (workspace.path / "root.txt").write_text("root\n", encoding="utf-8")

    steps = workspace.tree().commit("Refused")

    assert steps[-1] == Step(repo=".", outcome=Outcome.FAILED, detail="lint says no")
