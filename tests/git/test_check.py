from pathlib import Path

from mainboard.doctor import Verdict
from mainboard.git import Finding

from .conftest import FOREIGN, OWNED, Workspace, posix_only


def _found(findings: list[Finding]) -> set[tuple[str, str, Verdict]]:
    return {(finding.repo, finding.check, finding.verdict) for finding in findings}


def test_a_fresh_clone_is_consistent_apart_from_its_detached_submodule(
    workspace: Workspace,
) -> None:
    findings = workspace.tree().check()
    assert _found(findings) == {("packages/lib", "branch", Verdict.WARN)}
    assert findings[0].detail.endswith("commit or pull puts it on main")


def test_work_in_progress_warns_and_names_what_the_next_push_carries(
    workspace: Workspace,
) -> None:
    """An unpushed library commit the unpushed root records is pending, not broken."""
    workspace.git(workspace.lib, "switch", "-q", "main")
    loose = workspace.forge.commit(workspace.lib, "lib work", {"lib.txt": "new\n"})
    workspace.forge.commit(workspace.path, "Record it", {})
    workspace.git(workspace.path, "submodule", "deinit", "-q", "-f", "references/ref")
    workspace.forge.submodule(workspace.path, workspace.forge.url(FOREIGN, "dep"), "staged")

    findings = workspace.tree().check()

    assert _found(findings) == {
        (".", "branch", Verdict.WARN),
        (".", "pointer", Verdict.WARN),
        ("packages/lib", "branch", Verdict.WARN),
    }
    details = {finding.detail for finding in findings}
    assert f"packages/lib@{loose[:7]} is not pushed yet; push carries it" in details
    assert "1 commits not on origin/main; push" in details


def test_a_published_parent_recording_an_unserved_commit_fails(workspace: Workspace) -> None:
    loose = workspace.forge.commit(workspace.ref, "local only", {"x.txt": "x\n"})
    workspace.forge.publish(workspace.path, {})

    findings = workspace.tree().check()

    assert (".", "pointer", Verdict.FAIL) in _found(findings)
    assert (
        f"references/ref@{loose[:7]}: no branch of its remote holds it, and the published parent "
        "already records it"
    ) in {finding.detail for finding in findings}


def test_a_remote_that_does_not_answer_is_named_on_every_finding_it_shapes(
    workspace: Workspace,
) -> None:
    workspace.forge.commit(workspace.ref, "local only", {"x.txt": "x\n"})
    workspace.forge.commit(workspace.path, "Record it", {})
    workspace.git(workspace.ref, "remote", "set-url", "origin", str(workspace.path / "gone"))
    workspace.git(
        workspace.lib, "remote", "set-url", "origin", str(workspace.forge.bare(OWNED, "gone"))
    )

    findings = workspace.tree().check()

    assert ("packages/lib", "fetch", Verdict.WARN) in _found(findings)
    pointer = next(finding for finding in findings if finding.check == "pointer")
    assert pointer.verdict is Verdict.FAIL
    assert "gone" in pointer.detail


def test_a_checkout_off_its_recorded_pointer_warns(workspace: Workspace) -> None:
    upstream = workspace.forge.publish(workspace.forge.seed(FOREIGN, "ref"), {"up.txt": "up\n"})
    workspace.git(workspace.ref, "fetch", "-q", "origin")
    workspace.git(workspace.ref, "checkout", "-q", "--detach", upstream)

    findings = workspace.tree().check()

    assert (".", "checkout", Verdict.WARN) in _found(findings)


def test_divergence_behind_and_untracked_branches_are_each_named(workspace: Workspace) -> None:
    colleague = workspace.colleague()
    colleague.forge.publish(colleague.path, {"theirs.txt": "theirs\n"})
    colleague.forge.publish(colleague.lib, {"theirs.txt": "theirs\n"})
    workspace.forge.commit(workspace.path, "mine", {"mine.txt": "mine\n"})
    workspace.git(workspace.lib, "switch", "-q", "-c", "feature")

    findings = workspace.tree().check()

    assert _found(findings) == {
        (".", "branch", Verdict.FAIL),
        ("packages/lib", "branch", Verdict.WARN),
    }
    assert {finding.detail for finding in findings} == {
        "diverged from origin/main: 1 ahead, 1 behind",
        "feature tracks no upstream; push sets one",
    }
    workspace.git(workspace.path, "reset", "-q", "--hard", "HEAD~1")
    workspace.git(workspace.lib, "switch", "-q", "--detach", "HEAD")
    behind = {finding.detail for finding in workspace.tree().check()}
    assert "1 commits behind origin/main; pull" in behind


def test_a_file_over_the_ceiling_in_head_fails(workspace: Workspace) -> None:
    workspace.forge.commit(workspace.path, "heavy", {"heavy.txt": "x" * 2048})

    findings = workspace.tree(ceiling_mb=0.001).check()

    assert (".", "size", Verdict.FAIL) in _found(findings)
    assert "heavy.txt is 0.00195 MB, over the 0.001 MB ceiling" in {f.detail for f in findings}


@posix_only
def test_lfs_content_with_no_git_lfs_here_fails(workspace: Workspace, fake_lfs: Path) -> None:
    (fake_lfs / "code").write_text("1", encoding="utf-8")
    workspace.forge.commit(workspace.path, "lfs", {".gitattributes": "*.bin filter=lfs\n"})

    findings = workspace.tree().check()

    assert (".", "lfs", Verdict.FAIL) in _found(findings)
    (fake_lfs / "code").write_text("0", encoding="utf-8")
    assert (".", "lfs", Verdict.FAIL) not in _found(workspace.tree().check())


def test_a_gitmodules_entry_with_no_gitlink_is_named_by_check_and_skipped_by_every_verb(
    workspace: Workspace,
) -> None:
    """Git refuses such an entry as a pathspec, which failed a whole pull row
    (`packages/sqlalchemy-cockroachdb/cockroach-proto`, 2026-09-26)."""
    for field, value in (("path", "gone"), ("url", workspace.forge.url(FOREIGN, "gone"))):
        workspace.git(
            workspace.path, "config", "-f", ".gitmodules", f"submodule.gone.{field}", value
        )
    workspace.forge.publish(workspace.path, {})
    stale = [finding for finding in workspace.tree().check() if finding.check == "gitmodules"]
    assert [(finding.repo, finding.verdict, finding.detail) for finding in stale] == [
        (".", Verdict.WARN, "gone: stale .gitmodules entry with no gitlink; remove it")
    ]
    steps = workspace.tree().pull()
    assert "gone" not in {step.repo for step in steps}
    assert all(step.outcome.settled for step in steps)
