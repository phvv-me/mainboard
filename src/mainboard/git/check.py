from functools import cached_property
from typing import TYPE_CHECKING

from ..core.section import Verdict
from .report import Finding

if TYPE_CHECKING:
    from .repo import Repo
    from .tree import Tree

_MEGABYTE = 1 << 20


class Check:
    """Every way the tree could fail somebody cloning it, or the next push, found at once.

    Each owned repository is fetched first, with the foreign submodules its pointers name, so
    the answer is about the remotes as they are now rather than as this machine last heard. A
    fetch writes only remote-tracking refs, so a foreign checkout is read and left as it was.

    A `fail` is something that is broken for somebody else already or cannot be repaired by the
    next push: a pointer whose commit the submodule's remote does not hold while the parent that
    records it is published (or the commit exists nowhere this machine can see), a branch that
    diverged from its upstream, a file in HEAD over the size ceiling, LFS content with no
    git-lfs to move it. A `warn` is the ordinary state of work in progress: detached, unpushed,
    behind, a checkout off its recorded pointer, a pointer the next push will carry.
    """

    def __init__(self, tree: Tree) -> None:
        self.tree = tree
        self.unfetched: dict[str, str] = {}

    def run(self) -> list[Finding]:
        """Fetch the tree, then every finding for every owned repository, parents first."""
        repos = self.tree.owned()
        foreign = [
            child
            for repo in repos
            for child in repo.children
            if child.initialized and not child.owned
        ]
        self.unfetched = self.tree.fetch([*repos, *foreign])
        return [finding for repo in repos for finding in self._findings(repo)]

    @cached_property
    def lfs_installed(self) -> bool:
        return self.tree.root.git.ok("lfs", "version")

    def _findings(self, repo: Repo) -> list[Finding]:
        return [
            *self._fetched(repo),
            *self._line(repo),
            *self._pointers(repo),
            *self._sizes(repo),
            *self._lfs(repo),
        ]

    def _fetched(self, repo: Repo) -> list[Finding]:
        """A remote that did not answer, so what follows reads the refs it last gave."""
        if complaint := self.unfetched.get(repo.name):
            return [_warn(repo, "fetch", f"{complaint}; reading the refs fetched last")]
        return []

    @staticmethod
    def _line(repo: Repo) -> list[Finding]:
        """Where HEAD stands against the branch it should be on and that branch's upstream."""
        findings: list[Finding] = []
        branch = repo.branch()
        upstream = repo.upstream()
        if not branch:
            detail = f"detached at {repo.short('HEAD')}; commit or pull puts it on {repo.trunk()}"
            findings.append(_warn(repo, "branch", detail))
        elif not upstream:
            findings.append(_warn(repo, "branch", f"{branch} tracks no upstream; push sets one"))
        ahead, behind = repo.counts(upstream)
        if ahead and behind:
            detail = f"diverged from {upstream}: {ahead} ahead, {behind} behind"
            findings.append(_fail(repo, "branch", detail))
        elif ahead:
            findings.append(_warn(repo, "branch", f"{ahead} commits not on {upstream}; push"))
        elif behind:
            findings.append(_warn(repo, "branch", f"{behind} commits behind {upstream}; pull"))
        return findings

    def _pointers(self, repo: Repo) -> list[Finding]:
        """Every submodule pointer HEAD records, against what that submodule's remote holds."""
        findings: list[Finding] = []
        recorded = repo.pointers()
        published = bool(repo.homes(repo.head()))
        for child in repo.children:
            if not (commit := recorded.get(repo.relative(child), "")):
                continue
            short = commit[:7]
            if not child.initialized:
                detail = f"{child.name} is not checked out, so {short} is unverified"
                findings.append(_warn(repo, "pointer", detail))
            elif not child.homes(commit):
                findings.append(self._unserved(repo, child, commit, published=published))
            elif child.head() != commit:
                detail = (
                    f"{child.name} is checked out at {child.short('HEAD')}, HEAD records {short}"
                )
                findings.append(_warn(repo, "checkout", detail))
        return findings

    def _unserved(self, repo: Repo, child: Repo, commit: str, *, published: bool) -> Finding:
        """A pointer no branch of the submodule's remote holds, pending only if a push fixes it."""
        where = f"{child.name}@{commit[:7]}"
        if child.owned and not published and child.exists(commit):
            return _warn(repo, "pointer", f"{where} is not pushed yet; push carries it")
        reason = self.unfetched.get(child.name, "no branch of its remote holds it")
        if published:
            reason += ", and the published parent already records it"
        return _fail(repo, "pointer", f"{where}: {reason}")

    def _sizes(self, repo: Repo) -> list[Finding]:
        """Every file in HEAD over the ceiling, which a GitHub push refuses past 100 MB."""
        ceiling = self.tree.policy.ceiling_bytes
        listing = repo.git.out("ls-tree", "-r", "-l", "-z", "HEAD")
        # Each entry is `mode type object size`, a tab, then the path; a blob is a file.
        entries = (entry.partition("\t") for entry in listing.split("\0") if entry)
        return [
            _fail(
                repo,
                "size",
                f"{path} is {int(size) / _MEGABYTE:.3g} MB, over the "
                f"{self.tree.policy.ceiling_mb:g} MB ceiling",
            )
            for meta, _, path in entries
            for _, kind, _, size in [meta.split()]
            if kind == "blob" and int(size) > ceiling
        ]

    def _lfs(self, repo: Repo) -> list[Finding]:
        """LFS content this machine has no git-lfs to fetch or push."""
        if repo.lfs() and not self.lfs_installed:
            return [_fail(repo, "lfs", "keeps files in Git LFS and git-lfs is not installed")]
        return []


def _warn(repo: Repo, check: str, detail: str) -> Finding:
    return Finding(repo=repo.name, check=check, verdict=Verdict.WARN, detail=detail)


def _fail(repo: Repo, check: str, detail: str) -> Finding:
    return Finding(repo=repo.name, check=check, verdict=Verdict.FAIL, detail=detail)
