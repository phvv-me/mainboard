from pathlib import Path
from tempfile import TemporaryDirectory

from patos import FrozenModel, Runtime

from ..constants import REMOTE
from ..process import Git

_NON_SYMBOLIC_REF_NAME_FORMAT = "--format=%(if)%(symref)%(then)%(else)%(refname:short)%(end)"


class Remote(FrozenModel):
    """The `origin` of one repository: what it holds and whether it hands a commit out.

    git: git run in the repository's working tree.
    url: where this repository pushes, which a commit is asked for when no branch this clone
        knows holds it.
    """

    git: Runtime[Git]
    url: str

    def _offered(self, commit: str) -> bool:
        """Whether `origin` serves `commit` when asked for it alone, without its history or tree.

        The fetch lands in a scratch repository, so this clone's objects and shallow boundary
        stay as they were; a remote that ignores the tree filter sends one snapshot at most.
        """
        with TemporaryDirectory() as scratch:
            probe = Git(Path(scratch))
            probe.out("init", "--quiet", "--bare")
            asked = ("fetch", "--quiet", "--depth=1", "--filter=tree:0", self.url, commit)
            return probe.run(*asked, network=True).returncode == 0

    def fetch(self) -> str:
        """Fetch from `origin`, pruning branches it deleted, and return git's complaint if any."""
        found = self.git.run("fetch", "--prune", "--quiet", REMOTE, network=True)
        return found.stderr.strip() if found.returncode else ""

    def homes(self, commit: str) -> list[str]:
        """The remote branches holding `commit`, empty when none does or it is unknown here.

        This is what decides whether a parent pointer at `commit` can be cloned: a commit no
        remote branch reaches is one the remote may refuse to serve.
        """
        found = self.git.run(
            "for-each-ref",
            _NON_SYMBOLIC_REF_NAME_FORMAT,
            "--contains",
            commit,
            f"refs/remotes/{REMOTE}/",
        )
        return [] if found.returncode else found.stdout.split()

    def serves(self, commit: str) -> bool:
        """Whether `origin` hands out `commit`, which a clone of a parent pointing at it needs.

        A remote branch this clone knows holding it answers without the network. Otherwise the
        remote itself is asked for that one commit, since a clone's refs see only part of it: a
        shallow clone cannot trace a branch past its boundary, and a single-branch clone never
        tracks the other branches that may hold it.
        """
        return bool(self.homes(commit)) or self._offered(commit)
