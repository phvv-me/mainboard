import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mainboard.git import Tree
from mainboard.manifest.schema.git import GitPolicy

# A machine's own git configuration never reaches these tests: no signing key, no credential
# helper, no LFS filter, no default branch other than the one the fixtures assume. Local-path
# submodules need `protocol.file.allow`, which git refuses by default since 2.38. Every remote
# is spelled `forge:<owner>/<name>.git` and the configuration says where the forge is, so one
# tree built per session is copied into each test and still reaches only that test's remotes.
_CONFIG = """[user]
\tname = Tree Test
\temail = tree@example.com
[init]
\tdefaultBranch = main
[protocol "file"]
\tallow = always
[advice]
\tdetachedHead = false
[url "{remotes}"]
\tinsteadOf = forge:
"""

# The owner every fixture tree declares as its own beside the root's, and the one it does not.
OWNED = "phvv-me"
FOREIGN = "other"

# The workspace manifest the fixture root carries, so the CLI finds a workspace and its `[git]`.
MANIFEST = f"""[workspace]
name = "tree"

[git]
owners = ["{OWNED}"]
ceiling-mb = 0.001
"""

# What a git-lfs stand-in does: log every call beside itself and exit with the code in `code`.
_FAKE_LFS = """#!/bin/sh
here=$(dirname "$0")
echo "$@" >> "$here/calls"
exit "$(cat "$here/code")"
"""

posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="a git-lfs stand-in is a shell script, which Windows cannot run",
)


class Forge:
    """Bare remotes under `<root>/remotes/<owner>/<name>.git`, and working clones of them.

    root: the directory every remote and clone lives under.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def git(where: Path, *args: str) -> str:
        """Run git in `where` for the fixture's own setup, failing the test on any error."""
        done = subprocess.run(
            ["git", "-C", str(where), *args], capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, done.stderr
        return done.stdout.strip()

    @staticmethod
    def url(owner: str, name: str) -> str:
        """The URL a remote is known by, the way a GitHub remote names its owner."""
        return f"forge:{owner}/{name}.git"

    def bare(self, owner: str, name: str) -> Path:
        """The bare remote itself."""
        return self.root / "remotes" / owner / f"{name}.git"

    def seed(self, owner: str, name: str) -> Path:
        """The seeding clone of a remote, where a test can make commits and push them."""
        return self.root / "seeds" / owner / name

    def remote(self, owner: str, name: str, files: dict[str, str]) -> Path:
        """A bare remote seeded with one commit on `main` holding `files`, answering the seed."""
        bare = self.bare(owner, name)
        bare.mkdir(parents=True)
        self.git(bare, "init", "-q", "--bare")
        seed = self.clone(self.url(owner, name), self.seed(owner, name))
        self.commit(seed, "seed", files)
        self.git(seed, "push", "-q", "origin", "main")
        return seed

    def clone(self, url: str, dest: Path) -> Path:
        """A clone of `url` at `dest`, every submodule checked out."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.git(dest.parent, "clone", "-q", "--recurse-submodules", url, dest.name)
        return dest

    def commit(self, repo: Path, message: str, files: dict[str, str]) -> str:
        """Write `files` into `repo`, commit everything, and answer the new commit."""
        for name, text in files.items():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(text, encoding="utf-8")
        self.git(repo, "add", "-A")
        self.git(repo, "commit", "-q", "--allow-empty", "-m", message)
        return self.git(repo, "rev-parse", "HEAD")

    def hook(self, owner: str, name: str, script: str) -> None:
        """Install a `pre-receive` hook on a remote, which is how a remote refuses a push."""
        hook = self.bare(owner, name) / "hooks" / "pre-receive"
        hook.write_text(script, encoding="utf-8", newline="\n")
        hook.chmod(0o755)

    def submodule(self, parent: Path, url: str, path: str, *flags: str) -> None:
        """Add the remote at `url` as a submodule of `parent` at `path`."""
        self.git(parent, "submodule", "add", "-q", *flags, url, path)

    def build(self) -> None:
        """The fixture tree, pushed to its remotes and cloned at `work`."""
        self.remote(FOREIGN, "dep", {"dep.txt": "dep\n"})
        self.remote(FOREIGN, "ref", {"ref.txt": "ref\n"})
        lib = self.remote(OWNED, "lib", {"lib.txt": "lib\n"})
        self.submodule(lib, self.url(FOREIGN, "dep"), "vendor/dep")
        self.commit(lib, "vendor dep", {})
        self.git(lib, "push", "-q", "origin", "main")
        root = self.remote("Pedrexus", "projects", {"mainboard.toml": MANIFEST})
        self.submodule(root, self.url(OWNED, "lib"), "packages/lib", "-b", "main")
        self.submodule(root, self.url(FOREIGN, "ref"), "references/ref")
        self.commit(root, "add submodules", {})
        self.git(root, "push", "-q", "origin", "main")
        self.clone(self.url("Pedrexus", "projects"), self.root / "work")


class Workspace:
    """The fixture tree, checked out: an owned root and library, two foreign references.

    `.` is `Pedrexus/projects`, owned as the root. `packages/lib` is `phvv-me/lib`, owned by
    declaration and following `main` by `.gitmodules`. `references/ref` is `other/ref`, foreign
    and following its remote's HEAD, and `packages/lib/vendor/dep` is `other/dep`, foreign and
    under an owned parent, where the walk stops.
    """

    def __init__(self, forge: Forge, path: Path) -> None:
        self.forge = forge
        self.path = path
        self.lib = path / "packages" / "lib"
        self.ref = path / "references" / "ref"
        self.dep = self.lib / "vendor" / "dep"

    def tree(self, **policy: float | list[str]) -> Tree:
        """The tree at this checkout, owning `phvv-me` and whatever else `policy` says."""
        return Tree(self.path, GitPolicy.model_validate({"owners": [OWNED], **policy}))

    def git(self, where: Path, *args: str) -> str:
        """Run git in one of this checkout's repositories."""
        return self.forge.git(where, *args)

    def head(self, where: Path) -> str:
        """The commit a repository of this checkout is on."""
        return self.git(where, "rev-parse", "HEAD")

    def colleague(self) -> Workspace:
        """A second checkout of the same remotes, the other machine in a pull or a push race."""
        url = self.forge.url("Pedrexus", "projects")
        return Workspace(self.forge, self.forge.clone(url, self.forge.root / "colleague"))


def _configure(config: Path, forge: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every git at `config`, whose `forge:` remotes live under `forge`."""
    remotes = f"{(forge / 'remotes').as_uri()}/"
    config.write_text(_CONFIG.format(remotes=remotes), encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture(scope="session")
def template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The fixture tree built once: forty git calls are the slow part of every test here."""
    root = tmp_path_factory.mktemp("template")
    with pytest.MonkeyPatch.context() as monkeypatch:
        _configure(root / "gitconfig", root / "forge", monkeypatch)
        Forge(root / "forge").build()
    return root / "forge"


@pytest.fixture(autouse=True)
def isolated_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every git in these tests reads the fixture configuration and nothing of this machine's."""
    _configure(tmp_path / "gitconfig", tmp_path / "forge", monkeypatch)


@pytest.fixture
def workspace(template: Path, tmp_path: Path) -> Workspace:
    """A private copy of the fixture tree and its remotes, every submodule detached."""
    forge = Forge(tmp_path / "forge")
    shutil.copytree(template, forge.root, symlinks=True)
    return Workspace(forge, forge.root / "work")


@pytest.fixture
def fake_lfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A git-lfs first on PATH that records its calls and exits zero until told otherwise."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "git-lfs"
    script.write_text(_FAKE_LFS, encoding="utf-8", newline="\n")
    script.chmod(0o755)
    (bin_dir / "code").write_text("0", encoding="utf-8")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir
