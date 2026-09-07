import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

from mainboard import Board, ComputePath, HostFacts
from mainboard.compute import Survey
from mainboard.deps import Change, Dependencies
from mainboard.dispatch import HostSetup
from mainboard.dispatch.state import MonitorReport
from mainboard.doctor import Doctor, Section
from mainboard.monitor import Monitor
from mainboard.scaffold import Scaffold, Scaffolded

# What a stand-in is handed, what it hands back, and what one recorded call looks like. The
# verbs pass names and commands positionally and everything else by keyword, so the option
# values are exactly the scalar kinds a flag parses into.
type Owner = Board | Dependencies | Doctor | Monitor | Scaffold | Survey
type Option = str | int | float | bool | dict[str, str] | None
type Positional = str | tuple[str, ...]
type Answer = (
    int
    | None
    | HostFacts
    | HostSetup
    | MonitorReport
    | Scaffolded
    | SimpleNamespace
    | list[Change]
    | list[ComputePath]
    | list[Section]
    | Iterator[MonitorReport]
)
type Relayed = tuple[str, str, tuple[Positional, ...], dict[str, Option]]


class Lab:
    """A workspace on disk with a repository, a submodule and one job, the shape a closure walks.

    The job at `research/camp/experiments/node/run.py` imports a sibling node through its
    package, a distribution under `packages/core/src` that keeps a data file and an unused
    module, and a distribution inside the submodule at `packages/sub`. A second campaign under
    `research/other` and an ignored `data/` directory stand beside them and must never ship.

    root: the workspace root.
    """

    JOB = "research/camp/experiments/node/run.py"
    HOME = "research/camp"
    DISTRIBUTIONS = ("packages/core/src", "packages/sub/src")
    # Where the lab's compiled target environment would sit, the shape a dispatched job's
    # closure reads a distribution's installed shape from.
    ENVIRONMENT = ".mainboard/envs/default/.pixi/envs/default"

    def __init__(self, root: Path) -> None:
        self.root = root

    def compiled(self, distribution: str, *extensions: Path, imports: str = "") -> Path:
        """A compiled target environment holding `distribution`, its RECORD recording `extensions`.

        Each extension is spelled as an install spells one, as a site-packages-relative row,
        however far that climbs; a missing file is written, so a record of an extension inside
        the source tree points at a file that is really there. Answers the prefix, so a test
        hands the closure a target environment without a pixi in sight.

        distribution: the dist-info name, free to differ from the import it claims.
        extensions: absolute paths of compiled extensions the RECORD records.
        imports: a top-level import root declared in `top_level.txt`, empty for a distribution
            whose RECORD's own paths must speak for it.
        """
        prefix = self.root / Lab.ENVIRONMENT
        site = prefix / "lib/python3.14/site-packages"
        info = site / f"{distribution}-0.0.1.dist-info"
        info.mkdir(parents=True, exist_ok=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 0.0.1\n", encoding="utf-8"
        )
        if imports:
            (info / "top_level.txt").write_text(f"{imports}\n", encoding="utf-8")
        for extension in extensions:
            extension.parent.mkdir(parents=True, exist_ok=True)
            extension.write_text("not an elf, a stub\n", encoding="utf-8")
        rows = [os.path.relpath(extension, site).replace(os.sep, "/") for extension in extensions]
        rows.append(f"{info.relative_to(site).as_posix()}/RECORD,,")
        (info / "RECORD").write_text("\n".join(rows) + "\n", encoding="utf-8")
        return prefix

    def git(self, *args: str, cwd: Path | None = None) -> str:
        """Run git in the workspace (or `cwd`), answering its stripped stdout."""
        done = subprocess.run(
            ["git", "-C", str(cwd or self.root), *args], check=True, capture_output=True, text=True
        )
        return done.stdout.strip()

    def write(self, path: str, text: str) -> Path:
        """Write `text` at the workspace-relative `path`, creating its directories."""
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf-8")
        return file

    def commit(self, message: str = "more", *, cwd: Path | None = None) -> str:
        """Stage and commit everything in the workspace (or `cwd`), answering the commit."""
        self.git("add", "-A", cwd=cwd)
        self.git(
            "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", message, cwd=cwd
        )
        return self.git("rev-parse", "HEAD", cwd=cwd)


def build_lab(root: Path) -> Lab:
    """Materialise a `Lab` at `root`: two repositories, one job, one commit each."""
    lab = Lab(root)
    subsource = root.parent / f"{root.name}-sub"
    subsource.mkdir(parents=True)
    Lab(subsource).write("src/sub/__init__.py", "")
    Lab(subsource).write("src/sub/thing.py", "THING = 1\n")
    lab.git("init", "-q", cwd=subsource)
    lab.commit("sub", cwd=subsource)
    root.mkdir(parents=True, exist_ok=True)
    lab.git("init", "-q")
    lab.write(
        "mainboard.toml",
        '[workspace]\nname = "lab"\n\n'
        '[hosts.defaults]\nsync = { include = ["research", "packages"] }\n\n'
        '[hosts.gold]\nkind = "ssh"\nroot = "/repo"\n\n'
        '[tracking]\nmode = "off"\n',
    )
    lab.write(".gitignore", "data/\n__pycache__/\n*.pyc\n*_generated.py\n")
    lab.write("packages/core/src/core/__init__.py", "from .util import helper\n")
    lab.write("packages/core/src/core/util.py", "def helper() -> int:\n    return 1\n")
    lab.write("packages/core/src/core/spare.py", "UNUSED = True\n")
    lab.write("packages/core/src/core/data.txt", "resource\n")
    lab.write("research/camp/registry.toml", "[node]\n")
    lab.write("research/camp/experiments/__init__.py", "")
    lab.write("research/camp/experiments/helper/__init__.py", "")
    lab.write(
        "research/camp/experiments/helper/tools.py",
        "from core import helper\n\n\ndef tool() -> int:\n    return helper()\n",
    )
    lab.write("research/camp/experiments/node/__init__.py", "")
    lab.write("research/camp/experiments/node/node.md", "# node\n")
    lab.write(
        Lab.JOB,
        "import sub.thing\n"
        "from cyclopts import App\n"
        "from mainboard.jobs import job\n"
        "\n"
        "from ..helper.tools import tool\n"
        "\n"
        "app = job(\n"
        '    needs=("data/corpus",),\n'
        '    resources=("research/camp/registry.toml",),\n'
        '    fetch="research/camp/experiments/node/evidence",\n'
        ")(App())\n"
        "\n"
        "\n"
        "@app.default\n"
        "def main(x: int = 1) -> int:\n"
        "    return tool() * x + sub.thing.THING\n"
        "\n"
        "\n"
        "def plain() -> int:\n"
        "    return 7\n",
    )
    lab.write("research/other/experiments/__init__.py", "")
    lab.write("research/other/experiments/node/__init__.py", "")
    lab.write("research/other/experiments/node/run.py", "def main() -> None:\n    pass\n")
    lab.write("data/corpus/a.txt", "corpus\n")
    lab.git(
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(subsource),
        "packages/sub",
    )
    lab.commit("lab")
    return lab
