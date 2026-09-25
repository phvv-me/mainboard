import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

from mainboard import Board, ComputePath, HostFacts
from mainboard.compute import Survey
from mainboard.deps import Change, Dependencies
from mainboard.dispatch import HostSetup
from mainboard.dispatch.dispatcher import Dispatcher
from mainboard.dispatch.state import MonitorReport
from mainboard.doctor import Doctor, Section
from mainboard.monitor import Monitor
from mainboard.probe.stress import StressReport
from mainboard.scaffold import Scaffold, Scaffolded

# What a stand-in is handed, what it hands back, and what one recorded call looks like. The
# verbs pass names and commands positionally and everything else by keyword, so the option
# values are exactly the scalar kinds a flag parses into.
type Owner = Board | Dependencies | Dispatcher | Doctor | Monitor | Scaffold | Survey
type Option = str | int | float | bool | dict[str, str] | None
type Positional = str | tuple[str, ...] | list[str]
type Answer = (
    int
    | None
    | Path
    | HostFacts
    | HostSetup
    | MonitorReport
    | Scaffolded
    | SimpleNamespace
    | StressReport
    | list[Change]
    | list[ComputePath]
    | list[Section]
    | Iterator[MonitorReport]
)
type Relayed = tuple[str, str, tuple[Positional, ...], dict[str, Option]]


class Launcher:
    """The `local` command builder the lanes verb collects a lane through, recorded instead.

    The collection is a real subprocess of this tool inside the workspace environment, so what
    belongs to the verb is the command line it indexed together and how it reads what came
    back. Calling the built command answers `printed` and runs nothing.

    printed: what the collection prints, its `CELL {json}` lines among any other output.
    """

    def __init__(self, printed: str = "") -> None:
        self.printed = printed
        self.argv: list[str] = []

    def __getitem__(self, tokens: str | list[str]) -> Launcher:
        self.argv.extend([tokens] if isinstance(tokens, str) else tokens)
        return self

    def __call__(self) -> str:
        return self.printed


class Lab:
    """A workspace on disk with two packages and one job, the shape a closure walks.

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

    def write(self, path: str, text: str) -> Path:
        """Write `text` at the workspace-relative `path`, creating its directories."""
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf-8")
        return file


def build_lab(root: Path) -> Lab:
    """Create plain source directories and one job, with no Git dependency."""
    lab = Lab(root)
    root.mkdir(parents=True, exist_ok=True)
    lab.write("packages/sub/src/sub/__init__.py", "")
    lab.write("packages/sub/src/sub/thing.py", "THING = 1\n")
    lab.write(
        "mainboard.toml",
        """[workspace]
name = "lab"

[hosts.defaults]
sync = { include = ["research", "packages"] }

[hosts.gold]
kind = "ssh"
root = "/repo"

[tracking]
mode = "off"
""",
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
        """import sub.thing
from cyclopts import App
from mainboard.jobs import job

from ..helper.tools import tool

app = job(
    needs=("data/corpus",),
    resources=("research/camp/registry.toml",),
    fetch="research/camp/experiments/node/evidence",
)(App())


@app.default
def main(x: int = 1) -> int:
    return tool() * x + sub.thing.THING


def plain() -> int:
    return 7
""",
    )
    lab.write("research/other/experiments/__init__.py", "")
    lab.write("research/other/experiments/node/__init__.py", "")
    lab.write("research/other/experiments/node/run.py", "def main() -> None:\n    pass\n")
    lab.write("data/corpus/a.txt", "corpus\n")
    return lab
