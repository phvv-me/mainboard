from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.jobs import Declaration
from mainboard.jobs.target import Target, dotted, home_of

from ..support import Lab


def test_a_job_is_spelled_by_file_and_name_and_an_ordinary_command_is_left_alone(
    lab: Lab,
) -> None:
    """`path/to/file.py::name` is a job; `python -m foo` and a task name are not."""
    spelled = Target.spelled([f"{Lab.JOB}::app", "--x", "3"], lab.root)
    assert spelled == Target(file=Lab.JOB, name="app", args=("--x", "3"))
    assert spelled.spelling == f"{Lab.JOB}::app --x 3"
    assert spelled.node == "research/camp/experiments/node"
    assert Target.spelled(["python", "-m", "foo"], lab.root) is None
    assert Target.spelled(["test", "--quiet"], lab.root) is None
    assert Target.spelled([], lab.root) is None
    # A `.py` that is not there is an ordinary token, a named target inside one is a mistake.
    assert Target.spelled(["missing.py"], lab.root) is None
    with pytest.raises(MissionError, match="no job file at missing.py"):
        Target.spelled(["missing.py::main"], lab.root)


def test_a_bare_file_means_app_then_main_and_refuses_a_file_defining_neither(lab: Lab) -> None:
    assert Target.spelled([Lab.JOB], lab.root).name == "app"
    only_main = lab.write("research/camp/only_main.py", "def main() -> None:\n    pass\n")
    assert Target.spelled([str(only_main)], lab.root) == Target(
        file="research/camp/only_main.py", name="main"
    )
    lab.write("research/camp/neither.py", "x = 1\n")
    with pytest.raises(MissionError, match="defines neither `app` or `main`"):
        Target.spelled(["./research/camp/neither.py"], lab.root)


def test_an_absolute_file_is_rerooted_and_one_outside_the_workspace_is_refused(
    lab: Lab, tmp_path: Path
) -> None:
    assert Target.spelled([str(lab.root / Lab.JOB)], lab.root).file == Lab.JOB
    outside = tmp_path / "elsewhere.py"
    outside.write_text("app = 1\n", encoding="utf-8")
    with pytest.raises(MissionError, match="outside the workspace"):
        Target.spelled([str(outside)], lab.root)


def test_the_declaration_is_read_off_the_job_file(lab: Lab) -> None:
    assert Target.spelled([Lab.JOB], lab.root).declaration(lab.root) == Declaration(
        needs=("data/corpus",),
        resources=("research/camp/registry.toml",),
        fetch="research/camp/experiments/node/evidence",
    )
    assert Target.spelled([f"{Lab.JOB}::plain"], lab.root).declaration(lab.root) == Declaration()


def test_a_module_is_imported_from_where_its_package_chain_stops(lab: Lab) -> None:
    """Two parents carry `__init__.py`, so the file is `experiments.node.run` from above them."""
    file = lab.root / Lab.JOB
    home = home_of(file, root=lab.root)
    assert home == lab.root / Lab.HOME
    assert dotted(file, home=home) == "experiments.node.run"
    assert dotted(lab.root / "research/camp/experiments/node/__init__.py", home=home) == (
        "experiments.node"
    )
    # A bare script is imported from its own directory, and nothing above the root is climbed.
    script = lab.write("research/camp/script.py", "app = 1\n")
    assert home_of(script, root=lab.root) == script.parent
    assert home_of(lab.root / "top.py", root=lab.root) == lab.root


@given(args=st.lists(st.text(min_size=1).filter(lambda token: "\n" not in token), max_size=4))
def test_a_spelling_round_trips_through_the_shell(args: list[str]) -> None:
    """What the records call a job is what `shlex.split` hands back to the runner."""
    import shlex

    target = Target(file="a/run.py", name="app", args=tuple(args))
    tokens = shlex.split(target.spelling)
    assert tokens[0] == "a/run.py::app"
    assert tuple(tokens[1:]) == tuple(args)
