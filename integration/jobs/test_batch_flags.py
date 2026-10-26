"""The resource flags of `job submit` reach a `--job` batch, and a spec file refuses them."""

from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.verbs.options import Declared


def test_inline_jobs_take_the_resource_flags_as_their_defaults(tmp_path: Path) -> None:
    declared = Declared(job=("crimson:echo one", "gold:echo two"))

    spec = declared.spec(tmp_path, name="wave", resources={"walltime": "00:07:00", "gpus": 2})

    assert [(job.target, job.walltime, job.gpus) for job in spec.jobs] == [
        ("crimson", "00:07:00", 2),
        ("gold", "00:07:00", 2),
    ]


def test_a_spec_file_refuses_flags_it_would_ignore(tmp_path: Path) -> None:
    (tmp_path / "fleet.toml").write_text(
        '[[jobs]]\ntarget = "gold"\ncommand = "echo hi"\n', encoding="utf-8"
    )

    with pytest.raises(MissionError, match="--walltime, --mem-gb"):
        Declared(batch="fleet.toml").spec(
            tmp_path, resources={"walltime": "00:07:00", "mem_gb": 8}
        )

    assert Declared(batch="fleet.toml").spec(tmp_path, resources={}).jobs[0].target == "gold"
