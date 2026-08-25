from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.batch import BatchSpec

_FILE = """
name = "sweep"

[defaults]
walltime = "00:10:00"
runtime_s = 120

[[jobs]]
name = "big"
target = "miyabi-g"
command = "python -m train"
data = ["data/shard.npz"]
mem_gb = 100

[[jobs]]
target = "gold"
command = "python -m eval"
"""


def test_a_spec_file_names_the_batch_layers_its_defaults_and_labels_every_job(
    tmp_path: Path,
) -> None:
    """A batch says a shared walltime once, and a job that named no label still has one."""
    (tmp_path / "sweep.toml").write_text(_FILE)
    spec = BatchSpec.load(tmp_path / "sweep.toml")
    assert spec.name == "sweep"
    assert [job.name for job in spec.jobs] == ["big", "gold-2"]
    assert [job.walltime for job in spec.jobs] == ["00:10:00", "00:10:00"]
    assert [job.runtime_s for job in spec.jobs] == [120.0, 120.0]
    assert spec.jobs[0].data == ("data/shard.npz",)
    assert spec.jobs[0].mem_gb == 100


def test_a_spec_file_falls_back_to_its_own_stem_for_the_batch_name(tmp_path: Path) -> None:
    (tmp_path / "nightly.toml").write_text('[[jobs]]\ntarget = "gold"\ncommand = "true"\n')
    assert BatchSpec.load(tmp_path / "nightly.toml").name == "nightly"


@pytest.mark.parametrize(
    ("text", "detail"),
    [
        ("", "no batch spec at"),
        ("[[jobs]\n", "not valid TOML"),
        ('[[jobs]]\ncommand = "true"\n', "unusable job"),
    ],
    ids=["a spec file that is not there", "a spec that is not TOML", "a job missing its target"],
)
def test_an_unusable_spec_is_one_sentence_naming_what_is_wrong(
    tmp_path: Path, text: str, detail: str
) -> None:
    path = tmp_path / "broken.toml"
    if text:
        path.write_text(text)
    with pytest.raises(MissionError, match=detail):
        BatchSpec.load(path)


def test_jobs_typed_at_the_command_line_split_at_the_first_colon() -> None:
    """A host alias never carries a colon and a command routinely does, so the first one splits."""
    spec = BatchSpec.inline("quick", ["gold:python -c 'print(1)'", "miyabi-g:bash -lc 'echo x'"])
    assert [(job.target, job.command) for job in spec.jobs] == [
        ("gold", "python -c 'print(1)'"),
        ("miyabi-g", "bash -lc 'echo x'"),
    ]
    with pytest.raises(MissionError, match="target:command"):
        BatchSpec.inline("quick", ["python -m foo"])


@pytest.mark.parametrize(
    ("jobs", "detail"),
    [
        ([{"name": "a", "target": "gold", "command": "x"}] * 2, "job names repeat"),
        ([], "declares no jobs"),
    ],
    ids=["two jobs under one name", "a batch declaring nothing"],
)
def test_a_batch_refuses_a_declaration_its_receipts_could_not_key(
    jobs: list[dict[str, str]], detail: str
) -> None:
    """Every receipt line is keyed by job name, so a repeat would fold two jobs into one row."""
    with pytest.raises(ValueError, match=detail):
        BatchSpec.of("clash", jobs)


def test_the_batch_id_follows_the_declaration_rather_than_the_moment_it_was_read() -> None:
    """The same spec addresses the same receipts, so prepare, run and watch share one stream."""
    declared = BatchSpec.inline("smoke", ["gold:true"])
    assert declared.batch_id == BatchSpec.inline("smoke", ["gold:true"]).batch_id
    assert declared.batch_id.startswith("smoke-")
    assert declared.batch_id != BatchSpec.inline("smoke", ["gold:false"]).batch_id


def test_a_job_hands_the_board_only_the_resources_it_declares() -> None:
    [job] = BatchSpec.of("one", [{"target": "gold", "command": "true", "gpus": 2}]).jobs
    assert job.submission() == {
        "queue": "",
        "walltime": "",
        "mem_gb": 0,
        "gpus": 2,
        "gpu_name": "",
        "max_usd": 0.0,
        "nodes": 1,
        "env": "",
        "container": "",
        "fetch": None,
        "node": "",
    }
