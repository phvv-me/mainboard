import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest

from mainboard import HfDataset, HfModel, Needs, RepoFile
from mainboard.experiments import Stageable

if TYPE_CHECKING:
    from pathlib import Path

    from mainboard.experiments.data import Declaration


@pytest.mark.parametrize(
    ("declared", "key"),
    [
        pytest.param(HfModel(repo="org/model"), "org/model@main", id="an-unpinned-model"),
        pytest.param(
            HfModel(repo="org/model", revision="v2"), "org/model@v2", id="a-pinned-model"
        ),
        pytest.param(HfDataset(repo="org/ds"), "org/ds@main", id="an-unpinned-dataset"),
        pytest.param(HfDataset(repo="org/ds", revision="v1"), "org/ds@v1", id="a-pinned-dataset"),
    ],
)
def test_an_hf_declaration_keys_on_repo_at_revision_reading_an_unpinned_one_as_main(
    declared: Declaration, key: str
) -> None:
    assert declared.key == key


@pytest.mark.parametrize(
    ("declared", "present", "absent"),
    [
        pytest.param(
            HfModel(repo="org/model"),
            ("hf download org/model",),
            ("--repo-type", "--revision", "--include"),
            id="a-model-needs-no-repo-type-flag",
        ),
        pytest.param(
            HfModel(repo="org/model", revision="v2"),
            ("--revision v2",),
            ("--repo-type", "--include"),
            id="a-pinned-model-carries-its-revision",
        ),
        pytest.param(
            HfDataset(repo="org/ds"),
            ("--repo-type dataset", "hf download org/ds"),
            ("--revision", "--include"),
            id="a-dataset-declares-its-repo-type",
        ),
        pytest.param(
            HfDataset(repo="org/ds", include="*.parquet", revision="v1"),
            ("--repo-type dataset", "--revision v1", "--include '*.parquet'"),
            (),
            id="a-narrowed-dataset-carries-every-flag",
        ),
    ],
)
def test_an_hf_download_roots_its_cache_under_the_work_root_and_carries_only_declared_flags(
    declared: Declaration, present: Sequence[str], absent: Sequence[str]
) -> None:
    command = declared.command("/work/proj")
    assert command.startswith("HF_HOME=/work/proj/.cache/huggingface hf download ")
    assert all(flag in command for flag in present)
    assert not any(flag in command for flag in absent)


def test_a_repo_file_keys_on_its_current_content_and_stages_as_a_preflight_check(
    tmp_path: Path,
) -> None:
    target = tmp_path / "tokenizer.json"
    target.write_text("{}", encoding="utf-8")
    assert RepoFile(path=str(target)).key == hashlib.sha256(b"{}").hexdigest()
    checked_in = RepoFile(path="research/tok/tokenizer.json")
    assert checked_in.command("/work/proj") == "test -f research/tok/tokenizer.json"


def test_needs_reports_only_the_unstaged_declarations_and_emits_one_command_each(
    tmp_path: Path,
) -> None:
    target = tmp_path / "fixture.json"
    target.write_text("x", encoding="utf-8")
    staged = HfModel(repo="org/present", revision="v1")
    missing = HfDataset(repo="org/missing")
    checked_in = RepoFile(path=str(target))
    needs = Needs((staged, missing, checked_in))
    assert isinstance(needs, tuple)
    assert needs.verify({staged.key}) == [missing, checked_in]
    assert needs.verify({item.key for item in needs}) == []
    assert needs.staging_commands("/work/proj") == [
        item.command("/work/proj") for item in (staged, missing, checked_in)
    ]


@pytest.mark.parametrize(
    "declared",
    [
        pytest.param(HfModel(repo="org/m"), id="model"),
        pytest.param(HfDataset(repo="org/d"), id="dataset"),
        pytest.param(RepoFile(path="tokenizer.json"), id="repo-file"),
    ],
)
def test_every_declaration_satisfies_the_stageable_protocol(declared: Stageable) -> None:
    assert isinstance(declared, Stageable)
