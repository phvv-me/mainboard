import hashlib
from typing import TYPE_CHECKING

import pytest

from mainboard import HfDataset, HfModel, Needs, RepoFile
from mainboard.experiments import Stageable

if TYPE_CHECKING:
    from pathlib import Path


def test_hf_model_key_defaults_an_unpinned_revision_to_main() -> None:
    assert HfModel(repo="org/model").key == "org/model@main"


def test_hf_model_key_uses_the_pinned_revision() -> None:
    assert HfModel(repo="org/model", revision="v2").key == "org/model@v2"


def test_hf_model_command_sets_hf_home_under_the_work_root_with_no_repo_type_flag() -> None:
    command = HfModel(repo="org/model").command("/work/proj")
    assert command.startswith("HF_HOME=/work/proj/.cache/huggingface")
    assert "hf download org/model" in command
    assert "--repo-type" not in command
    assert "--revision" not in command


def test_hf_model_command_pins_the_revision_when_given() -> None:
    command = HfModel(repo="org/model", revision="v2").command("/work/proj")
    assert "--revision v2" in command


def test_hf_dataset_key_defaults_an_unpinned_revision_to_main() -> None:
    assert HfDataset(repo="org/ds").key == "org/ds@main"


def test_hf_dataset_command_carries_the_repo_type_and_include_glob() -> None:
    command = HfDataset(repo="org/ds", include="*.parquet", revision="v1").command("/work/proj")
    assert "--repo-type dataset" in command
    assert "--revision v1" in command
    assert "--include '*.parquet'" in command
    assert "org/ds" in command


def test_hf_dataset_command_omits_include_when_unset() -> None:
    assert "--include" not in HfDataset(repo="org/ds").command("/work/proj")


def test_repo_file_key_is_the_sha256_of_its_current_content(tmp_path: Path) -> None:
    target = tmp_path / "tokenizer.json"
    target.write_text("{}")
    assert RepoFile(path=str(target)).key == hashlib.sha256(b"{}").hexdigest()


def test_repo_file_command_is_a_preflight_existence_check() -> None:
    item = RepoFile(path="research/tok/tokenizer.json")
    assert item.command("/work/proj") == "test -f research/tok/tokenizer.json"


@pytest.mark.parametrize(
    "item",
    [HfModel(repo="org/m"), HfDataset(repo="org/d"), RepoFile(path="tokenizer.json")],
)
def test_declarations_satisfy_the_stageable_protocol(item: Stageable) -> None:
    assert isinstance(item, Stageable)


def test_needs_is_a_plain_tuple_of_declarations() -> None:
    needs = Needs((HfModel(repo="org/m"),))
    assert isinstance(needs, tuple)
    assert len(needs) == 1


def test_needs_verify_returns_only_the_items_missing_from_what_is_already_staged(
    tmp_path: Path,
) -> None:
    target = tmp_path / "f.json"
    target.write_text("x")
    present = HfModel(repo="org/present", revision="v1")
    missing_model = HfModel(repo="org/missing")
    file_item = RepoFile(path=str(target))
    needs = Needs((present, missing_model, file_item))
    assert needs.verify({present.key}) == [missing_model, file_item]


def test_needs_verify_returns_nothing_missing_when_every_key_is_already_present() -> None:
    item = HfModel(repo="org/m", revision="v1")
    needs = Needs((item,))
    assert needs.verify({item.key}) == []


def test_needs_staging_commands_emits_one_command_per_declared_item() -> None:
    needs = Needs((HfModel(repo="org/m"), HfDataset(repo="org/d")))
    commands = needs.staging_commands("/work/proj")
    assert len(commands) == 2
    assert all("/work/proj" in command for command in commands)
