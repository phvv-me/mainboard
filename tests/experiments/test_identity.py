import hashlib
import json

from mainboard.experiments import run_id, study_id
from mainboard.experiments.identity import labelled_study, study_label


def test_run_id_matches_the_canonical_json_sha256_prefix() -> None:
    config = {"bits": 2, "codec": "e8", "model": "meta-llama/Llama-3-8b"}
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    assert run_id(config) == expected
    assert len(expected) == 16


def test_run_id_is_stable_under_key_order() -> None:
    assert run_id({"a": 1, "b": 2}) == run_id({"b": 2, "a": 1})


def test_run_id_differs_for_different_configs() -> None:
    assert run_id({"a": 1}) != run_id({"a": 2})


def test_study_id_is_twelve_hex_chars() -> None:
    identity, _ = study_id(
        experiment="joint-search", config_space={"bits": [1, 2, 4]}, git_sha="a"
    )
    assert len(identity) == 12
    assert all(char in "0123456789abcdef" for char in identity)


def test_study_id_is_stable_for_the_same_inputs() -> None:
    first, _ = study_id(experiment="e", config_space={"a": 1}, git_sha="s")
    second, _ = study_id(experiment="e", config_space={"a": 1}, git_sha="s")
    assert first == second


def test_study_id_changes_with_git_sha() -> None:
    first, _ = study_id(experiment="e", config_space={}, git_sha="aaa")
    second, _ = study_id(experiment="e", config_space={}, git_sha="bbb")
    assert first != second


def test_study_id_changes_with_experiment() -> None:
    first, _ = study_id(experiment="e1", config_space={}, git_sha="s")
    second, _ = study_id(experiment="e2", config_space={}, git_sha="s")
    assert first != second


def test_study_id_changes_with_config_space() -> None:
    first, _ = study_id(experiment="e", config_space={"bits": [1]}, git_sha="s")
    second, _ = study_id(experiment="e", config_space={"bits": [1, 2]}, git_sha="s")
    assert first != second


def test_study_id_config_space_is_key_order_independent() -> None:
    first, _ = study_id(experiment="e", config_space={"a": 1, "b": 2}, git_sha="s")
    second, _ = study_id(experiment="e", config_space={"b": 2, "a": 1}, git_sha="s")
    assert first == second


def test_study_id_slug_embeds_the_experiment_and_the_id_prefix() -> None:
    identity, slug = study_id(experiment="joint-search", config_space={}, git_sha="s")
    assert slug == f"joint-search-{identity[:6]}"


def test_study_label_names_a_study_and_optionally_one_of_its_trials() -> None:
    assert study_label("abc123") == "study:abc123"
    assert study_label("abc123", trial="bits-2") == "study:abc123/bits-2"


def test_labelled_study_reads_the_id_back_off_either_shape() -> None:
    assert labelled_study("study:abc123") == "abc123"
    assert labelled_study("study:abc123/bits-2") == "abc123"


def test_labelled_study_is_empty_for_a_name_that_is_not_a_study_label() -> None:
    assert not labelled_study("nightly-bench")
