import hashlib
import json

from hypothesis import given
from hypothesis import strategies as st

from mainboard.experiments import run_id, study_id
from mainboard.experiments.identity import labelled_study, labelled_trial, study_label

from ..strategies import TEXT, WORDS

# A trial config or a study's config space as the layer above hands one over, JSON-native
# values only. Every key is lowercase, so an uppercase key is guaranteed to be a new one.
_SPACE = st.dictionaries(WORDS, st.integers() | WORDS | st.booleans(), max_size=4)


@given(config=_SPACE)
def test_a_trial_identity_is_the_canonical_json_sha256_prefix_whatever_the_key_order(
    *, config: dict[str, int | str | bool]
) -> None:
    """Byte for byte the id an in-process research run resolves the same config to."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    identity = run_id(config)
    assert identity == hashlib.sha256(canonical.encode()).hexdigest()[:16]
    assert len(identity) == 16
    assert identity == run_id(dict(reversed(list(config.items()))))
    assert identity != run_id({**config, "EXTRA": 1})


@given(experiment=WORDS, space=_SPACE, git_sha=WORDS)
def test_a_study_identity_moves_with_its_experiment_config_space_or_git_sha(
    *, experiment: str, space: dict[str, int | str | bool], git_sha: str
) -> None:
    identity, slug = study_id(experiment=experiment, config_space=space, git_sha=git_sha)
    assert len(identity) == 12
    assert set(identity) <= set("0123456789abcdef")
    assert slug == f"{experiment}-{identity[:6]}"
    reordered = dict(reversed(list(space.items())))
    assert study_id(experiment=experiment, config_space=reordered, git_sha=git_sha)[0] == identity
    moved = {
        study_id(experiment=f"{experiment}X", config_space=space, git_sha=git_sha)[0],
        study_id(experiment=experiment, config_space={**space, "EXTRA": 1}, git_sha=git_sha)[0],
        study_id(experiment=experiment, config_space=space, git_sha=f"{git_sha}X")[0],
    }
    assert identity not in moved


@given(study=WORDS, trial=WORDS, other=TEXT)
def test_a_dispatch_label_round_trips_the_study_id_it_names(
    *, study: str, trial: str, other: str
) -> None:
    assert study_label(study) == f"study:{study}"
    assert study_label(study, trial=trial) == f"study:{study}/{trial}"
    assert labelled_study(study_label(study)) == study
    assert labelled_study(study_label(study, trial=trial)) == study
    assert not labelled_study(f"nightly-{other}")
    assert labelled_trial(study_label(study, trial=trial)) == trial
    assert not labelled_trial(study_label(study))
    assert not labelled_trial(f"nightly-{other}")
