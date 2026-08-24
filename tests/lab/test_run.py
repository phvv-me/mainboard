from pathlib import Path

from mainboard import Project
from mainboard.lab import Run
from mainboard.lab.experiment import DeclaredExperiment, Fixture
from mainboard.lab.run import default_dataset_resolver


def test_a_run_resolves_a_dataset_through_the_resolver_it_was_injected_with(
    context: Run, tmp_path: Path
) -> None:
    assert default_dataset_resolver("org/dataset") == Path(Project().out_dir) / "data/org/dataset"
    assert context.dataset("org/dataset") == default_dataset_resolver("org/dataset")
    injected = Run(
        model_id="gpt2",
        config=DeclaredExperiment(),
        artifact_dir=tmp_path,
        dataset_resolver=lambda name: tmp_path / "custom" / name,
    )
    assert injected.dataset("org/dataset") == tmp_path / "custom/org/dataset"


def test_a_run_writes_a_named_artifact_under_its_own_dir_and_opens_a_dormant_span(
    tmp_path: Path,
) -> None:
    run = Run(model_id="gpt2", config=DeclaredExperiment(), artifact_dir=tmp_path / "trial")
    written: list[tuple[Path, Fixture]] = []

    def writer(path: Path, payload: Fixture) -> None:
        written.append((path, payload))
        path.write_text("ok", encoding="utf-8")

    path = run.artifact("out.txt", writer, {"score": 1.0})
    assert path == tmp_path / "trial/out.txt"
    assert path.read_text(encoding="utf-8") == "ok"
    assert written == [(path, {"score": 1.0})]
    with run.span("measure"):
        pass
