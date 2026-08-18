from pathlib import Path

from mainboard import Project
from mainboard.lab import Run
from mainboard.lab.experiment import DeclaredExperiment, Fixture
from mainboard.lab.run import default_dataset_resolver


def test_default_dataset_resolver_lives_under_the_project_cache() -> None:
    resolved = default_dataset_resolver("org/dataset")
    assert resolved == Path(Project().out_dir) / "data" / "org/dataset"


def test_run_dataset_uses_the_default_resolver(tmp_path: Path) -> None:
    run = Run(model_id="gpt2", config=DeclaredExperiment(), artifact_dir=tmp_path)
    assert run.dataset("org/dataset") == default_dataset_resolver("org/dataset")


def test_run_dataset_uses_an_injected_resolver(tmp_path: Path) -> None:
    run = Run(
        model_id="gpt2",
        config=DeclaredExperiment(),
        artifact_dir=tmp_path,
        dataset_resolver=lambda name: tmp_path / "custom" / name,
    )
    assert run.dataset("org/dataset") == tmp_path / "custom" / "org/dataset"


def test_run_artifact_writes_under_artifact_dir_and_runs_the_writer(tmp_path: Path) -> None:
    run = Run(model_id="gpt2", config=DeclaredExperiment(), artifact_dir=tmp_path / "trial")
    calls: list[tuple[Path, Fixture]] = []

    def writer(path: Path, payload: Fixture) -> None:
        calls.append((path, payload))
        path.write_text("ok", encoding="utf-8")

    path = run.artifact("out.txt", writer, {"score": 1.0})
    assert path == tmp_path / "trial" / "out.txt"
    assert path.read_text() == "ok"
    assert calls == [(path, {"score": 1.0})]


def test_run_span_is_dormant_without_an_active_profiler(tmp_path: Path) -> None:
    run = Run(model_id="gpt2", config=DeclaredExperiment(), artifact_dir=tmp_path)
    with run.span("measure"):
        pass
