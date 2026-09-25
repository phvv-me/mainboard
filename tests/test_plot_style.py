import pytest
from pydantic import ValidationError

from mainboard.manifest import Manifest
from mainboard.manifest.loading import load_plot_config
from mainboard.manifest.schema.plot import PlotStyle


def test_plot_settings_are_typed_without_importing_plotting_dependencies() -> None:
    manifest = Manifest.model_validate(
        {"workspace": {"name": "demo"}, "plots": {"paper": {"dpi": 600, "figsize": [3.25, 2]}}}
    )
    assert manifest.plots["paper"] == PlotStyle(dpi=600, figsize=(3.25, 2))
    assert "plots" in Manifest.uncompiled


@pytest.mark.parametrize(
    "settings",
    [{"dpi": 0}, {"figsize": [1, -1]}, {"pallete": "deep"}, {"palette": []}],
)
def test_invalid_plot_settings_are_refused(settings: dict) -> None:
    with pytest.raises(ValidationError):
        PlotStyle.model_validate(settings)


def test_project_style_overlays_only_explicit_fields(tmp_path) -> None:
    root = tmp_path / "mainboard.toml"
    root.write_text("""[workspace]
name="demo"
[plots.paper]
palette=["#745399", "#b7282e"]
figsize=[3.25,2.1]
dpi=600
[plots.paper.rc]
"font.size"=9
"axes.labelsize"=8
[plots.paper.colors]
reference="#555555"
ours="#745399"
""")
    project = tmp_path / "plots.toml"
    project.write_text("""[workspace]
name="figures"
[plots.paper]
figsize=[5.5,2.1]
[plots.paper.rc]
"axes.labelsize"=10
[plots.paper.colors]
ours="#b7282e"
""")
    style = load_plot_config(root, project).plots["paper"]
    assert style.palette == ["#745399", "#b7282e"]
    assert style.dpi == 600
    assert style.figsize == (5.5, 2.1)
    assert style.rc == {"font.size": 9, "axes.labelsize": 10}
    assert style.colors == {"reference": "#555555", "ours": "#b7282e"}


def test_project_can_explicitly_reset_a_shared_default() -> None:
    shared = PlotStyle(dpi=600, theme="dark_background", figsize=(5, 3))
    project = PlotStyle(dpi=300, theme="default", figsize=None)
    assert project.merged(shared) == project
