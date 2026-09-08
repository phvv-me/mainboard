import pytest
from pydantic import ValidationError

from mainboard.manifest import Manifest
from mainboard.manifest.schema.plot import PlotStyle


def test_plot_settings_are_typed_without_importing_plotting_dependencies() -> None:
    manifest = Manifest.model_validate(
        {"workspace": {"name": "demo"}, "plots": {"paper": {"dpi": 600, "figsize": [3.25, 2]}}}
    )
    assert manifest.plots["paper"] == PlotStyle(dpi=600, figsize=(3.25, 2))
    assert "plots" in Manifest.uncompiled


@pytest.mark.parametrize("settings", [{"dpi": 0}, {"figsize": [1, -1]}, {"pallete": "deep"}])
def test_invalid_plot_settings_are_refused(settings: dict) -> None:
    with pytest.raises(ValidationError):
        PlotStyle.model_validate(settings)
