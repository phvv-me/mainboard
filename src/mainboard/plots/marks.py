"""Native Matplotlib rounded bars for manuscript figures.

Geometry adapted from the repository's MIT-licensed plotting primitives.
Original copyright notice and permission are retained in LICENSE-marks.
"""

import math
from enum import StrEnum, auto
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox, IdentityTransform

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.backend_bases import RendererBase
    from matplotlib.transforms import BboxBase, Transform

_RADIUS = 1.25
_BAR_GAP = 0.75


class Orientation(StrEnum):
    """Which axis a bar grows along: the value axis, with categories on the other one."""

    VERTICAL = auto()
    HORIZONTAL = auto()

    @property
    def value_axis(self) -> int:
        """The coordinate index the value runs on: x for horizontal bars, y for vertical ones."""
        return 0 if self is Orientation.HORIZONTAL else 1

    def point(self, value: float, *, position: float) -> tuple[float, float]:
        """The (x, y) of `value` on the value axis at category `position`."""
        return (value, position) if self is Orientation.HORIZONTAL else (position, value)


class RoundedBar(FancyBboxPatch):
    """One bar with a rounded data end and a flat baseline.

    The corner radius and the gap to the neighbouring bar are points, never data, so every bar
    in a figure rounds identically however long it is and whatever scale its axis carries. The
    box is therefore rebuilt in display space at each draw: it runs a radius past the baseline
    and is clipped there, which cuts that end square while the data end keeps its corners.
    """

    @property
    def _pixels(self) -> float:
        """Display pixels in one point at the figure's resolution."""
        return self.figure.dpi / 72

    def __init__(
        self,
        orientation: Orientation,
        position: float,
        *,
        value: float,
        baseline: float,
        thickness: float,
        color: str,
        label: str | None = None,
        radius: float = _RADIUS,
        gap: float = _BAR_GAP,
    ) -> None:
        super().__init__(
            (0.0, 0.0),
            0.0,
            0.0,
            boxstyle="round,pad=0",
            transform=IdentityTransform(),
            facecolor=color,
            edgecolor="none",
            label=label,
        )
        self.orientation = orientation
        self.position = position
        self.value = value
        self.baseline = baseline
        self.thickness = thickness
        self.radius = radius
        self.gap = gap

    def _clip_box(self, bbox: BboxBase, *, foot: np.ndarray, head: np.ndarray) -> Bbox:
        """The axes box clipped at the baseline, which cuts the bar's flat end square."""
        along = self.orientation.value_axis
        limits = list(bbox.extents)
        limits[along if head[along] - foot[along] >= 0 else along + 2] = foot[along]
        return Bbox.from_extents(*limits)

    def _display_ends(self, transform: Transform) -> tuple[np.ndarray, np.ndarray]:
        """The bar's baseline end and value end, in display pixels."""
        half = self.thickness / 2
        foot, head = transform.transform(
            [
                self.orientation.point(self.baseline, position=self.position - half),
                self.orientation.point(self.value, position=self.position + half),
            ]
        )
        return foot, head

    def _outline(self, *, foot: np.ndarray, head: np.ndarray) -> tuple[list[float], float]:
        """The bounds box in display pixels, and the corner radius that rounds the data end."""
        along = self.orientation.value_axis
        across = 1 - along
        span = max(abs(head[across] - foot[across]) - self.gap * self._pixels, 1.0)
        length = head[along] - foot[along]
        radius = min(self.radius * self._pixels, span / 2, abs(length))
        start = foot[along] - math.copysign(radius, length)
        box = [0.0, 0.0, 0.0, 0.0]
        box[along], box[along + 2] = min(start, head[along]), abs(head[along] - start)
        box[across], box[across + 2] = (foot[across] + head[across]) / 2 - span / 2, span
        return box, radius

    def draw(self, renderer: RendererBase) -> None:
        self.reshape()
        super().draw(renderer)

    def reshape(self) -> None:
        """Lay the box out in display pixels for the axes as it now stands."""
        axes = self.axes
        if axes is None:
            raise RuntimeError("a bar takes its shape from an axes, so add it to one first")
        foot, head = self._display_ends(axes.transData)
        self.set_visible(bool(np.isfinite([*foot, *head]).all()))
        if not self.get_visible():
            return
        box, radius = self._outline(foot=foot, head=head)
        self.set_bounds(*box)
        self.set_boxstyle("round", pad=0, rounding_size=radius)
        self.set_clip_box(self._clip_box(axes.bbox, foot=foot, head=head))


def _corners(bar: RoundedBar, baseline: float) -> list[tuple[float, float]]:
    """The data-space corners of one bar at its value end and at its baseline end."""
    return [
        bar.orientation.point(end, position=bar.position + edge * bar.thickness / 2)
        for end in (bar.value, baseline)
        for edge in (-1, 1)
    ]


def bars(
    axis: Axes,
    positions: Sequence[float],
    values: Iterable[float],
    *,
    thickness: float,
    color: str,
    baseline: float = 0.0,
    orientation: Orientation = Orientation.VERTICAL,
    label: str | None = None,
    radius: float = _RADIUS,
    gap: float = _BAR_GAP,
) -> list[RoundedBar]:
    """One series of rounded bars, `values` at `positions`, drawn as patches on `axis`.

    A patch is invisible to autoscaling, so the series hands the axes both ends of every bar and
    asks for the view: bars alone frame themselves instead of landing outside the unit square.

    thickness: the slot one bar fills in category units; the surface gap is taken out of it.
    baseline: where every bar starts, in value units, and where its flat end is cut.
    label: the legend entry, carried by the first bar so the series appears once.
    """
    drawn = [
        RoundedBar(
            orientation,
            position,
            value=value,
            baseline=baseline,
            thickness=thickness,
            color=color,
            label=label if index == 0 else None,
            radius=radius,
            gap=gap,
        )
        for index, (position, value) in enumerate(zip(positions, values, strict=True))
    ]
    for bar in drawn:
        axis.add_patch(bar)
    corners: list[tuple[float, float]] = []
    for bar in drawn:
        if math.isfinite(bar.value):
            corners.extend(_corners(bar, baseline))
    axis.update_datalim(corners)
    axis.autoscale_view()
    return drawn
