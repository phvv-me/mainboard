"""Render font previews with Core Text, never installing a font or launching a UI."""

import json
import math
import struct
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Self

from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ConfigDict, Field, ValidationError, model_validator

from ..core.errors import MissionError
from .xcrun import xcrun

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..core.execution.tool import Runner
    from ..render.values import Node

_LONGEST = 8192
_FONTS = frozenset({".ttf", ".otf", ".woff", ".woff2", ".ttc", ".otc"})
# What opens a PNG file, and its header chunk.
_PNG = b"\x89PNG\r\n\x1a\n"
_HEADER = b"IHDR"


@Parameter(name="*")
class Preview(FrozenModel):
    """What a preview shows and how large: its text, width, font size and pixel scale."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    text: str
    """preview text; whitespace alone is invalid."""
    width: float = 320
    """logical width in points."""
    size: float = 24
    """font size in points."""
    scale: float = 3
    """pixels per point."""

    @model_validator(mode="after")
    def _drawable(self) -> Self:
        """Refuse text nothing would show and a size nothing could draw."""
        if not self.text.strip() or len(self.text) > _LONGEST:
            raise MissionError("font previews need 1-8192 characters of nonblank text")
        if not all(
            math.isfinite(value) and value > 0 for value in (self.width, self.size, self.scale)
        ):
            raise MissionError("font preview dimensions must be positive and finite")
        return self


class FontPreview(FrozenModel):
    """One rendered face: its source file, the image written, and the image's size."""

    source: Path
    image: Path
    font: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    def document(self) -> dict[str, Node]:
        return {
            "source": str(self.source),
            "image": str(self.image),
            "font": self.font,
            "width": self.width,
            "height": self.height,
        }

    def verify(self, output: Path) -> None:
        """Refuse an image somewhere unexpected, linked, missing, or not the PNG reported."""
        expected = output / (self.source.stem + ".png")
        if self.image != expected or self.image.is_symlink() or not self.image.is_file():
            raise MissionError("native renderer reported an unexpected or missing image")
        with self.image.open("rb") as image:
            header = image.read(24)
        if (
            len(header) != 24
            or header[:8] != _PNG
            or header[12:16] != _HEADER
            or struct.unpack(">II", header[16:24]) != (self.width, self.height)
        ):
            raise MissionError(f"invalid font preview image: {self.image}")


class FontPreviews(FrozenModel):
    images: list[FontPreview] = Field(min_length=1)


class FontRenderer:
    """The first face of each font rendered by the native Swift helper, every image checked."""

    @staticmethod
    def _fonts(paths: Iterable[Path]) -> set[Path]:
        """Every font file `paths` name, a directory standing for the fonts directly inside it."""
        fonts = {
            font for path in paths for font in FontRenderer._found(path.expanduser().resolve())
        }
        if not fonts or len({file.stem.casefold() for file in fonts}) != len(fonts):
            raise MissionError("font previews need files with distinct output stems")
        return fonts

    @staticmethod
    def _found(path: Path) -> set[Path]:
        """The fonts `path` names: itself, or the font files directly inside a directory."""
        if path.is_dir():
            return {
                file.resolve()
                for file in path.iterdir()
                if file.is_file() and file.suffix.lower() in _FONTS
            }
        if path.is_file():
            return {path}
        raise MissionError(f"font source does not exist: {path}")

    def __init__(self, runner: Runner) -> None:
        self.runner = runner

    def render(self, paths: Iterable[Path], preview: Preview, output: Path) -> list[FontPreview]:
        """Render each font `paths` name into the fresh directory `output`."""
        fonts = FontRenderer._fonts(paths)
        output = output.expanduser().resolve()
        if output.exists():
            raise MissionError(f"refusing to overwrite font previews at {output}")
        request = {
            "fonts": [str(file) for file in sorted(fonts)],
            "output": str(output),
            **preview.model_dump(),
        }
        with TemporaryDirectory(prefix="mainboard-fonts-") as temporary:
            asked = Path(temporary) / "request.json"
            asked.write_text(json.dumps(request))
            helper = Path(__file__).with_name("fonts.swift")
            payload = xcrun(self.runner, "swift", str(helper), str(asked))
        try:
            report = FontPreviews.model_validate_json(payload)
        except ValidationError as fault:
            raise MissionError(f"invalid native font preview report: {fault}") from fault
        if len(report.images) != len(fonts) or {item.source for item in report.images} != fonts:
            raise MissionError("native renderer did not report every requested font exactly once")
        for item in report.images:
            item.verify(output)
        return report.images
