"""Read Apple's installed toolchain and test evidence without launching the IDE."""

from .bundle.core import BundleChecks, BundleReport
from .evidence import BuildReport, TestReport
from .fonts import Preview
from .toolchain import Apple
from .xcode import XcodeRun

__all__ = [
    "Apple",
    "BuildReport",
    "BundleChecks",
    "BundleReport",
    "Preview",
    "TestReport",
    "XcodeRun",
]
