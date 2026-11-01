__lazy_modules__ = [
    "mainboard.log.core",
]

from .core import EventDict, configure, logger, sinks

__all__ = ["EventDict", "configure", "logger", "sinks"]
