"""Stand-ins for the optional annotation libraries, with the shape their stubs in `stubs/` state,
and the scripts a fresh interpreter runs against them, printing JSON. Each stand-in logs its calls
into a module-level list the scripts print."""

ROCTX = """
LOG = []


def rangeStart(message):
    LOG.append(["start", message])
    return len(LOG)


def rangeStop(range_id):
    LOG.append(["stop", range_id])
"""

OS_SIGNPOST = """
import enum

LOG = []


class Signposter:
    class Category(enum.Enum):
        PointsOfInterest = "PointsOfInterest"

    def __init__(self, subsystem, category):
        LOG.append(["init", subsystem, category.value])

    def begin_interval(self, msg):
        LOG.append(["begin", msg])

        def end_interval(end_msg=None):
            LOG.append(["end", end_msg or msg])

        return end_interval
"""

REGISTERED = """
import json

from mainboard.profile.tracer import Tracer

Tracer.detect()
print(json.dumps(sorted(tracer.__name__ for tracer in Tracer.implementations())))
"""

SIGNPOST_INTERVALS = """
import json

import os_signpost
from mainboard.profile.providers.apple.tracer import SignpostTracer

tracer = SignpostTracer()
closer = tracer.start("outer")
closer()
tracer.start("detached")()
print(json.dumps(os_signpost.LOG))
"""

ROCTX_RANGES = """
import json

import roctx
from mainboard.profile.providers.amd.tracer import RoctxTracer

tracer = RoctxTracer()
closer = tracer.start("outer")
closer()
tracer.start("detached")()
print(json.dumps(roctx.LOG))
"""
