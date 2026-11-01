import re

from mainboard import ExecutionPlan
from mainboard.manifest import HostProfile

HOST = "fakehost"
ROOT = "/srv/jobs"
SYSTEM = re.compile(r"\bpython3\b|-m pip\b|\bpip install\b|--break-system-packages")
PLAN = ExecutionPlan(
    host=HOST, profile=HostProfile(kind="ssh", root=ROOT, platform="linux-64"), env="default"
)
