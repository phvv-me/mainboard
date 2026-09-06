# Remote dispatch: sync a workspace to a host, submit a job, poll and pull results back.

from .dispatcher import Dispatcher, Handle, Verdict
from .onboard import HostSetup
from .provenance import Source
from .shared import now
from .shipment import Shipment
from .snapshots import Snapshots
from .sync import GitignoreFilter, SyncLock
from .targets import Facts, resolve, smallest_fit, ssh_hosts
from .transport import DaemonDown, HostUnreachable, SshTransport

__all__ = [
    "DaemonDown",
    "Dispatcher",
    "Facts",
    "GitignoreFilter",
    "Handle",
    "HostSetup",
    "HostUnreachable",
    "Snapshots",
    "Shipment",
    "Snapshots",
    "Source",
    "SshTransport",
    "SyncLock",
    "Verdict",
    "now",
    "resolve",
    "smallest_fit",
    "ssh_hosts",
]
