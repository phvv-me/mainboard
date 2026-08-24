# Remote dispatch: sync a workspace to a host, submit a job, poll and pull results back.

from .dispatcher import Dispatcher, Handle, Verdict
from .onboard import HostSetup
from .shared import now
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
    "SshTransport",
    "SyncLock",
    "Verdict",
    "now",
    "resolve",
    "smallest_fit",
    "ssh_hosts",
]
