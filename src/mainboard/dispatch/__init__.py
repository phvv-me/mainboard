# Remote dispatch: sync a workspace to a host, submit a job, poll and pull results back.

from .dispatcher import Dispatcher, Handle, Order, Verdict
from .hosts.landing import renter
from .hosts.onboard import HostSetup, Onboarding
from .jobs.arch import arch, sm
from .jobs.commandline import joined, vetted
from .jobs.evidence import printed
from .jobs.vocabulary import Allocation, JobState, Request, Resources
from .mirroring.shipment import Shipment
from .mirroring.snapshots import Snapshots
from .mirroring.spool import Beat, Spool
from .mirroring.sync import GitignoreFilter, SyncLock
from .provenance import Source, SourceTree
from .shared import announce, now
from .transport.core import DaemonDown, HostUnreachable, SshTransport
from .transport.shells import Dialect, open_shell
from .transport.targets import Facts, resolve, rooted, ssh_hosts
from .wrapping import connection

__all__ = [
    "Allocation",
    "Beat",
    "DaemonDown",
    "Dialect",
    "Dispatcher",
    "Facts",
    "GitignoreFilter",
    "Handle",
    "HostSetup",
    "HostUnreachable",
    "JobState",
    "Onboarding",
    "Order",
    "Request",
    "Resources",
    "Shipment",
    "Snapshots",
    "Source",
    "SourceTree",
    "Spool",
    "SshTransport",
    "SyncLock",
    "Verdict",
    "announce",
    "arch",
    "connection",
    "joined",
    "now",
    "open_shell",
    "printed",
    "renter",
    "resolve",
    "rooted",
    "sm",
    "ssh_hosts",
    "vetted",
]
