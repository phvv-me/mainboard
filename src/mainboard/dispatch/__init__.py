# Remote dispatch: sync a workspace to a host, submit a job, poll and pull results back.

from .dispatcher import Dispatcher, Handle, Verdict
from .onboard import HostSetup, Onboarding, RemoteShell
from .shared import STATE_DIR, logger, now, state_dir
from .sync import GitignoreFilter, SyncLock
from .targets import Facts, resolve, smallest_fit, ssh_hosts
from .transport import DaemonDown, HostUnreachable, SshTransport
from .vocabulary import VERDICTS, JobState, Resources

__all__ = [
    "STATE_DIR",
    "VERDICTS",
    "DaemonDown",
    "Dispatcher",
    "Facts",
    "GitignoreFilter",
    "Handle",
    "HostSetup",
    "HostUnreachable",
    "JobState",
    "Onboarding",
    "RemoteShell",
    "Resources",
    "SshTransport",
    "SyncLock",
    "Verdict",
    "logger",
    "now",
    "resolve",
    "smallest_fit",
    "ssh_hosts",
    "state_dir",
]
