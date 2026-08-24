from mainboard import dispatch
from mainboard.dispatch.shared import logger, state_dir


def test_the_package_names_its_own_paths_logger_and_public_surface() -> None:
    assert state_dir() == ".mainboard/dispatch"
    assert logger.name == "mainboard.dispatch"
    assert {
        "Dispatcher",
        "Handle",
        "Verdict",
        "GitignoreFilter",
        "SyncLock",
        "Facts",
        "resolve",
        "smallest_fit",
        "ssh_hosts",
        "DaemonDown",
        "HostUnreachable",
        "SshTransport",
    } <= set(dispatch.__all__)
    assert dispatch.Dispatcher is not None
