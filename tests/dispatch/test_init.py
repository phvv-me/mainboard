from mainboard import dispatch


def test_the_package_names_its_own_paths_logger_and_public_surface() -> None:
    assert dispatch.STATE_DIR == ".mainboard/dispatch"
    assert dispatch.logger.name == "mainboard.dispatch"
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
        "VERDICTS",
    } <= set(dispatch.__all__)
    assert dispatch.Dispatcher is not None
