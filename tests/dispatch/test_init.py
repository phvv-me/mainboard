from mainboard import dispatch


def test_state_dir_is_under_the_project_out_dir() -> None:
    assert dispatch.STATE_DIR == ".mainboard/dispatch"


def test_logger_is_named_for_the_subsystem() -> None:
    assert dispatch.logger.name == "mainboard.dispatch"


def test_package_reexports_the_public_surface() -> None:
    expected = {
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
    }
    assert expected <= set(dispatch.__all__)
    assert dispatch.Dispatcher is not None
