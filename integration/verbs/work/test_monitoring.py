"""`job monitor status`, `stop` and `start` on a machine whose service managers hold nothing."""

from pathlib import Path


def test_status_says_no_pass_is_installed_and_names_the_install_and_exits_one(
    mb, managers: dict[str, str]
) -> None:
    ran = mb("job", "monitor", "status", env=managers)

    assert (ran.code, "no periodic pass installed" in ran.said) == (1, True)
    assert "run `mb job list --every 20m`" in ran.said


def test_stop_with_no_pass_loaded_says_so_and_asks_the_manager_only_what_is_loaded(
    mb, managers: dict[str, str]
) -> None:
    ran = mb("job", "monitor", "stop", env=managers)
    asked = Path(managers["FAKE_MANAGER_LOG"]).read_text(encoding="utf-8")

    assert (ran.code, "nothing to stop" in ran.said) == (0, True)
    assert asked and "bootout" not in asked and " stop " not in asked


def test_start_with_no_pass_installed_is_refused_naming_the_install(
    mb, managers: dict[str, str]
) -> None:
    ran = mb("job", "monitor", "start", env=managers)

    assert (ran.code, "no periodic pass is installed to start" in ran.said) == (1, True)
    assert "`mb job list --every 20m` installs one" in ran.said
