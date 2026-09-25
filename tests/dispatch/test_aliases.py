import os
import sys
from typing import TYPE_CHECKING

from mainboard.dispatch.aliases import SshAliases
from mainboard.dispatch.transport import Endpoint

if TYPE_CHECKING:
    from pathlib import Path

# A config somebody wrote by hand, which no alias block may ever disturb.
_OWN = "Host *\n  ServerAliveInterval 120\n\nHost gold\n  User pedro\n"


def test_an_alias_goes_first_is_replaced_whole_and_leaves_the_rest_as_it_was(
    tmp_path: Path,
) -> None:
    """A block sits above a broad `Host *`, so the rental's address is the one ssh uses."""
    config = tmp_path / ".ssh" / "config"
    aliases = SshAliases(config)
    aliases.add("box", Endpoint(address="1.2.3.4"))
    config.write_text(config.read_text() + _OWN)
    aliases.add("box", Endpoint(address="5.6.7.8", port=2222, user="root", identity="~/k"))
    aliases.add("other", Endpoint(address="9.9.9.9"))
    text = config.read_text()
    assert text.index("Host other") < text.index("Host box") < text.index("Host *")
    assert "1.2.3.4" not in text
    assert f"  IdentityFile {os.path.expanduser('~/k')}\n  IdentitiesOnly yes" in text
    assert f"UserKnownHostsFile {os.devnull}" in text
    if sys.platform != "win32":
        assert config.stat().st_mode & 0o777 == 0o600
    aliases.remove("box")
    aliases.remove("other")
    assert config.read_text() == _OWN


def test_removing_from_a_config_that_does_not_exist_writes_nothing(tmp_path: Path) -> None:
    SshAliases(tmp_path / "config").remove("box")
    assert not (tmp_path / "config").exists()


def test_removing_the_only_alias_leaves_an_empty_config(tmp_path: Path) -> None:
    aliases = SshAliases(tmp_path / "config")
    aliases.add("box", Endpoint(address="1.2.3.4"))
    aliases.remove("box")
    assert aliases.path.read_text() == ""
