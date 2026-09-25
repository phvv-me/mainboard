# The vendor Protocols are imported only under TYPE_CHECKING, so importing each module here is
# what executes (and covers) their stub definitions.

from importlib import import_module

import pytest


@pytest.mark.parametrize("vendor", ["amd", "apple", "nvidia"])
def test_protocol_modules_import_cleanly(vendor: str) -> None:
    assert import_module(f"mainboard.profile.providers.{vendor}.protocols")
