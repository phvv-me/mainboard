# The one hook a consumer implements. A declaration carries a word table, coverage axes and the
# read and write halves of every tracked flag, none of which survive an ini string.
#
# No return annotation on purpose: pluggy evaluates a hookspec's annotations when it registers
# the spec, and naming `Declaration` would pull a dataframe engine into every pytest session on a
# machine this tool is installed beside. The caller in `pytest_plugin` annotates a local instead.

import pytest


@pytest.hookspec(firstresult=True)
def pytest_trials_declaration():
    """This workspace's `mainboard.trials.Declaration`.

    Answering None, or not implementing the hook, leaves the plugin completely inert.
    """
