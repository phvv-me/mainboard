# The one hook a consumer implements, and the whole of what the plugin asks of it.
#
# A declaration cannot be an ini setting: it carries a word table, a coverage axis list and the
# read and write halves of every tracked flag, none of which survive a string. So it is a hook,
# which is pytest's own answer for exactly this, and a consumer's conftest reduces to implementing
# it.
#
# The signature carries no return annotation on purpose. pluggy evaluates a hookspec's annotations
# when it registers the spec, so naming the declaration type here would import it, and importing
# it pulls a dataframe engine into every pytest session on a machine this tool is installed
# beside. The caller in `pytest_plugin` annotates what it received instead, where the annotation
# is a local and is never evaluated at all.

import pytest


@pytest.hookspec(firstresult=True)
def pytest_trials_declaration():
    """What this workspace's trials are: its universe, its settle words and its tracked flags.

    Returns one `mainboard.trials.Declaration`. Answering None, or not implementing this at all,
    leaves the plugin completely inert, which is what every session with no trials in it gets.
    """
