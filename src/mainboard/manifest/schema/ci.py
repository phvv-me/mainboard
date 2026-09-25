from ...core.base import Declared


class Ci(Declared):
    """The `[ci]` table: the hosts that stand in for the platforms this center is not.

    `ci --matrix` runs a package's gate here and on each of these at once. A host of this
    machine's own family is passed over, since the local run already covers that platform, so
    the one list serves a macOS center and a Windows one alike.

    hosts: host aliases, each declaring its `platform`, the gate runs on beside this machine.
    """

    hosts: list[str] = []
