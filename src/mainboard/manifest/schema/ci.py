from ...core.base import Declared


class Ci(Declared):
    """The `[ci]` table: host aliases (each declaring `platform`) `ci --matrix` also runs on.

    A host of this machine's own family is skipped, so one list serves any center.
    """

    hosts: list[str] = []
