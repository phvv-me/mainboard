from patos import FrozenModel

from .session import Session


class Verbs(FrozenModel):
    """A noun's verbs, bound to the one session every verb of a CLI run shares."""

    session: Session
