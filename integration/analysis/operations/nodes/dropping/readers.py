from patos import FrozenModel


class Dropped(FrozenModel):
    """What dropping the nodes written as some text on some lines of one file made of it.

    edits: the kinds of edit each drop made, none for a refused one.
    refusals: why each drop was refused, empty text for one that was not.
    taken: whether the plan took each drop's edits, which a refused drop has none to give.
    text: the file after the plan.
    """

    edits: list[list[str]]
    refusals: list[str]
    taken: list[bool]
    text: str
