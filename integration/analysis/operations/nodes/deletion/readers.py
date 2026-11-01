from patos import FrozenModel


class Deletion(FrozenModel):
    """What deleting the nodes starting on some lines of one file made of it, in one plan.

    edits: the kinds of edit each deletion made, none for a refused one.
    refusals: why each deletion was refused, empty text for one that was not.
    taken: whether the plan took each deletion's edits, which a refused deletion has none to give.
    text: the file after the plan.
    """

    edits: list[list[str]]
    refusals: list[str]
    taken: list[bool]
    text: str
