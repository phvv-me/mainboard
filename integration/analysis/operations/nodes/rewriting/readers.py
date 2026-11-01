from patos import FrozenModel


class Rewritten(FrozenModel):
    """What a `rewrite-node` row made of a file of the corpus, in one plan.

    kinds: the kind of each edit made, none for a refusal.
    why: the reason the operation refused, empty when it did not.
    text: the file after the plan, the file as it was for a refusal.
    """

    kinds: list[str]
    why: str
    text: str
