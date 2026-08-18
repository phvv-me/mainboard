from patos import FrozenModel
from pydantic import ConfigDict


def kebab(name: str) -> str:
    """The TOML spelling of a field name, underscores becoming hyphens."""
    return name.replace("_", "-")


class Declared(FrozenModel):
    """A frozen manifest table whose TOML keys are kebab-case.

    Fields stay snake_case in Python and kebab-case in the file
    (`max-walltime`, `mem-gb`), with both spellings accepted on input so
    programmatic construction never needs the alias.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", populate_by_name=True, alias_generator=kebab
    )
