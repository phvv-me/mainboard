# A TOML value as tomllib parses it. The PEP 695 alias lets pydantic resolve the recursion, and
# covariant `Sequence`/`Mapping` accept a concrete `list[str]` or `dict[str, str]`.
from collections.abc import Mapping, Sequence

type Toml = str | int | float | bool | Sequence[Toml] | Mapping[str, Toml]
