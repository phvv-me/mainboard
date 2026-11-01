from collections.abc import Sequence

_MODEL = """\
from pydantic import BaseModel, model_validator


class Box(BaseModel):
    a: int
    b: int = 3

"""
_FIELD_MODEL = _MODEL.replace("model_validator", "field_validator")
_CHECK = """\
    @model_validator(mode='after')
    def check(self):
        if self.b > 10:
            raise ValueError('big')
        return self
"""
_CHECK_FIXED = """\
    @field_validator("b")
    @classmethod
    def check(cls, b):
        if b > 10:
            raise ValueError('big')
        return b
"""
_OTHER = """\
    @model_validator(mode='after')
    def other(self):
        if self.a > 10:
            raise ValueError('big')
        return self
"""
_OTHER_FIXED = """\
    @field_validator("a")
    @classmethod
    def other(cls, a):
        if a > 10:
            raise ValueError('big')
        return a
"""
_TYPED = """\
from typing import Self

from pydantic import BaseModel, model_validator


class Box(BaseModel):
    a: int
    b: int | None = None

    @model_validator(mode='after')
    def check(self) -> Self:
        if self.b is not None and self.b > 10:
            raise ValueError('big')
        return self
"""
_TYPED_FIXED = """
from pydantic import BaseModel, field_validator


class Box(BaseModel):
    a: int
    b: int | None = None

    @field_validator("b")
    @classmethod
    def check(cls, b: int | None) -> int | None:
        if b is not None and b > 10:
            raise ValueError('big')
        return b
"""
_QUALIFIED = """\
import pydantic


class Box(pydantic.BaseModel):
    a: int
    b: int = 3

"""
_DECORATED = _MODEL + _CHECK.replace("    def check", "    @some\n    def check")
_NESTED = (
    _MODEL
    + """\
    @model_validator(mode='after')
    def check(self):
        def inner():
            return self.b

        if inner() > 10:
            raise ValueError('big')
        return self
"""
)
_OTHER_RETURN = (
    _MODEL
    + """\
    @model_validator(mode='after')
    def check(self):
        if self.b > 10:
            return other
        return self
"""
)
_ANNOTATED = _MODEL + _CHECK.replace("check(self)", "check(self) -> 'Box'")
_BEFORE = (
    _MODEL
    + """\
    @model_validator(mode='before')
    @classmethod
    def check(cls, data):
        return data
"""
)


def cases() -> dict[str, tuple[str, str]]:
    """The files the PY-PYDA0001 fix reads, by project path, each with the text it must leave.

    A file the fix must refuse is its own expectation.
    """
    return {
        "typed.py": (_TYPED, _TYPED_FIXED),
        "carriage_returns.py": (
            (_MODEL + _CHECK).replace("\n", "\r\n"),
            (_FIELD_MODEL + _CHECK_FIXED).replace("\n", "\r\n"),
        ),
        "two.py": (
            _MODEL + _CHECK + "\n" + _OTHER,
            _FIELD_MODEL + _CHECK_FIXED + "\n" + _OTHER_FIXED,
        ),
        "qualified.py": (
            _QUALIFIED + _CHECK.replace("@model_validator", "@pydantic.model_validator"),
            _QUALIFIED + _CHECK_FIXED.replace("@field_validator", "@pydantic.field_validator"),
        ),
        "decorated.py": (_DECORATED, _DECORATED),
        "nested.py": (_NESTED, _NESTED),
        "other_return.py": (_OTHER_RETURN, _OTHER_RETURN),
        "annotated.py": (_ANNOTATED, _ANNOTATED),
        "before.py": (_BEFORE, _BEFORE),
    }


def validated(fields: Sequence[str]) -> str:
    """A model holding one validator per field, each reading only its field."""
    members = [f"    {name}: int\n" for name in fields]
    validators = [
        f"    @model_validator(mode='after')\n    def check_{name}(self):\n"
        f"        if self.{name} > 10:\n            raise ValueError('big {name}')\n"
        "        return self\n"
        for name in fields
    ]
    return (
        "from pydantic import BaseModel, model_validator\n\n\nclass Box(BaseModel):\n"
        + "".join(members)
        + "\n"
        + "\n".join(validators)
    )


def narrowed(fields: Sequence[str]) -> str:
    """The model `validated` becomes: a field validator per field, reading its argument."""
    members = [f"    {name}: int\n" for name in fields]
    validators = [
        f'    @field_validator("{name}")\n    @classmethod\n'
        f"    def check_{name}(cls, {name}):\n"
        f"        if {name} > 10:\n            raise ValueError('big {name}')\n"
        f"        return {name}\n"
        for name in fields
    ]
    return (
        "from pydantic import BaseModel, field_validator\n\n\nclass Box(BaseModel):\n"
        + "".join(members)
        + "\n"
        + "\n".join(validators)
    )
