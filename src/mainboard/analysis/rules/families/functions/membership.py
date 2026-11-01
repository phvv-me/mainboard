"""The vocabularies mcmr's kernel keeps as Rust constants, each a scalar macro asking whether a
name belongs to one."""

from collections.abc import Sequence
from typing import ClassVar

from ....catalog import Function
from ....engine.facts import Setting

_BINDING_DECORATORS = [
    "abstractmethod",
    "abstractproperty",
    "cache",
    "cached_property",
    "classmethod",
    "deleter",
    "final",
    "getter",
    "lru_cache",
    "override",
    "overload",
    "property",
    "setter",
    "staticmethod",
]
_LIFECYCLE_NAMES = [
    "__init_subclass__",
    "__post_init__",
    "model_post_init",
    "setUp",
    "setUpClass",
    "setup_method",
    "tearDown",
    "tearDownClass",
    "teardown_method",
]
_MODEL_FOUNDATIONS = [
    "BaseModel",
    "Component",
    "FlexModel",
    "FrozenFlexModel",
    "FrozenModel",
    "Model",
    "RootModel",
    "SQLModel",
]
_VALIDATION_EXCEPTIONS = ["PydanticCustomError", "TypeError", "ValidationError", "ValueError"]
_VALIDATOR_DECORATORS = ["field_validator", "model_validator", "root_validator", "validator"]
_PROPERTY_DECORATORS = ["property", "cached_property", "setter", "getter", "deleter"]
_TENSOR_TYPES = ["Array", "DeviceArray", "NDArray", "Tensor", "ndarray"]
_TENSOR_ANNOTATIONS = [
    "BFloat16",
    "Bool",
    "Complex",
    "Complex64",
    "Complex128",
    "Float",
    "Float16",
    "Float32",
    "Float64",
    "Inexact",
    "Int",
    "Int16",
    "Int32",
    "Int64",
    "Int8",
    "Integer",
    "Key",
    "Num",
    "Real",
    "Shaped",
    "UInt8",
]
_TENSOR_LIBRARIES = ["cupy", "jax", "jaxtyping", "numpy", "torch", "torchtyping"]
_BEHAVIOR_TYPES = [
    "binary_operator",
    "boolean_operator",
    "call",
    "comparison_operator",
    "conditional_expression",
    "dictionary_comprehension",
    "generator_expression",
    "list_comprehension",
    "named_expression",
    "not_operator",
    "set_comprehension",
    "unary_operator",
    "union_type",
]
_READING_BUILTINS = [
    "len",
    "iter",
    "reversed",
    "sorted",
    "enumerate",
    "sum",
    "min",
    "max",
    "any",
    "all",
    "next",
]


class _Membership(Function):
    """Whether `name` is one of the vocabulary `members` a subclass names."""

    name: str

    members: ClassVar[Sequence[str]]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Setting) -> None:
        if "members" in cls.__dict__:
            cls.expression = "name IN (" + ", ".join(f"'{value}'" for value in cls.members) + ")"
        super().__pydantic_init_subclass__(**kwargs)


class FunctionsBindingDecorator(_Membership):
    """Whether a decorator says how the language binds a member rather than who calls it."""

    members = _BINDING_DECORATORS


class FunctionsPropertyDecorator(_Membership):
    """Whether a decorator makes a function a property or an accessor of one."""

    members = _PROPERTY_DECORATORS


class FunctionsValidatorDecorator(_Membership):
    """Whether a decorator is one of Pydantic's validators."""

    members = _VALIDATOR_DECORATORS


class FunctionsModelFoundation(_Membership):
    """Whether a class name is a model foundation a method's owner may base."""

    members = _MODEL_FOUNDATIONS


class FunctionsLifecycleName(_Membership):
    """Whether a function name is one a language or a model library calls itself."""

    members = _LIFECYCLE_NAMES


class FunctionsValidationException(_Membership):
    """Whether an exception name belongs to the validation vocabulary."""

    members = _VALIDATION_EXCEPTIONS


class FunctionsBehaviorType(_Membership):
    """Whether a syntax node type is an expression that does something rather than names
    something (a boolean operation counts once per chain, which the caller settles)."""

    members = _BEHAVIOR_TYPES


class FunctionsTensorType(_Membership):
    """Whether a name is an array type some libraries share the spelling of."""

    members = _TENSOR_TYPES


class FunctionsTensorWrapper(_Membership):
    """Whether a name is a jaxtyping wrapper that states a dtype and a shape at once."""

    members = _TENSOR_ANNOTATIONS


class FunctionsTensorLibrary(_Membership):
    """Whether a module name is a library whose array types the tensor vocabulary describes."""

    members = _TENSOR_LIBRARIES


class FunctionsReadingBuiltin(_Membership):
    """Whether a name is a builtin that only reads its argument."""

    members = _READING_BUILTINS
