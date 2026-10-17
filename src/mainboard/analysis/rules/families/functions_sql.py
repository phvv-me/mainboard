"""SQL the FunctionFact and ParameterFact derivations share: the vocabularies mcmr's kernel keeps
as Rust constants, and the scalar macros that read a decorator, a base or a docstring."""

from ...catalog import Function

BINDING_DECORATORS = [
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
LIFECYCLE_NAMES = [
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
MODEL_FOUNDATIONS = [
    "BaseModel",
    "Component",
    "FlexModel",
    "FrozenFlexModel",
    "FrozenModel",
    "Model",
    "RootModel",
    "SQLModel",
]
VALIDATION_EXCEPTIONS = ["PydanticCustomError", "TypeError", "ValidationError", "ValueError"]
VALIDATOR_DECORATORS = ["field_validator", "model_validator", "root_validator", "validator"]
PROPERTY_DECORATORS = ["property", "cached_property", "setter", "getter", "deleter"]
TENSOR_TYPES = ["Array", "DeviceArray", "NDArray", "Tensor", "ndarray"]
TENSOR_ANNOTATIONS = [
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
DTYPE_WORDS = [
    "bfloat16",
    "complex128",
    "complex64",
    "dtype",
    "float16",
    "float32",
    "float64",
    "int16",
    "int32",
    "int64",
    "int8",
    "uint8",
]
TENSOR_LIBRARIES = ["cupy", "jax", "jaxtyping", "numpy", "torch", "torchtyping"]
BEHAVIOR_TYPES = [
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
READING_BUILTINS = [
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


def listed(values: list[str]) -> str:
    """A SQL list literal of strings."""
    return "[" + ", ".join(f"'{value}'" for value in values) + "]"


def members(values: list[str]) -> str:
    """The SQL test that `name` is one of `values`."""
    return "name IN (" + ", ".join(f"'{value}'" for value in values) + ")"


class FunctionsDecoratorName(Function):
    """What a decorator is called: its text without arguments or module path."""

    text: str

    expression = "regexp_extract(trim(split_part(text, '(', 1)), '[^.]*$')"


class FunctionsBaseName(Function):
    """What a base class is called: its text without module path, type arguments or call."""

    text: str

    expression = "regexp_extract(rtrim(trim(regexp_extract(text, '^[^\\[(]*')), '.'), '[^.]*$')"


class FunctionsVisibility(Function):
    """The visibility a function name implies in its scope (`module`, `method` or `nested`)."""

    name: str
    scope: str

    expression = (
        "CASE WHEN starts_with(name, '__') AND ends_with(name, '__') THEN 'public' "
        "WHEN starts_with(name, '__') THEN 'private' "
        "WHEN starts_with(name, '_') THEN CASE WHEN scope = 'method' THEN 'protected' "
        "ELSE 'internal' END ELSE 'public' END"
    )


class FunctionsDocstring(Function):
    """A docstring's value as mcmr states it: the summary line trimmed, then the rest dedented
    and stripped of the line breaks around it, a blank line between them."""

    raw: str

    expression = (
        "(WITH norm AS (SELECT replace(replace(raw, chr(13) || chr(10), chr(10)), chr(13), "
        "chr(10)) AS text), "
        "split AS (SELECT split_part(text, chr(10), 1) AS summary, "
        "substr(text, length(split_part(text, chr(10), 1)) + 2) AS rest FROM norm), "
        "lines AS (SELECT summary, rest, string_split(rest, chr(10)) AS items FROM split), "
        "margin AS (SELECT summary, items, coalesce(list_min(list_transform(list_filter(items, "
        "lambda x: trim(x, ' \t\r') <> ''), "
        "lambda x: length(x) - length(ltrim(x, ' \t')))), 0) AS width "
        "FROM lines), body AS (SELECT trim(summary) AS summary, trim(array_to_string("
        "list_transform(items, lambda x: CASE WHEN trim(x, ' \t\r') = '' THEN '' "
        "ELSE substr(x, width + 1) END), chr(10)), chr(10)) AS rest FROM margin) "
        "SELECT CASE WHEN rest = '' THEN summary ELSE summary || chr(10) || chr(10) || rest END "
        "FROM body)"
    )


class FunctionsUnescaped(Function):
    """A non-raw string literal's value: the escapes for a backslash, a quote, a line break, a tab
    and a carriage return resolved (a backslash before a line break joins the lines)."""

    text: str

    expression = (
        "replace(replace(replace(replace(replace(replace(replace(replace(text, "
        "chr(92) || chr(92), chr(1)), chr(92) || chr(10), ''), chr(92) || 'n', chr(10)), "
        "chr(92) || 't', chr(9)), chr(92) || 'r', chr(13)), chr(92) || chr(39), chr(39)), "
        "chr(92) || chr(34), chr(34)), chr(1), chr(92))"
    )


class FunctionsBindingDecorator(Function):
    """Whether a decorator says how the language binds a member rather than who calls it."""

    name: str

    expression = members(BINDING_DECORATORS)


class FunctionsPropertyDecorator(Function):
    """Whether a decorator makes a function a property or an accessor of one."""

    name: str

    expression = members(PROPERTY_DECORATORS)


class FunctionsValidatorDecorator(Function):
    """Whether a decorator is one of Pydantic's validators."""

    name: str

    expression = members(VALIDATOR_DECORATORS)


class FunctionsModelFoundation(Function):
    """Whether a class name is a model foundation a method's owner may base."""

    name: str

    expression = members(MODEL_FOUNDATIONS)


class FunctionsLifecycleName(Function):
    """Whether a function name is one a language or a model library calls itself."""

    name: str

    expression = members(LIFECYCLE_NAMES)


class FunctionsValidationException(Function):
    """Whether an exception name belongs to the validation vocabulary."""

    name: str

    expression = members(VALIDATION_EXCEPTIONS)


class FunctionsBehaviorType(Function):
    """Whether a syntax node type is an expression that does something rather than names
    something (a boolean operation counts once per chain, which the caller settles)."""

    name: str

    expression = members(BEHAVIOR_TYPES)


class FunctionsTensorType(Function):
    """Whether a name is an array type some libraries share the spelling of."""

    name: str

    expression = members(TENSOR_TYPES)


class FunctionsTensorWrapper(Function):
    """Whether a name is a jaxtyping wrapper that states a dtype and a shape at once."""

    name: str

    expression = members(TENSOR_ANNOTATIONS)


class FunctionsTensorLibrary(Function):
    """Whether a module name is a library whose array types the tensor vocabulary describes."""

    name: str

    expression = members(TENSOR_LIBRARIES)


class FunctionsReadingBuiltin(Function):
    """Whether a name is a builtin that only reads its argument."""

    name: str

    expression = members(READING_BUILTINS)


class FunctionsMentionsDtype(Function):
    """Whether a lower-cased text names a dtype."""

    text: str

    expression = (
        "list_bool_or(list_transform(" + listed(DTYPE_WORDS) + ", lambda w: contains(text, w)))"
    )


class FunctionsDeviceRole(Function):
    """What a CUDA JIT compiles a function as (`kernel`, `device`, empty for host code), from the
    absolute name its decorator stands for and the decorator's text."""

    target: str
    decorator: str

    expression = (
        "CASE WHEN regexp_extract(target, '[^.]*$') = 'jit' "
        "AND regexp_extract(target, '^(.*)\\.[^.]*$', 1) "
        "IN ('cuda', 'numba.cuda', 'patos.cuda.typed.cuda') "
        "THEN CASE WHEN contains(regexp_replace(decorator, '\\s', '', 'g'), 'device=True') "
        "THEN 'device' ELSE 'kernel' END "
        "WHEN regexp_extract(target, '^(.*)\\.[^.]*$', 1) = 'patos.cuda.typed' "
        "OR starts_with(regexp_extract(target, '^(.*)\\.[^.]*$', 1), 'patos.cuda.typed.') "
        "THEN CASE regexp_extract(target, '[^.]*$') WHEN 'kernel' THEN 'kernel' "
        "WHEN 'device' THEN 'device' WHEN 'ptx' THEN 'device' ELSE '' END ELSE '' END"
    )


class FunctionsStorePosition(Function):
    """Whether an identifier sits where its name is bound or merely written, not read: given the
    type of its parent, its place among the parent's children, the place of the `in` of an
    enclosing `for` and the type of its grandparent."""

    parent_type: str
    sibling: int
    later: int
    grand_type: str

    expression = (
        "(parent_type IN ('function_definition', 'class_definition', 'parameters', "
        "'lambda_parameters', 'pattern_list', 'tuple_pattern', 'list_pattern', "
        "'list_splat_pattern', 'dictionary_splat_pattern', 'as_pattern_target', "
        "'global_statement', 'nonlocal_statement', 'dotted_name', 'aliased_import', "
        "'splat_type') "
        "OR (parent_type IN ('typed_parameter', 'default_parameter', 'typed_default_parameter', "
        "'keyword_argument', 'assignment', 'augmented_assignment', 'keyword_pattern', "
        "'named_expression') AND sibling = 0) "
        "OR (parent_type = 'attribute' AND sibling > 0) "
        "OR (parent_type IN ('tuple', 'list') AND grand_type = 'as_pattern_target') "
        "OR (parent_type IN ('for_statement', 'for_in_clause') AND sibling < later))"
    )


class FunctionsRootName(Function):
    """The name a dotted chain starts from, given the type and the name of its node: a name is
    itself, an attribute of plain names its first name, anything else has none here."""

    kind: str
    name: str

    expression = (
        "CASE WHEN kind = 'identifier' THEN name "
        "WHEN kind = 'attribute' AND NOT regexp_matches(name, '[()\\[\\]]') "
        "THEN split_part(name, '.', 1) END"
    )


class FunctionsOperand(Function):
    """The place of a node's first operand among its children: after the keyword of a starred
    or awaited expression, first otherwise."""

    kind: str

    expression = "CASE WHEN kind IN ('list_splat', 'await') THEN 1 ELSE 0 END"


class FunctionsCut(Function):
    """Bytes `start` to `stop` of a line (UTF-8 columns); `plain` says the line is ASCII, where
    a character is a byte and `substr` is cheap."""

    text: str
    plain: bool
    start: int
    stop: int

    expression = (
        "CASE WHEN plain THEN substr(text, start + 1, stop - start) "
        "ELSE decode(array_slice(encode(text), start + 1, stop)) END"
    )
