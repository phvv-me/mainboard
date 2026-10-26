"""The constants of `symbols_common` other modules share."""

STORES = """
    AND NOT (p.type IN ('assignment', 'augmented_assignment') AND i.sibling_index = 0)
    AND NOT (p.type IN ('for_statement', 'for_in_clause') AND i.sibling_index = 1)
    AND NOT (p.type = 'named_expression' AND i.sibling_index = 0)
    AND NOT (p.type = 'expression_list' AND q.type = 'delete_statement')
    AND p.type NOT IN ('pattern_list', 'tuple_pattern', 'list_pattern', 'list_splat_pattern',
                       'dictionary_splat_pattern', 'as_pattern_target', 'delete_statement')
"""

BUILTIN_LIST = (
    "string_split('"
    "abs all any bool bytes callable classmethod dict dir enumerate filter float format frozenset "
    "getattr hasattr hash id int isinstance issubclass iter len list map max min next object open "
    "ord print property range repr reversed round set setattr slice sorted staticmethod str sum "
    "super tuple type vars zip Exception ValueError TypeError KeyError RuntimeError "
    "NotImplementedError AttributeError IndexError StopIteration OSError SystemExit "
    "KeyboardInterrupt GeneratorExit BaseException BaseExceptionGroup ExceptionGroup LookupError "
    "ImportError ModuleNotFoundError MemoryError NameError OverflowError RecursionError "
    "ReferenceError StopAsyncIteration SyntaxError SystemError UnicodeError ZeroDivisionError "
    "ArithmeticError AssertionError BufferError EOFError FileNotFoundError FloatingPointError "
    "PermissionError TimeoutError UnboundLocalError Warning bytearray complex compile delattr "
    "divmod eval exec globals hex input locals memoryview oct pow aiter anext ascii bin "
    "breakpoint chr help license NotImplemented Ellipsis None True False', ' ')"
)
