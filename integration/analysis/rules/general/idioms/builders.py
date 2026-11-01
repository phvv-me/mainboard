from collections.abc import Sequence

_NARROW = """\
def label(code):
  if code == 1:
    return 'low'
  elif code == 2:
    return 'mid'
  elif code == 3:
    return 'high'
  return 'none'
"""
_NARROW_MATCHED = """\
def label(code):
  match code:
    case 1:
      return 'low'
    case 2:
      return 'mid'
    case 3:
      return 'high'
  return 'none'
"""
_LOOP = """\
def run(items):
    for code in items:
        if code == 1:
            continue
        elif code == 2:

            use(code)
            break
        elif code == 3:
            # a note
            use(3)
        else:
            raise ValueError(code)
"""
_LOOP_MATCHED = """\
def run(items):
    for code in items:
        match code:
            case 1:
                continue
            case 2:

                use(code)
                break
            case 3:
                # a note
                use(3)
            case _:
                raise ValueError(code)
"""
_BETWEEN = """\
def label(code):
    if code == 1:
        return 1
    # the second
    elif code == 2:
        return 2
    # the third
    elif code == 3:
        return 3
"""
_BETWEEN_MATCHED = """\
def label(code):
    match code:
        case 1:
            return 1
        # the second
        case 2:
            return 2
        # the third
        case 3:
            return 3
"""
_ARMS = """\
def label(code):
    if code == 1:
        return 1
    elif code == 2:
        return 2
    elif code == 3:
        return 3
"""
_ARMS_MATCHED = """\
def label(code):
    match code:
        case 1:
            return 1
        case 2:
            return 2
        case 3:
            return 3
"""
_MIXED = """\
def label(code):
    if code == 'a':
        return 1
    elif code == 'b':
        return 2
    elif code == 3:
        return 3
    elif code == 4.5:
        return 4
"""
_MIXED_MATCHED = """\
def label(code):
    match code:
        case 'a':
            return 1
        case 'b':
            return 2
        case 3:
            return 3
        case 4.5:
            return 4
"""
_NESTED = """\
def label(code, other):
    if code == 1:
        if other == 1:
            return 1
        elif other == 2:
            return 2
        elif other == 3:
            return 3
    elif code == 2:
        return 2
    elif code == 3:
        return 3
"""
_NESTED_MATCHED = """\
def label(code, other):
    match code:
        case 1:
            match other:
                case 1:
                    return 1
                case 2:
                    return 2
                case 3:
                    return 3
        case 2:
            return 2
        case 3:
            return 3
"""
_NOTED = """\
def label(code):
    if code == 1:  # one
        return 1
    elif code == 2:  # two
        return 2
    elif code == 3:
        return 3
"""
_NOTED_MATCHED = """\
def label(code):
    match code:
        case 1:  # one
            return 1
        case 2:  # two
            return 2
        case 3:
            return 3
"""
_PARENTHESIZED = _ARMS.replace("code == 1", "(code == 1)").replace("code == 2", "(code == 2)")
_UNEVEN = _ARMS.replace("        return 2", "      return 2")
_OTHER_SUBJECT = _ARMS.replace("elif code == 2", "elif kind == 2")


def cases() -> dict[str, tuple[str, str]]:
    """The files the ALL-BRAN0001 fix reads, by project path, each with the text it must leave.

    A file the fix must refuse is its own expectation.
    """
    return {
        "two_spaces.py": (_NARROW, _NARROW_MATCHED),
        "loop.py": (_LOOP, _LOOP_MATCHED),
        "between.py": (_BETWEEN, _BETWEEN_MATCHED),
        "carriage_returns.py": (_ARMS.replace("\n", "\r\n"), _ARMS_MATCHED.replace("\n", "\r\n")),
        "mixed.py": (_MIXED, _MIXED_MATCHED),
        "nested.py": (_NESTED, _NESTED_MATCHED),
        "noted.py": (_NOTED, _NOTED_MATCHED),
        "parenthesized.py": (_PARENTHESIZED, _PARENTHESIZED),
        "uneven.py": (_UNEVEN, _UNEVEN),
        "other_subject.py": (_OTHER_SUBJECT, _OTHER_SUBJECT),
    }


def chain(values: Sequence[int], *, default: bool) -> str:
    """A function choosing among `values` by an if chain, with an else when `default`."""
    arms = [
        f"    {'if' if place == 0 else 'elif'} code == {value}:\n        use({value})\n"
        for place, value in enumerate(values)
    ]
    return "def label(code):\n" + "".join(arms) + ("    else:\n        use(None)\n" * default)


def matched(values: Sequence[int], *, default: bool) -> str:
    """The function `chain` becomes."""
    arms = [f"        case {value}:\n            use({value})\n" for value in values]
    return (
        "def label(code):\n    match code:\n"
        + "".join(arms)
        + ("        case _:\n            use(None)\n" * default)
    )
