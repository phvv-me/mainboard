_HOISTED = """\
def load(path):
    try:
        text = path.read_text()
    except OSError:
        return None
    return parse(text)
"""
_WIDE = """\
def load(path):
    try:
        text = path.read_text()
        return parse(text)
    except OSError:
        return None
"""
_TWO_SPACES = """\
def load(path):
  try:
    text = path.read_text()
    return parse(text)
  except OSError:
    return None
"""
_TWO_SPACES_FIXED = """\
def load(path):
  try:
    text = path.read_text()
  except OSError:
    return None
  return parse(text)
"""
_NOTED = """\
def load(path):
    try:
        text = path.read_text()
        return parse(text)
    # missing
    except FileNotFoundError:
        return None
    # other
    except OSError:
        raise
"""
_NOTED_FIXED = """\
def load(path):
    try:
        text = path.read_text()
    # missing
    except FileNotFoundError:
        return None
    # other
    except OSError:
        raise
    return parse(text)
"""
_LOOP = """\
def load(paths):
    for path in paths:
        try:
            text = path.read_text()
            use(text)
            more(text)
        except OSError:
            continue
"""
_LOOP_FIXED = """\
def load(paths):
    for path in paths:
        try:
            text = path.read_text()
        except OSError:
            continue
        use(text)
        more(text)
"""
_CLOSURE = """\
def load(path):
    try:
        text = path.read_text()

        def inner():
            return text

        return inner
    except OSError:
        return None
"""
_CLOSURE_FIXED = """\
def load(path):
    try:
        text = path.read_text()
    except OSError:
        return None

    def inner():
        return text

    return inner
"""
_METHOD = """\
class Reader:
    def load(self, path):
        try:
            text = path.read_text()
            return parse(text)
        except OSError:
            return None
"""
_METHOD_FIXED = """\
class Reader:
    def load(self, path):
        try:
            text = path.read_text()
        except OSError:
            return None
        return parse(text)
"""
_MODULE = """\
try:
    text = open("x").read()
    parse(text)
except OSError:
    pass
"""
_MODULE_FIXED = """\
try:
    text = open("x").read()
except OSError:
    pass
else:
    parse(text)
"""
_AWAITING = """\
async def load(path):
    try:
        text = await path.read_text()
        return await parse(text)
    except OSError:
        return None
"""
_AWAITING_FIXED = """\
async def load(path):
    try:
        text = await path.read_text()
    except OSError:
        return None
    return await parse(text)
"""
_TWICE = """\
def load(a, b):
    try:
        x = a.read()
        use(x)
    except OSError:
        return None
    try:
        y = b.read()
        use(y)
    except OSError:
        return None
"""
_TWICE_FIXED = """\
def load(a, b):
    try:
        x = a.read()
    except OSError:
        return None
    use(x)
    try:
        y = b.read()
    except OSError:
        return None
    use(y)
"""
_TRAILING = _WIDE.replace("parse(text)", "parse(text)  # parse")
_TABS = _WIDE.replace("    ", "\t")
_STRING = _WIDE.replace("parse(text)", 'parse("""a\nb""")')
_ELSE = _WIDE.replace("return parse(text)", "parse(text)") + "    else:\n        use(text)\n"
_SAME_LINE = """\
def load(path):
    try: text = path.read_text(); parse(text)
    except OSError: return None
"""


def cases() -> dict[str, tuple[str, str]]:
    """The files the PY-EXCE0002 fix reads, by project path, each with the text it must leave.

    A file the fix must refuse is its own expectation.
    """
    return {
        "carriage_returns.py": (_WIDE.replace("\n", "\r\n"), _HOISTED.replace("\n", "\r\n")),
        "two_spaces.py": (_TWO_SPACES, _TWO_SPACES_FIXED),
        "noted_handlers.py": (_NOTED, _NOTED_FIXED),
        "loop.py": (_LOOP, _LOOP_FIXED),
        "closure.py": (_CLOSURE, _CLOSURE_FIXED),
        "method.py": (_METHOD, _METHOD_FIXED),
        "module.py": (_MODULE, _MODULE_FIXED),
        "awaiting.py": (_AWAITING, _AWAITING_FIXED),
        "twice.py": (_TWICE, _TWICE_FIXED),
        "trailing.py": (_TRAILING, _HOISTED.replace("parse(text)", "parse(text)  # parse")),
        "tabs.py": (_TABS, _TABS),
        "multiline.py": (_STRING, _STRING),
        "with_else.py": (_ELSE, _ELSE),
        "same_line.py": (_SAME_LINE, _SAME_LINE),
    }
