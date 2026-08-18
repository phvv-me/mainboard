import shlex
from string.templatelib import Interpolation, Template


def sh(template: Template) -> str:
    """A shell line from a t-string, every interpolation quoted on the way in.

    `sh(t"cd {root} && {command}")` renders with each interpolated value
    passed through `shlex.quote`, so a hostile path or argument cannot break
    out of its word. Passing a plain string is a `TypeError` by construction,
    which makes unquoted composition unrepresentable at the call site.
    """
    parts: list[str] = []
    for item in _templated(template):
        if isinstance(item, Interpolation):
            parts.append(shlex.quote(str(item.value)))
        else:
            parts.append(item)
    return "".join(parts)


def script(template: Template) -> str:
    """A shell fragment from a t-string, interpolations landed verbatim.

    The companion for composing trusted, already-quoted fragments (a `sh`
    result, a rendered activation snippet) into a larger line, keeping the
    t-string type discipline while opting out of double quoting.
    """
    parts: list[str] = []
    for item in _templated(template):
        if isinstance(item, Interpolation):
            parts.append(str(item.value))
        else:
            parts.append(item)
    return "".join(parts)


def _templated(template: Template) -> Template:
    if not isinstance(template, Template):
        raise TypeError(
            f"expected a t-string, got {type(template).__name__}; "
            'write t"..." so interpolations stay quotable'
        )
    return template
