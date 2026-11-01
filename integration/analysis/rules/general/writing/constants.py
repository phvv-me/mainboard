"""The functions the local naming rule is tried on, each named by what it tests so that a question
the model was asked is found by its function's name."""

# The name the model double gives every local.
NAMED = "row_count"
# Each function as it is and as the rename leaves it, by the case it states.
RENAMES = {
    "a-plain-local": (
        """\
def plain(rows):
    ab = len(rows)
    return ab
""",
        """\
def plain(rows):
    row_count = len(rows)
    return row_count
""",
    ),
    "a-closure-and-a-lambda-reading-it": (
        """\
def closure(rows):
    ab = len(rows)

    def inner():
        return ab + 1

    return inner() + (lambda: ab)()
""",
        """\
def closure(rows):
    row_count = len(rows)

    def inner():
        return row_count + 1

    return inner() + (lambda: row_count)()
""",
    ),
    "an-augmented-assignment-and-a-keyword-of-the-same-name": (
        """\
def counted(rows):
    ab = 0
    for row in rows:
        ab += row
    return dict(ab=ab)
""",
        """\
def counted(rows):
    row_count = 0
    for row in rows:
        row_count += row
    return dict(ab=row_count)
""",
    ),
    "a-nested-parameter-of-the-same-name": (
        """\
def shadowed(rows):
    ab = len(rows)

    def inner(ab):
        return ab

    return inner(ab)
""",
        """\
def shadowed(rows):
    row_count = len(rows)

    def inner(ab):
        return ab

    return inner(row_count)
""",
    ),
    "an-attribute-and-a-comment-of-the-same-name": (
        """\
def stored(self, rows):
    ab = len(rows)  # the ab
    self.ab = ab
    return self.ab
""",
        """\
def stored(self, rows):
    row_count = len(rows)  # the ab
    self.ab = row_count
    return self.ab
""",
    ),
    "a-decorated-async-function": (
        """\
@register
async def waiting(rows):
    ab = await rows.count()
    return ab
""",
        """\
@register
async def waiting(rows):
    row_count = await rows.count()
    return row_count
""",
    ),
    "a-method": (
        """\
class Holder:
    def total(self, rows):
        ab = sum(rows)
        return ab
""",
        """\
class Holder:
    def total(self, rows):
        row_count = sum(rows)
        return row_count
""",
    ),
}
# Two locals renamed one at a time, as the double names both alike; either may go first.
PAIRED = (
    """\
def paired(rows):
    ab = len(rows)
    cd = ab + 1
    return cd
""",
    (
        """\
def paired(rows):
    row_count = len(rows)
    cd = row_count + 1
    return cd
""",
        """\
def paired(rows):
    ab = len(rows)
    row_count = ab + 1
    return row_count
""",
    ),
)
# The functions whose rename cannot be made: asked nothing, since the rename is tried first.
UNFOLLOWED = {
    "a-parameter-assigned-again": """\
def given(ab):
    ab = ab or 1
    return ab
""",
    "a-name-declared-global": """\
def declared():
    global ab
    ab = 1
    return ab
""",
    "a-scope-reading-names-by-their-spelling": """\
def spelled():
    ab = 1
    return locals()["ab"]
""",
    "an-attribute-and-no-local": """\
def attached(self):
    self.ab = 1
""",
    "a-string-spelling-the-name": """\
def quoted(obj):
    ab = 1
    return getattr(obj, "ab", ab)
""",
}
# The functions that already use the name the model gives, asked and then refused.
CLASHES = {
    "a-name-read-from-the-module": """\
def clashing(rows):
    ab = len(rows)
    return ab + row_count
""",
    "a-name-used-by-a-nested-function": """\
def nesting(rows):
    ab = len(rows)

    def inner():
        return row_count

    return inner() + ab
""",
}
TYPESCRIPT = """\
function typed(rows: number[]): number {
  const ab = rows.length;
  return ab;
}
"""
