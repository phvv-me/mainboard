"""The kinds C arithmetic gives its operands, by the rules patos types device arithmetic by."""

from .....catalog import Function

_INTEGER = "'^[iu](8|16|32|64)$'"


def _width(kind: str) -> str:
    return f"CAST(substr({kind}, 2) AS INTEGER)"


class CudaPromoted(Function):
    """The kind an operand of C arithmetic takes first: an integer narrower than 32 bits is
    `i32`."""

    kind: str

    expression = "CASE WHEN regexp_matches(kind, '^[iu](8|16)$') THEN 'i32' ELSE kind END"


class CudaMeet(Function):
    """The kind two integer operands meet in under the C rules patos types device arithmetic by:
    two integers of different signedness meet in the unsigned one when it is at least as wide as
    the signed one, else in the signed one, each narrower than 32 bits promoted to `i32` first.
    Null for two of one signedness, which Numba types."""

    lhs: str
    rhs: str

    expression = f"""CASE WHEN regexp_matches(lhs, {_INTEGER}) AND regexp_matches(rhs, {_INTEGER})
        AND substr(cuda_promoted(lhs), 1, 1) <> substr(cuda_promoted(rhs), 1, 1)
      THEN CASE WHEN {_width("CASE WHEN starts_with(cuda_promoted(lhs), 'u') THEN cuda_promoted(lhs) ELSE cuda_promoted(rhs) END")}
                  >= {_width("CASE WHEN starts_with(cuda_promoted(lhs), 'i') THEN cuda_promoted(lhs) ELSE cuda_promoted(rhs) END")}
           THEN CASE WHEN starts_with(cuda_promoted(lhs), 'u') THEN cuda_promoted(lhs)
                     ELSE cuda_promoted(rhs) END
           ELSE CASE WHEN starts_with(cuda_promoted(lhs), 'i') THEN cuda_promoted(lhs)
                     ELSE cuda_promoted(rhs) END END END"""


class CudaCombined(Function):
    """The kind two operands of arithmetic, `min`, `max` or a comparison meet in: `cuda_meet`,
    else for two integers of one signedness the 64 bit kind of that signedness."""

    lhs: str
    rhs: str

    expression = f"""coalesce(cuda_meet(lhs, rhs),
      CASE WHEN regexp_matches(lhs, {_INTEGER}) AND regexp_matches(rhs, {_INTEGER})
                AND substr(lhs, 1, 1) = substr(rhs, 1, 1)
           THEN CASE WHEN starts_with(lhs, 'i') THEN 'i64' ELSE 'u64' END END)"""


class CudaShifted(Function):
    """The kind `lhs` shifted by `rhs` takes: their meeting when it is 64 bits wide, else the
    64 bit kind of the signedness of `lhs`."""

    lhs: str
    rhs: str

    expression = f"""CASE WHEN regexp_matches(coalesce(cuda_meet(lhs, rhs), ''), '64$')
           THEN cuda_meet(lhs, rhs)
           WHEN regexp_matches(lhs, {_INTEGER}) AND regexp_matches(rhs, {_INTEGER})
           THEN CASE WHEN starts_with(lhs, 'i') THEN 'i64' ELSE 'u64' END END"""


class CudaOperated(Function):
    """The kind `lhs op rhs` takes (`op` the operator as written), null where it is not
    told: a division, a power, a matrix product."""

    op: str
    lhs: str
    rhs: str

    expression = """CASE WHEN op IN ('/', '**', '@') THEN NULL
                         WHEN op IN ('<<', '>>') THEN cuda_shifted(lhs, rhs)
                         ELSE cuda_combined(lhs, rhs) END"""
