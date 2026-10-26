"""The constants of `functions` other modules share."""

# One piece of a parameter list at its top level: a character outside brackets, strings and
# comments, or a bracketed group holding one more bracketed group at most.
PARAMETER_LIST_ATOM = (
    r"""(?:[^()\[\]{}"'#]|[(\[{](?:[^()\[\]{}"'#]|[(\[{][^()\[\]{}"'#]*[)\]}])*[)\]}])"""
)
