# PowerShell completion, the one shell cyclopts writes no script for.
#
# The script is static: every command path and the options each takes, read from the CLI tree
# when the script is printed, so a completion costs PowerShell a table lookup rather than a
# Python start. bash, zsh and fish come from cyclopts itself.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from cyclopts import App


def _walk(node: App, path: tuple[str, ...] = ()) -> Iterator[tuple[str, list[str]]]:
    """Each command path with the words that may follow it: subcommands, then options."""
    commands = sorted(
        name
        for name, sub in getattr(node, "_commands", {}).items()
        if not name.startswith("-") and sub.show
    )
    # A group takes no options of its own, and cyclopts cannot list the arguments of one.
    arguments = node.assemble_argument_collection() if node.default_command else ()
    options = sorted(
        name for argument in arguments for name in argument.names if name.startswith("--")
    )
    yield " ".join(path), [*commands, *options]
    for name in commands:
        yield from _walk(node[name], (*path, name))


def powershell(app: App, names: Sequence[str]) -> str:
    """A script registering completion of `names` (the tool's aliases) in PowerShell.

    `mb completion powershell | Out-String | Invoke-Expression` in `$PROFILE` keeps it.
    """
    rows = "\n".join(
        f"    '{path}' = @({', '.join(repr(word) for word in words)})"
        for path, words in _walk(app)
    )
    return f"""Register-ArgumentCompleter -Native -CommandName {",".join(names)} -ScriptBlock {{
  param($word, $ast, $cursor)
  $table = @{{
{rows}
  }}
  $typed = @($ast.CommandElements | Select-Object -Skip 1 | ForEach-Object {{ "$_" }})
  if ($word) {{ $typed = @($typed | Select-Object -SkipLast 1) }}
  $key = (@($typed | Where-Object {{ $_ -notlike '-*' }})) -join ' '
  while ($key -and -not $table.ContainsKey($key)) {{
    $key = (@($key -split ' ' | Select-Object -SkipLast 1)) -join ' '
  }}
  $table[$key] | Where-Object {{ $_ -like "$word*" }} | ForEach-Object {{
    [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
  }}
}}
"""
