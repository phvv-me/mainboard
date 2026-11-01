from itertools import zip_longest


def table(printed: str) -> list[dict[str, str]]:
    """The rows a compact listing printed: a tab-separated header, then one line per row.

    A line that ends in empty columns may have lost its tabs, so a short line reads empty cells.
    """
    header, *lines = printed.strip("\n").splitlines()
    names = header.split("\t")
    return [dict(zip_longest(names, line.split("\t"), fillvalue="")) for line in lines]
