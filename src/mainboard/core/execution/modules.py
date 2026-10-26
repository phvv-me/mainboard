from importlib.util import find_spec
from pathlib import Path


def submodules(package: str) -> list[str]:
    """The dotted name of every module file under the package named `package`, none imported.

    Namespace portions are included.

    `pkgutil.walk_packages` stops at a directory without an initializer, so a package whose
    subpackages are namespace portions read as empty and its rules never registered (2026-10-24).
    """
    spec = find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(f"no package {package!r}", name=package)
    home = Path(next(iter(spec.submodule_search_locations)))
    names = []
    for path in sorted(home.rglob("*.py")):
        parts = path.relative_to(home).with_suffix("").parts
        kept = parts[:-1] if parts[-1] == "__init__" else parts
        names.append(".".join([package, *kept]))
    return names
