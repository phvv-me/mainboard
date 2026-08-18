import platform

_UNIX_FAMILIES = frozenset({"linux", "osx"})


def platform_selectors(platform_name: str) -> tuple[str, ...]:
    """The `[on.*]` overlay keys covering `platform_name`, most specific first.

    A pixi target names either a concrete platform or a family, so `linux-64`
    is covered by `linux-64`, `linux` and `unix`, while a bare family key like
    `linux` is covered by itself and `unix`.

    platform_name: a pixi platform string such as `linux-aarch64`.
    """
    family = platform_name.split("-", maxsplit=1)[0]
    selectors = (platform_name,) if family == platform_name else (platform_name, family)
    return (*selectors, "unix") if family in _UNIX_FAMILIES else selectors


def current_platform() -> str:
    """This machine as a pixi platform string, `linux-64` style.

    The default solve surface when a manifest declares no platforms, so a
    zero-config workspace installs on the machine it lives on.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = "aarch64" if machine in {"arm64", "aarch64"} else "64"
    if system == "darwin":
        return "osx-arm64" if arch == "aarch64" else "osx-64"
    if system == "windows":
        return "win-64"
    return f"linux-{arch}"
