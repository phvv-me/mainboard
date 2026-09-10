import platform

_UNIX_FAMILIES = frozenset({"linux", "osx"})


def platform_family(platform_name: str) -> str:
    """The operating-system half of a pixi platform string: `linux-aarch64` -> `linux`.

    The one place the family is read out of a platform name, so an overlay key and a virtual
    package floor always agree about which machines a platform stands for.

    platform_name: a pixi platform string such as `osx-arm64`, or a bare family such as `linux`.
    """
    return platform_name.split("-", maxsplit=1)[0]


def platform_selectors(platform_name: str) -> tuple[str, ...]:
    """The `[on.*]` overlay keys covering `platform_name`, most specific first.

    A pixi target names either a concrete platform or a family, so `linux-64`
    is covered by `linux-64`, `linux` and `unix`, while a bare family key like
    `linux` is covered by itself and `unix`.

    platform_name: a pixi platform string such as `linux-aarch64`.
    """
    family = platform_family(platform_name)
    selectors = (platform_name,) if family == platform_name else (platform_name, family)
    return (*selectors, "unix") if family in _UNIX_FAMILIES else selectors


def pixi_platform(system: str, machine: str) -> str:
    """The pixi platform string for one kernel and machine pair, `linux-64` style.

    system: the kernel as `uname -s`, `platform.system()` or a probe spells it.
    machine: the architecture as `uname -m` or `PROCESSOR_ARCHITECTURE` spells it.
    """
    arch = "aarch64" if machine.lower() in {"arm64", "aarch64"} else "64"
    if system.lower() == "darwin":
        return "osx-arm64" if arch == "aarch64" else "osx-64"
    if system.lower() == "windows":
        return "win-64"
    return f"linux-{arch}"


def current_platform() -> str:
    """This machine as a pixi platform string.

    The default solve surface when a manifest declares no platforms, so a
    zero-config workspace installs on the machine it lives on.
    """
    return pixi_platform(platform.system(), platform.machine())
