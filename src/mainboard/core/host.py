import platform

_UNIX_FAMILIES = frozenset({"linux", "osx"})


def platform_family(platform_name: str) -> str:
    """The operating-system half of a pixi platform string (`linux-aarch64` -> `linux`), a bare
    family (`linux`) answering itself.

    The one place the family is read, so an overlay key and a virtual package floor always agree
    about which machines a platform stands for.
    """
    return platform_name.split("-", maxsplit=1)[0]


def platform_selectors(platform_name: str) -> tuple[str, ...]:
    """The `[on.*]` overlay keys covering a pixi platform or family, most specific first.

    `linux-64` is covered by `linux-64`, `linux` and `unix`; a bare family `linux` by itself and
    `unix`.
    """
    family = platform_family(platform_name)
    selectors = (platform_name,) if family == platform_name else (platform_name, family)
    return (*selectors, "unix") if family in _UNIX_FAMILIES else selectors


def pixi_platform(system: str, machine: str) -> str:
    """The pixi platform string (`linux-64` style) for one kernel and machine pair.

    system: the kernel as `uname -s`, `platform.system()` or a probe spells it.
    machine: the architecture as `uname -m` or `PROCESSOR_ARCHITECTURE` spells it.
    """
    arm = machine.lower() in {"arm64", "aarch64"}
    match system.lower():
        case "darwin":
            return "osx-arm64" if arm else "osx-64"
        case "windows":
            return "win-64"
    return "linux-aarch64" if arm else "linux-64"


def current_platform() -> str:
    """This machine as a pixi platform string, the solve surface of a manifest declaring none."""
    return pixi_platform(platform.system(), platform.machine())
