from typing import NamedTuple

_ARCHITECTURE_BY_MAJOR = {
    6: "Pascal",
    7: "Volta",
    8: "Ampere",
    9: "Hopper",
    10: "Blackwell",
    12: "Blackwell",
}
_ARCHITECTURE_BY_CAPABILITY = {(7, 5): "Turing", (8, 9): "Ada"}


class ComputeCapability(NamedTuple):
    """CUDA compute capability as a (major, minor) pair, so 9.0 > 8.10 compares correctly."""

    major: int
    minor: int

    def __repr__(self) -> str:
        return f"ComputeCapability({self.major}, {self.minor})"

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"

    @property
    def architecture(self) -> str:
        """NVIDIA architecture family, the `cuda.core`-free fallback name; `Unknown` if unmapped.

        Ada (8.9) and Turing (7.5) share a major with Ampere/Volta, so the exact pair wins.
        """
        exact = _ARCHITECTURE_BY_CAPABILITY.get((self.major, self.minor))
        return exact or _ARCHITECTURE_BY_MAJOR.get(self.major, "Unknown")

    @property
    def sm(self) -> str:
        """The dot-free `sm_NN` target `nvcc`/Triton key a build by, e.g. `sm_90`."""
        return f"sm_{self.major}{self.minor}"
