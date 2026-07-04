from typing import NamedTuple

ARCHITECTURE_BY_MAJOR = {
    6: "Pascal",
    7: "Volta",
    8: "Ampere",
    9: "Hopper",
    10: "Blackwell",
    12: "Blackwell",
}
ARCHITECTURE_BY_CAPABILITY = {
    (7, 5): "Turing",
    (8, 9): "Ada",
}


class ComputeCapability(NamedTuple):
    """CUDA compute capability as a comparable (major, minor) pair.

    Comparison operators work correctly across two-digit minor versions:
    ``ComputeCapability(9, 0) > ComputeCapability(8, 10)`` is True.
    """

    major: int
    minor: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"

    def __repr__(self) -> str:
        return f"ComputeCapability({self.major}, {self.minor})"

    @property
    def sm(self) -> str:
        """The `sm_NN` target string for this capability, e.g. `sm_90`.

        The stable, dot-free identifier `nvcc`/Triton use to key a build per
        architecture — and the natural key for a per-arch config table.
        """
        return f"sm_{self.major}{self.minor}"

    @property
    def architecture(self) -> str:
        """Human-readable NVIDIA architecture family for this capability.

        Ada (8.9) and Turing (7.5) share a major with Ampere/Volta, so the
        exact pair is checked first and the rest maps by major as a
        `cuda.core`-free fallback. Returns `Unknown` for unmapped majors.
        """
        exact = ARCHITECTURE_BY_CAPABILITY.get((self.major, self.minor))
        return exact or ARCHITECTURE_BY_MAJOR.get(self.major, "Unknown")
