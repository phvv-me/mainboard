"""Where the PY-CUDA rules do not look by default: tests, and code that measures or demonstrates
rather than serves. A project changes either in the rule's `exempt` setting."""

TESTS = (
    r"(^|/)(tests?|testing)/",
    r"(^|/)test_[^/]*\.py$",
    r"(^|/)conftest\.py$",
)
MEASURING = (
    *TESTS,
    r"(^|/)(bench|benchmarks?|experiments?|examples?|papers?|scripts)/",
)
