import string

from hypothesis import strategies as st

# The vocabulary the tools accept wherever a name is asked for, a package, a host, a task, an
# environment. Short and lowercase, because every one of those names ends up in a command line,
# a table cell or a file name, and a test that reads back what it generated should stay legible
# when Hypothesis prints the falsifying example.
WORDS = st.text(string.ascii_lowercase, min_size=1, max_size=8)

# A relative posix path, the shape a sync rule, a scratch root or a workspace member takes.
PATHS = st.lists(WORDS, min_size=1, max_size=3).map("/".join)

# A version requirement as a manifest spells one, the values the dependency tables really hold.
SPECS = st.sampled_from(["*", ">=1.0", "==2.3.4", ">=3.14,<4", "~=1.2"])

# Free text that survives a round trip through a terminal, a TOML file and a shell word, so a
# rendering property fails on the code's own mistake rather than on an unpaired surrogate.
TEXT = st.text(
    st.characters(codec="utf-8", exclude_categories=("Cs", "Cc")),
    max_size=24,
)
