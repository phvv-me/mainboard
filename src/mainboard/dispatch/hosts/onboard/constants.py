from ....core.project import Project

# The tool's distribution and the script every release of it installs, so nothing below spells
# the name of the binary it installs, and a host still on an older release answers to it.
TOOL = Project().package

# Where the tool's own source sits inside a synced workspace that vendors it.
SOURCE = f"packages/{TOOL}"
