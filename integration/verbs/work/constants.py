# The service managers' programs as a machine with nothing installed answers them, each noting
# the command it was given in the file named by FAKE_MANAGER_LOG. launchd finds no service (113)
# and systemd reports every unit inactive.
MANAGERS = {
    "launchctl": """#!/bin/sh
echo "launchctl $*" >> "$FAKE_MANAGER_LOG"
case "$2" in
  */*/*) echo "Could not find service" >&2; exit 113 ;;
esac
""",
    "systemctl": """#!/bin/sh
echo "systemctl $*" >> "$FAKE_MANAGER_LOG"
echo "ActiveState=inactive"
""",
    "loginctl": """#!/bin/sh
echo "loginctl $*" >> "$FAKE_MANAGER_LOG"
echo "Linger=yes"
""",
}
