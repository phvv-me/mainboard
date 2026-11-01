#!/bin/sh
# Prepare a throwaway Linux CI runner for the gate (integration/gates/README.md, "In CI").
# It changes the whole machine, so never run it on a developer's host.
set -eu
probe() { bwrap --ro-bind / / --unshare-net --unshare-pid true; }
{ command -v bwrap && command -v zsh; } >/dev/null || {
    sudo apt-get update
    sudo apt-get install -y bubblewrap zsh
}
probe 2>/dev/null || sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
probe
