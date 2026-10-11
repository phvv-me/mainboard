#!/bin/sh
# usage: linux-runner.sh
# Prepare a throwaway Linux CI runner for the gate: bubblewrap, which the sandbox is made of, and
# zsh, a login shell test_managed_python drives (macOS ships it, an Ubuntu runner does not, and a
# skipped shell is an unexpected skip). Where AppArmor restricts unprivileged user namespaces
# (Ubuntu 24.04's default) the restriction is lifted too. That changes the whole machine, so only
# a throwaway runner runs this, never a developer's host.
set -eu
probe() { bwrap --ro-bind / / --unshare-net --unshare-pid true; }
{ command -v bwrap && command -v zsh; } >/dev/null || {
    sudo apt-get update
    sudo apt-get install -y bubblewrap zsh
}
probe 2>/dev/null || sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
probe
