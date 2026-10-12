#!/usr/bin/env bash

set -euo pipefail

# Ensure the token is removed from this shell even if authentication fails.
trap 'unset GITHUB_TOKEN' EXIT

if ! command -v gh >/dev/null 2>&1; then
    echo "GitHub CLI (gh) not found; installing it with apt-get..."

    if [[ ${EUID} -eq 0 ]]; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y gh
    elif command -v sudo >/dev/null 2>&1; then
        sudo apt-get update
        sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y gh
    else
        echo "Error: installing gh requires root privileges or sudo." >&2
        exit 1
    fi
fi

if [[ -z ${GITHUB_TOKEN:-} ]]; then
    echo "Error: GITHUB_TOKEN is not set or is empty." >&2
    exit 1
fi

echo "Authenticating GitHub CLI with GITHUB_TOKEN..."
echo "$GITHUB_TOKEN" | gh auth login --with-token

echo "Configuring Git to use GitHub CLI credentials..."
gh auth setup-git

echo "Verifying GitHub CLI authentication..."
gh auth status

unset GITHUB_TOKEN
trap - EXIT

echo "GitHub CLI authentication and Git credential setup completed."
