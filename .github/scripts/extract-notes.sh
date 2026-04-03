#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2023 Tiny DFT Development Team <https://github.com/molmod/acid/blob/main/AUTHORS.md>
# SPDX-License-Identifier: GPL-3.0-or-later

# Usage: .github/scripts/extract-notes.sh OWNER/SLUG GITREF

IFS='/'; read -ra REPOSITORY <<<"${1}"
OWNER=${REPOSITORY[0]}
SLUG=${REPOSITORY[1]}
GITREF=${2}

if [[ "${GITREF}" == refs/tags/* ]]; then
    TAG="${GITREF#refs/tags/}"
    VERSION="${TAG#v}"
else
    TAG="unreleased"
    VERSION="Unreleased"
fi
DASHTAG="${TAG//./-}"

# Extract the release notes from the changelog
sed -n "/## \[${VERSION}\]/, /## /{ /##/!p }" docs/source/development/changelog.md > notes.md

# Add a link to the release notes
URL="https://${OWNER}.github.io/${SLUG}/development/changelog.html#${DASHTAG}"
echo "See [changelog#${TAG}](${URL}) for more details." >> notes.md

# Remove leading and trailing empty lines
sed -e :a -e '/./,$!d;/^\n*$/{$d;N;};/\n$/ba' -i notes.md
