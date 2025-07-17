#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Building uubed-docs (documentation site)..."

# Get current version and update mkdocs config
echo "Getting version information..."
CURRENT_VERSION=$(python3 version.py)
echo "Building version: $CURRENT_VERSION"

# Install dependencies
echo "Installing dependencies..."
python3 -m pip install --break-system-packages -r requirements.txt

# Build the MkDocs site
echo "Building MkDocs site..."
python3 -m mkdocs build

echo "uubed-docs build complete. Static site is in the site/ directory."
echo "Version: $CURRENT_VERSION"
