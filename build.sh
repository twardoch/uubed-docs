#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Building uubed-docs (documentation site)..."

# Build the MkDocs site
mkdocs build

echo "uubed-docs build complete. Static site is in the site/ directory."
