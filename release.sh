#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting release process for uubed-docs...${NC}"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}Error: Not in a git repository${NC}"
    exit 1
fi

# Check if working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${RED}Error: Working directory is not clean. Please commit or stash changes.${NC}"
    git status --short
    exit 1
fi

# Get current version
CURRENT_VERSION=$(python3 version.py)
echo -e "${YELLOW}Current version: $CURRENT_VERSION${NC}"

# Check if we're on a tag
if git describe --tags --exact-match HEAD > /dev/null 2>&1; then
    echo -e "${GREEN}On tag: $(git describe --tags --exact-match HEAD)${NC}"
    IS_TAG_RELEASE=true
else
    echo -e "${YELLOW}Not on a tag, this is a development build${NC}"
    IS_TAG_RELEASE=false
fi

# Run tests
echo -e "${GREEN}Running tests...${NC}"
./test.sh

# Build documentation
echo -e "${GREEN}Building documentation...${NC}"
./build.sh

# Create release artifacts
echo -e "${GREEN}Creating release artifacts...${NC}"
RELEASE_DIR="release"
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"

# Copy built site
cp -r site "$RELEASE_DIR/site"

# Create version info file
python3 version.py --info > "$RELEASE_DIR/version-info.txt"

# Create archive
ARCHIVE_NAME="uubed-docs-$CURRENT_VERSION"
tar -czf "$RELEASE_DIR/$ARCHIVE_NAME.tar.gz" -C site .
zip -r "$RELEASE_DIR/$ARCHIVE_NAME.zip" site/

# Create checksums
cd "$RELEASE_DIR"
sha256sum "$ARCHIVE_NAME.tar.gz" > "$ARCHIVE_NAME.tar.gz.sha256"
sha256sum "$ARCHIVE_NAME.zip" > "$ARCHIVE_NAME.zip.sha256"
cd ..

# Display release information
echo -e "${GREEN}Release artifacts created:${NC}"
ls -la "$RELEASE_DIR/"

echo -e "${GREEN}Version info:${NC}"
cat "$RELEASE_DIR/version-info.txt"

if [ "$IS_TAG_RELEASE" = true ]; then
    echo -e "${GREEN}✓ Release build completed successfully!${NC}"
    echo -e "${YELLOW}Artifacts are ready in the '$RELEASE_DIR' directory.${NC}"
    echo -e "${YELLOW}You can now upload these to GitHub releases or deploy the site.${NC}"
else
    echo -e "${GREEN}✓ Development build completed successfully!${NC}"
    echo -e "${YELLOW}This is a development build. Create a git tag for official releases.${NC}"
fi

echo -e "${GREEN}Release process completed!${NC}"