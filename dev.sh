#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_help() {
    echo -e "${GREEN}uubed-docs Development Script${NC}"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  serve     Start development server"
    echo "  test      Run test suite"
    echo "  build     Build documentation"
    echo "  release   Create release"
    echo "  clean     Clean build artifacts"
    echo "  version   Show version information"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 serve     # Start development server"
    echo "  $0 test      # Run all tests"
    echo "  $0 build     # Build documentation"
    echo "  $0 release   # Create release artifacts"
}

install_deps() {
    echo -e "${BLUE}Installing dependencies...${NC}"
    python3 -m pip install --break-system-packages -r requirements.txt
    python3 -m pip install --break-system-packages pytest pyyaml
}

serve() {
    echo -e "${GREEN}Starting development server...${NC}"
    install_deps
    python3 -m mkdocs serve
}

test() {
    echo -e "${GREEN}Running test suite...${NC}"
    ./test.sh
}

build() {
    echo -e "${GREEN}Building documentation...${NC}"
    ./build.sh
}

release() {
    echo -e "${GREEN}Creating release...${NC}"
    ./release.sh
}

clean() {
    echo -e "${GREEN}Cleaning build artifacts...${NC}"
    rm -rf site/
    rm -rf release/
    rm -rf .pytest_cache/
    rm -rf __pycache__/
    find . -name "*.pyc" -delete
    find . -name "*.pyo" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}Clean completed!${NC}"
}

version() {
    echo -e "${GREEN}Version information:${NC}"
    python3 version.py --info
}

# Main script logic
case "${1:-help}" in
    serve)
        serve
        ;;
    test)
        test
        ;;
    build)
        build
        ;;
    release)
        release
        ;;
    clean)
        clean
        ;;
    version)
        version
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac