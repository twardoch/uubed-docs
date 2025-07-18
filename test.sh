#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Running uubed-docs test suite..."

# Install test dependencies
echo "Installing test dependencies..."
python3 -m pip install --break-system-packages -r requirements.txt
python3 -m pip install --break-system-packages pytest pyyaml

# Run version tests
echo "Testing version management..."
python3 -m pytest tests/test_version.py -v

# Run documentation tests
echo "Testing documentation structure..."
python3 -m pytest tests/test_docs.py -v

# Run build tests
echo "Testing build system..."
python3 -m pytest tests/test_build.py -v

# Run mkdocs build test
echo "Testing mkdocs build..."
python3 -m mkdocs build --verbose

# Validate generated site
echo "Validating generated site..."
if [ -f "site/index.html" ]; then
    echo "✓ Site generated successfully"
else
    echo "✗ Site generation failed"
    exit 1
fi

# Check for broken links in built site (basic check)
echo "Checking for basic HTML validity..."
if command -v python3 > /dev/null; then
    python3 -c "
import html.parser
import sys
from pathlib import Path

class SimpleHTMLParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.errors = []
    
    def error(self, message):
        self.errors.append(message)

site_dir = Path('site')
if site_dir.exists():
    for html_file in site_dir.rglob('*.html'):
        try:
            content = html_file.read_text(encoding='utf-8')
            parser = SimpleHTMLParser()
            parser.feed(content)
            if parser.errors:
                print(f'HTML errors in {html_file}: {parser.errors}')
                sys.exit(1)
        except Exception as e:
            print(f'Error parsing {html_file}: {e}')
            sys.exit(1)
    print('✓ HTML validation passed')
else:
    print('✗ Site directory not found')
    sys.exit(1)
"
fi

echo "All tests passed! ✓"