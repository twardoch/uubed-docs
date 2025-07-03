#!/usr/bin/env python3
"""Add Jekyll front matter to all markdown files."""

import os
import re
from pathlib import Path

def add_front_matter(filepath):
    """Add Jekyll front matter to a markdown file if it doesn't have it."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if file already has front matter
    if content.startswith('---\n'):
        return False
    
    # Extract title from first H1
    title_match = re.match(r'^#\s+(.+)$', content, re.MULTILINE)
    title = title_match.group(1) if title_match else filepath.stem.replace('-', ' ').title()
    
    # Create front matter
    front_matter = f"""---
layout: page
title: {title}
---

"""
    
    # Write back with front matter
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(front_matter + content)
    
    return True

def main():
    """Process all markdown files in docs directory."""
    docs_dir = Path(__file__).parent.parent
    processed = 0
    
    for filepath in docs_dir.rglob('*.md'):
        # Skip files in _site, _layouts, _includes
        if any(part.startswith('_') for part in filepath.parts):
            continue
        
        # Skip index.md (already processed)
        if filepath.name == 'index.md':
            continue
            
        if add_front_matter(filepath):
            processed += 1
            print(f"Added front matter to: {filepath.relative_to(docs_dir)}")
    
    print(f"\nProcessed {processed} files")

if __name__ == '__main__':
    main()