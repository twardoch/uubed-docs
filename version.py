#!/usr/bin/env python3
"""Version management for uubed-docs."""

import subprocess
import re
import sys
from pathlib import Path


def get_git_version():
    """Get version from git tags."""
    try:
        # Get the latest tag
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            check=True
        )
        latest_tag = result.stdout.strip()
        
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        commit_hash = result.stdout.strip()
        
        # Check if we're on a tag
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # We're on a tag
            return latest_tag
        else:
            # We're not on a tag, add commit hash
            return f"{latest_tag}-dev+{commit_hash}"
            
    except subprocess.CalledProcessError:
        # No tags found or git not available
        return "0.0.0-dev"


def validate_semver(version):
    """Validate semantic version format."""
    pattern = r'^v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$'
    return re.match(pattern, version) is not None


def get_version():
    """Get the current version."""
    version = get_git_version()
    
    # Remove 'v' prefix if present
    if version.startswith('v'):
        version = version[1:]
    
    return version


def get_version_info():
    """Get detailed version information."""
    version = get_version()
    
    try:
        # Get git info
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        commit_hash = result.stdout.strip()
        
        result = subprocess.run(
            ["git", "show", "-s", "--format=%cd", "--date=iso"],
            capture_output=True,
            text=True,
            check=True
        )
        commit_date = result.stdout.strip()
        
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True
        )
        branch = result.stdout.strip()
        
        return {
            "version": version,
            "commit_hash": commit_hash,
            "commit_date": commit_date,
            "branch": branch
        }
    except subprocess.CalledProcessError:
        return {
            "version": version,
            "commit_hash": "unknown",
            "commit_date": "unknown", 
            "branch": "unknown"
        }


def update_mkdocs_version():
    """Update version in mkdocs.yml."""
    mkdocs_file = Path("mkdocs.yml")
    if not mkdocs_file.exists():
        return
    
    version = get_version()
    content = mkdocs_file.read_text()
    
    # Update site_name with version
    content = re.sub(
        r'site_name: (.+)',
        f'site_name: \\1 v{version}',
        content
    )
    
    mkdocs_file.write_text(content)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--info":
            info = get_version_info()
            print(f"Version: {info['version']}")
            print(f"Commit: {info['commit_hash']}")
            print(f"Date: {info['commit_date']}")
            print(f"Branch: {info['branch']}")
        elif sys.argv[1] == "--update-mkdocs":
            update_mkdocs_version()
            print("Updated mkdocs.yml with version")
        else:
            print(get_version())
    else:
        print(get_version())