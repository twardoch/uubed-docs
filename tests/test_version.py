#!/usr/bin/env python3
"""Tests for version management."""

import unittest
import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from version import get_version, validate_semver, get_version_info


class TestVersion(unittest.TestCase):
    """Test version management functions."""

    def test_get_version_format(self):
        """Test that get_version returns a valid format."""
        version = get_version()
        self.assertIsInstance(version, str)
        self.assertGreater(len(version), 0)
        
    def test_validate_semver(self):
        """Test semantic version validation."""
        valid_versions = [
            "1.0.0",
            "1.2.3",
            "1.0.0-alpha",
            "1.0.0-alpha.1",
            "1.0.0+build.1",
            "1.0.0-alpha+build.1",
            "v1.0.0",
        ]
        
        for version in valid_versions:
            with self.subTest(version=version):
                self.assertTrue(validate_semver(version))
                
        invalid_versions = [
            "1.0",
            "1.0.0.0",
            "1.0.0-",
            "1.0.0+",
            "not.a.version",
            "",
        ]
        
        for version in invalid_versions:
            with self.subTest(version=version):
                self.assertFalse(validate_semver(version))
                
    def test_version_info_structure(self):
        """Test that version info has required fields."""
        info = get_version_info()
        
        required_fields = ["version", "commit_hash", "commit_date", "branch"]
        for field in required_fields:
            with self.subTest(field=field):
                self.assertIn(field, info)
                self.assertIsInstance(info[field], str)


if __name__ == "__main__":
    unittest.main()