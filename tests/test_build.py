#!/usr/bin/env python3
"""Tests for build system."""

import unittest
import subprocess
import tempfile
import shutil
from pathlib import Path


class TestBuild(unittest.TestCase):
    """Test build system functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.repo_root = Path(__file__).parent.parent
        self.original_dir = Path.cwd()
        
    def tearDown(self):
        """Clean up after tests."""
        # Change back to original directory
        if Path.cwd() != self.original_dir:
            subprocess.run(["cd", str(self.original_dir)], shell=True)
            
    def test_mkdocs_build(self):
        """Test that mkdocs build works."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to repo directory
            subprocess.run(["cd", str(self.repo_root)], shell=True)
            
            # Run mkdocs build
            result = subprocess.run(
                ["python3", "-m", "mkdocs", "build", "--site-dir", temp_dir],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            
            self.assertEqual(result.returncode, 0, f"Build failed: {result.stderr}")
            
            # Check that output directory exists and has content
            output_dir = Path(temp_dir)
            self.assertTrue(output_dir.exists())
            self.assertTrue((output_dir / "index.html").exists())
            
    def test_mkdocs_serve_check(self):
        """Test that mkdocs serve command would work (dry run)."""
        result = subprocess.run(
            ["python3", "-m", "mkdocs", "build", "--verbose"],
            cwd=self.repo_root,
            capture_output=True,
            text=True
        )
        
        self.assertEqual(result.returncode, 0, f"Serve check failed: {result.stderr}")
        
    def test_build_script_exists(self):
        """Test that build script exists and is executable."""
        build_script = self.repo_root / "build.sh"
        self.assertTrue(build_script.exists())
        
        # Check if it's executable
        import stat
        mode = build_script.stat().st_mode
        self.assertTrue(mode & stat.S_IEXEC)
        
    def test_build_script_runs(self):
        """Test that build script runs successfully."""
        build_script = self.repo_root / "build.sh"
        
        result = subprocess.run(
            ["bash", str(build_script)],
            cwd=self.repo_root,
            capture_output=True,
            text=True
        )
        
        self.assertEqual(result.returncode, 0, f"Build script failed: {result.stderr}")
        
        # Check that site directory was created
        site_dir = self.repo_root / "site"
        self.assertTrue(site_dir.exists())
        self.assertTrue((site_dir / "index.html").exists())


if __name__ == "__main__":
    unittest.main()