---
layout: page
title: Installation Guide
description: How to install UUBED on your system
---

# Installation Guide

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- pip package manager
- 10 MB of disk space

### Recommended Requirements
- Python 3.10+ for optimal performance
- C++ compiler for building native extensions (optional)
- 64-bit operating system

### Supported Platforms
- **Linux**: Ubuntu 20.04+, Debian 10+, RHEL 8+, and compatible distributions
- **macOS**: 10.15 (Catalina) or later
- **Windows**: Windows 10 or later (64-bit)

## Installation Methods

### Method 1: Install from PyPI (Recommended)

The simplest way to install uubed is using pip:

```bash
pip install uubed
```

To install with all optional dependencies:

```bash
pip install uubed[all]
```

### Method 2: Install from Source

For the latest development version or to contribute:

```bash
# Clone the repository
git clone https://github.com/twardoch/uubed.git
cd uubed

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Method 3: Using Poetry

If you prefer Poetry for dependency management:

```bash
# Clone the repository
git clone https://github.com/twardoch/uubed.git
cd uubed

# Install using Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Method 4: Using Conda

For Conda users:

```bash
# Using conda-forge channel
conda install -c conda-forge uubed

# Or using pip within conda environment
conda create -n uubed-env python=3.10
conda activate uubed-env
pip install uubed
```

## Optional Dependencies

### Performance Extensions

For maximum performance, install the native extensions:

```bash
pip install uubed[native]
```

This requires a C++ compiler:
- **Linux**: `sudo apt-get install build-essential` (Ubuntu/Debian) or `sudo yum groupinstall "Development Tools"` (RHEL/CentOS)
- **macOS**: Install Xcode Command Line Tools: `xcode-select --install`
- **Windows**: Install Visual Studio Build Tools or MinGW-w64

### Machine Learning Integration

For ML/AI integrations:

```bash
pip install uubed[ml]
```

This includes:
- NumPy for array operations
- Support for common embedding formats
- Optimized vectorized operations

### Development Dependencies

For contributing to uubed:

```bash
pip install uubed[dev]
```

Includes:
- pytest for testing
- black for code formatting
- mypy for type checking
- sphinx for documentation

## Verification

### Basic Verification

Verify your installation:

```bash
python -c "import uubed; print(uubed.__version__)"
```

### Run Tests

To ensure everything is working correctly:

```bash
# Install test dependencies
pip install uubed[test]

# Run the test suite
python -m pytest --pyargs uubed
```

### Performance Check

Check if native extensions are available:

```python
import uubed

# Check for native acceleration
if uubed.has_native_extensions():
    print("Native extensions are available!")
else:
    print("Using pure Python implementation")

# Run a benchmark
uubed.benchmark()
```

## Configuration

### Environment Variables

Configure uubed behavior using environment variables:

```bash
# Set default encoding variant
export UUBED_DEFAULT_VARIANT="eq64"

# Enable debug logging
export UUBED_DEBUG="1"

# Set performance mode
export UUBED_PERFORMANCE_MODE="aggressive"
```

### Configuration File

Create a configuration file at `~/.uubed/config.json`:

```json
{
  "default_variant": "eq64",
  "performance": {
    "use_native": true,
    "chunk_size": 8192,
    "parallel_threshold": 1048576
  },
  "logging": {
    "level": "INFO",
    "format": "simple"
  }
}
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'uubed'

**Solution**: Ensure pip installation completed successfully:
```bash
pip install --upgrade pip
pip install uubed --force-reinstall
```

#### Native extensions not building

**Solution**: Install development tools:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev build-essential

# macOS
xcode-select --install

# Windows
# Install Visual Studio Build Tools from Microsoft
```

#### Performance issues

**Solution**: Check if native extensions are loaded:
```python
import uubed
print(uubed.get_implementation_info())
```

### Getting Help

If you encounter issues:

1. Check the [FAQ](https://github.com/twardoch/uubed/wiki/FAQ)
2. Search [existing issues](https://github.com/twardoch/uubed/issues)
3. Join our [Discord community](https://discord.gg/uubed)
4. Open a [new issue](https://github.com/twardoch/uubed/issues/new)

## Next Steps

Now that you have uubed installed:

1. Follow the [Quick Start Guide](quickstart.md) to learn basic usage
2. Read about [QuadB64 Fundamentals](theory/quadb64-fundamentals.md) to understand the theory
3. Explore the [API Reference](api.md) for detailed documentation
4. Check out [Examples](https://github.com/twardoch/uubed/tree/main/examples) for real-world usage

## Upgrading

### Upgrade to Latest Version

```bash
pip install --upgrade uubed
```

### Check for Updates

```python
import uubed
uubed.check_for_updates()
```

### Migration Between Versions

When upgrading between major versions, check the [Migration Guide](reference/migration.md) for breaking changes and update instructions.