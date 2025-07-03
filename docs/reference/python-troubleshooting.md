---
layout: default
title: "Python Library Troubleshooting"
parent: "Reference"
nav_order: 4
description: "Python-specific troubleshooting guide for UUBED library issues"
---

# Python Library Troubleshooting

This guide covers Python-specific issues and solutions when using the UUBED library.

## Installation Issues

### Problem: "No module named 'uubed._native'"

**Symptom**: ImportError when trying to use uubed after installation.

**Solution**:
```bash
# Ensure you have the latest pip
pip install --upgrade pip

# Reinstall uubed with proper wheel support
pip install --force-reinstall uubed
```

### Problem: Build fails on M1/M2 Macs

**Symptom**: Compilation errors during installation on Apple Silicon.

**Solution**:
```bash
# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Set the correct target
rustup target add aarch64-apple-darwin

# Install with maturin
pip install maturin
maturin develop --release
```

### Problem: Using uv instead of pip

**Solution**: If you're using `uv` (recommended):
```bash
# Install with uv
uv pip install uubed

# For development installation
uv pip install -e ".[dev]"
```

## Performance Issues

### Problem: Encoding is slower than expected

**Symptom**: Performance doesn't match benchmarks.

**Possible causes and solutions**:

1. **Native module not loaded**:
   ```python
   import uubed
   print(uubed._has_native)  # Should be True
   ```

2. **Debug build instead of release**:
   ```bash
   # Reinstall with release optimizations
   pip install --force-reinstall uubed
   ```

3. **Input data type issues**:
   ```python
   # Ensure proper data types
   import numpy as np
   embedding = np.array(data, dtype=np.float32)  # Use float32
   ```

## Encoding Issues

### Problem: "Invalid embedding values" error

**Symptom**: ValueError when encoding embeddings.

**Solution**:
```python
# For float embeddings, ensure normalization
import numpy as np

# Convert float32 to uint8
embedding = np.clip(embedding * 255, 0, 255).astype(np.uint8)

# Or use the library's normalization
encoded = encode(embedding, method="eq64", validate=True)
```

### Problem: Decoded embedding doesn't match original

**Symptom**: Round-trip encoding/decoding produces different values.

**Note**: Only Eq64 supports exact round-trip decoding. Other methods are lossy by design.

```python
# Use Eq64 for exact round-trip
encoded = encode(embedding, method="eq64")
decoded = decode(encoded)  # Only works with eq64
```

## Integration Issues

### Problem: Search engine still shows substring matches

**Symptom**: Despite using uubed, substring pollution persists.

**Solutions**:

1. **Ensure proper field configuration**:
   ```json
   // Elasticsearch mapping
   {
     "mappings": {
       "properties": {
         "embedding": {
           "type": "keyword",  // Not "text"
           "index": true
         }
       }
     }
   }
   ```

2. **Use exact match queries**:
   ```python
   # Use term query, not match query
   query = {"term": {"embedding": encoded_string}}
   ```

## Memory Issues

### Problem: High memory usage with large batches

**Symptom**: Memory errors when encoding many embeddings.

**Solution**:
```python
# Process in smaller batches
batch_size = 1000
for i in range(0, len(embeddings), batch_size):
    batch = embeddings[i:i+batch_size]
    encoded_batch = [encode(emb) for emb in batch]
    # Process batch...
```

## Python-Specific Error Messages

### "Embedding must be 1-dimensional"

Ensure your embedding is flattened:
```python
embedding = embedding.flatten()
```

### "Cannot encode NaN values"

Check for and handle NaN values:
```python
if np.isnan(embedding).any():
    embedding = np.nan_to_num(embedding, 0)
```

### "Incompatible alphabet at position X"

This indicates corrupted encoded data. Ensure encoded strings aren't modified.

## Native Module Detection

```python
import uubed

# Check if native acceleration is available
if uubed._has_native:
    print("Native Rust acceleration available")
else:
    print("Using pure Python implementation")
```

## Type Hints and IDE Support

UUBED provides comprehensive type hints. If your IDE isn't recognizing them:

```bash
# Regenerate type stubs
pip install --force-reinstall uubed

# For development
pip install mypy
mypy --install-types
```

## Testing with uvx hatch

When contributing or testing:

```bash
# Run tests with hatch (recommended)
uvx hatch test

# Run specific test
uvx hatch test tests/test_encoding.py::test_eq64

# Run with coverage
uvx hatch test --cov
```

## Common Development Issues

### Problem: Import errors during development

When developing locally:
```bash
# Install in editable mode
pip install -e .

# Or with uv
uv pip install -e .
```

### Problem: FFI errors with PyO3

Ensure compatible Python and Rust versions:
```bash
# Check Python version
python --version  # Should be 3.8+

# Update Rust
rustup update

# Clean and rebuild
cargo clean
maturin develop --release
```

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/twardoch/uubed/issues)
2. Review the [API documentation](../api)
3. Ask in [GitHub Discussions](https://github.com/twardoch/uubed/discussions)
4. File a bug report with:
   - Python version (`python --version`)
   - uubed version (`pip show uubed`)
   - Minimal reproducible example
   - Full error traceback

## Debug Information Script

Use this script to gather debug information:

```python
import sys
import platform
import numpy as np

try:
    import uubed
    uubed_version = uubed.__version__
    has_native = uubed._has_native
except Exception as e:
    uubed_version = f"Error: {e}"
    has_native = False

print("=== Debug Information ===")
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.machine()}")
print(f"NumPy: {np.__version__}")
print(f"UUBED: {uubed_version}")
print(f"Native support: {has_native}")
```