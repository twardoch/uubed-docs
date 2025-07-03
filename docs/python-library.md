---
layout: default
title: Python Library
nav_order: 6
description: "Complete guide to the UUBED Python library, including installation, usage, and API reference."
---

# UUBED Python Library

**uubed** is a high-performance library for encoding embedding vectors into position-safe strings that solve the "substring pollution" problem in search systems.

## Key Features

* **Position-Safe Encoding**: QuadB64 family prevents false substring matches
* **Blazing Fast**: 40-105x faster than pure Python with Rust acceleration
* **Multiple Encoding Methods**: Full precision, SimHash, Top-k, Z-order
* **Search Engine Friendly**: No more substring pollution in Elasticsearch/Solr
* **Easy Integration**: Simple API, works with any vector database

## Quick Example

```python
import numpy as np
from uubed import encode

# Create a sample embedding
embedding = np.random.rand(384).astype(np.float32)

# Encode to position-safe string
encoded = encode(embedding, method="auto")
print(f"Encoded: {encoded[:50]}...")
```

## Project Structure

The uubed project is organized across multiple repositories:

* [uubed](https://github.com/twardoch/uubed) - Main project hub
* [uubed-rs](https://github.com/twardoch/uubed-rs) - High-performance Rust implementation
* [uubed-py](https://github.com/twardoch/uubed-py) - Python bindings and API
* [uubed-docs](https://github.com/twardoch/uubed-docs) - Comprehensive documentation

## Installation

### Using pip

```bash
pip install uubed
```

For maximum performance, install from source to get the native Rust acceleration:

```bash
git clone https://github.com/twardoch/uubed-py
cd uubed-py
pip install -e .
```

### Using uv (recommended)

```bash
uv pip install uubed
```

## Python API Overview

The Python library provides a simple, high-level interface to all UUBED encoding methods:

### Main Functions

- **`encode(embedding, method="auto", validate=True)`** - Encode an embedding to a position-safe string
- **`decode(encoded_string)`** - Decode back to the original embedding (Eq64 only)

### Encoding Methods

- **`eq64`** - Full precision encoding with complete reversibility
- **`shq64`** - SimHash for locality-sensitive hashing
- **`t8q64`** - Top-k indices for sparse representations
- **`zoq64`** - Z-order encoding for spatial queries

### Performance

With native Rust acceleration:
- Eq64: 234 MB/s encoding speed
- Shq64: 105 MB/s with similarity preservation
- T8q64: 94 MB/s for sparse encoding
- Zoq64: 168 MB/s for spatial encoding

## Next Steps

- [Installation Guide](installation) - Detailed installation instructions
- [Quickstart Tutorial](quickstart) - Get started in 5 minutes
- [API Reference](api) - Complete API documentation
- [Performance Guide](performance) - Optimization tips and benchmarks