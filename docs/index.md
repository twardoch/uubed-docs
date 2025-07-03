---
layout: default
title: Home
nav_order: 1
description: "Position-Safe Encoding for Substring-Based Search Systems - A comprehensive guide to the QuadB64 encoding family and the uubed library."
permalink: /
---

> Welcome to the ultimate guide for QuadB64, the encoding that fixes search engines! If you're tired of irrelevant results because of messy data, this is your new best friend. It's like giving your data a unique, unforgeable ID card that also tells you exactly where it belongs.

# The QuadB64 Codex

Imagine you're a librarian, but instead of books, you're organizing billions of tiny, digital snippets of information. Traditional methods are like throwing all the snippets into a giant pile, making it impossible to find anything specific without accidentally grabbing a bunch of unrelated junk. The QuadB64 Codex is your revolutionary new system, giving every snippet a precise, unchangeable address that prevents any mix-ups.

Imagine you're a detective, and you're trying to find a specific piece of evidence in a massive, chaotic crime scene. Old methods leave behind a lot of false leads and confusing trails. The QuadB64 Codex is your advanced forensic toolkit, allowing you to pinpoint exact data locations without any misleading clues, ensuring you only find what's truly relevant.

## Position-Safe Encoding for Substring-Based Search Systems

Welcome to **The QuadB64 Codex**, the comprehensive guide to the QuadB64 encoding family and the `uubed` library. This documentation covers everything from theoretical foundations to practical implementations of position-safe encoding schemes designed for modern search systems.

## What is QuadB64?

QuadB64 is a revolutionary family of encoding schemes that solve the **substring pollution problem** inherent in traditional Base64 encoding. When Base64-encoded data is indexed by search engines or vector databases, arbitrary substrings can match across unrelated documents, leading to false positives and degraded search quality.

The QuadB64 family provides **position-safe** encodings that preserve locality and prevent spurious matches, making them ideal for:

- 🔍 **Search Engines**: Prevent false matches in substring-based search
- 🗄️ **Vector Databases**: Maintain locality in embedded representations
- 📊 **Data Analysis**: Preserve meaningful patterns in encoded data
- 🔐 **Security Applications**: Reduce information leakage through encoding

## Key Features

### 🎯 Position-Safe Encoding
Every position in a QuadB64-encoded string carries positional information, preventing arbitrary substring matches.

### 🧩 Multiple Encoding Schemes
Choose from various encoding strategies optimized for different use cases:
- **Eq64**: Full embeddings with position markers
- **Shq64**: SimHash-based compact representations
- **T8q64**: Top-k index encoding for dimensionality reduction
- **Zoq64**: Z-order curve encoding for spatial locality

### ⚡ High Performance
Native implementations with SIMD optimizations ensure encoding/decoding speeds comparable to standard Base64.

### 🔧 Easy Integration
Simple Python API with drop-in replacements for standard Base64 operations.

## Quick Example

```python
from uubed import encode_eq64, decode_eq64

# Encode binary data with position safety
data = b"Hello, QuadB64!"
encoded = encode_eq64(data)
print(f"Encoded: {encoded}")

# Decode back to original
decoded = decode_eq64(encoded)
assert decoded == data
```

## Getting Started

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **[Quick Start](quickstart.md)**

    ---

    Get up and running with QuadB64 in minutes

    [:octicons-arrow-right-24: Quick Start Guide](quickstart.md)

-   :material-book-open:{ .lg .middle } **[Theory](theory/introduction.md)**

    ---

    Understand the mathematical foundations

    [:octicons-arrow-right-24: Read the Theory](theory/introduction.md)

-   :material-api:{ .lg .middle } **[API Reference](api.md)**

    ---

    Complete API documentation

    [:octicons-arrow-right-24: Browse API](api.md)

-   :material-chart-line:{ .lg .middle } **[Benchmarks](reference/benchmarks.md)**

    ---

    Performance comparisons and analysis

    [:octicons-arrow-right-24: View Benchmarks](reference/benchmarks.md)

</div>

## Why QuadB64?

Traditional Base64 encoding was designed for email attachments, not modern search systems. When search engines index Base64-encoded content, they treat it as regular text, leading to:

- **False Positives**: Random substrings match across unrelated documents
- **Poor Relevance**: Search results contaminated with irrelevant matches
- **Wasted Resources**: Indexing and searching meaningless character sequences

QuadB64 solves these problems by making every encoded position unique and meaningful, dramatically improving search quality while maintaining encoding efficiency.

## Project Status

The `uubed` library is under active development. Current features include:

- ✅ Core encoding/decoding algorithms
- ✅ Python bindings with type hints
- ✅ Comprehensive test suite
- 🚧 Native performance optimizations
- 🚧 Additional encoding schemes
- 📅 Streaming API support

## Contributing

We welcome contributions! Please see our [Contributing Guide](contributing/guidelines.md) for details on how to get involved.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/twardoch/uubed/blob/main/LICENSE) file for details.