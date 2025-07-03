---
layout: page
title: QuadB64 Encoding Family
description: Overview of all encoding methods in the QuadB64 family
---

# QuadB64 Encoding Family

The QuadB64 family consists of multiple encoding schemes, each optimized for specific use cases:

## Core Encodings

### [Q64 - Quadrant Base64](q64)
The foundational encoding that uses position-dependent alphabets to ensure substring safety.

### [EQ64 - Extended Quadrant Base64](eq64)
Enhanced version with optimized alphabet rotation for better distribution.

### [ShQ64 - Sharded Quadrant Base64](shq64)
Designed for distributed systems with built-in sharding support.

### [T8Q64 - Top-8 Bits Quadrant Base64](t8q64)
Optimized for sparse high-dimensional vectors, prioritizing the most significant bits.

### [ZoQ64 - Zoned Quadrant Base64](zoq64)
Supports variable-length embeddings with zone-based encoding.

### [MQ64 - Matryoshka Quadrant Base64](mq64)
Implements nested resolution layers for progressive refinement.

## Choosing an Encoding Method

| Method | Best For | Key Feature |
|--------|----------|-------------|
| Q64 | General use | Balanced performance |
| EQ64 | High-throughput systems | Optimized alphabets |
| ShQ64 | Distributed databases | Built-in sharding |
| T8Q64 | Sparse vectors | Bit prioritization |
| ZoQ64 | Variable embeddings | Flexible zones |
| MQ64 | Progressive search | Multi-resolution |

## Common Characteristics

All QuadB64 encodings share these properties:

- **Position-safe**: No false substring matches
- **Search-friendly**: Optimized for text indexing
- **Reversible**: Lossless encoding/decoding
- **Compact**: Efficient space utilization
- **Fast**: SIMD-optimized implementations