# Quick Start Guide

## Installation

```bash
pip install uubed
```

Or install from source with native acceleration:

```bash
git clone https://github.com/twardoch/uubed.git
cd uubed
pip install maturin
maturin develop --release
```

## Basic Usage

```python
import numpy as np
from uubed import encode, decode

# Create a random embedding
embedding = np.random.randint(0, 256, 256, dtype=np.uint8)

# Full precision encoding
full_code = encode(embedding, method="eq64")
print(f"Full: {full_code[:50]}...")  # AQgxASgz...

# Compact similarity hash (16 characters)
compact_code = encode(embedding, method="shq64")
print(f"Compact: {compact_code}")  # QRsTUvWxYZabcdef

# Decode back (only works for eq64)
decoded = decode(full_code)
assert np.array_equal(embedding, np.frombuffer(decoded, dtype=np.uint8))
```

## Why QuadB64?

Regular Base64 in search engines causes **substring pollution**:
- Query: "find similar to 'abc'"
- Problem: "abc" matches everywhere!
- Result: False positives pollute results

QuadB64 solution:
- Position-dependent alphabets
- "abc" can only match at specific positions
- Result: Clean, accurate similarity search

## Encoding Methods

### Eq64 - Full Embeddings
- **Use case**: Need full precision
- **Size**: 2n characters (n = embedding bytes)
- **Features**: Lossless encoding/decoding

### Shq64 - SimHash
- **Use case**: Fast similarity comparison
- **Size**: 16 characters (64-bit hash)
- **Features**: Preserves cosine similarity

### T8q64 - Top-k Indices
- **Use case**: Sparse representation
- **Size**: 16 characters (8 indices)
- **Features**: Captures most important features

### Zoq64 - Z-order
- **Use case**: Spatial/prefix search
- **Size**: 8 characters
- **Features**: Nearby points share prefixes

## Performance

With native Rust acceleration:
- **Q64 encoding**: 40-105x faster than pure Python
- **SimHash**: 1.7-9.7x faster with parallel processing
- **Z-order**: 60-1600x faster with efficient bit manipulation
- **Throughput**: > 230 MB/s for Q64 encoding

## Advanced Usage

```python
# Batch encoding
embeddings = [np.random.randint(0, 256, 256, dtype=np.uint8) for _ in range(100)]
codes = [encode(emb, method="shq64") for emb in embeddings]

# Custom parameters
custom_hash = encode(embedding, method="shq64", planes=128)  # More bits
sparse_repr = encode(embedding, method="t8q64", k=16)  # More indices

# Check native acceleration
from uubed.native_wrapper import is_native_available
print(f"Native module: {is_native_available()}")
```

## Integration Examples

### With Vector Databases

```python
# Store embeddings in Pinecone/Weaviate/etc
encoded = encode(embedding, method="shq64")
db.upsert(id="doc123", values=embedding, metadata={"q64_code": encoded})

# Search with substring matching disabled
results = db.query(
    vector=query_embedding,
    filter={"q64_code": {"$eq": target_code}}  # Exact match only
)
```

### With Search Engines

```python
# Index documents with position-safe codes
doc = {
    "id": "123",
    "content": "...",
    "embedding_code": encode(embedding, method="eq64")
}

# Search without substring pollution
# The code "AQgx" will NOT match "gxAQ" or "QgxA"
```