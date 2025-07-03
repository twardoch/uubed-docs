---
title: Quick Start Guide
keywords: quickstart, installation, setup, getting started
sidebar: home_sidebar
permalink: quickstart.html
summary: Get up and running with QuadB64 in minutes. This guide covers installation, basic usage, and common integration patterns.
---

# Quick Start Guide

Get up and running with QuadB64 in minutes! This guide covers installation, basic usage, and common integration patterns.

## Installation

The simplest way to install uubed:

```bash
pip install uubed
```

For maximum performance with native extensions:

```bash
pip install uubed[native]
```

For development or latest features:

```bash
git clone https://github.com/twardoch/uubed.git
cd uubed
pip install -e ".[dev]"
```

See the [Installation Guide](installation.md) for detailed instructions and troubleshooting.

## Basic Usage

### Simple Text Encoding

```python
from uubed import encode_eq64, decode_eq64

# Encode any binary data
data = b"Hello, QuadB64 World!"
encoded = encode_eq64(data)
print(f"Encoded: {encoded}")
# Output: SGVs.bG8s.IFFV.YWRC.NjQg.V29y.bGQh

# Decode back to original
decoded = decode_eq64(encoded)
assert decoded == data
print(f"Decoded: {decoded}")
# Output: b'Hello, QuadB64 World!'
```

### Working with Embeddings

```python
import numpy as np
from uubed import encode, decode

# Create a sample embedding (e.g., from an ML model)
embedding = np.random.rand(768).astype(np.float32)

# Convert to bytes
embedding_bytes = embedding.tobytes()

# Full precision encoding with Eq64
full_code = encode(embedding_bytes, method="eq64")
print(f"Full encoding length: {len(full_code)} chars")

# Compact similarity hash with Shq64
compact_code = encode(embedding_bytes, method="shq64")
print(f"Compact hash: {compact_code}")  # 16 characters

# Decode back (only works for eq64)
decoded_bytes = decode(full_code)
decoded_embedding = np.frombuffer(decoded_bytes, dtype=np.float32)
assert np.allclose(embedding, decoded_embedding)
```

## Why QuadB64?

### The Problem with Traditional Base64

When search engines index Base64-encoded data, they treat it as regular text:

```python
# Two completely different embeddings
embedding1 = model.encode("cats are cute")
embedding2 = model.encode("quantum physics")

# Traditional Base64 encoding
import base64
b64_1 = base64.b64encode(embedding1.tobytes()).decode()
b64_2 = base64.b64encode(embedding2.tobytes()).decode()

# Substring pollution: random matches!
# "YWJj" might appear in both encodings by chance
# Search engines will falsely match these unrelated documents
```

### The QuadB64 Solution

QuadB64 uses position-dependent encoding to prevent false matches:

```python
# QuadB64 encoding
from uubed import encode_eq64
q64_1 = encode_eq64(embedding1.tobytes())
q64_2 = encode_eq64(embedding2.tobytes())

# Position-safe: "YWJj" at position 0 ≠ "YWJj" at position 4
# No false substring matches between unrelated documents!
```

Key benefits:
- ✅ **No substring pollution**: Position-dependent alphabets
- ✅ **Search accuracy**: Only genuine matches are found
- ✅ **Easy integration**: Drop-in replacement for Base64
- ✅ **High performance**: Minimal overhead vs Base64

## Encoding Methods

uubed provides multiple encoding schemes optimized for different use cases:

### Eq64 - Full Embeddings
Perfect for when you need lossless encoding:

```python
from uubed import encode_eq64, decode_eq64

data = b"Your binary data here"
encoded = encode_eq64(data)  # Position-safe, dots every 4 chars
decoded = decode_eq64(encoded)  # Get original data back
```

- **Size**: ~1.33x original (same as Base64)
- **Use cases**: Full embeddings, binary files, any lossless encoding
- **Features**: Complete reversibility, position safety

### Shq64 - SimHash Variant
Compact similarity-preserving hashes:

```python
from uubed import encode_shq64

# 768-dimensional embedding
embedding = model.encode("sample text")
hash_code = encode_shq64(embedding.tobytes())
print(hash_code)  # 16-character hash like "QRsT.UvWx.YZab.cdef"
```

- **Size**: Always 16 characters (64-bit hash)
- **Use cases**: Deduplication, similarity search, clustering
- **Features**: Preserves cosine similarity, extremely compact

### T8q64 - Top-k Indices
Sparse representation capturing most important features:

```python
from uubed import encode_t8q64

# Encode top-8 most significant indices
sparse_code = encode_t8q64(embedding.tobytes(), k=8)
```

- **Size**: 16 characters (8 indices + magnitudes)
- **Use cases**: Sparse embeddings, feature selection
- **Features**: Captures most informative dimensions

### Zoq64 - Z-order Curve
Spatial locality-preserving encoding:

```python
from uubed import encode_zoq64

# 2D or higher dimensional data
spatial_code = encode_zoq64(coordinates)
```

- **Size**: Variable (based on precision needs)
- **Use cases**: Geospatial data, multi-dimensional indexing
- **Features**: Nearby points have similar prefixes

## Performance

QuadB64 is designed for production workloads:

| Operation | Pure Python | With Native Extensions | Speedup |
|-----------|-------------|----------------------|---------|
| Eq64 encoding | 5.5 MB/s | 230+ MB/s | 40-105x |
| Shq64 hashing | 12 MB/s | 117 MB/s | 9.7x |
| T8q64 sparse | 8 MB/s | 156 MB/s | 19.5x |
| Zoq64 spatial | 0.3 MB/s | 480 MB/s | 1600x |

Check if native extensions are available:

```python
from uubed import has_native_extensions

if has_native_extensions():
    print("🚀 Native acceleration enabled!")
else:
    print("Using pure Python implementation")
```

## Common Patterns

### Batch Processing

```python
from uubed import encode_batch

# Process multiple embeddings efficiently
embeddings = [model.encode(text) for text in documents]
encoded_batch = encode_batch(embeddings, method="shq64")

# Parallel processing for large datasets
from concurrent.futures import ProcessPoolExecutor

def process_chunk(chunk):
    return [encode_eq64(emb.tobytes()) for emb in chunk]

with ProcessPoolExecutor() as executor:
    chunks = [embeddings[i:i+100] for i in range(0, len(embeddings), 100)]
    results = list(executor.map(process_chunk, chunks))
```

### Configuration Options

```python
from uubed import Config

# Custom configuration
config = Config(
    default_variant="eq64",
    use_native=True,
    chunk_size=8192,
    num_threads=4
)

# Apply configuration
encoded = encode(data, config=config)
```

## Real-World Integration

### Vector Databases (Pinecone, Weaviate, Qdrant)

```python
from uubed import encode_shq64
import pinecone

# Initialize your vector database
index = pinecone.Index("my-index")

# Store embeddings with QuadB64 codes
for doc_id, text in documents.items():
    embedding = model.encode(text)
    q64_code = encode_shq64(embedding.tobytes())
    
    index.upsert(
        vectors=[(doc_id, embedding.tolist())],
        metadata={doc_id: {"text": text, "q64_code": q64_code}}
    )

# Similarity search without substring pollution
query_embedding = model.encode(query_text)
query_code = encode_shq64(query_embedding.tobytes())

# Find exact code matches (no false positives!)
results = index.query(
    vector=query_embedding.tolist(),
    filter={"q64_code": {"$eq": query_code}},
    top_k=10
)
```

### Elasticsearch / OpenSearch

```python
from elasticsearch import Elasticsearch
from uubed import encode_eq64

es = Elasticsearch()

# Index documents with position-safe encoding
doc = {
    "title": "Introduction to QuadB64",
    "content": "QuadB64 solves substring pollution...",
    "embedding": embedding.tolist(),
    "embedding_q64": encode_eq64(embedding.tobytes())
}

es.index(index="docs", id="doc1", body=doc)

# Search with exact matching on encoded field
query = {
    "query": {
        "bool": {
            "must": [
                {"match": {"content": "QuadB64"}},
                {"term": {"embedding_q64.keyword": target_code}}
            ]
        }
    }
}

results = es.search(index="docs", body=query)
```

### LangChain Integration

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from uubed import encode_shq64

# Custom embeddings wrapper
class QuadB64Embeddings(OpenAIEmbeddings):
    def embed_documents(self, texts):
        embeddings = super().embed_documents(texts)
        # Add QuadB64 codes to metadata
        return [(emb, {"q64": encode_shq64(np.array(emb).tobytes())}) 
                for emb in embeddings]

# Use with vector store
embeddings = QuadB64Embeddings()
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)
```

## Next Steps

- 📖 Read the [Theory](theory/introduction.md) behind QuadB64
- 🔧 Explore the [API Reference](api.md) for detailed documentation
- 🚀 Check out [Performance Benchmarks](reference/benchmarks.md)
- 💡 See more [Examples](https://github.com/twardoch/uubed/tree/main/examples)

## Need Help?

- 💬 Join our [Discord Community](https://discord.gg/uubed)
- 🐛 Report issues on [GitHub](https://github.com/twardoch/uubed/issues)
- 📧 Contact: support@uubed.io