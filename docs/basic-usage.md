---
layout: default
title: Basic Usage
nav_order: 4
description: "Essential patterns and common operations for using the uubed library effectively in your applications."
---

> This guide is your quick-start manual for using QuadB64. It shows you how to encode and decode different types of data, from simple text to complex AI embeddings, and how to make sure it runs super fast. Think of it as learning the secret handshake to a club that keeps your data safe and speedy.

# Basic Usage Guide

## Getting Started with uubed

This guide covers the essential patterns and common operations you'll use with the uubed library. After reading this, you'll understand how to effectively use QuadB64 encoding in your applications.

## Core Concepts

### Import Patterns

```python
# Basic imports
from uubed import encode_eq64, decode_eq64
from uubed import encode_shq64, encode_t8q64, encode_zoq64

# Advanced imports
from uubed import encode, decode, Config
from uubed import has_native_extensions, benchmark
```

### The Unified API

The `encode()` and `decode()` functions provide a unified interface:

```python
from uubed import encode, decode

# Specify encoding method
data = b"Hello, world!"
encoded = encode(data, method="eq64")
decoded = decode(encoded)

# Or use variant-specific functions
encoded = encode_eq64(data)  # Same result
decoded = decode_eq64(encoded)  # Same result
```

## Working with Different Data Types

### Text Data

```python
from uubed import encode_eq64, decode_eq64

# String to bytes conversion
text = "QuadB64 prevents substring pollution!"
data = text.encode('utf-8')

# Encode and decode
encoded = encode_eq64(data)
decoded = decode_eq64(encoded)
recovered_text = decoded.decode('utf-8')

assert text == recovered_text
print(f"Original: {text}")
print(f"Encoded:  {encoded}")
print(f"Recovered: {recovered_text}")
```

### Binary Files

```python
# Read binary file
with open("image.jpg", "rb") as f:
    image_data = f.read()

# Encode for storage in text-based systems
encoded = encode_eq64(image_data)

# Store in database, JSON, etc.
document = {
    "id": "img_001",
    "filename": "image.jpg",
    "size": len(image_data),
    "data": encoded  # Safe for text storage
}

# Later: retrieve and decode
decoded_data = decode_eq64(document["data"])
assert decoded_data == image_data
```

### NumPy Arrays

```python
import numpy as np
from uubed import encode_eq64, decode_eq64

# Create array
arr = np.random.rand(100, 50).astype(np.float32)

# Encode array
encoded = encode_eq64(arr.tobytes())

# Decode and reconstruct
decoded_bytes = decode_eq64(encoded)
reconstructed = np.frombuffer(decoded_bytes, dtype=np.float32)
reconstructed = reconstructed.reshape(100, 50)

assert np.array_equal(arr, reconstructed)
```

### ML Embeddings

```python
# Typical ML workflow
from sentence_transformers import SentenceTransformer
from uubed import encode_shq64, encode_eq64

model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
texts = [
    "Machine learning is fascinating",
    "Deep learning uses neural networks",
    "I love pizza"
]

embeddings = model.encode(texts)

# Full precision encoding (reversible)
full_codes = [encode_eq64(emb.tobytes()) for emb in embeddings]

# Compact similarity hashes (irreversible but fast comparison)
hash_codes = [encode_shq64(emb.tobytes()) for emb in embeddings]

print("Full codes (first 30 chars):")
for i, code in enumerate(full_codes):
    print(f"  {i}: {code[:30]}...")

print("\nCompact hashes:")
for i, code in enumerate(hash_codes):
    print(f"  {i}: {code}")
```

## Configuration and Performance

### Check Native Extensions

```python
from uubed import has_native_extensions, get_implementation_info

# Check if native extensions are available
if has_native_extensions():
    print("🚀 Native acceleration enabled!")
else:
    print("⚠️  Using pure Python implementation")
    print("Install native extensions: pip install uubed[native]")

# Get detailed implementation info
info = get_implementation_info()
print(f"Implementation: {info['implementation']}")
print(f"Version: {info['version']}")
print(f"Features: {info['features']}")
```

### Performance Configuration

```python
from uubed import Config, encode_eq64

# Create configuration
config = Config(
    use_native=True,          # Use native implementation if available
    chunk_size=8192,          # Process in 8KB chunks
    num_threads=4,            # Parallel processing threads
    validate_input=True       # Validate input data
)

# Use configuration
large_data = b"x" * 1000000  # 1MB of data
encoded = encode_eq64(large_data, config=config)
```

### Benchmarking

```python
from uubed import benchmark

# Run performance benchmark
results = benchmark()
print("Performance Results:")
print(f"Eq64 encoding: {results['eq64_encode_mb_per_sec']:.1f} MB/s")
print(f"Shq64 hashing: {results['shq64_encode_mb_per_sec']:.1f} MB/s")
print(f"Native available: {results['native_available']}")
```

## Error Handling

### Validation

```python
from uubed import validate_eq64, ValidationError

encoded = "SGVs.bG8s.IFFV.YWRC.NjQh"

# Basic validation
if validate_eq64(encoded):
    decoded = decode_eq64(encoded)
else:
    print("Invalid encoding")

# Detailed validation
try:
    validation = validate_eq64(encoded, detailed=True)
    if not validation['valid']:
        print(f"Validation failed: {validation['errors']}")
except ValidationError as e:
    print(f"Validation error: {e}")
```

### Exception Handling

```python
from uubed import DecodingError, EncodingError

try:
    # This will fail - invalid encoding
    decoded = decode_eq64("invalid.encoding.here")
except DecodingError as e:
    print(f"Decoding failed: {e}")

try:
    # This might fail - very large input
    huge_data = b"x" * (1024 * 1024 * 1024)  # 1GB
    encoded = encode_eq64(huge_data)
except EncodingError as e:
    print(f"Encoding failed: {e}")
```

## Batch Processing

### Encoding Multiple Items

```python
from uubed import encode_batch
from concurrent.futures import ProcessPoolExecutor

# Prepare data
documents = ["doc1", "doc2", "doc3"] * 1000
embeddings = [model.encode(doc) for doc in documents]

# Method 1: Built-in batch encoding
encoded_batch = encode_batch(
    [emb.tobytes() for emb in embeddings],
    method="eq64",
    num_workers=4
)

# Method 2: Manual parallel processing
def encode_chunk(chunk):
    return [encode_eq64(emb.tobytes()) for emb in chunk]

chunk_size = 100
chunks = [embeddings[i:i+chunk_size] 
          for i in range(0, len(embeddings), chunk_size)]

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(encode_chunk, chunks))
    all_encoded = [item for chunk in results for item in chunk]
```

### Streaming Processing

```python
from uubed import StreamEncoder

# For very large files or continuous data
encoder = StreamEncoder("eq64")

def process_large_file(input_path, output_path):
    with open(input_path, "rb") as input_file:
        with open(output_path, "w") as output_file:
            while True:
                chunk = input_file.read(4096)  # 4KB chunks
                if not chunk:
                    break
                
                encoded_chunk = encoder.encode_chunk(chunk)
                output_file.write(encoded_chunk + "\n")
            
            # Write any remaining data
            final_chunk = encoder.finalize()
            if final_chunk:
                output_file.write(final_chunk + "\n")
```

## Common Patterns

### Data Pipeline Integration

```python
# ETL pipeline with QuadB64
class DataPipeline:
    def __init__(self, variant="eq64"):
        self.variant = variant
        
    def extract(self, source):
        """Extract data from source"""
        # Your extraction logic
        return raw_data
    
    def transform(self, data):
        """Transform and encode data"""
        processed = self.process_data(data)
        
        if self.variant == "eq64":
            return encode_eq64(processed)
        elif self.variant == "shq64":
            return encode_shq64(processed)
        else:
            return encode(processed, method=self.variant)
    
    def load(self, encoded_data, destination):
        """Load encoded data to destination"""
        # Store in database, search engine, etc.
        destination.store(encoded_data)

# Usage
pipeline = DataPipeline("shq64")
result = pipeline.transform(input_data)
```

### Database Integration

```python
import sqlite3
from uubed import encode_eq64, decode_eq64

# Setup database
conn = sqlite3.connect('embeddings.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS vectors (
    id INTEGER PRIMARY KEY,
    content TEXT,
    embedding_eq64 TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

# Insert with encoding
def store_embedding(content: str, embedding: np.ndarray):
    encoded = encode_eq64(embedding.tobytes())
    cursor.execute(
        "INSERT INTO vectors (content, embedding_eq64) VALUES (?, ?)",
        (content, encoded)
    )
    conn.commit()

# Retrieve with decoding
def get_embedding(vector_id: int) -> np.ndarray:
    cursor.execute(
        "SELECT embedding_eq64 FROM vectors WHERE id = ?",
        (vector_id,)
    )
    encoded = cursor.fetchone()[0]
    decoded_bytes = decode_eq64(encoded)
    return np.frombuffer(decoded_bytes, dtype=np.float32)
```

### Web API Integration

```python
from flask import Flask, request, jsonify
from uubed import encode_eq64, decode_eq64

app = Flask(__name__)

@app.route('/encode', methods=['POST'])
def encode_endpoint():
    try:
        # Get binary data from request
        data = request.get_data()
        
        # Encode
        encoded = encode_eq64(data)
        
        return jsonify({
            'encoded': encoded,
            'original_size': len(data),
            'encoded_size': len(encoded)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/decode', methods=['POST'])
def decode_endpoint():
    try:
        # Get encoded string from request
        data = request.json
        encoded = data['encoded']
        
        # Decode
        decoded = decode_eq64(encoded)
        
        # Return as base64 for JSON compatibility
        import base64
        return jsonify({
            'decoded': base64.b64encode(decoded).decode(),
            'size': len(decoded)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
```

## Best Practices Summary

### Do's ✅

1. **Choose the right variant**: 
   - Eq64 for lossless encoding
   - Shq64 for similarity comparison
   - T8q64 for sparse data
   - Zoq64 for spatial data

2. **Use native extensions**: Install with `pip install uubed[native]`

3. **Validate untrusted input**: Use `validate_*()` functions

4. **Handle errors gracefully**: Wrap in try-catch blocks

5. **Batch when possible**: Better performance for multiple items

### Don'ts ❌

1. **Don't modify encoded strings**: They become invalid

2. **Don't mix variants**: Each has specific use cases

3. **Don't ignore performance**: Check for native extensions

4. **Don't store without validation**: Validate critical encoded data

5. **Don't assume reversibility**: Only Eq64 is reversible

## Next Steps

- Explore [Advanced Features](advanced-features.md)
- Learn about [Integration Patterns](integration/overview.md)
- Read [Performance Tuning](performance/optimization.md)
- Check out [Real-world Examples](examples/)