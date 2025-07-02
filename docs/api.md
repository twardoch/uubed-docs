# API Reference

## Main Functions

### `encode(embedding, method="auto", **kwargs)`

Encode embedding vector using specified method.

**Parameters:**
- `embedding` (Union[List[int], np.ndarray, bytes]): Vector to encode (0-255 integers)
- `method` (str): Encoding method - "eq64", "shq64", "t8q64", "zoq64", or "auto"
- `**kwargs`: Method-specific parameters

**Returns:**
- `str`: Position-safe encoded string

**Example:**
```python
import numpy as np
from uubed import encode

embedding = np.random.randint(0, 256, 256, dtype=np.uint8)
encoded = encode(embedding, method="eq64")
```

### `decode(encoded, method=None)`

Decode encoded string back to bytes.

**Parameters:**
- `encoded` (str): Encoded string
- `method` (str, optional): Encoding method (auto-detected if None)

**Returns:**
- `bytes`: Original bytes

**Note:** Only eq64 supports full decoding. Other methods are lossy compressions.

**Example:**
```python
from uubed import encode, decode

data = bytes(range(32))
encoded = encode(data, method="eq64")
decoded = decode(encoded)
assert data == decoded
```

## Encoding Methods

### Eq64 - Full Embedding Encoder

Encodes full embeddings with position-safe QuadB64.

**Method:** `"eq64"`

**Characteristics:**
- Lossless encoding/decoding
- 2 characters per byte
- No dots in native version (dots were in Python prototype)

### Shq64 - SimHash Encoder

Generates locality-sensitive hash using random projections.

**Method:** `"shq64"`

**Parameters:**
- `planes` (int, default=64): Number of random hyperplanes

**Characteristics:**
- Lossy compression to 64 bits
- Preserves cosine similarity
- Fixed output size (16 characters for 64 planes)

### T8q64 - Top-k Indices Encoder

Encodes the indices of the k highest values.

**Method:** `"t8q64"`

**Parameters:**
- `k` (int, default=8): Number of top indices to keep

**Characteristics:**
- Sparse representation
- Captures most important features
- Fixed output size (2k characters)

### Zoq64 - Z-order Encoder

Encodes using Z-order (Morton) curve for spatial locality.

**Method:** `"zoq64"`

**Characteristics:**
- Spatial locality preservation
- Nearby points share prefixes
- Fixed output size (8 characters)

## Native Acceleration

### `is_native_available()`

Check if native Rust acceleration is available.

**Returns:**
- `bool`: True if native module is loaded

**Example:**
```python
from uubed.native_wrapper import is_native_available

if is_native_available():
    print("Using Rust acceleration!")
else:
    print("Using pure Python implementation")
```

## Position-Safe Alphabets

QuadB64 uses different alphabets for different character positions:

```python
ALPHABETS = [
    "ABCDEFGHIJKLMNOP",  # positions 0, 4, 8, ...
    "QRSTUVWXYZabcdef",  # positions 1, 5, 9, ...
    "ghijklmnopqrstuv",  # positions 2, 6, 10, ...
    "wxyz0123456789-_",  # positions 3, 7, 11, ...
]
```

This ensures that a substring like "abc" can only match at specific positions, eliminating false positives in search engines.

## Performance Tips

1. **Use native module**: Install from source for 40-100x speedup
2. **Batch operations**: Process multiple embeddings together
3. **Choose appropriate method**:
   - `eq64`: When you need full precision
   - `shq64`: For fast similarity comparison
   - `t8q64`: For sparse representations
   - `zoq64`: For spatial/prefix searches

## Error Handling

```python
from uubed import encode, decode

try:
    # Invalid input
    encode([256, 300], method="eq64")  # Values must be 0-255
except ValueError as e:
    print(f"Encoding error: {e}")

try:
    # Invalid decode
    decode("invalid_string")
except ValueError as e:
    print(f"Decoding error: {e}")
```