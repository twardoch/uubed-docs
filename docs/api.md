---
layout: default
title: API Reference
nav_order: 5
description: "Complete API documentation for the UUBED library, including all encoding functions, classes, and utilities."
---

> This is your go-to guide for all the buttons and levers in the QuadB64 library. It tells you exactly how to use the `encode` and `decode` functions, what different encoding methods do, and how to make sure everything runs smoothly.

# API Reference

Imagine you have a universal remote control for all your data encoding needs. This API reference is the instruction manual for that remote, explaining what each button does and how to combine them for the best results. 

Imagine you're a chef, and QuadB64 is your kitchen. This API is your recipe book, detailing every ingredient (parameters), every cooking method (encoding methods), and how to combine them to create delicious, perfectly encoded data dishes.

## Main Functions

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

## Low-Level Q64 Codec

### `q64_encode(data_bytes)`

Base Q64 encoding without dots. Used internally by other encoders.

**Parameters:**
- `data_bytes` (bytes): Raw bytes to encode

**Returns:**
- `str`: Q64 encoded string

**Example:**
```python
from uubed.encoders.q64 import q64_encode

encoded = q64_encode(b"Hello World")
```

### `q64_decode(encoded_string)`

Base Q64 decoding.

**Parameters:**
- `encoded_string` (str): Q64 encoded string

**Returns:**
- `bytes`: Decoded bytes

**Example:**
```python
from uubed.encoders.q64 import q64_decode

decoded = q64_decode(encoded_string)
```

## Module-Level Imports

### Encoder Modules

Each encoding method has its own module with specialized functions:

```python
# Eq64 (Full Precision)
from uubed.encoders.eq64 import eq64_encode, eq64_decode

# Shq64 (SimHash)
from uubed.encoders.shq64 import simhash_q64

# T8q64 (Top-k)
from uubed.encoders.t8q64 import top_k_q64

# Zoq64 (Z-order)
from uubed.encoders.zoq64 import z_order_q64
```

### Constants

```python
from uubed.encoders.q64 import Q64_ALPHABETS, Q64_REVERSE

# Position-dependent alphabets
print(Q64_ALPHABETS[0])  # 'ABCD...XYZ'  (uppercase)
print(Q64_ALPHABETS[1])  # 'abcd...xyz'  (lowercase)
print(Q64_ALPHABETS[2])  # 'AaBb...YyZz' (mixed case)
print(Q64_ALPHABETS[3])  # '0123...+/-_' (digits/symbols)

# Reverse lookup table for decoding
char_value = Q64_REVERSE[position][character]
```

## Type Hints

UUBED provides comprehensive type hints for better IDE support:

```python
from typing import Union, Literal
import numpy as np
from numpy.typing import NDArray

def encode(
    embedding: Union[NDArray[np.float32], list, bytes],
    method: Literal["auto", "eq64", "shq64", "t8q64", "zoq64"] = "auto",
    validate: bool = True
) -> str: ...
```

## Native Module Detection

```python
import uubed

if uubed._has_native:
    print("Native Rust acceleration available")
else:
    print("Using pure Python implementation")
```

You can also use:

```python
from uubed.native_wrapper import is_native_available

if is_native_available():
    print("Rust acceleration enabled!")
```