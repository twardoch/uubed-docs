# Custom Encoding Variants

Learn how to create custom encoding variants and extend uubed's functionality for specialized use cases.

## Overview

uubed's modular architecture allows you to create custom encoding variants for specific requirements:

- **Domain-specific encodings**: Optimized for particular data types
- **Hybrid methods**: Combining multiple encoding strategies
- **Experimental variants**: Testing new encoding approaches
- **Performance-tuned versions**: Optimized for specific hardware

## Creating Custom Encoders

### Basic Encoder Interface

```python
from uubed.encoders.base import BaseEncoder
import numpy as np

class CustomEncoder(BaseEncoder):
    def __init__(self, **kwargs):
        super().__init__()
        self.params = kwargs
    
    def encode(self, data: np.ndarray) -> str:
        """Encode numpy array to string."""
        # Your custom encoding logic here
        processed = self.preprocess(data)
        encoded = self.apply_encoding(processed)
        return self.format_output(encoded)
    
    def decode(self, encoded: str) -> np.ndarray:
        """Decode string back to numpy array."""
        # Your custom decoding logic here
        parsed = self.parse_input(encoded)
        decoded = self.apply_decoding(parsed)
        return self.postprocess(decoded)
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess input data."""
        # Normalization, scaling, etc.
        return data
    
    def apply_encoding(self, data: np.ndarray) -> bytes:
        """Apply core encoding algorithm."""
        raise NotImplementedError
    
    def format_output(self, encoded_bytes: bytes) -> str:
        """Format encoded bytes as string."""
        from uubed.encoders.q64 import q64_encode
        return q64_encode(encoded_bytes)
```

### Example: Sparse Vector Encoder

```python
class SparseVectorEncoder(BaseEncoder):
    """Encoder optimized for sparse vectors."""
    
    def __init__(self, threshold=0.01, max_indices=32):
        super().__init__()
        self.threshold = threshold
        self.max_indices = max_indices
    
    def encode(self, data: np.ndarray) -> str:
        # Find significant indices
        significant_mask = np.abs(data) > self.threshold
        indices = np.where(significant_mask)[0]
        values = data[indices]
        
        # Limit to top-k by magnitude
        if len(indices) > self.max_indices:
            top_k = np.argsort(np.abs(values))[-self.max_indices:]
            indices = indices[top_k]
            values = values[top_k]
        
        # Pack indices and values
        packed_data = self.pack_sparse_data(indices, values)
        
        # Apply position-safe encoding
        from uubed.encoders.q64 import q64_encode
        return q64_encode(packed_data)
    
    def pack_sparse_data(self, indices, values):
        """Pack sparse indices and values into bytes."""
        # Custom packing format for sparse data
        packed = bytearray()
        
        # Number of elements (1 byte)
        packed.append(len(indices))
        
        # Indices (2 bytes each, supports up to 65535 dimensions)
        for idx in indices:
            packed.extend(idx.to_bytes(2, 'little'))
        
        # Values (quantized to uint8)
        quantized_values = self.quantize_values(values)
        packed.extend(quantized_values)
        
        return bytes(packed)
    
    def quantize_values(self, values):
        """Quantize float values to uint8."""
        # Normalize to [0, 1] then scale to [0, 255]
        normalized = (values - values.min()) / (values.max() - values.min())
        return (normalized * 255).astype(np.uint8)
```

### Example: Multi-Scale Encoder

```python
class MultiScaleEncoder(BaseEncoder):
    """Encoder that processes data at multiple scales."""
    
    def __init__(self, scales=[1, 2, 4], base_method="shq64"):
        super().__init__()
        self.scales = scales
        self.base_method = base_method
        self.encoders = self.create_scale_encoders()
    
    def create_scale_encoders(self):
        """Create encoders for each scale."""
        from uubed.encoders import get_encoder
        encoders = {}
        
        for scale in self.scales:
            # Adjust parameters based on scale
            params = self.get_scale_params(scale)
            encoders[scale] = get_encoder(self.base_method, **params)
        
        return encoders
    
    def encode(self, data: np.ndarray) -> str:
        scale_encodings = []
        
        for scale in self.scales:
            # Downsample data for this scale
            scaled_data = self.downsample(data, scale)
            
            # Encode at this scale
            scale_encoding = self.encoders[scale].encode(scaled_data)
            scale_encodings.append(scale_encoding)
        
        # Combine scale encodings
        return self.combine_encodings(scale_encodings)
    
    def downsample(self, data, scale):
        """Downsample data by averaging."""
        if scale == 1:
            return data
        
        # Simple average pooling
        new_size = len(data) // scale
        downsampled = np.zeros(new_size)
        
        for i in range(new_size):
            start_idx = i * scale
            end_idx = min(start_idx + scale, len(data))
            downsampled[i] = np.mean(data[start_idx:end_idx])
        
        return downsampled
```

## Hybrid Encoding Strategies

### Adaptive Encoder

```python
class AdaptiveEncoder(BaseEncoder):
    """Encoder that chooses method based on data characteristics."""
    
    def __init__(self):
        super().__init__()
        self.method_selectors = {
            'sparse': lambda x: np.count_nonzero(x) / len(x) < 0.1,
            'dense': lambda x: np.std(x) > 0.5,
            'uniform': lambda x: np.std(x) < 0.1
        }
        
        self.method_encoders = {
            'sparse': SparseVectorEncoder(),
            'dense': get_encoder('shq64'),
            'uniform': get_encoder('eq64')
        }
    
    def encode(self, data: np.ndarray) -> str:
        # Analyze data characteristics
        selected_method = self.select_method(data)
        
        # Encode with selected method
        encoding = self.method_encoders[selected_method].encode(data)
        
        # Prepend method identifier
        method_id = list(self.method_encoders.keys()).index(selected_method)
        return f"{method_id}:{encoding}"
    
    def select_method(self, data):
        """Select encoding method based on data characteristics."""
        for method, selector in self.method_selectors.items():
            if selector(data):
                return method
        
        # Default to dense if no specific pattern detected
        return 'dense'
```

### Ensemble Encoder

```python
class EnsembleEncoder(BaseEncoder):
    """Encoder that combines multiple encoding methods."""
    
    def __init__(self, methods=['shq64', 't8q64'], weights=None):
        super().__init__()
        self.methods = methods
        self.weights = weights or [1.0] * len(methods)
        self.encoders = [get_encoder(method) for method in methods]
    
    def encode(self, data: np.ndarray) -> str:
        # Encode with each method
        encodings = []
        for encoder in self.encoders:
            encoding = encoder.encode(data)
            encodings.append(encoding)
        
        # Combine encodings with weights
        combined = self.combine_weighted_encodings(encodings, self.weights)
        return combined
    
    def combine_weighted_encodings(self, encodings, weights):
        """Combine multiple encodings with weights."""
        # Simple concatenation with weight indicators
        combined_parts = []
        for encoding, weight in zip(encodings, weights):
            weight_byte = int(weight * 255).to_bytes(1, 'little')
            combined_parts.append(weight_byte + encoding.encode('utf-8'))
        
        # Encode the combined result
        from uubed.encoders.q64 import q64_encode
        return q64_encode(b''.join(combined_parts))
```

## Registration System

### Register Custom Encoder

```python
from uubed.encoders import register_encoder

# Register your custom encoder
register_encoder('sparse', SparseVectorEncoder)
register_encoder('multiscale', MultiScaleEncoder)
register_encoder('adaptive', AdaptiveEncoder)

# Now you can use it with the standard API
from uubed import encode
result = encode(data, method='sparse', threshold=0.05)
```

### Plugin System

```python
# Create a plugin file: uubed_custom_plugin.py
class CustomEncoderPlugin:
    def get_encoders(self):
        return {
            'my_custom': MyCustomEncoder,
            'experimental_v1': ExperimentalEncoder
        }
    
    def get_config_schema(self):
        return {
            'my_custom': {
                'param1': {'type': 'int', 'default': 10},
                'param2': {'type': 'float', 'default': 0.5}
            }
        }

# Register plugin
from uubed.plugins import register_plugin
register_plugin(CustomEncoderPlugin())
```

## Performance Optimization

### SIMD-Optimized Encoder

```python
class SIMDOptimizedEncoder(BaseEncoder):
    """Encoder with SIMD optimizations."""
    
    def __init__(self):
        super().__init__()
        self.simd_available = self.check_simd_support()
    
    def encode(self, data: np.ndarray) -> str:
        if self.simd_available:
            return self.encode_simd(data)
        else:
            return self.encode_fallback(data)
    
    def encode_simd(self, data):
        """SIMD-optimized encoding."""
        # Use numba or other SIMD libraries
        import numba
        
        @numba.jit(nopython=True)
        def simd_process(arr):
            # Vectorized operations
            result = np.zeros_like(arr)
            for i in numba.prange(len(arr)):
                result[i] = self.process_element_simd(arr[i])
            return result
        
        processed = simd_process(data)
        return self.finalize_encoding(processed)
```

### GPU-Accelerated Encoder

```python
class GPUEncoder(BaseEncoder):
    """GPU-accelerated encoder using CuPy."""
    
    def __init__(self):
        super().__init__()
        self.gpu_available = self.check_gpu_support()
    
    def encode(self, data: np.ndarray) -> str:
        if not self.gpu_available:
            return super().encode(data)
        
        try:
            import cupy as cp
            
            # Transfer to GPU
            gpu_data = cp.asarray(data)
            
            # GPU processing
            processed = self.process_on_gpu(gpu_data)
            
            # Transfer back to CPU
            result = cp.asnumpy(processed)
            
            return self.finalize_encoding(result)
        
        except Exception:
            # Fallback to CPU
            return super().encode(data)
```

## Testing Custom Encoders

### Unit Tests

```python
import unittest
import numpy as np

class TestCustomEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = SparseVectorEncoder(threshold=0.1)
        self.test_data = np.random.rand(100)
    
    def test_encode_decode_roundtrip(self):
        """Test that encode/decode preserves important information."""
        encoded = self.encoder.encode(self.test_data)
        decoded = self.encoder.decode(encoded)
        
        # Test that significant values are preserved
        significant_mask = np.abs(self.test_data) > 0.1
        np.testing.assert_allclose(
            self.test_data[significant_mask],
            decoded[significant_mask],
            rtol=0.1
        )
    
    def test_compression_ratio(self):
        """Test compression efficiency."""
        original_size = len(self.test_data.tobytes())
        encoded = self.encoder.encode(self.test_data)
        compressed_size = len(encoded.encode('utf-8'))
        
        compression_ratio = compressed_size / original_size
        self.assertLess(compression_ratio, 0.5)  # At least 50% compression
```

### Benchmark Tests

```python
def benchmark_custom_encoder():
    """Benchmark custom encoder performance."""
    import time
    
    encoder = SparseVectorEncoder()
    test_data = np.random.rand(1000, 512)
    
    # Benchmark encoding
    start_time = time.time()
    for data in test_data:
        encoded = encoder.encode(data)
    encoding_time = time.time() - start_time
    
    print(f"Encoding rate: {len(test_data) / encoding_time:.2f} vectors/sec")
    
    # Benchmark quality
    quality_scores = []
    for data in test_data[:100]:  # Sample for quality check
        encoded = encoder.encode(data)
        decoded = encoder.decode(encoded)
        quality = calculate_similarity(data, decoded)
        quality_scores.append(quality)
    
    print(f"Average quality: {np.mean(quality_scores):.3f}")
```

## Deployment and Distribution

### Packaging Custom Encoders

```python
# setup.py for custom encoder package
from setuptools import setup, find_packages

setup(
    name="uubed-custom-encoders",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["uubed>=1.0.0"],
    entry_points={
        'uubed.encoders': [
            'sparse = my_encoders:SparseVectorEncoder',
            'multiscale = my_encoders:MultiScaleEncoder',
        ]
    }
)
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

# Install uubed and custom encoders
RUN pip install uubed uubed-custom-encoders

# Copy configuration
COPY custom_config.toml /etc/uubed/config.toml

# Set up entrypoint
ENTRYPOINT ["python", "-m", "uubed.server"]
```

## Best Practices

1. **Follow Interface**: Implement the BaseEncoder interface correctly
2. **Error Handling**: Add robust error handling and validation
3. **Documentation**: Document parameters and behavior thoroughly
4. **Testing**: Include comprehensive unit and integration tests
5. **Performance**: Benchmark and optimize critical paths
6. **Backward Compatibility**: Maintain compatibility with existing data

## Related Topics

- [Implementation Architecture](architecture.md)
- [Performance Optimization](../performance/optimization.md)
- [API Reference](../api.md)