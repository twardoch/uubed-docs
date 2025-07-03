---
layout: default
title: Advanced Features and Configuration
parent: Implementation
nav_order: 3
description: "Sophisticated usage patterns, troubleshooting techniques, and advanced configuration options for power users and system integrators working with QuadB64 in complex environments."
---

# Advanced Features and Configuration

## Overview

This chapter covers sophisticated usage patterns, troubleshooting techniques, and advanced configuration options for power users and system integrators working with QuadB64 in complex environments.

## Advanced Configuration

### Custom Variant Creation

Beyond the standard QuadB64 variants (Eq64, Shq64, T8q64, Zoq64), you can create specialized variants for specific use cases:

```python
from uubed.core import VariantBuilder, AlphabetGenerator, PositionFunction

class DomainSpecificVariant:
    """Create QuadB64 variants optimized for specific domains"""
    
    @staticmethod
    def create_genomic_variant():
        """Variant optimized for genomic data encoding"""
        
        def genomic_alphabet_generator(position: int) -> str:
            # Optimize for ATCG frequency in genomic data
            # Place nucleotide-like characters first
            base = "ATCGNatcgn0123456789BDEFHIJKLMOPQRSUVWXYZ+/bdefhijklmopqrsuvwxyz"
            
            # Rotate based on codon position (groups of 3)
            codon_position = (position // 3) % 3
            rotation = codon_position * 21  # Distribute across alphabet
            
            return base[rotation:] + base[:rotation]
        
        def genomic_position_function(position: int, context: dict = None) -> int:
            # Account for reading frame in genomic sequences
            reading_frame = context.get('reading_frame', 0) if context else 0
            return (position + reading_frame) % (2**24)
        
        return VariantBuilder.create_variant(
            name="Genomic64",
            alphabet_generator=genomic_alphabet_generator,
            position_function=genomic_position_function,
            metadata={
                'domain': 'genomics',
                'optimized_for': 'nucleotide_sequences',
                'reading_frame_aware': True
            }
        )
    
    @staticmethod 
    def create_financial_variant():
        """Variant optimized for financial data encoding"""
        
        def financial_alphabet_generator(position: int) -> str:
            # Prioritize numeric characters for financial data
            base = "0123456789.$+-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/"
            
            # Rotate based on decimal precision context
            precision_rotation = (position * 11) % 64
            return base[precision_rotation:] + base[:precision_rotation]
        
        def financial_position_function(position: int, context: dict = None) -> int:
            # Incorporate timestamp for temporal consistency
            timestamp = context.get('timestamp', 0) if context else 0
            time_factor = int(timestamp / 3600) % 1000  # Hour-based rotation
            return (position * 13 + time_factor) % (2**28)
        
        return VariantBuilder.create_variant(
            name="Financial64",
            alphabet_generator=financial_alphabet_generator,
            position_function=financial_position_function,
            metadata={
                'domain': 'finance',
                'temporal_consistency': True,
                'precision_optimized': True
            }
        )

# Usage example
genomic_encoder = DomainSpecificVariant.create_genomic_variant()
financial_encoder = DomainSpecificVariant.create_financial_variant()

# Encode genomic sequence with reading frame context
dna_sequence = b"ATCGATCGATCG" * 100
encoded_dna = genomic_encoder.encode(dna_sequence, context={'reading_frame': 1})

# Encode financial data with timestamp context
financial_data = b'{"price": 123.45, "volume": 10000}'
encoded_financial = financial_encoder.encode(
    financial_data, 
    context={'timestamp': time.time()}
)
```

### Performance Tuning for Specific Use Cases

#### High-Throughput Streaming Configuration

```python
class HighThroughputConfig:
    """Configuration optimized for high-throughput streaming scenarios"""
    
    def __init__(self):
        self.batch_size = 8192  # Optimal for network packets
        self.worker_threads = min(32, os.cpu_count() * 2)
        self.memory_pool_size = 64 * 1024 * 1024  # 64MB pool
        self.enable_compression = True
        self.cache_size = 100000
        
    def create_streaming_encoder(self):
        """Create encoder optimized for streaming"""
        
        encoder = StreamingEncoder(
            batch_size=self.batch_size,
            max_workers=self.worker_threads,
            memory_pool_size=self.memory_pool_size,
            cache_config={
                'max_size': self.cache_size,
                'ttl_seconds': 300,  # 5 minute TTL
                'eviction_policy': 'lru'
            }
        )
        
        # Enable hardware acceleration if available
        if self._has_simd_support():
            encoder.enable_simd_acceleration()
            
        return encoder
    
    def _has_simd_support(self) -> bool:
        """Detect available SIMD instruction sets"""
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            return any(flag in cpu_info.get('flags', []) 
                      for flag in ['avx2', 'sse4_1', 'neon'])
        except ImportError:
            return False

class StreamingEncoder:
    """High-performance streaming encoder"""
    
    def __init__(self, batch_size=8192, max_workers=8, 
                 memory_pool_size=64*1024*1024, cache_config=None):
        
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.memory_pool = MemoryPool(memory_pool_size)
        self.cache = LRUCache(**(cache_config or {}))
        self.statistics = StreamingStats()
        
        # Pre-allocate working buffers
        self.working_buffers = [
            self.memory_pool.get_buffer(batch_size * 2)
            for _ in range(max_workers)
        ]
        
    def encode_stream(self, data_stream, output_stream):
        """Encode data stream with optimal batching"""
        
        batch_buffer = bytearray()
        batch_count = 0
        
        try:
            for chunk in data_stream:
                batch_buffer.extend(chunk)
                
                # Process when batch is full
                if len(batch_buffer) >= self.batch_size:
                    self._process_batch(batch_buffer, output_stream, batch_count)
                    batch_buffer.clear()
                    batch_count += 1
            
            # Process remaining data
            if batch_buffer:
                self._process_batch(batch_buffer, output_stream, batch_count)
                
        finally:
            self.executor.shutdown(wait=True)
    
    def _process_batch(self, batch_data: bytearray, output_stream, batch_id: int):
        """Process a single batch with position-aware encoding"""
        
        # Calculate global position offset
        global_position = batch_id * self.batch_size
        
        # Submit to thread pool
        future = self.executor.submit(
            self._encode_batch_worker, 
            bytes(batch_data), 
            global_position
        )
        
        # Get result and write to output
        try:
            encoded_data = future.result(timeout=30.0)
            output_stream.write(encoded_data)
            output_stream.flush()
            
            self.statistics.record_batch(len(batch_data), len(encoded_data))
            
        except Exception as e:
            self.statistics.record_error(str(e))
            raise RuntimeError(f"Batch {batch_id} encoding failed: {e}")
    
    def _encode_batch_worker(self, data: bytes, position: int) -> bytes:
        """Worker function for batch encoding"""
        
        # Check cache first
        cache_key = (hash(data), position)
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
            
        # Encode with position context
        encoded = encode_eq64_with_position(data, position)
        
        # Cache result
        self.cache.put(cache_key, encoded)
        
        return encoded.encode('utf-8')

class StreamingStats:
    """Statistics collection for streaming operations"""
    
    def __init__(self):
        self.total_bytes_in = 0
        self.total_bytes_out = 0
        self.batch_count = 0
        self.error_count = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    def record_batch(self, bytes_in: int, bytes_out: int):
        with self._lock:
            self.total_bytes_in += bytes_in
            self.total_bytes_out += bytes_out
            self.batch_count += 1
    
    def record_error(self, error_msg: str):
        with self._lock:
            self.error_count += 1
    
    def get_stats(self) -> dict:
        with self._lock:
            elapsed = time.time() - self.start_time
            return {
                'total_input_mb': self.total_bytes_in / 1024 / 1024,
                'total_output_mb': self.total_bytes_out / 1024 / 1024,
                'compression_ratio': self.total_bytes_out / max(self.total_bytes_in, 1),
                'throughput_mb_s': (self.total_bytes_in / 1024 / 1024) / max(elapsed, 0.001),
                'batches_processed': self.batch_count,
                'error_rate': self.error_count / max(self.batch_count, 1),
                'elapsed_time': elapsed
            }
```

#### Low-Latency Configuration

```python
class LowLatencyConfig:
    """Configuration optimized for low-latency scenarios"""
    
    def __init__(self):
        # Minimize context switching and memory allocation
        self.enable_pre_allocation = True
        self.disable_threading = True  # Single-threaded for consistency
        self.cache_warmup = True
        self.use_stack_buffers = True
        
    def create_low_latency_encoder(self):
        """Create encoder optimized for minimum latency"""
        
        encoder = LowLatencyEncoder()
        
        if self.cache_warmup:
            encoder.warmup_cache()
            
        return encoder

class LowLatencyEncoder:
    """Encoder optimized for minimal latency"""
    
    def __init__(self):
        # Pre-allocate all possible buffers
        self.buffer_cache = {}
        self.alphabet_cache = {}
        self.precomputed_tables = self._build_lookup_tables()
        
        # Pre-generate common alphabets
        for pos in range(1024):  # Cache first 1024 positions
            self.alphabet_cache[pos] = self._generate_alphabet(pos)
    
    def encode_minimal_latency(self, data: bytes, position: int = 0) -> str:
        """Encode with minimal latency - no dynamic allocation"""
        
        if len(data) == 0:
            return ""
            
        # Use stack-allocated buffer for small data
        if len(data) <= 1024:
            return self._encode_stack_buffer(data, position)
        else:
            return self._encode_pre_allocated(data, position)
    
    def _encode_stack_buffer(self, data: bytes, position: int) -> str:
        """Encode using stack-allocated buffer"""
        
        # Get pre-computed alphabet
        alphabet = self.alphabet_cache.get(position % 1024)
        if not alphabet:
            alphabet = self._generate_alphabet(position)
        
        # Direct encoding without memory allocation
        result_chars = []
        
        for i in range(0, len(data), 3):
            chunk = data[i:i+3]
            encoded_chunk = self._encode_chunk_direct(chunk, alphabet, position + i)
            result_chars.extend(encoded_chunk)
        
        return ''.join(result_chars)
    
    def _encode_chunk_direct(self, chunk: bytes, alphabet: str, position: int) -> list:
        """Direct chunk encoding without intermediate allocations"""
        
        # Pad chunk to 3 bytes
        while len(chunk) < 3:
            chunk += b'\x00'
        
        # Convert to 24-bit integer
        value = (chunk[0] << 16) | (chunk[1] << 8) | chunk[2]
        
        # Extract 6-bit groups using lookup table
        indices = [
            (value >> 18) & 0x3F,
            (value >> 12) & 0x3F,
            (value >> 6) & 0x3F,
            value & 0x3F
        ]
        
        # Apply position-dependent rotation
        rotation = (position // 3) % 64
        rotated_indices = [(idx + rotation) % 64 for idx in indices]
        
        # Map to alphabet characters
        return [alphabet[idx] for idx in rotated_indices]
    
    def warmup_cache(self):
        """Pre-compute common operations to minimize latency"""
        
        # Warm up with common data patterns
        test_patterns = [
            b'\x00' * 16,  # Null pattern
            b'\xFF' * 16,  # All ones
            bytes(range(16)),  # Sequential
            b'Hello, World!' * 2  # Text pattern
        ]
        
        for pattern in test_patterns:
            for pos in range(0, 64, 8):
                self.encode_minimal_latency(pattern, pos)
```

### Integration with Exotic Platforms

#### Embedded Systems Integration

```python
class EmbeddedSystemsConfig:
    """Configuration for resource-constrained embedded systems"""
    
    def __init__(self, memory_limit_kb=256, cpu_mhz=100):
        self.memory_limit = memory_limit_kb * 1024
        self.cpu_frequency = cpu_mhz * 1000000
        self.enable_power_saving = True
        self.use_lookup_tables = memory_limit_kb > 64  # Only if sufficient memory
        
    def create_embedded_encoder(self):
        """Create encoder suitable for embedded systems"""
        
        if self.memory_limit < 32 * 1024:  # Very constrained
            return MinimalEncoder(self.memory_limit)
        elif self.memory_limit < 128 * 1024:  # Moderately constrained  
            return CompactEncoder(self.memory_limit)
        else:  # Relatively unconstrained
            return OptimizedEmbeddedEncoder(self.memory_limit)

class MinimalEncoder:
    """Minimal encoder for severely memory-constrained environments"""
    
    def __init__(self, memory_limit: int):
        self.memory_limit = memory_limit
        # No caches or lookup tables - compute everything on-demand
        
    def encode(self, data: bytes, position: int = 0) -> str:
        """Encode with minimal memory footprint"""
        
        if len(data) == 0:
            return ""
        
        result = []
        
        # Process one character at a time to minimize memory usage
        for i in range(0, len(data), 3):
            chunk = data[i:min(i+3, len(data))]
            encoded = self._encode_chunk_minimal(chunk, position + i)
            result.append(encoded)
        
        return ''.join(result)
    
    def _encode_chunk_minimal(self, chunk: bytes, position: int) -> str:
        """Encode chunk with zero additional memory allocation"""
        
        # Generate alphabet on-demand (no caching)
        alphabet = self._generate_alphabet_minimal(position)
        
        # Pad chunk
        padded = chunk + b'\x00' * (3 - len(chunk))
        
        # Convert to integer and extract indices
        value = (padded[0] << 16) | (padded[1] << 8) | padded[2]
        
        # Extract and rotate indices
        rotation = (position // 3) % 64
        indices = [
            ((value >> 18) & 0x3F + rotation) % 64,
            ((value >> 12) & 0x3F + rotation) % 64,
            ((value >> 6) & 0x3F + rotation) % 64,
            ((value & 0x3F) + rotation) % 64
        ]
        
        # Build result directly
        return ''.join(alphabet[idx] for idx in indices[:len(chunk)+1])
    
    def _generate_alphabet_minimal(self, position: int) -> str:
        """Generate alphabet without caching"""
        base = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        rotation = (position // 3) % 64
        return base[rotation:] + base[:rotation]
```

#### GPU Acceleration Integration

```python
try:
    import cupy as cp  # GPU acceleration with CuPy
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class GPUAcceleratedEncoder:
    """QuadB64 encoder with GPU acceleration for large datasets"""
    
    def __init__(self):
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU acceleration requires CuPy")
        
        self.device = cp.cuda.Device()
        self.memory_pool = cp.get_default_memory_pool()
        self._compile_kernels()
    
    def _compile_kernels(self):
        """Compile CUDA kernels for QuadB64 operations"""
        
        # CUDA kernel for parallel encoding
        encode_kernel_code = """
        extern "C" __global__
        void quadb64_encode_kernel(
            const unsigned char* input,
            char* output,
            const char* alphabet,
            int* positions,
            int data_size,
            int chunk_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int chunk_start = idx * chunk_size;
            
            if (chunk_start >= data_size) return;
            
            // Process chunk
            int chunk_end = min(chunk_start + chunk_size, data_size);
            int position = positions[idx];
            
            // Generate position-dependent alphabet
            char local_alphabet[64];
            int rotation = (position / 3) % 64;
            
            for (int i = 0; i < 64; i++) {
                local_alphabet[i] = alphabet[(i + rotation) % 64];
            }
            
            // Encode chunk
            for (int i = chunk_start; i < chunk_end; i += 3) {
                // Get 3 bytes (pad with zeros if needed)
                unsigned int value = 0;
                for (int j = 0; j < 3 && i + j < data_size; j++) {
                    value |= (input[i + j] << (16 - 8 * j));
                }
                
                // Extract 6-bit groups
                int out_idx = (i / 3) * 4;
                output[out_idx + 0] = local_alphabet[(value >> 18) & 0x3F];
                output[out_idx + 1] = local_alphabet[(value >> 12) & 0x3F];
                output[out_idx + 2] = local_alphabet[(value >> 6) & 0x3F];
                output[out_idx + 3] = local_alphabet[value & 0x3F];
            }
        }
        """
        
        self.encode_kernel = cp.RawKernel(encode_kernel_code, 'quadb64_encode_kernel')
    
    def encode_gpu(self, data_list: list, positions: list = None) -> list:
        """Encode multiple data chunks on GPU"""
        
        if not positions:
            positions = list(range(len(data_list)))
        
        # Prepare GPU memory
        max_data_size = max(len(data) for data in data_list)
        max_output_size = ((max_data_size + 2) // 3) * 4
        
        # Allocate GPU memory
        gpu_input = cp.zeros(max_data_size, dtype=cp.uint8)
        gpu_output = cp.zeros(max_output_size, dtype=cp.int8)
        gpu_alphabet = cp.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"), dtype=cp.int8)
        
        results = []
        
        for i, (data, position) in enumerate(zip(data_list, positions)):
            # Copy data to GPU
            data_array = cp.array(list(data), dtype=cp.uint8)
            gpu_input[:len(data)] = data_array
            
            # Set up kernel parameters
            chunk_size = 192  # Process 192 bytes per thread (64 output chars)
            num_chunks = (len(data) + chunk_size - 1) // chunk_size
            
            positions_array = cp.array([position + i * chunk_size for i in range(num_chunks)], dtype=cp.int32)
            
            # Launch kernel
            threads_per_block = min(256, num_chunks)
            blocks = (num_chunks + threads_per_block - 1) // threads_per_block
            
            self.encode_kernel(
                (blocks,), (threads_per_block,),
                (gpu_input, gpu_output, gpu_alphabet, positions_array, len(data), chunk_size)
            )
            
            # Copy result back to CPU
            output_size = ((len(data) + 2) // 3) * 4
            result_bytes = cp.asnumpy(gpu_output[:output_size])
            result_str = ''.join(chr(b) for b in result_bytes if b != 0)
            results.append(result_str)
        
        return results
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Performance Degradation

**Symptoms:**
- Encoding speed significantly slower than expected
- High CPU usage with low throughput
- Memory usage growing over time

**Diagnostic Steps:**

```python
class PerformanceDiagnostic:
    """Diagnostic tools for performance issues"""
    
    def diagnose_performance_issue(self, encoder, test_data: bytes):
        """Comprehensive performance diagnosis"""
        
        diagnostics = {
            'system_info': self._get_system_info(),
            'encoder_config': self._analyze_encoder_config(encoder),
            'memory_usage': self._analyze_memory_usage(),
            'cpu_utilization': self._analyze_cpu_usage(),
            'bottlenecks': self._identify_bottlenecks(encoder, test_data)
        }
        
        return self._generate_performance_report(diagnostics)
    
    def _identify_bottlenecks(self, encoder, test_data: bytes) -> dict:
        """Identify specific performance bottlenecks"""
        
        import time
        import tracemalloc
        
        bottlenecks = {}
        
        # Test different aspects
        test_scenarios = {
            'small_data': test_data[:100],
            'medium_data': test_data[:10000],
            'large_data': test_data,
            'repeated_small': test_data[:100] * 100
        }
        
        for scenario, data in test_scenarios.items():
            # Start memory tracking
            tracemalloc.start()
            start_time = time.perf_counter()
            
            # Run encoding
            try:
                result = encoder.encode(data)
                end_time = time.perf_counter()
                
                # Get memory stats
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                bottlenecks[scenario] = {
                    'duration': end_time - start_time,
                    'throughput_mb_s': len(data) / (end_time - start_time) / 1024 / 1024,
                    'memory_current_mb': current / 1024 / 1024,
                    'memory_peak_mb': peak / 1024 / 1024,
                    'efficiency_score': len(result) / (end_time - start_time) / peak
                }
                
            except Exception as e:
                bottlenecks[scenario] = {'error': str(e)}
        
        return bottlenecks

# Usage
diagnostic = PerformanceDiagnostic()
test_data = b"sample data" * 10000
encoder = SomeQuadB64Encoder()

report = diagnostic.diagnose_performance_issue(encoder, test_data)
print(report)
```

**Common Solutions:**

1. **Enable Native Extensions:**
   ```python
   # Check if native extensions are loaded
   if not encoder.has_native_support():
       print("Installing native extensions...")
       install_native_extensions()
   ```

2. **Optimize Memory Pool Size:**
   ```python
   # Increase memory pool for large datasets
   encoder.configure_memory_pool(size_mb=128)
   ```

3. **Adjust Thread Pool Size:**
   ```python
   # Optimize for your CPU count
   optimal_workers = min(32, os.cpu_count() * 2)
   encoder.set_worker_count(optimal_workers)
   ```

#### Issue 2: Encoding Inconsistencies

**Symptoms:**
- Different results for same input
- Position-dependent variations unexpected
- Decoding failures

**Diagnostic Steps:**

```python
class ConsistencyDiagnostic:
    """Diagnostic tools for encoding consistency issues"""
    
    def test_encoding_consistency(self, encoder, test_cases: list):
        """Test encoding consistency across multiple runs"""
        
        results = {}
        
        for i, test_data in enumerate(test_cases):
            case_results = []
            
            # Run same encoding multiple times
            for run in range(5):
                try:
                    encoded = encoder.encode(test_data, position=0)
                    case_results.append(encoded)
                except Exception as e:
                    case_results.append(f"ERROR: {e}")
            
            # Check consistency
            unique_results = set(case_results)
            results[f"case_{i}"] = {
                'consistent': len(unique_results) == 1,
                'results': case_results,
                'unique_count': len(unique_results)
            }
        
        return results
    
    def test_position_consistency(self, encoder, data: bytes):
        """Test position-dependent behavior"""
        
        position_tests = {}
        
        for position in [0, 1, 2, 3, 63, 64, 65, 127, 128]:
            try:
                encoded = encoder.encode(data, position=position)
                decoded = encoder.decode(encoded, position=position)
                
                position_tests[position] = {
                    'encoded': encoded,
                    'roundtrip_success': decoded == data,
                    'encoded_length': len(encoded)
                }
            except Exception as e:
                position_tests[position] = {'error': str(e)}
        
        return position_tests
```

**Solutions:**

1. **Verify Position Parameter Usage:**
   ```python
   # Ensure consistent position usage
   position = calculate_global_position(chunk_index, chunk_size)
   encoded = encoder.encode(data, position=position)
   ```

2. **Check Thread Safety:**
   ```python
   # Use thread-safe encoder for concurrent access
   encoder = ThreadSafeEncoder(base_encoder)
   ```

#### Issue 3: Memory Leaks

**Symptoms:**
- Memory usage continuously increasing
- Out of memory errors in long-running processes
- Slow garbage collection

**Diagnostic Steps:**

```python
import gc
import tracemalloc
import weakref

class MemoryLeakDetector:
    """Detect and analyze memory leaks in QuadB64 operations"""
    
    def __init__(self):
        self.snapshots = []
        self.tracked_objects = []
    
    def start_tracking(self):
        """Start memory leak tracking"""
        tracemalloc.start()
        gc.collect()  # Clean start
        self.snapshots.append(tracemalloc.take_snapshot())
    
    def check_leaks(self, operation_name: str = ""):
        """Check for memory leaks since last snapshot"""
        gc.collect()  # Force garbage collection
        
        current_snapshot = tracemalloc.take_snapshot()
        self.snapshots.append(current_snapshot)
        
        if len(self.snapshots) >= 2:
            top_stats = current_snapshot.compare_to(
                self.snapshots[-2], 'lineno'
            )
            
            print(f"Memory diff for {operation_name}:")
            for stat in top_stats[:10]:
                print(stat)
    
    def track_object(self, obj):
        """Track specific object for garbage collection"""
        weak_ref = weakref.ref(obj, lambda x: print(f"Object {id(obj)} collected"))
        self.tracked_objects.append(weak_ref)

# Usage
leak_detector = MemoryLeakDetector()
encoder = SomeQuadB64Encoder()

leak_detector.start_tracking()

# Perform operations
for i in range(1000):
    data = b"test data" * 100
    encoded = encoder.encode(data)
    
    if i % 100 == 0:
        leak_detector.check_leaks(f"iteration_{i}")
```

**Solutions:**

1. **Proper Cache Management:**
   ```python
   # Configure cache with appropriate limits
   encoder.configure_cache(max_size=10000, ttl_seconds=300)
   ```

2. **Explicit Cleanup:**
   ```python
   # Periodically clean up resources
   if iteration_count % 1000 == 0:
       encoder.cleanup_caches()
       gc.collect()
   ```

### Debugging Techniques

#### Enable Detailed Logging

```python
import logging

# Configure QuadB64 logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('uubed')

# Enable detailed operation logging
encoder.enable_debug_logging(logger)

# Add custom log handlers
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
```

#### Step-by-Step Encoding Analysis

```python
class StepByStepDebugger:
    """Debug QuadB64 encoding step by step"""
    
    def debug_encode(self, data: bytes, position: int = 0):
        """Debug encoding process step by step"""
        
        print(f"Debugging encoding of {len(data)} bytes at position {position}")
        print(f"Input data: {data[:50]}..." if len(data) > 50 else f"Input data: {data}")
        print()
        
        # Step 1: Alphabet generation
        alphabet = self._debug_alphabet_generation(position)
        
        # Step 2: Data chunking
        chunks = self._debug_data_chunking(data)
        
        # Step 3: Chunk encoding
        encoded_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_position = position + i * 3
            encoded_chunk = self._debug_chunk_encoding(chunk, chunk_position, alphabet)
            encoded_chunks.append(encoded_chunk)
        
        # Step 4: Final assembly
        result = ''.join(encoded_chunks)
        print(f"Final result: {result}")
        
        return result
    
    def _debug_alphabet_generation(self, position: int) -> str:
        """Debug alphabet generation"""
        base_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        rotation = (position // 3) % 64
        
        alphabet = base_alphabet[rotation:] + base_alphabet[:rotation]
        
        print(f"Alphabet generation:")
        print(f"  Position: {position}")
        print(f"  Rotation: {rotation}")
        print(f"  Base:     {base_alphabet}")
        print(f"  Rotated:  {alphabet}")
        print()
        
        return alphabet
    
    def _debug_data_chunking(self, data: bytes) -> list:
        """Debug data chunking process"""
        chunks = [data[i:i+3] for i in range(0, len(data), 3)]
        
        print(f"Data chunking:")
        print(f"  Total bytes: {len(data)}")
        print(f"  Chunks: {len(chunks)}")
        
        for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
            print(f"  Chunk {i}: {chunk} ({[hex(b) for b in chunk]})")
        
        if len(chunks) > 5:
            print(f"  ... and {len(chunks) - 5} more chunks")
        print()
        
        return chunks
    
    def _debug_chunk_encoding(self, chunk: bytes, position: int, alphabet: str) -> str:
        """Debug individual chunk encoding"""
        # Pad chunk to 3 bytes
        padded_chunk = chunk + b'\x00' * (3 - len(chunk))
        
        # Convert to 24-bit integer
        value = (padded_chunk[0] << 16) | (padded_chunk[1] << 8) | padded_chunk[2]
        
        # Extract 6-bit indices
        indices = [
            (value >> 18) & 0x3F,
            (value >> 12) & 0x3F,
            (value >> 6) & 0x3F,
            value & 0x3F
        ]
        
        # Map to alphabet
        chars = [alphabet[idx] for idx in indices]
        result = ''.join(chars[:len(chunk)+1])
        
        print(f"Chunk encoding at position {position}:")
        print(f"  Input: {chunk} -> {[hex(b) for b in padded_chunk]}")
        print(f"  24-bit value: {value:024b} ({value})")
        print(f"  6-bit indices: {indices}")
        print(f"  Characters: {chars} -> '{result}'")
        print()
        
        return result

# Usage
debugger = StepByStepDebugger()
test_data = b"Hello, QuadB64!"
result = debugger.debug_encode(test_data, position=5)
```

This comprehensive advanced features guide provides power users with the tools and techniques needed to optimize QuadB64 for specific use cases, troubleshoot issues, and implement custom solutions for complex requirements.