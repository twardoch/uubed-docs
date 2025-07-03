---
layout: default
title: Performance Tuning
parent: Performance
nav_order: 2
description: "Comprehensive strategies to maximize QuadB64 performance in production environments with native extensions, batch processing, and optimization techniques."
---

> This guide is your ultimate cheat sheet for making QuadB64 run at lightning speed. It covers everything from installing the right bits to fine-tuning your code, ensuring your data encoding is as fast as a cheetah on a caffeine high.

# Performance Tuning Guide

## Overview

Imagine you're a Formula 1 race engineer, and QuadB64 is your high-performance engine. This guide provides you with all the advanced techniques and secret tweaks to push that engine to its absolute limits, ensuring it wins every data processing race.

Imagine you're a master blacksmith, and QuadB64 is the finest steel. This guide teaches you the ancient art of tempering, forging, and sharpening that steel to create an encoding blade that cuts through data with unparalleled speed and precision.

This guide provides comprehensive strategies to maximize QuadB64 performance in production environments. Whether you're processing millions of embeddings or optimizing real-time encoding, these techniques will help you achieve optimal throughput and efficiency.

## Quick Performance Checklist

Before diving into detailed optimizations, ensure these basics are covered:

- ✅ **Native Extensions**: Install with `pip install uubed[native]`
- ✅ **Verify Installation**: Check `has_native_extensions()` returns `True`
- ✅ **Batch Processing**: Process multiple items together when possible
- ✅ **Appropriate Variant**: Choose the right encoding for your use case
- ✅ **Hardware**: Use modern CPUs with SIMD support

## Native Extension Optimization

### Installation and Verification

```python
from uubed import has_native_extensions, get_implementation_info

# Check native status
if has_native_extensions():
    print("🚀 Native acceleration enabled")
    info = get_implementation_info()
    print(f"Implementation: {info['implementation']}")
    print(f"SIMD features: {info['simd_features']}")
else:
    print("❌ Using pure Python - install native extensions")
    print("Run: pip install uubed[native]")
```

### Performance Impact

| Operation | Pure Python | Native | Speedup |
|-----------|-------------|--------|---------|
| Eq64 (1MB) | 182ms | 4.3ms | **42x** |
| Shq64 (1MB) | 85ms | 8.5ms | **10x** |
| T8q64 (1MB) | 127ms | 6.4ms | **20x** |
| Zoq64 (1MB) | 3333ms | 2.1ms | **1587x** |

### Troubleshooting Native Extensions

```python
# If native extensions fail to load
import uubed
import sys

print("Python version:", sys.version)
print("Platform:", sys.platform)
print("Native available:", uubed.has_native_extensions())

# Check for common issues
try:
    from uubed._native import core
    print("Native module loaded successfully")
except ImportError as e:
    print(f"Native module failed: {e}")
    print("Solution: Reinstall with: pip install --force-reinstall uubed[native]")
```

## Batch Processing Strategies

### Optimal Batch Sizes

```python
from uubed import encode_batch, Config
import numpy as np

# Test different batch sizes to find optimal
def find_optimal_batch_size(data_samples, max_batch_size=1000):
    results = {}
    
    for batch_size in [10, 50, 100, 500, 1000]:
        if batch_size > len(data_samples):
            continue
            
        import time
        batch = data_samples[:batch_size]
        
        start = time.time()
        encoded = encode_batch(batch, method="eq64")
        duration = time.time() - start
        
        throughput = len(batch) / duration
        results[batch_size] = throughput
        
    optimal = max(results, key=results.get)
    print(f"Optimal batch size: {optimal} (throughput: {results[optimal]:.1f} items/sec)")
    return optimal

# Usage
embeddings = [np.random.rand(768).astype(np.float32) for _ in range(500)]
embedding_bytes = [emb.tobytes() for emb in embeddings]
optimal_size = find_optimal_batch_size(embedding_bytes)
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from uubed import encode_eq64
import multiprocessing

def parallel_encode_process(data_chunks, num_workers=None):
    """CPU-bound parallel processing with processes"""
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    def encode_chunk(chunk):
        return [encode_eq64(item) for item in chunk]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(encode_chunk, data_chunks))
    
    # Flatten results
    return [item for chunk in results for item in chunk]

def parallel_encode_thread(data_chunks, num_workers=4):
    """I/O-bound parallel processing with threads"""
    def encode_chunk(chunk):
        return [encode_eq64(item) for item in chunk]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(encode_chunk, data_chunks))
    
    return [item for chunk in results for item in chunk]

# Usage example
large_dataset = [b"data" + str(i).encode() for i in range(10000)]

# Split into chunks
chunk_size = 250
chunks = [large_dataset[i:i+chunk_size] 
          for i in range(0, len(large_dataset), chunk_size)]

# Process in parallel
encoded_parallel = parallel_encode_process(chunks)
```

### Streaming for Large Datasets

```python
from uubed import StreamEncoder

class HighThroughputProcessor:
    def __init__(self, method="eq64", buffer_size=8192):
        self.encoder = StreamEncoder(method)
        self.buffer_size = buffer_size
        
    def process_file(self, input_path, output_path):
        """Process large files with minimal memory usage"""
        with open(input_path, "rb") as infile:
            with open(output_path, "w") as outfile:
                
                while True:
                    chunk = infile.read(self.buffer_size)
                    if not chunk:
                        break
                    
                    # Encode chunk
                    encoded = self.encoder.encode_chunk(chunk)
                    outfile.write(encoded + "\n")
                
                # Finalize
                final = self.encoder.finalize()
                if final:
                    outfile.write(final + "\n")

# Usage
processor = HighThroughputProcessor(buffer_size=64*1024)  # 64KB chunks
processor.process_file("large_embeddings.bin", "encoded.eq64")
```

## Memory Optimization

### Memory-Efficient Patterns

```python
import gc
from uubed import encode_eq64

def memory_efficient_batch_encode(data_generator, batch_size=100):
    """Process data without loading everything into memory"""
    results = []
    batch = []
    
    for item in data_generator:
        batch.append(item)
        
        if len(batch) >= batch_size:
            # Process batch
            encoded_batch = [encode_eq64(item) for item in batch]
            results.extend(encoded_batch)
            
            # Clear batch and force garbage collection
            batch.clear()
            gc.collect()
    
    # Process remaining items
    if batch:
        encoded_batch = [encode_eq64(item) for item in batch]
        results.extend(encoded_batch)
    
    return results

# Generator for large datasets
def embedding_generator(num_embeddings):
    """Generate embeddings on-demand to save memory"""
    for i in range(num_embeddings):
        # Generate embedding (could be from model, file, etc.)
        embedding = np.random.rand(768).astype(np.float32)
        yield embedding.tobytes()

# Process 1 million embeddings with low memory usage
encoded = memory_efficient_batch_encode(
    embedding_generator(1_000_000), 
    batch_size=500
)
```

### Memory Monitoring

```python
import psutil
import os

def monitor_memory_usage(func, *args, **kwargs):
    """Monitor memory usage during function execution"""
    process = psutil.Process(os.getpid())
    
    # Initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory: {initial_memory:.1f} MB")
    
    # Execute function
    result = func(*args, **kwargs)
    
    # Final memory
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    peak_memory = process.memory_info().peak_wss / 1024 / 1024 if hasattr(process.memory_info(), 'peak_wss') else final_memory
    
    print(f"Final memory: {final_memory:.1f} MB")
    print(f"Memory increase: {final_memory - initial_memory:.1f} MB")
    
    return result

# Usage
def encode_large_batch():
    data = [b"test" * 1000 for _ in range(10000)]
    return [encode_eq64(item) for item in data]

result = monitor_memory_usage(encode_large_batch)
```

## Platform-Specific Optimizations

### CPU Architecture Optimization

```python
from uubed import get_cpu_features, Config

def optimize_for_cpu():
    """Configure QuadB64 for optimal CPU performance"""
    features = get_cpu_features()
    
    config = Config()
    
    # SIMD optimizations
    if 'avx512' in features:
        config.simd_level = 'avx512'
        config.chunk_size = 8192  # Larger chunks for AVX-512
    elif 'avx2' in features:
        config.simd_level = 'avx2'
        config.chunk_size = 4096
    elif 'sse4_2' in features:
        config.simd_level = 'sse4_2'
        config.chunk_size = 2048
    else:
        config.simd_level = 'none'
        config.chunk_size = 1024
    
    # Thread configuration
    import multiprocessing
    config.num_threads = min(multiprocessing.cpu_count(), 8)
    
    print(f"Optimized for CPU with {features}")
    print(f"Configuration: {config}")
    
    return config

# Apply optimization
optimized_config = optimize_for_cpu()
```

### ARM vs x86 Considerations

```python
import platform

def get_platform_config():
    """Get platform-optimized configuration"""
    arch = platform.machine().lower()
    
    if arch in ['arm64', 'aarch64']:
        # ARM optimization
        return Config(
            chunk_size=4096,  # ARM prefers smaller chunks
            num_threads=4,     # ARM cores are often more numerous
            use_neon=True      # ARM SIMD
        )
    elif arch in ['x86_64', 'amd64']:
        # x86 optimization
        return Config(
            chunk_size=8192,   # x86 handles larger chunks well
            num_threads=min(8, multiprocessing.cpu_count()),
            use_avx=True       # x86 SIMD
        )
    else:
        # Generic configuration
        return Config()

config = get_platform_config()
```

## Variant-Specific Optimizations

### Eq64 Optimization

```python
from uubed import Eq64Encoder

# For high-throughput Eq64 encoding
class OptimizedEq64Encoder:
    def __init__(self):
        self.config = Config(
            chunk_size=8192,        # Large chunks for better throughput
            validate_input=False,   # Skip validation for trusted input
            use_native=True,        # Always use native if available
            parallel_threshold=1024 # Parallelize for inputs > 1KB
        )
        self.encoder = Eq64Encoder(self.config)
    
    def encode_batch_optimized(self, data_list):
        """Optimized batch encoding for Eq64"""
        # Pre-allocate result list
        results = [None] * len(data_list)
        
        # Process in optimal batch sizes
        batch_size = 100
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            batch_results = self.encoder.encode_batch(batch)
            results[i:i+len(batch_results)] = batch_results
        
        return results

# Usage
encoder = OptimizedEq64Encoder()
large_dataset = [np.random.bytes(3072) for _ in range(10000)]  # 768-dim float32
encoded = encoder.encode_batch_optimized(large_dataset)
```

### Shq64 Optimization

```python
from uubed import Shq64Encoder

# Optimize for similarity hashing
class OptimizedShq64Encoder:
    def __init__(self, num_bits=64):
        self.config = Config(
            shingle_size=4,         # Optimal for most text
            num_planes=num_bits,    # 64-bit hashes
            use_vectorized=True,    # Vectorized operations
            cache_features=True     # Cache feature extraction
        )
        self.encoder = Shq64Encoder(self.config)
    
    def encode_embeddings(self, embeddings):
        """Optimized for ML embeddings"""
        # Convert to optimal format
        if hasattr(embeddings[0], 'tobytes'):
            byte_data = [emb.tobytes() for emb in embeddings]
        else:
            byte_data = embeddings
        
        return self.encoder.encode_batch(byte_data)

# Usage for high-volume similarity hashing
encoder = OptimizedShq64Encoder()
embeddings = [model.encode(text) for text in documents]
hashes = encoder.encode_embeddings(embeddings)
```

## Database Integration Optimization

### Connection Pool Management

```python
import psycopg2.pool
from uubed import encode_eq64

class OptimizedDBInserter:
    def __init__(self, db_config, pool_size=10):
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=pool_size,
            **db_config
        )
    
    def batch_insert_embeddings(self, embeddings_data, batch_size=1000):
        """Optimized batch insertion with QuadB64 encoding"""
        conn = self.pool.getconn()
        try:
            cursor = conn.cursor()
            
            # Prepare batch insert
            insert_query = """
                INSERT INTO embeddings (id, vector_eq64, metadata)
                VALUES %s
            """
            
            # Process in batches
            for i in range(0, len(embeddings_data), batch_size):
                batch = embeddings_data[i:i+batch_size]
                
                # Encode batch
                encoded_batch = [
                    (item['id'], encode_eq64(item['vector']), item['metadata'])
                    for item in batch
                ]
                
                # Bulk insert
                psycopg2.extras.execute_values(
                    cursor, insert_query, encoded_batch,
                    template=None, page_size=batch_size
                )
            
            conn.commit()
            
        finally:
            self.pool.putconn(conn)

# Usage
db_config = {
    'host': 'localhost',
    'database': 'vectors',
    'user': 'user',
    'password': 'pass'
}

inserter = OptimizedDBInserter(db_config)
```

### Index Strategy

```sql
-- Optimized database schema for QuadB64
CREATE TABLE embeddings (
    id BIGSERIAL PRIMARY KEY,
    content TEXT,
    vector_eq64 TEXT NOT NULL,
    vector_shq64 CHAR(19),  -- Fixed length for better performance
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for different query patterns
CREATE INDEX idx_eq64_prefix ON embeddings(LEFT(vector_eq64, 16));  -- Prefix matching
CREATE INDEX idx_shq64_exact ON embeddings(vector_shq64);           -- Exact matching
CREATE INDEX idx_combined ON embeddings(vector_shq64, created_at);  -- Combined queries

-- Partial indexes for common patterns
CREATE INDEX idx_recent_shq64 ON embeddings(vector_shq64) 
WHERE created_at > NOW() - INTERVAL '30 days';
```

## Performance Monitoring

### Benchmarking Framework

```python
import time
import statistics
from uubed import encode_eq64, encode_shq64

class PerformanceBenchmark:
    def __init__(self):
        self.results = {}
    
    def benchmark_function(self, func, data, iterations=5, name=None):
        """Benchmark a function with multiple iterations"""
        if name is None:
            name = func.__name__
        
        times = []
        for i in range(iterations):
            start = time.perf_counter()
            result = func(data)
            end = time.perf_counter()
            times.append(end - start)
        
        # Calculate statistics
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        
        # Calculate throughput
        data_size = len(data) if hasattr(data, '__len__') else 1
        throughput = data_size / mean_time
        
        self.results[name] = {
            'mean_time': mean_time,
            'std_time': std_time,
            'throughput': throughput,
            'times': times
        }
        
        return result
    
    def print_results(self):
        """Print benchmark results"""
        print("\nPerformance Benchmark Results:")
        print("-" * 60)
        
        for name, stats in self.results.items():
            print(f"{name}:")
            print(f"  Mean time: {stats['mean_time']*1000:.2f}ms ± {stats['std_time']*1000:.2f}ms")
            print(f"  Throughput: {stats['throughput']:.1f} items/sec")
            print()

# Usage
benchmark = PerformanceBenchmark()

# Test data
test_embeddings = [np.random.rand(768).astype(np.float32).tobytes() for _ in range(1000)]

# Benchmark different variants
benchmark.benchmark_function(
    lambda data: [encode_eq64(item) for item in data],
    test_embeddings,
    name="Eq64_Batch"
)

benchmark.benchmark_function(
    lambda data: [encode_shq64(item) for item in data],
    test_embeddings,
    name="Shq64_Batch"
)

benchmark.print_results()
```

### Real-time Performance Monitoring

```python
import threading
import time
from collections import deque

class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.times = deque(maxlen=window_size)
        self.lock = threading.Lock()
        
    def record_operation(self, duration):
        """Record operation duration"""
        with self.lock:
            self.times.append(duration)
    
    def get_stats(self):
        """Get current performance statistics"""
        with self.lock:
            if not self.times:
                return None
            
            times_list = list(self.times)
            return {
                'count': len(times_list),
                'mean': statistics.mean(times_list),
                'median': statistics.median(times_list),
                'p95': statistics.quantiles(times_list, n=20)[18] if len(times_list) > 20 else max(times_list),
                'throughput': len(times_list) / sum(times_list) if sum(times_list) > 0 else 0
            }

# Global monitor instance
monitor = PerformanceMonitor()

def monitored_encode(data):
    """Encoding function with performance monitoring"""
    start = time.perf_counter()
    result = encode_eq64(data)
    duration = time.perf_counter() - start
    
    monitor.record_operation(duration)
    return result

# Monitoring thread
def print_stats_periodically():
    while True:
        time.sleep(10)  # Print stats every 10 seconds
        stats = monitor.get_stats()
        if stats:
            print(f"Performance: {stats['throughput']:.1f} ops/sec, "
                  f"P95: {stats['p95']*1000:.2f}ms")

# Start monitoring
monitor_thread = threading.Thread(target=print_stats_periodically, daemon=True)
monitor_thread.start()
```

## Troubleshooting Performance Issues

### Common Performance Problems

```python
def diagnose_performance_issues():
    """Automated performance diagnostics"""
    issues = []
    
    # Check native extensions
    if not has_native_extensions():
        issues.append({
            'issue': 'Native extensions not available',
            'impact': 'High (10-100x slower)',
            'solution': 'Install with: pip install uubed[native]'
        })
    
    # Check memory usage
    import psutil
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > 80:
        issues.append({
            'issue': f'High memory usage ({memory_percent:.1f}%)',
            'impact': 'Medium (may cause swapping)',
            'solution': 'Use streaming processing or smaller batches'
        })
    
    # Check CPU usage
    cpu_count = psutil.cpu_count()
    if cpu_count < 4:
        issues.append({
            'issue': f'Limited CPU cores ({cpu_count})',
            'impact': 'Medium (limits parallelization)',
            'solution': 'Use smaller batch sizes and fewer worker threads'
        })
    
    return issues

# Run diagnostics
issues = diagnose_performance_issues()
for issue in issues:
    print(f"⚠️  {issue['issue']}")
    print(f"   Impact: {issue['impact']}")
    print(f"   Solution: {issue['solution']}")
    print()
```

### Performance Regression Testing

```python
def regression_test():
    """Test for performance regressions"""
    baseline_times = {
        'eq64_1kb': 0.001,   # 1ms baseline
        'shq64_1kb': 0.0005, # 0.5ms baseline
    }
    
    # Test current performance
    test_data = b"x" * 1024  # 1KB test data
    
    # Eq64 test
    start = time.perf_counter()
    encode_eq64(test_data)
    eq64_time = time.perf_counter() - start
    
    # Shq64 test
    start = time.perf_counter()
    encode_shq64(test_data)
    shq64_time = time.perf_counter() - start
    
    # Check for regressions
    results = {}
    
    if eq64_time > baseline_times['eq64_1kb'] * 1.5:  # 50% slower
        results['eq64'] = 'REGRESSION'
    else:
        results['eq64'] = 'OK'
    
    if shq64_time > baseline_times['shq64_1kb'] * 1.5:
        results['shq64'] = 'REGRESSION'
    else:
        results['shq64'] = 'OK'
    
    return results

# Run regression test
regression_results = regression_test()
print("Performance regression test results:", regression_results)
```

## Summary

Key performance optimization strategies:

1. **Always use native extensions** - 10-100x performance improvement
2. **Batch process data** - Better throughput and resource utilization
3. **Choose appropriate variants** - Match encoding to use case
4. **Monitor memory usage** - Use streaming for large datasets
5. **Platform-specific tuning** - Optimize for your CPU architecture
6. **Database optimization** - Use proper indexing and connection pooling

Following these guidelines will ensure you get maximum performance from QuadB64 in production environments.