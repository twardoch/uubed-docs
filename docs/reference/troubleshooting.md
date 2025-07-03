---
layout: default
title: "Troubleshooting"
parent: "Reference"
nav_order: 3
description: "Common issues and solutions for QuadB64 implementation and usage"
---

# Troubleshooting Guide

## Quick Diagnostic Checklist

If you're experiencing issues with QuadB64, start with this quick checklist:

### Installation Issues
- [ ] Python version 3.8+ installed
- [ ] Package installed via `pip install uubed` 
- [ ] Native extensions compiled successfully
- [ ] All dependencies resolved

### Runtime Issues  
- [ ] Input data is valid bytes object
- [ ] Position parameter used consistently
- [ ] Sufficient memory available
- [ ] No concurrent access without thread safety

### Performance Issues
- [ ] Native extensions enabled
- [ ] Appropriate batch size configured
- [ ] Memory pool properly sized
- [ ] CPU/thread count optimized

## Common Issues and Solutions

### Issue 1: Import Errors

#### Problem: "No module named 'uubed'"

```python
ImportError: No module named 'uubed'
```

**Solution:**
```bash
# Install the package
pip install uubed

# Verify installation
python -c "import uubed; print(uubed.__version__)"

# If still failing, check your Python environment
which python
pip list | grep uubed
```

#### Problem: "Failed to load native extension"

```python
RuntimeError: Failed to load native extension for QuadB64
```

**Diagnosis:**
```python
import uubed
print(f"Native support available: {uubed.has_native_support()}")
print(f"Available variants: {uubed.get_available_variants()}")

# Check system requirements
import platform
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.machine()}")
```

**Solutions:**

1. **Install development tools:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential python3-dev
   
   # CentOS/RHEL
   sudo yum groupinstall "Development Tools"
   sudo yum install python3-devel
   
   # macOS
   xcode-select --install
   
   # Windows
   # Install Visual Studio Build Tools
   ```

2. **Force rebuild:**
   ```bash
   pip uninstall uubed
   pip install --no-cache-dir --force-reinstall uubed
   ```

3. **Use Python fallback:**
   ```python
   # Temporary workaround
   import uubed
   uubed.config.force_python_implementation = True
   ```

### Issue 2: Encoding/Decoding Errors

#### Problem: "Invalid input data"

```python
TypeError: Input must be bytes or bytearray, got str
```

**Solution:**
```python
# Convert string to bytes first
text = "Hello, World!"
data = text.encode('utf-8')  # Convert to bytes
encoded = uubed.encode_eq64(data)

# Or use the text encoding helper
encoded = uubed.encode_text(text, encoding='utf-8')
```

#### Problem: Position-dependent decoding failures

```python
ValueError: Decoding failed - position mismatch
```

**Diagnosis:**
```python
def diagnose_position_issue(data, encoded_result, position):
    """Diagnose position-related encoding issues"""
    
    print(f"Original data: {data}")
    print(f"Position used: {position}")
    print(f"Encoded result: {encoded_result}")
    
    # Test roundtrip
    try:
        decoded = uubed.decode_eq64(encoded_result, position=position)
        print(f"Roundtrip successful: {decoded == data}")
        if decoded != data:
            print(f"Decoded result: {decoded}")
            print(f"Difference: {set(data) - set(decoded)}")
    except Exception as e:
        print(f"Roundtrip failed: {e}")
    
    # Test with different positions
    print("\nTesting nearby positions:")
    for test_pos in [position-1, position, position+1]:
        try:
            test_decoded = uubed.decode_eq64(encoded_result, position=test_pos)
            print(f"Position {test_pos}: {'✓' if test_decoded == data else '✗'}")
        except:
            print(f"Position {test_pos}: ERROR")

# Usage
data = b"test data"
position = 42
encoded = uubed.encode_eq64(data, position=position)
diagnose_position_issue(data, encoded, position)
```

**Solution:**
```python
# Ensure consistent position usage
def safe_encode_decode(data, position=0):
    """Safe encoding with position tracking"""
    
    # Store position with encoded data
    encoded = uubed.encode_eq64(data, position=position)
    
    # For storage, include position information
    stored_data = {
        'encoded': encoded,
        'position': position,
        'checksum': hash(data)  # For verification
    }
    
    return stored_data

def safe_decode(stored_data):
    """Safe decoding with position validation"""
    
    decoded = uubed.decode_eq64(
        stored_data['encoded'], 
        position=stored_data['position']
    )
    
    # Verify checksum
    if hash(decoded) != stored_data['checksum']:
        raise ValueError("Decoded data checksum mismatch")
    
    return decoded
```

### Issue 3: Performance Problems

#### Problem: Slow encoding performance

**Diagnosis:**
```python
import time
import uubed

def benchmark_encoding_performance():
    """Benchmark encoding performance and identify bottlenecks"""
    
    test_sizes = [1024, 10240, 102400, 1048576]  # 1KB to 1MB
    results = {}
    
    for size in test_sizes:
        data = b"x" * size
        
        # Time multiple runs
        times = []
        for _ in range(10):
            start = time.perf_counter()
            encoded = uubed.encode_eq64(data)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = sum(times) / len(times)
        throughput = size / avg_time / 1024 / 1024  # MB/s
        
        results[size] = {
            'avg_time_ms': avg_time * 1000,
            'throughput_mb_s': throughput,
            'encoded_size': len(encoded)
        }
        
        print(f"Size: {size:>8} bytes | "
              f"Time: {avg_time*1000:>6.2f} ms | "
              f"Throughput: {throughput:>6.2f} MB/s")
    
    return results

# Run benchmark
print("Performance Benchmark:")
benchmark_encoding_performance()

# Check native support
print(f"\nNative extensions: {uubed.has_native_support()}")
print(f"SIMD support: {uubed.has_simd_support()}")
```

**Solutions:**

1. **Enable native extensions:**
   ```python
   # Verify native support is enabled
   if not uubed.has_native_support():
       print("Native extensions not available - reinstalling...")
       # See installation troubleshooting above
   ```

2. **Optimize for your use case:**
   ```python
   # For many small encodings - use batch processing
   def batch_encode(data_list, batch_size=100):
       results = []
       for i in range(0, len(data_list), batch_size):
           batch = data_list[i:i+batch_size]
           batch_results = uubed.encode_batch_eq64(batch)
           results.extend(batch_results)
       return results
   
   # For streaming data - use streaming encoder
   encoder = uubed.StreamingEncoder(buffer_size=8192)
   for chunk in data_stream:
       encoded_chunk = encoder.encode_chunk(chunk)
       # Process encoded_chunk
   ```

3. **Configure memory pool:**
   ```python
   # Increase memory pool for large datasets
   uubed.config.memory_pool_size = 64 * 1024 * 1024  # 64MB
   uubed.config.enable_memory_pool = True
   ```

#### Problem: Memory usage growing over time

**Diagnosis:**
```python
import gc
import tracemalloc

def diagnose_memory_usage():
    """Diagnose memory usage patterns"""
    
    tracemalloc.start()
    
    # Simulate workload
    for i in range(1000):
        data = b"test data" * 100
        encoded = uubed.encode_eq64(data)
        
        if i % 100 == 0:
            current, peak = tracemalloc.get_traced_memory()
            print(f"Iteration {i}: Current={current/1024/1024:.1f}MB, "
                  f"Peak={peak/1024/1024:.1f}MB")
            
            # Force garbage collection
            collected = gc.collect()
            print(f"  Garbage collected: {collected} objects")
    
    tracemalloc.stop()

diagnose_memory_usage()
```

**Solutions:**
```python
# Configure cache limits
uubed.config.cache_size_limit = 10000  # Maximum cached items
uubed.config.cache_ttl_seconds = 300   # 5 minute TTL

# Periodic cleanup
def periodic_cleanup():
    uubed.clear_caches()
    gc.collect()

# Call periodically in long-running processes
```

### Issue 4: Thread Safety Issues

#### Problem: Inconsistent results in multi-threaded code

```python
# Problematic code
import threading
import uubed

results = []

def worker(data):
    encoded = uubed.encode_eq64(data)  # Not thread-safe
    results.append(encoded)

threads = [threading.Thread(target=worker, args=(b"data",)) for _ in range(10)]
```

**Solution:**
```python
import threading
import uubed
from concurrent.futures import ThreadPoolExecutor

# Thread-safe approach 1: Use thread-local encoders
thread_local = threading.local()

def get_thread_encoder():
    if not hasattr(thread_local, 'encoder'):
        thread_local.encoder = uubed.ThreadSafeEncoder()
    return thread_local.encoder

def worker(data):
    encoder = get_thread_encoder()
    return encoder.encode_eq64(data)

# Thread-safe approach 2: Use process pool for isolation  
def process_data(data_list):
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Each thread gets its own encoder instance
        futures = [
            executor.submit(uubed.encode_eq64, data) 
            for data in data_list
        ]
        results = [future.result() for future in futures]
    return results
```

### Issue 5: Integration Issues

#### Problem: Database storage encoding issues

```python
# Problematic approach
encoded = uubed.encode_eq64(data)
cursor.execute("INSERT INTO table (data) VALUES (?)", (encoded,))
# May cause encoding issues depending on database
```

**Solution:**
```python
import base64
import json

# Safe database storage
def store_encoded_data(cursor, data, position=0):
    """Safely store QuadB64-encoded data in database"""
    
    # Encode with QuadB64
    quad_encoded = uubed.encode_eq64(data, position=position)
    
    # Create storage record
    storage_record = {
        'data': quad_encoded,
        'position': position,
        'encoding': 'quadb64_eq64',
        'checksum': hash(data)
    }
    
    # Store as JSON for database compatibility
    json_data = json.dumps(storage_record)
    
    cursor.execute(
        "INSERT INTO encoded_data (record) VALUES (?)", 
        (json_data,)
    )

def retrieve_encoded_data(cursor, record_id):
    """Safely retrieve QuadB64-encoded data from database"""
    
    cursor.execute(
        "SELECT record FROM encoded_data WHERE id = ?", 
        (record_id,)
    )
    
    result = cursor.fetchone()
    if not result:
        raise ValueError(f"No record found with id {record_id}")
    
    storage_record = json.loads(result[0])
    
    # Decode data
    decoded = uubed.decode_eq64(
        storage_record['data'],
        position=storage_record['position']
    )
    
    # Verify checksum
    if hash(decoded) != storage_record['checksum']:
        raise ValueError("Data integrity check failed")
    
    return decoded
```

#### Problem: Web API encoding issues

```python
# Problematic approach - binary data in JSON
import json

data = b"binary data"
encoded = uubed.encode_eq64(data)
response = json.dumps({"data": encoded})  # May have encoding issues
```

**Solution:**
```python
import json
import uubed

# Safe web API approach
def create_api_response(data, position=0):
    """Create web API response with QuadB64 data"""
    
    # Encode data
    encoded = uubed.encode_eq64(data, position=position)
    
    # Create safe response
    response = {
        'data': encoded,
        'encoding': 'quadb64_eq64',
        'position': position,
        'metadata': {
            'original_size': len(data),
            'encoded_size': len(encoded),
            'timestamp': time.time()
        }
    }
    
    # JSON is safe with QuadB64 (text-based)
    return json.dumps(response)

def parse_api_response(json_response):
    """Parse web API response with QuadB64 data"""
    
    response = json.loads(json_response)
    
    # Validate response format
    required_fields = ['data', 'encoding', 'position']
    for field in required_fields:
        if field not in response:
            raise ValueError(f"Missing required field: {field}")
    
    if response['encoding'] != 'quadb64_eq64':
        raise ValueError(f"Unsupported encoding: {response['encoding']}")
    
    # Decode data
    decoded = uubed.decode_eq64(
        response['data'],
        position=response['position']
    )
    
    return decoded
```

## Advanced Debugging

### Enable Debug Logging

```python
import logging
import uubed

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('uubed')

# Enable QuadB64 debug mode
uubed.enable_debug_mode(True)

# Add custom handler
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Now operations will show detailed logging
data = b"debug test"
encoded = uubed.encode_eq64(data)  # Will show debug info
```

### Step-by-Step Encoding Analysis

```python
def debug_encoding_step_by_step(data, position=0):
    """Debug encoding process in detail"""
    
    print(f"=== Debug Encoding Analysis ===")
    print(f"Input: {data} (length: {len(data)})")
    print(f"Position: {position}")
    print()
    
    # Step 1: Show alphabet generation
    alphabet = uubed.debug.get_alphabet_for_position(position)
    rotation = (position // 3) % 64
    print(f"Alphabet rotation: {rotation}")
    print(f"Alphabet: {alphabet}")
    print()
    
    # Step 2: Show chunking
    chunks = [data[i:i+3] for i in range(0, len(data), 3)]
    print(f"Data chunks ({len(chunks)} total):")
    for i, chunk in enumerate(chunks):
        chunk_pos = position + i * 3
        print(f"  Chunk {i}: {chunk} at position {chunk_pos}")
    print()
    
    # Step 3: Show encoding process
    encoded_parts = []
    for i, chunk in enumerate(chunks):
        chunk_pos = position + i * 3
        encoded_chunk = uubed.debug.encode_chunk_verbose(chunk, chunk_pos)
        encoded_parts.append(encoded_chunk['result'])
        
        print(f"Chunk {i} encoding:")
        print(f"  Input bytes: {chunk}")
        print(f"  Hex values: {[hex(b) for b in chunk]}")
        print(f"  24-bit value: {encoded_chunk['value']:024b}")
        print(f"  6-bit indices: {encoded_chunk['indices']}")
        print(f"  Output chars: {encoded_chunk['chars']}")
        print(f"  Result: '{encoded_chunk['result']}'")
        print()
    
    final_result = ''.join(encoded_parts)
    print(f"Final encoded result: '{final_result}'")
    
    return final_result

# Usage
test_data = b"Hello!"
result = debug_encoding_step_by_step(test_data, position=5)
```

### Performance Profiling

```python
import cProfile
import pstats
import io

def profile_encoding_operation(data_list):
    """Profile encoding operation for performance analysis"""
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run encoding operations
    results = []
    for data in data_list:
        encoded = uubed.encode_eq64(data)
        results.append(encoded)
    
    profiler.disable()
    
    # Generate profile report
    profile_output = io.StringIO()
    stats = pstats.Stats(profiler, stream=profile_output)
    stats.sort_stats('cumulative')
    stats.print_stats()
    
    print("Performance Profile:")
    print(profile_output.getvalue())
    
    return results

# Generate test data
test_data = [b"sample data" * 100 for _ in range(100)]
profile_encoding_operation(test_data)
```

## Getting Help

### Community Resources

1. **GitHub Issues**: [Report bugs and request features](https://github.com/twardoch/uubed/issues)
2. **Documentation**: [Complete documentation](https://uubed.readthedocs.io)
3. **Examples**: [Code examples repository](https://github.com/twardoch/uubed-examples)

### Creating Bug Reports

When reporting issues, include:

```python
import uubed
import platform
import sys

def generate_bug_report():
    """Generate comprehensive bug report information"""
    
    report = {
        'uubed_version': uubed.__version__,
        'python_version': sys.version,
        'platform': platform.platform(),
        'architecture': platform.machine(),
        'native_support': uubed.has_native_support(),
        'simd_support': uubed.has_simd_support(),
        'available_variants': uubed.get_available_variants()
    }
    
    print("=== Bug Report Information ===")
    for key, value in report.items():
        print(f"{key}: {value}")
    
    return report

# Include this information in bug reports
generate_bug_report()
```

### Performance Baseline

Use this baseline test to compare performance:

```python
import time
import uubed

def run_performance_baseline():
    """Standard performance baseline test"""
    
    # Test data
    small_data = b"Hello, World!" * 10
    medium_data = b"x" * 10240  # 10KB
    large_data = b"x" * 1048576  # 1MB
    
    tests = [
        ("Small (130B)", small_data),
        ("Medium (10KB)", medium_data), 
        ("Large (1MB)", large_data)
    ]
    
    print("=== Performance Baseline ===")
    print(f"QuadB64 version: {uubed.__version__}")
    print(f"Native extensions: {uubed.has_native_support()}")
    print()
    
    for name, data in tests:
        # Warm up
        for _ in range(10):
            uubed.encode_eq64(data)
        
        # Measure
        times = []
        for _ in range(100):
            start = time.perf_counter()
            encoded = uubed.encode_eq64(data)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = sum(times) / len(times)
        throughput = len(data) / avg_time / 1024 / 1024
        
        print(f"{name:>12}: {avg_time*1000:>6.2f} ms | {throughput:>6.2f} MB/s")

run_performance_baseline()
```

Include this baseline output when reporting performance issues.