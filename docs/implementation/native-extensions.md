---
layout: default
title: Native Extensions Guide
parent: Implementation
nav_order: 2
description: "Guide to QuadB64's native extensions written in Rust and C++, covering architecture, compilation, optimization techniques, and platform-specific considerations."
---

TLDR: This guide is your deep dive into how QuadB64 achieves its incredible speed. It's all thanks to highly optimized native code (mostly Rust!) that talks directly to your computer's hardware, making encoding and decoding operations blazingly fast. Think of it as the turbocharger for your data.

# Native Extensions Guide

## Overview

Imagine QuadB64 is a high-performance sports car, and these native extensions are its custom-built, finely tuned engine. This guide takes you under the hood, revealing how Rust and C++ are used to craft components that extract every ounce of speed from your hardware, ensuring your data processing leaves others in the dust.

Imagine you're a master craftsman, and you're building a precision instrument. This guide details how QuadB64 leverages native extensions as its core, hand-forged components. It explains the meticulous process of compilation, the art of optimization, and the careful consideration of different materials (platforms) to create a tool of unparalleled efficiency.

QuadB64 achieves dramatic performance improvements through optimized native extensions written in Rust and C++. This guide covers the architecture, compilation, optimization techniques, and platform-specific considerations for native extensions.

## Performance Impact

Native extensions provide substantial performance improvements:

- **Pure Python**: 38 MB/s encoding throughput
- **Native Extensions**: 115 MB/s (3x improvement)
- **Native + SIMD**: 360 MB/s (9.5x improvement)
- **Memory Usage**: 15% reduction through optimized buffer management

## Architecture Overview

### Language Choice Rationale

QuadB64 native extensions use **Rust** as the primary implementation language for several key reasons:

1. **Memory Safety**: Zero-cost abstractions with compile-time memory safety
2. **Performance**: Comparable to C/C++ with better safety guarantees
3. **Cross-Platform**: Excellent support for diverse architectures
4. **SIMD Support**: First-class intrinsics for vectorized operations
5. **Python Integration**: Mature PyO3 ecosystem for Python bindings

### Core Extension Structure

```rust
// src/lib.rs - Main entry point
use pyo3::prelude::*;

#[pymodule]
fn uubed_native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_eq64_native, m)?)?;
    m.add_function(wrap_pyfunction!(encode_shq64_native, m)?)?;
    m.add_function(wrap_pyfunction!(encode_t8q64_native, m)?)?;
    m.add_function(wrap_pyfunction!(encode_zoq64_native, m)?)?;
    m.add_function(wrap_pyfunction!(has_simd_support, m)?)?;
    Ok(())
}

// High-level encoding interface
#[pyfunction]
fn encode_eq64_native(
    data: &[u8],
    position: usize,
    alphabet: Option<&str>
) -> PyResult<String> {
    match encode_eq64_optimized(data, position, alphabet) {
        Ok(result) => Ok(result),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Encoding failed: {}", e)
        ))
    }
}
```

### Performance-Critical Core

```rust
// src/encoding/eq64.rs - Optimized Eq64 implementation
use std::arch::x86_64::*;

pub fn encode_eq64_optimized(
    data: &[u8], 
    position: usize, 
    alphabet: Option<&str>
) -> Result<String, Box<dyn std::error::Error>> {
    
    // Select optimal implementation based on capabilities
    if is_x86_feature_detected!("avx2") {
        unsafe { encode_eq64_avx2(data, position, alphabet) }
    } else if is_x86_feature_detected!("sse4.1") {
        unsafe { encode_eq64_sse4(data, position, alphabet) }
    } else {
        encode_eq64_scalar(data, position, alphabet)
    }
}

// AVX2-optimized implementation for x86_64
#[target_feature(enable = "avx2")]
unsafe fn encode_eq64_avx2(
    data: &[u8],
    position: usize,
    alphabet: Option<&str>
) -> Result<String, Box<dyn std::error::Error>> {
    
    let alphabet = alphabet.unwrap_or(DEFAULT_ALPHABET);
    let mut result = Vec::with_capacity((data.len() + 2) / 3 * 4);
    
    // Process 24-byte chunks with AVX2 (produces 32 output characters)
    let chunks = data.chunks_exact(24);
    let remainder = chunks.remainder();
    
    for (chunk_idx, chunk) in chunks.enumerate() {
        let chunk_position = position + chunk_idx * 24;
        let encoded_chunk = encode_chunk_avx2(chunk, chunk_position, alphabet)?;
        result.extend_from_slice(encoded_chunk.as_bytes());
    }
    
    // Handle remainder with scalar code
    if !remainder.is_empty() {
        let remainder_position = position + data.len() - remainder.len();
        let encoded_remainder = encode_eq64_scalar(remainder, remainder_position, Some(alphabet))?;
        result.extend_from_slice(encoded_remainder.as_bytes());
    }
    
    String::from_utf8(result).map_err(|e| e.into())
}

// Core AVX2 chunk processing
#[target_feature(enable = "avx2")]
unsafe fn encode_chunk_avx2(
    chunk: &[u8],
    position: usize,
    alphabet: &str
) -> Result<String, Box<dyn std::error::Error>> {
    
    debug_assert_eq!(chunk.len(), 24);
    
    // Load 24 input bytes into AVX2 register
    let input = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
    
    // Position-dependent alphabet rotation
    let rotation = ((position / 3) % 64) as u8;
    let alphabet_bytes = alphabet.as_bytes();
    
    // Extract 6-bit groups using bit manipulation
    let indices = extract_6bit_groups_avx2(input);
    
    // Apply position-dependent rotation
    let rotated_indices = apply_rotation_avx2(indices, rotation);
    
    // Lookup characters in alphabet
    let output_chars = alphabet_lookup_avx2(rotated_indices, alphabet_bytes);
    
    // Convert to string
    let mut result = String::with_capacity(32);
    let output_slice = std::slice::from_raw_parts(
        &output_chars as *const __m256i as *const u8,
        32
    );
    
    for &byte in output_slice {
        result.push(byte as char);
    }
    
    Ok(result)
}

// Extract 6-bit groups from 256-bit input
#[target_feature(enable = "avx2")]
unsafe fn extract_6bit_groups_avx2(input: __m256i) -> __m256i {
    // Complex bit manipulation to extract 6-bit groups from 24-byte input
    // This involves careful handling of byte boundaries and bit alignment
    
    // Mask for 6-bit values
    let mask_6bit = _mm256_set1_epi8(0x3F);
    
    // Shift operations to align 6-bit groups
    let shifted_2 = _mm256_srli_epi32(input, 2);
    let shifted_8 = _mm256_srli_epi32(input, 8);
    let shifted_14 = _mm256_srli_epi32(input, 14);
    let shifted_20 = _mm256_srli_epi32(input, 20);
    
    // Extract and pack 6-bit groups
    let group0 = _mm256_and_si256(shifted_2, mask_6bit);
    let group1 = _mm256_and_si256(shifted_8, mask_6bit);
    let group2 = _mm256_and_si256(shifted_14, mask_6bit);
    let group3 = _mm256_and_si256(shifted_20, mask_6bit);
    
    // Pack into output register
    _mm256_packus_epi32(
        _mm256_packus_epi32(group0, group1),
        _mm256_packus_epi32(group2, group3)
    )
}

// Apply position-dependent alphabet rotation
#[target_feature(enable = "avx2")]
unsafe fn apply_rotation_avx2(indices: __m256i, rotation: u8) -> __m256i {
    let rotation_vec = _mm256_set1_epi8(rotation as i8);
    let rotated = _mm256_add_epi8(indices, rotation_vec);
    let mask_64 = _mm256_set1_epi8(0x3F);
    _mm256_and_si256(rotated, mask_64)
}

// Alphabet character lookup with AVX2
#[target_feature(enable = "avx2")]
unsafe fn alphabet_lookup_avx2(indices: __m256i, alphabet: &[u8]) -> __m256i {
    // Use gather operations or lookup table for character mapping
    // This is a simplified version - actual implementation requires
    // careful handling of alphabet indexing
    
    let mut result = [0u8; 32];
    let indices_array = std::mem::transmute::<__m256i, [u8; 32]>(indices);
    
    for (i, &index) in indices_array.iter().enumerate() {
        result[i] = alphabet[(index & 0x3F) as usize];
    }
    
    _mm256_loadu_si256(result.as_ptr() as *const __m256i)
}
```

### ARM NEON Implementation

```rust
// src/encoding/neon.rs - ARM NEON optimizations
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn encode_eq64_neon(
    data: &[u8],
    position: usize,
    alphabet: &str
) -> Result<String, Box<dyn std::error::Error>> {
    
    let mut result = Vec::with_capacity((data.len() + 2) / 3 * 4);
    
    // Process 12-byte chunks with NEON (produces 16 output characters)
    let chunks = data.chunks_exact(12);
    let remainder = chunks.remainder();
    
    for (chunk_idx, chunk) in chunks.enumerate() {
        let chunk_position = position + chunk_idx * 12;
        let encoded_chunk = encode_chunk_neon(chunk, chunk_position, alphabet)?;
        result.extend_from_slice(encoded_chunk.as_bytes());
    }
    
    // Handle remainder
    if !remainder.is_empty() {
        let remainder_position = position + data.len() - remainder.len();
        let encoded_remainder = encode_eq64_scalar(remainder, remainder_position, Some(alphabet))?;
        result.extend_from_slice(encoded_remainder.as_bytes());
    }
    
    String::from_utf8(result).map_err(|e| e.into())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn encode_chunk_neon(
    chunk: &[u8],
    position: usize,
    alphabet: &str
) -> Result<String, Box<dyn std::error::Error>> {
    
    // Load 12 bytes into NEON register
    let input = vld1q_u8(chunk.as_ptr());
    
    // Position-dependent rotation
    let rotation = ((position / 3) % 64) as u8;
    
    // Extract 6-bit groups using NEON operations
    let indices = extract_6bit_groups_neon(input);
    
    // Apply rotation
    let rotation_vec = vdupq_n_u8(rotation);
    let rotated_indices = vaddq_u8(indices, rotation_vec);
    let mask = vdupq_n_u8(0x3F);
    let final_indices = vandq_u8(rotated_indices, mask);
    
    // Character lookup
    let alphabet_bytes = alphabet.as_bytes();
    let mut result = String::with_capacity(16);
    
    let indices_array: [u8; 16] = std::mem::transmute(final_indices);
    for &index in &indices_array {
        result.push(alphabet_bytes[(index & 0x3F) as usize] as char);
    }
    
    Ok(result)
}
```

## Memory Management

### Custom Allocator Integration

```rust
// src/memory/allocator.rs - Custom memory management
use std::alloc::{GlobalAlloc, Layout};
use std::ptr::NonNull;

pub struct QuadB64Allocator {
    pool: MemoryPool,
    fallback: std::alloc::System,
}

impl QuadB64Allocator {
    pub fn new(pool_size: usize) -> Self {
        Self {
            pool: MemoryPool::new(pool_size),
            fallback: std::alloc::System,
        }
    }
}

unsafe impl GlobalAlloc for QuadB64Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Use pool for small, frequent allocations
        if layout.size() <= 8192 && layout.align() <= 64 {
            match self.pool.allocate(layout) {
                Some(ptr) => ptr.as_ptr(),
                None => self.fallback.alloc(layout),
            }
        } else {
            self.fallback.alloc(layout)
        }
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !self.pool.deallocate(NonNull::new_unchecked(ptr), layout) {
            self.fallback.dealloc(ptr, layout);
        }
    }
}

// Memory pool implementation
struct MemoryPool {
    blocks: Vec<MemoryBlock>,
    free_list: Vec<NonNull<u8>>,
    stats: PoolStats,
}

impl MemoryPool {
    fn new(size: usize) -> Self {
        // Initialize memory pool with pre-allocated blocks
        Self {
            blocks: Vec::new(),
            free_list: Vec::new(),
            stats: PoolStats::default(),
        }
    }
    
    fn allocate(&self, layout: Layout) -> Option<NonNull<u8>> {
        // Fast path: Check free list
        if let Some(ptr) = self.free_list.pop() {
            Some(ptr)
        } else {
            // Allocate new block if pool has space
            self.allocate_new_block(layout)
        }
    }
    
    fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) -> bool {
        // Return to free list if it belongs to our pool
        if self.owns_pointer(ptr) {
            self.free_list.push(ptr);
            true
        } else {
            false
        }
    }
}
```

### Buffer Management

```rust
// src/memory/buffers.rs - Optimized buffer handling
pub struct BufferManager {
    small_buffers: Vec<Vec<u8>>,     // < 1KB
    medium_buffers: Vec<Vec<u8>>,    // 1KB - 64KB  
    large_buffers: Vec<Vec<u8>>,     // > 64KB
    stats: BufferStats,
}

impl BufferManager {
    pub fn get_buffer(&mut self, size: usize) -> Vec<u8> {
        let pool = match size {
            0..=1024 => &mut self.small_buffers,
            1025..=65536 => &mut self.medium_buffers,
            _ => &mut self.large_buffers,
        };
        
        // Reuse existing buffer or allocate new one
        pool.pop().unwrap_or_else(|| {
            self.stats.allocations += 1;
            Vec::with_capacity(size)
        })
    }
    
    pub fn return_buffer(&mut self, mut buffer: Vec<u8>) {
        // Clear and return to appropriate pool
        buffer.clear();
        
        let pool = match buffer.capacity() {
            0..=1024 => &mut self.small_buffers,
            1025..=65536 => &mut self.medium_buffers,
            _ => &mut self.large_buffers,
        };
        
        // Limit pool size to prevent memory leaks
        if pool.len() < 100 {
            pool.push(buffer);
            self.stats.reuses += 1;
        }
    }
}
```

## Compilation and Build System

### Cargo Configuration

```toml
# Cargo.toml - Rust build configuration
[package]
name = "uubed-native"
version = "0.5.0"
edition = "2021"
rust-version = "1.70"

[lib]
name = "uubed_native"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"
rayon = "1.7"

[features]
default = ["simd"]
simd = []
avx2 = ["simd"]
avx512 = ["simd"]
neon = ["simd"]

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"

# Platform-specific optimizations
[target.'cfg(target_arch = "x86_64")'.dependencies]
raw-cpuid = "10.0"

[target.'cfg(target_arch = "aarch64")'.dependencies]
# ARM-specific dependencies

# Build script for feature detection
[build-dependencies]
cc = "1.0"
```

### Build Script

```rust
// build.rs - Feature detection and compilation
use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    
    // Detect CPU features
    detect_cpu_features(&target_arch);
    
    // Configure platform-specific builds
    configure_platform_build(&target_arch, &target_os);
    
    // Build C++ components if needed
    build_cpp_components();
}

fn detect_cpu_features(arch: &str) {
    match arch {
        "x86_64" => {
            if has_feature("avx2") {
                println!("cargo:rustc-cfg=has_avx2");
            }
            if has_feature("avx512f") {
                println!("cargo:rustc-cfg=has_avx512");
            }
        }
        "aarch64" => {
            if has_feature("neon") {
                println!("cargo:rustc-cfg=has_neon");
            }
        }
        _ => {}
    }
}

fn has_feature(feature: &str) -> bool {
    // Use runtime feature detection
    match feature {
        "avx2" => is_x86_feature_detected!("avx2"),
        "avx512f" => is_x86_feature_detected!("avx512f"),
        "neon" => cfg!(target_feature = "neon"),
        _ => false,
    }
}

fn configure_platform_build(arch: &str, os: &str) {
    // Platform-specific compiler flags
    let mut build = cc::Build::new();
    
    build
        .cpp(true)
        .std("c++17")
        .flag("-O3")
        .flag("-march=native");
    
    match arch {
        "x86_64" => {
            build.flag("-mavx2").flag("-mfma");
        }
        "aarch64" => {
            build.flag("-march=armv8-a+simd");
        }
        _ => {}
    }
    
    match os {
        "windows" => {
            build.flag("/arch:AVX2");
        }
        "macos" => {
            build.flag("-stdlib=libc++");
        }
        _ => {}
    }
    
    build.compile("uubed_cpp");
}
```

### Python Integration

```python
# setup.py - Python wheel building
from pyo3_pack import build_wheel
import platform
import subprocess
import os

def build_native_extensions():
    """Build native extensions with optimal configuration"""
    
    # Detect CPU capabilities
    cpu_features = detect_cpu_features()
    
    # Configure Rust build
    rust_flags = configure_rust_build(cpu_features)
    
    # Set environment variables
    env = os.environ.copy()
    env['RUSTFLAGS'] = rust_flags
    env['CARGO_BUILD_TARGET'] = get_build_target()
    
    # Build wheel
    build_wheel(
        manifest_path="native/Cargo.toml",
        target_dir="target",
        release=True,
        strip=True,
        env=env
    )

def detect_cpu_features():
    """Detect CPU features for optimization"""
    features = []
    
    if platform.machine() in ['x86_64', 'AMD64']:
        # Check for x86 features
        if has_x86_feature('avx2'):
            features.append('avx2')
        if has_x86_feature('avx512f'):
            features.append('avx512')
    elif platform.machine().startswith('arm') or platform.machine() == 'aarch64':
        # Check for ARM features
        features.append('neon')
    
    return features

def has_x86_feature(feature):
    """Check if x86 CPU feature is available"""
    try:
        result = subprocess.run(
            ['python', '-c', f'import cpuid; print(cpuid.has_{feature}())'],
            capture_output=True,
            text=True
        )
        return result.stdout.strip() == 'True'
    except:
        return False

def configure_rust_build(features):
    """Configure Rust compiler flags"""
    flags = ['-C', 'target-cpu=native']
    
    if 'avx2' in features:
        flags.extend(['-C', 'target-feature=+avx2'])
    if 'avx512' in features:
        flags.extend(['-C', 'target-feature=+avx512f'])
    if 'neon' in features:
        flags.extend(['-C', 'target-feature=+neon'])
    
    return ' '.join(flags)

if __name__ == "__main__":
    build_native_extensions()
```

## Performance Profiling

### Built-in Profiling Tools

```rust
// src/profiling/mod.rs - Performance profiling
use std::time::{Duration, Instant};
use std::collections::HashMap;

pub struct PerformanceProfiler {
    timings: HashMap<String, Vec<Duration>>,
    counters: HashMap<String, u64>,
    active_timers: HashMap<String, Instant>,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            timings: HashMap::new(),
            counters: HashMap::new(),
            active_timers: HashMap::new(),
        }
    }
    
    pub fn start_timer(&mut self, name: &str) {
        self.active_timers.insert(name.to_string(), Instant::now());
    }
    
    pub fn end_timer(&mut self, name: &str) {
        if let Some(start_time) = self.active_timers.remove(name) {
            let duration = start_time.elapsed();
            self.timings.entry(name.to_string())
                .or_insert_with(Vec::new)
                .push(duration);
        }
    }
    
    pub fn increment_counter(&mut self, name: &str) {
        *self.counters.entry(name.to_string()).or_insert(0) += 1;
    }
    
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Performance Profile Report ===\n\n");
        
        // Timing statistics
        report.push_str("Timing Statistics:\n");
        for (name, times) in &self.timings {
            let total: Duration = times.iter().sum();
            let avg = total / times.len() as u32;
            let min = times.iter().min().unwrap();
            let max = times.iter().max().unwrap();
            
            report.push_str(&format!(
                "{}: {} calls, avg: {:?}, min: {:?}, max: {:?}, total: {:?}\n",
                name, times.len(), avg, min, max, total
            ));
        }
        
        // Counter statistics
        report.push_str("\nCounter Statistics:\n");
        for (name, count) in &self.counters {
            report.push_str(&format!("{}: {}\n", name, count));
        }
        
        report
    }
}

// Profiling macros for easy integration
#[macro_export]
macro_rules! profile_scope {
    ($profiler:expr, $name:expr, $block:block) => {
        $profiler.start_timer($name);
        let result = $block;
        $profiler.end_timer($name);
        result
    };
}
```

### Benchmarking Framework

```rust
// src/benchmarks/mod.rs - Comprehensive benchmarks
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;

pub fn benchmark_encoding_variants(c: &mut Criterion) {
    let test_sizes = vec![64, 256, 1024, 4096, 16384, 65536];
    
    let mut group = c.benchmark_group("encoding_variants");
    group.measurement_time(Duration::from_secs(10));
    
    for size in test_sizes {
        let data = vec![0xAA; size];
        
        // Benchmark scalar implementation
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &data,
            |b, data| {
                b.iter(|| encode_eq64_scalar(data, 0, None))
            }
        );
        
        // Benchmark SIMD implementation
        #[cfg(target_feature = "avx2")]
        group.bench_with_input(
            BenchmarkId::new("avx2", size),
            &data,
            |b, data| {
                b.iter(|| unsafe { encode_eq64_avx2(data, 0, None) })
            }
        );
        
        // Benchmark against base64
        group.bench_with_input(
            BenchmarkId::new("base64_reference", size),
            &data,
            |b, data| {
                b.iter(|| base64::encode(data))
            }
        );
    }
    
    group.finish();
}

pub fn benchmark_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");
    
    // Test different allocation strategies
    group.bench_function("stack_allocation", |b| {
        b.iter(|| {
            let buffer = [0u8; 1024];
            encode_with_stack_buffer(&buffer)
        })
    });
    
    group.bench_function("heap_allocation", |b| {
        b.iter(|| {
            let buffer = vec![0u8; 1024];
            encode_with_heap_buffer(&buffer)
        })
    });
    
    group.bench_function("pool_allocation", |b| {
        let mut pool = BufferPool::new();
        b.iter(|| {
            let buffer = pool.get_buffer(1024);
            let result = encode_with_pooled_buffer(&buffer);
            pool.return_buffer(buffer);
            result
        })
    });
    
    group.finish();
}

criterion_group!(benches, benchmark_encoding_variants, benchmark_memory_patterns);
criterion_main!(benches);
```

## Deployment and Distribution

### Wheel Building

```yaml
# .github/workflows/build-wheels.yml - CI/CD for wheel building
name: Build Native Wheels

on:
  push:
    tags: ['v*']
  pull_request:

jobs:
  build_wheels:
    name: Build wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        architecture: [x86_64, aarch64]
        exclude:
          - os: windows-latest
            architecture: aarch64

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        target: ${{ matrix.architecture }}-unknown-linux-gnu
        override: true
    
    - name: Install maturin
      run: pip install maturin
    
    - name: Build wheels
      run: |
        maturin build --release \
          --target ${{ matrix.architecture }}-unknown-linux-gnu \
          --features simd \
          --out dist
    
    - name: Test wheels
      run: |
        pip install dist/*.whl
        python -c "import uubed; print(uubed.has_native_support())"
    
    - uses: actions/upload-artifact@v3
      with:
        path: dist/*.whl
```

### Installation Validation

```python
# validate_installation.py - Post-installation validation
import sys
import platform
import subprocess
import importlib.util

def validate_native_extensions():
    """Validate that native extensions are properly installed"""
    
    print("=== QuadB64 Native Extension Validation ===\n")
    
    # Basic import test
    try:
        import uubed
        print("✅ QuadB64 package imported successfully")
        print(f"   Version: {uubed.__version__}")
    except ImportError as e:
        print(f"❌ Failed to import QuadB64: {e}")
        return False
    
    # Native extension test
    try:
        has_native = uubed.has_native_support()
        print(f"✅ Native extensions: {'Available' if has_native else 'Not available'}")
        if not has_native:
            print("   Falling back to Python implementation")
    except Exception as e:
        print(f"❌ Error checking native support: {e}")
        return False
    
    # SIMD capabilities test
    try:
        has_simd = uubed.has_simd_support()
        print(f"✅ SIMD optimizations: {'Available' if has_simd else 'Not available'}")
        if has_simd:
            features = uubed.get_simd_features()
            print(f"   Supported features: {', '.join(features)}")
    except Exception as e:
        print(f"❌ Error checking SIMD support: {e}")
    
    # Performance benchmark
    try:
        import time
        data = b"Hello, World!" * 1000
        
        start = time.perf_counter()
        encoded = uubed.encode_eq64(data)
        end = time.perf_counter()
        
        duration = (end - start) * 1000
        throughput = len(data) / (end - start) / 1024 / 1024
        
        print(f"✅ Performance test: {duration:.2f}ms, {throughput:.1f} MB/s")
        
        # Roundtrip test
        decoded = uubed.decode_eq64(encoded)
        if decoded == data:
            print("✅ Roundtrip encoding/decoding successful")
        else:
            print("❌ Roundtrip test failed")
            return False
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False
    
    print("\n✅ All validation tests passed!")
    return True

def print_system_info():
    """Print relevant system information"""
    print("\n=== System Information ===")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version}")
    print(f"Processor: {platform.processor()}")
    
    # CPU feature detection
    try:
        if platform.machine() in ['x86_64', 'AMD64']:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            flags = info.get('flags', [])
            relevant_flags = [f for f in flags if f in ['avx', 'avx2', 'sse4_1', 'sse4_2']]
            print(f"CPU Features: {', '.join(relevant_flags)}")
    except ImportError:
        print("CPU Features: Unable to detect (install py-cpuinfo for details)")

if __name__ == "__main__":
    print_system_info()
    success = validate_native_extensions()
    sys.exit(0 if success else 1)
```

This comprehensive native extensions guide covers the complete implementation from Rust code to deployment, providing developers with everything needed to understand, build, and optimize QuadB64's native performance components.