---
layout: default
title: Performance
nav_order: 9
has_children: true
description: "Performance optimization, benchmarks, and platform-specific tuning for QuadB64 encodings."
---

# Performance & Optimization

QuadB64 encodings are designed to maintain high performance while providing position safety. This section covers benchmarks, optimization strategies, and platform-specific tuning guides.

## In This Section

- **[Benchmarks](benchmarks/)** - Comprehensive performance comparisons
- **[Optimization](optimization/)** - General optimization strategies  
- **[Platform Tuning](platform-tuning/)** - Platform-specific performance tips

## Performance Overview

Our benchmarks show that QuadB64 encodings achieve:
- 85-95% of Base64 speed with SIMD optimizations
- Minimal memory overhead (15-30% depending on encoding)
- Excellent scalability across different data sizes
- Strong performance on both x86_64 and ARM64 architectures

Whether you're processing small messages or large datasets, the QuadB64 family provides the performance you need without sacrificing position safety.