---
layout: default
title: Glossary
parent: Reference
nav_order: 5
description: "Comprehensive glossary of terms, concepts, and technical definitions used throughout the UUBED project."
---

# UUBED Glossary

A comprehensive glossary of terms, concepts, and technical definitions used throughout the UUBED project.

## Core Concepts

### QuadB64
The family of position-safe Base64 encoding variants developed for the UUBED project. Unlike standard Base64, QuadB64 encodings preserve locality and avoid substring pollution in search engines.

### Substring Pollution
The problem where encoded data fragments appear as false positives in search results because they match partial queries. Standard Base64 encoding is particularly susceptible to this issue.

### Position-Safe Encoding
An encoding scheme where the position of a character within the encoded string affects its value, preventing arbitrary substrings from being valid encodings.

### Matryoshka Embeddings
Nested embedding representations where earlier dimensions contain more important information. UUBED's Mq64 encoding provides specialized support for these hierarchical structures.

## Encoding Variants

### Eq64 (Embeddings QuadB64)
The primary encoding scheme for full embedding vectors. Uses position-safe alphabets to ensure no substring pollution. Optimized for dense vector representations with full reversibility.

### Shq64 (SimHash QuadB64)
A variant designed for SimHash fingerprints and other binary hash representations. Provides compact encoding while maintaining locality-sensitive hashing properties.

### T8q64 (Top-k QuadB64)
Specialized encoding for top-k sparse representations where only the indices and values of the k largest components are stored. Ideal for sparse vector applications.

### Zoq64 (Z-order QuadB64)
Encoding variant using Z-order (Morton) encoding for spatial data and multi-dimensional indices. Preserves locality across dimensions for efficient spatial queries.

### Mq64 (Matryoshka QuadB64)
Advanced encoding for hierarchical embeddings with progressive refinement capabilities. Supports decoding at multiple dimensional resolutions (64, 128, 256, 512, 1024+).

## Technical Terms

### SIMD (Single Instruction, Multiple Data)
Parallel processing technique used in UUBED for accelerating encoding/decoding operations. Implementations include AVX2, AVX-512, and NEON for different CPU architectures.

### FFI (Foreign Function Interface)
The interface layer that allows the Rust core to be called from other languages, primarily Python via PyO3. Enables 40-100x performance improvements over pure Python.

### Locality Preservation
The property of an encoding scheme where similar inputs produce similar encoded outputs, maintaining neighborhood relationships. Critical for similarity search applications.

### Progressive Decoding
The ability to decode an encoded string partially, retrieving only the first N dimensions. Particularly useful with Matryoshka embeddings for adaptive quality/performance trade-offs.

## Performance Terms

### Vectorization
The process of converting sequential operations into parallel SIMD operations for improved performance. UUBED uses CPU-specific vectorization for optimal speed.

### Zero-Copy Operations
Data processing techniques that avoid unnecessary memory allocations and copies, crucial for high-performance encoding. Achieved through careful memory management in Rust.

### Batch Processing
Encoding or decoding multiple vectors simultaneously to amortize overhead and improve throughput. Recommended for production workloads.

### Throughput
Measured in MB/s (megabytes per second), indicates how much data can be encoded/decoded per unit time. UUBED achieves 94-234 MB/s depending on the encoding method.

## Implementation Details

### PyO3
The Rust library used to create Python bindings for the uubed-rs core implementation. Provides seamless integration between Rust performance and Python ease-of-use.

### Position-Dependent Alphabets
The four different 16-character alphabets used in QuadB64, cycled based on character position:
- Positions 0,4,8...: Uppercase letters
- Positions 1,5,9...: Mixed case letters  
- Positions 2,6,10...: Lowercase letters
- Positions 3,7,11...: Digits and symbols

### Chunk Size
The number of bytes processed as a unit during encoding. QuadB64 processes data in groups that align with the position-safe alphabet system.

### Checksum
Error detection mechanism included in some encoding variants (like Mq64) to verify data integrity during transmission or storage.

## Search and Retrieval

### Vector Database
Specialized databases (Pinecone, Weaviate, Qdrant, ChromaDB) optimized for storing and searching high-dimensional vectors. Primary use case for UUBED encodings.

### Cosine Similarity
Common similarity metric for comparing embedding vectors, measuring the cosine of the angle between them. Preserved by UUBED's locality-preserving encodings.

### Approximate Nearest Neighbor (ANN)
Search algorithms that find similar vectors efficiently by trading exact accuracy for speed. UUBED encodings maintain properties that benefit ANN algorithms.

### Embedding Model
Machine learning models that convert text, images, or other data into high-dimensional vector representations. Examples include OpenAI's text-embedding-3, Cohere's embed models, and open-source alternatives.

## Use Cases

### Semantic Search
Finding documents or data points based on meaning rather than exact keyword matches. UUBED encodings enable efficient semantic search in traditional search engines.

### RAG (Retrieval-Augmented Generation)
AI pattern combining vector search with language models. UUBED encodings facilitate the storage and retrieval of document embeddings for RAG systems.

### Multimodal Search
Searching across different data types (text, images, audio) using unified vector representations. UUBED supports encoding vectors from multimodal models like CLIP.

---

*This glossary is continuously updated as the project evolves. For corrections or additions, please submit an issue or pull request to the documentation repository.*