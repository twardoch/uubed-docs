# Vector Database Integration

This guide covers integrating uubed with popular vector databases for efficient embedding storage and retrieval.

## Supported Vector Databases

uubed provides built-in connectors for major vector databases:

- **Pinecone**: Cloud-native vector database
- **Weaviate**: Open-source vector search engine
- **Qdrant**: Vector similarity search engine
- **ChromaDB**: AI-native embedding database

## Quick Setup

### Pinecone Integration

```python
from uubed.integrations.vectordb import get_connector

# Initialize Pinecone connector with uubed encoding
connector = get_connector(
    "pinecone", 
    api_key="your-api-key",
    environment="us-west1-gcp",
    encoding_method="shq64"
)

# Insert vectors with automatic encoding
vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
documents = ["Doc 1", "Doc 2"]
connector.insert_vectors(vectors, documents)
```

### Weaviate Integration

```python
# Initialize Weaviate connector
connector = get_connector(
    "weaviate",
    url="http://localhost:8080",
    encoding_method="eq64"
)

# Create collection and insert data
connector.create_collection("my_collection")
connector.insert_vectors(vectors, documents)
```

## Encoding Strategies

### Method Selection

Choose encoding methods based on your use case:

- **eq64**: Lossless encoding, perfect for exact retrieval
- **shq64**: Lossy compression, good for similarity search
- **t8q64**: Top-k encoding, efficient for sparse vectors
- **zoq64**: Z-order encoding, preserves spatial locality

### Performance Considerations

```python
# For large-scale deployments
connector = get_connector(
    "qdrant",
    host="localhost",
    port=6333,
    encoding_method="shq64",
    encoding_kwargs={"planes": 128}  # Higher compression
)
```

## Advanced Usage

### Batch Operations

```python
# Efficient batch insertion
from uubed.streaming import batch_encode

# Encode large batches efficiently
encoded_batch = batch_encode(
    large_vector_list,
    method="shq64",
    batch_size=1000
)

connector.batch_insert(encoded_batch, metadata_list)
```

### Custom Encoding

```python
# Custom encoding parameters for specific use cases
connector = get_connector(
    "chromadb",
    persist_directory="./chroma_db",
    encoding_method="mq64",
    encoding_kwargs={
        "levels": [64, 128, 256, 512]  # Matryoshka encoding
    }
)
```

## Migration Guide

### From Raw Vectors

```python
# Migrate existing vector database to use uubed encoding
from uubed.integrations.vectordb import migrate_collection

migrate_collection(
    source_connector=old_connector,
    target_connector=uubed_connector,
    collection_name="embeddings",
    encoding_method="shq64"
)
```

## Best Practices

1. **Consistent Encoding**: Use the same method across your entire dataset
2. **Batch Processing**: Use batch operations for better performance
3. **Method Selection**: Choose encoding based on accuracy vs. compression needs
4. **Monitoring**: Track encoding/decoding performance and accuracy metrics

## Troubleshooting

See [Python Troubleshooting](../reference/python-troubleshooting.md) for common issues and solutions.

## Related Topics

- [Search Engines](search-engines.md)
- [Performance Optimization](../performance/optimization.md)
- [API Reference](../api.md)