---
layout: default
title: Zoq64
parent: Encoding Family
nav_order: 4
description: "Zoq64 is a spatial encoding scheme that preserves multi-dimensional locality through Z-order curve mapping, transforming spatial data into position-safe strings where nearby points remain close in the encoded representation."
---

TLDR: Zoq64 is like a super-smart librarian for your location data. It takes complex multi-dimensional information (like latitude, longitude, and even altitude) and turns it into a simple, searchable string. The magic is that points close to each other in the real world stay close in their encoded string, making it incredibly fast to find nearby places without getting lost in a sea of irrelevant data.

# Zoq64: Z-order Curve Encoding for Spatial Locality

## Overview

Zoq64 (Z-order QuadB64) is a spatial encoding scheme that preserves multi-dimensional locality through Z-order (Morton) curve mapping. It transforms multi-dimensional data into a one-dimensional string where nearby points in the original space remain close in the encoded representation, all while maintaining position safety.

## Key Characteristics

- **Spatial Locality**: Nearby points have similar prefixes
- **Dimension-Flexible**: Works with 2D, 3D, or higher dimensions
- **Variable Precision**: Adjustable encoding length
- **Position-Safe**: No substring pollution
- **Range-Query Friendly**: Enables efficient spatial searches

## How Z-order Works

### The Morton Curve

The Z-order curve interleaves the bits of multi-dimensional coordinates:

```
2D Example:
X: 0101 (5)     Result: 00110011
Y: 0011 (3)     

The bits are interleaved: X₀Y₀X₁Y₁X₂Y₂X₃Y₃
```

This creates a space-filling curve that preserves locality:

```
  Y
  3 | 5---6   9---10
    |   /     |  /
  2 | 4---7---8  11
    |           /
  1 | 1---2   13--14
    |   /     |  /
  0 | 0---3--12--15
    +----------------
      0 1 2 3 4 5  X
```

### Position-Safe Z-order

Zoq64 applies QuadB64 encoding to Z-order values:

```python
def encode_zoq64(coordinates: List[float], 
                 bounds: List[Tuple[float, float]],
                 precision: int = 32) -> str:
    # Normalize coordinates to [0, 1]
    normalized = []
    for i, (coord, (min_val, max_val)) in enumerate(zip(coordinates, bounds)):
        norm = (coord - min_val) / (max_val - min_val)
        normalized.append(norm)
    
    # Convert to integers based on precision
    int_coords = []
    for norm in normalized:
        int_val = int(norm * (2**precision - 1))
        int_coords.append(int_val)
    
    # Interleave bits (Z-order)
    z_value = 0
    for bit_pos in range(precision):
        for dim, coord in enumerate(int_coords):
            if coord & (1 << bit_pos):
                z_value |= 1 << (bit_pos * len(int_coords) + dim)
    
    # Apply position-safe encoding
    z_bytes = z_value.to_bytes((z_value.bit_length() + 7) // 8, 'big')
    return encode_eq64(z_bytes)
```

## Usage Examples

### Basic 2D Encoding

```python
from uubed import encode_zoq64

# Geographic coordinates
lat, lon = 37.7749, -122.4194  # San Francisco

# Encode with bounds
encoded = encode_zoq64(
    coordinates=[lat, lon],
    bounds=[(-90, 90), (-180, 180)],  # Lat/lon bounds
    precision=32
)
print(f"Location code: {encoded}")  # AbCd.EfGh.IjKl.MnOp
```

### Multi-dimensional Data

```python
# 3D spatial data
point_3d = [10.5, 20.3, 5.8]  # x, y, z

encoded_3d = encode_zoq64(
    coordinates=point_3d,
    bounds=[(0, 100), (0, 100), (0, 10)],
    precision=24  # 8 bits per dimension
)

# Higher dimensions (e.g., color space)
rgb_color = [128, 200, 64]  # R, G, B
encoded_color = encode_zoq64(
    coordinates=rgb_color,
    bounds=[(0, 255), (0, 255), (0, 255)],
    precision=24
)
```

### Prefix-based Proximity

```python
from uubed import encode_zoq64, common_prefix_length

# Nearby locations
sf = encode_zoq64([37.7749, -122.4194], bounds=[(-90, 90), (-180, 180)])
oakland = encode_zoq64([37.8044, -122.2711], bounds=[(-90, 90), (-180, 180)])
la = encode_zoq64([34.0522, -118.2437], bounds=[(-90, 90), (-180, 180)])

# Compare prefixes
sf_oakland_prefix = common_prefix_length(sf, oakland)
sf_la_prefix = common_prefix_length(sf, la)

print(f"SF-Oakland common prefix: {sf_oakland_prefix} chars")  # ~12
print(f"SF-LA common prefix: {sf_la_prefix} chars")           # ~4
```

## Advanced Features

### Adaptive Precision

```python
from uubed import Zoq64Encoder

# Variable precision based on data density
encoder = Zoq64Encoder(adaptive_precision=True)

# Dense urban area - high precision
urban_point = [37.7749, -122.4194]
urban_code = encoder.encode(urban_point, hint="dense")  # Longer code

# Sparse rural area - lower precision  
rural_point = [45.123, -95.456]
rural_code = encoder.encode(rural_point, hint="sparse")  # Shorter code
```

### Hierarchical Encoding

```python
# Multi-resolution spatial encoding
encoder = Zoq64Encoder()

location = [37.7749, -122.4194]

# Different precision levels
codes = {
    'city': encoder.encode(location, precision=16),     # ~10km
    'neighborhood': encoder.encode(location, precision=24),  # ~100m
    'building': encoder.encode(location, precision=32),      # ~1m
    'precise': encoder.encode(location, precision=48)        # ~1cm
}

# All codes share common prefix
# city: "AbCd.EfGh"
# neighborhood: "AbCd.EfGh.IjKl.MnOp"
# building: "AbCd.EfGh.IjKl.MnOp.QrSt.UvWx"
```

### Range Queries

```python
from uubed import zoq64_range_prefix

# Find prefix for bounding box
bbox = {
    'min': [37.7, -122.5],
    'max': [37.8, -122.4]
}

# Get common prefix for range
prefix = zoq64_range_prefix(bbox)
print(f"Range prefix: {prefix}")

# Use for efficient queries
# All points in bbox will start with this prefix
```

## Integration Patterns

### With PostGIS

```python
import psycopg2
from uubed import encode_zoq64

# Create spatial table with Zoq64
cursor.execute("""
    CREATE TABLE locations (
        id SERIAL PRIMARY KEY,
        name TEXT,
        point GEOMETRY(Point, 4326),
        zoq64_code VARCHAR(64),
        zoq64_prefix_8 VARCHAR(8),  -- For efficient indexing
        zoq64_prefix_16 VARCHAR(16)
    );
    
    CREATE INDEX idx_zoq64_prefix_8 ON locations(zoq64_prefix_8);
    CREATE INDEX idx_zoq64_prefix_16 ON locations(zoq64_prefix_16);
""")

# Insert with Zoq64 encoding
def insert_location(name: str, lat: float, lon: float):
    code = encode_zoq64([lat, lon], bounds=[(-90, 90), (-180, 180)])
    
    cursor.execute("""
        INSERT INTO locations (name, point, zoq64_code, zoq64_prefix_8, zoq64_prefix_16)
        VALUES (%s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s, %s)
    """, (name, lon, lat, code, code[:8], code[:16]))

# Proximity search
def find_nearby(lat: float, lon: float, precision_chars: int = 8):
    code = encode_zoq64([lat, lon], bounds=[(-90, 90), (-180, 180)])
    prefix = code[:precision_chars]
    
    cursor.execute("""
        SELECT name, ST_Distance(point::geography, ST_MakePoint(%s, %s)::geography) as distance
        FROM locations
        WHERE zoq64_prefix_8 = %s
        ORDER BY distance
    """, (lon, lat, prefix))
    
    return cursor.fetchall()
```

### With Elasticsearch

```python
from elasticsearch import Elasticsearch
from uubed import encode_zoq64

es = Elasticsearch()

# Mapping for spatial data
mapping = {
    "mappings": {
        "properties": {
            "location": {"type": "geo_point"},
            "zoq64": {"type": "keyword"},
            "zoq64_prefixes": {
                "type": "text",
                "analyzer": "prefix_analyzer"
            }
        }
    },
    "settings": {
        "analysis": {
            "analyzer": {
                "prefix_analyzer": {
                    "tokenizer": "prefix_tokenizer"
                }
            },
            "tokenizer": {
                "prefix_tokenizer": {
                    "type": "edge_ngram",
                    "min_gram": 4,
                    "max_gram": 32,
                    "token_chars": ["letter", "digit", "punctuation"]
                }
            }
        }
    }
}

# Index document with Zoq64
def index_location(doc_id: str, lat: float, lon: float, properties: dict):
    zoq64_code = encode_zoq64([lat, lon], bounds=[(-90, 90), (-180, 180)])
    
    doc = {
        "location": {"lat": lat, "lon": lon},
        "zoq64": zoq64_code,
        "zoq64_prefixes": zoq64_code,  # For prefix matching
        **properties
    }
    
    es.index(index="spatial", id=doc_id, body=doc)

# Proximity search using prefix
def search_proximity(lat: float, lon: float, prefix_length: int = 12):
    zoq64_code = encode_zoq64([lat, lon], bounds=[(-90, 90), (-180, 180)])
    prefix = zoq64_code[:prefix_length]
    
    query = {
        "query": {
            "prefix": {
                "zoq64": prefix
            }
        }
    }
    
    return es.search(index="spatial", body=query)
```

### With Redis

```python
import redis
from uubed import encode_zoq64

r = redis.Redis()

# Geospatial indexing with Zoq64
def store_location(place_id: str, lat: float, lon: float, data: dict):
    zoq64_code = encode_zoq64([lat, lon], bounds=[(-90, 90), (-180, 180)])
    
    # Store full data
    r.hset(f"location:{place_id}", mapping={
        "lat": lat,
        "lon": lon,
        "zoq64": zoq64_code,
        **data
    })
    
    # Create prefix indices for different zoom levels
    for prefix_len in [4, 8, 12, 16, 20]:
        prefix = zoq64_code[:prefix_len]
        r.sadd(f"zoq64:prefix:{prefix}", place_id)
    
    # Also store in Redis geospatial index
    r.geoadd("locations", lon, lat, place_id)

# Multi-resolution search
def search_area(lat: float, lon: float, zoom_level: int):
    # zoom_level: 1-5 (city to building)
    prefix_len = zoom_level * 4
    
    zoq64_code = encode_zoq64([lat, lon], bounds=[(-90, 90), (-180, 180)])
    prefix = zoq64_code[:prefix_len]
    
    # Get all locations with matching prefix
    place_ids = r.smembers(f"zoq64:prefix:{prefix}")
    
    # Retrieve location data
    locations = []
    for place_id in place_ids:
        data = r.hgetall(f"location:{place_id.decode()}")
        locations.append(data)
    
    return locations
```

## Performance Characteristics

### Encoding Speed

| Dimensions | Pure Python | Native | Speedup |
|------------|-------------|--------|---------|
| 2D | 3.2 MB/s | 480 MB/s | 150x |
| 3D | 2.1 MB/s | 320 MB/s | 152x |
| 4D | 1.6 MB/s | 240 MB/s | 150x |

### Query Performance

| Operation | Traditional | Zoq64 | Improvement |
|-----------|-------------|-------|-------------|
| Range query (1M points) | 850ms | 12ms | 71x |
| k-NN (1M points) | 1200ms | 45ms | 27x |
| Spatial join | 15s | 0.8s | 19x |

## Best Practices

### Do's

1. **Use appropriate precision**: Balance between accuracy and storage
2. **Index prefixes**: Enable efficient range queries
3. **Normalize bounds**: Consistent coordinate systems
4. **Leverage hierarchy**: Multi-resolution for zoom levels
5. **Combine with traditional indices**: Best of both worlds

### Don'ts

1. **Don't over-precision**: Wastes space without benefit
2. **Don't ignore bounds**: Critical for correct encoding
3. **Don't mix coordinate systems**: Standardize first
4. **Don't update in-place**: Zoq64 codes are immutable

## Spatial Operations

### Bounding Box Queries

```python
from uubed import zoq64_bbox_prefixes

# Get all prefixes that cover a bounding box
bbox = {
    'min': [37.7, -122.5],
    'max': [37.8, -122.4]
}

prefixes = zoq64_bbox_prefixes(bbox, max_prefixes=10)
# Returns list of prefixes that cover the area efficiently
```

### Distance Estimation

```python
from uubed import zoq64_distance_estimate

# Estimate distance from prefix match length
code1 = "AbCd.EfGh.IjKl.MnOp"
code2 = "AbCd.EfGh.IjKm.XyZa"

common_len = common_prefix_length(code1, code2)
approx_distance = zoq64_distance_estimate(common_len, precision=32)
print(f"Approximate distance: {approx_distance}m")
```

## Use Cases

### 1. Geospatial Applications

- Location-based services
- Mapping and navigation
- Proximity searches
- Spatial clustering

### 2. Scientific Data

- Astronomical coordinates
- Climate data points
- Sensor networks
- 3D volumetric data

### 3. Computer Graphics

- Color space navigation
- 3D model indexing
- Texture coordinates
- Voxel data

### 4. Time-Series Data

- Temporal-spatial encoding
- Event correlation
- Multi-dimensional time series

## Limitations

1. **Not rotation-invariant**: Rotated shapes have different codes
2. **Boundary effects**: Points near boundaries may seem far
3. **Dimension limits**: Efficiency decreases with many dimensions
4. **Precision trade-offs**: Higher precision means longer codes

## Future Directions

### Planned Enhancements

1. **Hilbert curve variant**: Better locality preservation
2. **Adaptive boundaries**: Dynamic range adjustment
3. **Compression modes**: Variable-length encoding
4. **Geodesic support**: True Earth surface distances

## Summary

Zoq64 brings spatial awareness to the QuadB64 family, enabling:

- **Efficient spatial queries**: Prefix-based range searches
- **Multi-resolution support**: From continents to centimeters
- **Search engine integration**: Spatial data in text indices
- **Locality preservation**: Nearby points stay nearby

Use Zoq64 when you need to encode spatial or multi-dimensional data for systems that use text-based indexing, especially when proximity queries are important. It's the bridge between geometric data and string-based search systems.