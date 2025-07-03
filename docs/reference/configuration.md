# Configuration Options

This page documents all configuration options available in uubed for customizing encoding behavior and performance.

## Global Configuration

### Configuration File Format

uubed supports configuration via TOML files:

```toml
# uubed.toml
[encoding]
default_method = "shq64"
validate_inputs = true
performance_mode = "balanced"

[encoding.shq64]
default_planes = 64
seed = 42

[encoding.t8q64]
default_k = 8

[logging]
level = "info"
format = "json"
```

### Environment Variables

Common configuration options can be set via environment variables:

```bash
export UUBED_DEFAULT_METHOD=shq64
export UUBED_VALIDATE_INPUTS=true
export UUBED_LOG_LEVEL=debug
```

## Method-Specific Configuration

### shq64 Configuration

```python
# Python configuration
config = {
    "planes": 64,          # Number of hash planes (must be multiple of 8)
    "seed": 42,            # Random seed for reproducibility
    "normalize": True      # Normalize input vectors
}
```

### t8q64 Configuration

```python
config = {
    "k": 8,                # Number of top elements to preserve
    "threshold": 0.1       # Minimum value threshold
}
```

### eq64 Configuration

```python
config = {
    "compression": False,  # Enable additional compression
    "precision": "high"    # Encoding precision level
}
```

## Performance Configuration

### Memory Management

```toml
[memory]
max_batch_size = 10000
streaming_threshold = 100000
cache_size = "1GB"
```

### Parallel Processing

```toml
[performance]
num_threads = "auto"      # Number of threads (auto-detect)
simd_enabled = true       # Enable SIMD optimizations
batch_processing = true   # Enable batch processing
```

## Validation Settings

```toml
[validation]
strict_mode = true        # Enable strict input validation
float_range_check = true  # Validate float value ranges
dimension_check = true    # Validate embedding dimensions
```

## Logging Configuration

```toml
[logging]
level = "info"           # debug, info, warning, error
output = "console"       # console, file, both
file_path = "/var/log/uubed.log"
rotation = "daily"       # daily, weekly, size-based
max_size = "100MB"
```

## Integration-Specific Settings

### Vector Database Connectors

```toml
[connectors.pinecone]
timeout = 30
retry_attempts = 3
batch_size = 1000

[connectors.weaviate]
connection_timeout = 10
read_timeout = 60
```

## Advanced Configuration

### Custom Encoding Parameters

```python
from uubed.config import set_global_config

# Set custom defaults
set_global_config({
    "encoding": {
        "default_method": "mq64",
        "mq64": {
            "levels": [64, 128, 256, 512, 1024]
        }
    }
})
```

### Runtime Configuration

```python
# Override configuration at runtime
from uubed import encode
from uubed.config import override_config

with override_config({"validation.strict_mode": False}):
    result = encode(data, method="shq64")
```

## Configuration Precedence

Configuration is loaded in the following order (later overrides earlier):

1. Default built-in configuration
2. System-wide configuration file (`/etc/uubed/config.toml`)
3. User configuration file (`~/.config/uubed/config.toml`)
4. Project configuration file (`./uubed.toml`)
5. Environment variables
6. Runtime overrides

## Validation and Testing

### Configuration Validation

```bash
# Validate configuration file
uubed config validate ./uubed.toml

# Show effective configuration
uubed config show
```

### Performance Testing

```bash
# Test encoding performance with current configuration
uubed benchmark --config ./uubed.toml
```

## Migration Guide

### From Version 1.0 to 1.1

```toml
# Old format (deprecated)
[global]
method = "shq64"

# New format
[encoding]
default_method = "shq64"
```

## Related Topics

- [Performance Optimization](../performance/optimization.md)
- [API Reference](../api.md)
- [Troubleshooting](troubleshooting.md)