# Contributing Guidelines

Thank you for your interest in contributing to uubed! This guide will help you understand our development process and contribute effectively.

## Getting Started

### Before You Begin

1. **Read the Documentation**: Familiarize yourself with uubed's concepts and API
2. **Check Existing Issues**: Look for existing issues or discussions related to your idea
3. **Development Setup**: Follow the [development setup guide](setup.md)
4. **Code of Conduct**: Read and agree to follow our [Code of Conduct](code-of-conduct.md)

### Finding Ways to Contribute

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new functionality
- **Documentation**: Improve guides, examples, and API documentation
- **Code Contributions**: Fix bugs, implement features, or optimize performance
- **Testing**: Add test cases or improve test coverage
- **Examples**: Create usage examples and tutorials

## Development Process

### Repository Structure

uubed is organized as a multi-repository project:

- **`uubed-py/`**: Python implementation and API
- **`uubed-rs/`**: Rust implementation for performance-critical components
- **`uubed-docs/`**: Documentation and guides

### Branch Strategy

- **`main`**: Stable release branch
- **`develop`**: Development branch for integration
- **`feature/*`**: Feature development branches
- **`bugfix/*`**: Bug fix branches
- **`hotfix/*`**: Critical fixes for production issues

### Workflow

1. **Fork the Repository**: Create your own fork
2. **Create a Branch**: Use descriptive branch names
3. **Make Changes**: Follow coding standards and best practices
4. **Test Thoroughly**: Ensure all tests pass
5. **Submit a Pull Request**: Provide clear description and context

## Coding Standards

### Python Code Style

We use these tools for Python code quality:

- **Formatting**: `ruff format` (Black-compatible)
- **Linting**: `ruff check`
- **Type Checking**: `mypy`
- **Testing**: `pytest`

#### Code Style Rules

```python
# Good: Clear function names and type hints
def encode_embedding(
    data: np.ndarray, 
    method: EncodingMethod = "auto"
) -> str:
    """Encode embedding data using specified method.
    
    Args:
        data: Input embedding array
        method: Encoding method to use
        
    Returns:
        Encoded string representation
        
    Raises:
        UubedValidationError: If input validation fails
    """
    validate_embedding_input(data, method)
    return _encode_impl(data, method)

# Bad: Unclear names and missing documentation
def enc(d, m="auto"):
    return _enc_impl(d, m)
```

#### Documentation Standards

```python
# Use Google-style docstrings
def process_batch(
    embeddings: List[np.ndarray],
    batch_size: int = 1000,
    **kwargs: Any
) -> List[str]:
    """Process embeddings in batches for efficiency.
    
    This function processes large lists of embeddings by splitting them
    into smaller batches to manage memory usage and improve performance.
    
    Args:
        embeddings: List of embedding arrays to process
        batch_size: Maximum number of embeddings per batch
        **kwargs: Additional arguments passed to encoding function
        
    Returns:
        List of encoded strings corresponding to input embeddings
        
    Raises:
        UubedValidationError: If any embedding fails validation
        UubedResourceError: If batch processing exceeds memory limits
        
    Example:
        >>> embeddings = [np.random.rand(128) for _ in range(5000)]
        >>> encoded = process_batch(embeddings, batch_size=500)
        >>> len(encoded) == len(embeddings)
        True
    """
```

### Rust Code Style

Follow standard Rust conventions:

- **Formatting**: `cargo fmt`
- **Linting**: `cargo clippy`
- **Documentation**: `cargo doc`
- **Testing**: `cargo test`

#### Rust Style Guidelines

```rust
// Good: Well-documented public API
/// Encodes a vector using the specified Q64 variant.
///
/// # Arguments
///
/// * `data` - The input vector as a slice of bytes
/// * `method` - The encoding method to use
///
/// # Returns
///
/// Returns `Ok(String)` with the encoded result, or `Err(UubedError)`
/// if encoding fails.
///
/// # Examples
///
/// ```
/// use uubed_rs::{encode_q64, Q64Method};
///
/// let data = vec![1, 2, 3, 4];
/// let encoded = encode_q64(&data, Q64Method::Eq64)?;
/// ```
pub fn encode_q64(data: &[u8], method: Q64Method) -> Result<String, UubedError> {
    validate_input(data)?;
    match method {
        Q64Method::Eq64 => encode_eq64(data),
        Q64Method::Shq64 => encode_shq64(data),
        // ... other methods
    }
}

// Bad: Poor naming and no documentation
pub fn enc(d: &[u8], m: u8) -> Result<String, Box<dyn Error>> {
    // implementation
}
```

### Error Handling

#### Python Error Handling

```python
# Use custom exception hierarchy
from uubed.exceptions import UubedValidationError, UubedEncodingError

def validate_embedding_input(data: np.ndarray, method: str) -> None:
    """Validate embedding input with detailed error messages."""
    if data is None:
        raise UubedValidationError(
            "Embedding cannot be None",
            parameter="data",
            expected="numpy array or list",
            received="None"
        )
    
    if data.size == 0:
        raise UubedValidationError(
            "Embedding cannot be empty",
            parameter="data", 
            expected="non-empty array",
            received="empty array"
        )
```

#### Rust Error Handling

```rust
// Use Result types and custom error enum
#[derive(Debug, thiserror::Error)]
pub enum UubedError {
    #[error("Validation failed: {message}")]
    Validation { message: String },
    
    #[error("Encoding failed: {message}")]
    Encoding { message: String },
    
    #[error("IO error: {source}")]
    Io { #[from] source: std::io::Error },
}

pub fn validate_input(data: &[u8]) -> Result<(), UubedError> {
    if data.is_empty() {
        return Err(UubedError::Validation {
            message: "Input data cannot be empty".to_string(),
        });
    }
    Ok(())
}
```

## Testing Guidelines

### Test Structure

#### Python Tests

```python
# Use descriptive test class and method names
class TestEmbeddingValidation:
    """Test embedding input validation functionality."""
    
    def test_valid_numpy_array_passes_validation(self):
        """Test that valid numpy arrays pass validation."""
        data = np.random.rand(128).astype(np.float32)
        # Should not raise any exception
        validate_embedding_input(data, "shq64")
    
    def test_empty_array_raises_validation_error(self):
        """Test that empty arrays raise UubedValidationError."""
        with pytest.raises(UubedValidationError, match="cannot be empty"):
            validate_embedding_input(np.array([]), "shq64")
    
    @pytest.mark.parametrize("method", ["eq64", "shq64", "t8q64"])
    def test_encoding_roundtrip_preserves_data(self, method):
        """Test encode/decode roundtrip for all methods."""
        original_data = np.random.randint(0, 256, 64, dtype=np.uint8)
        
        encoded = encode(original_data, method=method)
        if method in ["eq64", "mq64"]:  # Lossless methods
            decoded = decode(encoded, method=method)
            np.testing.assert_array_equal(original_data, decoded)
```

#### Rust Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_valid_input_encodes_successfully() {
        let data = vec![1, 2, 3, 4, 5];
        let result = encode_q64(&data, Q64Method::Eq64);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_empty_input_returns_error() {
        let data = vec![];
        let result = encode_q64(&data, Q64Method::Eq64);
        assert!(matches!(result, Err(UubedError::Validation { .. })));
    }
    
    #[test]
    fn test_roundtrip_preserves_data() {
        let original = vec![10, 20, 30, 40];
        let encoded = encode_q64(&original, Q64Method::Eq64).unwrap();
        let decoded = decode_q64(&encoded, Q64Method::Eq64).unwrap();
        assert_eq!(original, decoded);
    }
}
```

### Test Coverage

Aim for high test coverage:

- **Python**: Use `pytest-cov` to measure coverage
- **Rust**: Use `cargo tarpaulin` for coverage analysis

```bash
# Python coverage
uvx hatch run test-cov

# Rust coverage  
cargo tarpaulin --out Html
```

### Performance Tests

```python
# Python benchmarks
@pytest.mark.benchmark
def test_encoding_performance():
    """Benchmark encoding performance."""
    data = np.random.rand(1000, 512)
    
    start_time = time.time()
    for embedding in data:
        encode(embedding, method="shq64")
    duration = time.time() - start_time
    
    rate = len(data) / duration
    assert rate > 100, f"Encoding rate too slow: {rate:.2f} embeddings/sec"
```

```rust
// Rust benchmarks
#[bench]
fn bench_encode_eq64(b: &mut Bencher) {
    let data = vec![42u8; 1024];
    b.iter(|| {
        black_box(encode_q64(&data, Q64Method::Eq64).unwrap())
    });
}
```

## Pull Request Process

### Before Submitting

1. **Update Dependencies**: Ensure you're using latest versions
2. **Run All Tests**: Make sure nothing is broken
3. **Check Code Style**: Run formatters and linters
4. **Update Documentation**: Add or update relevant docs
5. **Add Tests**: Include tests for new functionality

### PR Description Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Added unit tests for new functionality
- [ ] All existing tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced

## Related Issues
Closes #123
```

### Review Process

1. **Automated Checks**: CI must pass
2. **Code Review**: At least one maintainer review required
3. **Testing**: Reviewer may run additional tests
4. **Documentation**: Check that docs are updated appropriately
5. **Approval**: Maintainer approval required for merge

## Communication

### GitHub Issues

When creating issues:

- **Use Templates**: Follow the provided issue templates
- **Be Specific**: Include reproduction steps for bugs
- **Provide Context**: Explain the use case for features
- **Include Environment**: OS, Python/Rust versions, etc.

### Discussions

Use GitHub Discussions for:

- **Questions**: General usage questions
- **Ideas**: Brainstorming new features
- **Announcements**: Project updates and releases

### Code Review Feedback

When giving feedback:

- **Be Constructive**: Suggest improvements, don't just point out problems
- **Be Specific**: Reference exact lines or provide examples
- **Be Respectful**: Remember there's a person behind the code

When receiving feedback:

- **Be Open**: Consider all suggestions carefully
- **Ask Questions**: If feedback isn't clear, ask for clarification
- **Iterate**: Make changes and push updates promptly

## Recognition

Contributors are recognized in:

- **CONTRIBUTORS.md**: All contributors listed
- **Release Notes**: Major contributions highlighted
- **Documentation**: Examples and guides credited to authors

## Questions?

If you have questions about contributing:

1. Check this guide and the [setup documentation](setup.md)
2. Search existing issues and discussions
3. Create a new discussion or issue
4. Reach out to maintainers directly if needed

Thank you for contributing to uubed! 🎉

## Related Topics

- [Development Setup](setup.md)
- [Code of Conduct](code-of-conduct.md)
- [API Reference](../api.md)