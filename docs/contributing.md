---
layout: default
title: Contributing
nav_order: 9
description: "Guide for contributing to the UUBED project across all repositories"
---

# Contributing to UUBED

Thank you for your interest in contributing to UUBED! This guide will help you get started with contributing to any part of the project.

## Project Structure

The UUBED project is organized across multiple repositories:

- **[uubed](https://github.com/twardoch/uubed)** - Main project hub and coordination
- **[uubed-rs](https://github.com/twardoch/uubed-rs)** - Rust core implementation
- **[uubed-py](https://github.com/twardoch/uubed-py)** - Python bindings and API
- **[uubed-docs](https://github.com/twardoch/uubed-docs)** - Documentation site

## Getting Started

### 1. Fork and Clone

Fork the appropriate repository and clone it locally:

```bash
git clone https://github.com/YOUR_USERNAME/uubed-py.git
cd uubed-py
```

### 2. Set Up Development Environment

#### For Python Development (uubed-py):

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies (using uv is recommended)
uv pip install -e ".[dev]"

# Or with regular pip
pip install -e ".[dev]"
```

#### For Rust Development (uubed-rs):

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build the project
cargo build --release

# Run tests
cargo test
```

#### For Documentation (uubed-docs):

```bash
# Install Jekyll dependencies
bundle install

# Serve documentation locally
bundle exec jekyll serve
```

### 3. Make Your Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the coding standards

3. Write or update tests

4. Update documentation if needed

### 4. Run Tests

#### Python:
```bash
# Using hatch (recommended)
uvx hatch test

# Or with pytest
python -m pytest tests/
```

#### Rust:
```bash
cargo test
cargo clippy
cargo fmt --check
```

## Coding Standards

### Python Code Style

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Docstrings for all public functions

Example:
```python
def encode_embedding(
    data: np.ndarray,
    method: str = "eq64"
) -> str:
    """
    Encode an embedding vector to a position-safe string.
    
    Args:
        data: The embedding vector to encode
        method: Encoding method to use
        
    Returns:
        Position-safe encoded string
    """
    ...
```

### Rust Code Style

- Follow Rust formatting guidelines
- Use `cargo fmt` before committing
- Add documentation comments

Example:
```rust
/// Encode bytes using Q64 position-safe encoding
/// 
/// # Arguments
/// * `data` - The bytes to encode
/// 
/// # Returns
/// The encoded string
pub fn q64_encode(data: &[u8]) -> String {
    // Implementation
}
```

### Documentation Style

- Use clear, concise language
- Include code examples
- Follow Jekyll/Markdown conventions
- Add front matter to all pages

## Testing

### Writing Tests

Add tests for new features:

```python
# Python test example
def test_new_feature():
    result = new_feature(input_data)
    assert result == expected_output
```

```rust
// Rust test example
#[test]
fn test_new_feature() {
    let result = new_feature(&input_data);
    assert_eq!(result, expected_output);
}
```

### Performance Benchmarks

When adding performance-critical code:

1. Add benchmarks to track performance
2. Compare with existing implementations
3. Document performance characteristics

## Documentation

### Code Documentation

- All public APIs must have docstrings/doc comments
- Include examples in documentation
- Explain complex algorithms

### User Documentation

- Update relevant .md files in docs/
- Add examples for new features
- Update the API reference

## Pull Request Process

### 1. Before Submitting

- [ ] All tests pass
- [ ] Code is formatted correctly
- [ ] Documentation is updated
- [ ] CHANGELOG.md entry added
- [ ] Commit messages are clear

### 2. PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Performance benchmarks run

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG updated
```

### 3. Review Process

- Address reviewer feedback promptly
- Keep PRs focused and manageable
- Be patient and respectful

## Reporting Issues

### Bug Reports

When reporting bugs, include:

1. Clear description of the issue
2. Minimal reproducible example
3. System information:
   ```python
   import sys
   import platform
   import uubed
   
   print(f"Python: {sys.version}")
   print(f"Platform: {platform.platform()}")
   print(f"UUBED: {uubed.__version__}")
   ```
4. Full error traceback

### Feature Requests

For feature requests:

1. Search existing issues first
2. Clearly describe the use case
3. Provide examples if possible
4. Explain why it would benefit the project

## Development Workflow

### For Cross-Repository Changes

If your change affects multiple repositories:

1. Create branches in all affected repos
2. Submit PRs in dependency order:
   - uubed-rs → uubed-py → uubed-docs
3. Link related PRs in descriptions

### Working with FFI

When modifying the Python-Rust interface:

1. Update Rust functions in uubed-rs
2. Regenerate Python bindings
3. Update Python wrapper code
4. Test thoroughly across platforms

## Release Process

Releases follow semantic versioning:

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes

### Release Checklist

1. [ ] All tests pass
2. [ ] Documentation updated
3. [ ] CHANGELOG.md updated
4. [ ] Version bumped appropriately
5. [ ] Release notes written

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on what is best for the community

### Communication

- Use GitHub Issues for bugs/features
- Use GitHub Discussions for questions
- Be clear and concise in communication
- Provide context and examples

## Getting Help

If you need help:

1. Check the [documentation](https://uubed.readthedocs.io)
2. Search [GitHub Issues](https://github.com/twardoch/uubed/issues)
3. Ask in [GitHub Discussions](https://github.com/twardoch/uubed/discussions)
4. Reach out to maintainers

## Recognition

Contributors are recognized in:

- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to UUBED!