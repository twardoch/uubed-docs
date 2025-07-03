# Development Setup

This guide will help you set up a development environment for contributing to uubed.

## Prerequisites

### System Requirements

- **Python 3.10+** (3.11 or 3.12 recommended)
- **Rust 1.70+** (for native extensions)
- **Git** (for version control)
- **Docker** (optional, for testing)

### Development Tools

```bash
# Install uv (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Node.js (for documentation)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

## Repository Setup

### Clone the Repository

```bash
# Clone the main repository
git clone https://github.com/twardoch/uubed.git
cd uubed

# Initialize submodules
git submodule update --init --recursive
```

### Project Structure

```
uubed/
├── uubed-py/          # Python implementation
├── uubed-rs/          # Rust implementation  
├── uubed-docs/        # Documentation
├── README.md          # Main project README
├── PLAN.md            # Development plan
└── TODO.md            # Current tasks
```

## Python Development Setup

### Using uv (Recommended)

```bash
cd uubed-py

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in development mode
uv pip install -e ".[test,dev]"

# Install pre-commit hooks
uv pip install pre-commit
pre-commit install
```

### Using Hatch

```bash
cd uubed-py

# Install hatch if not already installed
pip install hatch

# Run tests
hatch run test

# Run linting
hatch run lint

# Format code
hatch run format
```

### Verify Python Setup

```bash
# Run tests to verify setup
uvx hatch test

# Check that import works
python -c "import uubed; print(uubed.__version__)"

# Run benchmarks
python -m uubed.cli bench --size 10 --dims 128
```

## Rust Development Setup

### Build Rust Components

```bash
cd uubed-rs

# Build in development mode
cargo build

# Run tests
cargo test

# Run benchmarks
cargo bench
```

### IDE Setup for Rust

Add to `.vscode/settings.json`:

```json
{
    "rust-analyzer.cargo.features": ["all"],
    "rust-analyzer.checkOnSave.command": "clippy"
}
```

## Documentation Setup

### MkDocs Setup

```bash
cd uubed-docs

# Install documentation dependencies
pip install mkdocs-material mkdocstrings[python]

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

### Verify Documentation

```bash
# Check for broken links
mkdocs build 2>&1 | grep -i warning

# Serve locally and test
mkdocs serve --dev-addr 127.0.0.1:8000
```

## Development Workflow

### Code Style and Linting

#### Python

```bash
# Format code
hatch run format

# Lint code
hatch run lint

# Type checking
hatch run type-check

# Run all checks
hatch run lint:all
```

#### Rust

```bash
# Format code
cargo fmt

# Lint code
cargo clippy -- -D warnings

# Check documentation
cargo doc --no-deps
```

### Running Tests

#### Python Tests

```bash
# Run all tests
uvx hatch test

# Run specific test file
uvx hatch test tests/test_api.py

# Run with coverage
uvx hatch run test-cov

# Run performance tests
uvx hatch test -m benchmark
```

#### Rust Tests

```bash
# Run unit tests
cargo test

# Run integration tests
cargo test --test integration

# Run with output
cargo test -- --nocapture

# Run benchmarks
cargo bench
```

### Cross-Language Testing

```bash
# Test Python-Rust integration
cd uubed-py
python tests/test_native_integration.py

# Verify FFI bindings
python -c "from uubed.native_wrapper import is_native_available; print(is_native_available())"
```

## IDE Configuration

### VS Code Setup

Install recommended extensions:

```json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.black-formatter",
        "ms-python.flake8",
        "ms-python.mypy-type-checker",
        "rust-lang.rust-analyzer",
        "tamasfe.even-better-toml",
        "charliermarsh.ruff"
    ]
}
```

Configuration (`.vscode/settings.json`):

```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "rust-analyzer.cargo.loadOutDirsFromCheck": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/target": true,
        "**/.pytest_cache": true
    }
}
```

### PyCharm Setup

1. Open the `uubed-py` directory
2. Configure Python interpreter to use the virtual environment
3. Enable pytest as test runner
4. Install Rust plugin for `uubed-rs` development

## Environment Variables

Create `.env` file in project root:

```bash
# Development settings
UUBED_LOG_LEVEL=debug
UUBED_VALIDATE_INPUTS=true

# Testing settings  
PYTEST_ADDOPTS="-v --tb=short"

# Rust settings
RUST_LOG=debug
RUST_BACKTRACE=1
```

## Docker Development

### Build Development Container

```bash
# Build development image
docker build -f Dockerfile.dev -t uubed-dev .

# Run development container
docker run -it --rm \
    -v $(pwd):/workspace \
    -p 8000:8000 \
    uubed-dev bash
```

### Docker Compose

```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  uubed-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/workspace
      - /workspace/target  # Rust build cache
      - /workspace/.venv   # Python venv cache
    ports:
      - "8000:8000"  # Documentation server
      - "8080:8080"  # Development server
    environment:
      - RUST_LOG=debug
      - UUBED_LOG_LEVEL=debug
```

## Common Issues and Solutions

### Python Import Errors

```bash
# Reinstall in development mode
uv pip install -e .

# Check PYTHONPATH
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

### Rust Compilation Issues

```bash
# Clean and rebuild
cargo clean && cargo build

# Update Rust toolchain
rustup update

# Check for conflicts
cargo tree --duplicates
```

### Test Failures

```bash
# Clear caches
rm -rf .pytest_cache __pycache__ target/

# Rebuild everything
uvx hatch env prune
cargo clean
uvx hatch run test
```

### Documentation Build Issues

```bash
# Clear MkDocs cache
rm -rf site/

# Reinstall dependencies
pip install --upgrade mkdocs-material mkdocstrings

# Check for missing files
mkdocs build --strict
```

## Performance Profiling

### Python Profiling

```bash
# Profile with cProfile
python -m cProfile -o profile.stats scripts/benchmark.py

# Analyze with snakeviz
pip install snakeviz
snakeviz profile.stats
```

### Rust Profiling

```bash
# Profile with perf (Linux)
cargo build --release
perf record --call-graph=dwarf target/release/benchmark
perf report

# Profile with Instruments (macOS)
cargo instruments -t "Time Profiler" --release --bench benchmark
```

## Continuous Integration

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### GitHub Actions Testing

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test-python:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v1
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: uv pip install -e ".[test]"
      
      - name: Run tests
        run: uvx hatch test
```

## Next Steps

After setup:

1. Read the [Contributing Guidelines](guidelines.md)
2. Check the [current TODO list](../TODO.md)
3. Look for issues labeled "good first issue"
4. Join the development discussion

## Getting Help

- **Documentation**: Check this guide and [troubleshooting](../reference/troubleshooting.md)
- **Issues**: Open a GitHub issue for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Code Review**: Submit PRs for feedback on your contributions

## Related Topics

- [Contributing Guidelines](guidelines.md)
- [Code of Conduct](code-of-conduct.md)
- [API Reference](../api.md)