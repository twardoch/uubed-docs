# uubed-docs

Documentation for the uubed project - Position-safe encoding for substring-based search systems.

## 🚀 Quick Start

### Development Setup

```bash
# Clone the repository
git clone https://github.com/twardoch/uubed-docs.git
cd uubed-docs

# Install dependencies
pip install -r requirements.txt

# Start development server
./dev.sh serve
```

### Available Scripts

- `./dev.sh serve` - Start development server
- `./dev.sh test` - Run test suite
- `./dev.sh build` - Build documentation
- `./dev.sh release` - Create release artifacts
- `./dev.sh clean` - Clean build artifacts
- `./dev.sh version` - Show version information

## 📦 Build System

This project uses a git-tag-based semversioning system with comprehensive testing and CI/CD.

### Version Management

Versions are automatically determined from git tags:
- On a tag: `v1.0.0` → `1.0.0`
- Development: `v1.0.0-dev+abc1234` (latest tag + commit)
- No tags: `0.0.0-dev`

```bash
# Get current version
python version.py

# Get detailed version info
python version.py --info

# Update mkdocs.yml with version
python version.py --update-mkdocs
```

### Testing

Comprehensive test suite covering:
- Documentation structure validation
- Internal link checking
- Build system functionality
- Version management
- HTML validity

```bash
# Run all tests
./test.sh

# Run specific test categories
python -m pytest tests/test_version.py -v
python -m pytest tests/test_docs.py -v
python -m pytest tests/test_build.py -v
```

### Building

```bash
# Build documentation
./build.sh

# Or using MkDocs directly
mkdocs build
```

### Releases

```bash
# Create release artifacts
./release.sh

# This creates:
# - release/site/ - Built documentation
# - release/uubed-docs-{version}.tar.gz - Compressed site
# - release/uubed-docs-{version}.zip - Compressed site
# - release/version-info.txt - Version details
# - release/*.sha256 - Checksums
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

1. **CI Pipeline** (`.github/workflows/ci.yml`)
   - Runs on push/PR to main/develop
   - Tests on Python 3.9-3.12
   - Linting and formatting checks
   - Security scanning
   - Build validation

2. **Release Pipeline** (`.github/workflows/release.yml`)
   - Triggered on git tags (`v*`)
   - Creates GitHub releases
   - Deploys to GitHub Pages
   - Multi-platform release artifacts

3. **PR Pipeline** (`.github/workflows/pr.yml`)
   - Validation for pull requests
   - Preview deployments
   - Automated comments

### Creating a Release

1. **Tag a release:**
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

2. **GitHub Actions will:**
   - Run full test suite
   - Build documentation
   - Create release artifacts
   - Create GitHub release
   - Deploy to GitHub Pages

## 📁 Project Structure

```
uubed-docs/
├── docs/                 # Documentation source
├── tests/                # Test suite
├── .github/workflows/    # GitHub Actions
├── scripts/              # Utility scripts
├── version.py           # Version management
├── build.sh             # Build script
├── test.sh              # Test runner
├── release.sh           # Release script
├── dev.sh               # Development helper
├── mkdocs.yml           # MkDocs configuration
├── requirements.txt     # Python dependencies
├── pytest.ini          # Test configuration
├── pyproject.toml       # Python project config
└── .flake8             # Linting configuration
```

## 🧪 Testing Strategy

### Test Categories

1. **Version Tests** (`tests/test_version.py`)
   - Semantic version validation
   - Git tag parsing
   - Version info structure

2. **Documentation Tests** (`tests/test_docs.py`)
   - MkDocs configuration validation
   - Required pages existence
   - Markdown structure validation
   - Internal link checking

3. **Build Tests** (`tests/test_build.py`)
   - MkDocs build functionality
   - Build script execution
   - Output validation

### Running Tests

```bash
# All tests
./test.sh

# Specific test file
python -m pytest tests/test_version.py -v

# With coverage
python -m pytest tests/ --cov=. --cov-report=html

# Test markers
python -m pytest -m "not slow"  # Skip slow tests
python -m pytest -m "unit"      # Only unit tests
```

## 🛠️ Development

### Code Quality

- **Linting**: flake8
- **Formatting**: black
- **Import sorting**: isort
- **Type checking**: Built-in validation

```bash
# Format code
black .
isort .

# Check linting
flake8 .
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `./test.sh`
5. Submit a pull request

### Release Process

1. Update version in git tag
2. Push tag to trigger release
3. GitHub Actions handles the rest
4. Monitor the release workflow

## 📚 Documentation

This project uses MkDocs with the Material theme. The documentation is built from the `docs/` directory and deployed to GitHub Pages.

### Local Development

```bash
# Start development server
mkdocs serve

# Build documentation
mkdocs build

# Deploy to GitHub Pages (maintainers only)
mkdocs gh-deploy
```

## 🔧 Configuration

### Environment Variables

- `GOOGLE_ANALYTICS_KEY` - Google Analytics tracking ID
- `GITHUB_TOKEN` - GitHub token for releases

### Required Tools

- Python 3.9+
- Git
- MkDocs
- pytest

## 📊 Monitoring

### Build Status

[![CI/CD Pipeline](https://github.com/twardoch/uubed-docs/actions/workflows/ci.yml/badge.svg)](https://github.com/twardoch/uubed-docs/actions/workflows/ci.yml)
[![Release](https://github.com/twardoch/uubed-docs/actions/workflows/release.yml/badge.svg)](https://github.com/twardoch/uubed-docs/actions/workflows/release.yml)

### Links

- **Documentation**: https://twardoch.github.io/uubed-docs/
- **Repository**: https://github.com/twardoch/uubed-docs
- **Issues**: https://github.com/twardoch/uubed-docs/issues
- **Releases**: https://github.com/twardoch/uubed-docs/releases

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: Comprehensive guides available in the docs
- **Community**: Join discussions in GitHub Discussions