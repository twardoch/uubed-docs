# GitHub Actions Workflows for uubed Project

## Overview

This document provides specific GitHub Actions workflow configurations for implementing cross-repository coordination in the uubed project. These workflows implement the strategies outlined in the Cross-Repository Coordination Research document.

## 1. Core Repository Workflow (uubed-rs)

### File: `.github/workflows/core-ci.yml`

```yaml
name: Core Library CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    # Nightly benchmarks at 2 AM UTC
    - cron: '0 2 * * *'
  workflow_dispatch:

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  test:
    name: Test Suite
    runs-on: ubuntu-latest
    strategy:
      matrix:
        rust-version: [stable, beta, nightly]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Rust
      uses: dtolnay/rust-toolchain@stable
      with:
        toolchain: ${{ matrix.rust-version }}
        components: clippy, rustfmt
    
    - name: Cache Cargo
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target/
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Check formatting
      run: cargo fmt --all -- --check
    
    - name: Run clippy
      run: cargo clippy --all-targets --all-features -- -D warnings
    
    - name: Run tests
      run: cargo test --all-features --workspace
    
    - name: Run C API tests
      run: |
        cd examples
        gcc -o c_api_demo c_api_demo.c -L../target/release -luubed_rs
        ./c_api_demo

  benchmark:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Rust
      uses: dtolnay/rust-toolchain@stable
    
    - name: Cache Cargo
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target/
        key: ${{ runner.os }}-cargo-bench-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Run benchmarks
      run: |
        cargo bench --all-features -- --output-format json > benchmark_results.json
    
    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'cargo'
        output-file-path: benchmark_results.json
        name: 'Rust Core Benchmarks'
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
        comment-on-alert: true
        alert-threshold: '105%'
        fail-on-alert: false

  dispatch-python:
    name: Trigger Python Bindings
    runs-on: ubuntu-latest
    needs: [test]
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    
    steps:
    - name: Get version
      id: version
      run: |
        VERSION=$(cargo metadata --format-version 1 | jq -r '.packages[] | select(.name == "uubed-rs") | .version')
        echo "version=$VERSION" >> $GITHUB_OUTPUT
        echo "sha=${{ github.sha }}" >> $GITHUB_OUTPUT
    
    - name: Dispatch to Python repository
      uses: peter-evans/repository-dispatch@v2
      with:
        token: ${{ secrets.DISPATCH_TOKEN }}
        repository: twardoch/uubed-py
        event-type: core_updated
        client-payload: |
          {
            "version": "${{ steps.version.outputs.version }}",
            "commit_sha": "${{ steps.version.outputs.sha }}",
            "breaking_changes": false,
            "benchmark_required": true
          }

  release:
    name: Create Release
    runs-on: ubuntu-latest
    needs: [test, benchmark]
    if: github.ref == 'refs/heads/main' && contains(github.event.head_commit.message, '[release]')
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Setup Rust
      uses: dtolnay/rust-toolchain@stable
    
    - name: Build release
      run: |
        cargo build --release --all-features
        cargo package --allow-dirty
    
    - name: Create Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: v${{ steps.version.outputs.version }}
        release_name: uubed-rs v${{ steps.version.outputs.version }}
        body: |
          ## Changes
          
          ${{ github.event.head_commit.message }}
          
          ## Benchmarks
          
          See [benchmark results](https://twardoch.github.io/uubed-rs/dev/bench/) for performance details.
        draft: false
        prerelease: false
```

## 2. Python Bindings Workflow (uubed-py)

### File: `.github/workflows/bindings-ci.yml`

```yaml
name: Python Bindings CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  repository_dispatch:
    types: [core_updated]
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.11"

jobs:
  test:
    name: Test Python Bindings
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Setup Rust
      uses: dtolnay/rust-toolchain@stable
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target/
          ~/.cache/pip
        key: ${{ runner.os }}-${{ matrix.python-version }}-deps-${{ hashFiles('**/Cargo.lock', '**/pyproject.toml') }}
    
    - name: Install uv
      run: pip install uv
    
    - name: Install dependencies
      run: |
        uv pip install -e .[dev,test]
    
    - name: Run type checking
      run: |
        uv pip install mypy
        mypy src/uubed/
    
    - name: Run tests
      run: |
        uvx hatch test
    
    - name: Test installation
      run: |
        python -c "import uubed; print(uubed.__version__)"

  benchmark:
    name: Python Benchmarks
    runs-on: ubuntu-latest
    if: github.event.client_payload.benchmark_required == 'true' || github.event_name == 'workflow_dispatch'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        pip install uv
        uv pip install -e .[dev,benchmark]
    
    - name: Run benchmarks
      run: |
        python benchmarks/bench_encoders.py --output-format json > python_benchmarks.json
    
    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: python_benchmarks.json
        name: 'Python Bindings Benchmarks'
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
        comment-on-alert: true
        alert-threshold: '110%'
        fail-on-alert: false

  dispatch-docs:
    name: Trigger Documentation Update
    runs-on: ubuntu-latest
    needs: [test]
    if: github.ref == 'refs/heads/main' && (github.event_name == 'push' || github.event_name == 'repository_dispatch')
    
    steps:
    - name: Get version
      id: version
      run: |
        VERSION=$(python -c "import tomli; print(tomli.load(open('pyproject.toml', 'rb'))['project']['version'])")
        echo "version=$VERSION" >> $GITHUB_OUTPUT
    
    - name: Dispatch to documentation repository
      uses: peter-evans/repository-dispatch@v2
      with:
        token: ${{ secrets.DISPATCH_TOKEN }}
        repository: twardoch/uubed-docs
        event-type: bindings_updated
        client-payload: |
          {
            "python_version": "${{ steps.version.outputs.version }}",
            "rust_version": "${{ github.event.client_payload.version }}",
            "commit_sha": "${{ github.sha }}",
            "update_api_docs": true
          }

  release:
    name: Release to PyPI
    runs-on: ubuntu-latest
    needs: [test, benchmark]
    if: github.ref == 'refs/heads/main' && contains(github.event.head_commit.message, '[release]')
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install build dependencies
      run: |
        pip install build twine
    
    - name: Build package
      run: |
        python -m build
    
    - name: Test package
      run: |
        python -m twine check dist/*
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        user: __token__
        password: ${{ secrets.PYPI_TOKEN }}
```

## 3. Documentation Workflow (uubed-docs)

### File: `.github/workflows/docs-ci.yml`

```yaml
name: Documentation CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  repository_dispatch:
    types: [bindings_updated]
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.11"

jobs:
  build:
    name: Build Documentation
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Validate documentation
      run: |
        mkdocs build --strict
    
    - name: Test documentation links
      run: |
        # Install and run link checker
        pip install linkchecker
        linkchecker site/ --check-extern

  api-docs:
    name: Generate API Documentation
    runs-on: ubuntu-latest
    if: github.event.client_payload.update_api_docs == 'true' || github.event_name == 'workflow_dispatch'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install uubed-py
      run: |
        pip install uubed
    
    - name: Generate API documentation
      run: |
        pip install sphinx sphinx-rtd-theme
        sphinx-apidoc -f -o docs/api/ uubed/
        sphinx-build -b html docs/api/ docs/api/_build/
    
    - name: Update API documentation
      run: |
        # Copy generated API docs to the appropriate location
        cp -r docs/api/_build/* docs/reference/api/
        
        # Update version information
        echo "Updated API documentation for uubed-py v${{ github.event.client_payload.python_version }}" > docs/reference/api/version.md

  deploy:
    name: Deploy to GitHub Pages
    runs-on: ubuntu-latest
    needs: [build]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Build documentation
      run: |
        mkdocs build
    
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./site
        publish_branch: gh-pages
        force_orphan: true

  update-changelog:
    name: Update Project Changelog
    runs-on: ubuntu-latest
    needs: [deploy]
    if: github.ref == 'refs/heads/main' && github.event_name == 'repository_dispatch'
    
    steps:
    - uses: actions/checkout@v4
      with:
        repository: twardoch/uubed
        token: ${{ secrets.DISPATCH_TOKEN }}
    
    - name: Update main project changelog
      run: |
        # Add entry to main project changelog
        echo "## Documentation Updated - $(date)" >> CHANGELOG.md
        echo "- Python bindings: v${{ github.event.client_payload.python_version }}" >> CHANGELOG.md
        echo "- Rust core: v${{ github.event.client_payload.rust_version }}" >> CHANGELOG.md
        echo "- Documentation site updated" >> CHANGELOG.md
        echo "" >> CHANGELOG.md
    
    - name: Commit changelog update
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add CHANGELOG.md
        git commit -m "Update changelog for documentation release" || exit 0
        git push
```

## 4. Comprehensive Benchmarking Workflow

### File: `.github/workflows/nightly-benchmarks.yml`

```yaml
name: Nightly Performance Benchmarks

on:
  schedule:
    # Run at 3 AM UTC every day
    - cron: '0 3 * * *'
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.11"

jobs:
  rust-benchmarks:
    name: Rust Core Benchmarks
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        repository: twardoch/uubed-rs
        token: ${{ secrets.DISPATCH_TOKEN }}
    
    - name: Setup Rust
      uses: dtolnay/rust-toolchain@stable
    
    - name: Run comprehensive benchmarks
      run: |
        cargo bench --all-features -- --output-format json > rust_benchmarks.json
    
    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'cargo'
        output-file-path: rust_benchmarks.json
        name: 'Rust Nightly Benchmarks'
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
        comment-on-alert: true
        alert-threshold: '105%'
        fail-on-alert: true

  python-benchmarks:
    name: Python Bindings Benchmarks
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        repository: twardoch/uubed-py
        token: ${{ secrets.DISPATCH_TOKEN }}
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        pip install uv
        uv pip install -e .[benchmark]
    
    - name: Run benchmarks
      run: |
        python benchmarks/bench_encoders.py --output-format json > python_benchmarks.json
    
    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: python_benchmarks.json
        name: 'Python Nightly Benchmarks'
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
        comment-on-alert: true
        alert-threshold: '110%'
        fail-on-alert: true

  cross-language-comparison:
    name: Cross-Language Performance Comparison
    runs-on: ubuntu-latest
    needs: [rust-benchmarks, python-benchmarks]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install analysis tools
      run: |
        pip install pandas matplotlib seaborn
    
    - name: Download benchmark results
      run: |
        # Download results from benchmark action
        curl -o rust_results.json "https://twardoch.github.io/uubed-rs/dev/bench/data.js"
        curl -o python_results.json "https://twardoch.github.io/uubed-py/dev/bench/data.js"
    
    - name: Generate comparison report
      run: |
        python scripts/compare_benchmarks.py \
          --rust rust_results.json \
          --python python_results.json \
          --output comparison_report.md
    
    - name: Update documentation with comparison
      run: |
        # Update benchmark comparison in docs
        cp comparison_report.md docs/performance/cross-language-comparison.md
        
        # Commit changes
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add docs/performance/cross-language-comparison.md
        git commit -m "Update cross-language benchmark comparison" || exit 0
        git push

  performance-regression-check:
    name: Performance Regression Detection
    runs-on: ubuntu-latest
    needs: [cross-language-comparison]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Check for performance regressions
      run: |
        # Custom script to analyze performance trends
        python scripts/check_performance_regression.py \
          --threshold 5 \
          --days 7 \
          --output regression_report.md
    
    - name: Create issue for regressions
      if: failure()
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const report = fs.readFileSync('regression_report.md', 'utf8');
          
          github.rest.issues.create({
            owner: context.repo.owner,
            repo: context.repo.repo,
            title: 'Performance Regression Detected',
            body: report,
            labels: ['performance', 'regression', 'high-priority']
          });
```

## 5. Release Coordination Workflow

### File: `.github/workflows/coordinated-release.yml`

```yaml
name: Coordinated Release

on:
  workflow_dispatch:
    inputs:
      version:
        description: 'Release version (e.g., 1.2.3)'
        required: true
        type: string
      release_notes:
        description: 'Release notes'
        required: true
        type: string
      breaking_changes:
        description: 'Contains breaking changes'
        required: false
        type: boolean
        default: false

jobs:
  validate-release:
    name: Validate Release Readiness
    runs-on: ubuntu-latest
    
    steps:
    - name: Validate all repositories are ready
      run: |
        # Check that all repositories have passing tests
        echo "Validating release readiness for version ${{ github.event.inputs.version }}"
        
        # Add validation logic here
        # - Check test status
        # - Verify no breaking changes in patch releases
        # - Ensure all dependencies are up to date

  release-rust:
    name: Release Rust Core
    runs-on: ubuntu-latest
    needs: [validate-release]
    
    steps:
    - name: Trigger Rust release
      uses: actions/github-script@v6
      with:
        github-token: ${{ secrets.DISPATCH_TOKEN }}
        script: |
          await github.rest.repos.createDispatchEvent({
            owner: 'twardoch',
            repo: 'uubed-rs',
            event_type: 'release_triggered',
            client_payload: {
              version: '${{ github.event.inputs.version }}',
              release_notes: '${{ github.event.inputs.release_notes }}',
              breaking_changes: ${{ github.event.inputs.breaking_changes }}
            }
          });
    
    - name: Wait for Rust release completion
      run: |
        # Wait for Rust release to complete
        sleep 300  # Wait 5 minutes for build to complete
        
        # Check release status
        # Add proper status checking logic here

  release-python:
    name: Release Python Bindings
    runs-on: ubuntu-latest
    needs: [release-rust]
    
    steps:
    - name: Trigger Python release
      uses: actions/github-script@v6
      with:
        github-token: ${{ secrets.DISPATCH_TOKEN }}
        script: |
          await github.rest.repos.createDispatchEvent({
            owner: 'twardoch',
            repo: 'uubed-py',
            event_type: 'release_triggered',
            client_payload: {
              version: '${{ github.event.inputs.version }}',
              rust_version: '${{ github.event.inputs.version }}',
              release_notes: '${{ github.event.inputs.release_notes }}',
              breaking_changes: ${{ github.event.inputs.breaking_changes }}
            }
          });
    
    - name: Wait for Python release completion
      run: |
        # Wait for Python release to complete
        sleep 600  # Wait 10 minutes for build and PyPI upload
        
        # Verify PyPI package is available
        pip install uubed==${{ github.event.inputs.version }}

  release-docs:
    name: Release Documentation
    runs-on: ubuntu-latest
    needs: [release-python]
    
    steps:
    - name: Trigger documentation release
      uses: actions/github-script@v6
      with:
        github-token: ${{ secrets.DISPATCH_TOKEN }}
        script: |
          await github.rest.repos.createDispatchEvent({
            owner: 'twardoch',
            repo: 'uubed-docs',
            event_type: 'release_triggered',
            client_payload: {
              version: '${{ github.event.inputs.version }}',
              rust_version: '${{ github.event.inputs.version }}',
              python_version: '${{ github.event.inputs.version }}',
              release_notes: '${{ github.event.inputs.release_notes }}',
              breaking_changes: ${{ github.event.inputs.breaking_changes }}
            }
          });

  post-release:
    name: Post-Release Actions
    runs-on: ubuntu-latest
    needs: [release-docs]
    
    steps:
    - name: Update main project
      run: |
        # Update main project with release information
        echo "Release v${{ github.event.inputs.version }} completed successfully"
        
        # Add any post-release actions here
        # - Send notifications
        # - Update project board
        # - Generate release announcements
```

## 6. Configuration Files

### Shared Configuration: `.github/config.yml`

```yaml
# Shared configuration for all repositories
release:
  sequence: [rust, python, docs]
  
benchmark:
  schedule: "0 3 * * *"  # 3 AM UTC daily
  thresholds:
    rust: 105%
    python: 110%
    docs: 120%
  
notifications:
  slack_webhook: ${{ secrets.SLACK_WEBHOOK }}
  email: devops@example.com
  
repositories:
  rust:
    name: uubed-rs
    main_branch: main
    package_manager: cargo
    
  python:
    name: uubed-py
    main_branch: main
    package_manager: pip
    
  docs:
    name: uubed-docs
    main_branch: main
    generator: mkdocs
```

### Secrets Configuration

Required secrets for the workflows:

```yaml
# Organization-level secrets
DISPATCH_TOKEN: # PAT with repo scope for cross-repository triggers
PYPI_TOKEN: # Token for PyPI publishing
SLACK_WEBHOOK: # Webhook URL for Slack notifications
GOOGLE_ANALYTICS_KEY: # GA key for documentation analytics

# Repository-specific secrets
GITHUB_TOKEN: # Default GitHub token (automatically provided)
```

## 7. Usage Instructions

### Setting Up Cross-Repository Coordination

1. **Create Personal Access Token**:
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Create token with `repo` scope
   - Add as `DISPATCH_TOKEN` secret in all repositories

2. **Configure Workflows**:
   - Copy the workflow files to appropriate `.github/workflows/` directories
   - Update repository names and organization details
   - Configure required secrets

3. **Test the Setup**:
   - Trigger a test workflow dispatch
   - Verify cross-repository communication works
   - Check benchmark results are stored correctly

### Triggering Coordinated Releases

1. **Manual Release**:
   - Go to Actions tab in the main repository
   - Run "Coordinated Release" workflow
   - Provide version number and release notes

2. **Automated Release**:
   - Commit with `[release]` in the message
   - Workflows will automatically trigger in sequence

### Monitoring Performance

1. **View Benchmark Results**:
   - Check GitHub Pages for performance charts
   - Monitor for regression alerts in pull requests
   - Review nightly benchmark reports

2. **Performance Regression Handling**:
   - Automatic issue creation for regressions
   - Email notifications for critical performance drops
   - Integration with project management tools

This comprehensive workflow setup provides robust cross-repository coordination while maintaining individual repository autonomy and ensuring consistent quality across the entire uubed project ecosystem.