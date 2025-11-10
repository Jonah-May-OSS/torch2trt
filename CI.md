# Continuous Integration

torch2trt uses GitHub Actions for continuous integration testing.

## Workflow Status

The main test workflow runs automatically on:
- Pushes to `main`, `master`, or `develop` branches
- Pull requests targeting `main`, `master`, or `develop` branches
- Manual workflow dispatch

## Workflow Details

### Test Matrix

Tests run on multiple Python versions:
- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11

### Test Execution

Due to the dependency on CUDA and TensorRT, the CI environment runs with CPU-only PyTorch and executes a limited subset of tests:
- Version utility tests
- Flattener tests
- Dataset tests

These tests verify basic functionality without requiring GPU hardware.

### Full Test Suite

For comprehensive testing including:
- Converter tests (require TensorRT)
- Model conversion tests (require CUDA)
- Performance benchmarks (require GPU)

Tests should be run on a system with:
- NVIDIA GPU
- CUDA Toolkit
- TensorRT installed

See [TESTING.md](TESTING.md) for complete testing instructions.

## Adding Workflow Badge to README

To add a workflow status badge to the README, use:

```markdown
[![Tests](https://github.com/Jonah-May-OSS/torch2trt/actions/workflows/tests.yml/badge.svg)](https://github.com/Jonah-May-OSS/torch2trt/actions/workflows/tests.yml)
```

## Customizing the Workflow

The workflow file is located at `.github/workflows/tests.yml`. You can customize:
- Python versions to test
- Test execution commands
- Additional validation steps
- Artifact uploads
- Notification settings

## Local Testing Before Push

Before pushing changes, run tests locally:

```bash
# Quick validation
python run_tests.py --features

# Full test suite (requires GPU)
python run_tests.py --coverage
```

This helps catch issues before they trigger CI failures.
