# Testing Guide for torch2trt

This document describes how to run tests for the torch2trt project.

## Prerequisites

### System Requirements

- NVIDIA GPU (for most tests)
- CUDA Toolkit installed
- TensorRT installed (version 8.x or 10.x)
- Python 3.8 or higher

### Python Dependencies

Install the testing dependencies:

```bash
pip install -r requirements-test.txt
```

Install torch2trt in development mode:

```bash
pip install -e .
```

## Test Organization

Tests are organized into three categories:

### 1. Converter Tests (`tests/converter_tests/`)
Tests for individual PyTorch operator converters. These tests verify that specific operations (like Conv2d, ReLU, etc.) are correctly converted to TensorRT.

### 2. Feature Tests (`tests/feature_tests/`)
Tests for torch2trt features like:
- Dynamic shapes
- Model saving/loading
- FP16 mode
- Flattening utilities
- Dataset calibration

### 3. Model Tests (`tests/model_tests/`)
End-to-end tests for complete model conversions:
- TorchVision models (ResNet, VGG, etc.)
- Timm models (MaxViT, etc.)
- Segmentation models

## Running Tests

### Quick Start

There are three ways to run tests:

**1. Using the Python test runner (simplest):**
```bash
# Run all tests
python run_tests.py

# Run specific categories
python run_tests.py --converters
python run_tests.py --features
python run_tests.py --models

# Run with coverage
python run_tests.py --coverage

# Run in parallel
python run_tests.py --parallel
```

**2. Using pytest directly:**
```bash
pytest tests/
```

**3. Using the Makefile:**
```bash
make test
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Categories

```bash
# Run only converter tests
pytest tests/converter_tests/

# Run only feature tests
pytest tests/feature_tests/

# Run only model tests
pytest tests/model_tests/
```

### Run Specific Test Files

```bash
# Run a specific test file
pytest tests/feature_tests/test_save_load.py

# Run with verbose output
pytest tests/feature_tests/test_save_load.py -v
```

### Run Specific Test Functions

```bash
# Run a specific test function
pytest tests/converter_tests/test_converters.py::test_add

# Run tests matching a pattern
pytest tests/converter_tests/test_converters.py -k "test_conv"
```

### Parallel Test Execution

For faster test execution, run tests in parallel:

```bash
# Run tests on 4 CPU cores
pytest tests/ -n 4

# Run tests on all available cores
pytest tests/ -n auto
```

### Coverage Report

Generate a coverage report:

```bash
# Run tests with coverage
pytest tests/ --cov=torch2trt --cov-report=html

# View the coverage report
# Open htmlcov/index.html in a browser
```

## Using Makefile

For convenience, common test commands are available in the Makefile:

```bash
# Run all tests
make test

# Run specific test category
make test-converters
make test-features
make test-models

# Run tests with coverage
make test-coverage

# Clean test artifacts
make clean-test
```

## Test Markers

Tests can be marked with pytest markers for selective execution:

```bash
# Run only GPU tests
pytest -m gpu

# Run all except slow tests
pytest -m "not slow"

# Run converter tests
pytest -m converter
```

## Continuous Integration

Tests are automatically run via GitHub Actions on:
- Push to main/master/develop branches
- Pull requests to main/master/develop branches

Note: CI runs with CPU-only PyTorch, so only a subset of tests that don't require CUDA/TensorRT will execute successfully.

## Troubleshooting

### CUDA Out of Memory

If tests fail with CUDA out of memory errors:

```bash
# Run tests sequentially
pytest tests/ -n 0

# Or run smaller test subsets
pytest tests/feature_tests/
```

### TensorRT Not Found

Ensure TensorRT is properly installed and the library path is set:

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.x/dist-packages/tensorrt:$LD_LIBRARY_PATH
```

### Missing GPU

Some tests require a GPU. If running on CPU-only system, skip GPU tests:

```bash
pytest tests/ -m "not gpu"
```

## Writing New Tests

### Test Structure

Follow the existing test pattern:

```python
import pytest
import torch
import torch2trt


def test_my_feature():
    # Create a simple PyTorch model
    model = torch.nn.Conv2d(3, 3, 1).cuda().eval()
    
    # Create sample input
    x = torch.randn(1, 3, 224, 224).cuda()
    
    # Convert to TensorRT
    model_trt = torch2trt.torch2trt(model, [x])
    
    # Test outputs match
    y = model(x)
    y_trt = model_trt(x)
    
    assert torch.allclose(y, y_trt, atol=1e-3, rtol=1e-3)
```

### Test Markers

Add appropriate markers to your tests:

```python
@pytest.mark.gpu
def test_requires_gpu():
    pass

@pytest.mark.slow
def test_long_running():
    pass
```

### Parameterized Tests

Use pytest parametrization for testing multiple configurations:

```python
@pytest.mark.parametrize("fp16_mode,tol", [(False, 1e-3), (True, 1e-2)])
def test_with_params(fp16_mode, tol):
    # Test code here
    pass
```

## Additional Resources

- [PyTest Documentation](https://docs.pytest.org/)
- [torch2trt Documentation](https://nvidia-ai-iot.github.io/torch2trt)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
