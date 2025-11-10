# Automated Testing Infrastructure - Quick Reference

## What Was Added

This automated testing infrastructure provides comprehensive test automation for torch2trt.

### Files Created

1. **pytest.ini** - Pytest configuration
2. **.github/workflows/tests.yml** - GitHub Actions CI/CD workflow
3. **requirements-test.txt** - Test dependencies
4. **TESTING.md** - Comprehensive testing guide
5. **CI.md** - Continuous integration documentation
6. **Makefile** - Convenient test commands
7. **run_tests.py** - Python test runner script

### Files Modified

1. **.gitignore** - Added test artifacts
2. **README.md** - Added testing section

## Quick Start

### Install Test Dependencies

```bash
pip install -r requirements-test.txt
```

### Run Tests (3 ways)

**Option 1: Python script (easiest)**
```bash
python run_tests.py
python run_tests.py --features
python run_tests.py --coverage
```

**Option 2: Pytest directly**
```bash
pytest tests/
pytest tests/feature_tests/
pytest tests/converter_tests/ -v
```

**Option 3: Makefile**
```bash
make test
make test-features
make test-coverage
```

## Test Organization

- **tests/converter_tests/** - Individual operator converter tests
- **tests/feature_tests/** - Feature-level tests (save/load, dynamic shapes, etc.)
- **tests/model_tests/** - End-to-end model conversion tests

## Continuous Integration

Tests run automatically on:
- Push to main/master/develop
- Pull requests
- Manual workflow dispatch

View status: `.github/workflows/tests.yml`

## Key Features

✅ Automated CI/CD via GitHub Actions
✅ Test categorization with pytest markers
✅ Parallel test execution support
✅ Code coverage reporting
✅ Multiple ways to run tests
✅ Comprehensive documentation
✅ Easy-to-use test runner script

## Important Notes

⚠️ Most tests require CUDA and TensorRT
⚠️ CI runs limited CPU-only tests
⚠️ For full coverage, run on GPU-enabled system

## Documentation

- **TESTING.md** - Complete testing guide
- **CI.md** - CI/CD documentation
- **pytest.ini** - Configuration details

## Common Commands

```bash
# Run all tests
make test

# Run specific category
make test-features

# Run with coverage
make test-coverage

# Clean test artifacts
make clean-test

# Show help
make help
```

## Test Markers

Filter tests by category:

```bash
pytest -m converter  # Converter tests
pytest -m feature    # Feature tests
pytest -m model      # Model tests
pytest -m gpu        # GPU-required tests
pytest -m slow       # Slow tests
```

## Next Steps

1. Review TESTING.md for detailed instructions
2. Run `make test` to verify setup
3. Check CI.md for workflow customization
4. Add new tests following existing patterns

## Support

For issues or questions:
- Check TESTING.md troubleshooting section
- Review existing test examples in tests/
- Refer to pytest documentation
