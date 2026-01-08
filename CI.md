# Continuous Integration

torch2trt uses GitHub Actions for continuous integration testing.

## Workflow Status

The main test workflow runs automatically on:
- Pushes to `main`, `master`, or `develop` branches
- Pull requests targeting `main`, `master`, or `develop` branches
- Manual workflow dispatch

## Workflow Details

### Test Job

The workflow runs tests on a self-hosted runner with GPU/CUDA/TensorRT support:
- Uses Python 3.13
- Runs ruff linter for code quality checks
- Runs the complete test suite including:
  - Converter tests (require TensorRT)
  - Model conversion tests (require CUDA)
  - Feature tests (full coverage)
  - Performance benchmarks (require GPU)
- Generates coverage reports
- Only runs on the main repository (not forks)

**Note**: Since torch2trt requires TensorRT, which in turn requires a GPU, all tests must run on a GPU-enabled system. There are no CPU-only tests.

### Requirements for Self-Hosted Runner

The self-hosted runner must have:
- NVIDIA GPU
- CUDA Toolkit installed
- TensorRT installed
- Python 3.13+
- GitHub Actions runner configured

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

Before pushing changes, run tests and linter locally:

```bash
# Run linter
python run_tests.py --lint
make lint

# Quick validation
python run_tests.py --features

# Full test suite (requires GPU)
python run_tests.py --coverage
```

This helps catch issues before they trigger CI failures.
