.PHONY: help test test-converters test-features test-models test-coverage test-fast clean-test install-test-deps

help:
	@echo "torch2trt Testing Commands"
	@echo "=========================="
	@echo ""
	@echo "make test              - Run all tests"
	@echo "make test-converters   - Run converter tests only"
	@echo "make test-features     - Run feature tests only"
	@echo "make test-models       - Run model tests only"
	@echo "make test-fast         - Run tests in parallel"
	@echo "make test-coverage     - Run tests with coverage report"
	@echo "make clean-test        - Clean test artifacts"
	@echo "make install-test-deps - Install test dependencies"

install-test-deps:
	pip install -r requirements-test.txt

test:
	pytest tests/ -v

test-converters:
	pytest tests/converter_tests/ -v

test-features:
	pytest tests/feature_tests/ -v

test-models:
	pytest tests/model_tests/ -v

test-fast:
	pytest tests/ -v -n auto

test-coverage:
	pytest tests/ --cov=torch2trt --cov-report=html --cov-report=term
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

clean-test:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf *.pth
	rm -rf .test_model.pth
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
