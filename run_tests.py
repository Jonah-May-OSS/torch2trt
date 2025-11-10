#!/usr/bin/env python3
"""
Simple test runner script for torch2trt.
This provides an alternative to using pytest directly or the Makefile.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --converters       # Run converter tests only
    python run_tests.py --features         # Run feature tests only
    python run_tests.py --models           # Run model tests only
    python run_tests.py --coverage         # Run with coverage
    python run_tests.py --parallel         # Run in parallel
    python run_tests.py --lint             # Run ruff linter
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run torch2trt tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--converters",
        action="store_true",
        help="Run converter tests only"
    )
    parser.add_argument(
        "--features",
        action="store_true",
        help="Run feature tests only"
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Run model tests only"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run tests with coverage report"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--lint",
        action="store_true",
        help="Run ruff linter instead of tests"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--filter",
        "-k",
        type=str,
        help="Run tests matching the given expression"
    )

    args = parser.parse_args()

    # Run linter if requested
    if args.lint:
        cmd = ["ruff", "check", "torch2trt/", "tests/", "run_tests.py"]
        print(f"Running: {' '.join(cmd)}")
        print("-" * 80)
        try:
            result = subprocess.run(cmd, check=False)
            sys.exit(result.returncode)
        except FileNotFoundError:
            print("\nError: ruff not found. Install test dependencies with:")
            print("  pip install -r requirements-test.txt")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nLinting interrupted by user")
            sys.exit(130)

    # Build pytest command
    cmd = ["pytest"]

    # Determine test path
    if args.converters:
        cmd.append("tests/converter_tests/")
    elif args.features:
        cmd.append("tests/feature_tests/")
    elif args.models:
        cmd.append("tests/model_tests/")
    else:
        cmd.append("tests/")

    # Add options
    if args.verbose:
        cmd.append("-v")

    if args.parallel:
        cmd.extend(["-n", "auto"])

    if args.coverage:
        cmd.extend([
            "--cov=torch2trt",
            "--cov-report=html",
            "--cov-report=term"
        ])

    if args.filter:
        cmd.extend(["-k", args.filter])

    # Run tests
    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)

    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print("\nError: pytest not found. Install test dependencies with:")
        print("  pip install -r requirements-test.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
