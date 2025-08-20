#!/usr/bin/env python3
"""
Test runner for CAAF
"""

import sys
import pytest

def main():
    """Run all tests with coverage."""
    args = [
        "-v",  # Verbose
        "--cov=src",  # Coverage for src directory
        "--cov-report=term-missing",  # Show missing lines
        "--cov-report=html",  # Generate HTML report
        "tests/"  # Test directory
    ]
    
    # Add any command line arguments
    args.extend(sys.argv[1:])
    
    # Run tests
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
        print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n❌ Some tests failed")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())