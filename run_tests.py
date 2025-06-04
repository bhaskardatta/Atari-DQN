#!/usr/bin/env python3
# filepath: /Users/bhaskar/Desktop/atari/run_tests.py
"""
Test runner for the adaptive RL project.
"""

import unittest
import sys
import os
from pathlib import Path
import argparse

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def discover_and_run_tests(test_dir='tests', pattern='test_*.py', verbosity=2):
    """
    Discover and run all tests in the test directory.
    
    Args:
        test_dir: Directory containing test files
        pattern: Pattern to match test files
        verbosity: Test output verbosity level
        
    Returns:
        TestResult object
    """
    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def run_specific_test(test_module, test_class=None, test_method=None, verbosity=2):
    """
    Run a specific test module, class, or method.
    
    Args:
        test_module: Test module name (e.g., 'test_dqn_components')
        test_class: Optional test class name
        test_method: Optional test method name
        verbosity: Test output verbosity level
        
    Returns:
        TestResult object
    """
    # Import the test module
    module = __import__(f'tests.{test_module}', fromlist=[test_module])
    
    # Create test suite
    loader = unittest.TestLoader()
    
    if test_class and test_method:
        # Run specific method
        suite = loader.loadTestsFromName(f'{test_class}.{test_method}', module)
    elif test_class:
        # Run specific class
        suite = loader.loadTestsFromTestCase(getattr(module, test_class))
    else:
        # Run entire module
        suite = loader.loadTestsFromModule(module)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Run tests for adaptive RL project')
    parser.add_argument('--module', type=str, help='Specific test module to run')
    parser.add_argument('--class', dest='test_class', type=str, help='Specific test class to run')
    parser.add_argument('--method', type=str, help='Specific test method to run')
    parser.add_argument('--verbose', '-v', action='count', default=1, 
                        help='Increase verbosity (use -v, -vv, or -vvv)')
    parser.add_argument('--pattern', type=str, default='test_*.py',
                        help='Pattern to match test files')
    parser.add_argument('--coverage', action='store_true',
                        help='Run with coverage analysis (requires coverage.py)')
    args = parser.parse_args()
    
    print("🧪 Running Adaptive RL Tests")
    print("=" * 40)
    
    # Check if coverage is requested
    if args.coverage:
        try:
            import coverage
            cov = coverage.Coverage()
            cov.start()
            print("📊 Coverage analysis enabled")
        except ImportError:
            print("❌ Coverage analysis requested but coverage.py not installed")
            print("   Install with: pip install coverage")
            args.coverage = False
    
    try:
        if args.module:
            # Run specific module/class/method
            print(f"Running tests from module: {args.module}")
            if args.test_class:
                print(f"  Class: {args.test_class}")
            if args.method:
                print(f"  Method: {args.method}")
            
            result = run_specific_test(
                args.module, 
                args.test_class, 
                args.method, 
                verbosity=args.verbose
            )
        else:
            # Run all tests
            print("Running all tests...")
            result = discover_and_run_tests(
                pattern=args.pattern,
                verbosity=args.verbose
            )
        
        # Stop coverage if enabled
        if args.coverage:
            cov.stop()
            cov.save()
        
        # Print results summary
        print("\n" + "=" * 40)
        print("📋 TEST SUMMARY")
        print("=" * 40)
        
        tests_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
        
        print(f"Tests run: {tests_run}")
        print(f"Failures: {failures}")
        print(f"Errors: {errors}")
        print(f"Skipped: {skipped}")
        
        if failures == 0 and errors == 0:
            print("✅ All tests passed!")
            success = True
        else:
            print("❌ Some tests failed!")
            success = False
        
        # Show coverage report if enabled
        if args.coverage:
            print("\n" + "=" * 40)
            print("📊 COVERAGE REPORT")
            print("=" * 40)
            cov.report(show_missing=True)
            
            # Save HTML coverage report
            try:
                cov.html_report(directory='htmlcov')
                print("\n📄 HTML coverage report saved to: htmlcov/index.html")
            except Exception as e:
                print(f"Could not generate HTML report: {e}")
        
        # Print failure details if any
        if failures > 0:
            print("\n" + "=" * 40)
            print("❌ FAILURE DETAILS")
            print("=" * 40)
            for test, traceback in result.failures:
                print(f"\nFAILED: {test}")
                print(traceback)
        
        if errors > 0:
            print("\n" + "=" * 40)
            print("💥 ERROR DETAILS")
            print("=" * 40)
            for test, traceback in result.errors:
                print(f"\nERROR: {test}")
                print(traceback)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
