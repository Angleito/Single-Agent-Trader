#!/usr/bin/env python3
"""
VuManChu Testing Suite Runner.

Simple script to run VuManChu tests with different configurations.
Provides easy access to the comprehensive testing suite.

Usage:
    python run_vumanchu_tests.py [test_type]

Test Types:
    quick       - Run essential tests only (default)
    full        - Run complete test suite  
    performance - Run performance benchmarks only
    accuracy    - Run accuracy tests only
    validation  - Run manual validation with reports
    real-data   - Prompt for real data file and validate
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"Running: {description or cmd}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✓ Completed successfully in {elapsed:.2f}s")
    else:
        print(f"\n✗ Failed with return code {result.returncode}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='VuManChu Testing Suite Runner')
    parser.add_argument('test_type', nargs='?', default='quick', 
                       choices=['quick', 'full', 'performance', 'accuracy', 'validation', 'real-data'],
                       help='Type of tests to run')
    parser.add_argument('--data-file', type=str, help='CSV file with real market data')
    parser.add_argument('--output-dir', type=str, default='test_output', help='Output directory')
    
    args = parser.parse_args()
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print(f"VuManChu Testing Suite Runner")
    print(f"Test Type: {args.test_type}")
    print(f"Working Directory: {os.getcwd()}")
    
    success = True
    
    if args.test_type == 'quick':
        print("\nRunning QUICK tests - essential components only...")
        
        # Run core parameter tests
        success &= run_command(
            "poetry run pytest tests/test_vumanchu_complete.py::TestPineScriptParameters -v",
            "Pine Script Parameter Verification"
        )
        
        # Run individual component tests
        success &= run_command(
            "poetry run pytest tests/test_vumanchu_complete.py::TestIndividualComponents -v",
            "Individual Component Tests"
        )
        
        # Run basic integration test
        success &= run_command(
            "poetry run pytest tests/test_vumanchu_complete.py::TestCombinedVuManChuIndicators::test_indicator_calculator_initialization -v",
            "Basic Integration Test"
        )
        
    elif args.test_type == 'full':
        print("\nRunning FULL test suite - comprehensive validation...")
        
        # Run all pytest tests
        success &= run_command(
            "poetry run pytest tests/test_vumanchu_complete.py -v --tb=short",
            "Complete Pytest Suite"
        )
        
        # Run manual validation
        success &= run_command(
            f"python tests/validate_vumanchu_implementation.py --full --output-dir {args.output_dir}",
            "Manual Validation Suite"
        )
        
    elif args.test_type == 'performance':
        print("\nRunning PERFORMANCE tests - benchmarking and optimization...")
        
        # Run performance tests
        success &= run_command(
            "poetry run pytest tests/test_vumanchu_complete.py::TestPerformance -v",
            "Performance Tests"
        )
        
        # Run performance validation
        success &= run_command(
            f"python tests/validate_vumanchu_implementation.py --performance --output-dir {args.output_dir}",
            "Performance Validation"
        )
        
    elif args.test_type == 'accuracy':
        print("\nRunning ACCURACY tests - Pine Script compliance and formula verification...")
        
        # Run accuracy-focused tests
        success &= run_command(
            "poetry run pytest tests/test_vumanchu_complete.py::TestPineScriptParameters -v",
            "Parameter Accuracy Tests"
        )
        
        success &= run_command(
            "poetry run pytest tests/test_vumanchu_complete.py::TestCipherAIntegration -v",
            "Cipher A Accuracy Tests"
        )
        
        success &= run_command(
            "poetry run pytest tests/test_vumanchu_complete.py::TestCipherBIntegration -v",
            "Cipher B Accuracy Tests"
        )
        
        # Run accuracy validation
        success &= run_command(
            f"python tests/validate_vumanchu_implementation.py --accuracy --output-dir {args.output_dir}",
            "Accuracy Validation"
        )
        
    elif args.test_type == 'validation':
        print("\nRunning VALIDATION tests - manual validation with detailed reports...")
        
        # Run manual validation only
        success &= run_command(
            f"python tests/validate_vumanchu_implementation.py --full --output-dir {args.output_dir}",
            "Complete Manual Validation"
        )
        
    elif args.test_type == 'real-data':
        print("\nRunning REAL DATA validation...")
        
        data_file = args.data_file
        if not data_file:
            print("\nEnter path to your CSV file with market data:")
            print("Required columns: timestamp (or date), open, high, low, close, volume")
            data_file = input("CSV file path: ").strip()
        
        if not data_file or not os.path.exists(data_file):
            print(f"Error: Data file not found: {data_file}")
            sys.exit(1)
        
        print(f"Using data file: {data_file}")
        
        # Run real data validation
        success &= run_command(
            f"python tests/validate_vumanchu_implementation.py --data-file '{data_file}' --full --output-dir {args.output_dir}",
            "Real Market Data Validation"
        )
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    if success:
        print("✓ All tests completed successfully!")
        print(f"✓ Check output directory: {args.output_dir}")
        
        if args.test_type in ['full', 'validation', 'real-data']:
            print(f"✓ Detailed reports available in: {args.output_dir}")
            
        print("\nNext steps:")
        print("1. Review test reports for any warnings")
        print("2. Check performance benchmarks meet requirements")
        print("3. Validate signal accuracy with real market data")
        print("4. Deploy to production environment")
        
    else:
        print("✗ Some tests failed!")
        print("Please review the error messages above and fix issues before proceeding.")
        sys.exit(1)
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()