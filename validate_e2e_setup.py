#!/usr/bin/env python3
"""
Validation script for VuManChu E2E Testing Suite setup.

This script validates that all required files and dependencies are in place
for the comprehensive E2E testing suite.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

def check_file_exists(file_path: Path) -> Tuple[bool, str]:
    """Check if a file exists and return status with message."""
    if file_path.exists():
        return True, f"✓ {file_path}"
    else:
        return False, f"✗ Missing: {file_path}"

def check_directory_exists(dir_path: Path) -> Tuple[bool, str]:
    """Check if a directory exists and return status with message."""
    if dir_path.exists() and dir_path.is_dir():
        return True, f"✓ {dir_path}/"
    else:
        return False, f"✗ Missing directory: {dir_path}/"

def validate_e2e_setup() -> bool:
    """Validate the complete E2E testing setup."""
    print("VuManChu E2E Testing Suite - Setup Validation")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    all_checks_passed = True
    
    # Required files
    required_files = [
        # Main E2E test file
        project_root / "tests" / "test_e2e_vumanchu_docker.py",
        
        # Test data generator
        project_root / "tests" / "data" / "generate_test_market_data.py",
        project_root / "tests" / "data" / "__init__.py",
        
        # Docker configuration
        project_root / "docker" / "test-compose.yml",
        project_root / "docker" / "test-config.yml",
        
        # Test runner script
        project_root / "scripts" / "run_docker_tests.py",
        
        # Documentation
        project_root / "tests" / "README_E2E_DOCKER_TESTING.md",
        project_root / "docker" / "README.md",
        
        # Project files
        project_root / "pyproject.toml",
        project_root / "docker-compose.yml",
        project_root / "Dockerfile.minimal",
    ]
    
    # Required directories
    required_directories = [
        project_root / "tests",
        project_root / "tests" / "data",
        project_root / "docker",
        project_root / "scripts",
        project_root / "bot",
        project_root / "bot" / "indicators",
    ]
    
    print("\n1. Checking Required Directories:")
    print("-" * 30)
    for directory in required_directories:
        passed, message = check_directory_exists(directory)
        print(message)
        if not passed:
            all_checks_passed = False
    
    print("\n2. Checking Required Files:")
    print("-" * 30)
    for file_path in required_files:
        passed, message = check_file_exists(file_path)
        print(message)
        if not passed:
            all_checks_passed = False
    
    print("\n3. Checking File Permissions:")
    print("-" * 30)
    executable_files = [
        project_root / "tests" / "data" / "generate_test_market_data.py",
        project_root / "scripts" / "run_docker_tests.py",
    ]
    
    for file_path in executable_files:
        if file_path.exists():
            if os.access(file_path, os.X_OK):
                print(f"✓ {file_path} (executable)")
            else:
                print(f"⚠ {file_path} (not executable - may need chmod +x)")
        else:
            print(f"✗ {file_path} (missing)")
            all_checks_passed = False
    
    print("\n4. Checking VuManChu Implementation Files:")
    print("-" * 30)
    vumanchu_files = [
        project_root / "bot" / "indicators" / "vumanchu.py",
        project_root / "bot" / "indicators" / "wavetrend.py",
        project_root / "bot" / "indicators" / "cipher_a_signals.py",
        project_root / "bot" / "indicators" / "cipher_b_signals.py",
        project_root / "bot" / "indicators" / "ema_ribbon.py",
        project_root / "bot" / "indicators" / "rsimfi.py",
    ]
    
    for file_path in vumanchu_files:
        passed, message = check_file_exists(file_path)
        print(message)
        if not passed:
            all_checks_passed = False
    
    print("\n5. Checking Python Dependencies:")
    print("-" * 30)
    try:
        import pandas
        print("✓ pandas")
    except ImportError:
        print("✗ pandas (required)")
        all_checks_passed = False
    
    try:
        import numpy
        print("✓ numpy") 
    except ImportError:
        print("✗ numpy (required)")
        all_checks_passed = False
    
    try:
        import pytest
        print("✓ pytest")
    except ImportError:
        print("✗ pytest (required for testing)")
        all_checks_passed = False
    
    try:
        import psutil
        print("✓ psutil")
    except ImportError:
        print("✗ psutil (required for performance monitoring)")
        all_checks_passed = False
    
    try:
        import yaml
        print("✓ PyYAML")
    except ImportError:
        print("⚠ PyYAML (required for test runner script)")
    
    print("\n6. Checking Docker Availability:")
    print("-" * 30)
    import subprocess
    
    try:
        result = subprocess.run(
            ["docker", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            print(f"✓ Docker: {result.stdout.strip()}")
        else:
            print("✗ Docker not responding")
            all_checks_passed = False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("✗ Docker not found")
        all_checks_passed = False
    
    try:
        result = subprocess.run(
            ["docker-compose", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            print(f"✓ Docker Compose: {result.stdout.strip()}")
        else:
            print("✗ Docker Compose not responding")
            all_checks_passed = False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("✗ Docker Compose not found")
        all_checks_passed = False
    
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("✓ ALL CHECKS PASSED - E2E Testing Suite is ready!")
        print("\nNext steps:")
        print("1. Run: python scripts/run_docker_tests.py setup")
        print("2. Run: python scripts/run_docker_tests.py comprehensive")
    else:
        print("✗ SOME CHECKS FAILED - Please fix the issues above")
        print("\nInstallation commands:")
        print("1. Install dependencies: poetry install")
        print("2. Make scripts executable: chmod +x scripts/run_docker_tests.py")
        print("3. Install Docker: https://docs.docker.com/get-docker/")
    
    print("\nFor detailed usage instructions, see:")
    print("- tests/README_E2E_DOCKER_TESTING.md")
    print("- python scripts/run_docker_tests.py --help")
    
    return all_checks_passed

if __name__ == "__main__":
    success = validate_e2e_setup()
    sys.exit(0 if success else 1)