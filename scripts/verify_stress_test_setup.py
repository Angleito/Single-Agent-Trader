#!/usr/bin/env python3
"""
Quick verification script for stress test setup.
Validates that all required components are available and properly configured.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def print_status(message, status="INFO"):
    """Print status message with color coding."""
    colors = {
        "INFO": "\033[0;36m",  # Cyan
        "SUCCESS": "\033[0;32m",  # Green
        "WARNING": "\033[1;33m",  # Yellow
        "ERROR": "\033[0;31m",  # Red
        "RESET": "\033[0m",  # Reset
    }

    color = colors.get(status, colors["INFO"])
    reset = colors["RESET"]
    print(f"{color}[{status}] {message}{reset}")


def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print_status(f"✓ {description}: {filepath}", "SUCCESS")
        return True
    print_status(f"✗ {description}: {filepath}", "ERROR")
    return False


def check_command_available(command, description):
    """Check if a command is available."""
    try:
        subprocess.run([command, "--version"], capture_output=True, check=True)
        print_status(f"✓ {description} is available", "SUCCESS")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_status(f"✗ {description} is not available", "ERROR")
        return False


def check_python_dependencies():
    """Check Python dependencies."""
    required_packages = ["aiohttp", "websockets", "docker", "psutil", "requests"]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print_status(f"✓ Python package '{package}' is available", "SUCCESS")
        except ImportError:
            print_status(f"✗ Python package '{package}' is missing", "ERROR")
            missing_packages.append(package)

    if missing_packages:
        print_status(
            f"Install missing packages: pip install {' '.join(missing_packages)}",
            "WARNING",
        )
        return False

    return True


def validate_config_file(config_path):
    """Validate the configuration file."""
    try:
        with open(config_path) as f:
            config = json.load(f)

        required_keys = [
            "resource_constraints",
            "test_parameters",
            "service_endpoints",
            "test_scenarios",
        ]

        for key in required_keys:
            if key in config:
                print_status(f"✓ Config section '{key}' present", "SUCCESS")
            else:
                print_status(f"✗ Config section '{key}' missing", "ERROR")
                return False

        return True

    except json.JSONDecodeError as e:
        print_status(f"✗ Invalid JSON in config file: {e}", "ERROR")
        return False
    except Exception as e:
        print_status(f"✗ Error reading config file: {e}", "ERROR")
        return False


def check_docker_setup():
    """Check Docker setup."""
    try:
        # Check if Docker is running
        result = subprocess.run(["docker", "info"], capture_output=True, check=True)
        print_status("✓ Docker is running", "SUCCESS")

        # Check if docker-compose is available
        subprocess.run(["docker-compose", "--version"], capture_output=True, check=True)
        print_status("✓ Docker Compose is available", "SUCCESS")

        return True

    except subprocess.CalledProcessError:
        print_status("✗ Docker is not running or not accessible", "ERROR")
        return False
    except FileNotFoundError:
        print_status("✗ Docker or Docker Compose not installed", "ERROR")
        return False


def main():
    """Main verification function."""
    print("🔍 Stress Test Setup Verification")
    print("=" * 50)

    all_checks_passed = True

    # Check required files
    print("\n📁 Checking Required Files:")
    required_files = [
        ("stress_test_low_resource.py", "Main stress test script"),
        ("docker-compose.stress-test.yml", "Docker Compose configuration"),
        ("Dockerfile.stress-test", "Stress test runner Dockerfile"),
        ("run_stress_tests.sh", "Stress test runner script"),
        ("stress_test_config.json", "Test configuration file"),
        (".env", "Environment configuration"),
    ]

    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False

    # Check directories
    print("\n📂 Checking Directories:")
    required_dirs = ["stress_test_results", "logs", "data"]

    for dirname in required_dirs:
        if Path(dirname).exists():
            print_status(f"✓ Directory '{dirname}' exists", "SUCCESS")
        else:
            print_status(
                f"⚠ Directory '{dirname}' missing (will be created)", "WARNING"
            )
            try:
                Path(dirname).mkdir(parents=True, exist_ok=True)
                print_status(f"✓ Created directory '{dirname}'", "SUCCESS")
            except Exception as e:
                print_status(f"✗ Failed to create '{dirname}': {e}", "ERROR")
                all_checks_passed = False

    # Check system dependencies
    print("\n🛠️ Checking System Dependencies:")
    system_deps = [
        ("docker", "Docker"),
        ("python3", "Python 3"),
    ]

    for command, description in system_deps:
        if not check_command_available(command, description):
            all_checks_passed = False

    # Check Docker setup
    print("\n🐳 Checking Docker Setup:")
    if not check_docker_setup():
        all_checks_passed = False

    # Check Python dependencies
    print("\n🐍 Checking Python Dependencies:")
    if not check_python_dependencies():
        all_checks_passed = False

    # Validate configuration
    print("\n⚙️ Validating Configuration:")
    if not validate_config_file("stress_test_config.json"):
        all_checks_passed = False

    # Check permissions
    print("\n🔐 Checking Permissions:")
    if os.access("run_stress_tests.sh", os.X_OK):
        print_status("✓ run_stress_tests.sh is executable", "SUCCESS")
    else:
        print_status("✗ run_stress_tests.sh is not executable", "ERROR")
        print_status("Run: chmod +x run_stress_tests.sh", "WARNING")
        all_checks_passed = False

    # Environment check
    print("\n🌍 Environment Check:")
    if os.getenv("STRESS_TEST_ENV"):
        print_status(f"✓ STRESS_TEST_ENV={os.getenv('STRESS_TEST_ENV')}", "SUCCESS")
    else:
        print_status("⚠ STRESS_TEST_ENV not set (will use default)", "WARNING")

    # Final summary
    print("\n" + "=" * 50)
    if all_checks_passed:
        print_status("🎉 All checks passed! Stress test setup is ready.", "SUCCESS")
        print_status("Run: ./run_stress_tests.sh", "INFO")
        sys.exit(0)
    else:
        print_status("❌ Some checks failed. Please fix the issues above.", "ERROR")
        print_status(
            "See STRESS_TEST_README.md for detailed setup instructions.", "INFO"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
