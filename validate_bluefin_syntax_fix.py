#!/usr/bin/env python3
"""
Validation script for Bluefin SDK v2 SyntaxWarning fix.

This script validates that the fix is working correctly by:
1. Checking Docker logs for SyntaxWarning messages
2. Testing the Bluefin service health endpoint
3. Verifying environment variables are set correctly
"""

import subprocess
import sys


def run_command(cmd: list[str]) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def check_docker_logs() -> tuple[bool, list[str]]:
    """Check Docker logs for SyntaxWarning messages."""
    print("🔍 Checking Docker logs for SyntaxWarning messages...")

    # Check if Docker Compose is running
    exit_code, stdout, stderr = run_command(
        ["docker-compose", "ps", "-q", "bluefin-service"]
    )

    if exit_code != 0:
        print("⚠️  Docker Compose not running. Starting services...")
        print("   Run: docker-compose up -d bluefin-service")
        return False, ["Docker Compose not running"]

    if not stdout.strip():
        print("⚠️  Bluefin service not running. Starting service...")
        print("   Run: docker-compose up -d bluefin-service")
        return False, ["Bluefin service not running"]

    # Get logs from the last 5 minutes
    exit_code, stdout, stderr = run_command(
        ["docker-compose", "logs", "--since", "5m", "bluefin-service"]
    )

    if exit_code != 0:
        print(f"❌ Failed to get Docker logs: {stderr}")
        return False, [f"Failed to get logs: {stderr}"]

    # Check for SyntaxWarning messages
    syntax_warnings = []
    lines = stdout.split("\n")

    for line in lines:
        if "SyntaxWarning" in line:
            syntax_warnings.append(line.strip())

    if syntax_warnings:
        print(f"❌ Found {len(syntax_warnings)} SyntaxWarning message(s) in logs:")
        for warning in syntax_warnings:
            print(f"   {warning}")
        return False, syntax_warnings
    print("✅ No SyntaxWarning messages found in Docker logs")
    return True, []


def check_bluefin_service_health() -> bool:
    """Check if the Bluefin service is healthy."""
    print("🏥 Checking Bluefin service health...")

    # Check service health endpoint
    exit_code, stdout, stderr = run_command(
        [
            "curl",
            "-f",
            "--connect-timeout",
            "10",
            "--max-time",
            "15",
            "http://localhost:8081/health",
        ]
    )

    if exit_code == 0:
        print("✅ Bluefin service is healthy")
        return True
    print(f"❌ Bluefin service health check failed: {stderr}")

    # Try to get more info about the service
    exit_code, stdout, stderr = run_command(["docker-compose", "ps", "bluefin-service"])

    if exit_code == 0:
        print(f"Service status:\n{stdout}")

    return False


def check_environment_variables() -> bool:
    """Check if Docker environment variables are set correctly."""
    print("🔧 Checking Docker environment variables...")

    # Get environment variables from the running container
    exit_code, stdout, stderr = run_command(
        [
            "docker-compose",
            "exec",
            "-T",
            "bluefin-service",
            "printenv",
            "PYTHONWARNINGS",
        ]
    )

    if exit_code == 0:
        python_warnings = stdout.strip()
        print(f"📋 PYTHONWARNINGS in container: {python_warnings}")

        if "SyntaxWarning" in python_warnings:
            print("✅ SyntaxWarning suppression found in container environment")
            return True
        print("❌ SyntaxWarning not found in PYTHONWARNINGS")
        return False
    print(f"⚠️  Could not check container environment: {stderr}")
    return False


def validate_fix() -> bool:
    """Main validation function."""
    print("🔧 Validating Bluefin SDK v2 SyntaxWarning Fix")
    print("=" * 60)

    success = True

    # Check 1: Docker logs
    logs_clean, warnings = check_docker_logs()
    if not logs_clean:
        success = False

    print()

    # Check 2: Environment variables
    env_correct = check_environment_variables()
    if not env_correct:
        success = False

    print()

    # Check 3: Service health
    service_healthy = check_bluefin_service_health()
    if not service_healthy:
        print(
            "⚠️  Service health check failed, but this may not be related to the SyntaxWarning fix"
        )

    print("\n" + "=" * 60)
    print("📊 Validation Summary:")
    print(f"   Docker Logs Clean: {'✅' if logs_clean else '❌'}")
    print(f"   Environment Variables: {'✅' if env_correct else '❌'}")
    print(f"   Service Health: {'✅' if service_healthy else '⚠️'}")

    if success:
        print("\n🎉 SyntaxWarning fix validation PASSED!")
        print("   No SyntaxWarning messages should appear in future Docker logs.")
    else:
        print("\n❌ SyntaxWarning fix validation FAILED!")
        print("   Please check the Docker configuration and service logs.")

    print("\n📝 Manual verification commands:")
    print("   docker-compose logs bluefin-service | grep -i syntax")
    print("   docker-compose exec bluefin-service printenv PYTHONWARNINGS")
    print("   curl -f http://localhost:8081/health")

    return success


if __name__ == "__main__":
    success = validate_fix()
    sys.exit(0 if success else 1)
