#!/bin/bash
# Validation script for Docker entrypoint scripts
# This script performs basic validation without requiring container environment

set -euo pipefail

echo "🔍 Validating Docker entrypoint scripts..."

# Check syntax of main entrypoint script
echo "Checking main entrypoint script syntax..."
if bash -n scripts/docker-entrypoint.sh; then
    echo "✅ Main entrypoint script syntax: VALID"
else
    echo "❌ Main entrypoint script syntax: INVALID"
    exit 1
fi

# Check syntax of Bluefin entrypoint script
echo "Checking Bluefin entrypoint script syntax..."
if bash -n services/docker-entrypoint-bluefin.sh; then
    echo "✅ Bluefin entrypoint script syntax: VALID"
else
    echo "❌ Bluefin entrypoint script syntax: INVALID"
    exit 1
fi

# Check file permissions
echo "Checking file permissions..."
if [[ -x scripts/docker-entrypoint.sh ]]; then
    echo "✅ Main entrypoint script: EXECUTABLE"
else
    echo "❌ Main entrypoint script: NOT EXECUTABLE"
    exit 1
fi

if [[ -x services/docker-entrypoint-bluefin.sh ]]; then
    echo "✅ Bluefin entrypoint script: EXECUTABLE"
else
    echo "❌ Bluefin entrypoint script: NOT EXECUTABLE"
    exit 1
fi

# Check that Dockerfiles reference the entrypoint scripts
echo "Checking Dockerfile integration..."
if grep -q "docker-entrypoint.sh" Dockerfile; then
    echo "✅ Main Dockerfile: References entrypoint script"
else
    echo "❌ Main Dockerfile: Missing entrypoint script reference"
    exit 1
fi

if grep -q "docker-entrypoint.sh" services/Dockerfile.bluefin; then
    echo "✅ Bluefin Dockerfile: References entrypoint script"
else
    echo "❌ Bluefin Dockerfile: Missing entrypoint script reference"
    exit 1
fi

# Check script headers and required functions
echo "Checking script structure..."
required_functions=("display_banner" "check_user" "setup_directories" "verify_python_environment" "setup_environment" "perform_health_checks" "main")

for script in "scripts/docker-entrypoint.sh" "services/docker-entrypoint-bluefin.sh"; do
    echo "  Checking functions in: $script"
    for func in "${required_functions[@]}"; do
        if grep -q "^${func}()" "$script"; then
            echo "    ✅ Function found: $func"
        else
            echo "    ❌ Function missing: $func"
            exit 1
        fi
    done
done

echo ""
echo "🎉 All entrypoint script validations PASSED!"
echo ""
echo "The entrypoint scripts are ready for container deployment."
echo "They will:"
echo "  • Create required directories with proper permissions"
echo "  • Test write permissions and setup fallbacks if needed"
echo "  • Verify Python environment"
echo "  • Perform health checks"
echo "  • Provide clear error messages"
echo "  • Maintain backwards compatibility"