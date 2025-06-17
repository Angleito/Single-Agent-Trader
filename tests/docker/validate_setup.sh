#!/bin/bash

# Simple validation script to ensure test suite is ready

echo "🔍 Validating WebSocket Test Suite Setup..."

# Check if all test files exist
TEST_FILES=(
    "setup_test_environment.sh"
    "test_websocket_connectivity.py"
    "test_message_flow.py"
    "test_fault_tolerance.sh"
    "test_websocket_load.py"
    "integration_test.sh"
    "measure_latency.py"
    "run_all_tests.sh"
    "README.md"
    "requirements.txt"
)

ALL_GOOD=true

for FILE in "${TEST_FILES[@]}"; do
    if [ -f "tests/docker/$FILE" ]; then
        echo "✅ $FILE"
    else
        echo "❌ $FILE missing"
        ALL_GOOD=false
    fi
done

# Check if test files are executable
EXEC_FILES=(
    "setup_test_environment.sh"
    "test_fault_tolerance.sh"
    "integration_test.sh"
    "run_all_tests.sh"
)

echo -e "\n🔧 Checking executable permissions..."
for FILE in "${EXEC_FILES[@]}"; do
    if [ -x "tests/docker/$FILE" ]; then
        echo "✅ $FILE is executable"
    else
        echo "❌ $FILE is not executable"
        ALL_GOOD=false
    fi
done

# Check Python dependencies
echo -e "\n📦 Checking Python dependencies..."
python -c "import websockets" 2>/dev/null && echo "✅ websockets" || echo "❌ websockets (run: pip install -r tests/docker/requirements.txt)"
python -c "import psutil" 2>/dev/null && echo "✅ psutil" || echo "❌ psutil (run: pip install -r tests/docker/requirements.txt)"
python -c "import numpy" 2>/dev/null && echo "✅ numpy" || echo "❌ numpy (run: pip install -r tests/docker/requirements.txt)"
python -c "import docker" 2>/dev/null && echo "✅ docker" || echo "❌ docker (run: pip install -r tests/docker/requirements.txt)"

if [ "$ALL_GOOD" = true ]; then
    echo -e "\n✅ WebSocket test suite is ready!"
    echo "Run: ./tests/docker/run_all_tests.sh"
else
    echo -e "\n❌ Some issues need to be fixed before running tests"
fi