#!/usr/bin/env python3
"""
Test script for WebSocket integration validation.

This script validates the WebSocket integration without requiring all dependencies
to be installed. It checks import structure, configuration, and message schemas.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_websocket_publisher_structure():
    """Test WebSocket publisher file structure and basic validation."""
    print("🔍 Testing WebSocket Publisher structure...")
    
    # Check if WebSocket publisher file exists
    websocket_file = project_root / "bot" / "websocket_publisher.py"
    if not websocket_file.exists():
        print("❌ WebSocket publisher file not found")
        return False
    
    # Read and validate the file contains key components
    content = websocket_file.read_text()
    
    required_components = [
        "class WebSocketPublisher",
        "async def publish_market_data",
        "async def publish_ai_decision", 
        "async def publish_trading_decision",
        "async def publish_indicator_data",
        "async def publish_trade_execution",
        "async def publish_position_update",
        "async def initialize",
        "async def close"
    ]
    
    missing_components = []
    for component in required_components:
        if component not in content:
            missing_components.append(component)
    
    if missing_components:
        print(f"❌ Missing components: {missing_components}")
        return False
    
    print("✅ WebSocket publisher structure is valid")
    return True

def test_configuration_structure():
    """Test WebSocket configuration in bot settings."""
    print("🔍 Testing configuration structure...")
    
    # Check if config file exists and has WebSocket settings
    config_file = project_root / "bot" / "config.py"
    if not config_file.exists():
        print("❌ Config file not found")
        return False
    
    content = config_file.read_text()
    
    required_config = [
        "enable_websocket_publishing",
        "websocket_dashboard_url",
        "websocket_publish_interval",
        "websocket_max_retries",
        "websocket_retry_delay",
        "websocket_timeout",
        "websocket_queue_size"
    ]
    
    missing_config = []
    for config in required_config:
        if config not in content:
            missing_config.append(config)
    
    if missing_config:
        print(f"❌ Missing configuration: {missing_config}")
        return False
    
    print("✅ Configuration structure is valid")
    return True

def test_main_integration():
    """Test main.py integration points."""
    print("🔍 Testing main.py integration...")
    
    main_file = project_root / "bot" / "main.py"
    if not main_file.exists():
        print("❌ Main file not found")
        return False
    
    content = main_file.read_text()
    
    required_integrations = [
        "from .websocket_publisher import WebSocketPublisher",
        "self.websocket_publisher = WebSocketPublisher",
        "publish_market_data",
        "publish_indicator_data", 
        "publish_ai_decision",
        "publish_trading_decision",
        "publish_trade_execution",
        "publish_position_update"
    ]
    
    missing_integrations = []
    for integration in required_integrations:
        if integration not in content:
            missing_integrations.append(integration)
    
    if missing_integrations:
        print(f"❌ Missing integrations: {missing_integrations}")
        return False
    
    print("✅ Main.py integration is valid")
    return True

def test_example_env():
    """Test example.env configuration."""
    print("🔍 Testing example.env configuration...")
    
    env_file = project_root / "example.env"
    if not env_file.exists():
        print("❌ example.env file not found")
        return False
    
    content = env_file.read_text()
    
    required_env_vars = [
        "SYSTEM__ENABLE_WEBSOCKET_PUBLISHING",
        "SYSTEM__WEBSOCKET_DASHBOARD_URL",
        "SYSTEM__WEBSOCKET_PUBLISH_INTERVAL"
    ]
    
    missing_env_vars = []
    for var in required_env_vars:
        if var not in content:
            missing_env_vars.append(var)
    
    if missing_env_vars:
        print(f"❌ Missing environment variables: {missing_env_vars}")
        return False
    
    print("✅ example.env configuration is valid")
    return True

def test_dashboard_compatibility():
    """Test dashboard message schema compatibility."""
    print("🔍 Testing dashboard compatibility...")
    
    # Check if dashboard WebSocket message schemas exist
    dashboard_ws_file = project_root / "dashboard" / "frontend" / "src" / "websocket.ts"
    if not dashboard_ws_file.exists():
        print("❌ Dashboard WebSocket file not found")
        return False
    
    content = dashboard_ws_file.read_text()
    
    # Check for message types our publisher sends
    required_message_types = [
        "TradingLoopMessage",
        "AIDecisionMessage", 
        "TradingDecisionMessage",
        "SystemStatusMessage",
        "LLMRequestMessage",
        "LLMResponseMessage"
    ]
    
    missing_message_types = []
    for msg_type in required_message_types:
        if msg_type not in content:
            missing_message_types.append(msg_type)
    
    if missing_message_types:
        print(f"❌ Missing message types: {missing_message_types}")
        return False
    
    print("✅ Dashboard compatibility is valid")
    return True

def main():
    """Run all tests and provide summary."""
    print("🚀 Running WebSocket Integration Tests...\n")
    
    tests = [
        test_websocket_publisher_structure,
        test_configuration_structure,
        test_main_integration,
        test_example_env,
        test_dashboard_compatibility
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
            print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! WebSocket integration is ready.")
        print("\n📝 To enable WebSocket publishing:")
        print("1. Set SYSTEM__ENABLE_WEBSOCKET_PUBLISHING=true in your .env file")
        print("2. Start the dashboard: docker-compose up")
        print("3. Run the trading bot with WebSocket publishing enabled")
        return True
    else:
        print("❌ Some tests failed. Please review the integration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)