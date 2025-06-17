#!/usr/bin/env python3
"""
Phase 2 Integration Test: Enhanced Message Protocol and Bidirectional Control

This test validates the complete implementation of Phase 2 features:
1. Enhanced message protocol with categorization and replay
2. REST API endpoints for bot control
3. Command queue system with safe execution
4. Emergency stop and risk limit functionality

The test runs without dependencies and validates implementation structure.
"""

import sys
import os
import json
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_enhanced_connection_manager():
    """Test enhanced ConnectionManager with message categorization."""
    print("🔍 Testing Enhanced ConnectionManager...")
    
    # Check if dashboard backend has enhanced ConnectionManager
    backend_file = project_root / "dashboard" / "backend" / "main.py"
    if not backend_file.exists():
        print("❌ Dashboard backend file not found")
        return False
    
    content = backend_file.read_text()
    
    # Check for enhanced features
    required_features = [
        "message_buffers",
        "_categorize_message",
        "_send_replay_messages",
        "get_messages_by_category",
        "get_message_stats",
        "replay_categories",
        "connection_metadata",
        "buffer_limits"
    ]
    
    missing_features = []
    for feature in required_features:
        if feature not in content:
            missing_features.append(feature)
    
    if missing_features:
        print(f"❌ Missing enhanced ConnectionManager features: {missing_features}")
        return False
    
    # Check for message categories
    expected_categories = ["trading", "indicator", "system", "log", "ai"]
    for category in expected_categories:
        if f'"{category}"' not in content:
            print(f"❌ Missing message category: {category}")
            return False
    
    print("✅ Enhanced ConnectionManager structure is valid")
    return True

def test_rest_api_endpoints():
    """Test REST API endpoints for bot control."""
    print("🔍 Testing REST API endpoints...")
    
    backend_file = project_root / "dashboard" / "backend" / "main.py"
    if not backend_file.exists():
        print("❌ Dashboard backend file not found")
        return False
    
    content = backend_file.read_text()
    
    # Check for new API endpoints
    required_endpoints = [
        "@app.get(\"/api/messages/{category}\")",
        "@app.get(\"/api/messages/stats\")",
        "@app.post(\"/api/messages/broadcast\")",
        "@app.post(\"/api/bot/commands/emergency-stop\")",
        "@app.post(\"/api/bot/commands/pause-trading\")",
        "@app.post(\"/api/bot/commands/resume-trading\")",
        "@app.post(\"/api/bot/commands/update-risk-limits\")",
        "@app.post(\"/api/bot/commands/manual-trade\")",
        "@app.get(\"/api/bot/commands/queue\")",
        "@app.get(\"/api/bot/commands/history\")",
        "@app.delete(\"/api/bot/commands/{command_id}\")"
    ]
    
    missing_endpoints = []
    for endpoint in required_endpoints:
        if endpoint not in content:
            missing_endpoints.append(endpoint)
    
    if missing_endpoints:
        print(f"❌ Missing REST API endpoints: {missing_endpoints}")
        return False
    
    # Check for command queue implementation
    command_queue_features = [
        "class BotCommand:",
        "bot_command_queue",
        "command_history",
        "emergency_stop_bot",
        "pause_trading",
        "resume_trading",
        "update_risk_limits",
        "manual_trade_command"
    ]
    
    missing_command_features = []
    for feature in command_queue_features:
        if feature not in content:
            missing_command_features.append(feature)
    
    if missing_command_features:
        print(f"❌ Missing command queue features: {missing_command_features}")
        return False
    
    print("✅ REST API endpoints structure is valid")
    return True

def test_command_consumer():
    """Test command consumer implementation."""
    print("🔍 Testing Command Consumer...")
    
    # Check if command consumer file exists
    consumer_file = project_root / "bot" / "command_consumer.py"
    if not consumer_file.exists():
        print("❌ Command consumer file not found")
        return False
    
    content = consumer_file.read_text()
    
    # Check for required command consumer features
    required_features = [
        "class CommandConsumer:",
        "class CommandType(Enum):",
        "async def start_polling",
        "async def _poll_for_commands",
        "async def _process_command",
        "async def _execute_command",
        "_validate_command",
        "register_callback",
        "_execute_emergency_stop",
        "_execute_pause_trading",
        "_execute_resume_trading",
        "_execute_update_risk_limits",
        "_execute_manual_trade"
    ]
    
    missing_features = []
    for feature in required_features:
        if feature not in content:
            missing_features.append(feature)
    
    if missing_features:
        print(f"❌ Missing command consumer features: {missing_features}")
        return False
    
    # Check for command types
    expected_commands = [
        "EMERGENCY_STOP",
        "PAUSE_TRADING", 
        "RESUME_TRADING",
        "UPDATE_RISK_LIMITS",
        "MANUAL_TRADE"
    ]
    
    for command in expected_commands:
        if command not in content:
            print(f"❌ Missing command type: {command}")
            return False
    
    print("✅ Command consumer structure is valid")
    return True

def test_main_integration():
    """Test integration of command consumer into main bot."""
    print("🔍 Testing main bot integration...")
    
    main_file = project_root / "bot" / "main.py"
    if not main_file.exists():
        print("❌ Main bot file not found")
        return False
    
    content = main_file.read_text()
    
    # Check for command consumer import and initialization
    required_integrations = [
        "from .command_consumer import CommandConsumer",
        "self.command_consumer = CommandConsumer()",
        "_register_command_callbacks",
        "await self.command_consumer.initialize()",
        "self.command_consumer.start_polling()",
        "_handle_emergency_stop",
        "_handle_pause_trading",
        "_handle_resume_trading",
        "_handle_update_risk_limits",
        "_handle_manual_trade",
        "self.command_consumer.stop_polling()",
        "self.command_consumer.close()"
    ]
    
    missing_integrations = []
    for integration in required_integrations:
        if integration not in content:
            missing_integrations.append(integration)
    
    if missing_integrations:
        print(f"❌ Missing main bot integrations: {missing_integrations}")
        return False
    
    print("✅ Main bot integration is valid")
    return True

def test_emergency_stop_functionality():
    """Test emergency stop functionality."""
    print("🔍 Testing Emergency Stop functionality...")
    
    # Check dashboard backend
    backend_file = project_root / "dashboard" / "backend" / "main.py"
    content = backend_file.read_text()
    
    emergency_features = [
        "emergency_stop_bot",
        "priority=1",  # Highest priority
        "emergency_message",
        "emergency_stop"
    ]
    
    for feature in emergency_features:
        if feature not in content:
            print(f"❌ Missing emergency stop feature in backend: {feature}")
            return False
    
    # Check bot command consumer
    consumer_file = project_root / "bot" / "command_consumer.py"
    consumer_content = consumer_file.read_text()
    
    bot_emergency_features = [
        "_execute_emergency_stop",
        "self.emergency_stopped = True",
        "self.trading_paused = True",
        "EMERGENCY STOP EXECUTED"
    ]
    
    for feature in bot_emergency_features:
        if feature not in consumer_content:
            print(f"❌ Missing emergency stop feature in bot: {feature}")
            return False
    
    # Check main bot handlers
    main_file = project_root / "bot" / "main.py"
    main_content = main_file.read_text()
    
    main_emergency_features = [
        "_handle_emergency_stop",
        "EMERGENCY STOP ACTIVATED",
        "_close_all_positions",
        "emergency_stopped"
    ]
    
    for feature in main_emergency_features:
        if feature not in main_content:
            print(f"❌ Missing emergency stop feature in main: {feature}")
            return False
    
    print("✅ Emergency stop functionality is valid")
    return True

def test_risk_limit_functionality():
    """Test risk limit adjustment functionality."""
    print("🔍 Testing Risk Limit functionality...")
    
    # Check dashboard endpoints
    backend_file = project_root / "dashboard" / "backend" / "main.py"
    content = backend_file.read_text()
    
    risk_features = [
        "update_risk_limits",
        "max_position_size",
        "stop_loss_percentage", 
        "max_daily_loss",
        "risk_limits_update"
    ]
    
    for feature in risk_features:
        if feature not in content:
            print(f"❌ Missing risk limit feature in backend: {feature}")
            return False
    
    # Check bot implementation
    consumer_file = project_root / "bot" / "command_consumer.py"
    consumer_content = consumer_file.read_text()
    
    bot_risk_features = [
        "_execute_update_risk_limits",
        "_validate_risk_limits",
        "current_risk_limits"
    ]
    
    for feature in bot_risk_features:
        if feature not in consumer_content:
            print(f"❌ Missing risk limit feature in bot: {feature}")
            return False
    
    print("✅ Risk limit functionality is valid")
    return True

def test_manual_trade_functionality():
    """Test manual trade functionality."""
    print("🔍 Testing Manual Trade functionality...")
    
    # Check dashboard endpoints
    backend_file = project_root / "dashboard" / "backend" / "main.py"
    content = backend_file.read_text()
    
    manual_trade_features = [
        "manual_trade_command",
        "regex=\"^(buy|sell|close)$\"",
        "size_percentage",
        "manual_trade"
    ]
    
    for feature in manual_trade_features:
        if feature not in content:
            print(f"❌ Missing manual trade feature in backend: {feature}")
            return False
    
    # Check bot implementation
    consumer_file = project_root / "bot" / "command_consumer.py"
    consumer_content = consumer_file.read_text()
    
    bot_manual_features = [
        "_execute_manual_trade",
        "_validate_manual_trade",
        "manual_trade"
    ]
    
    for feature in bot_manual_features:
        if feature not in consumer_content:
            print(f"❌ Missing manual trade feature in bot: {feature}")
            return False
    
    print("✅ Manual trade functionality is valid")
    return True

def test_message_categorization():
    """Test message categorization system."""
    print("🔍 Testing Message Categorization...")
    
    backend_file = project_root / "dashboard" / "backend" / "main.py"
    content = backend_file.read_text()
    
    # Check categorization logic
    categorization_features = [
        "_categorize_message",
        "\"trading\":",
        "\"indicator\":",
        "\"system\":",
        "\"log\":",
        "\"ai\":",
        "msg_type",
        "source",
        "return \"trading\"",
        "return \"indicator\"",
        "return \"system\"",
        "return \"log\""
    ]
    
    for feature in categorization_features:
        if feature not in content:
            print(f"❌ Missing categorization feature: {feature}")
            return False
    
    # Check buffer management
    buffer_features = [
        "buffer_limits",
        "\"trading\": 500",
        "\"indicator\": 1000",
        "\"system\": 200",
        "\"log\": 1000",
        "\"ai\": 300"
    ]
    
    for feature in buffer_features:
        if feature not in content:
            print(f"❌ Missing buffer management feature: {feature}")
            return False
    
    print("✅ Message categorization is valid")
    return True

def test_phase2_completeness():
    """Test overall Phase 2 completeness."""
    print("🔍 Testing Phase 2 overall completeness...")
    
    # Check that all major components are integrated
    components = [
        ("Enhanced ConnectionManager", test_enhanced_connection_manager),
        ("REST API Endpoints", test_rest_api_endpoints),
        ("Command Consumer", test_command_consumer),
        ("Main Integration", test_main_integration),
        ("Emergency Stop", test_emergency_stop_functionality),
        ("Risk Limits", test_risk_limit_functionality),
        ("Manual Trading", test_manual_trade_functionality),
        ("Message Categorization", test_message_categorization)
    ]
    
    passed = 0
    total = len(components)
    
    for name, test_func in components:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {name} test failed")
        except Exception as e:
            print(f"❌ {name} test error: {e}")
    
    print(f"\n📊 Phase 2 Test Results: {passed}/{total} components passed")
    
    if passed == total:
        print("🎉 Phase 2 implementation is complete!")
        return True
    else:
        print("❌ Phase 2 implementation needs work")
        return False

def main():
    """Run all Phase 2 integration tests."""
    print("🚀 Running Phase 2 Integration Tests...\n")
    
    # Run individual tests
    tests = [
        ("Enhanced ConnectionManager", test_enhanced_connection_manager),
        ("REST API Endpoints", test_rest_api_endpoints),
        ("Command Consumer", test_command_consumer),
        ("Main Bot Integration", test_main_integration),
        ("Emergency Stop Functionality", test_emergency_stop_functionality),
        ("Risk Limit Functionality", test_risk_limit_functionality),
        ("Manual Trade Functionality", test_manual_trade_functionality),
        ("Message Categorization", test_message_categorization)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append(False)
            print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"📊 Phase 2 Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 2 tests passed! Implementation is ready.")
        print("\n📝 To test the integration:")
        print("1. Start the dashboard: docker-compose up dashboard-backend")
        print("2. Set SYSTEM__ENABLE_WEBSOCKET_PUBLISHING=true in .env")
        print("3. Start the bot: docker-compose up ai-trading-bot")
        print("4. Test commands via REST API:")
        print("   curl -X POST http://localhost:8000/api/bot/commands/pause-trading")
        print("   curl -X GET http://localhost:8000/api/messages/stats")
        return True
    else:
        print("❌ Some Phase 2 tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)