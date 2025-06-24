#!/usr/bin/env python3
"""
Direct test for functional risk management types without full bot import.
"""

import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Direct imports without going through bot package
sys.path.insert(0, str(project_root / "bot" / "fp" / "types"))

try:
    # Import base types first
    from base import Maybe, Some, Nothing
    from result import Result, Success, Failure
    
    print("✓ Successfully imported base types")
    
    # Import risk types directly
    import risk
    
    print("✓ Successfully imported risk module")
    
    # Test basic functionality
    risk_params = risk.RiskParameters(
        max_position_size=Decimal("25"),
        max_leverage=Decimal("5"),
        stop_loss_pct=Decimal("2"),
        take_profit_pct=Decimal("4")
    )
    print(f"✓ Created RiskParameters: {risk_params}")
    
    # Test circuit breaker
    cb_state = risk.create_circuit_breaker_state()
    print(f"✓ Created circuit breaker: {cb_state.state}")
    
    cb_state = risk.record_circuit_breaker_failure(cb_state, "test", "test message")
    print(f"✓ Recorded failure: {cb_state.failure_count}")
    
    # Test emergency stop
    es_state = risk.create_emergency_stop_state()
    print(f"✓ Created emergency stop: {es_state.is_stopped}")
    
    # Test portfolio exposure
    positions = [
        {"symbol": "BTC-USD", "size": 0.1, "price": 50000},
        {"symbol": "ETH-USD", "size": 2, "price": 3000}
    ]
    
    exposure = risk.calculate_portfolio_exposure(positions, Decimal("50000"))
    print(f"✓ Calculated portfolio exposure: {exposure.portfolio_heat:.1f}%")
    
    # Test balance validation
    import balance_validation
    
    config = balance_validation.create_default_balance_config()
    print(f"✓ Created balance config: min={config.min_balance}")
    
    balance_result = balance_validation.validate_balance_range(
        Decimal("5000"), config.balance_range
    )
    print(f"✓ Balance validation: {balance_result.is_valid}")
    
    print("\n✅ All direct functional risk types are working!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"❌ Test error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)