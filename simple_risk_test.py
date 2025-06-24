#!/usr/bin/env python3
"""
Simple test for functional risk management types.
"""

import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    # Test importing the risk types directly
    from bot.fp.types.risk import (
        RiskParameters,
        RiskLimits,
        MarginInfo,
        create_circuit_breaker_state,
        record_circuit_breaker_failure,
    )
    
    print("✓ Successfully imported basic risk types")
    
    # Test basic functionality
    risk_params = RiskParameters(
        max_position_size=Decimal("25"),
        max_leverage=Decimal("5"),
        stop_loss_pct=Decimal("2"),
        take_profit_pct=Decimal("4")
    )
    print(f"✓ Created RiskParameters: {risk_params}")
    
    cb_state = create_circuit_breaker_state()
    print(f"✓ Created circuit breaker: {cb_state.state}")
    
    print("\n✅ Basic functional risk types are working!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)