#!/usr/bin/env python3
"""
Emergency Reset Script for Trading Bot
Resets circuit breaker and emergency stops to restore trading functionality.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add bot module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Legacy imports (maintained for compatibility)
from bot.config import settings
from bot.risk import RiskManager
from bot.trading_types import Position

# Functional imports (added for migration to functional programming patterns)
try:
    from bot.fp.strategies.risk_management import RiskAssessment, RiskLevel
    from bot.fp.types.config import Config
    from bot.fp.types.trading import PositionSide, PositionState

    FUNCTIONAL_AVAILABLE = True
except ImportError:
    # Functional implementations not available, continue with legacy
    FUNCTIONAL_AVAILABLE = False

logger = logging.getLogger(__name__)


async def main():
    """Reset emergency stops and circuit breakers."""
    print("🔧 Trading Bot Emergency Reset Tool")
    print("=" * 50)

    try:
        # Initialize risk manager
        risk_manager = RiskManager(
            max_position_size=settings.TRADING.MAX_POSITION_SIZE,
            leverage=settings.TRADING.LEVERAGE,
            max_daily_loss=settings.TRADING.MAX_DAILY_LOSS,
            risk_per_trade=settings.TRADING.RISK_PER_TRADE,
        )

        print("✅ Risk Manager initialized")

        # Check current emergency stop status
        if risk_manager.emergency_stop.is_stopped:
            print(
                f"🚨 Emergency stop is ACTIVE: {risk_manager.emergency_stop.stop_reason}"
            )
            print(f"   Triggered at: {risk_manager.emergency_stop.stop_timestamp}")

            # Reset emergency stop
            risk_manager.emergency_stop.reset_emergency_stop(manual_reset=True)
            print("✅ Emergency stop has been RESET")
        else:
            print("✅ No emergency stop active")

        # Reset circuit breaker
        if hasattr(risk_manager, "circuit_breaker"):
            risk_manager.circuit_breaker.reset()
            print("✅ Circuit breaker has been RESET")

        # Reset position error counters
        risk_manager._position_errors_count = 0
        print("✅ Position error counter RESET")

        # Create a flat position to validate
        flat_position = Position(
            symbol=settings.TRADING.SYMBOL,
            side="FLAT",
            size=0,
            entry_price=None,
            timestamp=None,
        )

        # Test risk evaluation with flat position
        current_price = 2.64  # Example SUI price in correct format
        risk_approved, final_action, risk_reason = risk_manager.evaluate_risk(
            action={"action": "HOLD", "size_pct": 0},
            position=flat_position,
            current_price=current_price,
        )

        print(f"✅ Risk evaluation test: {risk_reason}")

        print("\n🎉 Emergency reset completed successfully!")
        print("   Trading bot should now be able to resume normal operations.")
        print("   Monitor logs for any remaining issues.")

    except Exception as e:
        print(f"❌ Error during emergency reset: {e}")
        logger.exception("Emergency reset failed")
        return 1

    return 0


if __name__ == "__main__":
    asyncio.run(main())
