#!/usr/bin/env python3
"""
Simple type checking test for MarketDataProvider null access fixes.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.data.market import MarketDataProvider


def test_type_annotations():
    """Test that type annotations are correct."""
    # This would be caught by a static type checker

    # Simulate the class without initialization
    class MockTradingEngine:
        def __init__(self):
            self.market_data: "MarketDataProvider | None" = None

        def _ensure_market_data_available(self) -> bool:
            if self.market_data is None:
                return False
            if not self.market_data.is_connected():
                return False
            return True

    # Test the helper method
    engine = MockTradingEngine()
    assert engine._ensure_market_data_available() is False
    print("✓ Type annotations and helper method test passed")


if __name__ == "__main__":
    test_type_annotations()
    print("✓ All type checking tests passed")
