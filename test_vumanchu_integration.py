#!/usr/bin/env python3
"""VuManChu Indicators Integration Test"""

import numpy as np
import pandas as pd


def test_vumanchu_indicators():
    """Test VuManChu indicators functionality and accuracy."""
    print("=== VUMANCHU INDICATORS INTEGRATION TEST ===")
    print()

    # Test VuManChu indicators functionality
    try:
        from bot.indicators.vumanchu import VuManChuIndicators

        print("1. Testing VuManChu indicators initialization...")
        indicators = VuManChuIndicators()
        print("✅ VuManChu indicators initialized successfully")

        # Create sample market data
        print()
        print("2. Creating sample market data...")
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1H")
        np.random.seed(42)  # For reproducible results

        # Generate realistic OHLCV data
        base_price = 50000
        prices = []
        volume = []

        for i in range(100):
            if i == 0:
                open_price = base_price
            else:
                open_price = prices[i - 1]["close"]

            # Random walk with small changes
            change = np.random.normal(0, 0.02)  # 2% volatility
            close_price = open_price * (1 + change)

            high_price = max(open_price, close_price) * (
                1 + abs(np.random.normal(0, 0.01))
            )
            low_price = min(open_price, close_price) * (
                1 - abs(np.random.normal(0, 0.01))
            )

            prices.append(
                {
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                }
            )

            volume.append(np.random.uniform(1000, 10000))

        # Create DataFrame
        market_data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": [p["open"] for p in prices],
                "high": [p["high"] for p in prices],
                "low": [p["low"] for p in prices],
                "close": [p["close"] for p in prices],
                "volume": volume,
            }
        )

        print(
            f"✅ Created {len(market_data)} data points from {market_data.timestamp.min()} to {market_data.timestamp.max()}"
        )
        print(
            f"   Price range: ${market_data.low.min():.2f} - ${market_data.high.max():.2f}"
        )

        print()
        print("3. Testing VuManChu indicator calculations...")

        # Test if calculate method exists and works
        if hasattr(indicators, "calculate"):
            result = indicators.calculate(market_data)
            print("✅ VuManChu calculate() method works")
        elif hasattr(indicators, "calculate_all"):
            result = indicators.calculate_all(market_data)
            print("✅ VuManChu calculate_all() method works")
        else:
            print("❌ Neither calculate() nor calculate_all() method found")
            available_methods = [
                method for method in dir(indicators) if not method.startswith("_")
            ]
            print(f"   Available methods: {available_methods}")
            return False

        print(f"   Result type: {type(result)}")
        if isinstance(result, dict):
            print(f"   Result keys: {list(result.keys())}")
            for key, value in result.items():
                if hasattr(value, "__len__") and not isinstance(value, str):
                    print(f"   {key}: {len(value)} values")
                else:
                    print(f"   {key}: {value}")
        else:
            result_len = len(result) if hasattr(result, "__len__") else "N/A"
            print(f"   Result length: {result_len}")

        print("✅ VuManChu indicators calculation completed successfully")
        return True

    except Exception as e:
        import traceback

        print(f"❌ VuManChu indicators test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_vumanchu_indicators()
    exit(0 if success else 1)
