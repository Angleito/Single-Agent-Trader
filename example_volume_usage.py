"""
Example usage of the ScalpingVolumeSignals indicators.

This file demonstrates how to use the new volume-based indicators
for scalping strategies. Run this after installing dependencies.

Example usage:
    python example_volume_usage.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the volume indicators
from bot.indicators.scalping_volume import (
    ScalpingVWAP,
    OnBalanceVolume,
    VolumeMovingAverage,
    VolumeProfile,
    ScalpingVolumeSignals,
)


def create_sample_data(periods: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(seconds=periods * 15)  # 15-second intervals
    timestamps = [start_time + timedelta(seconds=i * 15) for i in range(periods)]
    
    # Generate price data with some trends
    base_price = 45000.0
    price_changes = np.random.normal(0, 0.001, periods)  # Small random changes
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Generate OHLC from prices
    highs = prices * (1 + np.abs(np.random.normal(0, 0.0005, periods)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.0005, periods)))
    
    # Generate volumes (higher volume during price moves)
    base_volume = 1000
    volume_multiplier = 1 + np.abs(price_changes) * 10
    volumes = base_volume * volume_multiplier * np.random.lognormal(0, 0.3, periods)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes,
    }).set_index('timestamp')


def demonstrate_vwap():
    """Demonstrate VWAP indicator usage."""
    print("=== VWAP Indicator Demo ===")
    
    # Create sample data
    data = create_sample_data(50)
    
    # Initialize VWAP
    vwap = ScalpingVWAP(rolling_periods=20)
    
    # Calculate VWAP
    result = vwap.calculate(data)
    print(f"Current VWAP: {result['vwap']['current']:.2f}")
    print(f"Upper Band 1: {result['vwap']['upper_band_1']:.2f}")
    print(f"Lower Band 1: {result['vwap']['lower_band_1']:.2f}")
    print(f"Price Position: {result['vwap']['position']}")
    print(f"Signal: {result['vwap']['signal']}")
    
    # Demonstrate real-time update
    print("\n--- Real-time Update ---")
    new_price = 45100.0
    new_volume = 1500.0
    rt_result = vwap.update_realtime(new_price, new_volume)
    print(f"Updated VWAP: {rt_result['vwap']['current']:.2f}")
    print(f"Position: {rt_result['vwap']['position']}")


def demonstrate_obv():
    """Demonstrate OBV indicator usage."""
    print("\n=== OBV Indicator Demo ===")
    
    # Create sample data
    data = create_sample_data(30)
    
    # Initialize OBV
    obv = OnBalanceVolume()
    
    # Calculate OBV
    result = obv.calculate(data)
    print(f"Current OBV: {result['obv']['value']:.0f}")
    print(f"Trend: {result['obv']['trend']}")
    print(f"Momentum: {result['obv']['momentum']:.3f}")
    print(f"Divergence: {result['obv']['divergence']}")
    
    # Demonstrate real-time update
    print("\n--- Real-time Update ---")
    rt_result = obv.update_realtime(45200.0, 1200.0, 45100.0)
    print(f"Updated OBV: {rt_result['obv']['value']:.0f}")
    print(f"Trend: {rt_result['obv']['trend']}")


def demonstrate_volume_ma():
    """Demonstrate Volume MA indicator usage."""
    print("\n=== Volume MA Indicator Demo ===")
    
    # Create sample data
    data = create_sample_data(40)
    
    # Initialize Volume MA
    volume_ma = VolumeMovingAverage()
    
    # Calculate Volume MA
    result = volume_ma.calculate(data)
    print(f"Current Volume: {result['volume_ma']['current']:.0f}")
    print(f"MA 5: {result['volume_ma']['ma_5']:.0f}")
    print(f"MA 10: {result['volume_ma']['ma_10']:.0f}")
    print(f"MA 20: {result['volume_ma']['ma_20']:.0f}")
    print(f"Relative Strength: {result['volume_ma']['relative_strength']:.2f}")
    print(f"Spike Detected: {result['volume_ma']['spike_detected']}")
    print(f"Trend: {result['volume_ma']['trend']}")


def demonstrate_volume_profile():
    """Demonstrate Volume Profile indicator usage."""
    print("\n=== Volume Profile Indicator Demo ===")
    
    # Create sample data
    data = create_sample_data(60)
    
    # Initialize Volume Profile
    volume_profile = VolumeProfile()
    
    # Calculate Volume Profile
    result = volume_profile.calculate(data)
    profile = result['volume_profile']
    print(f"Point of Control (POC): {profile['poc']:.2f}")
    print(f"High Volume Nodes: {len(profile['high_volume_nodes'])} levels")
    print(f"Low Volume Nodes: {len(profile['low_volume_nodes'])} levels")
    print(f"Volume Concentration: {profile['concentration']:.3f}")
    
    if profile['high_volume_nodes']:
        print(f"Top High Volume Node: {profile['high_volume_nodes'][0]:.2f}")


def demonstrate_combined_signals():
    """Demonstrate combined volume signals."""
    print("\n=== Combined Volume Signals Demo ===")
    
    # Create sample data
    data = create_sample_data(100)
    
    # Initialize combined signals
    volume_signals = ScalpingVolumeSignals()
    
    # Calculate all signals
    result = volume_signals.calculate(data)
    
    # Display consensus
    consensus = result['consensus']
    print(f"Volume Confirmation: {consensus['confirmation']}")
    print(f"Strength: {consensus['strength']:.3f}")
    print(f"Supporting Indicators: {consensus['supporting_indicators']}")
    print(f"Conflicting Indicators: {consensus['conflicting_indicators']}")
    
    # Display individual components
    print("\n--- Individual Components ---")
    print(f"VWAP Signal: {result['vwap']['signal']}")
    print(f"OBV Trend: {result['obv']['trend']}")
    print(f"Volume Spike: {result['volume_ma']['spike_detected']}")
    print(f"POC Level: {result['volume_profile']['poc']:.2f if result['volume_profile']['poc'] else 'N/A'}")


def main():
    """Run all demonstrations."""
    print("ScalpingVolumeSignals - Volume Indicators Demo")
    print("=" * 50)
    
    try:
        demonstrate_vwap()
        demonstrate_obv()
        demonstrate_volume_ma()
        demonstrate_volume_profile()
        demonstrate_combined_signals()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()