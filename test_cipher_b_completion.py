#!/usr/bin/env python3
"""
Test script to validate Cipher B completion and integration fixes.

This script tests the integration improvements to achieve 100% Cipher B completion.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data(length: int = 200) -> pd.DataFrame:
    """Create test OHLCV data for validation."""
    dates = pd.date_range(start=datetime.now() - timedelta(days=length), periods=length, freq='1H')
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, length)
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Create OHLCV data
    data = {
        'open': prices * (1 + np.random.normal(0, 0.001, length)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, length))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, length))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, length)
    }
    
    df = pd.DataFrame(data, index=dates)
    return df

def test_cipher_b_integration():
    """Test the Cipher B integration improvements."""
    try:
        from bot.indicators.vumanchu import CipherB
        
        logger.info("Creating test data...")
        test_df = create_test_data(200)
        
        logger.info("Initializing Cipher B...")
        cipher_b = CipherB()
        
        logger.info("Running Cipher B calculation...")
        result = cipher_b.calculate(test_df)
        
        # Check for expected columns
        expected_columns = [
            # Core WaveTrend
            'wt1', 'wt2', 'wt_cross_up', 'wt_cross_down',
            'wt_overbought', 'wt_oversold',
            
            # Advanced signals
            'buy_circle', 'sell_circle', 'gold_buy',
            'divergence_buy', 'divergence_sell',
            'small_circle_up', 'small_circle_down',
            
            # EMA Ribbon
            'ema1', 'ema2', 'ema3', 'ema4', 'ema5', 'ema6', 'ema7', 'ema8',
            'ema_ribbon_bullish', 'ema_ribbon_bearish',
            
            # Additional indicators
            'rsi', 'rsimfi', 'stoch_rsi_k', 'stoch_rsi_d',
            
            # Sommi patterns
            'sommi_flag_up', 'sommi_flag_down',
            'sommi_diamond_up', 'sommi_diamond_down',
            
            # Divergence signals
            'wt_divergence_bullish', 'wt_divergence_bearish',
            'rsi_divergence_bullish', 'rsi_divergence_bearish',
            
            # Legacy indicators
            'vwap', 'money_flow', 'wave',
            
            # Signal analysis
            'cipher_b_signal', 'cipher_b_strength', 'cipher_b_confidence'
        ]
        
        missing_columns = []
        present_columns = []
        
        for col in expected_columns:
            if col in result.columns:
                present_columns.append(col)
            else:
                missing_columns.append(col)
        
        completion_percentage = (len(present_columns) / len(expected_columns)) * 100
        
        logger.info(f"Cipher B Completion Analysis:")
        logger.info(f"Expected columns: {len(expected_columns)}")
        logger.info(f"Present columns: {len(present_columns)}")
        logger.info(f"Missing columns: {len(missing_columns)}")
        logger.info(f"Completion percentage: {completion_percentage:.1f}%")
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
        
        if completion_percentage >= 100.0:
            logger.info("✅ SUCCESS: Cipher B achieved 100% completion!")
        elif completion_percentage >= 90.0:
            logger.info(f"✅ GOOD: Cipher B achieved {completion_percentage:.1f}% completion")
        else:
            logger.warning(f"⚠️  NEEDS WORK: Cipher B only achieved {completion_percentage:.1f}% completion")
        
        # Test latest values method
        logger.info("Testing latest values method...")
        latest_values = cipher_b.get_latest_values(result)
        logger.info(f"Latest values keys: {len(latest_values)} items")
        
        # Test signal analysis
        logger.info("Testing signal analysis...")
        signal_analysis = cipher_b.get_all_signals(result)
        logger.info(f"Signal analysis completed successfully")
        
        # Test specific integrations
        logger.info("Testing specific integrations...")
        
        # RSI+MFI integration
        if 'rsimfi' in result.columns:
            rsimfi_values = result['rsimfi'].dropna()
            logger.info(f"✅ RSI+MFI integration: {len(rsimfi_values)} valid values")
        else:
            logger.warning("❌ RSI+MFI integration missing")
        
        # VWAP integration
        if 'vwap' in result.columns:
            vwap_values = result['vwap'].dropna()
            logger.info(f"✅ VWAP integration: {len(vwap_values)} valid values")
        else:
            logger.warning("❌ VWAP integration missing")
        
        # MFI integration
        if 'money_flow' in result.columns:
            mfi_values = result['money_flow'].dropna()
            logger.info(f"✅ MFI integration: {len(mfi_values)} valid values")
        else:
            logger.warning("❌ MFI integration missing")
        
        # Sommi patterns integration
        sommi_columns = ['sommi_flag_up', 'sommi_flag_down', 'sommi_diamond_up', 'sommi_diamond_down']
        sommi_present = [col for col in sommi_columns if col in result.columns]
        logger.info(f"✅ Sommi patterns integration: {len(sommi_present)}/{len(sommi_columns)} patterns")
        
        return completion_percentage, missing_columns, result
        
    except Exception as e:
        logger.error(f"Error in Cipher B integration test: {e}")
        raise

def test_cipher_a_vs_cipher_b():
    """Compare Cipher A and B to ensure both work properly."""
    try:
        from bot.indicators.vumanchu import CipherA, CipherB
        
        logger.info("Comparing Cipher A vs Cipher B...")
        test_df = create_test_data(200)
        
        cipher_a = CipherA()
        cipher_b = CipherB()
        
        result_a = cipher_a.calculate(test_df)
        result_b = cipher_b.calculate(test_df)
        
        logger.info(f"Cipher A columns: {len(result_a.columns)}")
        logger.info(f"Cipher B columns: {len(result_b.columns)}")
        
        # Check for common indicators
        common_indicators = ['wt1', 'wt2', 'rsi', 'ema1', 'ema2', 'ema8']
        
        for indicator in common_indicators:
            a_has = indicator in result_a.columns
            b_has = indicator in result_b.columns
            logger.info(f"{indicator}: Cipher A={a_has}, Cipher B={b_has}")
        
        return result_a, result_b
        
    except Exception as e:
        logger.error(f"Error in Cipher A vs B comparison: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting Cipher B completion validation...")
    
    try:
        # Test Cipher B integration
        completion_pct, missing_cols, result = test_cipher_b_integration()
        
        # Compare A vs B
        result_a, result_b = test_cipher_a_vs_cipher_b()
        
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Cipher B Completion: {completion_pct:.1f}%")
        
        if completion_pct >= 100.0:
            logger.info("🎉 CIPHER B INTEGRATION FIX SUCCESSFUL!")
            logger.info("All expected indicators are present and integrated.")
        else:
            logger.info(f"Still missing {len(missing_cols)} components:")
            for col in missing_cols:
                logger.info(f"  - {col}")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        exit(1)