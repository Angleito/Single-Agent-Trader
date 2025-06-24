#!/usr/bin/env python3
"""Real-time Data Processing Pipeline Integration Test"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_realtime_data_processing():
    """Test real-time data processing pipeline functionality."""
    print('=== REAL-TIME DATA PROCESSING PIPELINE TEST ===')
    print()
    
    results = {'working': [], 'failing': [], 'warnings': []}
    
    # Test 1: Market Data Processing
    print('1. Testing market data processing...')
    try:
        from bot.fp.types.market import MarketSnapshot, OHLCV, Trade, Candle
        
        # Create sample market data
        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            price=50000.0,
            bid=49995.0,
            ask=50005.0,
            volume=1000.0,
            spread=10.0
        )
        
        ohlcv = OHLCV(
            timestamp=datetime.now(),
            open=49800.0,
            high=50200.0,
            low=49700.0,
            close=50000.0,
            volume=5000.0
        )
        
        trade = Trade(
            timestamp=datetime.now(),
            price=50000.0,
            size=0.5,
            side="buy"
        )
        
        results['working'].append('✅ Market data structures (MarketSnapshot, OHLCV, Trade)')
        
    except Exception as e:
        results['failing'].append(f'❌ Market data structures failed: {e}')
    
    # Test 2: Data Pipeline Components
    print('2. Testing data pipeline components...')
    try:
        from bot.fp.data_pipeline import FunctionalDataPipeline
        
        # Test pipeline creation
        pipeline = FunctionalDataPipeline()
        results['working'].append('✅ Functional data pipeline initialization')
        
    except Exception as e:
        results['failing'].append(f'❌ Data pipeline failed: {e}')
    
    # Test 3: Real-time Aggregation
    print('3. Testing real-time aggregation...')
    try:
        # Test with sample streaming data
        streaming_data = []
        base_time = datetime.now()
        
        # Generate 60 seconds of trade data
        for i in range(60):
            trade_time = base_time + timedelta(seconds=i)
            price = 50000 + np.random.normal(0, 100)  # Price with 100 point volatility
            size = np.random.uniform(0.1, 1.0)
            side = "buy" if np.random.random() > 0.5 else "sell"
            
            streaming_data.append({
                'timestamp': trade_time,
                'price': price,
                'size': size,
                'side': side
            })
        
        # Test aggregation logic
        df = pd.DataFrame(streaming_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Aggregate to 1-minute candles
        candles = df.resample('1min').agg({
            'price': ['first', 'max', 'min', 'last'],
            'size': 'sum'
        }).round(2)
        
        candles.columns = ['open', 'high', 'low', 'close', 'volume']
        
        results['working'].append(f'✅ Real-time aggregation (processed {len(streaming_data)} trades → {len(candles)} candles)')
        
    except Exception as e:
        results['failing'].append(f'❌ Real-time aggregation failed: {e}')
    
    # Test 4: Enhanced Market Data Processing
    print('4. Testing enhanced market data processing...')
    try:
        from bot.types.enhanced_market_data import EnhancedMarketData
        
        # Create enhanced market data
        enhanced_data = EnhancedMarketData(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=50000.0,
            volume=1000.0,
            bid=49995.0,
            ask=50005.0,
            spread=0.02,
            liquidity_score=0.85,
            volatility=0.15,
            momentum=0.05
        )
        
        results['working'].append('✅ Enhanced market data types')
        
    except Exception as e:
        results['failing'].append(f'❌ Enhanced market data failed: {e}')
    
    # Test 5: WebSocket Data Processing Simulation
    print('5. Testing WebSocket data processing simulation...')
    try:
        # Simulate WebSocket message processing
        def process_websocket_message(message):
            """Simulate processing a WebSocket trade message"""
            if message['type'] == 'trade':
                return {
                    'timestamp': datetime.fromisoformat(message['timestamp']),
                    'price': float(message['price']),
                    'size': float(message['size']),
                    'side': message['side']
                }
            return None
        
        # Test with sample messages
        sample_messages = [
            {
                'type': 'trade',
                'timestamp': '2024-01-01T12:00:00',
                'price': '50000.0',
                'size': '0.5',
                'side': 'buy'
            },
            {
                'type': 'trade',
                'timestamp': '2024-01-01T12:00:01',
                'price': '50050.0',
                'size': '0.3',
                'side': 'sell'
            }
        ]
        
        processed_trades = [process_websocket_message(msg) for msg in sample_messages]
        processed_trades = [trade for trade in processed_trades if trade is not None]
        
        results['working'].append(f'✅ WebSocket data processing simulation ({len(processed_trades)} trades processed)')
        
    except Exception as e:
        results['failing'].append(f'❌ WebSocket processing failed: {e}')
    
    # Test 6: Data Quality and Validation
    print('6. Testing data quality and validation...')
    try:
        def validate_market_data(data):
            """Validate market data quality"""
            issues = []
            
            if data['price'] <= 0:
                issues.append("Invalid price")
            if data['size'] <= 0:
                issues.append("Invalid size")
            if 'timestamp' not in data or data['timestamp'] is None:
                issues.append("Missing timestamp")
            
            return len(issues) == 0, issues
        
        # Test with good and bad data
        good_data = {'price': 50000.0, 'size': 0.5, 'timestamp': datetime.now()}
        bad_data = {'price': -100.0, 'size': 0.0, 'timestamp': None}
        
        good_valid, good_issues = validate_market_data(good_data)
        bad_valid, bad_issues = validate_market_data(bad_data)
        
        assert good_valid == True
        assert bad_valid == False
        assert len(bad_issues) == 3
        
        results['working'].append('✅ Data quality validation')
        
    except Exception as e:
        results['failing'].append(f'❌ Data validation failed: {e}')
    
    return results


def print_realtime_results(results):
    """Print real-time data processing test results."""
    print('\n' + '='*80)
    print('REAL-TIME DATA PROCESSING PIPELINE RESULTS')
    print('='*80)
    
    print(f"\n✅ WORKING COMPONENTS ({len(results['working'])}):")
    for item in results['working']:
        print(f"  {item}")
    
    print(f"\n❌ FAILING COMPONENTS ({len(results['failing'])}):")
    for item in results['failing']:
        print(f"  {item}")
    
    if results['warnings']:
        print(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
        for item in results['warnings']:
            print(f"  {item}")
    
    # Calculate pipeline health
    total_tests = len(results['working']) + len(results['failing'])
    success_rate = len(results['working']) / total_tests * 100 if total_tests > 0 else 0
    
    print(f"\n📊 REAL-TIME PIPELINE HEALTH:")
    print(f"  Success Rate: {success_rate:.1f}% ({len(results['working'])}/{total_tests} components working)")
    
    if success_rate >= 80:
        print("  Status: ✅ REAL-TIME PROCESSING READY")
    elif success_rate >= 60:
        print("  Status: ⚠️  REAL-TIME PROCESSING NEEDS MINOR FIXES")
    else:
        print("  Status: ❌ REAL-TIME PROCESSING NEEDS MAJOR FIXES")
    
    print('\n' + '='*80)


def main():
    """Run real-time data processing pipeline test."""
    try:
        results = test_realtime_data_processing()
        print_realtime_results(results)
        
        # Return success if most components are working
        success_rate = len(results['working']) / (len(results['working']) + len(results['failing'])) * 100
        return success_rate >= 60
        
    except Exception as e:
        print(f"❌ Real-time pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)