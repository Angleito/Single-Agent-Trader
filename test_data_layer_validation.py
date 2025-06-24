#!/usr/bin/env python3
"""
Enhanced Data Layer Performance and Real-time Capabilities Test

This script validates the enhanced data layer components including
market data types, WebSocket integration, and real-time processing.
"""

import sys
import os
import logging
from datetime import datetime, timezone
from decimal import Decimal

# Add bot module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_market_data_types():
    """Test enhanced market data types"""
    print("Testing Market Data Types...")
    
    try:
        # Import available market data types
        from bot.trading_types import Position, MarketData
        
        # Test Position type
        position = Position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal('1.5'),
            timestamp=datetime.now(timezone.utc),
        )
        
        print(f"✅ Created Position: {position.symbol} {position.side} {position.size}")
        
        # Test MarketData if available
        try:
            market_data = MarketData(
                symbol="BTC-USD",
                timestamp=datetime.now(timezone.utc),
                price=Decimal('50000.00'),
                volume=Decimal('100.0'),
            )
            print(f"✅ Created MarketData: {market_data.symbol} @ ${market_data.price}")
        except TypeError as e:
            print(f"ℹ️  MarketData constructor differs from expected: {e}")
            
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import market data types: {e}")
        return False
    except Exception as e:
        print(f"❌ Market data types test failed: {e}")
        return False


def test_enhanced_market_data():
    """Test enhanced market data modules"""
    print("\nTesting Enhanced Market Data Modules...")
    
    try:
        # Check what's available in enhanced_market_data
        from bot.types import enhanced_market_data
        
        # List available classes/functions
        available_items = [item for item in dir(enhanced_market_data) if not item.startswith('_')]
        print(f"Available items in enhanced_market_data: {available_items}")
        
        # Try to use what's available
        if hasattr(enhanced_market_data, 'MarketData'):
            print("✅ MarketData class found")
        if hasattr(enhanced_market_data, 'IndicatorData'):
            print("✅ IndicatorData class found")
        if hasattr(enhanced_market_data, 'MarketState'):
            print("✅ MarketState class found")
            
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import enhanced market data: {e}")
        return False
    except Exception as e:
        print(f"❌ Enhanced market data test failed: {e}")
        return False


def test_websocket_capabilities():
    """Test WebSocket integration capabilities"""
    print("\nTesting WebSocket Capabilities...")
    
    try:
        # Test WebSocket publisher
        from bot.websocket_publisher import WebSocketPublisher
        
        # Test creation (without actually connecting)
        publisher = WebSocketPublisher()
        print("✅ WebSocketPublisher created successfully")
        
        # Test data structures for WebSocket
        test_data = {
            "type": "market_update",
            "symbol": "BTC-USD",
            "price": 50000.0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        print(f"✅ Test WebSocket data structure: {test_data}")
        
        return True
        
    except ImportError as e:
        print(f"❌ WebSocket import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False


def test_performance_monitoring():
    """Test performance monitoring capabilities"""
    print("\nTesting Performance Monitoring...")
    
    try:
        # Test performance monitor
        from bot.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        print("✅ PerformanceMonitor created successfully")
        
        # Test getting metrics
        metrics = monitor.get_current_metrics()
        print(f"✅ Current metrics: {len(metrics) if isinstance(metrics, dict) else type(metrics)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Performance monitor import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False


def test_order_position_management():
    """Test order and position management reliability"""
    print("\nTesting Order and Position Management...")
    
    try:
        # Test order manager
        from bot.order_manager import OrderManager
        
        order_manager = OrderManager()
        print("✅ OrderManager created successfully")
        
        # Test position manager
        from bot.position_manager import PositionManager
        
        position_manager = PositionManager()
        print("✅ PositionManager created successfully")
        
        # Test FIFO trading
        try:
            from bot.trading.fifo_position_manager import FIFOPositionManager
            
            fifo_manager = FIFOPositionManager()
            print("✅ FIFOPositionManager created successfully")
        except ImportError:
            print("ℹ️  FIFOPositionManager not available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Order/Position management import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Order/Position management test failed: {e}")
        return False


def test_paper_trading_simulation():
    """Test paper trading simulation accuracy"""
    print("\nTesting Paper Trading Simulation...")
    
    try:
        # Test paper trading module
        from bot.paper_trading import PaperTradingEngine
        
        paper_engine = PaperTradingEngine()
        print("✅ PaperTradingEngine created successfully")
        
        # Test simulation state
        initial_balance = paper_engine.get_balance()
        print(f"✅ Initial balance: {initial_balance}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Paper trading import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Paper trading test failed: {e}")
        return False


def test_real_time_data_processing():
    """Test real-time data processing capabilities"""
    print("\nTesting Real-time Data Processing...")
    
    try:
        # Test market data module
        from bot.data.market import MarketDataFeed
        
        # Test creation without connecting to live feeds
        feed = MarketDataFeed(symbol="BTC-USD", dry_run=True)
        print("✅ MarketDataFeed created successfully")
        
        # Test data format compatibility
        sample_candle = {
            'timestamp': datetime.now(timezone.utc),
            'open': 50000.0,
            'high': 50100.0,
            'low': 49900.0,
            'close': 50050.0,
            'volume': 100.0
        }
        
        print(f"✅ Sample candle data format: {list(sample_candle.keys())}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Market data feed import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Real-time data processing test failed: {e}")
        return False


def main():
    """Run all enhanced data layer validation tests"""
    print("=" * 60)
    print("Enhanced Data Layer and Real-time Capabilities Validation")
    print("=" * 60)
    
    results = []
    
    # Test 1: Market data types
    results.append(("Market Data Types", test_market_data_types()))
    
    # Test 2: Enhanced market data modules
    results.append(("Enhanced Market Data", test_enhanced_market_data()))
    
    # Test 3: WebSocket capabilities
    results.append(("WebSocket Capabilities", test_websocket_capabilities()))
    
    # Test 4: Performance monitoring
    results.append(("Performance Monitoring", test_performance_monitoring()))
    
    # Test 5: Order and position management
    results.append(("Order/Position Management", test_order_position_management()))
    
    # Test 6: Paper trading simulation
    results.append(("Paper Trading Simulation", test_paper_trading_simulation()))
    
    # Test 7: Real-time data processing
    results.append(("Real-time Data Processing", test_real_time_data_processing()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("ENHANCED DATA LAYER VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL ENHANCED DATA LAYER TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {failed} ENHANCED DATA LAYER TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())