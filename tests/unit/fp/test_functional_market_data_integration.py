"""
Test Functional Market Data Integration

This module tests the integration between functional market data types
and the existing real-time data processing infrastructure.
"""

import asyncio
import pytest
from datetime import datetime, UTC
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from bot.fp.adapters.market_data_adapter import (
    FunctionalMarketDataProcessor,
    create_functional_market_data_processor,
    create_integrated_market_data_system,
    integrate_with_existing_provider,
)
from bot.fp.adapters.indicator_adapter import (
    FunctionalIndicatorProcessor,
    create_indicator_processor,
    create_integrated_indicator_system,
)
from bot.fp.adapters.type_converters import (
    current_market_data_to_fp_candle,
    convert_candle_list_to_fp,
    create_connection_state,
    create_data_quality,
    validate_functional_candle,
)
from bot.fp.types.market import (
    ConnectionStatus,
    FPCandle,
    RealtimeUpdate,
)
from bot.fp.types.indicators import IndicatorConfig
from bot.trading_types import MarketData as CurrentMarketData


class TestFunctionalMarketDataProcessor:
    """Test functional market data processor."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = create_functional_market_data_processor("BTC-USD", "1m")
        
        assert processor.symbol == "BTC-USD"
        assert processor.interval == "1m"
        assert processor.get_connection_state().status == ConnectionStatus.DISCONNECTED
        assert len(processor.get_recent_candles()) == 0
        assert len(processor.get_recent_updates()) == 0
    
    def test_market_data_conversion(self):
        """Test conversion of market data to functional types."""
        # Create sample market data
        market_data = CurrentMarketData(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49500"),
            close=Decimal("50500"),
            volume=Decimal("100.5"),
        )
        
        # Convert to functional type
        fp_candle = current_market_data_to_fp_candle(market_data)
        
        # Verify conversion
        assert fp_candle.symbol == "BTC-USD"
        assert fp_candle.open == Decimal("50000")
        assert fp_candle.high == Decimal("51000")
        assert fp_candle.low == Decimal("49500")
        assert fp_candle.close == Decimal("50500")
        assert fp_candle.volume == Decimal("100.5")
        assert validate_functional_candle(fp_candle)
    
    def test_candle_properties(self):
        """Test functional candle properties."""
        fp_candle = FPCandle(
            timestamp=datetime.now(UTC),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49500"),
            close=Decimal("50500"),
            volume=Decimal("100.5"),
            symbol="BTC-USD",
        )
        
        # Test properties
        assert fp_candle.is_bullish  # close > open
        assert not fp_candle.is_bearish
        assert fp_candle.price_range == Decimal("1500")  # high - low
        assert fp_candle.body_size == Decimal("500")  # abs(close - open)
        assert fp_candle.upper_shadow == Decimal("500")  # high - max(open, close)
        assert fp_candle.lower_shadow == Decimal("0")  # min(open, close) - low
        assert fp_candle.vwap() == Decimal("50166.666666666666666666666667")
    
    def test_websocket_message_processing(self):
        """Test WebSocket message processing."""
        processor = create_functional_market_data_processor("BTC-USD", "1m")
        
        # Mock callback
        update_callback = MagicMock()
        processor.add_update_callback(update_callback)
        
        # Test ticker message
        ticker_message = {
            "channel": "ticker",
            "timestamp": datetime.now(UTC).isoformat(),
            "events": [{
                "type": "update",
                "tickers": [{
                    "product_id": "BTC-USD",
                    "price": "50000.00",
                    "volume_24h": "1000.0"
                }]
            }]
        }
        
        processor.process_websocket_message(ticker_message)
        
        # Verify callback was called
        assert update_callback.call_count >= 1
        
        # Verify updates are stored
        updates = processor.get_recent_updates()
        assert len(updates) >= 1
        assert updates[-1].update_type == "ticker"
        assert updates[-1].symbol == "BTC-USD"
    
    def test_data_quality_tracking(self):
        """Test data quality tracking."""
        processor = create_functional_market_data_processor("BTC-USD", "1m")
        
        # Initial quality should be perfect
        quality = processor.get_data_quality()
        assert quality.success_rate == 100.0
        assert quality.error_rate == 0.0
        
        # Simulate some data processing
        market_data = CurrentMarketData(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49500"),
            close=Decimal("50500"),
            volume=Decimal("100.5"),
        )
        
        processor._on_market_data_update(market_data)
        
        # Quality should reflect processed message
        quality = processor.get_data_quality()
        assert quality.messages_received >= 1
        assert quality.messages_processed >= 1
        assert quality.success_rate == 100.0
    
    def test_callback_management(self):
        """Test callback management."""
        processor = create_functional_market_data_processor("BTC-USD", "1m")
        
        # Test adding callbacks
        candle_callback = MagicMock()
        update_callback = MagicMock()
        stream_callback = MagicMock()
        
        processor.add_candle_callback(candle_callback)
        processor.add_update_callback(update_callback)
        processor.add_stream_callback(stream_callback)
        
        assert len(processor._candle_callbacks) == 1
        assert len(processor._update_callbacks) == 1
        assert len(processor._stream_callbacks) == 1
        
        # Test removing callbacks
        processor.remove_candle_callback(candle_callback)
        processor.remove_update_callback(update_callback)
        processor.remove_stream_callback(stream_callback)
        
        assert len(processor._candle_callbacks) == 0
        assert len(processor._update_callbacks) == 0
        assert len(processor._stream_callbacks) == 0


class TestFunctionalIndicatorProcessor:
    """Test functional indicator processor."""
    
    def test_initialization(self):
        """Test indicator processor initialization."""
        config = IndicatorConfig(ma_period=20, rsi_period=14)
        processor = create_indicator_processor(config)
        
        assert processor.config.ma_period == 20
        assert processor.config.rsi_period == 14
        assert len(processor._recent_results) == 0
    
    def test_indicator_calculation(self):
        """Test indicator calculations with functional candles."""
        processor = create_indicator_processor()
        
        # Create sample candles
        candles = []
        base_price = Decimal("50000")
        
        for i in range(100):  # Need sufficient data for indicators
            price_variation = Decimal(str(i * 10))  # Simple price progression
            candles.append(FPCandle(
                timestamp=datetime.now(UTC),
                open=base_price + price_variation,
                high=base_price + price_variation + Decimal("500"),
                low=base_price + price_variation - Decimal("200"),
                close=base_price + price_variation + Decimal("100"),
                volume=Decimal("10.0"),
                symbol="BTC-USD",
            ))
        
        # Process candles
        results = processor.process_candles(candles)
        
        # Verify results
        assert isinstance(results, dict)
        # Some indicators should be calculated with sufficient data
        assert len(results) >= 1
        
        # Check if any indicators were calculated
        for indicator_name, result in results.items():
            if result:
                assert result.timestamp is not None
                print(f"Calculated {indicator_name}: {result}")
    
    def test_indicator_history(self):
        """Test indicator history tracking."""
        processor = create_indicator_processor()
        
        # Create sample candles
        candles = []
        for i in range(50):
            candles.append(FPCandle(
                timestamp=datetime.now(UTC),
                open=Decimal("50000"),
                high=Decimal("51000"),
                low=Decimal("49500"),
                close=Decimal("50500"),
                volume=Decimal("10.0"),
                symbol="BTC-USD",
            ))
        
        # Process multiple times to build history
        processor.process_candles(candles[:25])
        processor.process_candles(candles)
        
        # Check history
        for indicator_name in processor._recent_results:
            history = processor.get_indicator_history(indicator_name)
            assert len(history.data) >= 1
            
            latest = processor.get_latest_result(indicator_name)
            assert latest is not None
    
    def test_integration_adapter(self):
        """Test integration adapter for current system compatibility."""
        functional_processor, integration_adapter = create_integrated_indicator_system()
        
        # Create sample candles
        candles = []
        for i in range(60):  # Sufficient data for indicators
            candles.append(FPCandle(
                timestamp=datetime.now(UTC),
                open=Decimal("50000"),
                high=Decimal("51000"),
                low=Decimal("49500"),
                close=Decimal("50500"),
                volume=Decimal("10.0"),
                symbol="BTC-USD",
            ))
        
        # Process and get current format
        current_indicators = integration_adapter.process_candles_and_get_current(candles)
        
        # Verify current format
        assert current_indicators.timestamp is not None
        # At least some indicators should be available
        indicator_values = [
            current_indicators.cipher_a_dot,
            current_indicators.cipher_b_wave,
            current_indicators.rsi,
            current_indicators.ema_fast,
            current_indicators.ema_slow,
        ]
        # At least one value should be not None
        assert any(value is not None for value in indicator_values)


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_data_flow(self):
        """Test end-to-end data flow from WebSocket to indicators."""
        # Create integrated system
        functional_processor, provider = create_integrated_market_data_system("BTC-USD", "1m")
        
        # Mock the existing provider
        with patch('bot.data.market.MarketDataProvider') as MockProvider:
            mock_provider = MockProvider.return_value
            mock_provider.wait_for_websocket_data = AsyncMock(return_value=True)
            mock_provider.subscribe_to_updates = MagicMock()
            mock_provider.unsubscribe_from_updates = MagicMock()
            
            # Start the functional processor
            await functional_processor.start(mock_provider)
            
            # Verify connection
            assert functional_processor.get_connection_state().status == ConnectionStatus.CONNECTED
            
            # Simulate market data update
            market_data = CurrentMarketData(
                symbol="BTC-USD",
                timestamp=datetime.now(UTC),
                open=Decimal("50000"),
                high=Decimal("51000"),
                low=Decimal("49500"),
                close=Decimal("50500"),
                volume=Decimal("100.5"),
            )
            
            # Process update
            functional_processor._on_market_data_update(market_data)
            
            # Verify functional data was created
            candles = functional_processor.get_recent_candles()
            assert len(candles) == 1
            assert candles[0].symbol == "BTC-USD"
            assert candles[0].close == Decimal("50500")
            
            updates = functional_processor.get_recent_updates()
            assert len(updates) >= 1
            assert updates[-1].symbol == "BTC-USD"
            
            # Stop processor
            await functional_processor.stop()
            assert functional_processor.get_connection_state().status == ConnectionStatus.DISCONNECTED
    
    def test_error_handling(self):
        """Test error handling in functional processing."""
        processor = create_functional_market_data_processor("BTC-USD", "1m")
        
        # Test with invalid market data
        invalid_data = CurrentMarketData(
            symbol="INVALID",
            timestamp=datetime.now(UTC),
            open=Decimal("-1"),  # Invalid negative price
            high=Decimal("0"),
            low=Decimal("-100"),
            close=Decimal("-50"),
            volume=Decimal("-10"),  # Invalid negative volume
        )
        
        # Should handle gracefully
        try:
            processor._on_market_data_update(invalid_data)
            # Check that data quality reflects the error
            quality = processor.get_data_quality()
            # System should continue functioning
            assert quality.messages_received >= 1
        except Exception as e:
            # Any exceptions should be caught and logged
            pytest.fail(f"Should handle invalid data gracefully, but got: {e}")
    
    def test_performance_with_large_dataset(self):
        """Test performance with a large dataset."""
        processor = create_functional_market_data_processor("BTC-USD", "1m")
        indicator_processor = create_indicator_processor()
        
        # Create large dataset
        large_dataset = []
        for i in range(1000):
            market_data = CurrentMarketData(
                symbol="BTC-USD",
                timestamp=datetime.now(UTC),
                open=Decimal(f"{50000 + i}"),
                high=Decimal(f"{51000 + i}"),
                low=Decimal(f"{49500 + i}"),
                close=Decimal(f"{50500 + i}"),
                volume=Decimal("100.5"),
            )
            large_dataset.append(market_data)
        
        # Process large dataset
        import time
        start_time = time.time()
        
        for market_data in large_dataset:
            processor._on_market_data_update(market_data)
        
        # Convert to functional and process indicators
        candles = processor.get_recent_candles()
        if len(candles) >= 100:  # Ensure sufficient data for indicators
            indicator_processor.process_candles(candles[-100:])
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 10.0, f"Processing took too long: {processing_time}s"
        
        # Verify data integrity
        assert len(processor.get_recent_candles()) == 1000
        assert processor.is_healthy()
    
    def test_connection_state_management(self):
        """Test connection state management."""
        # Test connection state creation and updates
        state = create_connection_state("wss://test.com", "CONNECTING")
        assert state.status == ConnectionStatus.CONNECTING
        assert state.url == "wss://test.com"
        assert not state.is_healthy()
        
        # Test connection state with healthy connection
        healthy_state = create_connection_state("wss://test.com", "CONNECTED")
        healthy_state = healthy_state.__class__(
            status=ConnectionStatus.CONNECTED,
            url=healthy_state.url,
            reconnect_attempts=healthy_state.reconnect_attempts,
            last_error=healthy_state.last_error,
            connected_at=datetime.now(UTC),
            last_message_at=datetime.now(UTC),
        )
        
        assert healthy_state.is_healthy()
    
    def test_data_quality_metrics(self):
        """Test data quality metrics calculation."""
        quality = create_data_quality(
            messages_received=100,
            messages_processed=95,
            validation_failures=5
        )
        
        assert quality.success_rate == 95.0
        assert quality.error_rate == 5.0
        
        # Test quality validation
        from bot.fp.adapters.type_converters import validate_data_quality
        assert validate_data_quality(quality, min_success_rate=90.0)
        assert not validate_data_quality(quality, min_success_rate=98.0)


if __name__ == "__main__":
    # Run some basic tests to verify functionality
    print("Testing Functional Market Data Integration...")
    
    # Test basic functionality
    test_processor = TestFunctionalMarketDataProcessor()
    test_processor.test_initialization()
    test_processor.test_market_data_conversion()
    test_processor.test_candle_properties()
    print("✓ Basic processor tests passed")
    
    # Test indicator functionality
    test_indicators = TestFunctionalIndicatorProcessor()
    test_indicators.test_initialization()
    test_indicators.test_indicator_calculation()
    print("✓ Indicator tests passed")
    
    # Test integration scenarios
    test_integration = TestIntegrationScenarios()
    test_integration.test_error_handling()
    test_integration.test_connection_state_management()
    test_integration.test_data_quality_metrics()
    print("✓ Integration tests passed")
    
    print("All functional market data integration tests completed successfully!")