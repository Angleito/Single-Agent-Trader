"""
Functional Programming Real-time Data Streaming Tests

This module tests real-time data streaming, functional data pipelines,
latency measurement, and streaming data processing using functional patterns.
"""

import pytest
import asyncio
import time
from datetime import datetime, UTC, timedelta
from decimal import Decimal
from typing import Any, AsyncIterator, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call
from concurrent.futures import ThreadPoolExecutor
import threading

from bot.fp.types.market import (
    Candle,
    Trade,
    Ticker,
    RealtimeUpdate,
    AggregatedData,
    MarketDataStream,
    ConnectionState,
    ConnectionStatus,
    DataQuality,
)
from bot.fp.data_pipeline import (
    FunctionalDataPipeline,
    StreamProcessor,
    DataStreamEffect,
    create_data_pipeline,
    process_streaming_data,
    aggregate_streaming_data,
    filter_streaming_data,
    transform_streaming_data,
)
from bot.fp.effects.market_data_aggregation import (
    aggregate_trades_to_candles,
    calculate_vwap_effect,
    calculate_streaming_stats,
    detect_price_anomalies,
    measure_latency_effect,
)
from bot.fp.core.either import Either, Left, Right
from bot.fp.core.option import Option, Some, None_ as NoneOption
from bot.fp.adapters.market_data_adapter import (
    FunctionalMarketDataProcessor,
    create_functional_market_data_processor,
)


class TestFunctionalDataPipeline:
    """Test functional data pipeline for real-time streaming."""
    
    def test_data_pipeline_creation(self):
        """Test data pipeline creation and initialization."""
        pipeline = create_data_pipeline(
            symbol="BTC-USD",
            buffer_size=1000,
            batch_size=100
        )
        
        assert pipeline.symbol == "BTC-USD"
        assert pipeline.buffer_size == 1000
        assert pipeline.batch_size == 100
        assert pipeline.is_active == False
        assert len(pipeline.processors) == 0
    
    def test_pipeline_processor_registration(self):
        """Test registering processors with the pipeline."""
        pipeline = create_data_pipeline("BTC-USD")
        
        # Mock processors
        candle_processor = MagicMock()
        trade_processor = MagicMock()
        
        # Register processors
        pipeline.add_processor("candle", candle_processor)
        pipeline.add_processor("trade", trade_processor)
        
        assert "candle" in pipeline.processors
        assert "trade" in pipeline.processors
        assert len(pipeline.processors) == 2
    
    async def test_pipeline_start_stop(self):
        """Test starting and stopping the data pipeline."""
        pipeline = create_data_pipeline("BTC-USD")
        
        # Start pipeline
        await pipeline.start()
        assert pipeline.is_active == True
        
        # Stop pipeline
        await pipeline.stop()
        assert pipeline.is_active == False
    
    async def test_pipeline_data_processing(self):
        """Test data processing through the pipeline."""
        pipeline = create_data_pipeline("BTC-USD", batch_size=5)
        
        # Mock processor
        processed_data = []
        def mock_processor(data_batch):
            processed_data.extend(data_batch)
        
        pipeline.add_processor("test", mock_processor)
        
        # Start pipeline
        await pipeline.start()
        
        # Add test data
        test_trades = [
            Trade(
                id=f"trade-{i}",
                timestamp=datetime.now(UTC),
                price=Decimal(f"{50000 + i}"),
                size=Decimal("0.1"),
                side="BUY",
                symbol="BTC-USD"
            )
            for i in range(10)
        ]
        
        # Process data
        for trade in test_trades:
            await pipeline.process_data(trade)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Stop pipeline
        await pipeline.stop()
        
        # Verify processing
        assert len(processed_data) == 10
        assert all(isinstance(item, Trade) for item in processed_data)


class TestStreamingDataEffects:
    """Test streaming data processing effects."""
    
    def test_process_streaming_data_effect(self):
        """Test streaming data processing effect."""
        stream_data = [
            Trade(
                id="1", timestamp=datetime.now(UTC), price=Decimal("50000"),
                size=Decimal("0.1"), side="BUY", symbol="BTC-USD"
            ),
            Trade(
                id="2", timestamp=datetime.now(UTC), price=Decimal("50100"),
                size=Decimal("0.2"), side="SELL", symbol="BTC-USD"
            ),
        ]
        
        result = process_streaming_data(stream_data)
        assert isinstance(result, Right)
        
        processed_stream = result.value
        assert len(processed_stream) == 2
        assert all(isinstance(item, Trade) for item in processed_stream)
    
    def test_filter_streaming_data_effect(self):
        """Test filtering streaming data."""
        stream_data = [
            Trade(
                id="1", timestamp=datetime.now(UTC), price=Decimal("50000"),
                size=Decimal("0.1"), side="BUY", symbol="BTC-USD"
            ),
            Trade(
                id="2", timestamp=datetime.now(UTC), price=Decimal("50100"),
                size=Decimal("0.2"), side="SELL", symbol="BTC-USD"
            ),
            Trade(
                id="3", timestamp=datetime.now(UTC), price=Decimal("50050"),
                size=Decimal("0.05"), side="BUY", symbol="BTC-USD"
            ),
        ]
        
        # Filter for buy trades only
        buy_filter = lambda trade: trade.side == "BUY"
        result = filter_streaming_data(stream_data, buy_filter)
        
        assert isinstance(result, Right)
        filtered_data = result.value
        assert len(filtered_data) == 2
        assert all(trade.side == "BUY" for trade in filtered_data)
    
    def test_transform_streaming_data_effect(self):
        """Test transforming streaming data."""
        trades = [
            Trade(
                id="1", timestamp=datetime.now(UTC), price=Decimal("50000"),
                size=Decimal("0.1"), side="BUY", symbol="BTC-USD"
            ),
            Trade(
                id="2", timestamp=datetime.now(UTC), price=Decimal("50100"),
                size=Decimal("0.2"), side="SELL", symbol="BTC-USD"
            ),
        ]
        
        # Transform to extract prices
        price_extractor = lambda trade: trade.price
        result = transform_streaming_data(trades, price_extractor)
        
        assert isinstance(result, Right)
        prices = result.value
        assert len(prices) == 2
        assert prices[0] == Decimal("50000")
        assert prices[1] == Decimal("50100")
    
    def test_aggregate_streaming_data_effect(self):
        """Test aggregating streaming data."""
        start_time = datetime.now(UTC)
        trades = [
            Trade(
                id=f"trade-{i}",
                timestamp=start_time + timedelta(seconds=i),
                price=Decimal(f"{50000 + i * 10}"),
                size=Decimal("0.1"),
                side="BUY" if i % 2 == 0 else "SELL",
                symbol="BTC-USD"
            )
            for i in range(10)
        ]
        
        # Aggregate by time window
        result = aggregate_streaming_data(trades, timedelta(minutes=1))
        assert isinstance(result, Right)
        
        aggregated = result.value
        assert aggregated.symbol == "BTC-USD"
        assert aggregated.trade_count == 10
        assert aggregated.volume_total == Decimal("1.0")  # 10 * 0.1


class TestTradeAggregationEffects:
    """Test trade aggregation to candles using effects."""
    
    def test_aggregate_trades_to_candles_effect(self):
        """Test aggregating trades to candles."""
        start_time = datetime.now(UTC).replace(second=0, microsecond=0)
        trades = []
        
        # Create trades over 2 minutes
        for minute in range(2):
            for second in range(0, 60, 10):  # Every 10 seconds
                trades.append(Trade(
                    id=f"trade-{minute}-{second}",
                    timestamp=start_time + timedelta(minutes=minute, seconds=second),
                    price=Decimal(f"{50000 + minute * 100 + second}"),
                    size=Decimal("0.1"),
                    side="BUY" if second % 20 == 0 else "SELL",
                    symbol="BTC-USD"
                ))
        
        # Aggregate to 1-minute candles
        result = aggregate_trades_to_candles(trades, timedelta(minutes=1))
        assert isinstance(result, Right)
        
        candles = result.value
        assert len(candles) == 2  # 2 minutes of data
        
        # Verify candle properties
        first_candle = candles[0]
        assert first_candle.symbol == "BTC-USD"
        assert first_candle.timestamp == start_time
        assert first_candle.open == Decimal("50000")  # First trade price
        assert first_candle.volume == Decimal("0.6")  # 6 trades * 0.1
    
    def test_vwap_calculation_effect(self):
        """Test VWAP calculation using effects."""
        trades = [
            Trade(
                id="1", timestamp=datetime.now(UTC), price=Decimal("50000"),
                size=Decimal("1.0"), side="BUY", symbol="BTC-USD"
            ),
            Trade(
                id="2", timestamp=datetime.now(UTC), price=Decimal("51000"),
                size=Decimal("2.0"), side="SELL", symbol="BTC-USD"
            ),
            Trade(
                id="3", timestamp=datetime.now(UTC), price=Decimal("49000"),
                size=Decimal("1.0"), side="BUY", symbol="BTC-USD"
            ),
        ]
        
        result = calculate_vwap_effect(trades)
        assert isinstance(result, Right)
        
        vwap = result.value
        # VWAP = (50000*1 + 51000*2 + 49000*1) / (1+2+1) = 200000/4 = 50000
        expected_vwap = Decimal("50250")  # (50000 + 102000 + 49000) / 4
        assert vwap == expected_vwap
    
    def test_streaming_stats_calculation(self):
        """Test streaming statistics calculation."""
        trades = [
            Trade(
                id=f"trade-{i}",
                timestamp=datetime.now(UTC),
                price=Decimal(f"{50000 + i * 100}"),
                size=Decimal("0.1"),
                side="BUY" if i % 2 == 0 else "SELL",
                symbol="BTC-USD"
            )
            for i in range(10)
        ]
        
        result = calculate_streaming_stats(trades)
        assert isinstance(result, Right)
        
        stats = result.value
        assert "count" in stats
        assert "volume_total" in stats
        assert "price_min" in stats
        assert "price_max" in stats
        assert "price_avg" in stats
        
        assert stats["count"] == 10
        assert stats["volume_total"] == Decimal("1.0")
        assert stats["price_min"] == Decimal("50000")
        assert stats["price_max"] == Decimal("50900")


class TestLatencyMeasurement:
    """Test latency measurement for streaming data."""
    
    def test_latency_measurement_effect(self):
        """Test latency measurement using effects."""
        # Create a message with timestamp
        message_timestamp = datetime.now(UTC) - timedelta(milliseconds=50)
        current_timestamp = datetime.now(UTC)
        
        result = measure_latency_effect(message_timestamp, current_timestamp)
        assert isinstance(result, Right)
        
        latency_ms = result.value
        assert 40 <= latency_ms <= 60  # Should be around 50ms
    
    def test_end_to_end_latency_measurement(self):
        """Test end-to-end latency measurement."""
        processor = create_functional_market_data_processor("BTC-USD", "1m")
        
        # Mock callback to measure latency
        latencies = []
        def latency_callback(update: RealtimeUpdate):
            if update.latency_ms is not None:
                latencies.append(update.latency_ms)
        
        processor.add_update_callback(latency_callback)
        
        # Simulate message with known timestamp
        message_time = datetime.now(UTC) - timedelta(milliseconds=25)
        ticker_message = {
            "channel": "ticker",
            "timestamp": message_time.isoformat(),
            "events": [{
                "type": "update",
                "tickers": [{
                    "product_id": "BTC-USD",
                    "price": "50000.00"
                }]
            }]
        }
        
        processor.process_websocket_message(ticker_message)
        
        # Check if latency was measured
        updates = processor.get_recent_updates()
        if updates and updates[-1].latency_ms is not None:
            assert 20 <= updates[-1].latency_ms <= 35  # Around 25ms plus processing
    
    def test_latency_distribution_tracking(self):
        """Test tracking latency distribution over time."""
        latencies = []
        
        # Simulate various latencies
        base_time = datetime.now(UTC)
        for i in range(100):
            message_time = base_time - timedelta(milliseconds=i % 50)  # 0-49ms ago
            current_time = base_time
            
            result = measure_latency_effect(message_time, current_time)
            if isinstance(result, Right):
                latencies.append(result.value)
        
        # Analyze distribution
        assert len(latencies) == 100
        assert min(latencies) >= 0
        assert max(latencies) <= 60  # Should be reasonable
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        assert 20 <= avg_latency <= 30  # Should be reasonable average


class TestPriceAnomalyDetection:
    """Test price anomaly detection in streaming data."""
    
    def test_price_anomaly_detection_effect(self):
        """Test price anomaly detection."""
        # Normal trades
        normal_trades = [
            Trade(
                id=f"trade-{i}",
                timestamp=datetime.now(UTC),
                price=Decimal(f"{50000 + i * 10}"),  # Small increments
                size=Decimal("0.1"),
                side="BUY",
                symbol="BTC-USD"
            )
            for i in range(10)
        ]
        
        # Add an anomalous trade
        anomalous_trade = Trade(
            id="anomaly",
            timestamp=datetime.now(UTC),
            price=Decimal("60000"),  # 20% jump
            size=Decimal("0.1"),
            side="BUY",
            symbol="BTC-USD"
        )
        
        all_trades = normal_trades + [anomalous_trade]
        
        result = detect_price_anomalies(all_trades, threshold_pct=10.0)
        assert isinstance(result, Right)
        
        anomalies = result.value
        assert len(anomalies) >= 1
        assert anomalies[0].id == "anomaly"
    
    def test_volume_anomaly_detection(self):
        """Test volume anomaly detection."""
        # Normal volume trades
        normal_trades = [
            Trade(
                id=f"trade-{i}",
                timestamp=datetime.now(UTC),
                price=Decimal("50000"),
                size=Decimal("0.1"),  # Normal size
                side="BUY",
                symbol="BTC-USD"
            )
            for i in range(10)
        ]
        
        # Add a high volume trade
        high_volume_trade = Trade(
            id="high_volume",
            timestamp=datetime.now(UTC),
            price=Decimal("50000"),
            size=Decimal("10.0"),  # 100x normal volume
            side="BUY",
            symbol="BTC-USD"
        )
        
        all_trades = normal_trades + [high_volume_trade]
        
        # Detect volume anomalies
        result = detect_price_anomalies(all_trades, volume_threshold=5.0)
        assert isinstance(result, Right)
        
        # Should detect the high volume trade
        anomalies = result.value
        volume_anomalies = [t for t in anomalies if t.size > Decimal("5.0")]
        assert len(volume_anomalies) >= 1


class TestRealTimeDataStream:
    """Test real-time data stream functionality."""
    
    def test_market_data_stream_creation(self):
        """Test creating a market data stream."""
        # Create connection states
        healthy_state = ConnectionState(
            status=ConnectionStatus.CONNECTED,
            url="wss://test.com",
            connected_at=datetime.now(UTC),
            last_message_at=datetime.now(UTC)
        )
        
        quality = DataQuality(
            timestamp=datetime.now(UTC),
            messages_received=100,
            messages_processed=100,
            validation_failures=0
        )
        
        stream = MarketDataStream(
            symbol="BTC-USD",
            exchanges=["coinbase"],
            connection_states={"coinbase": healthy_state},
            data_quality=quality,
            active=True
        )
        
        # Should be immutable
        with pytest.raises(AttributeError):
            stream.symbol = "ETH-USD"  # type: ignore
        
        # Health check should work
        assert stream.overall_health
        assert "coinbase" in stream.get_healthy_exchanges()
    
    def test_stream_health_monitoring(self):
        """Test stream health monitoring."""
        # Healthy stream
        healthy_state = ConnectionState(
            status=ConnectionStatus.CONNECTED,
            url="wss://test.com",
            connected_at=datetime.now(UTC),
            last_message_at=datetime.now(UTC)
        )
        
        good_quality = DataQuality(
            timestamp=datetime.now(UTC),
            messages_received=100,
            messages_processed=95,
            validation_failures=5
        )
        
        healthy_stream = MarketDataStream(
            symbol="BTC-USD",
            exchanges=["coinbase"],
            connection_states={"coinbase": healthy_state},
            data_quality=good_quality,
            active=True
        )
        
        assert healthy_stream.overall_health
        
        # Unhealthy stream - poor data quality
        poor_quality = DataQuality(
            timestamp=datetime.now(UTC),
            messages_received=100,
            messages_processed=80,
            validation_failures=20
        )
        
        unhealthy_stream = MarketDataStream(
            symbol="BTC-USD",
            exchanges=["coinbase"],
            connection_states={"coinbase": healthy_state},
            data_quality=poor_quality,
            active=True
        )
        
        assert not unhealthy_stream.overall_health
    
    def test_multi_exchange_stream(self):
        """Test multi-exchange data stream."""
        # Multiple exchanges with different states
        coinbase_state = ConnectionState(
            status=ConnectionStatus.CONNECTED,
            url="wss://coinbase.com",
            connected_at=datetime.now(UTC),
            last_message_at=datetime.now(UTC)
        )
        
        binance_state = ConnectionState(
            status=ConnectionStatus.ERROR,
            url="wss://binance.com",
            last_error="Connection failed"
        )
        
        quality = DataQuality(
            timestamp=datetime.now(UTC),
            messages_received=100,
            messages_processed=100,
            validation_failures=0
        )
        
        multi_stream = MarketDataStream(
            symbol="BTC-USD",
            exchanges=["coinbase", "binance"],
            connection_states={
                "coinbase": coinbase_state,
                "binance": binance_state
            },
            data_quality=quality,
            active=True
        )
        
        # Should have one healthy exchange
        healthy_exchanges = multi_stream.get_healthy_exchanges()
        assert len(healthy_exchanges) == 1
        assert "coinbase" in healthy_exchanges
        assert "binance" not in healthy_exchanges
        
        # Overall health should still be good (at least one healthy exchange)
        assert multi_stream.overall_health


class TestStreamingPerformance:
    """Test streaming performance characteristics."""
    
    def test_high_frequency_data_processing(self):
        """Test processing high-frequency streaming data."""
        processor = create_functional_market_data_processor("BTC-USD", "1m")
        
        # Generate high-frequency messages
        messages = []
        start_time = time.time()
        
        for i in range(1000):
            messages.append({
                "channel": "ticker",
                "timestamp": f"2024-01-01T12:00:{i % 60:02d}.{(i // 60) * 60:03d}Z",
                "events": [{
                    "type": "update",
                    "tickers": [{
                        "product_id": "BTC-USD",
                        "price": f"{50000 + i}.00"
                    }]
                }]
            })
        
        # Process all messages
        process_start = time.time()
        
        for message in messages:
            processor.process_websocket_message(message)
        
        process_end = time.time()
        processing_time = process_end - process_start
        
        # Should process quickly
        assert processing_time < 2.0  # Less than 2 seconds for 1000 messages
        
        # Verify all messages were processed (within queue limits)
        updates = processor.get_recent_updates()
        assert len(updates) >= 500  # At least half should be retained
    
    def test_concurrent_stream_processing(self):
        """Test concurrent processing of multiple streams."""
        processors = [
            create_functional_market_data_processor(f"SYMBOL-{i}", "1m")
            for i in range(5)
        ]
        
        # Process messages concurrently
        def process_symbol_messages(processor, symbol_id):
            for i in range(100):
                message = {
                    "channel": "ticker",
                    "timestamp": f"2024-01-01T12:00:{i:02d}Z",
                    "events": [{
                        "type": "update",
                        "tickers": [{
                            "product_id": f"SYMBOL-{symbol_id}",
                            "price": f"{50000 + i}.00"
                        }]
                    }]
                }
                processor.process_websocket_message(message)
        
        # Use thread pool for concurrent processing
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(process_symbol_messages, processors[i], i)
                for i in range(5)
            ]
            
            for future in futures:
                future.result()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete within reasonable time
        assert total_time < 5.0  # Less than 5 seconds for 5 concurrent streams
        
        # Verify all processors received messages
        for processor in processors:
            updates = processor.get_recent_updates()
            assert len(updates) >= 50  # Should have processed significant portion
    
    def test_memory_usage_streaming(self):
        """Test memory usage during streaming."""
        processor = create_functional_market_data_processor("BTC-USD", "1m")
        
        # Process many messages to test memory management
        for i in range(2000):  # More than queue limits
            message = {
                "channel": "ticker",
                "timestamp": f"2024-01-01T12:00:{i % 60:02d}.{i // 60:03d}Z",
                "events": [{
                    "type": "update",
                    "tickers": [{
                        "product_id": "BTC-USD",
                        "price": f"{50000 + i}.00"
                    }]
                }]
            }
            processor.process_websocket_message(message)
        
        # Check memory is bounded by queue limits
        updates = processor.get_recent_updates()
        candles = processor.get_recent_candles()
        
        assert len(updates) <= 500  # Queue limit for updates
        assert len(candles) <= 1000  # Queue limit for candles
        
        # Latest data should be preserved
        if updates:
            latest_update = updates[-1]
            # Should contain recent price data
            assert "51" in str(latest_update.data) or "2000" in str(latest_update.data)


class TestStreamingDataIntegration:
    """Test integration of streaming data with other components."""
    
    async def test_streaming_to_indicator_pipeline(self):
        """Test streaming data feeding into indicator calculations."""
        processor = create_functional_market_data_processor("BTC-USD", "1m")
        
        # Mock indicator processor
        indicator_results = []
        def indicator_callback(candle):
            # Mock indicator calculation
            result = {
                "symbol": candle.symbol,
                "timestamp": candle.timestamp,
                "rsi": 50.0,  # Mock RSI
                "ema": float(candle.close),
                "volume": float(candle.volume)
            }
            indicator_results.append(result)
        
        processor.add_candle_callback(indicator_callback)
        
        # Simulate market data
        from bot.trading_types import MarketData
        
        market_data = MarketData(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49500"),
            close=Decimal("50500"),
            volume=Decimal("100.5")
        )
        
        # Process through streaming pipeline
        processor._on_market_data_update(market_data)
        
        # Verify indicator pipeline was triggered
        assert len(indicator_results) == 1
        assert indicator_results[0]["symbol"] == "BTC-USD"
        assert indicator_results[0]["ema"] == 50500.0
    
    def test_streaming_data_persistence(self):
        """Test persistence of streaming data."""
        processor = create_functional_market_data_processor("BTC-USD", "1m")
        
        # Mock persistence layer
        persisted_data = []
        def persistence_callback(update):
            persisted_data.append({
                "timestamp": update.timestamp,
                "symbol": update.symbol,
                "type": update.update_type,
                "data": update.data
            })
        
        processor.add_update_callback(persistence_callback)
        
        # Process streaming updates
        ticker_message = {
            "channel": "ticker",
            "timestamp": "2024-01-01T12:00:00Z",
            "events": [{
                "type": "update",
                "tickers": [{
                    "product_id": "BTC-USD",
                    "price": "50000.00"
                }]
            }]
        }
        
        processor.process_websocket_message(ticker_message)
        
        # Verify persistence
        assert len(persisted_data) >= 1
        assert persisted_data[0]["symbol"] == "BTC-USD"
        assert persisted_data[0]["type"] == "ticker"


if __name__ == "__main__":
    # Run some basic functionality tests
    print("Testing Functional Real-time Data Streaming...")
    
    # Test data pipeline
    test_pipeline = TestFunctionalDataPipeline()
    test_pipeline.test_data_pipeline_creation()
    test_pipeline.test_pipeline_processor_registration()
    print("✓ Functional data pipeline tests passed")
    
    # Test streaming effects
    test_effects = TestStreamingDataEffects()
    test_effects.test_process_streaming_data_effect()
    test_effects.test_filter_streaming_data_effect()
    test_effects.test_transform_streaming_data_effect()
    test_effects.test_aggregate_streaming_data_effect()
    print("✓ Streaming data effects tests passed")
    
    # Test trade aggregation
    test_aggregation = TestTradeAggregationEffects()
    test_aggregation.test_aggregate_trades_to_candles_effect()
    test_aggregation.test_vwap_calculation_effect()
    test_aggregation.test_streaming_stats_calculation()
    print("✓ Trade aggregation effects tests passed")
    
    # Test latency measurement
    test_latency = TestLatencyMeasurement()
    test_latency.test_latency_measurement_effect()
    test_latency.test_end_to_end_latency_measurement()
    test_latency.test_latency_distribution_tracking()
    print("✓ Latency measurement tests passed")
    
    # Test anomaly detection
    test_anomaly = TestPriceAnomalyDetection()
    test_anomaly.test_price_anomaly_detection_effect()
    test_anomaly.test_volume_anomaly_detection()
    print("✓ Price anomaly detection tests passed")
    
    print("All functional real-time streaming tests completed successfully!")