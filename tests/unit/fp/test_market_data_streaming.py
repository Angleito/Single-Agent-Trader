"""
Functional Programming Market Data Streaming Tests

Test suite for market data streaming with functional programming patterns.
Tests immutable market data types, real-time streaming, and data aggregation.
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from bot.fp.effects.market_data import (
    MarketDataStream,
    StreamingConfig,
)
from bot.fp.effects.market_data_aggregation import (
    CandleAggregator,
    RealTimeCandle,
    TradeAggregator,
    create_trade_aggregator,
)
from bot.fp.types.trading import (
    FunctionalMarketData,
    FuturesMarketData,
)


class TestFunctionalMarketDataTypes:
    """Test immutable market data types for streaming."""

    def test_functional_market_data_creation(self):
        """Test creation of functional market data."""
        market_data = FunctionalMarketData(
            symbol="BTC-USD",
            timestamp=datetime.now(),
            open=Decimal(50000),
            high=Decimal(52000),
            low=Decimal(49000),
            close=Decimal(51000),
            volume=Decimal(100),
        )

        assert market_data.symbol == "BTC-USD"
        assert market_data.is_bullish
        assert market_data.price_change == Decimal(1000)
        assert market_data.price_change_percentage == 2.0

    def test_market_data_validation(self):
        """Test market data validation rules."""
        # Test invalid OHLCV relationships
        with pytest.raises(ValueError, match="High .* must be >= all other prices"):
            FunctionalMarketData(
                symbol="BTC-USD",
                timestamp=datetime.now(),
                open=Decimal(50000),
                high=Decimal(48000),  # High < Open (invalid)
                low=Decimal(47000),
                close=Decimal(49000),
                volume=Decimal(100),
            )

    def test_futures_market_data_with_funding(self):
        """Test futures market data with funding information."""
        base_data = FunctionalMarketData(
            symbol="BTC-PERP",
            timestamp=datetime.now(),
            open=Decimal(50000),
            high=Decimal(51000),
            low=Decimal(49500),
            close=Decimal(50500),
            volume=Decimal(1000),
        )

        futures_data = FuturesMarketData(
            base_data=base_data,
            open_interest=Decimal(5000000),
            funding_rate=0.0001,
            next_funding_time=datetime.now() + timedelta(hours=8),
            mark_price=Decimal(50505),
            index_price=Decimal(50500),
        )

        assert futures_data.symbol == "BTC-PERP"
        assert futures_data.basis == Decimal(5)  # mark - index
        assert abs(futures_data.funding_rate_8h_annualized - 0.1095) < 0.001  # ~0.1095

    def test_market_data_immutable_updates(self):
        """Test immutable updates to market data."""
        original = FunctionalMarketData(
            symbol="ETH-USD",
            timestamp=datetime.now(),
            open=Decimal(3000),
            high=Decimal(3100),
            low=Decimal(2950),
            close=Decimal(3050),
            volume=Decimal(500),
        )

        new_timestamp = datetime.now() + timedelta(minutes=1)
        updated = original.update_timestamp(new_timestamp)

        # Original unchanged
        assert original.timestamp != new_timestamp
        # New instance has updated timestamp
        assert updated.timestamp == new_timestamp
        assert updated.symbol == original.symbol


class TestStreamingConfig:
    """Test streaming configuration."""

    def test_streaming_config_creation(self):
        """Test creation of streaming configuration."""
        config = StreamingConfig(
            symbols=["BTC-USD", "ETH-USD"],
            interval="1m",
            buffer_size=1000,
            enable_aggregation=True,
            max_reconnect_attempts=5,
        )

        assert config.symbols == ["BTC-USD", "ETH-USD"]
        assert config.interval == "1m"
        assert config.buffer_size == 1000
        assert config.enable_aggregation is True

    def test_streaming_config_validation(self):
        """Test streaming configuration validation."""
        # Test invalid interval
        with pytest.raises(ValueError, match="Invalid interval"):
            StreamingConfig(symbols=["BTC-USD"], interval="invalid", buffer_size=1000)

        # Test empty symbols
        with pytest.raises(ValueError, match="At least one symbol required"):
            StreamingConfig(symbols=[], interval="1m", buffer_size=1000)


class MockMarketDataSource:
    """Mock market data source for testing."""

    def __init__(self, symbols: list[str], data_count: int = 100):
        self.symbols = symbols
        self.data_count = data_count
        self.current_index = 0

    async def stream_market_data(self) -> AsyncIterator[FunctionalMarketData]:
        """Stream mock market data."""
        base_time = datetime.now()

        for i in range(self.data_count):
            for symbol in self.symbols:
                # Generate realistic-looking OHLCV data
                base_price = 50000 if "BTC" in symbol else 3000
                price_variation = i * 10

                open_price = Decimal(str(base_price + price_variation))
                high_price = open_price + Decimal(100)
                low_price = open_price - Decimal(50)
                close_price = open_price + Decimal(25)
                volume = Decimal(100) + Decimal(str(i))

                market_data = FunctionalMarketData(
                    symbol=symbol,
                    timestamp=base_time + timedelta(minutes=i),
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                )

                yield market_data

                # Small delay to simulate real-time data
                await asyncio.sleep(0.001)


class TestMarketDataStream:
    """Test market data streaming functionality."""

    @pytest.mark.asyncio
    async def test_basic_market_data_streaming(self):
        """Test basic market data streaming."""
        symbols = ["BTC-USD", "ETH-USD"]
        mock_source = MockMarketDataSource(symbols, data_count=10)

        config = StreamingConfig(symbols=symbols, interval="1m", buffer_size=100)

        # Create stream
        stream = MarketDataStream(config, mock_source)

        # Collect streaming data
        collected_data = []
        async for data in mock_source.stream_market_data():
            collected_data.append(data)
            if len(collected_data) >= 20:  # 10 * 2 symbols
                break

        assert len(collected_data) == 20
        assert len([d for d in collected_data if d.symbol == "BTC-USD"]) == 10
        assert len([d for d in collected_data if d.symbol == "ETH-USD"]) == 10

    @pytest.mark.asyncio
    async def test_market_data_filtering(self):
        """Test filtering market data by symbol."""
        symbols = ["BTC-USD", "ETH-USD", "LTC-USD"]
        mock_source = MockMarketDataSource(symbols, data_count=5)

        # Filter for BTC only
        btc_data = []
        async for data in mock_source.stream_market_data():
            if data.symbol == "BTC-USD":
                btc_data.append(data)

        assert len(btc_data) == 5
        assert all(d.symbol == "BTC-USD" for d in btc_data)

    @pytest.mark.asyncio
    async def test_real_time_data_validation(self):
        """Test real-time validation of streaming data."""
        symbols = ["BTC-USD"]
        mock_source = MockMarketDataSource(symbols, data_count=10)

        invalid_data_count = 0
        valid_data_count = 0

        async for data in mock_source.stream_market_data():
            try:
                # Validate data consistency
                assert data.symbol in symbols
                assert data.high >= data.open
                assert data.high >= data.close
                assert data.low <= data.open
                assert data.low <= data.close
                assert data.volume >= 0
                valid_data_count += 1
            except AssertionError:
                invalid_data_count += 1

        assert valid_data_count == 10
        assert invalid_data_count == 0


class TestTradeAggregator:
    """Test trade aggregation for sub-minute intervals."""

    def test_trade_aggregator_creation(self):
        """Test creation of trade aggregator."""
        aggregator = TradeAggregator(
            symbol="BTC-USD", interval="1s", window_size=timedelta(seconds=1)
        )

        assert aggregator.symbol == "BTC-USD"
        assert aggregator.interval == "1s"
        assert aggregator.window_size == timedelta(seconds=1)

    def test_trade_aggregation_1_second(self):
        """Test 1-second trade aggregation."""
        aggregator = TradeAggregator(
            symbol="BTC-USD", interval="1s", window_size=timedelta(seconds=1)
        )

        base_time = datetime.now()

        # Add trades within 1 second
        trades = [
            {"price": 50000, "size": 0.1, "timestamp": base_time},
            {
                "price": 50010,
                "size": 0.2,
                "timestamp": base_time + timedelta(milliseconds=300),
            },
            {
                "price": 49995,
                "size": 0.15,
                "timestamp": base_time + timedelta(milliseconds=600),
            },
            {
                "price": 50005,
                "size": 0.1,
                "timestamp": base_time + timedelta(milliseconds=900),
            },
        ]

        for trade in trades:
            aggregator.add_trade(
                price=Decimal(str(trade["price"])),
                size=Decimal(str(trade["size"])),
                timestamp=trade["timestamp"],
            )

        # Get aggregated candle
        candle = aggregator.get_current_candle()

        assert candle is not None
        assert candle.symbol == "BTC-USD"
        assert candle.open == Decimal(50000)  # First trade price
        assert candle.high == Decimal(50010)  # Highest trade price
        assert candle.low == Decimal(49995)  # Lowest trade price
        assert candle.close == Decimal(50005)  # Last trade price
        assert candle.volume == Decimal("0.55")  # Sum of all sizes

    def test_trade_aggregation_multiple_windows(self):
        """Test trade aggregation across multiple windows."""
        aggregator = TradeAggregator(
            symbol="ETH-USD", interval="5s", window_size=timedelta(seconds=5)
        )

        base_time = datetime.now()

        # Trades in first window (0-5s)
        first_window_trades = [
            {"price": 3000, "size": 1.0, "timestamp": base_time + timedelta(seconds=1)},
            {"price": 3010, "size": 2.0, "timestamp": base_time + timedelta(seconds=3)},
        ]

        # Trades in second window (5-10s)
        second_window_trades = [
            {"price": 3015, "size": 1.5, "timestamp": base_time + timedelta(seconds=6)},
            {"price": 3005, "size": 1.0, "timestamp": base_time + timedelta(seconds=8)},
        ]

        # Add first window trades
        for trade in first_window_trades:
            aggregator.add_trade(
                price=Decimal(str(trade["price"])),
                size=Decimal(str(trade["size"])),
                timestamp=trade["timestamp"],
            )

        first_candle = aggregator.get_current_candle()
        assert first_candle.volume == Decimal("3.0")

        # Force window completion and start new window
        aggregator.complete_current_window()

        # Add second window trades
        for trade in second_window_trades:
            aggregator.add_trade(
                price=Decimal(str(trade["price"])),
                size=Decimal(str(trade["size"])),
                timestamp=trade["timestamp"],
            )

        second_candle = aggregator.get_current_candle()
        assert second_candle.volume == Decimal("2.5")

    def test_sub_minute_interval_support(self):
        """Test support for sub-minute intervals."""
        intervals = ["1s", "5s", "15s", "30s"]

        for interval in intervals:
            aggregator = create_trade_aggregator("BTC-USD", interval)
            assert aggregator.interval == interval

            # Test that it can handle trades
            aggregator.add_trade(
                price=Decimal(50000), size=Decimal("0.1"), timestamp=datetime.now()
            )

            candle = aggregator.get_current_candle()
            assert candle is not None
            assert candle.symbol == "BTC-USD"


class TestCandleAggregator:
    """Test candle aggregation functionality."""

    def test_candle_aggregator_creation(self):
        """Test creation of candle aggregator."""
        aggregator = CandleAggregator(
            symbol="BTC-USD", source_interval="1s", target_interval="1m"
        )

        assert aggregator.symbol == "BTC-USD"
        assert aggregator.source_interval == "1s"
        assert aggregator.target_interval == "1m"

    def test_1s_to_1m_aggregation(self):
        """Test aggregating 1-second candles to 1-minute candles."""
        aggregator = CandleAggregator(
            symbol="BTC-USD", source_interval="1s", target_interval="1m"
        )

        base_time = datetime.now().replace(second=0, microsecond=0)

        # Create 60 1-second candles
        source_candles = []
        for i in range(60):
            candle = RealTimeCandle(
                symbol="BTC-USD",
                timestamp=base_time + timedelta(seconds=i),
                open=Decimal(str(50000 + i)),
                high=Decimal(str(50020 + i)),
                low=Decimal(str(49980 + i)),
                close=Decimal(str(50010 + i)),
                volume=Decimal("1.0"),
                trade_count=10,
                interval="1s",
            )
            source_candles.append(candle)
            aggregator.add_candle(candle)

        # Get aggregated 1-minute candle
        minute_candle = aggregator.get_aggregated_candle()

        assert minute_candle is not None
        assert minute_candle.symbol == "BTC-USD"
        assert minute_candle.open == Decimal(50000)  # First candle's open
        assert minute_candle.close == Decimal(50069)  # Last candle's close
        assert minute_candle.high == Decimal(50079)  # Highest high
        assert minute_candle.low == Decimal(49980)  # Lowest low
        assert minute_candle.volume == Decimal("60.0")  # Sum of volumes
        assert minute_candle.trade_count == 600  # Sum of trade counts

    def test_partial_aggregation(self):
        """Test partial aggregation with incomplete windows."""
        aggregator = CandleAggregator(
            symbol="ETH-USD", source_interval="5s", target_interval="1m"
        )

        base_time = datetime.now().replace(second=0, microsecond=0)

        # Add only 6 candles (30 seconds of data)
        for i in range(6):
            candle = RealTimeCandle(
                symbol="ETH-USD",
                timestamp=base_time + timedelta(seconds=i * 5),
                open=Decimal(str(3000 + i)),
                high=Decimal(str(3010 + i)),
                low=Decimal(str(2990 + i)),
                close=Decimal(str(3005 + i)),
                volume=Decimal("2.0"),
                trade_count=20,
                interval="5s",
            )
            aggregator.add_candle(candle)

        # Should still be able to get partial aggregation
        partial_candle = aggregator.get_aggregated_candle()

        assert partial_candle is not None
        assert partial_candle.volume == Decimal("12.0")  # 6 * 2.0
        assert partial_candle.trade_count == 120  # 6 * 20


class TestRealTimeCandle:
    """Test real-time candle functionality."""

    def test_real_time_candle_creation(self):
        """Test creation of real-time candle."""
        candle = RealTimeCandle(
            symbol="BTC-USD",
            timestamp=datetime.now(),
            open=Decimal(50000),
            high=Decimal(50100),
            low=Decimal(49900),
            close=Decimal(50050),
            volume=Decimal("10.5"),
            trade_count=100,
            interval="1s",
        )

        assert candle.symbol == "BTC-USD"
        assert candle.interval == "1s"
        assert candle.trade_count == 100
        assert candle.is_bullish
        assert candle.price_change == Decimal(50)

    def test_real_time_candle_updates(self):
        """Test updating real-time candle with new trades."""
        base_time = datetime.now()

        candle = RealTimeCandle(
            symbol="ETH-USD",
            timestamp=base_time,
            open=Decimal(3000),
            high=Decimal(3000),
            low=Decimal(3000),
            close=Decimal(3000),
            volume=Decimal(0),
            trade_count=0,
            interval="1s",
        )

        # Update with new trade
        updated_candle = candle.update_with_trade(
            price=Decimal(3010), size=Decimal("1.5")
        )

        assert updated_candle.high == Decimal(3010)
        assert updated_candle.close == Decimal(3010)
        assert updated_candle.volume == Decimal("1.5")
        assert updated_candle.trade_count == 1

        # Original candle should be unchanged (immutable)
        assert candle.volume == Decimal(0)
        assert candle.trade_count == 0

    def test_candle_immutability(self):
        """Test that candles are immutable."""
        candle = RealTimeCandle(
            symbol="BTC-USD",
            timestamp=datetime.now(),
            open=Decimal(50000),
            high=Decimal(50100),
            low=Decimal(49900),
            close=Decimal(50050),
            volume=Decimal("10.5"),
            trade_count=100,
            interval="1s",
        )

        # Should not be able to modify fields
        with pytest.raises(AttributeError):
            candle.close = Decimal(51000)  # type: ignore


@pytest.mark.asyncio
class TestStreamingPerformance:
    """Test streaming performance and efficiency."""

    async def test_high_frequency_streaming(self):
        """Test high-frequency data streaming performance."""
        symbols = ["BTC-USD"]
        mock_source = MockMarketDataSource(symbols, data_count=1000)

        start_time = datetime.now()

        # Stream 1000 data points
        data_count = 0
        async for data in mock_source.stream_market_data():
            data_count += 1

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        assert data_count == 1000
        # Should complete within reasonable time (allowing for CI/CD environment)
        assert duration < 5.0

    async def test_concurrent_symbol_streaming(self):
        """Test concurrent streaming of multiple symbols."""
        symbols = ["BTC-USD", "ETH-USD", "LTC-USD", "ADA-USD"]

        async def stream_symbol(symbol: str) -> list[FunctionalMarketData]:
            source = MockMarketDataSource([symbol], data_count=100)
            data = []
            async for market_data in source.stream_market_data():
                data.append(market_data)
            return data

        # Stream all symbols concurrently
        tasks = [stream_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)

        assert len(results) == 4
        assert all(len(result) == 100 for result in results)

        # Verify symbol separation
        for i, symbol in enumerate(symbols):
            assert all(data.symbol == symbol for data in results[i])

    async def test_streaming_with_aggregation_performance(self):
        """Test performance of streaming with real-time aggregation."""
        aggregator = TradeAggregator(
            symbol="BTC-USD", interval="1s", window_size=timedelta(seconds=1)
        )

        base_time = datetime.now()

        # Simulate high-frequency trades
        start_time = datetime.now()

        for i in range(1000):
            aggregator.add_trade(
                price=Decimal(str(50000 + (i % 100))),
                size=Decimal("0.01"),
                timestamp=base_time + timedelta(milliseconds=i),
            )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Should handle 1000 trades efficiently
        assert duration < 1.0

        # Verify aggregation worked
        candle = aggregator.get_current_candle()
        assert candle is not None
        assert candle.volume == Decimal("10.0")  # 1000 * 0.01


class TestMarketDataValidation:
    """Test market data validation in streaming context."""

    def test_streaming_data_consistency(self):
        """Test consistency validation of streaming data."""

        def validate_ohlcv_consistency(data: FunctionalMarketData) -> bool:
            """Validate OHLCV relationships."""
            try:
                assert data.high >= data.open
                assert data.high >= data.close
                assert data.high >= data.low
                assert data.low <= data.open
                assert data.low <= data.close
                assert data.volume >= 0
                return True
            except AssertionError:
                return False

        # Test with valid data
        valid_data = FunctionalMarketData(
            symbol="BTC-USD",
            timestamp=datetime.now(),
            open=Decimal(50000),
            high=Decimal(50100),
            low=Decimal(49900),
            close=Decimal(50050),
            volume=Decimal(100),
        )

        assert validate_ohlcv_consistency(valid_data)

    def test_streaming_timestamp_validation(self):
        """Test timestamp validation in streaming data."""
        base_time = datetime.now()

        # Create sequence of market data with proper timestamps
        data_sequence = []
        for i in range(5):
            data = FunctionalMarketData(
                symbol="ETH-USD",
                timestamp=base_time + timedelta(minutes=i),
                open=Decimal(str(3000 + i)),
                high=Decimal(str(3010 + i)),
                low=Decimal(str(2990 + i)),
                close=Decimal(str(3005 + i)),
                volume=Decimal(100),
            )
            data_sequence.append(data)

        # Validate timestamps are in order
        for i in range(1, len(data_sequence)):
            assert data_sequence[i].timestamp > data_sequence[i - 1].timestamp

    def test_futures_data_validation(self):
        """Test validation of futures-specific data."""
        base_data = FunctionalMarketData(
            symbol="BTC-PERP",
            timestamp=datetime.now(),
            open=Decimal(50000),
            high=Decimal(50100),
            low=Decimal(49900),
            close=Decimal(50050),
            volume=Decimal(1000),
        )

        # Valid futures data
        futures_data = FuturesMarketData(
            base_data=base_data,
            open_interest=Decimal(5000000),
            funding_rate=0.0001,
            mark_price=Decimal(50055),
            index_price=Decimal(50050),
        )

        assert futures_data.open_interest > 0
        assert abs(futures_data.funding_rate) < 1.0  # Reasonable funding rate
        assert futures_data.mark_price > 0
        assert futures_data.index_price > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
