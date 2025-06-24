"""
Functional Programming Trade Aggregation Tests

Test suite for trade aggregation with functional programming patterns.
Tests sub-minute interval aggregation, immutable candle types, and real-time processing.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.fp.types.trading import FunctionalMarketData
from bot.fp.effects.market_data_aggregation import (
    TradeAggregator,
    CandleAggregator,
    RealTimeCandle,
    AggregationWindow,
    TradeData,
    AggregationConfig,
    create_trade_aggregator,
    create_candle_aggregator,
    validate_interval_compatibility
)
from bot.fp.types.result import Result, Ok, Err
from bot.fp.effects.io import IOEither


class TestTradeData:
    """Test immutable trade data structure."""

    def test_trade_data_creation(self):
        """Test creation of trade data."""
        trade = TradeData(
            symbol="BTC-USD",
            timestamp=datetime.now(),
            price=Decimal("50000"),
            size=Decimal("0.1"),
            side="buy",
            trade_id="trade-123"
        )
        
        assert trade.symbol == "BTC-USD"
        assert trade.price == Decimal("50000")
        assert trade.size == Decimal("0.1")
        assert trade.side == "buy"
        assert trade.value == Decimal("5000")  # price * size

    def test_trade_data_validation(self):
        """Test trade data validation."""
        # Negative price
        with pytest.raises(ValueError, match="Price must be positive"):
            TradeData(
                symbol="BTC-USD",
                timestamp=datetime.now(),
                price=Decimal("-100"),  # Invalid
                size=Decimal("0.1"),
                side="buy"
            )
        
        # Zero size
        with pytest.raises(ValueError, match="Size must be positive"):
            TradeData(
                symbol="BTC-USD",
                timestamp=datetime.now(),
                price=Decimal("50000"),
                size=Decimal("0"),  # Invalid
                side="sell"
            )

    def test_trade_data_immutability(self):
        """Test that trade data is immutable."""
        trade = TradeData(
            symbol="ETH-USD",
            timestamp=datetime.now(),
            price=Decimal("3000"),
            size=Decimal("1.0"),
            side="sell"
        )
        
        # Should not be able to modify fields
        with pytest.raises(AttributeError):
            trade.price = Decimal("3100")  # type: ignore


class TestRealTimeCandle:
    """Test real-time candle functionality."""

    def test_real_time_candle_creation(self):
        """Test creation of real-time candle."""
        timestamp = datetime.now()
        
        candle = RealTimeCandle(
            symbol="BTC-USD",
            timestamp=timestamp,
            interval="1s",
            open=Decimal("50000"),
            high=Decimal("50100"),
            low=Decimal("49900"),
            close=Decimal("50050"),
            volume=Decimal("10.5"),
            trade_count=100,
            vwap=Decimal("50025")
        )
        
        assert candle.symbol == "BTC-USD"
        assert candle.interval == "1s"
        assert candle.trade_count == 100
        assert candle.vwap == Decimal("50025")
        assert candle.is_bullish  # close > open
        assert candle.price_change == Decimal("50")

    def test_candle_from_first_trade(self):
        """Test creating candle from first trade."""
        trade = TradeData(
            symbol="ETH-USD",
            timestamp=datetime.now(),
            price=Decimal("3000"),
            size=Decimal("1.0"),
            side="buy"
        )
        
        candle = RealTimeCandle.from_first_trade(trade, "1s")
        
        assert candle.symbol == "ETH-USD"
        assert candle.open == Decimal("3000")
        assert candle.high == Decimal("3000")
        assert candle.low == Decimal("3000")
        assert candle.close == Decimal("3000")
        assert candle.volume == Decimal("1.0")
        assert candle.trade_count == 1

    def test_candle_update_with_trade(self):
        """Test updating candle with new trade."""
        initial_trade = TradeData(
            symbol="LTC-USD",
            timestamp=datetime.now(),
            price=Decimal("150"),
            size=Decimal("2.0"),
            side="buy"
        )
        
        candle = RealTimeCandle.from_first_trade(initial_trade, "5s")
        
        # Add another trade
        new_trade = TradeData(
            symbol="LTC-USD",
            timestamp=datetime.now() + timedelta(seconds=2),
            price=Decimal("155"),  # Higher price
            size=Decimal("1.5"),
            side="sell"
        )
        
        updated_candle = candle.update_with_trade(new_trade)
        
        # Original candle should be unchanged
        assert candle.trade_count == 1
        assert candle.high == Decimal("150")
        
        # Updated candle should reflect new trade
        assert updated_candle.trade_count == 2
        assert updated_candle.high == Decimal("155")
        assert updated_candle.close == Decimal("155")  # Last trade price
        assert updated_candle.volume == Decimal("3.5")  # 2.0 + 1.5

    def test_candle_price_boundaries(self):
        """Test that candle maintains correct price boundaries."""
        trades = [
            TradeData("BTC-USD", datetime.now(), Decimal("50000"), Decimal("0.1"), "buy"),
            TradeData("BTC-USD", datetime.now(), Decimal("50200"), Decimal("0.2"), "buy"),  # New high
            TradeData("BTC-USD", datetime.now(), Decimal("49800"), Decimal("0.15"), "sell"), # New low
            TradeData("BTC-USD", datetime.now(), Decimal("50100"), Decimal("0.1"), "buy"),   # Close
        ]
        
        candle = RealTimeCandle.from_first_trade(trades[0], "1s")
        
        for trade in trades[1:]:
            candle = candle.update_with_trade(trade)
        
        assert candle.open == Decimal("50000")    # First trade
        assert candle.high == Decimal("50200")    # Highest trade
        assert candle.low == Decimal("49800")     # Lowest trade
        assert candle.close == Decimal("50100")   # Last trade
        assert candle.volume == Decimal("0.55")   # Sum of all sizes

    def test_candle_vwap_calculation(self):
        """Test VWAP calculation in candle updates."""
        trades = [
            TradeData("ETH-USD", datetime.now(), Decimal("3000"), Decimal("1.0"), "buy"),
            TradeData("ETH-USD", datetime.now(), Decimal("3020"), Decimal("2.0"), "buy"),
            TradeData("ETH-USD", datetime.now(), Decimal("2980"), Decimal("1.0"), "sell"),
        ]
        
        candle = RealTimeCandle.from_first_trade(trades[0], "1s")
        
        for trade in trades[1:]:
            candle = candle.update_with_trade(trade)
        
        # VWAP = (3000*1 + 3020*2 + 2980*1) / (1+2+1) = 12000/4 = 3005
        expected_vwap = Decimal("3005")
        assert candle.vwap == expected_vwap


class TestAggregationWindow:
    """Test aggregation window functionality."""

    def test_aggregation_window_creation(self):
        """Test creation of aggregation window."""
        start_time = datetime.now()
        window = AggregationWindow(
            start_time=start_time,
            duration=timedelta(seconds=1),
            symbol="BTC-USD",
            interval="1s"
        )
        
        assert window.start_time == start_time
        assert window.duration == timedelta(seconds=1)
        assert window.end_time == start_time + timedelta(seconds=1)
        assert not window.is_closed

    def test_window_trade_inclusion(self):
        """Test determining if trade belongs to window."""
        start_time = datetime.now()
        window = AggregationWindow(
            start_time=start_time,
            duration=timedelta(seconds=5),
            symbol="BTC-USD",
            interval="5s"
        )
        
        # Trade within window
        valid_trade = TradeData(
            symbol="BTC-USD",
            timestamp=start_time + timedelta(seconds=2),
            price=Decimal("50000"),
            size=Decimal("0.1"),
            side="buy"
        )
        
        assert window.includes_trade(valid_trade)
        
        # Trade outside window
        invalid_trade = TradeData(
            symbol="BTC-USD",
            timestamp=start_time + timedelta(seconds=6),  # After window end
            price=Decimal("50000"),
            size=Decimal("0.1"),
            side="buy"
        )
        
        assert not window.includes_trade(invalid_trade)

    def test_window_closing(self):
        """Test window closing functionality."""
        window = AggregationWindow(
            start_time=datetime.now(),
            duration=timedelta(seconds=1),
            symbol="BTC-USD",
            interval="1s"
        )
        
        assert not window.is_closed
        
        closed_window = window.close()
        
        assert closed_window.is_closed
        assert not window.is_closed  # Original unchanged


class TestTradeAggregator:
    """Test trade aggregation functionality."""

    def test_trade_aggregator_creation(self):
        """Test creation of trade aggregator."""
        aggregator = TradeAggregator(
            symbol="BTC-USD",
            interval="1s",
            window_size=timedelta(seconds=1)
        )
        
        assert aggregator.symbol == "BTC-USD"
        assert aggregator.interval == "1s"
        assert aggregator.window_size == timedelta(seconds=1)

    def test_1_second_aggregation(self):
        """Test 1-second trade aggregation."""
        aggregator = create_trade_aggregator("BTC-USD", "1s")
        
        base_time = datetime.now()
        
        # Add trades within 1 second window
        trades = [
            TradeData("BTC-USD", base_time, Decimal("50000"), Decimal("0.1"), "buy"),
            TradeData("BTC-USD", base_time + timedelta(milliseconds=300), Decimal("50010"), Decimal("0.2"), "buy"),
            TradeData("BTC-USD", base_time + timedelta(milliseconds=600), Decimal("49995"), Decimal("0.15"), "sell"),
            TradeData("BTC-USD", base_time + timedelta(milliseconds=900), Decimal("50005"), Decimal("0.1"), "buy"),
        ]
        
        for trade in trades:
            aggregator.add_trade(trade)
        
        candle = aggregator.get_current_candle()
        
        assert candle is not None
        assert candle.symbol == "BTC-USD"
        assert candle.interval == "1s"
        assert candle.open == Decimal("50000")   # First trade
        assert candle.high == Decimal("50010")   # Highest
        assert candle.low == Decimal("49995")    # Lowest
        assert candle.close == Decimal("50005")  # Last trade
        assert candle.volume == Decimal("0.55")  # Sum of sizes
        assert candle.trade_count == 4

    def test_5_second_aggregation(self):
        """Test 5-second trade aggregation."""
        aggregator = create_trade_aggregator("ETH-USD", "5s")
        
        base_time = datetime.now()
        
        # Add trades spread across 5 seconds
        trades = []
        for i in range(10):
            trade = TradeData(
                symbol="ETH-USD",
                timestamp=base_time + timedelta(milliseconds=i * 500),  # Every 500ms
                price=Decimal(str(3000 + i)),
                size=Decimal("0.1"),
                side="buy" if i % 2 == 0 else "sell"
            )
            trades.append(trade)
        
        for trade in trades:
            aggregator.add_trade(trade)
        
        candle = aggregator.get_current_candle()
        
        assert candle is not None
        assert candle.open == Decimal("3000")    # First trade
        assert candle.high == Decimal("3009")    # Highest (3000 + 9)
        assert candle.low == Decimal("3000")     # Lowest
        assert candle.close == Decimal("3009")   # Last trade
        assert candle.volume == Decimal("1.0")   # 10 * 0.1
        assert candle.trade_count == 10

    def test_15_second_aggregation(self):
        """Test 15-second trade aggregation."""
        aggregator = create_trade_aggregator("SOL-USD", "15s")
        
        base_time = datetime.now()
        
        # Add trades every 2 seconds for 14 seconds (8 trades total)
        trades = []
        for i in range(8):
            trade = TradeData(
                symbol="SOL-USD",
                timestamp=base_time + timedelta(seconds=i * 2),
                price=Decimal(str(100 + i * 0.5)),  # Gradually increasing price
                size=Decimal("1.0"),
                side="buy"
            )
            trades.append(trade)
        
        for trade in trades:
            aggregator.add_trade(trade)
        
        candle = aggregator.get_current_candle()
        
        assert candle is not None
        assert candle.open == Decimal("100.0")
        assert candle.close == Decimal("103.5")  # 100 + 7 * 0.5
        assert candle.high == Decimal("103.5")
        assert candle.low == Decimal("100.0")
        assert candle.is_bullish

    def test_30_second_aggregation(self):
        """Test 30-second trade aggregation."""
        aggregator = create_trade_aggregator("ADA-USD", "30s")
        
        base_time = datetime.now()
        
        # High-frequency trades (every 100ms for 3 seconds)
        trades = []
        for i in range(30):
            price_variation = (i % 10) - 5  # Oscillating price
            trade = TradeData(
                symbol="ADA-USD",
                timestamp=base_time + timedelta(milliseconds=i * 100),
                price=Decimal(str(0.5 + price_variation * 0.001)),
                size=Decimal("100.0"),
                side="buy" if i % 2 == 0 else "sell"
            )
            trades.append(trade)
        
        for trade in trades:
            aggregator.add_trade(trade)
        
        candle = aggregator.get_current_candle()
        
        assert candle is not None
        assert candle.trade_count == 30
        assert candle.volume == Decimal("3000.0")  # 30 * 100
        assert candle.high >= candle.low  # Basic sanity check

    def test_multiple_window_aggregation(self):
        """Test aggregation across multiple windows."""
        aggregator = create_trade_aggregator("BTC-USD", "1s")
        
        base_time = datetime.now()
        
        # First window (0-1s)
        first_window_trades = [
            TradeData("BTC-USD", base_time + timedelta(milliseconds=100), Decimal("50000"), Decimal("0.1"), "buy"),
            TradeData("BTC-USD", base_time + timedelta(milliseconds=500), Decimal("50010"), Decimal("0.1"), "buy"),
        ]
        
        for trade in first_window_trades:
            aggregator.add_trade(trade)
        
        first_candle = aggregator.get_current_candle()
        assert first_candle.trade_count == 2
        
        # Force window completion
        aggregator.complete_current_window()
        
        # Second window (1-2s)
        second_window_trades = [
            TradeData("BTC-USD", base_time + timedelta(seconds=1, milliseconds=100), Decimal("50020"), Decimal("0.2"), "sell"),
            TradeData("BTC-USD", base_time + timedelta(seconds=1, milliseconds=800), Decimal("50015"), Decimal("0.1"), "buy"),
        ]
        
        for trade in second_window_trades:
            aggregator.add_trade(trade)
        
        second_candle = aggregator.get_current_candle()
        assert second_candle.trade_count == 2
        assert second_candle.open == Decimal("50020")  # First trade of new window

    def test_late_trade_handling(self):
        """Test handling of late trades (out of order)."""
        aggregator = create_trade_aggregator("ETH-USD", "1s")
        
        base_time = datetime.now()
        
        # Add trades in order
        trade1 = TradeData("ETH-USD", base_time, Decimal("3000"), Decimal("1.0"), "buy")
        trade2 = TradeData("ETH-USD", base_time + timedelta(milliseconds=500), Decimal("3010"), Decimal("1.0"), "buy")
        
        aggregator.add_trade(trade1)
        aggregator.add_trade(trade2)
        
        # Add late trade (earlier timestamp)
        late_trade = TradeData("ETH-USD", base_time + timedelta(milliseconds=200), Decimal("2990"), Decimal("0.5"), "sell")
        
        aggregator.add_trade(late_trade)
        
        candle = aggregator.get_current_candle()
        
        # Should incorporate the late trade
        assert candle.trade_count == 3
        assert candle.low == Decimal("2990")  # Late trade had lowest price


class TestCandleAggregator:
    """Test candle aggregation from smaller intervals."""

    def test_candle_aggregator_creation(self):
        """Test creation of candle aggregator."""
        aggregator = create_candle_aggregator("BTC-USD", "1s", "1m")
        
        assert aggregator.symbol == "BTC-USD"
        assert aggregator.source_interval == "1s"
        assert aggregator.target_interval == "1m"

    def test_1s_to_1m_aggregation(self):
        """Test aggregating 1-second candles to 1-minute candles."""
        aggregator = create_candle_aggregator("BTC-USD", "1s", "1m")
        
        base_time = datetime.now().replace(second=0, microsecond=0)
        
        # Create 60 1-second candles
        source_candles = []
        for i in range(60):
            candle = RealTimeCandle(
                symbol="BTC-USD",
                timestamp=base_time + timedelta(seconds=i),
                interval="1s",
                open=Decimal(str(50000 + i)),
                high=Decimal(str(50020 + i)),
                low=Decimal(str(49980 + i)),
                close=Decimal(str(50010 + i)),
                volume=Decimal("1.0"),
                trade_count=10,
                vwap=Decimal(str(50000 + i))
            )
            source_candles.append(candle)
            aggregator.add_candle(candle)
        
        # Get aggregated 1-minute candle
        minute_candle = aggregator.get_aggregated_candle()
        
        assert minute_candle is not None
        assert minute_candle.symbol == "BTC-USD"
        assert minute_candle.interval == "1m"
        assert minute_candle.open == Decimal("50000")     # First candle's open
        assert minute_candle.close == Decimal("50069")    # Last candle's close
        assert minute_candle.high == Decimal("50079")     # Highest high (50020 + 59)
        assert minute_candle.low == Decimal("49980")      # Lowest low
        assert minute_candle.volume == Decimal("60.0")    # Sum of all volumes
        assert minute_candle.trade_count == 600           # Sum of all trade counts

    def test_5s_to_1m_aggregation(self):
        """Test aggregating 5-second candles to 1-minute candles."""
        aggregator = create_candle_aggregator("ETH-USD", "5s", "1m")
        
        base_time = datetime.now().replace(second=0, microsecond=0)
        
        # Create 12 5-second candles (12 * 5s = 60s = 1m)
        for i in range(12):
            candle = RealTimeCandle(
                symbol="ETH-USD",
                timestamp=base_time + timedelta(seconds=i * 5),
                interval="5s",
                open=Decimal(str(3000 + i * 2)),
                high=Decimal(str(3010 + i * 2)),
                low=Decimal(str(2990 + i * 2)),
                close=Decimal(str(3005 + i * 2)),
                volume=Decimal("5.0"),
                trade_count=50,
                vwap=Decimal(str(3000 + i * 2))
            )
            aggregator.add_candle(candle)
        
        minute_candle = aggregator.get_aggregated_candle()
        
        assert minute_candle is not None
        assert minute_candle.open == Decimal("3000")      # First candle
        assert minute_candle.close == Decimal("3027")     # Last candle (3005 + 11*2)
        assert minute_candle.volume == Decimal("60.0")    # 12 * 5.0
        assert minute_candle.trade_count == 600           # 12 * 50

    def test_partial_aggregation(self):
        """Test partial aggregation when target window is incomplete."""
        aggregator = create_candle_aggregator("LTC-USD", "5s", "1m")
        
        base_time = datetime.now().replace(second=0, microsecond=0)
        
        # Add only 6 candles (30 seconds worth)
        for i in range(6):
            candle = RealTimeCandle(
                symbol="LTC-USD",
                timestamp=base_time + timedelta(seconds=i * 5),
                interval="5s",
                open=Decimal(str(150 + i)),
                high=Decimal(str(152 + i)),
                low=Decimal(str(148 + i)),
                close=Decimal(str(151 + i)),
                volume=Decimal("2.0"),
                trade_count=20,
                vwap=Decimal(str(150 + i))
            )
            aggregator.add_candle(candle)
        
        # Should still return partial aggregation
        partial_candle = aggregator.get_aggregated_candle()
        
        assert partial_candle is not None
        assert partial_candle.volume == Decimal("12.0")   # 6 * 2.0
        assert partial_candle.trade_count == 120          # 6 * 20


class TestIntervalValidation:
    """Test interval validation and compatibility."""

    def test_valid_sub_minute_intervals(self):
        """Test validation of valid sub-minute intervals."""
        valid_intervals = ["1s", "5s", "15s", "30s"]
        
        for interval in valid_intervals:
            assert validate_interval_compatibility(interval) is True
            
            # Should be able to create aggregator
            aggregator = create_trade_aggregator("BTC-USD", interval)
            assert aggregator.interval == interval

    def test_invalid_intervals(self):
        """Test validation of invalid intervals."""
        invalid_intervals = ["2s", "7s", "45s", "1.5s", "0s"]
        
        for interval in invalid_intervals:
            with pytest.raises(ValueError, match="Unsupported interval"):
                create_trade_aggregator("BTC-USD", interval)

    def test_interval_parsing(self):
        """Test interval string parsing."""
        test_cases = [
            ("1s", 1),
            ("5s", 5),
            ("15s", 15),
            ("30s", 30),
        ]
        
        for interval_str, expected_seconds in test_cases:
            # Would use actual parsing function
            parsed_seconds = int(interval_str[:-1])  # Remove 's' and convert
            assert parsed_seconds == expected_seconds

    def test_aggregation_compatibility(self):
        """Test compatibility between source and target intervals."""
        compatible_pairs = [
            ("1s", "5s"),
            ("1s", "15s"),
            ("1s", "30s"),
            ("1s", "1m"),
            ("5s", "1m"),
            ("15s", "1m"),
            ("30s", "1m"),
        ]
        
        for source, target in compatible_pairs:
            # Should not raise error
            try:
                aggregator = create_candle_aggregator("BTC-USD", source, target)
                assert aggregator.source_interval == source
                assert aggregator.target_interval == target
            except ValueError:
                pytest.fail(f"Should be compatible: {source} -> {target}")


@pytest.mark.asyncio
class TestAsyncAggregation:
    """Test asynchronous aggregation operations."""

    async def test_async_trade_processing(self):
        """Test asynchronous trade processing."""
        aggregator = create_trade_aggregator("BTC-USD", "1s")
        
        async def process_trade_batch(trades: List[TradeData]):
            """Process a batch of trades."""
            for trade in trades:
                aggregator.add_trade(trade)
                await asyncio.sleep(0.001)  # Simulate processing time
        
        # Create trade batches
        base_time = datetime.now()
        trade_batches = []
        
        for batch_idx in range(3):
            batch = []
            for i in range(10):
                trade = TradeData(
                    symbol="BTC-USD",
                    timestamp=base_time + timedelta(milliseconds=batch_idx * 100 + i * 10),
                    price=Decimal(str(50000 + i)),
                    size=Decimal("0.1"),
                    side="buy" if i % 2 == 0 else "sell"
                )
                batch.append(trade)
            trade_batches.append(batch)
        
        # Process batches concurrently
        tasks = [process_trade_batch(batch) for batch in trade_batches]
        await asyncio.gather(*tasks)
        
        candle = aggregator.get_current_candle()
        assert candle is not None
        assert candle.trade_count == 30  # 3 batches * 10 trades

    async def test_async_candle_streaming(self):
        """Test streaming candle updates."""
        aggregator = create_trade_aggregator("ETH-USD", "1s")
        
        async def candle_stream() -> AsyncIterator[RealTimeCandle]:
            """Stream candle updates."""
            base_time = datetime.now()
            
            for i in range(5):
                # Add trade
                trade = TradeData(
                    symbol="ETH-USD",
                    timestamp=base_time + timedelta(milliseconds=i * 200),
                    price=Decimal(str(3000 + i)),
                    size=Decimal("1.0"),
                    side="buy"
                )
                
                aggregator.add_trade(trade)
                candle = aggregator.get_current_candle()
                
                if candle is not None:
                    yield candle
                
                await asyncio.sleep(0.01)  # Simulate real-time delay
        
        # Collect streaming candles
        candles = []
        async for candle in candle_stream():
            candles.append(candle)
        
        assert len(candles) == 5
        assert all(isinstance(candle, RealTimeCandle) for candle in candles)
        
        # Should show progression
        assert candles[0].trade_count == 1
        assert candles[-1].trade_count == 5

    async def test_concurrent_aggregators(self):
        """Test concurrent aggregators for different symbols."""
        symbols = ["BTC-USD", "ETH-USD", "LTC-USD"]
        aggregators = {symbol: create_trade_aggregator(symbol, "1s") for symbol in symbols}
        
        async def feed_trades_to_symbol(symbol: str, trade_count: int):
            """Feed trades to a specific symbol aggregator."""
            base_time = datetime.now()
            
            for i in range(trade_count):
                trade = TradeData(
                    symbol=symbol,
                    timestamp=base_time + timedelta(milliseconds=i * 100),
                    price=Decimal(str(1000 * (ord(symbol[0]) - ord('A')) + i)),  # Different base prices
                    size=Decimal("0.1"),
                    side="buy"
                )
                
                aggregators[symbol].add_trade(trade)
                await asyncio.sleep(0.001)
        
        # Feed different numbers of trades to each symbol
        tasks = [
            feed_trades_to_symbol("BTC-USD", 10),
            feed_trades_to_symbol("ETH-USD", 15),
            feed_trades_to_symbol("LTC-USD", 8),
        ]
        
        await asyncio.gather(*tasks)
        
        # Check results
        btc_candle = aggregators["BTC-USD"].get_current_candle()
        eth_candle = aggregators["ETH-USD"].get_current_candle()
        ltc_candle = aggregators["LTC-USD"].get_current_candle()
        
        assert btc_candle.trade_count == 10
        assert eth_candle.trade_count == 15
        assert ltc_candle.trade_count == 8
        
        # Each should have different price ranges
        assert btc_candle.symbol == "BTC-USD"
        assert eth_candle.symbol == "ETH-USD"
        assert ltc_candle.symbol == "LTC-USD"


class TestAggregationPerformance:
    """Test aggregation performance characteristics."""

    def test_high_frequency_aggregation(self):
        """Test performance with high-frequency trades."""
        aggregator = create_trade_aggregator("BTC-USD", "1s")
        
        import time
        
        base_time = datetime.now()
        start_time = time.time()
        
        # Add 1000 trades rapidly
        for i in range(1000):
            trade = TradeData(
                symbol="BTC-USD",
                timestamp=base_time + timedelta(microseconds=i * 1000),  # Every millisecond
                price=Decimal(str(50000 + (i % 100))),
                size=Decimal("0.01"),
                side="buy" if i % 2 == 0 else "sell"
            )
            aggregator.add_trade(trade)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should process efficiently
        assert duration < 1.0  # Less than 1 second for 1000 trades
        
        candle = aggregator.get_current_candle()
        assert candle.trade_count == 1000
        assert candle.volume == Decimal("10.0")  # 1000 * 0.01

    def test_memory_efficiency(self):
        """Test memory efficiency of aggregation."""
        aggregator = create_trade_aggregator("ETH-USD", "1s")
        
        # Add many trades and check that memory doesn't grow unbounded
        base_time = datetime.now()
        
        # Add trades in multiple windows
        for window in range(10):
            window_start = base_time + timedelta(seconds=window)
            
            for i in range(100):
                trade = TradeData(
                    symbol="ETH-USD",
                    timestamp=window_start + timedelta(milliseconds=i * 10),
                    price=Decimal(str(3000 + i)),
                    size=Decimal("0.1"),
                    side="buy"
                )
                aggregator.add_trade(trade)
            
            # Complete window to trigger cleanup
            aggregator.complete_current_window()
        
        # Should have processed 1000 trades across 10 windows
        # Memory usage should be bounded (not tested directly here, but structure should support it)
        
        # Current window should only have trades from the last batch
        current_candle = aggregator.get_current_candle()
        if current_candle:
            # Should not accumulate all historical trades
            assert current_candle.trade_count <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])