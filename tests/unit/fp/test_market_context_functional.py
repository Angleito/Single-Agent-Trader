"""
Functional programming tests for market context analysis with immutable data structures.

This module tests:
1. Market context analysis with immutable data types
2. Pure functional operations on market data
3. Correlation analysis using functional patterns
4. Risk sentiment assessment with FP types
5. Momentum alignment calculations
6. Market regime detection using immutable structures
7. Error handling with Result/Either patterns
8. Performance with large immutable datasets
"""

import pytest
from datetime import datetime, UTC, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Callable
from unittest.mock import MagicMock, patch
import numpy as np

from bot.fp.types.market import (
    Candle,
    Trade,
    Ticker,
    OrderBook,
    MarketSnapshot,
    ConnectionState,
    ConnectionStatus,
    DataQuality,
    WebSocketMessage,
    AggregatedData,
    RealtimeUpdate,
)
from bot.fp.types.result import Result, Success, Failure
from bot.fp.types.base import Symbol, Money, Percentage


# Functional Market Context Analysis Types (these would be in actual FP analysis module)

@pytest.fixture
def immutable_btc_candles():
    """Create immutable BTC candle data for testing."""
    base_time = datetime.now(UTC)
    candles = []
    
    # Generate 100 candles with realistic price movements
    base_price = Decimal("50000")
    
    for i in range(100):
        # Simulate price movement
        price_change = Decimal(str(np.random.normal(0, 0.02)))  # 2% volatility
        new_price = base_price * (1 + price_change)
        
        # Generate OHLC with realistic relationships
        high = new_price * Decimal("1.005")  # 0.5% higher
        low = new_price * Decimal("0.995")   # 0.5% lower
        
        candle = Candle(
            timestamp=base_time + timedelta(minutes=i),
            open=base_price,
            high=high,
            low=low,
            close=new_price,
            volume=Decimal(str(np.random.uniform(50, 200))),
            symbol="BTC-USD",
            interval="1m",
            trades_count=int(np.random.uniform(100, 500)),
        )
        
        candles.append(candle)
        base_price = new_price
    
    return candles


@pytest.fixture
def immutable_nasdaq_candles():
    """Create immutable NASDAQ candle data for testing."""
    base_time = datetime.now(UTC)
    candles = []
    
    base_price = Decimal("15000")
    
    for i in range(100):
        # Simulate correlated movement with BTC
        price_change = Decimal(str(np.random.normal(0, 0.01)))  # 1% volatility
        new_price = base_price * (1 + price_change)
        
        high = new_price * Decimal("1.003")
        low = new_price * Decimal("0.997")
        
        candle = Candle(
            timestamp=base_time + timedelta(minutes=i),
            open=base_price,
            high=high,
            low=low,
            close=new_price,
            volume=Decimal(str(np.random.uniform(1000, 5000))),
            symbol="NASDAQ",
            interval="1m",
            trades_count=int(np.random.uniform(500, 2000)),
        )
        
        candles.append(candle)
        base_price = new_price
    
    return candles


@pytest.fixture
def immutable_trade_data():
    """Create immutable trade data for testing."""
    base_time = datetime.now(UTC)
    trades = []
    
    base_price = Decimal("50000")
    
    for i in range(200):
        # Alternate between buy and sell
        side = "BUY" if i % 2 == 0 else "SELL"
        
        # Small price variation
        price_variation = Decimal(str(np.random.uniform(-0.001, 0.001)))
        trade_price = base_price * (1 + price_variation)
        
        trade = Trade(
            id=f"trade_{i:06d}",
            timestamp=base_time + timedelta(seconds=i * 30),
            price=trade_price,
            size=Decimal(str(np.random.uniform(0.1, 2.0))),
            side=side,
            symbol="BTC-USD",
            exchange="test_exchange",
        )
        
        trades.append(trade)
    
    return trades


@pytest.fixture
def immutable_order_book():
    """Create immutable order book for testing."""
    base_price = Decimal("50000")
    
    # Generate bids (descending prices)
    bids = []
    for i in range(10):
        price = base_price - Decimal(str(i + 1))
        size = Decimal(str(np.random.uniform(0.5, 5.0)))
        bids.append((price, size))
    
    # Generate asks (ascending prices)
    asks = []
    for i in range(10):
        price = base_price + Decimal(str(i + 1))
        size = Decimal(str(np.random.uniform(0.5, 5.0)))
        asks.append((price, size))
    
    return OrderBook(
        bids=bids,
        asks=asks,
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def immutable_market_snapshots():
    """Create immutable market snapshots for testing."""
    base_time = datetime.now(UTC)
    snapshots = []
    
    base_price = Decimal("50000")
    
    for i in range(50):
        price_change = Decimal(str(np.random.normal(0, 0.01)))
        current_price = base_price * (1 + price_change)
        
        snapshot = MarketSnapshot(
            timestamp=base_time + timedelta(minutes=i * 5),
            symbol="BTC-USD",
            price=current_price,
            volume=Decimal(str(np.random.uniform(100, 1000))),
            bid=current_price - Decimal("10"),
            ask=current_price + Decimal("10"),
        )
        
        snapshots.append(snapshot)
        base_price = current_price
    
    return snapshots


class TestImmutableMarketDataTypes:
    """Test immutable market data type functionality."""
    
    def test_candle_immutability(self):
        """Test that candles are immutable."""
        candle = Candle(
            timestamp=datetime.now(UTC),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("100"),
            symbol="BTC-USD",
        )
        
        # Should not be able to modify fields
        with pytest.raises(AttributeError):
            candle.close = Decimal("52000")
        
        with pytest.raises(AttributeError):
            candle.volume = Decimal("200")
    
    def test_candle_validation(self):
        """Test candle data validation."""
        # Valid candle
        candle = Candle(
            timestamp=datetime.now(UTC),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("100"),
        )
        
        assert candle.is_bullish is True
        assert candle.price_range == Decimal("2000")
        assert candle.body_size == Decimal("500")
        
        # Test invalid high price
        with pytest.raises(ValueError, match="High .* must be >= all other prices"):
            Candle(
                timestamp=datetime.now(UTC),
                open=Decimal("50000"),
                high=Decimal("49000"),  # Invalid: lower than open
                low=Decimal("48000"),
                close=Decimal("50500"),
                volume=Decimal("100"),
            )
        
        # Test invalid low price
        with pytest.raises(ValueError, match="Low .* must be <= all other prices"):
            Candle(
                timestamp=datetime.now(UTC),
                open=Decimal("50000"),
                high=Decimal("51000"),
                low=Decimal("50500"),  # Invalid: higher than open
                close=Decimal("50200"),
                volume=Decimal("100"),
            )
    
    def test_trade_immutability_and_validation(self):
        """Test trade immutability and validation."""
        trade = Trade(
            id="trade_001",
            timestamp=datetime.now(UTC),
            price=Decimal("50000"),
            size=Decimal("0.5"),
            side="BUY",
            symbol="BTC-USD",
        )
        
        # Test immutability
        with pytest.raises(AttributeError):
            trade.price = Decimal("51000")
        
        # Test properties
        assert trade.value == Decimal("25000")  # 50000 * 0.5
        assert trade.is_buy() is True
        assert trade.is_sell() is False
        
        # Test validation
        with pytest.raises(ValueError, match="Price must be positive"):
            Trade(
                id="invalid_trade",
                timestamp=datetime.now(UTC),
                price=Decimal("-1000"),
                size=Decimal("0.5"),
                side="BUY",
            )
        
        with pytest.raises(ValueError, match="Side must be BUY or SELL"):
            Trade(
                id="invalid_side",
                timestamp=datetime.now(UTC),
                price=Decimal("50000"),
                size=Decimal("0.5"),
                side="INVALID",
            )
    
    def test_order_book_immutability_and_properties(self, immutable_order_book):
        """Test order book immutability and properties."""
        # Test immutability
        with pytest.raises(AttributeError):
            immutable_order_book.bids = []
        
        # Test properties
        assert immutable_order_book.best_bid is not None
        assert immutable_order_book.best_ask is not None
        assert immutable_order_book.mid_price is not None
        assert immutable_order_book.spread is not None
        assert immutable_order_book.spread > 0
        
        # Test depth calculations
        assert immutable_order_book.bid_depth > 0
        assert immutable_order_book.ask_depth > 0
        
        # Test spread in basis points
        spread_bps = immutable_order_book.get_spread_bps()
        assert spread_bps > 0
    
    def test_market_snapshot_immutability(self):
        """Test market snapshot immutability and derived properties."""
        snapshot = MarketSnapshot(
            timestamp=datetime.now(UTC),
            symbol="ETH-USD",
            price=Decimal("3000"),
            volume=Decimal("500"),
            bid=Decimal("2995"),
            ask=Decimal("3005"),
        )
        
        # Test immutability
        with pytest.raises(AttributeError):
            snapshot.price = Decimal("3100")
        
        # Test derived properties
        assert snapshot.spread == Decimal("10")  # 3005 - 2995
        
        # Test validation
        with pytest.raises(ValueError, match="Bid .* cannot be greater than ask"):
            MarketSnapshot(
                timestamp=datetime.now(UTC),
                symbol="ETH-USD",
                price=Decimal("3000"),
                volume=Decimal("500"),
                bid=Decimal("3010"),  # Invalid: higher than ask
                ask=Decimal("3005"),
            )


class TestFunctionalMarketAnalysis:
    """Test functional market analysis operations."""
    
    def test_pure_price_series_extraction(self, immutable_btc_candles):
        """Test pure function for extracting price series from candles."""
        def extract_close_prices(candles: List[Candle]) -> List[Decimal]:
            """Pure function to extract close prices."""
            return [candle.close for candle in candles]
        
        def extract_volume_series(candles: List[Candle]) -> List[Decimal]:
            """Pure function to extract volume series."""
            return [candle.volume for candle in candles]
        
        close_prices = extract_close_prices(immutable_btc_candles)
        volumes = extract_volume_series(immutable_btc_candles)
        
        assert len(close_prices) == len(immutable_btc_candles)
        assert len(volumes) == len(immutable_btc_candles)
        assert all(isinstance(price, Decimal) for price in close_prices)
        assert all(isinstance(volume, Decimal) for volume in volumes)
    
    def test_pure_candle_pattern_detection(self, immutable_btc_candles):
        """Test pure functions for candle pattern detection."""
        def find_bullish_candles(candles: List[Candle]) -> List[Candle]:
            """Pure function to find bullish candles."""
            return [candle for candle in candles if candle.is_bullish]
        
        def find_doji_candles(candles: List[Candle], threshold: float = 0.1) -> List[Candle]:
            """Pure function to find doji candles."""
            return [candle for candle in candles if candle.is_doji(threshold)]
        
        def calculate_average_volume(candles: List[Candle]) -> Decimal:
            """Pure function to calculate average volume."""
            if not candles:
                return Decimal("0")
            total = sum(candle.volume for candle in candles)
            return total / Decimal(len(candles))
        
        bullish = find_bullish_candles(immutable_btc_candles)
        doji = find_doji_candles(immutable_btc_candles)
        avg_volume = calculate_average_volume(immutable_btc_candles)
        
        assert len(bullish) <= len(immutable_btc_candles)
        assert len(doji) <= len(immutable_btc_candles)
        assert avg_volume > Decimal("0")
        assert all(candle.is_bullish for candle in bullish)
    
    def test_pure_correlation_calculation(self, immutable_btc_candles, immutable_nasdaq_candles):
        """Test pure correlation calculation between price series."""
        def extract_returns(candles: List[Candle]) -> List[float]:
            """Pure function to calculate returns from candles."""
            if len(candles) < 2:
                return []
            
            returns = []
            for i in range(1, len(candles)):
                prev_price = float(candles[i-1].close)
                curr_price = float(candles[i].close)
                if prev_price > 0:
                    return_val = (curr_price - prev_price) / prev_price
                    returns.append(return_val)
            return returns
        
        def calculate_correlation(series1: List[float], series2: List[float]) -> float:
            """Pure function to calculate correlation between two series."""
            if len(series1) != len(series2) or len(series1) < 2:
                return 0.0
            
            # Use numpy for correlation calculation
            correlation_matrix = np.corrcoef(series1, series2)
            return float(correlation_matrix[0, 1]) if not np.isnan(correlation_matrix[0, 1]) else 0.0
        
        btc_returns = extract_returns(immutable_btc_candles)
        nasdaq_returns = extract_returns(immutable_nasdaq_candles)
        
        # Align series length
        min_length = min(len(btc_returns), len(nasdaq_returns))
        btc_aligned = btc_returns[:min_length]
        nasdaq_aligned = nasdaq_returns[:min_length]
        
        correlation = calculate_correlation(btc_aligned, nasdaq_aligned)
        
        assert -1.0 <= correlation <= 1.0
        assert isinstance(correlation, float)
    
    def test_pure_volatility_calculation(self, immutable_btc_candles):
        """Test pure volatility calculation functions."""
        def calculate_realized_volatility(candles: List[Candle], periods: int = 20) -> Decimal:
            """Pure function to calculate realized volatility."""
            if len(candles) < periods:
                return Decimal("0")
            
            # Calculate returns
            returns = []
            for i in range(1, len(candles)):
                prev_price = candles[i-1].close
                curr_price = candles[i].close
                if prev_price > 0:
                    return_val = (curr_price - prev_price) / prev_price
                    returns.append(float(return_val))
            
            if len(returns) < periods:
                return Decimal("0")
            
            # Take last 'periods' returns
            recent_returns = returns[-periods:]
            
            # Calculate standard deviation
            mean_return = sum(recent_returns) / len(recent_returns)
            variance = sum((r - mean_return) ** 2 for r in recent_returns) / len(recent_returns)
            volatility = variance ** 0.5
            
            # Annualize (assuming minute data)
            annualized_vol = volatility * (365.25 * 24 * 60) ** 0.5
            
            return Decimal(str(annualized_vol))
        
        def calculate_price_range_volatility(candles: List[Candle]) -> Decimal:
            """Pure function to calculate volatility based on price ranges."""
            if not candles:
                return Decimal("0")
            
            ranges = [candle.price_range for candle in candles]
            prices = [candle.close for candle in candles]
            
            # Calculate average relative range
            relative_ranges = []
            for price_range, close_price in zip(ranges, prices):
                if close_price > 0:
                    relative_ranges.append(float(price_range / close_price))
            
            if not relative_ranges:
                return Decimal("0")
            
            avg_relative_range = sum(relative_ranges) / len(relative_ranges)
            return Decimal(str(avg_relative_range))
        
        realized_vol = calculate_realized_volatility(immutable_btc_candles)
        range_vol = calculate_price_range_volatility(immutable_btc_candles)
        
        assert realized_vol >= Decimal("0")
        assert range_vol >= Decimal("0")
    
    def test_pure_trend_analysis(self, immutable_btc_candles):
        """Test pure trend analysis functions."""
        def calculate_simple_moving_average(candles: List[Candle], periods: int) -> List[Decimal]:
            """Pure function to calculate SMA."""
            if len(candles) < periods:
                return []
            
            sma_values = []
            for i in range(periods - 1, len(candles)):
                window = candles[i - periods + 1:i + 1]
                avg = sum(candle.close for candle in window) / Decimal(periods)
                sma_values.append(avg)
            
            return sma_values
        
        def detect_trend_direction(candles: List[Candle], periods: int = 20) -> str:
            """Pure function to detect trend direction."""
            if len(candles) < periods + 1:
                return "INSUFFICIENT_DATA"
            
            sma = calculate_simple_moving_average(candles, periods)
            if len(sma) < 2:
                return "NEUTRAL"
            
            current_sma = sma[-1]
            previous_sma = sma[-2]
            
            if current_sma > previous_sma:
                return "UPTREND"
            elif current_sma < previous_sma:
                return "DOWNTREND"
            else:
                return "NEUTRAL"
        
        def calculate_trend_strength(candles: List[Candle], periods: int = 20) -> Decimal:
            """Pure function to calculate trend strength."""
            if len(candles) < periods:
                return Decimal("0")
            
            prices = [candle.close for candle in candles[-periods:]]
            
            # Calculate R-squared of linear regression
            n = len(prices)
            x_values = list(range(n))
            
            # Calculate means
            mean_x = sum(x_values) / n
            mean_y = sum(float(p) for p in prices) / n
            
            # Calculate slope and correlation
            numerator = sum((x - mean_x) * (float(prices[i]) - mean_y) for i, x in enumerate(x_values))
            denominator_x = sum((x - mean_x) ** 2 for x in x_values)
            denominator_y = sum((float(prices[i]) - mean_y) ** 2 for i in range(n))
            
            if denominator_x == 0 or denominator_y == 0:
                return Decimal("0")
            
            r_squared = (numerator ** 2) / (denominator_x * denominator_y)
            return Decimal(str(max(0, min(1, r_squared))))
        
        sma_20 = calculate_simple_moving_average(immutable_btc_candles, 20)
        trend_direction = detect_trend_direction(immutable_btc_candles)
        trend_strength = calculate_trend_strength(immutable_btc_candles)
        
        assert len(sma_20) == len(immutable_btc_candles) - 19
        assert trend_direction in ["UPTREND", "DOWNTREND", "NEUTRAL", "INSUFFICIENT_DATA"]
        assert Decimal("0") <= trend_strength <= Decimal("1")


class TestFunctionalOrderBookAnalysis:
    """Test functional order book analysis operations."""
    
    def test_pure_order_book_calculations(self, immutable_order_book):
        """Test pure order book calculation functions."""
        def calculate_order_book_imbalance(order_book: OrderBook) -> Decimal:
            """Pure function to calculate order book imbalance."""
            bid_depth = order_book.bid_depth
            ask_depth = order_book.ask_depth
            total_depth = bid_depth + ask_depth
            
            if total_depth == 0:
                return Decimal("0")
            
            return (bid_depth - ask_depth) / total_depth
        
        def calculate_weighted_mid_price(order_book: OrderBook) -> Decimal:
            """Pure function to calculate volume-weighted mid price."""
            if not order_book.best_bid or not order_book.best_ask:
                return Decimal("0")
            
            bid_price, bid_size = order_book.best_bid
            ask_price, ask_size = order_book.best_ask
            total_size = bid_size + ask_size
            
            if total_size == 0:
                return (bid_price + ask_price) / 2
            
            weighted_price = (bid_price * ask_size + ask_price * bid_size) / total_size
            return weighted_price
        
        def calculate_order_book_depth_at_levels(
            order_book: OrderBook, 
            levels: int = 5
        ) -> tuple[Decimal, Decimal]:
            """Pure function to calculate depth at specified levels."""
            bid_depth = sum(size for _, size in order_book.bids[:levels])
            ask_depth = sum(size for _, size in order_book.asks[:levels])
            return bid_depth, ask_depth
        
        imbalance = calculate_order_book_imbalance(immutable_order_book)
        weighted_mid = calculate_weighted_mid_price(immutable_order_book)
        bid_depth_5, ask_depth_5 = calculate_order_book_depth_at_levels(immutable_order_book, 5)
        
        assert -1 <= imbalance <= 1
        assert weighted_mid > 0
        assert bid_depth_5 > 0
        assert ask_depth_5 > 0
    
    def test_pure_liquidity_analysis(self, immutable_order_book):
        """Test pure liquidity analysis functions."""
        def calculate_effective_spread(order_book: OrderBook, trade_size: Decimal) -> Decimal:
            """Pure function to calculate effective spread for a trade size."""
            buy_price = order_book.price_impact("buy", trade_size)
            sell_price = order_book.price_impact("sell", trade_size)
            
            if buy_price is None or sell_price is None:
                return Decimal("0")
            
            mid_price = order_book.mid_price
            if mid_price is None or mid_price == 0:
                return Decimal("0")
            
            effective_spread = (buy_price - sell_price) / mid_price
            return effective_spread
        
        def analyze_market_depth_profile(
            order_book: OrderBook,
            max_distance_pct: Decimal = Decimal("0.01")  # 1%
        ) -> Dict[str, Decimal]:
            """Pure function to analyze market depth profile."""
            mid_price = order_book.mid_price
            if mid_price is None:
                return {"total_bid_depth": Decimal("0"), "total_ask_depth": Decimal("0")}
            
            max_distance = mid_price * max_distance_pct
            
            # Analyze bid side
            bid_depth_in_range = Decimal("0")
            for price, size in order_book.bids:
                if mid_price - price <= max_distance:
                    bid_depth_in_range += size
                else:
                    break
            
            # Analyze ask side
            ask_depth_in_range = Decimal("0")
            for price, size in order_book.asks:
                if price - mid_price <= max_distance:
                    ask_depth_in_range += size
                else:
                    break
            
            return {
                "total_bid_depth": bid_depth_in_range,
                "total_ask_depth": ask_depth_in_range,
                "depth_ratio": (
                    bid_depth_in_range / ask_depth_in_range 
                    if ask_depth_in_range > 0 else Decimal("0")
                ),
            }
        
        test_size = Decimal("1.0")
        effective_spread = calculate_effective_spread(immutable_order_book, test_size)
        depth_profile = analyze_market_depth_profile(immutable_order_book)
        
        assert effective_spread >= 0
        assert "total_bid_depth" in depth_profile
        assert "total_ask_depth" in depth_profile
        assert "depth_ratio" in depth_profile
        assert all(isinstance(v, Decimal) for v in depth_profile.values())


class TestFunctionalTradeAnalysis:
    """Test functional trade analysis operations."""
    
    def test_pure_trade_aggregation(self, immutable_trade_data):
        """Test pure trade aggregation functions."""
        def aggregate_trades_by_time_window(
            trades: List[Trade], 
            window_minutes: int = 5
        ) -> List[Dict[str, Any]]:
            """Pure function to aggregate trades by time windows."""
            if not trades:
                return []
            
            # Sort trades by timestamp
            sorted_trades = sorted(trades, key=lambda t: t.timestamp)
            
            aggregations = []
            current_window_start = sorted_trades[0].timestamp.replace(second=0, microsecond=0)
            window_delta = timedelta(minutes=window_minutes)
            
            current_window_trades = []
            
            for trade in sorted_trades:
                # Check if trade belongs to current window
                if trade.timestamp < current_window_start + window_delta:
                    current_window_trades.append(trade)
                else:
                    # Process current window
                    if current_window_trades:
                        agg = _aggregate_trade_window(current_window_trades, current_window_start)
                        aggregations.append(agg)
                    
                    # Start new window
                    current_window_start = trade.timestamp.replace(second=0, microsecond=0)
                    current_window_start = current_window_start.replace(
                        minute=(current_window_start.minute // window_minutes) * window_minutes
                    )
                    current_window_trades = [trade]
            
            # Process final window
            if current_window_trades:
                agg = _aggregate_trade_window(current_window_trades, current_window_start)
                aggregations.append(agg)
            
            return aggregations
        
        def _aggregate_trade_window(trades: List[Trade], window_start: datetime) -> Dict[str, Any]:
            """Helper function to aggregate trades within a window."""
            if not trades:
                return {}
            
            buy_trades = [t for t in trades if t.is_buy()]
            sell_trades = [t for t in trades if t.is_sell()]
            
            total_volume = sum(t.size for t in trades)
            buy_volume = sum(t.size for t in buy_trades)
            sell_volume = sum(t.size for t in sell_trades)
            
            # Calculate VWAP
            total_value = sum(t.value for t in trades)
            vwap = total_value / total_volume if total_volume > 0 else Decimal("0")
            
            return {
                "window_start": window_start,
                "trade_count": len(trades),
                "total_volume": total_volume,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "buy_sell_ratio": buy_volume / sell_volume if sell_volume > 0 else Decimal("0"),
                "vwap": vwap,
                "first_price": trades[0].price,
                "last_price": trades[-1].price,
                "min_price": min(t.price for t in trades),
                "max_price": max(t.price for t in trades),
            }
        
        aggregated = aggregate_trades_by_time_window(immutable_trade_data, 5)
        
        assert len(aggregated) > 0
        for agg in aggregated:
            assert "trade_count" in agg
            assert "total_volume" in agg
            assert "vwap" in agg
            assert agg["total_volume"] >= 0
            assert agg["trade_count"] > 0
    
    def test_pure_momentum_indicators(self, immutable_trade_data):
        """Test pure momentum indicator calculations from trades."""
        def calculate_trade_momentum(trades: List[Trade], lookback_count: int = 50) -> Decimal:
            """Pure function to calculate trade momentum."""
            if len(trades) < lookback_count:
                return Decimal("0")
            
            recent_trades = trades[-lookback_count:]
            
            # Calculate volume-weighted price momentum
            total_buy_volume = sum(t.size for t in recent_trades if t.is_buy())
            total_sell_volume = sum(t.size for t in recent_trades if t.is_sell())
            
            if total_buy_volume + total_sell_volume == 0:
                return Decimal("0")
            
            # Momentum based on buy/sell volume imbalance
            momentum = (total_buy_volume - total_sell_volume) / (total_buy_volume + total_sell_volume)
            return momentum
        
        def calculate_price_acceleration(trades: List[Trade], periods: int = 20) -> Decimal:
            """Pure function to calculate price acceleration from trades."""
            if len(trades) < periods + 1:
                return Decimal("0")
            
            recent_trades = trades[-periods:]
            prices = [float(t.price) for t in recent_trades]
            
            # Calculate first and second derivatives (velocity and acceleration)
            if len(prices) < 3:
                return Decimal("0")
            
            # Simple finite difference approximation
            velocities = []
            for i in range(1, len(prices)):
                velocity = prices[i] - prices[i-1]
                velocities.append(velocity)
            
            if len(velocities) < 2:
                return Decimal("0")
            
            accelerations = []
            for i in range(1, len(velocities)):
                acceleration = velocities[i] - velocities[i-1]
                accelerations.append(acceleration)
            
            if not accelerations:
                return Decimal("0")
            
            # Average acceleration
            avg_acceleration = sum(accelerations) / len(accelerations)
            return Decimal(str(avg_acceleration))
        
        momentum = calculate_trade_momentum(immutable_trade_data)
        acceleration = calculate_price_acceleration(immutable_trade_data)
        
        assert -1 <= momentum <= 1
        assert isinstance(acceleration, Decimal)


class TestFunctionalMarketRegimeDetection:
    """Test functional market regime detection with immutable data."""
    
    def test_pure_volatility_regime_detection(self, immutable_btc_candles):
        """Test pure volatility regime detection."""
        def detect_volatility_regime(candles: List[Candle], lookback: int = 20) -> str:
            """Pure function to detect volatility regime."""
            if len(candles) < lookback:
                return "INSUFFICIENT_DATA"
            
            recent_candles = candles[-lookback:]
            
            # Calculate realized volatility
            returns = []
            for i in range(1, len(recent_candles)):
                prev_price = recent_candles[i-1].close
                curr_price = recent_candles[i].close
                if prev_price > 0:
                    return_val = (curr_price - prev_price) / prev_price
                    returns.append(float(return_val))
            
            if len(returns) < 2:
                return "INSUFFICIENT_DATA"
            
            # Calculate standard deviation of returns
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = variance ** 0.5
            
            # Classify regime based on volatility thresholds
            if volatility > 0.03:  # 3% daily volatility
                return "HIGH_VOLATILITY"
            elif volatility < 0.01:  # 1% daily volatility
                return "LOW_VOLATILITY"
            else:
                return "NORMAL_VOLATILITY"
        
        def detect_trend_regime(candles: List[Candle], short_period: int = 10, long_period: int = 30) -> str:
            """Pure function to detect trend regime."""
            if len(candles) < long_period:
                return "INSUFFICIENT_DATA"
            
            # Calculate moving averages
            short_ma = sum(c.close for c in candles[-short_period:]) / Decimal(short_period)
            long_ma = sum(c.close for c in candles[-long_period:]) / Decimal(long_period)
            current_price = candles[-1].close
            
            # Classify trend regime
            if short_ma > long_ma and current_price > short_ma:
                return "STRONG_UPTREND"
            elif short_ma < long_ma and current_price < short_ma:
                return "STRONG_DOWNTREND"
            elif short_ma > long_ma:
                return "WEAK_UPTREND"
            elif short_ma < long_ma:
                return "WEAK_DOWNTREND"
            else:
                return "SIDEWAYS"
        
        vol_regime = detect_volatility_regime(immutable_btc_candles)
        trend_regime = detect_trend_regime(immutable_btc_candles)
        
        expected_vol_regimes = ["HIGH_VOLATILITY", "LOW_VOLATILITY", "NORMAL_VOLATILITY", "INSUFFICIENT_DATA"]
        expected_trend_regimes = ["STRONG_UPTREND", "STRONG_DOWNTREND", "WEAK_UPTREND", "WEAK_DOWNTREND", "SIDEWAYS", "INSUFFICIENT_DATA"]
        
        assert vol_regime in expected_vol_regimes
        assert trend_regime in expected_trend_regimes
    
    def test_pure_market_microstructure_analysis(self, immutable_trade_data, immutable_order_book):
        """Test pure market microstructure analysis."""
        def analyze_market_microstructure(
            trades: List[Trade], 
            order_book: OrderBook,
            window_minutes: int = 10
        ) -> Dict[str, Any]:
            """Pure function to analyze market microstructure."""
            if not trades:
                return {"error": "No trades provided"}
            
            # Filter recent trades
            cutoff_time = trades[-1].timestamp - timedelta(minutes=window_minutes)
            recent_trades = [t for t in trades if t.timestamp >= cutoff_time]
            
            if not recent_trades:
                return {"error": "No recent trades"}
            
            # Calculate microstructure metrics
            buy_trades = [t for t in recent_trades if t.is_buy()]
            sell_trades = [t for t in recent_trades if t.is_sell()]
            
            buy_volume = sum(t.size for t in buy_trades)
            sell_volume = sum(t.size for t in sell_trades)
            total_volume = buy_volume + sell_volume
            
            # Trade size distribution
            trade_sizes = [float(t.size) for t in recent_trades]
            avg_trade_size = sum(trade_sizes) / len(trade_sizes) if trade_sizes else 0
            
            # Price impact analysis
            if len(recent_trades) >= 2:
                price_changes = []
                for i in range(1, len(recent_trades)):
                    price_change = float(recent_trades[i].price - recent_trades[i-1].price)
                    price_changes.append(price_change)
                avg_price_impact = sum(abs(pc) for pc in price_changes) / len(price_changes)
            else:
                avg_price_impact = 0
            
            # Order book metrics
            spread = float(order_book.spread) if order_book.spread else 0
            mid_price = float(order_book.mid_price) if order_book.mid_price else 0
            
            return {
                "trade_count": len(recent_trades),
                "buy_volume": float(buy_volume),
                "sell_volume": float(sell_volume),
                "volume_imbalance": float((buy_volume - sell_volume) / total_volume) if total_volume > 0 else 0,
                "average_trade_size": avg_trade_size,
                "average_price_impact": avg_price_impact,
                "bid_ask_spread": spread,
                "mid_price": mid_price,
                "market_quality_score": _calculate_market_quality_score(
                    spread, avg_price_impact, total_volume, mid_price
                ),
            }
        
        def _calculate_market_quality_score(
            spread: float, 
            price_impact: float, 
            volume: Decimal, 
            mid_price: float
        ) -> float:
            """Helper function to calculate market quality score."""
            if mid_price == 0:
                return 0.0
            
            # Normalize metrics (lower spread and price impact = higher quality)
            spread_score = max(0, 1 - (spread / mid_price) * 10000)  # Spread in bps
            impact_score = max(0, 1 - price_impact / mid_price * 1000)
            volume_score = min(1, float(volume) / 1000)  # Normalize volume
            
            # Weighted average
            quality_score = (spread_score * 0.4 + impact_score * 0.4 + volume_score * 0.2)
            return max(0, min(1, quality_score))
        
        microstructure = analyze_market_microstructure(
            immutable_trade_data, 
            immutable_order_book
        )
        
        assert "trade_count" in microstructure
        assert "volume_imbalance" in microstructure
        assert "market_quality_score" in microstructure
        assert -1 <= microstructure["volume_imbalance"] <= 1
        assert 0 <= microstructure["market_quality_score"] <= 1


class TestFunctionalPerformanceOptimization:
    """Test performance optimization with functional programming patterns."""
    
    def test_lazy_evaluation_patterns(self):
        """Test lazy evaluation patterns for large datasets."""
        def create_large_candle_generator(count: int):
            """Generator function for creating large candle datasets."""
            base_time = datetime.now(UTC)
            base_price = Decimal("50000")
            
            for i in range(count):
                # Generate price movement
                price_change = Decimal(str(np.random.normal(0, 0.01)))
                new_price = base_price * (1 + price_change)
                
                yield Candle(
                    timestamp=base_time + timedelta(minutes=i),
                    open=base_price,
                    high=new_price * Decimal("1.005"),
                    low=new_price * Decimal("0.995"),
                    close=new_price,
                    volume=Decimal(str(np.random.uniform(50, 200))),
                    symbol="BTC-USD",
                )
                
                base_price = new_price
        
        def lazy_sma_calculation(candle_generator, periods: int):
            """Lazy SMA calculation using generator."""
            window = []
            
            for candle in candle_generator:
                window.append(candle.close)
                
                if len(window) > periods:
                    window.pop(0)
                
                if len(window) == periods:
                    sma = sum(window) / Decimal(periods)
                    yield (candle.timestamp, sma)
        
        # Test with large dataset
        large_dataset = create_large_candle_generator(10000)
        sma_generator = lazy_sma_calculation(large_dataset, 20)
        
        # Process only first 100 values
        sma_values = []
        for i, (timestamp, sma) in enumerate(sma_generator):
            if i >= 100:
                break
            sma_values.append(sma)
        
        assert len(sma_values) == 100
        assert all(isinstance(sma, Decimal) for sma in sma_values)
    
    def test_functional_data_pipeline_composition(self, immutable_btc_candles):
        """Test composing data processing pipelines functionally."""
        def compose(*functions):
            """Functional composition utility."""
            def composed_function(data):
                result = data
                for func in functions:
                    result = func(result)
                return result
            return composed_function
        
        # Define individual processing functions
        def extract_close_prices(candles: List[Candle]) -> List[Decimal]:
            return [candle.close for candle in candles]
        
        def calculate_returns(prices: List[Decimal]) -> List[Decimal]:
            if len(prices) < 2:
                return []
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0:
                    ret = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(ret)
            return returns
        
        def calculate_rolling_volatility(returns: List[Decimal], window: int = 20) -> List[Decimal]:
            if len(returns) < window:
                return []
            
            volatilities = []
            for i in range(window - 1, len(returns)):
                window_returns = returns[i - window + 1:i + 1]
                mean_return = sum(window_returns) / Decimal(window)
                variance = sum((r - mean_return) ** 2 for r in window_returns) / Decimal(window)
                volatility = variance.sqrt() if hasattr(variance, 'sqrt') else Decimal(str(float(variance) ** 0.5))
                volatilities.append(volatility)
            
            return volatilities
        
        # Compose the pipeline
        volatility_pipeline = compose(
            extract_close_prices,
            calculate_returns,
            lambda returns: calculate_rolling_volatility(returns, 10)
        )
        
        # Execute pipeline
        volatilities = volatility_pipeline(immutable_btc_candles)
        
        assert isinstance(volatilities, list)
        assert all(isinstance(vol, Decimal) for vol in volatilities)
        assert len(volatilities) <= len(immutable_btc_candles)
    
    def test_parallel_processing_patterns(self, immutable_btc_candles):
        """Test parallel processing patterns with immutable data."""
        def chunk_data(data: List[Any], chunk_size: int) -> List[List[Any]]:
            """Pure function to chunk data for parallel processing."""
            return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        def process_candle_chunk(candles: List[Candle]) -> Dict[str, Any]:
            """Pure function to process a chunk of candles."""
            if not candles:
                return {"volume_sum": Decimal("0"), "price_range": Decimal("0")}
            
            volume_sum = sum(candle.volume for candle in candles)
            prices = [candle.close for candle in candles]
            price_range = max(prices) - min(prices) if prices else Decimal("0")
            
            return {
                "volume_sum": volume_sum,
                "price_range": price_range,
                "candle_count": len(candles),
                "avg_volume": volume_sum / Decimal(len(candles)),
            }
        
        def merge_chunk_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Pure function to merge results from parallel processing."""
            if not results:
                return {}
            
            total_volume = sum(r["volume_sum"] for r in results)
            total_candles = sum(r["candle_count"] for r in results)
            max_price_range = max(r["price_range"] for r in results)
            
            return {
                "total_volume": total_volume,
                "total_candles": total_candles,
                "max_price_range": max_price_range,
                "avg_volume": total_volume / Decimal(total_candles) if total_candles > 0 else Decimal("0"),
            }
        
        # Simulate parallel processing
        chunks = chunk_data(immutable_btc_candles, 20)
        chunk_results = [process_candle_chunk(chunk) for chunk in chunks]
        final_result = merge_chunk_results(chunk_results)
        
        assert "total_volume" in final_result
        assert "total_candles" in final_result
        assert final_result["total_candles"] == len(immutable_btc_candles)
        assert final_result["total_volume"] > Decimal("0")


class TestFunctionalErrorHandling:
    """Test error handling patterns in functional market analysis."""
    
    def test_result_type_error_handling(self):
        """Test using Result types for error handling."""
        def safe_calculate_correlation(
            series1: List[float], 
            series2: List[float]
        ) -> Result[float, str]:
            """Safe correlation calculation returning Result type."""
            try:
                if len(series1) != len(series2):
                    return Failure("Series must have equal length")
                
                if len(series1) < 2:
                    return Failure("Need at least 2 data points")
                
                correlation_matrix = np.corrcoef(series1, series2)
                correlation = float(correlation_matrix[0, 1])
                
                if np.isnan(correlation):
                    return Failure("Correlation calculation resulted in NaN")
                
                return Success(correlation)
                
            except Exception as e:
                return Failure(f"Calculation error: {str(e)}")
        
        def safe_parse_candle_data(data: Dict[str, Any]) -> Result[Candle, str]:
            """Safe candle parsing returning Result type."""
            try:
                required_fields = ["timestamp", "open", "high", "low", "close", "volume"]
                
                for field in required_fields:
                    if field not in data:
                        return Failure(f"Missing required field: {field}")
                
                candle = Candle(
                    timestamp=data["timestamp"] if isinstance(data["timestamp"], datetime) 
                             else datetime.fromisoformat(data["timestamp"]),
                    open=Decimal(str(data["open"])),
                    high=Decimal(str(data["high"])),
                    low=Decimal(str(data["low"])),
                    close=Decimal(str(data["close"])),
                    volume=Decimal(str(data["volume"])),
                    symbol=data.get("symbol", "UNKNOWN"),
                )
                
                return Success(candle)
                
            except ValueError as e:
                return Failure(f"Invalid data format: {str(e)}")
            except Exception as e:
                return Failure(f"Parsing error: {str(e)}")
        
        # Test successful correlation
        valid_series1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        valid_series2 = [2.0, 4.0, 6.0, 8.0, 10.0]
        
        result = safe_calculate_correlation(valid_series1, valid_series2)
        assert result.is_success()
        assert result.success() == 1.0  # Perfect correlation
        
        # Test error cases
        mismatched_series = [1.0, 2.0, 3.0]
        result = safe_calculate_correlation(valid_series1, mismatched_series)
        assert result.is_failure()
        assert "equal length" in result.failure()
        
        # Test successful candle parsing
        valid_candle_data = {
            "timestamp": datetime.now(UTC),
            "open": "50000",
            "high": "51000",
            "low": "49000",
            "close": "50500",
            "volume": "100",
            "symbol": "BTC-USD",
        }
        
        candle_result = safe_parse_candle_data(valid_candle_data)
        assert candle_result.is_success()
        candle = candle_result.success()
        assert candle.symbol == "BTC-USD"
        
        # Test candle parsing error
        invalid_candle_data = {
            "timestamp": datetime.now(UTC),
            "open": "invalid_price",  # Invalid decimal
            "high": "51000",
            "low": "49000",
            "close": "50500",
            "volume": "100",
        }
        
        candle_result = safe_parse_candle_data(invalid_candle_data)
        assert candle_result.is_failure()
        assert "Invalid data format" in candle_result.failure()
    
    def test_graceful_degradation_patterns(self, immutable_btc_candles):
        """Test graceful degradation patterns for robust analysis."""
        def robust_trend_analysis(
            candles: List[Candle], 
            preferred_periods: int = 20,
            fallback_periods: int = 10,
            minimum_periods: int = 5
        ) -> Dict[str, Any]:
            """Robust trend analysis with fallback strategies."""
            if len(candles) < minimum_periods:
                return {
                    "trend": "INSUFFICIENT_DATA",
                    "confidence": 0.0,
                    "periods_used": 0,
                    "analysis_quality": "POOR",
                }
            
            # Determine periods to use based on available data
            if len(candles) >= preferred_periods:
                periods = preferred_periods
                quality = "HIGH"
            elif len(candles) >= fallback_periods:
                periods = fallback_periods
                quality = "MEDIUM"
            else:
                periods = len(candles)
                quality = "LOW"
            
            # Calculate trend using available data
            recent_candles = candles[-periods:]
            prices = [candle.close for candle in recent_candles]
            
            # Simple trend calculation
            start_price = prices[0]
            end_price = prices[-1]
            
            if start_price == 0:
                return {
                    "trend": "UNKNOWN",
                    "confidence": 0.0,
                    "periods_used": periods,
                    "analysis_quality": quality,
                }
            
            price_change = (end_price - start_price) / start_price
            
            # Determine trend direction and confidence
            if price_change > Decimal("0.02"):  # > 2%
                trend = "STRONG_UPTREND"
                confidence = min(1.0, float(abs(price_change)) * 10)
            elif price_change > Decimal("0.005"):  # > 0.5%
                trend = "WEAK_UPTREND"
                confidence = min(1.0, float(abs(price_change)) * 20)
            elif price_change < Decimal("-0.02"):  # < -2%
                trend = "STRONG_DOWNTREND"
                confidence = min(1.0, float(abs(price_change)) * 10)
            elif price_change < Decimal("-0.005"):  # < -0.5%
                trend = "WEAK_DOWNTREND"
                confidence = min(1.0, float(abs(price_change)) * 20)
            else:
                trend = "SIDEWAYS"
                confidence = 0.8  # High confidence in sideways detection
            
            # Adjust confidence based on data quality
            quality_multiplier = {"HIGH": 1.0, "MEDIUM": 0.8, "LOW": 0.6, "POOR": 0.0}
            final_confidence = confidence * quality_multiplier[quality]
            
            return {
                "trend": trend,
                "confidence": final_confidence,
                "periods_used": periods,
                "analysis_quality": quality,
                "price_change_pct": float(price_change * 100),
            }
        
        # Test with full dataset
        full_analysis = robust_trend_analysis(immutable_btc_candles)
        assert full_analysis["analysis_quality"] == "HIGH"
        assert full_analysis["periods_used"] == 20
        
        # Test with limited dataset
        limited_candles = immutable_btc_candles[:15]
        limited_analysis = robust_trend_analysis(limited_candles)
        assert limited_analysis["analysis_quality"] == "MEDIUM"
        assert limited_analysis["periods_used"] == 10
        
        # Test with very limited dataset
        minimal_candles = immutable_btc_candles[:3]
        minimal_analysis = robust_trend_analysis(minimal_candles)
        assert minimal_analysis["analysis_quality"] == "LOW"
        assert minimal_analysis["periods_used"] == 3
        
        # Test with insufficient data
        insufficient_candles = immutable_btc_candles[:2]
        insufficient_analysis = robust_trend_analysis(insufficient_candles)
        assert insufficient_analysis["trend"] == "INSUFFICIENT_DATA"
        assert insufficient_analysis["confidence"] == 0.0


if __name__ == "__main__":
    # Run some basic functionality tests
    print("Testing Functional Market Context Analysis...")
    
    # Test candle creation and immutability
    candle = Candle(
        timestamp=datetime.now(UTC),
        open=Decimal("50000"),
        high=Decimal("51000"),
        low=Decimal("49000"),
        close=Decimal("50500"),
        volume=Decimal("100"),
    )
    
    assert candle.is_bullish is True
    assert candle.price_range == Decimal("2000")
    print("✓ Immutable candle tests passed")
    
    # Test trade creation and validation
    trade = Trade(
        id="test_trade",
        timestamp=datetime.now(UTC),
        price=Decimal("50000"),
        size=Decimal("0.5"),
        side="BUY",
    )
    
    assert trade.value == Decimal("25000")
    assert trade.is_buy() is True
    print("✓ Immutable trade tests passed")
    
    # Test order book functionality
    bids = [(Decimal("49990"), Decimal("1.0")), (Decimal("49980"), Decimal("2.0"))]
    asks = [(Decimal("50010"), Decimal("1.5")), (Decimal("50020"), Decimal("2.5"))]
    
    order_book = OrderBook(
        bids=bids,
        asks=asks,
        timestamp=datetime.now(UTC),
    )
    
    assert order_book.spread == Decimal("20")  # 50010 - 49990
    assert order_book.mid_price == Decimal("50000")
    print("✓ Immutable order book tests passed")
    
    print("All functional market context analysis tests completed successfully!")