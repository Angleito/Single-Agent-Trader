#!/usr/bin/env python3
"""
TradingView Data Feed Formatter for AI Trading Bot Dashboard

Converts bot data into TradingView-compatible format for professional trading charts.
Implements Universal Data Feed (UDF) specification for TradingView Charting Library.

Features:
- OHLCV data formatting with proper timestamps
- AI decision markers as chart annotations
- Technical indicators formatting (RSI, EMA, VuManChu Cipher)
- Real-time data feed interface
- Symbol configuration and metadata
- Multiple timeframe support (1m, 5m, 15m, 1h, 4h, 1d)
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Resolution(Enum):
    """Supported chart resolutions/timeframes"""

    MIN_1 = "1"
    MIN_5 = "5"
    MIN_15 = "15"
    HOUR_1 = "60"
    HOUR_4 = "240"
    DAY_1 = "1D"
    WEEK_1 = "1W"
    MONTH_1 = "1M"


class DecisionType(Enum):
    """AI Trading Decision Types"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


@dataclass
class SymbolInfo:
    """TradingView symbol information"""

    name: str
    ticker: str
    description: str
    type: str = "crypto"
    session: str = "24x7"
    timezone: str = "Etc/UTC"
    minmov: int = 1
    pricescale: int = 100000
    has_intraday: bool = True
    has_daily: bool = True
    has_weekly_and_monthly: bool = True
    supported_resolutions: list[str] = None
    intraday_multipliers: list[str] = None

    def __post_init__(self):
        if self.supported_resolutions is None:
            self.supported_resolutions = ["1", "5", "15", "60", "240", "1D", "1W", "1M"]
        if self.intraday_multipliers is None:
            self.intraday_multipliers = ["1", "5", "15", "60", "240"]


@dataclass
class OHLCVBar:
    """OHLCV price bar data"""

    timestamp: int  # Unix timestamp in seconds
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OHLCVBar":
        """Create OHLCVBar from dictionary"""
        return cls(
            timestamp=int(data.get("timestamp", 0)),
            open=float(data.get("open", 0)),
            high=float(data.get("high", 0)),
            low=float(data.get("low", 0)),
            close=float(data.get("close", 0)),
            volume=float(data.get("volume", 0)),
        )


@dataclass
class AIDecisionMarker:
    """AI trading decision marker for chart annotations"""

    timestamp: int  # Unix timestamp in seconds
    decision: DecisionType
    price: float
    confidence: float
    reasoning: str
    indicator_values: dict[str, float] = None

    def __post_init__(self):
        if self.indicator_values is None:
            self.indicator_values = {}


@dataclass
class TechnicalIndicator:
    """Technical indicator data point"""

    timestamp: int
    name: str
    value: float | dict[str, float]
    parameters: dict[str, Any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class TradingViewDataFeed:
    """TradingView Universal Data Feed (UDF) formatter"""

    def __init__(self):
        self.symbols: dict[str, SymbolInfo] = {}
        self.price_data: dict[str, dict[str, list[OHLCVBar]]] = (
            {}
        )  # symbol -> resolution -> bars
        self.ai_decisions: dict[str, list[AIDecisionMarker]] = {}  # symbol -> decisions
        self.indicators: dict[str, dict[str, list[TechnicalIndicator]]] = (
            {}
        )  # symbol -> indicator_name -> values

        # Initialize default symbols
        self._init_default_symbols()

    def _init_default_symbols(self):
        """Initialize default cryptocurrency symbols"""
        default_symbols = [
            ("BTC-USD", "Bitcoin vs US Dollar", 100),
            ("ETH-USD", "Ethereum vs US Dollar", 100),
            ("DOGE-USD", "Dogecoin vs US Dollar", 100000),
            ("SOL-USD", "Solana vs US Dollar", 100),
            ("ADA-USD", "Cardano vs US Dollar", 10000),
        ]

        for ticker, description, pricescale in default_symbols:
            self.symbols[ticker] = SymbolInfo(
                name=ticker,
                ticker=ticker,
                description=description,
                pricescale=pricescale,
            )

    def get_symbol_info(self, symbol: str) -> dict[str, Any] | None:
        """Get symbol information in TradingView format"""
        if symbol not in self.symbols:
            return None

        symbol_info = self.symbols[symbol]
        return asdict(symbol_info)

    def get_symbols_list(self) -> list[dict[str, Any]]:
        """Get list of all available symbols"""
        return [
            {
                "symbol": symbol,
                "full_name": info.name,
                "description": info.description,
                "type": info.type,
                "ticker": info.ticker,
            }
            for symbol, info in self.symbols.items()
        ]

    def add_symbol(self, symbol_info: SymbolInfo):
        """Add new symbol to feed"""
        self.symbols[symbol_info.ticker] = symbol_info
        if symbol_info.ticker not in self.price_data:
            self.price_data[symbol_info.ticker] = {}
        if symbol_info.ticker not in self.ai_decisions:
            self.ai_decisions[symbol_info.ticker] = []
        if symbol_info.ticker not in self.indicators:
            self.indicators[symbol_info.ticker] = {}

    def add_price_data(self, symbol: str, resolution: str, bars: list[OHLCVBar]):
        """Add OHLCV price data for symbol and resolution"""
        if symbol not in self.price_data:
            self.price_data[symbol] = {}

        # Sort bars by timestamp
        sorted_bars = sorted(bars, key=lambda x: x.timestamp)
        self.price_data[symbol][resolution] = sorted_bars

        logger.info(
            "Added %s price bars for %s at %s resolution",
            len(sorted_bars),
            symbol,
            resolution,
        )

    def get_history(
        self,
        symbol: str,
        resolution: str,
        from_timestamp: int,
        to_timestamp: int,
        countback: int | None = None,
    ) -> dict[str, Any]:
        """
        Get historical data in TradingView format

        Returns:
        {
            "s": "ok"|"no_data"|"error",
            "t": [timestamps],
            "o": [open_prices],
            "h": [high_prices],
            "l": [low_prices],
            "c": [close_prices],
            "v": [volumes]
        }
        """
        try:
            if (
                symbol not in self.price_data
                or resolution not in self.price_data[symbol]
            ):
                return {"s": "no_data"}

            bars = self.price_data[symbol][resolution]

            # Filter bars by timestamp range
            filtered_bars = [
                bar for bar in bars if from_timestamp <= bar.timestamp <= to_timestamp
            ]

            # Apply countback limit if specified
            if countback and len(filtered_bars) > countback:
                filtered_bars = filtered_bars[-countback:]

            if not filtered_bars:
                return {"s": "no_data"}

            # Format data for TradingView
            response = {
                "s": "ok",
                "t": [bar.timestamp for bar in filtered_bars],
                "o": [bar.open for bar in filtered_bars],
                "h": [bar.high for bar in filtered_bars],
                "l": [bar.low for bar in filtered_bars],
                "c": [bar.close for bar in filtered_bars],
                "v": [bar.volume for bar in filtered_bars],
            }

            logger.info(
                "Returning %s bars for %s %s", len(filtered_bars), symbol, resolution
            )
            return response

        except Exception as e:
            logger.exception("Error getting history for %s", symbol)
            return {"s": "error", "errmsg": str(e)}

    def get_marks(
        self, symbol: str, from_timestamp: int, to_timestamp: int, _resolution: str
    ) -> list[dict[str, Any]]:
        """
        Get AI decision markers for chart annotations

        Returns list of marks in TradingView format:
        [
            {
                "id": "unique_id",
                "time": timestamp,
                "color": "red"|"green"|"blue",
                "text": "marker_text",
                "label": "B"|"S"|"H",
                "labelFontColor": "white",
                "minSize": 20
            }
        ]
        """
        try:
            if symbol not in self.ai_decisions:
                return []

            decisions = self.ai_decisions[symbol]

            # Filter decisions by timestamp range
            filtered_decisions = [
                decision
                for decision in decisions
                if from_timestamp <= decision.timestamp <= to_timestamp
            ]

            marks = []
            for i, decision in enumerate(filtered_decisions):
                # Determine marker color and label based on decision type
                color_map = {
                    DecisionType.BUY: ("green", "B"),
                    DecisionType.SELL: ("red", "S"),
                    DecisionType.HOLD: ("blue", "H"),
                    DecisionType.CLOSE_LONG: ("orange", "CL"),
                    DecisionType.CLOSE_SHORT: ("purple", "CS"),
                }

                color, label = color_map.get(decision.decision, ("gray", "?"))

                mark = {
                    "id": f"{symbol}_{decision.timestamp}_{i}",
                    "time": decision.timestamp,
                    "color": color,
                    "text": f"{decision.decision.value}: {decision.reasoning[:50]}...",
                    "label": label,
                    "labelFontColor": "white",
                    "minSize": 20,
                }
                marks.append(mark)

            logger.info("Returning %s AI decision marks for %s", len(marks), symbol)
            return marks

        except Exception:
            logger.exception("Error getting marks for %s", symbol)
            return []

    def get_timescale_marks(
        self, _symbol: str, _from_timestamp: int, _to_timestamp: int, _resolution: str
    ) -> list[dict[str, Any]]:
        """
        Get timescale marks (important events on time scale)
        Currently returns empty list but can be extended for important events
        """
        return []

    def add_ai_decision(self, symbol: str, decision: AIDecisionMarker):
        """Add AI trading decision marker"""
        if symbol not in self.ai_decisions:
            self.ai_decisions[symbol] = []

        self.ai_decisions[symbol].append(decision)

        # Keep only last 1000 decisions to prevent memory issues
        if len(self.ai_decisions[symbol]) > 1000:
            self.ai_decisions[symbol] = self.ai_decisions[symbol][-1000:]

        logger.info(
            "Added AI decision for %s: %s at %s",
            symbol,
            decision.decision.value,
            decision.price,
        )

    def add_technical_indicator(self, symbol: str, indicator: TechnicalIndicator):
        """Add technical indicator data point"""
        if symbol not in self.indicators:
            self.indicators[symbol] = {}

        if indicator.name not in self.indicators[symbol]:
            self.indicators[symbol][indicator.name] = []

        self.indicators[symbol][indicator.name].append(indicator)

        # Keep only last 1000 values to prevent memory issues
        if len(self.indicators[symbol][indicator.name]) > 1000:
            self.indicators[symbol][indicator.name] = self.indicators[symbol][
                indicator.name
            ][-1000:]

    def get_indicator_values(
        self, symbol: str, indicator_name: str, from_timestamp: int, to_timestamp: int
    ) -> list[dict[str, Any]]:
        """Get technical indicator values for specified time range"""
        try:
            if (
                symbol not in self.indicators
                or indicator_name not in self.indicators[symbol]
            ):
                return []

            indicators = self.indicators[symbol][indicator_name]

            # Filter by timestamp range
            filtered_indicators = [
                indicator
                for indicator in indicators
                if from_timestamp <= indicator.timestamp <= to_timestamp
            ]

            # Convert to TradingView study format
            study_data = []
            for indicator in filtered_indicators:
                data_point = {"time": indicator.timestamp, "value": indicator.value}
                if indicator.parameters:
                    data_point["parameters"] = indicator.parameters
                study_data.append(data_point)

            return study_data

        except Exception:
            logger.exception(
                "Error getting indicator values for %s %s",
                symbol,
                indicator_name,
            )
            return []

    def get_real_time_bar(self, symbol: str, resolution: str) -> dict[str, Any] | None:
        """Get the most recent bar for real-time updates"""
        try:
            if (
                symbol not in self.price_data
                or resolution not in self.price_data[symbol]
            ):
                return None

            bars = self.price_data[symbol][resolution]
            if not bars:
                return None

            latest_bar = bars[-1]

            return {
                "time": latest_bar.timestamp,
                "open": latest_bar.open,
                "high": latest_bar.high,
                "low": latest_bar.low,
                "close": latest_bar.close,
                "volume": latest_bar.volume,
            }

        except Exception:
            logger.exception("Error getting real-time bar for %s", symbol)
            return None

    def update_real_time_bar(
        self, symbol: str, resolution: str, bar_update: dict[str, Any]
    ):
        """Update the most recent bar with real-time data"""
        try:
            if (
                symbol not in self.price_data
                or resolution not in self.price_data[symbol]
            ):
                return

            bars = self.price_data[symbol][resolution]
            if not bars:
                return

            # Update the latest bar
            latest_bar = bars[-1]
            if "high" in bar_update and bar_update["high"] > latest_bar.high:
                latest_bar.high = bar_update["high"]
            if "low" in bar_update and bar_update["low"] < latest_bar.low:
                latest_bar.low = bar_update["low"]
            if "close" in bar_update:
                latest_bar.close = bar_update["close"]
            if "volume" in bar_update:
                latest_bar.volume = bar_update["volume"]

            logger.debug("Updated real-time bar for %s %s", symbol, resolution)

        except Exception:
            logger.exception("Error updating real-time bar for %s", symbol)

    def create_new_bar(self, symbol: str, resolution: str, bar_data: dict[str, Any]):
        """Create a new bar when timeframe completes"""
        try:
            if symbol not in self.price_data:
                self.price_data[symbol] = {}
            if resolution not in self.price_data[symbol]:
                self.price_data[symbol][resolution] = []

            new_bar = OHLCVBar.from_dict(bar_data)
            self.price_data[symbol][resolution].append(new_bar)

            # Keep only last 5000 bars to prevent memory issues
            if len(self.price_data[symbol][resolution]) > 5000:
                self.price_data[symbol][resolution] = self.price_data[symbol][
                    resolution
                ][-5000:]

            logger.info(
                "Created new bar for %s %s at %s", symbol, resolution, new_bar.timestamp
            )

        except Exception:
            logger.exception("Error creating new bar for %s", symbol)

    @staticmethod
    def resolution_to_seconds(resolution: str) -> int:
        """Convert TradingView resolution to seconds"""
        resolution_map = {
            "1": 60,  # 1 minute
            "5": 300,  # 5 minutes
            "15": 900,  # 15 minutes
            "60": 3600,  # 1 hour
            "240": 14400,  # 4 hours
            "1D": 86400,  # 1 day
            "1W": 604800,  # 1 week
            "1M": 2592000,  # 1 month (30 days)
        }
        return resolution_map.get(resolution, 60)

    @staticmethod
    def timestamp_to_unix(dt: datetime) -> int:
        """Convert datetime to Unix timestamp in seconds"""
        return int(dt.timestamp())

    @staticmethod
    def unix_to_datetime(timestamp: int) -> datetime:
        """Convert Unix timestamp to datetime"""
        return datetime.fromtimestamp(timestamp, tz=UTC)

    def export_data_summary(self) -> dict[str, Any]:
        """Export summary of all data for debugging"""
        return {
            "symbols": list(self.symbols.keys()),
            "price_data": {
                symbol: {
                    resolution: len(bars) for resolution, bars in resolutions.items()
                }
                for symbol, resolutions in self.price_data.items()
            },
            "ai_decisions": {
                symbol: len(decisions)
                for symbol, decisions in self.ai_decisions.items()
            },
            "indicators": {
                symbol: {
                    indicator: len(values) for indicator, values in indicators.items()
                }
                for symbol, indicators in self.indicators.items()
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }


# Global instance for use in FastAPI endpoints
tradingview_feed = TradingViewDataFeed()


# Utility functions for data conversion


def convert_bot_trade_to_ai_decision(
    trade_data: dict[str, Any], _symbol: str
) -> AIDecisionMarker | None:
    """Convert bot trade data to AI decision marker"""
    try:
        decision_map = {
            "buy": DecisionType.BUY,
            "sell": DecisionType.SELL,
            "hold": DecisionType.HOLD,
            "close_long": DecisionType.CLOSE_LONG,
            "close_short": DecisionType.CLOSE_SHORT,
        }

        decision_type = decision_map.get(
            trade_data.get("action", "").lower(), DecisionType.HOLD
        )

        return AIDecisionMarker(
            timestamp=int(trade_data.get("timestamp", time.time())),
            decision=decision_type,
            price=float(trade_data.get("price", 0)),
            confidence=float(trade_data.get("confidence", 0.5)),
            reasoning=trade_data.get("reasoning", "No reasoning provided"),
            indicator_values=trade_data.get("indicators", {}),
        )

    except Exception:
        logger.exception("Error converting bot trade to AI decision")
        return None


def convert_market_data_to_ohlcv(market_data: list[dict[str, Any]]) -> list[OHLCVBar]:
    """Convert market data to OHLCV bars"""
    try:
        bars = []
        for data_point in market_data:
            bar = OHLCVBar.from_dict(data_point)
            bars.append(bar)

        return sorted(bars, key=lambda x: x.timestamp)

    except Exception:
        logger.exception("Error converting market data to OHLCV")
        return []


def convert_indicator_data_to_technical_indicator(
    indicator_data: dict[str, Any], indicator_name: str, _symbol: str
) -> TechnicalIndicator | None:
    """Convert indicator data to technical indicator format"""
    try:
        return TechnicalIndicator(
            timestamp=int(indicator_data.get("timestamp", time.time())),
            name=indicator_name,
            value=indicator_data.get("value", 0),
            parameters=indicator_data.get("parameters", {}),
        )

    except Exception:
        logger.exception("Error converting indicator data")
        return None


# Example usage and test data generation
def generate_sample_data():
    """Generate sample data for testing TradingView integration"""
    import random

    # Get the global feed instance
    feed = tradingview_feed

    # Generate sample OHLCV data for BTC-USD
    now = int(time.time())
    sample_bars = []

    base_price = 65000.0
    for i in range(100):
        timestamp = now - (100 - i) * 60  # 1-minute bars

        # Simple random walk for demo data generation
        price_change = random.uniform(-0.02, 0.02)  # noqa: S311
        open_price = base_price * (1 + price_change)
        high_price = open_price * (1 + random.uniform(0, 0.01))  # noqa: S311
        low_price = open_price * (1 - random.uniform(0, 0.01))  # noqa: S311
        close_price = open_price * (1 + random.uniform(-0.01, 0.01))  # noqa: S311
        volume = random.uniform(10, 100)  # noqa: S311

        bar = OHLCVBar(
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
        )
        sample_bars.append(bar)
        base_price = close_price

    feed.add_price_data("BTC-USD", "1", sample_bars)

    # Generate sample AI decisions
    for i in range(10):
        decision_timestamp = now - (10 - i) * 600  # Every 10 minutes
        decisions = [DecisionType.BUY, DecisionType.SELL, DecisionType.HOLD]
        decision_type = decisions[i % 3]

        marker = AIDecisionMarker(
            timestamp=decision_timestamp,
            decision=decision_type,
            price=base_price + random.uniform(-100, 100),  # noqa: S311
            confidence=random.uniform(0.6, 0.95),  # noqa: S311
            reasoning=f"AI decision based on market analysis #{i + 1}",
            indicator_values={
                "rsi": random.uniform(30, 70),
                "ema": base_price,
            },
        )

        feed.add_ai_decision("BTC-USD", marker)

    # Generate sample technical indicators
    for i in range(50):
        indicator_timestamp = now - (50 - i) * 60

        # RSI indicator
        rsi_indicator = TechnicalIndicator(
            timestamp=indicator_timestamp,
            name="RSI",
            value=random.uniform(20, 80),  # noqa: S311
            parameters={"period": 14},
        )
        feed.add_technical_indicator("BTC-USD", rsi_indicator)

        # EMA indicator
        ema_indicator = TechnicalIndicator(
            timestamp=indicator_timestamp,
            name="EMA",
            value=base_price + random.uniform(-500, 500),  # noqa: S311
            parameters={"period": 20},
        )
        feed.add_technical_indicator("BTC-USD", ema_indicator)

    logger.info("Generated sample data for TradingView feed")


if __name__ == "__main__":
    # Test the data feed with sample data
    generate_sample_data()

    # Test getting historical data
    now = int(time.time())
    history = tradingview_feed.get_history("BTC-USD", "1", now - 3600, now)
    print("Sample history response:", json.dumps(history, indent=2))

    # Test getting AI decision marks
    marks = tradingview_feed.get_marks("BTC-USD", now - 3600, now, "1")
    print(f"Sample marks: {len(marks)} AI decisions")

    # Export data summary
    summary = tradingview_feed.export_data_summary()
    print("Data summary:", json.dumps(summary, indent=2))
