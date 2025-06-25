"""
Fixes for BluefinMarketDataProvider to ensure reliable market data flow.
This module contains patches and enhancements to address:
1. API connectivity test using invalid endpoints
2. WebSocket message handling improvements
3. Data validation and recovery mechanisms
4. Synthetic candle generation for testing
"""

import logging
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from bot.trading_types import MarketData

logger = logging.getLogger(__name__)


class BluefinDataFixes:
    """Collection of fixes for Bluefin market data issues."""

    @staticmethod
    def get_valid_test_endpoint() -> str:
        """
        Get a valid endpoint for API connectivity testing.

        The /time and /ping endpoints don't exist on Bluefin API.
        Use /exchangeInfo as it's a lightweight public endpoint.
        """
        return "/exchangeInfo"

    @staticmethod
    def validate_websocket_message(message: dict) -> bool:
        """
        Validate incoming WebSocket messages for proper structure.

        Args:
            message: WebSocket message dict

        Returns:
            True if message is valid trade/tick data
        """
        # Check for required fields in trade/tick messages
        required_fields = ["symbol", "price", "quantity", "timestamp"]

        # Handle different message types
        if "type" in message:
            msg_type = message.get("type")
            if msg_type in {"trade", "tick"}:
                return all(field in message for field in required_fields)
            if msg_type in {"kline", "candle"}:
                return all(
                    field in message
                    for field in [
                        "symbol",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "timestamp",
                    ]
                )

        # Check if it's a trade update format
        if "data" in message and isinstance(message["data"], dict):
            return BluefinDataFixes.validate_websocket_message(message["data"])

        return False

    @staticmethod
    def convert_18_decimal_price(price_str: str) -> Decimal:
        """
        Convert 18-decimal format prices to human-readable format.

        Bluefin uses 18 decimal places for price precision.
        """
        try:
            price_value = Decimal(price_str)
            # Convert from 18 decimal places
            return price_value / Decimal(1000000000000000000)
        except:
            logger.exception(f"Failed to convert price: {price_str}")
            return Decimal(0)

    @staticmethod
    def parse_candle_data(candle_array: list) -> MarketData | None:
        """
        Parse Bluefin candle array format into MarketData object.

        Bluefin returns candles as arrays:
        [timestamp, open, high, low, close, volume, close_time, quote_volume, trades, ...]
        """
        try:
            if len(candle_array) < 8:
                logger.warning(f"Invalid candle array length: {len(candle_array)}")
                return None

            # Convert 18-decimal prices
            open_price = BluefinDataFixes.convert_18_decimal_price(candle_array[1])
            high_price = BluefinDataFixes.convert_18_decimal_price(candle_array[2])
            low_price = BluefinDataFixes.convert_18_decimal_price(candle_array[3])
            close_price = BluefinDataFixes.convert_18_decimal_price(candle_array[4])
            volume = BluefinDataFixes.convert_18_decimal_price(candle_array[5])

            # Convert timestamp from milliseconds
            timestamp = datetime.fromtimestamp(candle_array[0] / 1000, tz=UTC)

            return MarketData(
                symbol=candle_array[11] if len(candle_array) > 11 else "UNKNOWN",
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                bid=close_price,  # Use close as bid approximation
                ask=close_price,  # Use close as ask approximation
                bid_size=Decimal(0),
                ask_size=Decimal(0),
                last_trade_price=close_price,
                last_trade_size=Decimal(0),
                trades_count=int(candle_array[8]) if len(candle_array) > 8 else 0,
                vwap=close_price,  # Approximate VWAP with close
                market_state="TRADING",
                sequence_number=int(time.time() * 1000),
            )

        except Exception as e:
            logger.exception(f"Failed to parse candle data: {e}")
            return None

    @staticmethod
    def generate_synthetic_candles(
        symbol: str,
        num_candles: int = 100,
        interval: str = "1m",
        base_price: Decimal = Decimal(2500),
        volatility: Decimal = Decimal("0.001"),
    ) -> list[MarketData]:
        """
        Generate synthetic candles for testing when real data is unavailable.

        This helps ensure the bot can start and indicators can be calculated
        even when market data feed has issues.
        """
        candles = []

        # Parse interval to get minutes
        interval_minutes = 1
        if interval.endswith("m"):
            interval_minutes = int(interval[:-1])
        elif interval.endswith("h"):
            interval_minutes = int(interval[:-1]) * 60
        elif interval.endswith("d"):
            interval_minutes = int(interval[:-1]) * 1440

        current_time = datetime.now(UTC)
        current_price = base_price

        for i in range(num_candles):
            # Calculate timestamp for this candle
            timestamp = current_time - timedelta(
                minutes=interval_minutes * (num_candles - i - 1)
            )

            # Generate OHLC with some randomness
            import random

            price_change = current_price * volatility * Decimal(random.uniform(-1, 1))

            open_price = current_price
            close_price = current_price + price_change
            high_price = max(open_price, close_price) * Decimal(
                1 + random.uniform(0, 0.001)
            )
            low_price = min(open_price, close_price) * Decimal(
                1 - random.uniform(0, 0.001)
            )

            # Generate volume
            volume = Decimal(random.uniform(100, 1000))

            candle = MarketData(
                symbol=symbol,
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                bid=close_price * Decimal("0.9999"),
                ask=close_price * Decimal("1.0001"),
                bid_size=Decimal(random.uniform(10, 100)),
                ask_size=Decimal(random.uniform(10, 100)),
                last_trade_price=close_price,
                last_trade_size=Decimal(random.uniform(1, 10)),
                trades_count=random.randint(10, 100),
                vwap=close_price,
                market_state="TRADING",
                sequence_number=int(timestamp.timestamp() * 1000),
            )

            candles.append(candle)
            current_price = close_price

        logger.info(f"Generated {len(candles)} synthetic candles for {symbol}")
        return candles

    @staticmethod
    def create_fallback_ticker_data(
        symbol: str, last_price: Decimal | None = None
    ) -> dict[str, Any]:
        """
        Create fallback ticker data when API is unavailable.
        """
        if last_price is None:
            # Default prices for common symbols
            default_prices = {
                "BTC-PERP": Decimal(108000),
                "ETH-PERP": Decimal(3400),
                "SUI-PERP": Decimal("2.50"),
                "SOL-PERP": Decimal(260),
            }
            last_price = default_prices.get(symbol, Decimal(100))

        return {
            "symbol": symbol,
            "price": str(int(last_price * Decimal(1000000000000000000))),
            "lastPrice": str(int(last_price * Decimal(1000000000000000000))),
            "bestBidPrice": str(
                int(last_price * Decimal("0.9999") * Decimal(1000000000000000000))
            ),
            "bestAskPrice": str(
                int(last_price * Decimal("1.0001") * Decimal(1000000000000000000))
            ),
            "indexPrice": str(int(last_price * Decimal(1000000000000000000))),
            "_24hrVolume": "1000000000000000000000",
            "_24hrPriceChange": "0",
            "_24hrPriceChangePercent": "0",
        }

    @staticmethod
    async def enhanced_connectivity_test(session, api_base_url: str) -> bool:
        """
        Enhanced API connectivity test using valid endpoints.
        """
        try:
            # Use exchangeInfo endpoint for connectivity test
            test_url = f"{api_base_url}/exchangeInfo"

            async with session.get(test_url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    # Verify it's actually returning exchange data
                    if isinstance(data, list) and len(data) > 0:
                        logger.debug(
                            "✅ API connectivity validated using /exchangeInfo"
                        )
                        return True

                logger.warning(
                    f"❌ API connectivity test failed with status: {response.status}"
                )
                return False

        except TimeoutError:
            logger.warning("❌ API connectivity test timed out")
            return False
        except Exception as e:
            logger.warning(f"❌ API connectivity validation failed: {e}")
            return False

    @staticmethod
    def create_candle_from_trades(
        trades: list[dict[str, Any]], interval_seconds: int
    ) -> MarketData | None:
        """
        Aggregate trades into a candle for custom intervals.
        """
        if not trades:
            return None

        try:
            # Sort trades by timestamp
            sorted_trades = sorted(trades, key=lambda x: x.get("timestamp", 0))

            # Get time boundaries
            first_trade = sorted_trades[0]
            sorted_trades[-1]

            # Calculate OHLC
            prices = [
                BluefinDataFixes.convert_18_decimal_price(t.get("price", "0"))
                for t in sorted_trades
            ]
            volumes = [
                BluefinDataFixes.convert_18_decimal_price(t.get("quantity", "0"))
                for t in sorted_trades
            ]

            open_price = prices[0]
            close_price = prices[-1]
            high_price = max(prices)
            low_price = min(prices)
            total_volume = sum(volumes)

            # Calculate VWAP
            vwap = (
                sum(p * v for p, v in zip(prices, volumes, strict=False)) / total_volume
                if total_volume > 0
                else close_price
            )

            # Create candle
            return MarketData(
                symbol=first_trade.get("symbol", "UNKNOWN"),
                timestamp=datetime.fromtimestamp(
                    first_trade["timestamp"] / 1000, tz=UTC
                ),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=total_volume,
                bid=close_price * Decimal("0.9999"),
                ask=close_price * Decimal("1.0001"),
                bid_size=Decimal(0),
                ask_size=Decimal(0),
                last_trade_price=close_price,
                last_trade_size=volumes[-1] if volumes else Decimal(0),
                trades_count=len(trades),
                vwap=vwap,
                market_state="TRADING",
                sequence_number=int(time.time() * 1000),
            )

        except Exception as e:
            logger.exception(f"Failed to create candle from trades: {e}")
            return None


# Export fix functions for easy patching
def patch_bluefin_market_provider():
    """
    Apply all fixes to BluefinMarketDataProvider.
    This can be called during bot initialization.
    """
    logger.info("Applying Bluefin market data fixes...")

    # The actual patching would be done in the main bot initialization
    # This is just a placeholder to show the intended usage
    return BluefinDataFixes
