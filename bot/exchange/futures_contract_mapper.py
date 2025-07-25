"""Map spot symbols to actual futures contracts."""

import logging
from datetime import UTC, datetime
from typing import ClassVar

logger = logging.getLogger(__name__)


class FuturesContractMapper:
    """Maps spot symbols to their corresponding futures contracts."""

    # Map base symbols to their futures contract prefixes
    FUTURES_PREFIXES: ClassVar[dict[str, str]] = {
        "ETH-USD": "ET",
        "BTC-USD": "BT",
        "SOL-USD": "SOL",
        "DOGE-USD": "DOGE",
        "LTC-USD": "LTC",
        "BCH-USD": "BCH",
    }

    @staticmethod
    def get_current_contract_month() -> str:
        """Get the current/next contract month in format DDMMMYY."""
        # Futures typically expire on the last Friday of the month
        # We should use the next month's contract if we're past expiry
        datetime.now(UTC)

        # For now, return the known active contract
        # In production, this would calculate based on current date
        return "27JUN25"

    @staticmethod
    def spot_to_futures_symbol(spot_symbol: str) -> str:
        """Convert spot symbol to futures contract symbol.

        Args:
            spot_symbol: Spot trading pair (e.g., 'ETH-USD')

        Returns:
            Futures contract symbol (e.g., 'ET-27JUN25-CDE')
        """
        prefix = FuturesContractMapper.FUTURES_PREFIXES.get(spot_symbol)
        if not prefix:
            logger.warning("No futures mapping for %s", spot_symbol)
            return spot_symbol

        month = FuturesContractMapper.get_current_contract_month()
        futures_symbol = f"{prefix}-{month}-CDE"

        logger.info("Mapped %s to futures contract %s", spot_symbol, futures_symbol)
        return futures_symbol

    @staticmethod
    def futures_to_spot_symbol(futures_symbol: str) -> str:
        """Convert futures contract symbol back to spot symbol.

        Args:
            futures_symbol: Futures contract (e.g., 'ET-27JUN25-CDE')

        Returns:
            Spot symbol (e.g., 'ETH-USD')
        """
        # Extract the prefix (everything before the first dash)
        parts = futures_symbol.split("-")
        if len(parts) < 3:
            return futures_symbol

        prefix = parts[0]

        # Reverse lookup
        for spot, fut_prefix in FuturesContractMapper.FUTURES_PREFIXES.items():
            if fut_prefix == prefix:
                logger.info("Mapped futures %s back to %s", futures_symbol, spot)
                return spot

        logger.warning("No spot mapping for futures %s", futures_symbol)
        return futures_symbol
