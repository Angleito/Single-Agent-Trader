"""
Symbol utilities for exchange integration.

Provides functions for symbol validation and conversion between different
exchange formats.
"""

import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)


class SymbolConversionError(Exception):
    """Exception raised when symbol conversion fails."""


class InvalidSymbolError(Exception):
    """Exception raised when symbol is invalid."""


def validate_symbol(symbol: str) -> bool:
    """
    Validate trading symbol format.

    Args:
        symbol: Trading symbol to validate (e.g., "BTC-USD", "ETH-USD", "BTC-PERP")

    Returns:
        True if symbol is valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False

    # Enhanced pattern for crypto pairs: BASE-QUOTE or BASE-PERP
    pattern = r"^[A-Z0-9]{2,10}-(USD|USDC|PERP)$"
    return bool(re.match(pattern, symbol.upper()))


def normalize_symbol(symbol: str, suffix: str = "PERP") -> str:
    """
    Normalize symbol format.

    Args:
        symbol: Symbol to normalize
        suffix: Suffix to add if not present

    Returns:
        Normalized symbol
    """
    if not symbol:
        raise InvalidSymbolError("Empty symbol")

    symbol = symbol.upper().strip()

    # If it already has the suffix, return as-is
    if symbol.endswith(f"-{suffix}"):
        return symbol

    # Extract base currency
    base = symbol.split("-")[0] if "-" in symbol else symbol

    return f"{base}-{suffix}"


def to_bluefin_perp(symbol: str) -> str:
    """
    Convert symbol to Bluefin perpetual format.

    Args:
        symbol: Standard symbol format (e.g., "BTC-USD")

    Returns:
        Bluefin perpetual symbol format
    """
    if not validate_symbol(symbol):
        raise ValueError(f"Invalid symbol format: {symbol}")

    # Convert BTC-USD to BTC-PERP format for Bluefin
    base = symbol.split("-")[0]
    perp_symbol = f"{base}-PERP"

    # Validate that the resulting symbol is supported on Bluefin
    if not is_bluefin_symbol_supported(perp_symbol):
        raise ValueError(f"Symbol {perp_symbol} is not supported on Bluefin")

    return perp_symbol


def to_coinbase_format(symbol: str) -> str:
    """
    Convert symbol to Coinbase format.

    Args:
        symbol: Standard symbol format

    Returns:
        Coinbase symbol format
    """
    if not validate_symbol(symbol):
        raise ValueError(f"Invalid symbol format: {symbol}")

    # Coinbase uses the same format
    return symbol.upper()


def parse_symbol(symbol: str) -> tuple[str, str]:
    """
    Parse symbol into base and quote components.

    Args:
        symbol: Trading symbol (e.g., "BTC-USD")

    Returns:
        Tuple of (base, quote) currencies
    """
    if not validate_symbol(symbol):
        raise ValueError(f"Invalid symbol format: {symbol}")

    parts = symbol.upper().split("-")
    return parts[0], parts[1]


def get_symbol_type(symbol: str) -> Literal["spot", "futures", "perpetual"]:
    """
    Determine the type of trading instrument from symbol.

    Args:
        symbol: Trading symbol

    Returns:
        Type of trading instrument
    """
    if not validate_symbol(symbol):
        raise ValueError(f"Invalid symbol format: {symbol}")

    symbol_upper = symbol.upper()

    if "PERP" in symbol_upper:
        return "perpetual"
    if any(month in symbol_upper for month in ["MAR", "JUN", "SEP", "DEC"]):
        return "futures"
    return "spot"


# Bluefin supported symbols (based on official documentation)
BLUEFIN_MAINNET_SYMBOLS = {
    "BTC-PERP": {"min_trade_size": 0.001, "max_trade_size": 100, "step_size": 0.001},
    "ETH-PERP": {"min_trade_size": 0.01, "max_trade_size": 1000, "step_size": 0.01},
    "SOL-PERP": {"min_trade_size": 0.1, "max_trade_size": 50000, "step_size": 0.1},
    "SUI-PERP": {"min_trade_size": 1.0, "max_trade_size": 1000000, "step_size": 1.0},
}

# Testnet typically has same symbols as mainnet but might be limited
BLUEFIN_TESTNET_SYMBOLS = {
    "BTC-PERP": {"min_trade_size": 0.001, "max_trade_size": 100, "step_size": 0.001},
    "ETH-PERP": {"min_trade_size": 0.01, "max_trade_size": 1000, "step_size": 0.01},
    "SOL-PERP": {"min_trade_size": 0.1, "max_trade_size": 50000, "step_size": 0.1},
    # SUI-PERP might not be available on testnet, using BTC-PERP as fallback
}


def is_bluefin_symbol_supported(symbol: str, network: str = "mainnet") -> bool:
    """
    Check if a symbol is supported on Bluefin.

    Args:
        symbol: Trading symbol (e.g., "BTC-PERP")
        network: Network type ("mainnet" or "testnet")

    Returns:
        True if symbol is supported, False otherwise
    """
    symbol = symbol.upper()
    if network.lower() in ["mainnet", "production", "sui_prod"]:
        return symbol in BLUEFIN_MAINNET_SYMBOLS
    if network.lower() in ["testnet", "staging", "sui_staging"]:
        return symbol in BLUEFIN_TESTNET_SYMBOLS
    # Default to mainnet for unknown networks
    return symbol in BLUEFIN_MAINNET_SYMBOLS


def get_bluefin_symbol_info(symbol: str, network: str = "mainnet") -> dict:
    """
    Get symbol trading information for Bluefin.

    Args:
        symbol: Trading symbol (e.g., "BTC-PERP")
        network: Network type ("mainnet" or "testnet")

    Returns:
        Dictionary with symbol trading parameters
    """
    symbol = symbol.upper()
    if network.lower() in ["mainnet", "production", "sui_prod"]:
        return BLUEFIN_MAINNET_SYMBOLS.get(symbol, {})
    if network.lower() in ["testnet", "staging", "sui_staging"]:
        return BLUEFIN_TESTNET_SYMBOLS.get(symbol, {})
    return BLUEFIN_MAINNET_SYMBOLS.get(symbol, {})


def get_testnet_symbol_fallback(symbol: str) -> str:
    """
    Get a fallback symbol for testnet if the requested symbol is not available.

    Args:
        symbol: Requested symbol (e.g., "SUI-PERP")

    Returns:
        Available testnet symbol
    """
    symbol = symbol.upper()

    # If the symbol is available on testnet, return it
    if symbol in BLUEFIN_TESTNET_SYMBOLS:
        return symbol

    # Fallback mapping for unavailable symbols
    fallback_map = {
        "SUI-PERP": "BTC-PERP",  # SUI-PERP might not be on testnet
        "AVAX-PERP": "ETH-PERP",
        "MATIC-PERP": "SOL-PERP",
    }

    return fallback_map.get(symbol, "BTC-PERP")  # Default to BTC-PERP


class BluefinSymbolConverter:
    """Symbol converter for Bluefin exchange integration."""

    def __init__(self, market_symbols_enum=None, network: str = "mainnet"):
        """Initialize converter with market symbols enum."""
        self.market_symbols_enum = market_symbols_enum
        self.network = network

    def validate_market_symbol(self, symbol: str) -> bool:
        """Validate if symbol can be converted to market symbol."""
        try:
            self.to_market_symbol(symbol)
        except (SymbolConversionError, InvalidSymbolError):
            return False
        else:
            return True

    def to_market_symbol(self, symbol: str):
        """Convert string symbol to MARKET_SYMBOLS enum value."""
        if not symbol or not isinstance(symbol, str):
            raise TypeError(f"Invalid symbol: {symbol}")

        # Normalize the symbol
        normalized = normalize_symbol(symbol, "PERP")
        base = normalized.split("-")[0].upper()

        if not self.market_symbols_enum:
            raise SymbolConversionError("MARKET_SYMBOLS enum not available")

        # Try different methods to find the symbol
        result = (
            self._try_direct_attribute_access(base)
            or self._try_case_variations(base)
            or self._try_enum_introspection(base)
        )

        if result:
            return result

        raise SymbolConversionError(
            f"Unknown symbol: {symbol} (base: {base}) - not found in MARKET_SYMBOLS enum"
        )

    def _try_direct_attribute_access(self, base: str):
        """Try direct attribute access on the enum."""
        if hasattr(self.market_symbols_enum, base):
            try:
                return getattr(self.market_symbols_enum, base)
            except AttributeError:
                pass
        return None

    def _try_case_variations(self, base: str):
        """Try different case variations of the base symbol."""
        for case_variant in [base.upper(), base.lower(), base.capitalize()]:
            if hasattr(self.market_symbols_enum, case_variant):
                try:
                    return getattr(self.market_symbols_enum, case_variant)
                except AttributeError:
                    continue
        return None

    def _try_enum_introspection(self, base: str):
        """Try to find symbol by introspecting enum values."""
        try:
            for attr_name in dir(self.market_symbols_enum):
                if attr_name.startswith("_"):
                    continue

                try:
                    attr_value = getattr(self.market_symbols_enum, attr_name)
                    if self._matches_symbol(attr_value, base):
                        return attr_value
                except (AttributeError, TypeError):
                    continue
        except Exception as e:
            logger.debug("Failed to introspect symbol during conversion: %s", e)
        return None

    def _matches_symbol(self, attr_value, base: str) -> bool:
        """Check if an enum attribute matches the base symbol."""
        # Check value attribute
        if hasattr(attr_value, "value"):
            try:
                value_attr = getattr(attr_value, "value", None)
                if value_attr is not None and str(value_attr).upper() == base.upper():
                    return True
            except (AttributeError, TypeError):
                pass

        # Check name attribute
        if hasattr(attr_value, "name"):
            try:
                name_attr = getattr(attr_value, "name", None)
                if name_attr is not None and str(name_attr).upper() == base.upper():
                    return True
            except (AttributeError, TypeError):
                pass

        # Direct string comparison as fallback
        try:
            if str(attr_value).upper() == base.upper():
                return True
        except (AttributeError, TypeError):
            pass

        return False

    def from_market_symbol(self, market_symbol) -> str:
        """Convert MARKET_SYMBOLS enum to string representation with robust attribute access."""
        # Try name attribute first
        if hasattr(market_symbol, "name"):
            try:
                name_attr = getattr(market_symbol, "name", None)
                if name_attr is not None:
                    return f"{name_attr}-PERP"
            except (AttributeError, TypeError):
                pass

        # Try value attribute as fallback
        if hasattr(market_symbol, "value"):
            try:
                value_attr = getattr(market_symbol, "value", None)
                if value_attr is not None:
                    return f"{value_attr}-PERP"
            except (AttributeError, TypeError):
                pass

        # Direct string conversion as final fallback
        try:
            symbol_str = str(market_symbol)
            # Clean up any enum prefixes
            if "MARKET_SYMBOLS." in symbol_str:
                symbol_str = symbol_str.replace("MARKET_SYMBOLS.", "")
        except Exception:
            # Ultimate fallback
            return "UNKNOWN-PERP"
        else:
            return f"{symbol_str}-PERP"
