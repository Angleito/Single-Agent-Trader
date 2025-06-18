"""
Symbol utilities for exchange integration.

Provides functions for symbol validation and conversion between different
exchange formats.
"""

import re
from typing import Literal


class SymbolConversionError(Exception):
    """Exception raised when symbol conversion fails."""
    pass


class InvalidSymbolError(Exception):
    """Exception raised when symbol is invalid."""
    pass


def validate_symbol(symbol: str) -> bool:
    """
    Validate trading symbol format.

    Args:
        symbol: Trading symbol to validate (e.g., "BTC-USD", "ETH-USD")

    Returns:
        True if symbol is valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False

    # Basic pattern for crypto pairs: BASE-QUOTE
    pattern = r"^[A-Z0-9]{2,10}-[A-Z]{3,4}$"
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
    if "-" in symbol:
        base = symbol.split("-")[0]
    else:
        base = symbol

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
    return f"{base}-PERP"


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
    elif any(month in symbol_upper for month in ["MAR", "JUN", "SEP", "DEC"]):
        return "futures"
    else:
        return "spot"


class BluefinSymbolConverter:
    """Symbol converter for Bluefin exchange integration."""

    def __init__(self, market_symbols_enum=None):
        """Initialize converter with market symbols enum."""
        self.market_symbols_enum = market_symbols_enum

    def validate_market_symbol(self, symbol: str) -> bool:
        """Validate if symbol can be converted to market symbol."""
        try:
            self.to_market_symbol(symbol)
            return True
        except (SymbolConversionError, InvalidSymbolError):
            return False

    def to_market_symbol(self, symbol: str):
        """Convert string symbol to MARKET_SYMBOLS enum value."""
        if not symbol or not isinstance(symbol, str):
            raise InvalidSymbolError(f"Invalid symbol: {symbol}")

        # Normalize the symbol
        normalized = normalize_symbol(symbol, "PERP")

        # Extract base currency
        base = normalized.split("-")[0].upper()

        if not self.market_symbols_enum:
            raise SymbolConversionError("MARKET_SYMBOLS enum not available")

        # Enhanced enum attribute access with multiple fallback approaches
        # Approach 1: Direct attribute access
        if hasattr(self.market_symbols_enum, base):
            try:
                return getattr(self.market_symbols_enum, base)
            except AttributeError:
                # Continue to fallback approaches
                pass
        
        # Approach 2: Try with different case variations
        for case_variant in [base.upper(), base.lower(), base.capitalize()]:
            if hasattr(self.market_symbols_enum, case_variant):
                try:
                    return getattr(self.market_symbols_enum, case_variant)
                except AttributeError:
                    continue
        
        # Approach 3: Try to find by value if the enum supports it
        try:
            for attr_name in dir(self.market_symbols_enum):
                if not attr_name.startswith('_'):
                    try:
                        attr_value = getattr(self.market_symbols_enum, attr_name)
                        # Check if the attribute value matches our base currency
                        if hasattr(attr_value, 'value') and str(attr_value.value).upper() == base.upper():
                            return attr_value
                        elif hasattr(attr_value, 'name') and str(attr_value.name).upper() == base.upper():
                            return attr_value
                        elif str(attr_value).upper() == base.upper():
                            return attr_value
                    except (AttributeError, TypeError):
                        continue
        except Exception:
            # If introspection fails, continue to final error
            pass

        raise SymbolConversionError(f"Unknown symbol: {symbol} (base: {base}) - not found in MARKET_SYMBOLS enum")

    def from_market_symbol(self, market_symbol) -> str:
        """Convert MARKET_SYMBOLS enum to string representation."""
        if hasattr(market_symbol, 'name'):
            return f"{market_symbol.name}-PERP"
        elif hasattr(market_symbol, 'value'):
            return f"{market_symbol.value}-PERP"
        else:
            return f"{str(market_symbol)}-PERP"
