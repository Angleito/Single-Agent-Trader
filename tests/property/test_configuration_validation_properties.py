"""
Property-based tests for configuration parameter validation.

Tests the functional programming configuration system with property-based
validation, security checks, and boundary condition testing.
"""

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given, settings

from bot.fp.types.base import Money, Percentage, Symbol, TimeInterval, TradingMode
from bot.fp.types.config import (
    APIKey,
    MomentumStrategyConfig,
    PrivateKey,
)
from bot.fp.types.result import Failure, Success


# Strategies for configuration testing
@st.composite
def api_key_strategy(draw, valid=True):
    """Generate API key strings for testing."""
    if valid:
        # Valid API keys: at least 10 characters
        length = draw(st.integers(min_value=10, max_value=100))
        chars = st.characters(
            min_codepoint=32, max_codepoint=126, blacklist_characters="\n\r\t"
        )
        return draw(st.text(alphabet=chars, min_size=length, max_size=length))
    # Invalid API keys: less than 10 characters
    length = draw(st.integers(min_value=0, max_value=9))
    chars = st.characters(
        min_codepoint=32, max_codepoint=126, blacklist_characters="\n\r\t"
    )
    return draw(st.text(alphabet=chars, min_size=length, max_size=length))


@st.composite
def private_key_strategy(draw, valid=True):
    """Generate private key strings for testing."""
    if valid:
        # Valid private keys: at least 20 characters
        length = draw(st.integers(min_value=20, max_value=200))
        chars = st.characters(
            min_codepoint=32, max_codepoint=126, blacklist_characters="\n\r\t"
        )
        return draw(st.text(alphabet=chars, min_size=length, max_size=length))
    # Invalid private keys: less than 20 characters
    length = draw(st.integers(min_value=0, max_value=19))
    chars = st.characters(
        min_codepoint=32, max_codepoint=126, blacklist_characters="\n\r\t"
    )
    return draw(st.text(alphabet=chars, min_size=length, max_size=length))


@st.composite
def percentage_strategy(draw, valid=True):
    """Generate percentage values for testing."""
    if valid:
        return draw(
            st.floats(
                min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        )
    # Invalid percentages: outside 0-1 range
    return draw(
        st.one_of(
            st.floats(min_value=-10.0, max_value=-0.001),
            st.floats(min_value=1.001, max_value=10.0),
            st.just(float("nan")),
            st.just(float("inf")),
            st.just(float("-inf")),
        )
    )


@st.composite
def money_strategy(draw, valid=True):
    """Generate money values for testing."""
    if valid:
        return draw(
            st.floats(
                min_value=0.01,
                max_value=1000000.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
    # Invalid money: negative or special values
    return draw(
        st.one_of(
            st.floats(min_value=-1000.0, max_value=-0.01),
            st.just(0.0),
            st.just(float("nan")),
            st.just(float("inf")),
            st.just(float("-inf")),
        )
    )


@st.composite
def symbol_strategy(draw, valid=True):
    """Generate trading symbol strings for testing."""
    if valid:
        bases = [
            "BTC",
            "ETH",
            "SOL",
            "ADA",
            "DOT",
            "MATIC",
            "LINK",
            "UNI",
            "AAVE",
            "ATOM",
        ]
        quotes = ["USD", "USDT", "USDC", "EUR", "BTC", "ETH"]

        base = draw(st.sampled_from(bases))
        quote = draw(st.sampled_from(quotes))
        assume(base != quote)

        return f"{base}-{quote}"
    # Invalid symbols: various malformed patterns
    return draw(
        st.one_of(
            st.just(""),  # Empty
            st.just("BTC"),  # Missing quote
            st.just("-USD"),  # Missing base
            st.just("BTC-"),  # Missing quote
            st.just("BTC_USD"),  # Wrong separator
            st.just("btc-usd"),  # Lowercase
            st.text(min_size=1, max_size=5).filter(
                lambda x: "-" not in x
            ),  # No separator
            st.text(min_size=20, max_size=50),  # Too long
        )
    )


@st.composite
def time_interval_strategy(draw, valid=True):
    """Generate time interval strings for testing."""
    if valid:
        return draw(
            st.sampled_from(
                [
                    "1s",
                    "5s",
                    "15s",
                    "30s",
                    "1m",
                    "3m",
                    "5m",
                    "15m",
                    "30m",
                    "1h",
                    "4h",
                    "12h",
                    "1d",
                    "1w",
                ]
            )
        )
    # Invalid intervals
    return draw(
        st.one_of(
            st.just(""),
            st.just("1x"),  # Invalid unit
            st.just("60s"),  # Valid but not standard
            st.just("2h"),  # Valid but not standard
            st.just("1"),  # Missing unit
            st.just("s"),  # Missing number
            st.text(min_size=1, max_size=10).filter(
                lambda x: x
                not in [
                    "1s",
                    "5s",
                    "15s",
                    "30s",
                    "1m",
                    "3m",
                    "5m",
                    "15m",
                    "30m",
                    "1h",
                    "4h",
                    "12h",
                    "1d",
                    "1w",
                ]
            ),
        )
    )


class TestAPIKeyProperties:
    """Property-based tests for API key validation."""

    @given(api_key_strategy(valid=True))
    @settings(max_examples=100, deadline=1000)
    def test_valid_api_key_creation_succeeds(self, key_string: str):
        """Property: Valid API keys should always create successful APIKey instances."""
        result = APIKey.create(key_string)

        assert isinstance(
            result, Success
        ), f"Valid API key should succeed: {key_string[:10]}..."
        api_key = result.success()
        assert isinstance(api_key, APIKey)

        # Test string representation masking
        str_repr = str(api_key)
        assert "APIKey(" in str_repr
        assert "***" in str_repr
        assert key_string[-4:] in str_repr  # Last 4 characters should be visible
        assert key_string[:-4] not in str_repr  # Rest should be hidden

    @given(api_key_strategy(valid=False))
    @settings(max_examples=50, deadline=1000)
    def test_invalid_api_key_creation_fails(self, key_string: str):
        """Property: Invalid API keys should always fail validation."""
        result = APIKey.create(key_string)

        assert isinstance(result, Failure), f"Invalid API key should fail: {key_string}"
        error_msg = result.failure()
        assert "too short" in error_msg

    @given(api_key_strategy(valid=True))
    @settings(max_examples=50, deadline=1000)
    def test_api_key_masking_security_property(self, key_string: str):
        """Property: API key masking should never expose the full key."""
        result = APIKey.create(key_string)
        assume(isinstance(result, Success))

        api_key = result.success()

        # Test various string operations don't expose the key
        assert key_string not in str(api_key)
        assert key_string not in repr(api_key)

        # But last 4 characters should be visible for identification
        if len(key_string) >= 4:
            assert key_string[-4:] in str(api_key)


class TestPrivateKeyProperties:
    """Property-based tests for private key validation."""

    @given(private_key_strategy(valid=True))
    @settings(max_examples=100, deadline=1000)
    def test_valid_private_key_creation_succeeds(self, key_string: str):
        """Property: Valid private keys should always create successful PrivateKey instances."""
        result = PrivateKey.create(key_string)

        assert isinstance(result, Success), "Valid private key should succeed"
        private_key = result.success()
        assert isinstance(private_key, PrivateKey)

        # Test complete masking
        str_repr = str(private_key)
        assert (
            str_repr == "PrivateKey(***)"
        ), f"Private key should be fully masked, got: {str_repr}"

    @given(private_key_strategy(valid=False))
    @settings(max_examples=50, deadline=1000)
    def test_invalid_private_key_creation_fails(self, key_string: str):
        """Property: Invalid private keys should always fail validation."""
        result = PrivateKey.create(key_string)

        assert isinstance(result, Failure), "Invalid private key should fail"
        error_msg = result.failure()
        assert "too short" in error_msg

    @given(private_key_strategy(valid=True))
    @settings(max_examples=50, deadline=1000)
    def test_private_key_complete_masking_security_property(self, key_string: str):
        """Property: Private keys should never expose any part of the actual key."""
        result = PrivateKey.create(key_string)
        assume(isinstance(result, Success))

        private_key = result.success()

        # Private key should be completely hidden
        assert key_string not in str(private_key)
        assert key_string not in repr(private_key)

        # No part of the key should be visible
        for i in range(1, min(len(key_string), 10)):
            assert key_string[:i] not in str(private_key)
            assert key_string[-i:] not in str(private_key)


class TestPercentageProperties:
    """Property-based tests for percentage validation."""

    @given(percentage_strategy(valid=True))
    @settings(max_examples=100, deadline=1000)
    def test_valid_percentage_creation_succeeds(self, value: float):
        """Property: Valid percentages should always create successful Percentage instances."""
        result = Percentage.create(value)

        assert isinstance(result, Success), f"Valid percentage should succeed: {value}"
        percentage = result.success()
        assert isinstance(percentage, Percentage)

        # Test value is preserved
        assert abs(float(percentage.value) - value) < 0.0001

    @given(percentage_strategy(valid=False))
    @settings(max_examples=50, deadline=1000)
    def test_invalid_percentage_creation_fails(self, value: float):
        """Property: Invalid percentages should always fail validation."""
        result = Percentage.create(value)

        assert isinstance(result, Failure), f"Invalid percentage should fail: {value}"
        error_msg = result.failure()
        assert any(
            keyword in error_msg.lower() for keyword in ["range", "invalid", "positive"]
        )

    @given(
        st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=1000)
    def test_percentage_arithmetic_properties(self, value: float):
        """Property: Percentage arithmetic should maintain valid ranges."""
        result = Percentage.create(value)
        assume(isinstance(result, Success))

        percentage = result.success()

        # Test that percentage maintains its properties
        assert 0 < percentage.value <= 1
        assert percentage.as_decimal() == percentage.value

        # Test basis points conversion
        bps = percentage.as_basis_points()
        assert 0 < bps <= 10000
        assert abs(bps - (value * 10000)) < 0.01


class TestMoneyProperties:
    """Property-based tests for money validation."""

    @given(money_strategy(valid=True))
    @settings(max_examples=100, deadline=1000)
    def test_valid_money_creation_succeeds(self, value: float):
        """Property: Valid money amounts should always create successful Money instances."""
        result = Money.create(value)

        assert isinstance(result, Success), f"Valid money should succeed: {value}"
        money = result.success()
        assert isinstance(money, Money)

        # Test value is preserved with proper precision
        assert abs(float(money.amount) - value) < 0.01
        assert money.amount > 0

    @given(money_strategy(valid=False))
    @settings(max_examples=50, deadline=1000)
    def test_invalid_money_creation_fails(self, value: float):
        """Property: Invalid money amounts should always fail validation."""
        result = Money.create(value)

        assert isinstance(result, Failure), f"Invalid money should fail: {value}"
        error_msg = result.failure()
        assert any(
            keyword in error_msg.lower()
            for keyword in ["positive", "invalid", "greater"]
        )


class TestSymbolProperties:
    """Property-based tests for trading symbol validation."""

    @given(symbol_strategy(valid=True))
    @settings(max_examples=100, deadline=1000)
    def test_valid_symbol_creation_succeeds(self, symbol_string: str):
        """Property: Valid symbols should always create successful Symbol instances."""
        result = Symbol.create(symbol_string)

        assert isinstance(
            result, Success
        ), f"Valid symbol should succeed: {symbol_string}"
        symbol = result.success()
        assert isinstance(symbol, Symbol)

        # Test symbol properties
        assert symbol.pair == symbol_string
        assert "-" in symbol.pair

        base, quote = symbol.pair.split("-")
        assert symbol.base_currency() == base
        assert symbol.quote_currency() == quote

    @given(symbol_strategy(valid=False))
    @settings(max_examples=50, deadline=1000)
    def test_invalid_symbol_creation_fails(self, symbol_string: str):
        """Property: Invalid symbols should always fail validation."""
        result = Symbol.create(symbol_string)

        assert isinstance(
            result, Failure
        ), f"Invalid symbol should fail: {symbol_string}"
        error_msg = result.failure()
        assert any(
            keyword in error_msg.lower()
            for keyword in ["invalid", "format", "separator"]
        )


class TestTimeIntervalProperties:
    """Property-based tests for time interval validation."""

    @given(time_interval_strategy(valid=True))
    @settings(max_examples=50, deadline=1000)
    def test_valid_time_interval_creation_succeeds(self, interval_string: str):
        """Property: Valid time intervals should always create successful TimeInterval instances."""
        result = TimeInterval.create(interval_string)

        assert isinstance(
            result, Success
        ), f"Valid time interval should succeed: {interval_string}"
        interval = result.success()
        assert isinstance(interval, TimeInterval)

        # Test interval properties
        assert interval.interval == interval_string
        assert interval.to_seconds() > 0

    @given(time_interval_strategy(valid=False))
    @settings(max_examples=30, deadline=1000)
    def test_invalid_time_interval_creation_fails(self, interval_string: str):
        """Property: Invalid time intervals should always fail validation."""
        result = TimeInterval.create(interval_string)

        assert isinstance(
            result, Failure
        ), f"Invalid time interval should fail: {interval_string}"
        error_msg = result.failure()
        assert any(
            keyword in error_msg.lower()
            for keyword in ["invalid", "unsupported", "format"]
        )


class TestMomentumStrategyConfigProperties:
    """Property-based tests for momentum strategy configuration."""

    @given(
        st.integers(min_value=1, max_value=1000),
        percentage_strategy(valid=True),
        percentage_strategy(valid=True),
        st.booleans(),
    )
    @settings(max_examples=100, deadline=1000)
    def test_valid_momentum_config_creation_succeeds(
        self,
        lookback: int,
        entry_threshold: float,
        exit_threshold: float,
        use_volume: bool,
    ):
        """Property: Valid momentum config should always create successful instances."""
        result = MomentumStrategyConfig.create(
            lookback_period=lookback,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            use_volume_confirmation=use_volume,
        )

        assert isinstance(result, Success), "Valid momentum config should succeed"
        config = result.success()
        assert isinstance(config, MomentumStrategyConfig)

        # Test config properties
        assert config.lookback_period == lookback
        assert config.use_volume_confirmation == use_volume
        assert 0 < config.entry_threshold.value <= 1
        assert 0 < config.exit_threshold.value <= 1

    @given(
        st.integers(min_value=-100, max_value=0),  # Invalid lookback
        percentage_strategy(valid=True),
        percentage_strategy(valid=True),
        st.booleans(),
    )
    @settings(max_examples=30, deadline=1000)
    def test_invalid_momentum_config_lookback_fails(
        self,
        lookback: int,
        entry_threshold: float,
        exit_threshold: float,
        use_volume: bool,
    ):
        """Property: Invalid lookback period should cause momentum config creation to fail."""
        result = MomentumStrategyConfig.create(
            lookback_period=lookback,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            use_volume_confirmation=use_volume,
        )

        assert isinstance(result, Failure), f"Invalid lookback should fail: {lookback}"
        error_msg = result.failure()
        assert "positive" in error_msg.lower()

    @given(
        st.integers(min_value=1, max_value=100),
        percentage_strategy(valid=False),  # Invalid entry threshold
        percentage_strategy(valid=True),
        st.booleans(),
    )
    @settings(max_examples=30, deadline=1000)
    def test_invalid_momentum_config_entry_threshold_fails(
        self,
        lookback: int,
        entry_threshold: float,
        exit_threshold: float,
        use_volume: bool,
    ):
        """Property: Invalid entry threshold should cause momentum config creation to fail."""
        result = MomentumStrategyConfig.create(
            lookback_period=lookback,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            use_volume_confirmation=use_volume,
        )

        assert isinstance(
            result, Failure
        ), f"Invalid entry threshold should fail: {entry_threshold}"
        error_msg = result.failure()
        assert "entry threshold" in error_msg.lower()


class TestConfigurationCombinations:
    """Property-based tests for complex configuration combinations."""

    @given(
        api_key_strategy(valid=True),
        private_key_strategy(valid=True),
        symbol_strategy(valid=True),
        time_interval_strategy(valid=True),
        st.sampled_from(["paper", "live", "backtest"]),
    )
    @settings(max_examples=50, deadline=2000)
    def test_complete_config_validation_properties(
        self,
        api_key: str,
        private_key: str,
        symbol: str,
        interval: str,
        trading_mode: str,
    ):
        """Property: Complete configuration should validate all components consistently."""
        # Test API key
        api_result = APIKey.create(api_key)
        assert isinstance(api_result, Success)

        # Test private key
        private_result = PrivateKey.create(private_key)
        assert isinstance(private_result, Success)

        # Test symbol
        symbol_result = Symbol.create(symbol)
        assert isinstance(symbol_result, Success)

        # Test interval
        interval_result = TimeInterval.create(interval)
        assert isinstance(interval_result, Success)

        # Test trading mode
        mode_result = TradingMode.create(trading_mode)
        assert isinstance(mode_result, Success)

        # All components should be individually valid
        api_obj = api_result.success()
        private_obj = private_result.success()
        symbol_obj = symbol_result.success()
        interval_obj = interval_result.success()
        mode_obj = mode_result.success()

        # Test that masking works consistently
        config_str = f"API: {api_obj}, Private: {private_obj}, Symbol: {symbol_obj}, Interval: {interval_obj}, Mode: {mode_obj}"

        # Sensitive data should be masked
        assert api_key not in config_str
        assert private_key not in config_str

        # Non-sensitive data should be visible
        assert symbol in config_str
        assert interval in config_str
        assert trading_mode in config_str


class TestConfigurationEdgeCases:
    """Property-based tests for configuration edge cases and boundary conditions."""

    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=100, deadline=1000)
    def test_arbitrary_string_api_key_validation(self, text: str):
        """Property: API key validation should handle arbitrary strings correctly."""
        result = APIKey.create(text)

        if len(text) >= 10:
            assert isinstance(
                result, Success
            ), f"String of length {len(text)} should be valid API key"
        else:
            assert isinstance(
                result, Failure
            ), f"String of length {len(text)} should be invalid API key"

    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=100, deadline=1000)
    def test_arbitrary_string_private_key_validation(self, text: str):
        """Property: Private key validation should handle arbitrary strings correctly."""
        result = PrivateKey.create(text)

        if len(text) >= 20:
            assert isinstance(
                result, Success
            ), f"String of length {len(text)} should be valid private key"
        else:
            assert isinstance(
                result, Failure
            ), f"String of length {len(text)} should be invalid private key"

    @given(st.floats(allow_nan=True, allow_infinity=True))
    @settings(max_examples=100, deadline=1000)
    def test_special_float_values_handling(self, value: float):
        """Property: Configuration should handle special float values appropriately."""
        # Test percentage
        percentage_result = Percentage.create(value)
        if 0 < value <= 1 and not (
            any(
                special(value)
                for special in [
                    float("nan").__eq__,
                    float("inf").__eq__,
                    float("-inf").__eq__,
                ]
            )
        ):
            assert isinstance(percentage_result, Success)
        else:
            assert isinstance(percentage_result, Failure)

        # Test money
        money_result = Money.create(value)
        if value > 0 and not (
            any(
                special(value)
                for special in [
                    float("nan").__eq__,
                    float("inf").__eq__,
                    float("-inf").__eq__,
                ]
            )
        ):
            assert isinstance(money_result, Success)
        else:
            assert isinstance(money_result, Failure)


if __name__ == "__main__":
    # Run property tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "--hypothesis-show-statistics"])
