"""Unit tests for technical indicators."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

# Legacy imports (maintained for compatibility)
from bot.indicators.vumanchu import CipherA, CipherB, VuManChuIndicators

# Functional imports (added for migration to functional programming patterns)
try:
    from bot.fp.indicators import (
        calculate_all_momentum_indicators,
        macd,
        rsi,
        stochastic,
    )
    from bot.fp.indicators.momentum import momentum_oscillator
    from bot.fp.indicators.vumanchu_functional import VuManchuState

    FUNCTIONAL_INDICATORS_AVAILABLE = True
except ImportError:
    # Functional implementations not available, continue with legacy
    FUNCTIONAL_INDICATORS_AVAILABLE = False


class TestCipherA:
    """Test cases for Cipher A indicator."""

    def test_cipher_a_initialization(self) -> None:
        """Test Cipher A initialization with default parameters."""
        cipher_a = CipherA()

        # Test default parameters - these may vary based on actual implementation
        assert hasattr(cipher_a, "wt_ma_length")
        assert hasattr(cipher_a, "wt_channel_length")
        assert hasattr(cipher_a, "wt_average_length")
        assert cipher_a.wt_ma_length > 0
        assert cipher_a.wt_channel_length > 0
        assert cipher_a.wt_average_length > 0

    def test_cipher_a_with_sample_data(self) -> None:
        """Test Cipher A calculation with sample data."""
        # Create sample OHLCV data
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        rng = np.random.default_rng(42)
        data = pd.DataFrame(
            {
                "open": rng.uniform(45000, 55000, 100),
                "high": rng.uniform(46000, 56000, 100),
                "low": rng.uniform(44000, 54000, 100),
                "close": rng.uniform(45000, 55000, 100),
                "volume": rng.uniform(10, 100, 100),
            },
            index=dates,
        )

        cipher_a = CipherA()
        result = cipher_a.calculate(data)

        # Check that new columns are added
        assert "wt1" in result.columns
        assert "wt2" in result.columns
        assert "rsi" in result.columns
        assert "cipher_a_signal" in result.columns
        assert "cipher_a_confidence" in result.columns

    def test_cipher_a_insufficient_data(self) -> None:
        """Test Cipher A with insufficient data."""
        # Create minimal data
        data = pd.DataFrame(
            {
                "open": [50000, 50100],
                "high": [50200, 50300],
                "low": [49800, 49900],
                "close": [50050, 50150],
                "volume": [10, 15],
            }
        )

        cipher_a = CipherA()
        result = cipher_a.calculate(data)

        # Should return original data without errors
        assert len(result) == 2


class TestCipherB:
    """Test cases for Cipher B indicator."""

    def test_cipher_b_initialization(self) -> None:
        """Test Cipher B initialization with default parameters."""
        cipher_b = CipherB()

        assert cipher_b.vwap_length == 20
        assert cipher_b.mf_length == 9
        assert cipher_b.signal_sensitivity == 1.0

    def test_cipher_b_with_sample_data(self) -> None:
        """Test Cipher B calculation with sample data."""
        # Create sample OHLCV data
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        rng = np.random.default_rng(43)
        data = pd.DataFrame(
            {
                "open": rng.uniform(45000, 55000, 100),
                "high": rng.uniform(46000, 56000, 100),
                "low": rng.uniform(44000, 54000, 100),
                "close": rng.uniform(45000, 55000, 100),
                "volume": rng.uniform(10, 100, 100),
            },
            index=dates,
        )

        cipher_b = CipherB()
        result = cipher_b.calculate(data)

        # Check that new columns are added
        assert "vwap" in result.columns
        assert "cipher_b_money_flow" in result.columns
        assert "cipher_b_wave" in result.columns
        assert "cipher_b_buy_signal" in result.columns


class TestVuManChuIndicators:
    """Test cases for the main indicator calculator."""

    def test_indicator_calculator_initialization(self) -> None:
        """Test indicator calculator initialization."""
        calc = VuManChuIndicators()

        assert calc.cipher_a is not None
        assert calc.cipher_b is not None

    def test_calculate_all_indicators(self) -> None:
        """Test calculation of all indicators."""
        # Create sample data
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        rng = np.random.default_rng(44)
        data = pd.DataFrame(
            {
                "open": rng.uniform(45000, 55000, 100),
                "high": rng.uniform(46000, 56000, 100),
                "low": rng.uniform(44000, 54000, 100),
                "close": rng.uniform(45000, 55000, 100),
                "volume": rng.uniform(10, 100, 100),
            },
            index=dates,
        )

        calc = VuManChuIndicators()
        result = calc.calculate_all(data)

        # Should have all indicator columns
        expected_columns = [
            "wt1",
            "wt2",
            "rsi",
            "cipher_a_signal",
            "cipher_a_confidence",
            "vwap",
            "cipher_b_money_flow",
            "cipher_b_wave",
            "cipher_b_buy_signal",
            "combined_signal",
            "combined_confidence",
        ]

        for col in expected_columns:
            assert col in result.columns

    def test_get_latest_state(self) -> None:
        """Test getting latest indicator state."""
        # Create sample data
        dates = pd.date_range("2024-01-01", periods=50, freq="1h")
        rng = np.random.default_rng(45)
        data = pd.DataFrame(
            {
                "open": rng.uniform(45000, 55000, 50),
                "high": rng.uniform(46000, 56000, 50),
                "low": rng.uniform(44000, 54000, 50),
                "close": rng.uniform(45000, 55000, 50),
                "volume": rng.uniform(10, 100, 50),
            },
            index=dates,
        )

        calc = VuManChuIndicators()
        result = calc.calculate_all(data)
        latest_state = calc.get_latest_state(result)

        # Should have latest values
        assert "rsi" in latest_state
        assert "vwap" in latest_state
        assert "cipher_a_signal" in latest_state
        assert isinstance(latest_state, dict)
