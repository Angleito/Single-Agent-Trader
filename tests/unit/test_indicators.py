"""Unit tests for technical indicators."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from bot.indicators.vumanchu import CipherA, CipherB, VuManChuIndicators


class TestCipherA:
    """Test cases for Cipher A indicator."""

    def test_cipher_a_initialization(self) -> None:
        """Test Cipher A initialization with default parameters."""
        cipher_a = CipherA()

        # Test default parameters - these may vary based on actual implementation
        assert hasattr(cipher_a, "wt_ma_length")
        assert hasattr(cipher_a, "wt_signal_length")
        assert cipher_a.wt_ma_length > 0
        assert cipher_a.wt_signal_length > 0

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
        assert "ema_fast" in result.columns
        assert "ema_slow" in result.columns
        assert "rsi" in result.columns
        assert "trend_dot" in result.columns
        assert "cipher_a_signal" in result.columns

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

        assert cipher_b.vwap_length == 14
        assert cipher_b.mfi_length == 14
        assert cipher_b.wave_length == 10
        assert cipher_b.wave_mult == 3.7

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
        assert "money_flow" in result.columns
        assert "wave" in result.columns
        assert "cipher_b_signal" in result.columns


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
            "ema_fast",
            "ema_slow",
            "rsi",
            "trend_dot",
            "cipher_a_signal",
            "vwap",
            "money_flow",
            "wave",
            "cipher_b_signal",
            "ema_200",
            "atr",
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
        assert "close" in latest_state
        assert "volume" in latest_state
        assert isinstance(latest_state, dict)
