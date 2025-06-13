"""Unit tests for technical indicators."""

import numpy as np
import pandas as pd

from bot.indicators.vumanchu import CipherA, CipherB, VuManChuIndicators


class TestCipherA:
    """Test cases for Cipher A indicator."""

    def test_cipher_a_initialization(self):
        """Test Cipher A initialization with default parameters."""
        cipher_a = CipherA()

        assert cipher_a.ema1_length == 9
        assert cipher_a.ema2_length == 21
        assert cipher_a.rsi_length == 14
        assert cipher_a.rsi_overbought == 80.0
        assert cipher_a.rsi_oversold == 20.0

    def test_cipher_a_with_sample_data(self):
        """Test Cipher A calculation with sample data."""
        # Create sample OHLCV data
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        data = pd.DataFrame(
            {
                "open": np.random.uniform(45000, 55000, 100),
                "high": np.random.uniform(46000, 56000, 100),
                "low": np.random.uniform(44000, 54000, 100),
                "close": np.random.uniform(45000, 55000, 100),
                "volume": np.random.uniform(10, 100, 100),
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

    def test_cipher_a_insufficient_data(self):
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

    def test_cipher_b_initialization(self):
        """Test Cipher B initialization with default parameters."""
        cipher_b = CipherB()

        assert cipher_b.vwap_length == 14
        assert cipher_b.mfi_length == 14
        assert cipher_b.wave_length == 10
        assert cipher_b.wave_mult == 3.7

    def test_cipher_b_with_sample_data(self):
        """Test Cipher B calculation with sample data."""
        # Create sample OHLCV data
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        data = pd.DataFrame(
            {
                "open": np.random.uniform(45000, 55000, 100),
                "high": np.random.uniform(46000, 56000, 100),
                "low": np.random.uniform(44000, 54000, 100),
                "close": np.random.uniform(45000, 55000, 100),
                "volume": np.random.uniform(10, 100, 100),
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

    def test_indicator_calculator_initialization(self):
        """Test indicator calculator initialization."""
        calc = VuManChuIndicators()

        assert calc.cipher_a is not None
        assert calc.cipher_b is not None

    def test_calculate_all_indicators(self):
        """Test calculation of all indicators."""
        # Create sample data
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        data = pd.DataFrame(
            {
                "open": np.random.uniform(45000, 55000, 100),
                "high": np.random.uniform(46000, 56000, 100),
                "low": np.random.uniform(44000, 54000, 100),
                "close": np.random.uniform(45000, 55000, 100),
                "volume": np.random.uniform(10, 100, 100),
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

    def test_get_latest_state(self):
        """Test getting latest indicator state."""
        # Create sample data
        dates = pd.date_range("2024-01-01", periods=50, freq="1h")
        data = pd.DataFrame(
            {
                "open": np.random.uniform(45000, 55000, 50),
                "high": np.random.uniform(46000, 56000, 50),
                "low": np.random.uniform(44000, 54000, 50),
                "close": np.random.uniform(45000, 55000, 50),
                "volume": np.random.uniform(10, 100, 50),
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
