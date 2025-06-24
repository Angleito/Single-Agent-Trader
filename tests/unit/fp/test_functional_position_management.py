"""
Functional programming position management tests.

This module tests immutable position tracking, functional P&L calculations,
pure position state transitions, and comprehensive safety validations using
functional programming patterns.
"""

import tempfile
import unittest
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock, patch

import pytest
from hypothesis import given, strategies as st

# Functional programming imports
from bot.fp.types.portfolio import (
    Position,
    Portfolio,
    PortfolioMetrics,
    TradeResult,
    open_position,
    close_position,
    update_position_price,
    calculate_position_pnl,
    calculate_portfolio_pnl,
)
from bot.fp.types.effects import Result, Ok, Err, Maybe, Some, Nothing
from bot.fp.types.trading import Order, OrderStatus
from bot.fp.types.market import MarketSnapshot
from bot.fp.pure.paper_trading_calculations import (
    calculate_position_value,
    calculate_unrealized_pnl_simple as calculate_unrealized_pnl,
    calculate_margin_requirement_simple as calculate_margin_requirement,
    calculate_fees_simple as calculate_fees,
    validate_position_size_simple as validate_position_size,
    calculate_stop_loss_distance,
    calculate_take_profit_distance,
)

# Legacy compatibility imports for adapter testing
from bot.position_manager import (
    PositionManager,
    PositionManagerError,
    PositionStateError,
    PositionValidationError,
)


class TestImmutablePositionTypes:
    """Test immutable position data types and their properties."""
    
    def test_position_immutability(self):
        """Test Position type is immutable."""
        position = Position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("51000.0")
        )
        
        # Should be frozen/immutable
        with pytest.raises(AttributeError):
            position.size = Decimal("1.0")
        
        with pytest.raises(AttributeError):
            position.current_price = Decimal("52000.0")
    
    def test_position_pnl_calculation(self):
        """Test Position P&L calculation properties."""
        # Long position with profit
        long_position = Position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("55000.0")
        )
        
        assert long_position.unrealized_pnl == Decimal("5000.0")
        assert long_position.value == Decimal("55000.0")
        assert long_position.return_pct == Decimal("10.0")  # 5000/50000 * 100
        
        # Short position with profit
        short_position = Position(
            symbol="ETH-USD",
            side="SHORT",
            size=Decimal("10.0"),
            entry_price=Decimal("3000.0"),
            current_price=Decimal("2800.0")
        )
        
        assert short_position.unrealized_pnl == Decimal("2000.0")  # 10 * (3000 - 2800)
        assert short_position.value == Decimal("28000.0")
        assert short_position.return_pct == Decimal("6.67")  # Rounded to 2 decimals
    
    def test_position_validation(self):
        """Test Position validation rules."""
        # Valid position
        valid_position = Position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("51000.0")
        )
        assert valid_position.symbol == "BTC-USD"
        
        # Invalid side should raise error
        with pytest.raises(ValueError, match="Invalid side"):
            Position(
                symbol="BTC-USD",
                side="INVALID",
                size=Decimal("1.0"),
                entry_price=Decimal("50000.0"),
                current_price=Decimal("51000.0")
            )
        
        # Zero size should be valid for flat positions
        flat_position = Position(
            symbol="BTC-USD",
            side="FLAT",
            size=Decimal("0.0"),
            entry_price=None,
            current_price=Decimal("50000.0")
        )
        assert flat_position.side == "FLAT"
        assert flat_position.unrealized_pnl == Decimal("0.0")
    
    def test_portfolio_immutability(self):
        """Test Portfolio type is immutable."""
        position = Position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("51000.0")
        )
        
        portfolio = Portfolio(
            positions=(position,),
            cash_balance=Decimal("10000.0")
        )
        
        # Should be frozen/immutable
        with pytest.raises(AttributeError):
            portfolio.cash_balance = Decimal("20000.0")
        
        # Positions tuple should be immutable
        with pytest.raises(AttributeError):
            portfolio.positions = ()
    
    def test_portfolio_calculations(self):
        """Test Portfolio-level calculations."""
        btc_position = Position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("52000.0")
        )
        
        eth_position = Position(
            symbol="ETH-USD",
            side="SHORT",
            size=Decimal("10.0"),
            entry_price=Decimal("3000.0"),
            current_price=Decimal("2900.0")
        )
        
        portfolio = Portfolio(
            positions=(btc_position, eth_position),
            cash_balance=Decimal("5000.0")
        )
        
        # Total value: cash + position values + unrealized PnL
        # Cash: 5000
        # BTC value: 52000, PnL: +2000
        # ETH value: 29000, PnL: +1000
        # Total: 5000 + 52000 + 29000 = 86000
        assert portfolio.total_value == Decimal("86000.0")
        
        # Unrealized PnL: 2000 + 1000 = 3000
        assert portfolio.unrealized_pnl == Decimal("3000.0")


class TestFunctionalPositionOperations:
    """Test functional position operations and state transitions."""
    
    def test_open_position_functional(self):
        """Test opening position using functional approach."""
        portfolio = Portfolio(
            positions=(),
            cash_balance=Decimal("100000.0")
        )
        
        # Open a new position
        new_portfolio = open_position(
            portfolio,
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("2.0"),
            entry_price=Decimal("50000.0")
        )
        
        # Original portfolio should be unchanged
        assert len(portfolio.positions) == 0
        assert portfolio.cash_balance == Decimal("100000.0")
        
        # New portfolio should have the position
        assert len(new_portfolio.positions) == 1
        
        position = new_portfolio.positions[0]
        assert position.symbol == "BTC-USD"
        assert position.side == "LONG"
        assert position.size == Decimal("2.0")
        assert position.entry_price == Decimal("50000.0")
        
        # Cash should be reduced by position cost
        assert new_portfolio.cash_balance == Decimal("0.0")  # 100000 - (2 * 50000)
    
    def test_close_position_functional(self):
        """Test closing position using functional approach."""
        position = Position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("55000.0")
        )
        
        portfolio = Portfolio(
            positions=(position,),
            cash_balance=Decimal("0.0")
        )
        
        # Close the position
        new_portfolio, trade_result = close_position(
            portfolio,
            symbol="BTC-USD",
            exit_price=Decimal("55000.0"),
            exit_time=datetime.now(UTC)
        )
        
        # Original portfolio unchanged
        assert len(portfolio.positions) == 1
        
        # New portfolio should have no positions
        assert len(new_portfolio.positions) == 0
        
        # Cash should reflect position exit value + PnL
        # Exit value: 1 * 55000 = 55000
        # PnL: 5000
        # Total cash: 0 + 55000 + 5000 = 60000
        assert new_portfolio.cash_balance == Decimal("60000.0")
        
        # Trade result should capture the trade
        assert trade_result is not None
        assert trade_result.symbol == "BTC-USD"
        assert trade_result.side == "LONG"
        assert trade_result.entry_price == Decimal("50000.0")
        assert trade_result.exit_price == Decimal("55000.0")
        assert trade_result.pnl == Decimal("5000.0")
        assert trade_result.return_pct == Decimal("10.0")
    
    def test_update_position_price_functional(self):
        """Test updating position price functionally."""
        position = Position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("50000.0")
        )
        
        # Update price
        updated_position = update_position_price(position, Decimal("52000.0"))
        
        # Original position unchanged
        assert position.current_price == Decimal("50000.0")
        assert position.unrealized_pnl == Decimal("0.0")
        
        # Updated position has new price and PnL
        assert updated_position.current_price == Decimal("52000.0")
        assert updated_position.unrealized_pnl == Decimal("2000.0")
        
        # Other fields should remain the same
        assert updated_position.symbol == position.symbol
        assert updated_position.side == position.side
        assert updated_position.size == position.size
        assert updated_position.entry_price == position.entry_price
    
    def test_portfolio_price_updates_functional(self):
        """Test updating portfolio prices functionally."""
        btc_position = Position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("50000.0")
        )
        
        eth_position = Position(
            symbol="ETH-USD",
            side="SHORT",
            size=Decimal("10.0"),
            entry_price=Decimal("3000.0"),
            current_price=Decimal("3000.0")
        )
        
        portfolio = Portfolio(
            positions=(btc_position, eth_position),
            cash_balance=Decimal("10000.0")
        )
        
        # Update prices
        price_updates = {
            "BTC-USD": Decimal("52000.0"),
            "ETH-USD": Decimal("2900.0")
        }
        
        updated_portfolio = portfolio.update_prices(price_updates)
        
        # Original portfolio unchanged
        assert portfolio.positions[0].current_price == Decimal("50000.0")
        assert portfolio.positions[1].current_price == Decimal("3000.0")
        assert portfolio.unrealized_pnl == Decimal("0.0")
        
        # Updated portfolio has new prices and PnL
        updated_btc = updated_portfolio.positions[0]
        updated_eth = updated_portfolio.positions[1]
        
        assert updated_btc.current_price == Decimal("52000.0")
        assert updated_btc.unrealized_pnl == Decimal("2000.0")
        
        assert updated_eth.current_price == Decimal("2900.0")
        assert updated_eth.unrealized_pnl == Decimal("1000.0")  # 10 * (3000 - 2900)
        
        assert updated_portfolio.unrealized_pnl == Decimal("3000.0")


class TestPurePnLCalculations:
    """Test pure P&L calculation functions."""
    
    def test_calculate_position_pnl_long(self):
        """Test P&L calculation for long positions."""
        size = Decimal("2.0")
        entry_price = Decimal("50000.0")
        current_price = Decimal("52000.0")
        
        pnl = calculate_position_pnl("LONG", size, entry_price, current_price)
        
        # Long PnL: size * (current - entry) = 2 * (52000 - 50000) = 4000
        assert pnl == Decimal("4000.0")
    
    def test_calculate_position_pnl_short(self):
        """Test P&L calculation for short positions."""
        size = Decimal("10.0")
        entry_price = Decimal("3000.0")
        current_price = Decimal("2800.0")
        
        pnl = calculate_position_pnl("SHORT", size, entry_price, current_price)
        
        # Short PnL: size * (entry - current) = 10 * (3000 - 2800) = 2000
        assert pnl == Decimal("2000.0")
    
    def test_calculate_position_pnl_flat(self):
        """Test P&L calculation for flat positions."""
        pnl = calculate_position_pnl(
            "FLAT", Decimal("0.0"), Decimal("50000.0"), Decimal("52000.0")
        )
        
        assert pnl == Decimal("0.0")
    
    def test_calculate_portfolio_pnl(self):
        """Test portfolio-level P&L calculation."""
        positions = [
            Position(
                symbol="BTC-USD",
                side="LONG",
                size=Decimal("1.0"),
                entry_price=Decimal("50000.0"),
                current_price=Decimal("52000.0")
            ),
            Position(
                symbol="ETH-USD",
                side="SHORT",
                size=Decimal("5.0"),
                entry_price=Decimal("3000.0"),
                current_price=Decimal("2900.0")
            )
        ]
        
        total_pnl = calculate_portfolio_pnl(positions)
        
        # BTC PnL: 1 * (52000 - 50000) = 2000
        # ETH PnL: 5 * (3000 - 2900) = 500
        # Total: 2000 + 500 = 2500
        assert total_pnl == Decimal("2500.0")


class TestPurePaperTradingCalculations:
    """Test pure paper trading calculation functions."""
    
    def test_calculate_position_value(self):
        """Test position value calculation."""
        size = Decimal("2.5")
        price = Decimal("50000.0")
        
        value = calculate_position_value(size, price)
        assert value == Decimal("125000.0")
    
    def test_calculate_unrealized_pnl_long(self):
        """Test unrealized P&L calculation for long positions."""
        side = "LONG"
        size = Decimal("1.0")
        entry_price = Decimal("50000.0")
        current_price = Decimal("53000.0")
        
        pnl = calculate_unrealized_pnl(side, size, entry_price, current_price)
        assert pnl == Decimal("3000.0")
    
    def test_calculate_unrealized_pnl_short(self):
        """Test unrealized P&L calculation for short positions."""
        side = "SHORT"
        size = Decimal("2.0")
        entry_price = Decimal("50000.0")
        current_price = Decimal("48000.0")
        
        pnl = calculate_unrealized_pnl(side, size, entry_price, current_price)
        assert pnl == Decimal("4000.0")  # 2 * (50000 - 48000)
    
    def test_calculate_margin_requirement(self):
        """Test margin requirement calculation."""
        position_value = Decimal("100000.0")
        leverage = Decimal("5.0")
        
        margin = calculate_margin_requirement(position_value, leverage)
        assert margin == Decimal("20000.0")  # 100000 / 5
    
    def test_calculate_margin_requirement_no_leverage(self):
        """Test margin requirement with no leverage (1x)."""
        position_value = Decimal("50000.0")
        leverage = Decimal("1.0")
        
        margin = calculate_margin_requirement(position_value, leverage)
        assert margin == Decimal("50000.0")
    
    def test_calculate_fees(self):
        """Test fee calculation."""
        position_value = Decimal("100000.0")
        fee_rate = Decimal("0.001")  # 0.1%
        
        fees = calculate_fees(position_value, fee_rate)
        assert fees == Decimal("100.0")
    
    def test_validate_position_size_valid(self):
        """Test position size validation for valid sizes."""
        # Valid sizes
        valid_sizes = [
            Decimal("0.001"),
            Decimal("1.0"),
            Decimal("100.0"),
        ]
        
        for size in valid_sizes:
            result = validate_position_size(size)
            assert isinstance(result, Ok)
            assert result.unwrap() == size
    
    def test_validate_position_size_invalid(self):
        """Test position size validation for invalid sizes."""
        # Invalid sizes
        invalid_sizes = [
            Decimal("0.0"),  # Zero size
            Decimal("-1.0"),  # Negative size
        ]
        
        for size in invalid_sizes:
            result = validate_position_size(size)
            assert isinstance(result, Err)
            assert "Invalid position size" in result.error
    
    def test_calculate_stop_loss_distance(self):
        """Test stop loss distance calculation."""
        entry_price = Decimal("50000.0")
        stop_loss_pct = Decimal("2.0")  # 2%
        
        distance = calculate_stop_loss_distance(entry_price, stop_loss_pct)
        expected = Decimal("1000.0")  # 50000 * 0.02
        
        assert distance == expected
    
    def test_calculate_take_profit_distance(self):
        """Test take profit distance calculation."""
        entry_price = Decimal("50000.0")
        take_profit_pct = Decimal("4.0")  # 4%
        
        distance = calculate_take_profit_distance(entry_price, take_profit_pct)
        expected = Decimal("2000.0")  # 50000 * 0.04
        
        assert distance == expected


class TestFunctionalPositionValidation:
    """Test functional position validation and error handling."""
    
    def test_position_validation_with_result_types(self):
        """Test position validation using Result types."""
        # Valid position data
        valid_result = validate_position_data(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.0")
        )
        
        assert isinstance(valid_result, Ok)
        position_data = valid_result.unwrap()
        assert position_data["symbol"] == "BTC-USD"
        assert position_data["side"] == "LONG"
    
    def test_position_validation_invalid_symbol(self):
        """Test position validation with invalid symbol."""
        invalid_result = validate_position_data(
            symbol="",  # Empty symbol
            side="LONG",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.0")
        )
        
        assert isinstance(invalid_result, Err)
        assert "Symbol cannot be empty" in invalid_result.error
    
    def test_position_validation_invalid_side(self):
        """Test position validation with invalid side."""
        invalid_result = validate_position_data(
            symbol="BTC-USD",
            side="INVALID",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.0")
        )
        
        assert isinstance(invalid_result, Err)
        assert "Invalid side" in invalid_result.error
    
    def test_position_validation_invalid_size(self):
        """Test position validation with invalid size."""
        invalid_result = validate_position_data(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.0"),  # Zero size
            entry_price=Decimal("50000.0")
        )
        
        assert isinstance(invalid_result, Err)
        assert "Size must be positive" in invalid_result.error
    
    def test_position_validation_invalid_price(self):
        """Test position validation with invalid price."""
        invalid_result = validate_position_data(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("1.0"),
            entry_price=Decimal("0.0")  # Zero price
        )
        
        assert isinstance(invalid_result, Err)
        assert "Entry price must be positive" in invalid_result.error


class TestPortfolioMetrics:
    """Test portfolio metrics calculation and aggregation."""
    
    def test_portfolio_metrics_from_trades(self):
        """Test portfolio metrics calculation from trade results."""
        trades = (
            TradeResult(
                trade_id="1",
                symbol="BTC-USD",
                side="LONG",
                entry_price=Decimal("50000.0"),
                exit_price=Decimal("52000.0"),
                size=Decimal("1.0"),
                entry_time=datetime.now(UTC),
                exit_time=datetime.now(UTC)
            ),
            TradeResult(
                trade_id="2",
                symbol="ETH-USD",
                side="SHORT",
                entry_price=Decimal("3000.0"),
                exit_price=Decimal("2900.0"),
                size=Decimal("10.0"),
                entry_time=datetime.now(UTC),
                exit_time=datetime.now(UTC)
            ),
            TradeResult(
                trade_id="3",
                symbol="BTC-USD",
                side="LONG",
                entry_price=Decimal("52000.0"),
                exit_price=Decimal("51000.0"),
                size=Decimal("0.5"),
                entry_time=datetime.now(UTC),
                exit_time=datetime.now(UTC)
            )
        )
        
        metrics = PortfolioMetrics.from_trades(trades)
        
        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2  # First two trades are winners
        assert metrics.losing_trades == 1   # Last trade is a loser
        assert metrics.win_rate == 2/3  # 66.67%
        
        # Check P&L calculations
        # Trade 1: 1 * (52000 - 50000) = 2000
        # Trade 2: 10 * (3000 - 2900) = 1000  
        # Trade 3: 0.5 * (51000 - 52000) = -500
        # Total: 2000 + 1000 - 500 = 2500
        assert metrics.total_pnl == Decimal("2500.0")
        
        # Gross profit: 2000 + 1000 = 3000
        assert metrics.gross_profit == Decimal("3000.0")
        
        # Gross loss: 500
        assert metrics.gross_loss == Decimal("500.0")
        
        # Profit factor: 3000 / 500 = 6.0
        assert metrics.profit_factor == 6.0
    
    def test_portfolio_metrics_empty_trades(self):
        """Test portfolio metrics with no trades."""
        metrics = PortfolioMetrics.from_trades(())
        
        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.total_pnl == Decimal("0.0")
        assert metrics.profit_factor == 0.0
        assert metrics.sharpe_ratio == 0.0


class TestPropertyBasedPositionTests:
    """Property-based tests for position calculations using hypothesis."""
    
    @given(
        size=st.decimals(min_value=Decimal("0.001"), max_value=Decimal("1000")),
        entry_price=st.decimals(min_value=Decimal("1"), max_value=Decimal("100000")),
        current_price=st.decimals(min_value=Decimal("1"), max_value=Decimal("100000"))
    )
    def test_position_pnl_properties(self, size, entry_price, current_price):
        """Test position P&L calculation properties."""
        # Test long position
        long_pnl = calculate_position_pnl("LONG", size, entry_price, current_price)
        expected_long = size * (current_price - entry_price)
        assert long_pnl == expected_long
        
        # Test short position  
        short_pnl = calculate_position_pnl("SHORT", size, entry_price, current_price)
        expected_short = size * (entry_price - current_price)
        assert short_pnl == expected_short
        
        # Long and short should be opposites
        assert long_pnl == -short_pnl
    
    @given(
        size=st.decimals(min_value=Decimal("0.001"), max_value=Decimal("1000")),
        price=st.decimals(min_value=Decimal("1"), max_value=Decimal("100000"))
    )
    def test_position_value_properties(self, size, price):
        """Test position value calculation properties."""
        value = calculate_position_value(size, price)
        
        # Value should be positive
        assert value > 0
        
        # Value should be proportional to size and price
        assert value == size * price
        
        # Double size should double value
        double_value = calculate_position_value(size * 2, price)
        assert double_value == value * 2
    
    @given(
        position_value=st.decimals(min_value=Decimal("1000"), max_value=Decimal("1000000")),
        leverage=st.decimals(min_value=Decimal("1"), max_value=Decimal("100"))
    )
    def test_margin_requirement_properties(self, position_value, leverage):
        """Test margin requirement calculation properties."""
        margin = calculate_margin_requirement(position_value, leverage)
        
        # Margin should be positive
        assert margin > 0
        
        # Margin should be less than or equal to position value
        assert margin <= position_value
        
        # Higher leverage should require less margin
        if leverage > 1:
            lower_leverage_margin = calculate_margin_requirement(position_value, leverage / 2)
            assert margin < lower_leverage_margin


class TestFunctionalPositionPersistence:
    """Test functional position state persistence and reconstruction."""
    
    def test_position_serialization(self):
        """Test position serialization to/from dict."""
        position = Position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("1.5"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("52000.0")
        )
        
        # Serialize to dict
        position_dict = position.to_dict()
        
        expected_dict = {
            "symbol": "BTC-USD",
            "side": "LONG",
            "size": "1.5",
            "entry_price": "50000.0",
            "current_price": "52000.0"
        }
        
        assert position_dict == expected_dict
        
        # Deserialize from dict
        reconstructed = Position.from_dict(position_dict)
        
        assert reconstructed.symbol == position.symbol
        assert reconstructed.side == position.side
        assert reconstructed.size == position.size
        assert reconstructed.entry_price == position.entry_price
        assert reconstructed.current_price == position.current_price
        assert reconstructed.unrealized_pnl == position.unrealized_pnl
    
    def test_portfolio_serialization(self):
        """Test portfolio serialization to/from dict."""
        positions = (
            Position(
                symbol="BTC-USD",
                side="LONG", 
                size=Decimal("1.0"),
                entry_price=Decimal("50000.0"),
                current_price=Decimal("51000.0")
            ),
            Position(
                symbol="ETH-USD",
                side="SHORT",
                size=Decimal("10.0"),
                entry_price=Decimal("3000.0"),
                current_price=Decimal("2950.0")
            )
        )
        
        portfolio = Portfolio(
            positions=positions,
            cash_balance=Decimal("25000.0")
        )
        
        # Serialize to dict
        portfolio_dict = portfolio.to_dict()
        
        assert portfolio_dict["cash_balance"] == "25000.0"
        assert len(portfolio_dict["positions"]) == 2
        
        # Deserialize from dict
        reconstructed = Portfolio.from_dict(portfolio_dict)
        
        assert reconstructed.cash_balance == portfolio.cash_balance
        assert len(reconstructed.positions) == len(portfolio.positions)
        
        for orig, recon in zip(portfolio.positions, reconstructed.positions):
            assert orig.symbol == recon.symbol
            assert orig.side == recon.side
            assert orig.size == recon.size
            assert orig.entry_price == recon.entry_price


class TestAdapterCompatibility:
    """Test compatibility between functional and legacy position management."""
    
    def test_fp_to_legacy_position_conversion(self):
        """Test converting FP Position to legacy format."""
        fp_position = Position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("52000.0")
        )
        
        # Convert to legacy format
        legacy_dict = fp_position.to_legacy_format()
        
        expected_keys = {
            "symbol", "side", "size", "entry_price", 
            "current_price", "unrealized_pnl", "value"
        }
        
        assert set(legacy_dict.keys()) == expected_keys
        assert legacy_dict["symbol"] == "BTC-USD"
        assert legacy_dict["side"] == "LONG"
        assert legacy_dict["size"] == 1.0  # Float for legacy compatibility
        assert legacy_dict["unrealized_pnl"] == 2000.0
    
    def test_legacy_to_fp_position_conversion(self):
        """Test converting legacy position data to FP Position."""
        legacy_data = {
            "symbol": "ETH-USD",
            "side": "SHORT",
            "size": 10.0,
            "entry_price": 3000.0,
            "current_price": 2900.0
        }
        
        # Convert to FP Position
        fp_position = Position.from_legacy_format(legacy_data)
        
        assert fp_position.symbol == "ETH-USD"
        assert fp_position.side == "SHORT"
        assert fp_position.size == Decimal("10.0")
        assert fp_position.entry_price == Decimal("3000.0")
        assert fp_position.current_price == Decimal("2900.0")
        assert fp_position.unrealized_pnl == Decimal("1000.0")


# Helper functions for testing
def validate_position_data(symbol: str, side: str, size: Decimal, entry_price: Decimal) -> Result[Dict, str]:
    """Validate position data and return Result type."""
    if not symbol:
        return Err("Symbol cannot be empty")
    
    if side not in ["LONG", "SHORT", "FLAT"]:
        return Err(f"Invalid side: {side}")
    
    if size <= 0 and side != "FLAT":
        return Err("Size must be positive for non-flat positions")
    
    if entry_price <= 0 and side != "FLAT":
        return Err("Entry price must be positive for non-flat positions")
    
    return Ok({
        "symbol": symbol,
        "side": side,
        "size": size,
        "entry_price": entry_price
    })


if __name__ == "__main__":
    pytest.main([__file__, "-v"])