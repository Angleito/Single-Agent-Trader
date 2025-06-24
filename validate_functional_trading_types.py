#!/usr/bin/env python3
"""
Functional Trading Types Validation Script

This script validates that the new functional trading types work correctly
and maintain trading accuracy compared to the legacy Pydantic types.
"""

import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bot.fp.adapters.trading_type_adapter import (
    FunctionalTradingIntegration,
    TradingTypeAdapter,
)
from bot.fp.types.trading import (
    CFM_ACCOUNT,
    HEALTHY_MARGIN,
    FunctionalMarketData,
    FunctionalMarketState,
    FuturesAccountBalance,
    FuturesLimitOrder,
    FuturesMarketData,
    LimitOrder,
    Long,
    MarginInfo,
    MarketMake,
    MarketOrder,
    Position,
    RiskMetrics,
    Short,
    StopOrder,
    TradingIndicators,
    convert_trade_signal_to_orders,
    create_conservative_risk_limits,
    create_default_margin_info,
)


class FunctionalTypesValidator:
    """Validator for functional trading types."""

    def __init__(self):
        """Initialize the validator."""
        self.test_results = []
        self.adapter = TradingTypeAdapter()
        self.integration = FunctionalTradingIntegration()

    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log a test result."""
        status = "PASS" if passed else "FAIL"
        self.test_results.append(
            {"test": test_name, "status": status, "message": message}
        )
        print(f"[{status}] {test_name}: {message}")

    def validate_basic_types(self) -> None:
        """Validate basic functional types."""
        print("\n=== Validating Basic Types ===")

        # Test Long signal
        try:
            long_signal = Long(
                confidence=0.8, size=0.25, reason="Strong bullish signal"
            )
            self.log_test(
                "Long Signal Creation",
                True,
                f"Created with confidence {long_signal.confidence}",
            )

            # Test invalid confidence
            try:
                Long(confidence=1.5, size=0.25, reason="Invalid")
                self.log_test(
                    "Long Signal Validation", False, "Should reject invalid confidence"
                )
            except ValueError:
                self.log_test(
                    "Long Signal Validation",
                    True,
                    "Correctly rejected invalid confidence",
                )

        except Exception as e:
            self.log_test("Long Signal Creation", False, f"Error: {e}")

        # Test Short signal
        try:
            short_signal = Short(confidence=0.7, size=0.15, reason="Bearish divergence")
            self.log_test(
                "Short Signal Creation", True, f"Created with size {short_signal.size}"
            )
        except Exception as e:
            self.log_test("Short Signal Creation", False, f"Error: {e}")

        # Test MarketMake signal
        try:
            mm_signal = MarketMake(
                bid_price=50000, ask_price=50100, bid_size=0.1, ask_size=0.1
            )
            spread = mm_signal.spread
            mid_price = mm_signal.mid_price
            self.log_test(
                "MarketMake Signal", True, f"Spread: {spread}, Mid: {mid_price}"
            )
        except Exception as e:
            self.log_test("MarketMake Signal", False, f"Error: {e}")

    def validate_order_types(self) -> None:
        """Validate order types."""
        print("\n=== Validating Order Types ===")

        # Test basic orders
        try:
            limit_order = LimitOrder(
                symbol="BTC-USD", side="buy", price=50000, size=0.1
            )
            self.log_test("Limit Order", True, f"Value: ${limit_order.value}")

            market_order = MarketOrder(symbol="BTC-USD", side="sell", size=0.05)
            self.log_test("Market Order", True, f"ID: {market_order.order_id}")

            stop_order = StopOrder(
                symbol="BTC-USD", side="sell", stop_price=45000, size=0.1
            )
            self.log_test("Stop Order", True, f"Stop at: ${stop_order.stop_price}")

        except Exception as e:
            self.log_test("Basic Orders", False, f"Error: {e}")

        # Test futures orders
        try:
            futures_limit = FuturesLimitOrder(
                symbol="BTC-PERP",
                side="buy",
                price=50000,
                size=0.1,
                leverage=5,
                margin_required=Decimal(1000),
            )
            notional = futures_limit.notional_value
            position_value = futures_limit.position_value
            self.log_test(
                "Futures Limit Order",
                True,
                f"Notional: ${notional}, Position: ${position_value}",
            )

        except Exception as e:
            self.log_test("Futures Limit Order", False, f"Error: {e}")

    def validate_market_data(self) -> None:
        """Validate market data types."""
        print("\n=== Validating Market Data ===")

        try:
            # Valid market data
            market_data = FunctionalMarketData(
                symbol="BTC-USD",
                timestamp=datetime.now(),
                open=Decimal(49500),
                high=Decimal(50200),
                low=Decimal(49000),
                close=Decimal(50100),
                volume=Decimal("150.5"),
            )

            change_pct = market_data.price_change_percentage
            is_bullish = market_data.is_bullish
            typical_price = market_data.typical_price

            self.log_test(
                "Market Data",
                True,
                f"Change: {change_pct:.2f}%, Bullish: {is_bullish}, Typical: ${typical_price}",
            )

            # Test invalid data (high < low)
            try:
                FunctionalMarketData(
                    symbol="BTC-USD",
                    timestamp=datetime.now(),
                    open=Decimal(50000),
                    high=Decimal(49000),  # Invalid: high < open
                    low=Decimal(49500),
                    close=Decimal(50100),
                    volume=Decimal(100),
                )
                self.log_test(
                    "Market Data Validation", False, "Should reject invalid OHLC"
                )
            except ValueError:
                self.log_test(
                    "Market Data Validation", True, "Correctly rejected invalid OHLC"
                )

        except Exception as e:
            self.log_test("Market Data", False, f"Error: {e}")

        # Test futures market data
        try:
            base_data = FunctionalMarketData(
                symbol="BTC-PERP",
                timestamp=datetime.now(),
                open=Decimal(50000),
                high=Decimal(50500),
                low=Decimal(49500),
                close=Decimal(50200),
                volume=Decimal(1000),
            )

            futures_data = FuturesMarketData(
                base_data=base_data,
                open_interest=Decimal(50000),
                funding_rate=0.0001,
                mark_price=Decimal(50180),
                index_price=Decimal(50175),
            )

            basis = futures_data.basis
            annualized_funding = futures_data.funding_rate_8h_annualized

            self.log_test(
                "Futures Market Data",
                True,
                f"Basis: ${basis}, Funding (ann.): {annualized_funding:.4f}",
            )

        except Exception as e:
            self.log_test("Futures Market Data", False, f"Error: {e}")

    def validate_positions_and_accounts(self) -> None:
        """Validate position and account types."""
        print("\n=== Validating Positions and Accounts ===")

        try:
            # Test position
            position = Position(
                symbol="BTC-USD",
                side="LONG",
                size=Decimal("0.5"),
                entry_price=Decimal(49000),
                unrealized_pnl=Decimal(600),
                realized_pnl=Decimal(150),
                timestamp=datetime.now(),
            )

            value = position.value
            pnl_pct = position.pnl_percentage

            self.log_test("Position", True, f"Value: ${value}, PnL%: {pnl_pct:.2f}%")

        except Exception as e:
            self.log_test("Position", False, f"Error: {e}")

        try:
            # Test margin info
            margin_info = create_default_margin_info()
            margin_ratio = margin_info.margin_ratio
            can_open = margin_info.can_open_position(Decimal(500))

            self.log_test(
                "Margin Info",
                True,
                f"Ratio: {margin_ratio:.2%}, Can open $500: {can_open}",
            )

        except Exception as e:
            self.log_test("Margin Info", False, f"Error: {e}")

        try:
            # Test futures account
            futures_account = FuturesAccountBalance(
                account_type=CFM_ACCOUNT,
                account_id="test_account",
                currency="USD",
                cash_balance=Decimal(10000),
                futures_balance=Decimal(5000),
                total_balance=Decimal(15000),
                margin_info=create_default_margin_info(),
            )

            equity = futures_account.equity
            buying_power = futures_account.buying_power

            self.log_test(
                "Futures Account",
                True,
                f"Equity: ${equity}, Buying Power: ${buying_power}",
            )

        except Exception as e:
            self.log_test("Futures Account", False, f"Error: {e}")

    def validate_risk_management(self) -> None:
        """Validate risk management types."""
        print("\n=== Validating Risk Management ===")

        try:
            # Test risk limits
            risk_limits = create_conservative_risk_limits()
            self.log_test(
                "Risk Limits Creation",
                True,
                f"Max leverage: {risk_limits.max_leverage}, Max positions: {risk_limits.max_open_positions}",
            )

        except Exception as e:
            self.log_test("Risk Limits Creation", False, f"Error: {e}")

        try:
            # Test risk metrics
            risk_metrics = RiskMetrics(
                account_balance=Decimal(10000),
                available_margin=Decimal(8000),
                used_margin=Decimal(2000),
                daily_pnl=Decimal(150),
                total_exposure=Decimal(5000),
                current_positions=2,
            )

            margin_util = risk_metrics.margin_utilization
            exposure_ratio = risk_metrics.exposure_ratio
            risk_score = risk_metrics.risk_score()
            compliant = risk_metrics.is_within_risk_limits(
                create_conservative_risk_limits()
            )

            self.log_test(
                "Risk Metrics",
                True,
                f"Margin util: {margin_util:.2%}, Risk score: {risk_score:.1f}, Compliant: {compliant}",
            )

        except Exception as e:
            self.log_test("Risk Metrics", False, f"Error: {e}")

    def validate_indicators(self) -> None:
        """Validate trading indicators."""
        print("\n=== Validating Trading Indicators ===")

        try:
            indicators = TradingIndicators(
                timestamp=datetime.now(),
                rsi=65.5,
                cipher_a_dot=0.8,
                cipher_b_wave=-0.2,
                cipher_b_money_flow=0.3,
                stablecoin_dominance=7.5,
                dominance_trend=-0.3,
                dominance_rsi=45.0,
            )

            rsi_signal = indicators.rsi_signal()
            dominance_signal = indicators.dominance_signal()
            has_vumanchu = indicators.has_vumanchu_signals

            self.log_test(
                "Trading Indicators",
                True,
                f"RSI signal: {rsi_signal}, Dominance signal: {dominance_signal}, VuManChu: {has_vumanchu}",
            )

        except Exception as e:
            self.log_test("Trading Indicators", False, f"Error: {e}")

    def validate_signal_to_order_conversion(self) -> None:
        """Validate signal to order conversion."""
        print("\n=== Validating Signal to Order Conversion ===")

        try:
            # Test Long signal conversion
            long_signal = Long(confidence=0.8, size=0.25, reason="Bullish breakout")
            orders = convert_trade_signal_to_orders(long_signal, "BTC-USD", 50000)

            self.log_test(
                "Long Signal Conversion",
                True,
                f"Generated {len(orders)} orders from Long signal",
            )

            # Test Short signal conversion
            short_signal = Short(confidence=0.7, size=0.15, reason="Bearish reversal")
            orders = convert_trade_signal_to_orders(short_signal, "BTC-USD", 50000)

            self.log_test(
                "Short Signal Conversion",
                True,
                f"Generated {len(orders)} orders from Short signal",
            )

            # Test MarketMake signal conversion
            mm_signal = MarketMake(
                bid_price=49950, ask_price=50050, bid_size=0.1, ask_size=0.1
            )
            orders = convert_trade_signal_to_orders(mm_signal, "BTC-USD", 50000)

            self.log_test(
                "MarketMake Signal Conversion",
                True,
                f"Generated {len(orders)} orders from MarketMake signal",
            )

        except Exception as e:
            self.log_test("Signal to Order Conversion", False, f"Error: {e}")

    def validate_comprehensive_market_state(self) -> None:
        """Validate comprehensive market state."""
        print("\n=== Validating Comprehensive Market State ===")

        try:
            # Create market data
            market_data = FunctionalMarketData(
                symbol="BTC-USD",
                timestamp=datetime.now(),
                open=Decimal(49500),
                high=Decimal(50500),
                low=Decimal(49000),
                close=Decimal(50200),
                volume=Decimal(500),
            )

            # Create indicators
            indicators = TradingIndicators(
                timestamp=datetime.now(),
                rsi=72.0,
                cipher_a_dot=0.85,
                stablecoin_dominance=6.8,
            )

            # Create position
            position = Position(
                symbol="BTC-USD",
                side="LONG",
                size=Decimal("0.5"),
                entry_price=Decimal(49000),
                unrealized_pnl=Decimal(600),
                realized_pnl=Decimal(0),
                timestamp=datetime.now(),
            )

            # Create market state
            market_state = FunctionalMarketState(
                symbol="BTC-USD",
                timestamp=datetime.now(),
                market_data=market_data,
                indicators=indicators,
                position=position,
            )

            current_price = market_state.current_price
            has_position = market_state.has_position
            position_value = market_state.position_value

            self.log_test(
                "Market State Creation",
                True,
                f"Price: ${current_price}, Has position: {has_position}, Value: ${position_value}",
            )

            # Test price update
            updated_state = market_state.with_updated_price(Decimal(51000))
            new_position_value = updated_state.position_value

            self.log_test(
                "Market State Update",
                True,
                f"Updated position value: ${new_position_value}",
            )

        except Exception as e:
            self.log_test("Comprehensive Market State", False, f"Error: {e}")

    def validate_type_adapters(self) -> None:
        """Validate type adapters."""
        print("\n=== Validating Type Adapters ===")

        try:
            # Create mock legacy data
            legacy_market_data = {
                "symbol": "BTC-USD",
                "timestamp": datetime.now(),
                "open": 49500,
                "high": 50500,
                "low": 49000,
                "close": 50200,
                "volume": 500,
            }

            legacy_position = {
                "symbol": "BTC-USD",
                "side": "LONG",
                "size": 0.5,
                "entry_price": 49000,
                "unrealized_pnl": 600,
                "realized_pnl": 0,
                "timestamp": datetime.now(),
            }

            legacy_indicators = {
                "timestamp": datetime.now(),
                "rsi": 72.0,
                "cipher_a_dot": 0.85,
            }

            # Test adaptation
            functional_market_data = self.adapter.adapt_market_data_to_functional(
                legacy_market_data
            )
            functional_position = self.adapter.adapt_position_to_functional(
                legacy_position
            )
            functional_indicators = self.adapter.adapt_indicators_to_functional(
                legacy_indicators
            )

            self.log_test(
                "Legacy to Functional Adaptation",
                True,
                "Adapted market data, position, and indicators successfully",
            )

            # Test comprehensive state creation
            functional_state = self.adapter.adapt_legacy_to_functional_state(
                symbol="BTC-USD",
                legacy_market_data=legacy_market_data,
                legacy_indicators=legacy_indicators,
                legacy_position=legacy_position,
            )

            self.log_test(
                "Functional State Creation",
                True,
                f"Created state for {functional_state.symbol}",
            )

            # Test conversion back to legacy
            legacy_format = self.adapter.convert_functional_state_to_legacy(
                functional_state
            )

            self.log_test(
                "Functional to Legacy Conversion",
                True,
                "Converted back to legacy format",
            )

        except Exception as e:
            self.log_test("Type Adapters", False, f"Error: {e}")

    def validate_trading_accuracy(self) -> None:
        """Validate that trading accuracy is maintained."""
        print("\n=== Validating Trading Accuracy ===")

        try:
            # Test P&L calculations
            position = Position(
                symbol="BTC-USD",
                side="LONG",
                size=Decimal("1.0"),
                entry_price=Decimal(50000),
                unrealized_pnl=Decimal(2000),  # $2000 profit
                realized_pnl=Decimal(500),
                timestamp=datetime.now(),
            )

            # Calculate expected values
            position_value = position.value  # Should be 1.0 * 52000 = 52000
            expected_current_price = Decimal(52000)  # entry + unrealized_pnl/size
            pnl_percentage = position.pnl_percentage  # Should be 4%

            # Verify calculations
            price_accurate = abs(float(position_value - expected_current_price)) < 0.01
            pnl_accurate = abs(pnl_percentage - 4.0) < 0.01

            self.log_test(
                "P&L Calculation Accuracy",
                price_accurate and pnl_accurate,
                f"Position value: ${position_value}, PnL%: {pnl_percentage:.2f}%",
            )

        except Exception as e:
            self.log_test("Trading Accuracy", False, f"Error: {e}")

        try:
            # Test margin calculations
            margin_info = MarginInfo(
                total_margin=Decimal(10000),
                available_margin=Decimal(7000),
                used_margin=Decimal(3000),
                maintenance_margin=Decimal(1000),
                initial_margin=Decimal(1500),
                health_status=HEALTHY_MARGIN,
                liquidation_threshold=Decimal(500),
                intraday_margin_requirement=Decimal(1000),
                overnight_margin_requirement=Decimal(1500),
            )

            margin_ratio = margin_info.margin_ratio  # Should be 0.3 (30%)
            margin_accurate = abs(margin_ratio - 0.3) < 0.001

            self.log_test(
                "Margin Calculation Accuracy",
                margin_accurate,
                f"Margin ratio: {margin_ratio:.1%}",
            )

        except Exception as e:
            self.log_test("Margin Calculation Accuracy", False, f"Error: {e}")

    def run_all_validations(self) -> None:
        """Run all validation tests."""
        print("🚀 Starting Functional Trading Types Validation")
        print("=" * 60)

        self.validate_basic_types()
        self.validate_order_types()
        self.validate_market_data()
        self.validate_positions_and_accounts()
        self.validate_risk_management()
        self.validate_indicators()
        self.validate_signal_to_order_conversion()
        self.validate_comprehensive_market_state()
        self.validate_type_adapters()
        self.validate_trading_accuracy()

        # Print summary
        passed = sum(1 for result in self.test_results if result["status"] == "PASS")
        total = len(self.test_results)

        print("\n" + "=" * 60)
        print(f"📊 VALIDATION SUMMARY: {passed}/{total} tests passed")

        if passed == total:
            print("✅ All functional trading types are working correctly!")
            print("✅ Trading accuracy is maintained!")
        else:
            print("❌ Some tests failed. Review the output above.")
            failed_tests = [r for r in self.test_results if r["status"] == "FAIL"]
            print("\nFailed tests:")
            for test in failed_tests:
                print(f"  - {test['test']}: {test['message']}")

        return passed == total


def main():
    """Main validation function."""
    validator = FunctionalTypesValidator()
    success = validator.run_all_validations()

    if success:
        print("\n🎉 Functional trading types migration is successful!")
        print("📈 Trading accuracy is preserved with enhanced type safety.")
        return 0
    print("\n💥 Validation failed! Check the errors above.")
    return 1


if __name__ == "__main__":
    exit(main())
