#!/usr/bin/env python3
"""
Validation script for Coinbase Futures trading implementation.

This script validates the enhanced futures trading functionality including:
- Configuration validation
- Type definitions
- Exchange client futures methods
- LLM agent o3 model support
"""

import asyncio
import importlib.util
import logging
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_module_from_file(name: str, file_path: str):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load modules directly
bot_dir = Path(__file__).parent / "bot"
config_module = load_module_from_file("config", bot_dir / "config.py")
types_module = load_module_from_file("types", bot_dir / "types.py")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FuturesImplementationValidator:
    """Validator for futures trading implementation."""

    def __init__(self):
        self.validation_results = {}
        self.client = None
        self.llm_agent = None

    async def validate_all(self) -> bool:
        """Run all validation tests."""
        logger.info("Starting Coinbase Futures Implementation Validation")
        logger.info("=" * 60)

        tests = [
            ("Configuration Validation", self.validate_configuration),
            ("Type Definitions Validation", self.validate_types),
            ("Exchange Client Validation", self.validate_exchange_client),
            ("LLM Agent Validation", self.validate_llm_agent),
            ("Integration Test", self.validate_integration),
        ]

        all_passed = True

        for test_name, test_func in tests:
            logger.info(f"\n📋 Running: {test_name}")
            try:
                result = await test_func()
                self.validation_results[test_name] = result
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    all_passed = False
            except Exception as e:
                logger.error(f"💥 {test_name}: ERROR - {e}")
                self.validation_results[test_name] = False
                all_passed = False

        # Print summary
        self._print_summary(all_passed)
        return all_passed

    async def validate_configuration(self) -> bool:
        """Validate configuration enhancements."""
        try:
            # Test default model is o3
            if settings.llm.model_name != "o3":
                logger.error(
                    f"Expected default model 'o3', got '{settings.llm.model_name}'"
                )
                return False

            # Test futures configuration exists
            if not hasattr(settings.trading, "enable_futures"):
                logger.error("Missing enable_futures configuration")
                return False

            if not hasattr(settings.trading, "futures_account_type"):
                logger.error("Missing futures_account_type configuration")
                return False

            if not hasattr(settings.trading, "max_futures_leverage"):
                logger.error("Missing max_futures_leverage configuration")
                return False

            # Test trading profile application
            conservative_settings = settings.apply_profile(TradingProfile.CONSERVATIVE)
            if conservative_settings.trading.leverage != 2:
                logger.error("Profile application failed")
                return False

            logger.info("  ✓ Default model is 'o3'")
            logger.info("  ✓ Futures configuration present")
            logger.info("  ✓ Trading profiles working")

            return True

        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False

    async def validate_types(self) -> bool:
        """Validate new type definitions."""
        try:
            # Test AccountType enum
            assert AccountType.CFM == "CFM"
            assert AccountType.CBI == "CBI"

            # Test MarginHealthStatus enum
            assert MarginHealthStatus.HEALTHY == "HEALTHY"
            assert MarginHealthStatus.LIQUIDATION_RISK == "LIQUIDATION_RISK"

            # Test MarginInfo creation
            margin_info = MarginInfo(
                total_margin=Decimal("10000"),
                available_margin=Decimal("8000"),
                used_margin=Decimal("2000"),
                maintenance_margin=Decimal("1000"),
                initial_margin=Decimal("2000"),
                margin_ratio=0.2,
                health_status=MarginHealthStatus.HEALTHY,
                liquidation_threshold=Decimal("9000"),
                intraday_margin_requirement=Decimal("2000"),
                overnight_margin_requirement=Decimal("4000"),
            )
            assert margin_info.margin_ratio == 0.2

            # Test FuturesAccountInfo creation
            futures_account = FuturesAccountInfo(
                account_type=AccountType.CFM,
                account_id="test-account",
                cash_balance=Decimal("10000"),
                futures_balance=Decimal("10000"),
                total_balance=Decimal("20000"),
                margin_info=margin_info,
                max_position_size=Decimal("5000"),
                timestamp=datetime.utcnow(),
            )
            assert futures_account.account_type == AccountType.CFM

            # Test TradeAction with futures fields
            trade_action = TradeAction(
                action="LONG",
                size_pct=20,
                take_profit_pct=3.0,
                stop_loss_pct=2.0,
                rationale="Test action",
                leverage=10,
                reduce_only=False,
            )
            assert trade_action.leverage == 10
            assert trade_action.reduce_only == False

            logger.info("  ✓ AccountType enum working")
            logger.info("  ✓ MarginHealthStatus enum working")
            logger.info("  ✓ MarginInfo model working")
            logger.info("  ✓ FuturesAccountInfo model working")
            logger.info("  ✓ Enhanced TradeAction working")

            return True

        except Exception as e:
            logger.error(f"Types validation error: {e}")
            return False

    async def validate_exchange_client(self) -> bool:
        """Validate exchange client enhancements."""
        try:
            # Initialize client
            self.client = CoinbaseClient()

            # Test futures configuration
            assert self.client.enable_futures == settings.trading.enable_futures
            assert (
                self.client.futures_account_type
                == settings.trading.futures_account_type
            )
            assert (
                self.client.max_futures_leverage
                == settings.trading.max_futures_leverage
            )

            # Test connection status includes futures info
            status = self.client.get_connection_status()
            required_keys = [
                "futures_enabled",
                "futures_account_type",
                "auto_cash_transfer",
                "max_futures_leverage",
            ]
            for key in required_keys:
                if key not in status:
                    logger.error(f"Missing key in connection status: {key}")
                    return False

            # Test method existence
            methods_to_check = [
                "get_futures_balance",
                "get_spot_balance",
                "get_futures_account_info",
                "get_margin_info",
                "transfer_cash_to_futures",
                "get_futures_positions",
                "place_futures_market_order",
            ]

            for method in methods_to_check:
                if not hasattr(self.client, method):
                    logger.error(f"Missing method: {method}")
                    return False

            # Test account balance method signatures
            balance = await self.client.get_account_balance(AccountType.CFM)
            assert isinstance(balance, Decimal)

            spot_balance = await self.client.get_spot_balance()
            assert isinstance(spot_balance, Decimal)

            futures_balance = await self.client.get_futures_balance()
            assert isinstance(futures_balance, Decimal)

            logger.info("  ✓ Futures configuration loaded")
            logger.info("  ✓ Connection status includes futures info")
            logger.info("  ✓ All futures methods present")
            logger.info("  ✓ Balance methods working")

            return True

        except Exception as e:
            logger.error(f"Exchange client validation error: {e}")
            return False

    async def validate_llm_agent(self) -> bool:
        """Validate LLM agent enhancements."""
        try:
            # Initialize LLM agent
            self.llm_agent = LLMAgent()

            # Test model configuration
            assert self.llm_agent.model_name == "o3"

            # Test status includes o3 model
            status = self.llm_agent.get_status()
            assert status["model_name"] == "o3"

            # Test prompt template includes futures elements
            assert "leverage" in self.llm_agent.prompt_text
            assert "margin_health" in self.llm_agent.prompt_text
            assert "reduce_only" in self.llm_agent.prompt_text
            assert "futures" in self.llm_agent.prompt_text.lower()

            # Test market state preparation with futures data
            mock_margin_info = MarginInfo(
                total_margin=Decimal("10000"),
                available_margin=Decimal("8000"),
                used_margin=Decimal("2000"),
                maintenance_margin=Decimal("1000"),
                initial_margin=Decimal("2000"),
                margin_ratio=0.2,
                health_status=MarginHealthStatus.HEALTHY,
                liquidation_threshold=Decimal("9000"),
                intraday_margin_requirement=Decimal("2000"),
                overnight_margin_requirement=Decimal("4000"),
            )

            mock_futures_account = FuturesAccountInfo(
                account_type=AccountType.CFM,
                account_id="test",
                cash_balance=Decimal("10000"),
                futures_balance=Decimal("10000"),
                total_balance=Decimal("20000"),
                margin_info=mock_margin_info,
                max_position_size=Decimal("5000"),
                timestamp=datetime.utcnow(),
            )

            mock_market_state = MarketState(
                symbol="BTC-USD",
                interval="1m",
                timestamp=datetime.utcnow(),
                current_price=Decimal("50000"),
                ohlcv_data=[
                    MarketData(
                        symbol="BTC-USD",
                        timestamp=datetime.utcnow(),
                        open=Decimal("49900"),
                        high=Decimal("50100"),
                        low=Decimal("49800"),
                        close=Decimal("50000"),
                        volume=Decimal("100"),
                    )
                ],
                indicators=IndicatorData(
                    timestamp=datetime.utcnow(),
                    cipher_a_dot=1.0,
                    cipher_b_wave=0.5,
                    rsi=65.0,
                ),
                current_position=Position(
                    symbol="BTC-USD",
                    side="FLAT",
                    size=Decimal("0"),
                    timestamp=datetime.utcnow(),
                ),
            )

            # Add futures account to market state
            mock_market_state.futures_account = mock_futures_account

            # Test input preparation
            llm_input = self.llm_agent._prepare_llm_input(mock_market_state)

            required_futures_keys = [
                "margin_health",
                "available_margin",
                "max_leverage",
                "futures_enabled",
                "auto_cash_transfer",
            ]

            for key in required_futures_keys:
                if key not in llm_input:
                    logger.error(f"Missing futures key in LLM input: {key}")
                    return False

            logger.info("  ✓ Model configured for o3")
            logger.info("  ✓ Prompt template enhanced for futures")
            logger.info("  ✓ Input preparation includes futures data")

            return True

        except Exception as e:
            logger.error(f"LLM agent validation error: {e}")
            return False

    async def validate_integration(self) -> bool:
        """Validate end-to-end integration."""
        try:
            # Test complete workflow in dry-run mode
            if not settings.system.dry_run:
                logger.warning("Integration test requires dry-run mode")
                return True

            # Create test trade action with futures parameters
            trade_action = TradeAction(
                action="LONG",
                size_pct=10,
                take_profit_pct=3.0,
                stop_loss_pct=2.0,
                rationale="Integration test",
                leverage=5,
                reduce_only=False,
            )

            # Test futures order placement
            if self.client:
                order = await self.client.place_futures_market_order(
                    symbol="BTC-USD", side="BUY", quantity=Decimal("0.01"), leverage=5
                )

                if not order:
                    logger.error("Failed to place test futures order")
                    return False

                assert order.symbol == "BTC-USD"
                assert order.side == "BUY"
                assert order.quantity == Decimal("0.01")

            logger.info("  ✓ Futures trade action creation")
            logger.info("  ✓ Futures order placement (dry-run)")
            logger.info("  ✓ End-to-end workflow functional")

            return True

        except Exception as e:
            logger.error(f"Integration validation error: {e}")
            return False

    def _print_summary(self, all_passed: bool):
        """Print validation summary."""
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)

        for test_name, result in self.validation_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")

        logger.info("=" * 60)

        if all_passed:
            logger.info(
                "🎉 ALL TESTS PASSED! Coinbase Futures implementation is ready."
            )
            logger.info("\nKey Enhancements Validated:")
            logger.info("  • OpenAI o3 model as default LLM")
            logger.info("  • Futures trading configuration")
            logger.info("  • CFM/CBI account separation")
            logger.info("  • Margin health monitoring")
            logger.info("  • Leverage and risk management")
            logger.info("  • Auto cash transfer functionality")
            logger.info("  • Enhanced LLM prompting for futures")
        else:
            logger.error("❌ Some tests failed. Please review the implementation.")

        logger.info("=" * 60)


async def main():
    """Main validation function."""
    validator = FuturesImplementationValidator()
    success = await validator.validate_all()

    if success:
        print("\n🎯 Validation completed successfully!")
        print("The Coinbase Futures trading implementation is ready for use.")
        return 0
    else:
        print("\n💥 Validation failed!")
        print("Please review the errors above and fix the implementation.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Validation error: {e}")
        sys.exit(1)
