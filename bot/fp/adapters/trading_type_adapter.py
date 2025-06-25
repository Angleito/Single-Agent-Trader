"""
Trading Type Adapter for Functional/Pydantic Compatibility

This module provides adapters and utilities to ensure seamless compatibility
between functional trading types and legacy Pydantic-based types.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from bot.fp.types.trading import (
    CBI_ACCOUNT,
    CFM_ACCOUNT,
    FunctionalMarketData,
    FunctionalMarketState,
    FuturesAccountBalance,
    FuturesLimitOrder,
    FuturesMarketOrder,
    FuturesStopOrder,
    LimitOrder,
    MarginInfo,
    MarketOrder,
    Position,
    RiskLimits,
    RiskMetrics,
    StopOrder,
    TradingIndicators,
    convert_functional_to_pydantic_position,
    convert_order_to_functional,
    convert_pydantic_to_functional_market_data,
    convert_pydantic_to_functional_position,
)

logger = logging.getLogger(__name__)


class TradingTypeAdapter:
    """
    Adapter for seamless conversion between functional and Pydantic trading types.

    This class provides methods to convert between the new functional types
    and legacy Pydantic types, ensuring backward compatibility during the
    migration process.
    """

    @staticmethod
    def adapt_market_data_to_functional(legacy_data: Any) -> FunctionalMarketData:
        """
        Convert legacy market data to functional equivalent.

        Args:
            legacy_data: Legacy market data (Pydantic or dict)

        Returns:
            FunctionalMarketData instance
        """
        try:
            if hasattr(legacy_data, "dict"):
                # Pydantic model
                return convert_pydantic_to_functional_market_data(legacy_data)
            if isinstance(legacy_data, dict):
                # Dictionary format
                return FunctionalMarketData(
                    symbol=legacy_data["symbol"],
                    timestamp=legacy_data.get("timestamp", datetime.now()),
                    open=Decimal(str(legacy_data["open"])),
                    high=Decimal(str(legacy_data["high"])),
                    low=Decimal(str(legacy_data["low"])),
                    close=Decimal(str(legacy_data["close"])),
                    volume=Decimal(str(legacy_data.get("volume", 0))),
                )
            raise ValueError(f"Unsupported market data format: {type(legacy_data)}")

        except Exception as e:
            logger.exception(f"Failed to adapt market data: {e}")
            raise ValueError(f"Market data adaptation failed: {e}")

    @staticmethod
    def adapt_position_to_functional(legacy_position: Any) -> Position:
        """
        Convert legacy position to functional equivalent.

        Args:
            legacy_position: Legacy position (Pydantic or dict)

        Returns:
            Position instance
        """
        try:
            if hasattr(legacy_position, "dict"):
                # Pydantic model
                return convert_pydantic_to_functional_position(legacy_position)
            if isinstance(legacy_position, dict):
                # Dictionary format
                return Position(
                    symbol=legacy_position["symbol"],
                    side=legacy_position["side"],
                    size=Decimal(str(legacy_position["size"])),
                    entry_price=(
                        Decimal(str(legacy_position["entry_price"]))
                        if legacy_position.get("entry_price")
                        else None
                    ),
                    unrealized_pnl=Decimal(
                        str(legacy_position.get("unrealized_pnl", 0))
                    ),
                    realized_pnl=Decimal(str(legacy_position.get("realized_pnl", 0))),
                    timestamp=legacy_position.get("timestamp", datetime.now()),
                )
            raise ValueError(f"Unsupported position format: {type(legacy_position)}")

        except Exception as e:
            logger.exception(f"Failed to adapt position: {e}")
            raise ValueError(f"Position adaptation failed: {e}")

    @staticmethod
    def adapt_order_to_functional(
        legacy_order: Any,
    ) -> LimitOrder | MarketOrder | StopOrder:
        """
        Convert legacy order to functional equivalent.

        Args:
            legacy_order: Legacy order (Pydantic or dict)

        Returns:
            Functional order instance
        """
        try:
            return convert_order_to_functional(legacy_order)
        except Exception as e:
            logger.exception(f"Failed to adapt order: {e}")
            raise ValueError(f"Order adaptation failed: {e}")

    @staticmethod
    def adapt_indicators_to_functional(legacy_indicators: Any) -> TradingIndicators:
        """
        Convert legacy indicators to functional equivalent.

        Args:
            legacy_indicators: Legacy indicators (Pydantic or dict)

        Returns:
            TradingIndicators instance
        """
        try:
            if hasattr(legacy_indicators, "dict"):
                # Pydantic model
                data = legacy_indicators.dict()
            elif isinstance(legacy_indicators, dict):
                data = legacy_indicators
            else:
                raise ValueError(
                    f"Unsupported indicators format: {type(legacy_indicators)}"
                )

            return TradingIndicators(
                timestamp=data.get("timestamp", datetime.now()),
                rsi=data.get("rsi"),
                macd=data.get("macd"),
                macd_signal=data.get("macd_signal"),
                macd_histogram=data.get("macd_histogram"),
                ema_fast=data.get("ema_fast"),
                ema_slow=data.get("ema_slow"),
                bollinger_upper=data.get("bollinger_upper"),
                bollinger_middle=data.get("bollinger_middle"),
                bollinger_lower=data.get("bollinger_lower"),
                volume_sma=data.get("volume_sma"),
                atr=data.get("atr"),
                cipher_a_dot=data.get("cipher_a_dot"),
                cipher_b_wave=data.get("cipher_b_wave"),
                cipher_b_money_flow=data.get("cipher_b_money_flow"),
                usdt_dominance=data.get("usdt_dominance"),
                usdc_dominance=data.get("usdc_dominance"),
                stablecoin_dominance=data.get("stablecoin_dominance"),
                dominance_trend=data.get("dominance_trend"),
                dominance_rsi=data.get("dominance_rsi"),
            )

        except Exception as e:
            logger.exception(f"Failed to adapt indicators: {e}")
            raise ValueError(f"Indicators adaptation failed: {e}")

    @staticmethod
    def create_futures_order_from_legacy(
        legacy_order: Any, leverage: int = 1, margin_required: Decimal = Decimal(0)
    ) -> FuturesLimitOrder | FuturesMarketOrder | FuturesStopOrder:
        """
        Convert legacy order to futures-specific functional equivalent.

        Args:
            legacy_order: Legacy order data
            leverage: Trading leverage
            margin_required: Required margin

        Returns:
            Futures order instance
        """
        try:
            order_type = getattr(legacy_order, "type", "MARKET").upper()

            common_fields = {
                "symbol": legacy_order.symbol,
                "side": legacy_order.side.lower(),
                "size": float(
                    getattr(legacy_order, "quantity", getattr(legacy_order, "size", 0))
                ),
                "leverage": leverage,
                "margin_required": margin_required,
                "reduce_only": getattr(legacy_order, "reduce_only", False),
                "order_id": getattr(legacy_order, "id", ""),
            }

            if order_type == "LIMIT":
                return FuturesLimitOrder(
                    price=float(legacy_order.price),
                    post_only=getattr(legacy_order, "post_only", False),
                    time_in_force=getattr(legacy_order, "time_in_force", "GTC"),
                    **common_fields,
                )
            if order_type == "STOP":
                return FuturesStopOrder(
                    stop_price=float(legacy_order.stop_price),
                    time_in_force=getattr(legacy_order, "time_in_force", "GTC"),
                    **common_fields,
                )
            # MARKET
            return FuturesMarketOrder(**common_fields)

        except Exception as e:
            logger.exception(f"Failed to create futures order: {e}")
            raise ValueError(f"Futures order creation failed: {e}")

    @staticmethod
    def adapt_account_balance_to_functional(
        legacy_balance: Any,
    ) -> FuturesAccountBalance:
        """
        Convert legacy account balance to functional equivalent.

        Args:
            legacy_balance: Legacy account balance data

        Returns:
            FuturesAccountBalance instance
        """
        try:
            if hasattr(legacy_balance, "dict"):
                data = legacy_balance.dict()
            elif isinstance(legacy_balance, dict):
                data = legacy_balance
            else:
                raise ValueError(f"Unsupported balance format: {type(legacy_balance)}")

            # Create margin info
            margin_data = data.get("margin_info", {})
            margin_info = MarginInfo(
                total_margin=Decimal(str(margin_data.get("total_margin", 0))),
                available_margin=Decimal(str(margin_data.get("available_margin", 0))),
                used_margin=Decimal(str(margin_data.get("used_margin", 0))),
                maintenance_margin=Decimal(
                    str(margin_data.get("maintenance_margin", 0))
                ),
                initial_margin=Decimal(str(margin_data.get("initial_margin", 0))),
                health_status=margin_data.get("health_status", "HEALTHY"),
                liquidation_threshold=Decimal(
                    str(margin_data.get("liquidation_threshold", 0))
                ),
                intraday_margin_requirement=Decimal(
                    str(margin_data.get("intraday_margin_requirement", 0))
                ),
                overnight_margin_requirement=Decimal(
                    str(margin_data.get("overnight_margin_requirement", 0))
                ),
                is_overnight_position=margin_data.get("is_overnight_position", False),
            )

            # Create account type
            account_type_str = data.get("account_type", "CFM")
            account_type = CFM_ACCOUNT if account_type_str == "CFM" else CBI_ACCOUNT

            return FuturesAccountBalance(
                account_type=account_type,
                account_id=data.get("account_id", ""),
                currency=data.get("currency", "USD"),
                cash_balance=Decimal(str(data.get("cash_balance", 0))),
                futures_balance=Decimal(str(data.get("futures_balance", 0))),
                total_balance=Decimal(str(data.get("total_balance", 0))),
                margin_info=margin_info,
                auto_cash_transfer_enabled=data.get("auto_cash_transfer_enabled", True),
                min_cash_transfer_amount=Decimal(
                    str(data.get("min_cash_transfer_amount", 100))
                ),
                max_cash_transfer_amount=Decimal(
                    str(data.get("max_cash_transfer_amount", 10000))
                ),
                max_leverage=data.get("max_leverage", 20),
                max_position_size=Decimal(str(data.get("max_position_size", 1000000))),
                current_positions_count=data.get("current_positions_count", 0),
                timestamp=data.get("timestamp", datetime.now()),
            )

        except Exception as e:
            logger.exception(f"Failed to adapt account balance: {e}")
            raise ValueError(f"Account balance adaptation failed: {e}")

    @staticmethod
    def adapt_legacy_to_functional_state(
        symbol: str,
        legacy_market_data: Any,
        legacy_indicators: Any,
        legacy_position: Any,
        legacy_account: Any = None,
    ) -> FunctionalMarketState:
        """
        Create functional market state from legacy components.

        Args:
            symbol: Trading symbol
            legacy_market_data: Legacy market data
            legacy_indicators: Legacy indicators
            legacy_position: Legacy position
            legacy_account: Legacy account data (optional)

        Returns:
            FunctionalMarketState instance
        """
        try:
            adapter = TradingTypeAdapter()

            # Convert components
            market_data = adapter.adapt_market_data_to_functional(legacy_market_data)
            indicators = adapter.adapt_indicators_to_functional(legacy_indicators)
            position = adapter.adapt_position_to_functional(legacy_position)

            account_balance = None
            if legacy_account:
                account_balance = adapter.adapt_account_balance_to_functional(
                    legacy_account
                )

            return FunctionalMarketState(
                symbol=symbol,
                timestamp=datetime.now(),
                market_data=market_data,
                indicators=indicators,
                position=position,
                account_balance=account_balance,
            )

        except Exception as e:
            logger.exception(f"Failed to create functional market state: {e}")
            raise ValueError(f"Market state creation failed: {e}")

    @staticmethod
    def convert_functional_state_to_legacy(
        functional_state: FunctionalMarketState,
    ) -> dict[str, Any]:
        """
        Convert functional market state back to legacy format.

        Args:
            functional_state: Functional market state

        Returns:
            Dictionary in legacy format
        """
        try:
            # Convert position back to legacy format
            legacy_position = convert_functional_to_pydantic_position(
                functional_state.position
            )

            return {
                "symbol": functional_state.symbol,
                "timestamp": functional_state.timestamp,
                "current_price": float(functional_state.current_price),
                "market_data": {
                    "symbol": functional_state.market_data.symbol,
                    "timestamp": functional_state.market_data.timestamp,
                    "open": float(functional_state.market_data.open),
                    "high": float(functional_state.market_data.high),
                    "low": float(functional_state.market_data.low),
                    "close": float(functional_state.market_data.close),
                    "volume": float(functional_state.market_data.volume),
                },
                "indicators": {
                    "timestamp": functional_state.indicators.timestamp,
                    "rsi": functional_state.indicators.rsi,
                    "cipher_a_dot": functional_state.indicators.cipher_a_dot,
                    "cipher_b_wave": functional_state.indicators.cipher_b_wave,
                    "cipher_b_money_flow": functional_state.indicators.cipher_b_money_flow,
                    "stablecoin_dominance": functional_state.indicators.stablecoin_dominance,
                    "dominance_trend": functional_state.indicators.dominance_trend,
                },
                "position": (
                    legacy_position.dict()
                    if hasattr(legacy_position, "dict")
                    else legacy_position
                ),
                "has_position": functional_state.has_position,
                "is_futures_market": functional_state.is_futures_market,
                "position_value": float(functional_state.position_value),
            }

        except Exception as e:
            logger.exception(f"Failed to convert functional state to legacy: {e}")
            raise ValueError(f"Legacy conversion failed: {e}")


class OrderExecutionAdapter:
    """
    Adapter for order execution that bridges functional and legacy systems.
    """

    @staticmethod
    def prepare_order_for_execution(
        functional_order: LimitOrder | MarketOrder | StopOrder,
        exchange_format: str = "coinbase",
    ) -> dict[str, Any]:
        """
        Convert functional order to exchange-compatible format.

        Args:
            functional_order: Functional order instance
            exchange_format: Target exchange format

        Returns:
            Exchange-compatible order dictionary
        """
        try:
            base_order = {
                "symbol": functional_order.symbol,
                "side": functional_order.side.upper(),
                "size": str(functional_order.size),
                "order_id": functional_order.order_id,
            }

            if isinstance(functional_order, LimitOrder):
                base_order.update(
                    {"type": "LIMIT", "price": str(functional_order.price)}
                )
            elif isinstance(functional_order, StopOrder):
                base_order.update(
                    {"type": "STOP", "stop_price": str(functional_order.stop_price)}
                )
            else:  # MarketOrder
                base_order.update({"type": "MARKET"})

            # Exchange-specific formatting
            if exchange_format.lower() == "coinbase":
                return OrderExecutionAdapter._format_for_coinbase(base_order)
            if exchange_format.lower() == "bluefin":
                return OrderExecutionAdapter._format_for_bluefin(base_order)
            return base_order

        except Exception as e:
            logger.exception(f"Failed to prepare order for execution: {e}")
            raise ValueError(f"Order preparation failed: {e}")

    @staticmethod
    def _format_for_coinbase(order: dict[str, Any]) -> dict[str, Any]:
        """Format order for Coinbase exchange."""
        # Coinbase-specific formatting
        coinbase_order = order.copy()
        coinbase_order["product_id"] = coinbase_order.pop("symbol")
        coinbase_order["client_order_id"] = coinbase_order.pop("order_id")

        if coinbase_order["type"] == "LIMIT":
            coinbase_order["limit_price"] = coinbase_order.pop("price")
        elif coinbase_order["type"] == "STOP":
            coinbase_order["stop_price"] = coinbase_order.get("stop_price")

        return coinbase_order

    @staticmethod
    def _format_for_bluefin(order: dict[str, Any]) -> dict[str, Any]:
        """Format order for Bluefin exchange."""
        # Bluefin-specific formatting
        bluefin_order = order.copy()

        # Convert size to integer (assuming base unit conversion)
        try:
            bluefin_order["quantity"] = int(
                float(bluefin_order.pop("size")) * 1e18
            )  # Convert to wei
        except ValueError:
            bluefin_order["quantity"] = bluefin_order.pop("size")

        if bluefin_order["type"] == "LIMIT":
            bluefin_order["price"] = int(float(bluefin_order.pop("price")) * 1e18)
        elif bluefin_order["type"] == "STOP":
            bluefin_order["trigger_price"] = int(
                float(bluefin_order.pop("stop_price")) * 1e18
            )

        return bluefin_order


class RiskAdapterMixin:
    """
    Mixin class providing risk management adaptation utilities.
    """

    @staticmethod
    def adapt_risk_metrics_to_functional(legacy_risk: Any) -> RiskMetrics:
        """
        Convert legacy risk metrics to functional equivalent.

        Args:
            legacy_risk: Legacy risk metrics

        Returns:
            RiskMetrics instance
        """
        try:
            if hasattr(legacy_risk, "dict"):
                data = legacy_risk.dict()
            elif isinstance(legacy_risk, dict):
                data = legacy_risk
            else:
                raise ValueError(f"Unsupported risk format: {type(legacy_risk)}")

            return RiskMetrics(
                account_balance=Decimal(str(data.get("account_balance", 0))),
                available_margin=Decimal(str(data.get("available_margin", 0))),
                used_margin=Decimal(str(data.get("used_margin", 0))),
                daily_pnl=Decimal(str(data.get("daily_pnl", 0))),
                total_exposure=Decimal(str(data.get("total_exposure", 0))),
                current_positions=data.get("current_positions", 0),
                max_daily_loss_reached=data.get("max_daily_loss_reached", False),
                value_at_risk_95=(
                    Decimal(str(data["value_at_risk_95"]))
                    if data.get("value_at_risk_95")
                    else None
                ),
                sharpe_ratio=data.get("sharpe_ratio"),
                sortino_ratio=data.get("sortino_ratio"),
                max_drawdown=data.get("max_drawdown"),
            )

        except Exception as e:
            logger.exception(f"Failed to adapt risk metrics: {e}")
            raise ValueError(f"Risk metrics adaptation failed: {e}")

    @staticmethod
    def validate_functional_risk_compliance(
        functional_state: FunctionalMarketState, risk_limits: RiskLimits
    ) -> dict[str, Any]:
        """
        Validate functional state against risk limits.

        Args:
            functional_state: Current market state
            risk_limits: Risk limits to check against

        Returns:
            Validation results dictionary
        """
        try:
            violations = []
            warnings = []

            # Check position size
            if functional_state.has_position:
                position_value = functional_state.position_value
                if position_value > risk_limits.max_position_size:
                    violations.append(
                        f"Position value {position_value} exceeds limit {risk_limits.max_position_size}"
                    )

            # Check risk metrics if available
            if functional_state.risk_metrics:
                metrics = functional_state.risk_metrics

                if not metrics.is_within_risk_limits(risk_limits):
                    violations.append("Risk metrics exceed defined limits")

                risk_score = metrics.risk_score()
                if risk_score > 80:
                    warnings.append(f"High risk score: {risk_score:.1f}")
                elif risk_score > 60:
                    warnings.append(f"Moderate risk score: {risk_score:.1f}")

            # Check account balance if available
            if functional_state.account_balance:
                margin_ratio = functional_state.account_balance.margin_info.margin_ratio
                if margin_ratio > 0.8:
                    violations.append(f"High margin utilization: {margin_ratio:.1%}")
                elif margin_ratio > 0.6:
                    warnings.append(f"Moderate margin utilization: {margin_ratio:.1%}")

            return {
                "compliant": len(violations) == 0,
                "violations": violations,
                "warnings": warnings,
                "timestamp": datetime.now(),
                "symbol": functional_state.symbol,
            }

        except Exception as e:
            logger.exception(f"Risk validation failed: {e}")
            return {
                "compliant": False,
                "violations": [f"Validation error: {e}"],
                "warnings": [],
                "timestamp": datetime.now(),
                "symbol": functional_state.symbol,
            }


# Integration utility class
class FunctionalTradingIntegration(TradingTypeAdapter, RiskAdapterMixin):
    """
    Complete integration utility combining all adaptation capabilities.
    """

    def __init__(self):
        """Initialize the integration utility."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.conversion_stats = {
            "market_data_conversions": 0,
            "position_conversions": 0,
            "order_conversions": 0,
            "failures": 0,
        }

    def batch_convert_legacy_data(
        self, legacy_data_batch: list[dict[str, Any]]
    ) -> list[FunctionalMarketState]:
        """
        Convert a batch of legacy data to functional market states.

        Args:
            legacy_data_batch: List of legacy data dictionaries

        Returns:
            List of functional market states
        """
        functional_states = []

        for item in legacy_data_batch:
            try:
                state = self.adapt_legacy_to_functional_state(
                    symbol=item["symbol"],
                    legacy_market_data=item["market_data"],
                    legacy_indicators=item["indicators"],
                    legacy_position=item["position"],
                    legacy_account=item.get("account"),
                )
                functional_states.append(state)
                self.conversion_stats["market_data_conversions"] += 1

            except Exception as e:
                self.logger.exception(f"Failed to convert legacy data item: {e}")
                self.conversion_stats["failures"] += 1

        return functional_states

    def get_conversion_statistics(self) -> dict[str, Any]:
        """Get conversion statistics."""
        return self.conversion_stats.copy()

    def reset_statistics(self) -> None:
        """Reset conversion statistics."""
        for key in self.conversion_stats:
            self.conversion_stats[key] = 0


# Global adapter instance for easy access
trading_adapter = FunctionalTradingIntegration()
