"""
Intelligent inventory management system for market making strategies.

This module provides sophisticated inventory tracking and rebalancing for market making,
integrating with VuManChu signals to inform directional bias decisions and manage
inventory risk while maintaining liquidity provision.
"""

import json
import logging
import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from ..error_handling import exception_handler
from ..trading_types import Order, Position
from ..utils.path_utils import get_data_directory, get_data_file_path

logger = logging.getLogger(__name__)


class InventoryMetrics:
    """Container for inventory risk metrics and statistics."""

    def __init__(
        self,
        symbol: str,
        net_position: Decimal,
        position_value: Decimal,
        imbalance_percentage: float,
        risk_score: float,
        max_position_limit: Decimal,
        rebalancing_threshold: float,
        time_weighted_exposure: Decimal,
        inventory_duration_hours: float,
    ):
        """Initialize inventory metrics."""
        self.symbol = symbol
        self.net_position = net_position
        self.position_value = position_value
        self.imbalance_percentage = imbalance_percentage
        self.risk_score = risk_score
        self.max_position_limit = max_position_limit
        self.rebalancing_threshold = rebalancing_threshold
        self.time_weighted_exposure = time_weighted_exposure
        self.inventory_duration_hours = inventory_duration_hours
        self.timestamp = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "symbol": self.symbol,
            "net_position": str(self.net_position),
            "position_value": str(self.position_value),
            "imbalance_percentage": self.imbalance_percentage,
            "risk_score": self.risk_score,
            "max_position_limit": str(self.max_position_limit),
            "rebalancing_threshold": self.rebalancing_threshold,
            "time_weighted_exposure": str(self.time_weighted_exposure),
            "inventory_duration_hours": self.inventory_duration_hours,
            "timestamp": self.timestamp.isoformat(),
        }


class RebalancingAction:
    """Represents a rebalancing trade recommendation."""

    def __init__(
        self,
        action_type: Literal["BUY", "SELL", "HOLD"],
        quantity: Decimal,
        urgency: Literal["LOW", "MEDIUM", "HIGH", "EMERGENCY"],
        reason: str,
        target_price: Decimal | None = None,
        vumanchu_bias: str | None = None,
        confidence: float = 0.5,
    ):
        """Initialize rebalancing action."""
        self.action_type = action_type
        self.quantity = quantity
        self.urgency = urgency
        self.reason = reason
        self.target_price = target_price
        self.vumanchu_bias = vumanchu_bias
        self.confidence = confidence
        self.timestamp = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert action to dictionary format."""
        return {
            "action_type": self.action_type,
            "quantity": str(self.quantity),
            "urgency": self.urgency,
            "reason": self.reason,
            "target_price": str(self.target_price) if self.target_price else None,
            "vumanchu_bias": self.vumanchu_bias,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


class VuManChuBias:
    """Container for VuManChu directional bias information."""

    def __init__(
        self,
        overall_bias: Literal["BULLISH", "BEARISH", "NEUTRAL"],
        cipher_a_signal: str | None = None,
        cipher_b_signal: str | None = None,
        wave_trend_direction: str | None = None,
        signal_strength: float = 0.0,
        confidence: float = 0.5,
    ):
        """Initialize VuManChu bias."""
        self.overall_bias = overall_bias
        self.cipher_a_signal = cipher_a_signal
        self.cipher_b_signal = cipher_b_signal
        self.wave_trend_direction = wave_trend_direction
        self.signal_strength = signal_strength
        self.confidence = confidence
        self.timestamp = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert bias to dictionary format."""
        return {
            "overall_bias": self.overall_bias,
            "cipher_a_signal": self.cipher_a_signal,
            "cipher_b_signal": self.cipher_b_signal,
            "wave_trend_direction": self.wave_trend_direction,
            "signal_strength": self.signal_strength,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


class InventoryManagerError(Exception):
    """Base exception for inventory manager errors."""


class InventoryValidationError(InventoryManagerError):
    """Exception raised when inventory validation fails."""


class InventoryRebalancingError(InventoryManagerError):
    """Exception raised when rebalancing operations fail."""


class InventoryManager:
    """
    Intelligent inventory management system for market making strategies.

    Features:
    - Real-time inventory tracking and position monitoring
    - VuManChu signal integration for directional bias
    - Risk-based rebalancing recommendations
    - Emergency flatten mechanisms for extreme imbalances
    - Comprehensive inventory analytics and reporting
    - Integration with order management and exchange systems
    """

    def __init__(
        self,
        symbol: str,
        max_position_pct: float = 10.0,  # Max position as % of account equity
        rebalancing_threshold: float = 5.0,  # Rebalancing trigger threshold %
        emergency_threshold: float = 15.0,  # Emergency flatten threshold %
        inventory_timeout_hours: float = 4.0,  # Max time to hold inventory
        data_dir: Path | None = None,
    ):
        """
        Initialize the inventory manager.

        Args:
            symbol: Trading symbol to manage inventory for
            max_position_pct: Maximum position size as percentage of account equity
            rebalancing_threshold: Threshold percentage to trigger rebalancing
            emergency_threshold: Threshold percentage to trigger emergency flattening
            inventory_timeout_hours: Maximum hours to hold inventory position
            data_dir: Directory for state persistence
        """
        self.symbol = symbol
        self.max_position_pct = max_position_pct
        self.rebalancing_threshold = rebalancing_threshold
        self.emergency_threshold = emergency_threshold
        self.inventory_timeout_hours = inventory_timeout_hours

        # Data directory setup
        if data_dir:
            self.data_dir = data_dir
        else:
            self.data_dir = get_data_directory() / "inventory"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Thread-safe inventory tracking
        self._lock = threading.RLock()
        self._current_position = Decimal(0)
        self._position_history: list[dict[str, Any]] = []
        self._rebalancing_history: list[dict[str, Any]] = []
        self._last_rebalancing_time: datetime | None = None

        # VuManChu integration
        self._current_vumanchu_bias: VuManChuBias | None = None
        self._bias_history: list[dict[str, Any]] = []

        # Risk tracking
        self._account_equity = Decimal(10000)  # Default equity
        self._average_entry_price: Decimal | None = None
        self._position_start_time: datetime | None = None
        self._total_realized_pnl = Decimal(0)
        self._inventory_turns_daily = 0
        self._max_inventory_seen = Decimal(0)

        # Performance metrics
        self._rebalancing_success_count = 0
        self._rebalancing_failure_count = 0
        self._emergency_flatten_count = 0
        self._inventory_metrics_history: list[InventoryMetrics] = []

        # State persistence
        self.state_file = get_data_file_path(f"inventory/{symbol}_inventory_state.json")
        self._load_state()

        logger.info(
            "🎯 Initialized InventoryManager for %s:\n"
            "  • Max position: %.1f%% of equity\n"
            "  • Rebalancing threshold: %.1f%%\n"
            "  • Emergency threshold: %.1f%%\n"
            "  • Inventory timeout: %.1f hours\n"
            "  • Current position: %s",
            symbol,
            max_position_pct,
            rebalancing_threshold,
            emergency_threshold,
            inventory_timeout_hours,
            self._current_position,
        )

    def track_position_changes(
        self, fills: list[Order], current_position: Position
    ) -> InventoryMetrics:
        """
        Track position changes from order fills and update inventory.

        Args:
            fills: List of recent order fills
            current_position: Current position from position manager

        Returns:
            Updated inventory metrics
        """
        try:
            with self._lock:
                # Process new fills
                for fill in fills:
                    self._process_fill(fill)

                # Update current position from position manager
                self._update_position_from_manager(current_position)

                # Calculate inventory metrics
                metrics = self._calculate_inventory_metrics()

                # Store metrics history
                self._inventory_metrics_history.append(metrics)

                # Keep only recent metrics (last 24 hours)
                cutoff_time = datetime.now(UTC) - timedelta(hours=24)
                self._inventory_metrics_history = [
                    m
                    for m in self._inventory_metrics_history
                    if m.timestamp >= cutoff_time
                ]

                # Save state
                self._save_state()

                logger.debug(
                    "📊 Inventory updated for %s: position=%s, imbalance=%.2f%%, risk=%.2f",
                    self.symbol,
                    metrics.net_position,
                    metrics.imbalance_percentage,
                    metrics.risk_score,
                )

                return metrics

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "fills_count": len(fills),
                    "current_position_side": current_position.side,
                    "inventory_tracking_error": True,
                },
                component="InventoryManager",
                operation="track_position_changes",
            )
            # Return safe fallback metrics
            return self._get_fallback_metrics()

    def calculate_inventory_imbalance(self) -> float:
        """
        Calculate current inventory imbalance as percentage.

        Returns:
            Imbalance percentage (positive = long bias, negative = short bias)
        """
        with self._lock:
            if self._account_equity <= 0:
                return 0.0

            position_value = abs(self._current_position) * (
                self._average_entry_price or Decimal(1)
            )
            max_position_value = self._account_equity * (
                Decimal(str(self.max_position_pct)) / Decimal(100)
            )

            if max_position_value <= 0:
                return 0.0

            # Calculate imbalance as percentage of maximum allowed position
            imbalance_pct = float((position_value / max_position_value) * 100)

            # Apply sign based on position direction
            if self._current_position < 0:
                imbalance_pct = -imbalance_pct

            return imbalance_pct

    def suggest_rebalancing_action(
        self,
        imbalance: float,
        vumanchu_bias: VuManChuBias,
        market_price: Decimal,
    ) -> RebalancingAction:
        """
        Suggest rebalancing action based on imbalance and VuManChu bias.

        Args:
            imbalance: Current inventory imbalance percentage
            vumanchu_bias: VuManChu directional bias
            market_price: Current market price

        Returns:
            Recommended rebalancing action
        """
        try:
            with self._lock:
                # Update VuManChu bias
                self._current_vumanchu_bias = vumanchu_bias
                self._bias_history.append(vumanchu_bias.to_dict())

                # Check for emergency conditions first
                emergency_action = self._check_emergency_conditions(
                    imbalance, market_price
                )
                if emergency_action:
                    return emergency_action

                # Check for timeout conditions
                timeout_action = self._check_timeout_conditions(imbalance, market_price)
                if timeout_action:
                    return timeout_action

                # Normal rebalancing logic with VuManChu integration
                return self._calculate_rebalancing_action(
                    imbalance, vumanchu_bias, market_price
                )

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "imbalance": imbalance,
                    "vumanchu_bias": vumanchu_bias.overall_bias,
                    "rebalancing_suggestion_error": True,
                },
                component="InventoryManager",
                operation="suggest_rebalancing_action",
            )
            return RebalancingAction(
                action_type="HOLD",
                quantity=Decimal(0),
                urgency="LOW",
                reason="Error in rebalancing calculation",
            )

    def execute_rebalancing_trade(
        self, action: RebalancingAction, market_price: Decimal
    ) -> bool:
        """
        Execute a rebalancing trade action.

        Args:
            action: Rebalancing action to execute
            market_price: Current market price

        Returns:
            True if trade was successfully executed
        """
        try:
            if action.action_type == "HOLD":
                logger.debug("🔒 Inventory rebalancing: HOLD - no action needed")
                return True

            logger.info(
                "⚖️ Executing inventory rebalancing for %s:\n"
                "  • Action: %s %s\n"
                "  • Urgency: %s\n"
                "  • Reason: %s\n"
                "  • Price: %s\n"
                "  • VuManChu bias: %s",
                self.symbol,
                action.action_type,
                action.quantity,
                action.urgency,
                action.reason,
                market_price,
                action.vumanchu_bias or "N/A",
            )

            # Record rebalancing attempt
            with self._lock:
                rebalancing_record = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "action": action.to_dict(),
                    "market_price": str(market_price),
                    "position_before": str(self._current_position),
                    "executed": False,
                }

                # Simulate execution for now (in real implementation, would interface with order manager)
                execution_success = self._simulate_rebalancing_execution(
                    action, market_price
                )

                rebalancing_record["executed"] = execution_success
                rebalancing_record["position_after"] = str(self._current_position)

                self._rebalancing_history.append(rebalancing_record)
                self._last_rebalancing_time = datetime.now(UTC)

                # Update success/failure counters
                if execution_success:
                    self._rebalancing_success_count += 1
                    logger.info("✅ Inventory rebalancing executed successfully")
                else:
                    self._rebalancing_failure_count += 1
                    logger.warning("❌ Inventory rebalancing execution failed")

                # Keep only recent rebalancing history
                cutoff_time = datetime.now(UTC) - timedelta(hours=24)
                self._rebalancing_history = [
                    r
                    for r in self._rebalancing_history
                    if datetime.fromisoformat(r["timestamp"]) >= cutoff_time
                ]

                self._save_state()
                return execution_success

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "action_type": action.action_type,
                    "quantity": str(action.quantity),
                    "rebalancing_execution_error": True,
                },
                component="InventoryManager",
                operation="execute_rebalancing_trade",
            )
            return False

    def get_inventory_metrics(self) -> InventoryMetrics:
        """
        Get current inventory metrics and risk assessment.

        Returns:
            Comprehensive inventory metrics
        """
        with self._lock:
            return self._calculate_inventory_metrics()

    def update_account_equity(self, new_equity: Decimal) -> None:
        """
        Update account equity for position sizing calculations.

        Args:
            new_equity: New account equity value
        """
        with self._lock:
            old_equity = self._account_equity
            self._account_equity = new_equity

            logger.debug(
                "💰 Account equity updated for %s: %s -> %s",
                self.symbol,
                old_equity,
                new_equity,
            )

    def get_position_summary(self) -> dict[str, Any]:
        """
        Get comprehensive position and inventory summary.

        Returns:
            Dictionary with position summary
        """
        with self._lock:
            metrics = self._calculate_inventory_metrics()

            # Calculate recent performance
            recent_rebalancing_count = len(
                [
                    r
                    for r in self._rebalancing_history
                    if datetime.fromisoformat(r["timestamp"])
                    >= datetime.now(UTC) - timedelta(hours=24)
                ]
            )

            recent_successful_rebalancing = len(
                [
                    r
                    for r in self._rebalancing_history
                    if datetime.fromisoformat(r["timestamp"])
                    >= datetime.now(UTC) - timedelta(hours=24)
                    and r.get("executed", False)
                ]
            )

            return {
                "symbol": self.symbol,
                "current_position": str(self._current_position),
                "position_value": str(metrics.position_value),
                "imbalance_percentage": metrics.imbalance_percentage,
                "risk_score": metrics.risk_score,
                "inventory_duration_hours": metrics.inventory_duration_hours,
                "account_equity": str(self._account_equity),
                "max_position_limit": str(metrics.max_position_limit),
                "rebalancing_threshold": self.rebalancing_threshold,
                "emergency_threshold": self.emergency_threshold,
                "total_realized_pnl": str(self._total_realized_pnl),
                "rebalancing_stats": {
                    "total_success": self._rebalancing_success_count,
                    "total_failure": self._rebalancing_failure_count,
                    "recent_24h_count": recent_rebalancing_count,
                    "recent_24h_success": recent_successful_rebalancing,
                    "emergency_flatten_count": self._emergency_flatten_count,
                    "last_rebalancing": (
                        self._last_rebalancing_time.isoformat()
                        if self._last_rebalancing_time
                        else None
                    ),
                },
                "vumanchu_bias": (
                    self._current_vumanchu_bias.to_dict()
                    if self._current_vumanchu_bias
                    else None
                ),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def reset_inventory(self) -> None:
        """Reset inventory state (for testing or emergency use)."""
        with self._lock:
            self._current_position = Decimal(0)
            self._position_history.clear()
            self._rebalancing_history.clear()
            self._bias_history.clear()
            self._average_entry_price = None
            self._position_start_time = None
            self._total_realized_pnl = Decimal(0)
            self._inventory_metrics_history.clear()

            logger.warning("🔄 Inventory state reset for %s", self.symbol)
            self._save_state()

    def _process_fill(self, fill: Order) -> None:
        """Process an order fill and update inventory."""
        if fill.symbol != self.symbol:
            return

        # Calculate position change
        if fill.side == "BUY":
            position_change = fill.filled_quantity
        else:  # SELL
            position_change = -fill.filled_quantity

        # Update current position
        old_position = self._current_position
        self._current_position += position_change

        # Update average entry price
        if fill.price and position_change != 0:
            self._update_average_entry_price(position_change, fill.price)

        # Track position start time
        if old_position == 0 and self._current_position != 0:
            self._position_start_time = datetime.now(UTC)
        elif self._current_position == 0:
            self._position_start_time = None

        # Record position change
        self._position_history.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "fill_id": fill.id,
                "side": fill.side,
                "quantity": str(fill.filled_quantity),
                "price": str(fill.price) if fill.price else None,
                "position_before": str(old_position),
                "position_after": str(self._current_position),
            }
        )

        logger.debug(
            "📈 Position updated for %s: %s -> %s (fill: %s %s)",
            self.symbol,
            old_position,
            self._current_position,
            fill.side,
            fill.filled_quantity,
        )

    def _update_position_from_manager(self, position: Position) -> None:
        """Update inventory from position manager data."""
        if position.symbol != self.symbol:
            return

        # Reconcile position if there's a discrepancy
        if position.side == "FLAT":
            target_position = Decimal(0)
        elif position.side == "LONG":
            target_position = position.size
        else:  # SHORT
            target_position = -position.size

        if target_position != self._current_position:
            logger.warning(
                "🔄 Position reconciliation for %s: %s -> %s",
                self.symbol,
                self._current_position,
                target_position,
            )
            self._current_position = target_position

        # Update entry price if available
        if position.entry_price and position.side != "FLAT":
            self._average_entry_price = position.entry_price

    def _calculate_inventory_metrics(self) -> InventoryMetrics:
        """Calculate comprehensive inventory metrics."""
        # Calculate position value
        position_value = abs(self._current_position) * (
            self._average_entry_price or Decimal(1)
        )

        # Calculate imbalance percentage
        imbalance_pct = self.calculate_inventory_imbalance()

        # Calculate risk score (0-100, higher = more risky)
        risk_score = self._calculate_risk_score(imbalance_pct)

        # Calculate maximum position limit
        max_position_limit = self._account_equity * (
            Decimal(str(self.max_position_pct)) / Decimal(100)
        )

        # Calculate time-weighted exposure
        time_weighted_exposure = self._calculate_time_weighted_exposure()

        # Calculate inventory duration
        inventory_duration = self._calculate_inventory_duration()

        return InventoryMetrics(
            symbol=self.symbol,
            net_position=self._current_position,
            position_value=position_value,
            imbalance_percentage=imbalance_pct,
            risk_score=risk_score,
            max_position_limit=max_position_limit,
            rebalancing_threshold=self.rebalancing_threshold,
            time_weighted_exposure=time_weighted_exposure,
            inventory_duration_hours=inventory_duration,
        )

    def _calculate_risk_score(self, imbalance_pct: float) -> float:
        """Calculate risk score based on multiple factors."""
        risk_factors = []

        # Imbalance risk (0-50 points)
        imbalance_risk = min(50, abs(imbalance_pct) * 2)
        risk_factors.append(imbalance_risk)

        # Duration risk (0-30 points)
        duration_hours = self._calculate_inventory_duration()
        duration_risk = min(30, (duration_hours / self.inventory_timeout_hours) * 30)
        risk_factors.append(duration_risk)

        # Size risk (0-20 points)
        position_ratio = abs(self._current_position) / max(
            self._account_equity / (self._average_entry_price or Decimal(1)), Decimal(1)
        )
        size_risk = min(20, float(position_ratio) * 200)
        risk_factors.append(size_risk)

        return sum(risk_factors)

    def _calculate_time_weighted_exposure(self) -> Decimal:
        """Calculate time-weighted exposure for risk assessment."""
        if not self._position_start_time or self._current_position == 0:
            return Decimal(0)

        hours_held = (
            datetime.now(UTC) - self._position_start_time
        ).total_seconds() / 3600
        return abs(self._current_position) * Decimal(str(hours_held))

    def _calculate_inventory_duration(self) -> float:
        """Calculate how long current inventory has been held."""
        if not self._position_start_time or self._current_position == 0:
            return 0.0

        return (datetime.now(UTC) - self._position_start_time).total_seconds() / 3600

    def _check_emergency_conditions(
        self, imbalance: float, market_price: Decimal
    ) -> RebalancingAction | None:
        """Check for emergency conditions requiring immediate action."""
        # Emergency threshold exceeded
        if abs(imbalance) >= self.emergency_threshold:
            self._emergency_flatten_count += 1

            if imbalance > 0:  # Long position, need to sell
                action_type = "SELL"
                quantity = abs(self._current_position)
            else:  # Short position, need to buy
                action_type = "BUY"
                quantity = abs(self._current_position)

            return RebalancingAction(
                action_type=action_type,
                quantity=quantity,
                urgency="EMERGENCY",
                reason=f"Emergency flatten: imbalance {imbalance:.1f}% exceeds {self.emergency_threshold}%",
                target_price=market_price,
                confidence=0.9,
            )

        return None

    def _check_timeout_conditions(
        self, imbalance: float, market_price: Decimal
    ) -> RebalancingAction | None:
        """Check for timeout conditions requiring position closure."""
        duration_hours = self._calculate_inventory_duration()

        if (
            duration_hours >= self.inventory_timeout_hours
            and self._current_position != 0
        ):
            if self._current_position > 0:  # Long position
                action_type = "SELL"
                quantity = abs(self._current_position)
            else:  # Short position
                action_type = "BUY"
                quantity = abs(self._current_position)

            return RebalancingAction(
                action_type=action_type,
                quantity=quantity,
                urgency="HIGH",
                reason=f"Inventory timeout: held for {duration_hours:.1f}h >= {self.inventory_timeout_hours}h",
                target_price=market_price,
                confidence=0.8,
            )

        return None

    def _calculate_rebalancing_action(
        self,
        imbalance: float,
        vumanchu_bias: VuManChuBias,
        market_price: Decimal,
    ) -> RebalancingAction:
        """Calculate normal rebalancing action with VuManChu integration."""
        # No rebalancing needed if within threshold
        if abs(imbalance) < self.rebalancing_threshold:
            return RebalancingAction(
                action_type="HOLD",
                quantity=Decimal(0),
                urgency="LOW",
                reason=f"Imbalance {imbalance:.1f}% within threshold {self.rebalancing_threshold}%",
                vumanchu_bias=vumanchu_bias.overall_bias,
                confidence=vumanchu_bias.confidence,
            )

        # Determine if VuManChu bias supports or conflicts with rebalancing
        bias_alignment = self._assess_bias_alignment(imbalance, vumanchu_bias)

        # Calculate rebalancing quantity
        if imbalance > 0:  # Long position - consider selling
            if (
                vumanchu_bias.overall_bias == "BEARISH"
                or bias_alignment == "CONFLICTING"
            ):
                # VuManChu supports selling or conflicts with long position
                action_type = "SELL"
                quantity = self._calculate_rebalancing_quantity(imbalance, "aggressive")
                urgency = "HIGH" if bias_alignment == "CONFLICTING" else "MEDIUM"
                reason = f"Rebalance long position: imbalance {imbalance:.1f}%, VuManChu {vumanchu_bias.overall_bias}"
            elif vumanchu_bias.overall_bias == "BULLISH":
                # VuManChu supports holding long - moderate rebalancing
                if abs(imbalance) > self.rebalancing_threshold * 1.5:
                    action_type = "SELL"
                    quantity = self._calculate_rebalancing_quantity(
                        imbalance, "conservative"
                    )
                    urgency = "LOW"
                    reason = f"Conservative rebalance: imbalance {imbalance:.1f}%, VuManChu bullish"
                else:
                    return RebalancingAction(
                        action_type="HOLD",
                        quantity=Decimal(0),
                        urgency="LOW",
                        reason="VuManChu bullish bias supports long position",
                        vumanchu_bias=vumanchu_bias.overall_bias,
                        confidence=vumanchu_bias.confidence,
                    )
            else:  # NEUTRAL
                action_type = "SELL"
                quantity = self._calculate_rebalancing_quantity(imbalance, "moderate")
                urgency = "MEDIUM"
                reason = (
                    f"Neutral rebalance: imbalance {imbalance:.1f}%, VuManChu neutral"
                )

        elif vumanchu_bias.overall_bias == "BULLISH" or bias_alignment == "CONFLICTING":
            # VuManChu supports buying or conflicts with short position
            action_type = "BUY"
            quantity = self._calculate_rebalancing_quantity(
                abs(imbalance), "aggressive"
            )
            urgency = "HIGH" if bias_alignment == "CONFLICTING" else "MEDIUM"
            reason = f"Rebalance short position: imbalance {imbalance:.1f}%, VuManChu {vumanchu_bias.overall_bias}"
        elif vumanchu_bias.overall_bias == "BEARISH":
            # VuManChu supports holding short - moderate rebalancing
            if abs(imbalance) > self.rebalancing_threshold * 1.5:
                action_type = "BUY"
                quantity = self._calculate_rebalancing_quantity(
                    abs(imbalance), "conservative"
                )
                urgency = "LOW"
                reason = f"Conservative rebalance: imbalance {imbalance:.1f}%, VuManChu bearish"
            else:
                return RebalancingAction(
                    action_type="HOLD",
                    quantity=Decimal(0),
                    urgency="LOW",
                    reason="VuManChu bearish bias supports short position",
                    vumanchu_bias=vumanchu_bias.overall_bias,
                    confidence=vumanchu_bias.confidence,
                )
        else:  # NEUTRAL
            action_type = "BUY"
            quantity = self._calculate_rebalancing_quantity(abs(imbalance), "moderate")
            urgency = "MEDIUM"
            reason = f"Neutral rebalance: imbalance {imbalance:.1f}%, VuManChu neutral"

        return RebalancingAction(
            action_type=action_type,
            quantity=quantity,
            urgency=urgency,
            reason=reason,
            target_price=market_price,
            vumanchu_bias=vumanchu_bias.overall_bias,
            confidence=vumanchu_bias.confidence,
        )

    def _assess_bias_alignment(
        self, imbalance: float, vumanchu_bias: VuManChuBias
    ) -> Literal["SUPPORTING", "CONFLICTING", "NEUTRAL"]:
        """Assess if VuManChu bias supports or conflicts with current position."""
        if imbalance > 0:  # Long position
            if vumanchu_bias.overall_bias == "BULLISH":
                return "SUPPORTING"
            if vumanchu_bias.overall_bias == "BEARISH":
                return "CONFLICTING"
        elif imbalance < 0:  # Short position
            if vumanchu_bias.overall_bias == "BEARISH":
                return "SUPPORTING"
            if vumanchu_bias.overall_bias == "BULLISH":
                return "CONFLICTING"

        return "NEUTRAL"

    def _calculate_rebalancing_quantity(
        self,
        imbalance_pct: float,
        style: Literal["conservative", "moderate", "aggressive"],
    ) -> Decimal:
        """Calculate rebalancing quantity based on imbalance and style."""
        base_position = abs(self._current_position)

        if style == "conservative":
            # Rebalance 25% of excess position
            excess_pct = max(0, imbalance_pct - self.rebalancing_threshold)
            rebalance_ratio = min(0.25, excess_pct / 100)
        elif style == "moderate":
            # Rebalance 50% of excess position
            excess_pct = max(0, imbalance_pct - self.rebalancing_threshold)
            rebalance_ratio = min(0.5, excess_pct / 100)
        else:  # aggressive
            # Rebalance 75% of excess position
            excess_pct = max(0, imbalance_pct - self.rebalancing_threshold)
            rebalance_ratio = min(0.75, excess_pct / 100)

        return base_position * Decimal(str(rebalance_ratio))

    def _simulate_rebalancing_execution(
        self, action: RebalancingAction, market_price: Decimal
    ) -> bool:
        """Simulate rebalancing execution (placeholder for real implementation)."""
        # In real implementation, this would interface with the order manager
        # For now, simulate successful execution and update position

        if action.action_type == "BUY":
            self._current_position += action.quantity
        elif action.action_type == "SELL":
            self._current_position -= action.quantity

        # Update average entry price
        if action.quantity > 0:
            self._update_average_entry_price(
                action.quantity if action.action_type == "BUY" else -action.quantity,
                market_price,
            )

        return True  # Simulate successful execution

    def _update_average_entry_price(
        self, position_change: Decimal, price: Decimal
    ) -> None:
        """Update weighted average entry price."""
        if self._average_entry_price is None:
            self._average_entry_price = price
            return

        # Calculate new weighted average
        old_position = self._current_position - position_change
        if old_position == 0:
            self._average_entry_price = price
        elif self._current_position == 0:
            # Position was closed, reset average price
            self._average_entry_price = None
        else:
            total_cost = (old_position * self._average_entry_price) + (
                position_change * price
            )
            self._average_entry_price = total_cost / self._current_position

    def _get_fallback_metrics(self) -> InventoryMetrics:
        """Get safe fallback metrics in case of errors."""
        return InventoryMetrics(
            symbol=self.symbol,
            net_position=Decimal(0),
            position_value=Decimal(0),
            imbalance_percentage=0.0,
            risk_score=0.0,
            max_position_limit=self._account_equity
            * (Decimal(str(self.max_position_pct)) / Decimal(100)),
            rebalancing_threshold=self.rebalancing_threshold,
            time_weighted_exposure=Decimal(0),
            inventory_duration_hours=0.0,
        )

    def _save_state(self) -> None:
        """Save inventory state to file."""
        try:
            # Ensure parent directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            state_data = {
                "symbol": self.symbol,
                "current_position": str(self._current_position),
                "average_entry_price": (
                    str(self._average_entry_price)
                    if self._average_entry_price
                    else None
                ),
                "position_start_time": (
                    self._position_start_time.isoformat()
                    if self._position_start_time
                    else None
                ),
                "account_equity": str(self._account_equity),
                "total_realized_pnl": str(self._total_realized_pnl),
                "rebalancing_success_count": self._rebalancing_success_count,
                "rebalancing_failure_count": self._rebalancing_failure_count,
                "emergency_flatten_count": self._emergency_flatten_count,
                "last_rebalancing_time": (
                    self._last_rebalancing_time.isoformat()
                    if self._last_rebalancing_time
                    else None
                ),
                "position_history": self._position_history[
                    -100:
                ],  # Keep last 100 entries
                "rebalancing_history": self._rebalancing_history[
                    -50:
                ],  # Keep last 50 entries
                "bias_history": self._bias_history[-50:],  # Keep last 50 entries
                "timestamp": datetime.now(UTC).isoformat(),
            }

            with self.state_file.open("w") as f:
                json.dump(state_data, f, indent=2)

            logger.debug("💾 Inventory state saved for %s", self.symbol)

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "state_file": str(self.state_file),
                    "inventory_state_save_error": True,
                },
                component="InventoryManager",
                operation="_save_state",
            )

    def _load_state(self) -> None:
        """Load inventory state from file."""
        try:
            if not self.state_file.exists():
                logger.info("📁 No existing inventory state file for %s", self.symbol)
                return

            with self.state_file.open() as f:
                state_data = json.load(f)

            # Restore state
            self._current_position = Decimal(state_data.get("current_position", "0"))

            avg_price_str = state_data.get("average_entry_price")
            self._average_entry_price = (
                Decimal(avg_price_str) if avg_price_str else None
            )

            start_time_str = state_data.get("position_start_time")
            self._position_start_time = (
                datetime.fromisoformat(start_time_str) if start_time_str else None
            )

            self._account_equity = Decimal(state_data.get("account_equity", "10000"))
            self._total_realized_pnl = Decimal(
                state_data.get("total_realized_pnl", "0")
            )
            self._rebalancing_success_count = state_data.get(
                "rebalancing_success_count", 0
            )
            self._rebalancing_failure_count = state_data.get(
                "rebalancing_failure_count", 0
            )
            self._emergency_flatten_count = state_data.get("emergency_flatten_count", 0)

            last_rebal_str = state_data.get("last_rebalancing_time")
            self._last_rebalancing_time = (
                datetime.fromisoformat(last_rebal_str) if last_rebal_str else None
            )

            self._position_history = state_data.get("position_history", [])
            self._rebalancing_history = state_data.get("rebalancing_history", [])
            self._bias_history = state_data.get("bias_history", [])

            logger.info(
                "📁 Inventory state loaded for %s: position=%s, avg_price=%s",
                self.symbol,
                self._current_position,
                self._average_entry_price,
            )

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "state_file": str(self.state_file),
                    "inventory_state_load_error": True,
                },
                component="InventoryManager",
                operation="_load_state",
            )
            logger.warning(
                "⚠️ Failed to load inventory state, starting fresh for %s", self.symbol
            )
