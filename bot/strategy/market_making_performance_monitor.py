"""
Market Making Performance Monitoring System.

This module provides comprehensive real-time performance monitoring specifically
for market making operations, tracking key metrics like P&L, spread capture,
fill rates, fee efficiency, and inventory turnover.

Key Features:
- Real-time P&L tracking (gross and net of fees)
- Spread capture rate monitoring
- Fill rate and order efficiency analytics
- Fee efficiency and profitability metrics
- Inventory turnover and imbalance tracking
- VuManChu signal effectiveness analysis
- Performance alerts and risk management
- Thread-safe concurrent access
- Exportable data for analysis
"""

import logging
import statistics
import threading
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, NamedTuple

from ..exchange.bluefin_fee_calculator import BluefinFeeCalculator
from ..performance_monitor import (
    AlertLevel,
    PerformanceAlert,
    PerformanceMetric,
    PerformanceMonitor,
    PerformanceThresholds,
)
from ..trading_types import IndicatorData
from .inventory_manager import InventoryMetrics
from .market_making_order_manager import ManagedOrder
from .market_making_strategy import DirectionalBias, SpreadCalculation

logger = logging.getLogger(__name__)


class MarketMakingAlert(str, Enum):
    """Market making specific alert types."""

    LOW_FILL_RATE = "LOW_FILL_RATE"
    POOR_SPREAD_CAPTURE = "POOR_SPREAD_CAPTURE"
    HIGH_FEE_RATIO = "HIGH_FEE_RATIO"
    INVENTORY_IMBALANCE = "INVENTORY_IMBALANCE"
    LOW_TURNOVER = "LOW_TURNOVER"
    NEGATIVE_PNL_STREAK = "NEGATIVE_PNL_STREAK"
    SIGNAL_INEFFECTIVENESS = "SIGNAL_INEFFECTIVENESS"
    EXCESSIVE_SLIPPAGE = "EXCESSIVE_SLIPPAGE"


class TradePnL(NamedTuple):
    """Individual trade P&L record."""

    trade_id: str
    timestamp: datetime
    side: str
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    gross_pnl: Decimal
    fees_paid: Decimal
    net_pnl: Decimal
    holding_time_seconds: float
    spread_captured: Decimal


class SpreadAnalysis(NamedTuple):
    """Spread capture analysis."""

    target_spread: Decimal
    actual_spread: Decimal
    capture_rate: float  # 0.0 to 1.0
    efficiency_score: float  # 0.0 to 1.0


class FillAnalysis(NamedTuple):
    """Order fill analysis."""

    total_orders: int
    filled_orders: int
    partially_filled_orders: int
    cancelled_orders: int
    fill_rate: float  # 0.0 to 1.0
    average_fill_time_seconds: float
    fill_efficiency_score: float  # 0.0 to 1.0


class SignalEffectiveness(NamedTuple):
    """VuManChu signal effectiveness metrics."""

    total_signals: int
    profitable_signals: int
    success_rate: float  # 0.0 to 1.0
    average_pnl_per_signal: Decimal
    signal_confidence_correlation: float
    best_performing_signal_type: str


class SessionMetrics(NamedTuple):
    """Trading session performance metrics."""

    session_start: datetime
    total_trades: int
    total_volume: Decimal
    gross_pnl: Decimal
    total_fees: Decimal
    net_pnl: Decimal
    roi_percentage: float
    sharpe_ratio: float
    max_drawdown: Decimal
    profit_factor: float


class MarketMakingThresholds(PerformanceThresholds):
    """Extended performance thresholds for market making."""

    def __init__(self):
        super().__init__()

        # Market making specific thresholds
        self.min_fill_rate = 0.3  # 30% minimum fill rate
        self.min_spread_capture_rate = 0.6  # 60% minimum spread capture
        self.max_fee_ratio = 0.5  # Max 50% of profit consumed by fees
        self.max_inventory_imbalance = 0.2  # 20% max inventory imbalance
        self.min_turnover_rate = 2.0  # Minimum 2x daily turnover
        self.max_negative_pnl_streak = 5  # Max 5 consecutive negative trades
        self.min_signal_effectiveness = 0.4  # 40% minimum signal success rate
        self.max_slippage_bps = 5  # 5 basis points max slippage


class MarketMakingPerformanceMonitor:
    """
    Comprehensive performance monitoring system for market making operations.

    Tracks real-time metrics including P&L, spread capture, fill rates, fee efficiency,
    inventory turnover, and signal effectiveness. Provides performance alerts and
    detailed analytics for strategy optimization.
    """

    def __init__(
        self,
        fee_calculator: BluefinFeeCalculator,
        symbol: str = "BTC-PERP",
        thresholds: MarketMakingThresholds | None = None,
        max_history_size: int = 10000,
    ):
        """
        Initialize the Market Making Performance Monitor.

        Args:
            fee_calculator: Fee calculator for accurate cost analysis
            symbol: Trading symbol to monitor
            thresholds: Performance thresholds (uses defaults if None)
            max_history_size: Maximum number of records to keep in memory
        """
        self.fee_calculator = fee_calculator
        self.symbol = symbol
        self.thresholds = thresholds or MarketMakingThresholds()
        self.max_history_size = max_history_size

        # Initialize base performance monitor
        self.base_monitor = PerformanceMonitor(self.thresholds)

        # Trade tracking
        self.completed_trades: deque[TradePnL] = deque(maxlen=max_history_size)
        self.active_positions: dict[str, dict[str, Any]] = (
            {}
        )  # order_id -> position data
        self.trade_counter = 0

        # Order tracking
        self.order_history: deque[dict[str, Any]] = deque(maxlen=max_history_size)
        self.fill_events: deque[dict[str, Any]] = deque(maxlen=max_history_size)

        # Spread tracking
        self.spread_targets: deque[SpreadCalculation] = deque(maxlen=max_history_size)
        self.spread_captures: deque[SpreadAnalysis] = deque(maxlen=max_history_size)

        # Signal tracking
        self.signal_events: deque[dict[str, Any]] = deque(maxlen=max_history_size)
        self.signal_outcomes: deque[dict[str, Any]] = deque(maxlen=max_history_size)

        # Inventory tracking
        self.inventory_snapshots: deque[InventoryMetrics] = deque(
            maxlen=max_history_size
        )
        self.inventory_turns: deque[float] = deque(maxlen=100)  # Daily turnover rates

        # Session tracking
        self.session_start = datetime.now(UTC)
        self.session_metrics: SessionMetrics | None = None

        # Performance caching
        self._cached_metrics: dict[str, Any] = {}
        self._cache_timestamp = datetime.min
        self._cache_ttl = timedelta(seconds=5)  # 5-second cache

        # Thread safety
        self._lock = threading.RLock()

        # Alert tracking
        self.recent_alerts: deque[PerformanceAlert] = deque(maxlen=100)
        self.alert_cooldowns: dict[str, datetime] = defaultdict(lambda: datetime.min)

        logger.info(
            "Initialized MarketMakingPerformanceMonitor for %s with %d max history",
            symbol,
            max_history_size,
        )

    def record_order_fill(
        self,
        managed_order: ManagedOrder,
        fill_quantity: Decimal,
        fill_price: Decimal,
        fill_timestamp: datetime,
        fees_paid: Decimal,
    ) -> None:
        """
        Record an order fill event for performance tracking.

        Args:
            managed_order: The managed order that was filled
            fill_quantity: Quantity that was filled
            fill_price: Price at which the fill occurred
            fill_timestamp: Timestamp of the fill
            fees_paid: Fees paid for the fill
        """
        with self._lock:
            try:
                # Record fill event
                fill_event = {
                    "order_id": managed_order.order.id,
                    "symbol": managed_order.order.symbol,
                    "side": managed_order.order.side,
                    "level": managed_order.level,
                    "quantity": fill_quantity,
                    "price": fill_price,
                    "fees": fees_paid,
                    "timestamp": fill_timestamp,
                    "target_price": managed_order.target_price,
                    "order_created": managed_order.created_at,
                    "fill_delay_seconds": (
                        fill_timestamp - managed_order.created_at
                    ).total_seconds(),
                }

                self.fill_events.append(fill_event)

                # Update active position tracking
                position_key = f"{managed_order.order.symbol}_{managed_order.level}"
                if position_key not in self.active_positions:
                    self.active_positions[position_key] = {
                        "side": managed_order.order.side,
                        "entry_price": fill_price,
                        "entry_time": fill_timestamp,
                        "quantity": fill_quantity,
                        "fees_paid": fees_paid,
                        "target_spread": None,  # Will be set by record_spread_target
                    }
                else:
                    # Handle partial fills or position updates
                    pos = self.active_positions[position_key]
                    total_quantity = pos["quantity"] + fill_quantity
                    weighted_price = (
                        pos["entry_price"] * pos["quantity"]
                        + fill_price * fill_quantity
                    ) / total_quantity
                    pos["entry_price"] = weighted_price
                    pos["quantity"] = total_quantity
                    pos["fees_paid"] += fees_paid

                # Check for completed round-trip trades
                self._check_for_completed_trades(fill_event)

                # Record performance metrics
                self.base_monitor.add_metric(
                    PerformanceMetric(
                        name="market_making.fill_event",
                        value=float(fill_quantity),
                        timestamp=fill_timestamp,
                        unit="quantity",
                        tags={
                            "symbol": managed_order.order.symbol,
                            "side": managed_order.order.side,
                            "level": str(managed_order.level),
                        },
                    )
                )

                logger.debug(
                    "Recorded fill: %s %s %.6f @ %.6f (fees: %.6f)",
                    managed_order.order.side,
                    managed_order.order.symbol,
                    float(fill_quantity),
                    float(fill_price),
                    float(fees_paid),
                )

            except Exception as e:
                logger.exception("Error recording order fill: %s", e)

    def record_spread_target(
        self,
        spread_calc: SpreadCalculation,
        current_price: Decimal,
        timestamp: datetime,
        vumanchu_bias: DirectionalBias | None = None,
    ) -> None:
        """
        Record a spread target for later capture rate analysis.

        Args:
            spread_calc: Calculated optimal spread
            current_price: Current market price
            timestamp: Timestamp of the spread calculation
            vumanchu_bias: Optional VuManChu bias information
        """
        with self._lock:
            try:
                self.spread_targets.append(spread_calc)

                # Track spread target metrics
                spread_pct = float(spread_calc.adjusted_spread / current_price * 100)

                self.base_monitor.add_metric(
                    PerformanceMetric(
                        name="market_making.spread_target",
                        value=spread_pct,
                        timestamp=timestamp,
                        unit="percentage",
                        tags={
                            "symbol": self.symbol,
                            "bias": (
                                vumanchu_bias.direction if vumanchu_bias else "neutral"
                            ),
                        },
                    )
                )

                logger.debug(
                    "Recorded spread target: %.3f%% (bias: %s)",
                    spread_pct,
                    vumanchu_bias.direction if vumanchu_bias else "neutral",
                )

            except Exception as e:
                logger.exception("Error recording spread target: %s", e)

    def record_vumanchu_signal(
        self,
        indicators: IndicatorData,
        bias: DirectionalBias,
        timestamp: datetime,
    ) -> str:
        """
        Record a VuManChu signal event for effectiveness tracking.

        Args:
            indicators: Indicator data containing signal values
            bias: Calculated directional bias
            timestamp: Signal timestamp

        Returns:
            Signal ID for later outcome tracking
        """
        with self._lock:
            try:
                signal_id = (
                    f"signal_{int(timestamp.timestamp())}_{len(self.signal_events)}"
                )

                signal_event = {
                    "signal_id": signal_id,
                    "timestamp": timestamp,
                    "bias_direction": bias.direction,
                    "bias_strength": bias.strength,
                    "bias_confidence": bias.confidence,
                    "cipher_a_dot": indicators.cipher_a_dot,
                    "cipher_b_wave": indicators.cipher_b_wave,
                    "cipher_b_money_flow": indicators.cipher_b_money_flow,
                    "rsi": indicators.rsi,
                    "ema_fast": indicators.ema_fast,
                    "ema_slow": indicators.ema_slow,
                    "outcome_recorded": False,
                }

                self.signal_events.append(signal_event)

                # Track signal metrics
                self.base_monitor.add_metric(
                    PerformanceMetric(
                        name="market_making.vumanchu_signal",
                        value=bias.strength,
                        timestamp=timestamp,
                        unit="strength",
                        tags={
                            "symbol": self.symbol,
                            "direction": bias.direction,
                            "confidence": f"{bias.confidence:.2f}",
                        },
                    )
                )

                logger.debug(
                    "Recorded VuManChu signal %s: %s (strength=%.2f, confidence=%.2f)",
                    signal_id,
                    bias.direction,
                    bias.strength,
                    bias.confidence,
                )

                return signal_id

            except Exception as e:
                logger.exception("Error recording VuManChu signal: %s", e)
                return ""

    def record_signal_outcome(
        self,
        signal_id: str,
        pnl_result: Decimal,
        success: bool,
        outcome_timestamp: datetime,
    ) -> None:
        """
        Record the outcome of a VuManChu signal for effectiveness analysis.

        Args:
            signal_id: ID of the signal to update
            pnl_result: P&L result from trades influenced by this signal
            success: Whether the signal led to profitable trades
            outcome_timestamp: Timestamp when outcome was determined
        """
        with self._lock:
            try:
                # Find the signal event
                signal_event = None
                for event in reversed(self.signal_events):
                    if event["signal_id"] == signal_id:
                        signal_event = event
                        break

                if not signal_event:
                    logger.warning(
                        "Signal ID not found for outcome recording: %s", signal_id
                    )
                    return

                # Record outcome
                outcome = {
                    "signal_id": signal_id,
                    "original_timestamp": signal_event["timestamp"],
                    "outcome_timestamp": outcome_timestamp,
                    "duration_seconds": (
                        outcome_timestamp - signal_event["timestamp"]
                    ).total_seconds(),
                    "pnl_result": pnl_result,
                    "success": success,
                    "bias_direction": signal_event["bias_direction"],
                    "bias_strength": signal_event["bias_strength"],
                    "bias_confidence": signal_event["bias_confidence"],
                }

                self.signal_outcomes.append(outcome)
                signal_event["outcome_recorded"] = True

                # Track signal effectiveness metrics
                self.base_monitor.add_metric(
                    PerformanceMetric(
                        name="market_making.signal_outcome",
                        value=float(pnl_result),
                        timestamp=outcome_timestamp,
                        unit="pnl",
                        tags={
                            "symbol": self.symbol,
                            "success": str(success),
                            "direction": signal_event["bias_direction"],
                        },
                    )
                )

                logger.debug(
                    "Recorded signal outcome %s: success=%s, pnl=%.6f",
                    signal_id,
                    success,
                    float(pnl_result),
                )

            except Exception as e:
                logger.exception("Error recording signal outcome: %s", e)

    def record_inventory_snapshot(self, inventory_metrics: InventoryMetrics) -> None:
        """
        Record an inventory snapshot for turnover and imbalance tracking.

        Args:
            inventory_metrics: Current inventory metrics
        """
        with self._lock:
            try:
                self.inventory_snapshots.append(inventory_metrics)

                # Track inventory metrics
                self.base_monitor.add_metric(
                    PerformanceMetric(
                        name="market_making.inventory_imbalance",
                        value=inventory_metrics.imbalance_percentage,
                        timestamp=inventory_metrics.timestamp,
                        unit="percentage",
                        tags={"symbol": inventory_metrics.symbol},
                    )
                )

                self.base_monitor.add_metric(
                    PerformanceMetric(
                        name="market_making.inventory_value",
                        value=float(inventory_metrics.position_value),
                        timestamp=inventory_metrics.timestamp,
                        unit="value",
                        tags={"symbol": inventory_metrics.symbol},
                    )
                )

                # Calculate and record turnover rate if we have enough history
                if len(self.inventory_snapshots) >= 2:
                    turnover_rate = self._calculate_inventory_turnover()
                    if turnover_rate > 0:
                        self.inventory_turns.append(turnover_rate)

                        self.base_monitor.add_metric(
                            PerformanceMetric(
                                name="market_making.inventory_turnover",
                                value=turnover_rate,
                                timestamp=inventory_metrics.timestamp,
                                unit="rate",
                                tags={"symbol": inventory_metrics.symbol},
                            )
                        )

                logger.debug(
                    "Recorded inventory snapshot: imbalance=%.2f%%, value=%.2f",
                    inventory_metrics.imbalance_percentage,
                    float(inventory_metrics.position_value),
                )

            except Exception as e:
                logger.exception("Error recording inventory snapshot: %s", e)

    def get_real_time_pnl(
        self, current_prices: dict[str, Decimal]
    ) -> dict[str, Decimal]:
        """
        Calculate real-time P&L including unrealized positions.

        Args:
            current_prices: Dictionary of symbol -> current price

        Returns:
            Dictionary with P&L breakdown
        """
        with self._lock:
            try:
                realized_pnl = sum(trade.net_pnl for trade in self.completed_trades)
                unrealized_pnl = Decimal(0)
                total_fees = sum(trade.fees_paid for trade in self.completed_trades)

                # Calculate unrealized P&L for active positions
                current_price = current_prices.get(self.symbol, Decimal(0))
                if current_price > 0:
                    for position in self.active_positions.values():
                        if position["side"] == "BUY":
                            unrealized_pnl += (
                                current_price - position["entry_price"]
                            ) * position["quantity"]
                        else:  # SELL
                            unrealized_pnl += (
                                position["entry_price"] - current_price
                            ) * position["quantity"]

                        total_fees += position["fees_paid"]

                gross_pnl = realized_pnl + unrealized_pnl
                net_pnl = gross_pnl - total_fees

                pnl_breakdown = {
                    "realized_pnl": realized_pnl,
                    "unrealized_pnl": unrealized_pnl,
                    "gross_pnl": gross_pnl,
                    "total_fees": total_fees,
                    "net_pnl": net_pnl,
                    "fee_percentage": (
                        (total_fees / abs(gross_pnl) * 100)
                        if gross_pnl != 0
                        else Decimal(0)
                    ),
                }

                # Track real-time P&L metrics
                self.base_monitor.add_metric(
                    PerformanceMetric(
                        name="market_making.real_time_pnl",
                        value=float(net_pnl),
                        timestamp=datetime.now(UTC),
                        unit="pnl",
                        tags={"symbol": self.symbol, "type": "net"},
                    )
                )

                return pnl_breakdown

            except Exception as e:
                logger.exception("Error calculating real-time P&L: %s", e)
                return {
                    "realized_pnl": Decimal(0),
                    "unrealized_pnl": Decimal(0),
                    "gross_pnl": Decimal(0),
                    "total_fees": Decimal(0),
                    "net_pnl": Decimal(0),
                    "fee_percentage": Decimal(0),
                }

    def get_performance_metrics(
        self, time_window: timedelta = timedelta(hours=1)
    ) -> dict[str, Any]:
        """
        Get comprehensive performance metrics for the specified time window.

        Args:
            time_window: Time window for metric calculation

        Returns:
            Dictionary with performance metrics
        """
        with self._lock:
            try:
                # Check cache
                current_time = datetime.now(UTC)
                if (current_time - self._cache_timestamp) < self._cache_ttl:
                    return self._cached_metrics.copy()

                cutoff_time = current_time - time_window

                # Filter data by time window
                recent_trades = [
                    t for t in self.completed_trades if t.timestamp >= cutoff_time
                ]
                recent_fills = [
                    f for f in self.fill_events if f["timestamp"] >= cutoff_time
                ]
                recent_spreads = [
                    s
                    for s in self.spread_captures
                    if hasattr(s, "timestamp") and s.timestamp >= cutoff_time
                ]
                recent_signals = [
                    s
                    for s in self.signal_outcomes
                    if s["outcome_timestamp"] >= cutoff_time
                ]

                # Calculate metrics
                metrics = {
                    "timestamp": current_time.isoformat(),
                    "time_window_hours": time_window.total_seconds() / 3600,
                    "symbol": self.symbol,
                    # P&L Metrics
                    "total_trades": len(recent_trades),
                    "gross_pnl": float(sum(t.gross_pnl for t in recent_trades)),
                    "net_pnl": float(sum(t.net_pnl for t in recent_trades)),
                    "total_fees": float(sum(t.fees_paid for t in recent_trades)),
                    "average_trade_pnl": (
                        float(
                            sum(t.net_pnl for t in recent_trades) / len(recent_trades)
                        )
                        if recent_trades
                        else 0.0
                    ),
                    "winning_trades": len([t for t in recent_trades if t.net_pnl > 0]),
                    "losing_trades": len([t for t in recent_trades if t.net_pnl < 0]),
                    "win_rate": (
                        len([t for t in recent_trades if t.net_pnl > 0])
                        / len(recent_trades)
                        if recent_trades
                        else 0.0
                    ),
                    # Fill Rate Metrics
                    "total_fills": len(recent_fills),
                    "buy_fills": len([f for f in recent_fills if f["side"] == "BUY"]),
                    "sell_fills": len([f for f in recent_fills if f["side"] == "SELL"]),
                    "average_fill_time": (
                        statistics.mean([f["fill_delay_seconds"] for f in recent_fills])
                        if recent_fills
                        else 0.0
                    ),
                    "fill_efficiency": self._calculate_fill_efficiency(recent_fills),
                    # Spread Metrics
                    "spread_capture_rate": self._calculate_spread_capture_rate(
                        recent_spreads
                    ),
                    "average_spread_captured": (
                        float(statistics.mean([s.capture_rate for s in recent_spreads]))
                        if recent_spreads
                        else 0.0
                    ),
                    # Fee Efficiency
                    "fee_efficiency_ratio": self._calculate_fee_efficiency_ratio(
                        recent_trades
                    ),
                    # Signal Effectiveness
                    "signal_success_rate": (
                        len([s for s in recent_signals if s["success"]])
                        / len(recent_signals)
                        if recent_signals
                        else 0.0
                    ),
                    "average_signal_pnl": (
                        float(
                            statistics.mean(
                                [float(s["pnl_result"]) for s in recent_signals]
                            )
                        )
                        if recent_signals
                        else 0.0
                    ),
                    # Inventory Metrics
                    "current_inventory_imbalance": (
                        self.inventory_snapshots[-1].imbalance_percentage
                        if self.inventory_snapshots
                        else 0.0
                    ),
                    "average_inventory_turnover": (
                        statistics.mean(list(self.inventory_turns))
                        if self.inventory_turns
                        else 0.0
                    ),
                    # Risk Metrics
                    "max_drawdown": self._calculate_max_drawdown(recent_trades),
                    "sharpe_ratio": self._calculate_sharpe_ratio(
                        recent_trades, time_window
                    ),
                    "profit_factor": self._calculate_profit_factor(recent_trades),
                    # Volume Metrics
                    "total_volume": float(
                        sum(t.quantity * t.entry_price for t in recent_trades)
                    ),
                    "average_trade_size": (
                        float(statistics.mean([t.quantity for t in recent_trades]))
                        if recent_trades
                        else 0.0
                    ),
                }

                # Cache results
                self._cached_metrics = metrics
                self._cache_timestamp = current_time

                return metrics.copy()

            except Exception as e:
                logger.exception("Error calculating performance metrics: %s", e)
                return {"error": str(e), "timestamp": datetime.now(UTC).isoformat()}

    def get_dashboard_data(self) -> dict[str, Any]:
        """
        Get real-time dashboard data for monitoring interfaces.

        Returns:
            Dictionary with dashboard-ready data
        """
        with self._lock:
            try:
                current_time = datetime.now(UTC)

                # Get metrics for different time windows
                metrics_1h = self.get_performance_metrics(timedelta(hours=1))
                metrics_24h = self.get_performance_metrics(timedelta(hours=24))
                metrics_7d = self.get_performance_metrics(timedelta(days=7))

                # Current session metrics
                session_duration = (
                    current_time - self.session_start
                ).total_seconds() / 3600

                dashboard_data = {
                    "timestamp": current_time.isoformat(),
                    "symbol": self.symbol,
                    "session_duration_hours": session_duration,
                    # Real-time metrics
                    "current_performance": {
                        "1h": {
                            "net_pnl": metrics_1h.get("net_pnl", 0.0),
                            "trades": metrics_1h.get("total_trades", 0),
                            "win_rate": metrics_1h.get("win_rate", 0.0),
                            "fill_efficiency": metrics_1h.get("fill_efficiency", 0.0),
                            "spread_capture": metrics_1h.get(
                                "spread_capture_rate", 0.0
                            ),
                        },
                        "24h": {
                            "net_pnl": metrics_24h.get("net_pnl", 0.0),
                            "trades": metrics_24h.get("total_trades", 0),
                            "win_rate": metrics_24h.get("win_rate", 0.0),
                            "volume": metrics_24h.get("total_volume", 0.0),
                        },
                        "7d": {
                            "net_pnl": metrics_7d.get("net_pnl", 0.0),
                            "trades": metrics_7d.get("total_trades", 0),
                            "sharpe_ratio": metrics_7d.get("sharpe_ratio", 0.0),
                            "max_drawdown": metrics_7d.get("max_drawdown", 0.0),
                        },
                    },
                    # Current status
                    "active_positions": len(self.active_positions),
                    "inventory_imbalance": (
                        self.inventory_snapshots[-1].imbalance_percentage
                        if self.inventory_snapshots
                        else 0.0
                    ),
                    "recent_signals": len(
                        [s for s in self.signal_events if not s["outcome_recorded"]]
                    ),
                    # Performance alerts
                    "recent_alerts": [
                        {
                            "level": alert.level.value,
                            "message": alert.message,
                            "timestamp": alert.timestamp.isoformat(),
                        }
                        for alert in list(self.recent_alerts)[-5:]  # Last 5 alerts
                    ],
                    # Health score
                    "health_score": self._calculate_health_score(metrics_1h),
                }

                return dashboard_data

            except Exception as e:
                logger.exception("Error generating dashboard data: %s", e)
                return {"error": str(e), "timestamp": datetime.now(UTC).isoformat()}

    def check_performance_alerts(self) -> list[PerformanceAlert]:
        """
        Check for performance alerts based on current metrics.

        Returns:
            List of new performance alerts
        """
        with self._lock:
            try:
                alerts = []
                current_time = datetime.now(UTC)
                metrics = self.get_performance_metrics(timedelta(hours=1))

                # Check fill rate
                fill_efficiency = metrics.get("fill_efficiency", 0.0)
                if fill_efficiency < self.thresholds.min_fill_rate:
                    if self._can_send_alert(MarketMakingAlert.LOW_FILL_RATE):
                        alerts.append(
                            PerformanceAlert(
                                level=AlertLevel.WARNING,
                                message=f"Low fill rate: {fill_efficiency:.1%} (threshold: {self.thresholds.min_fill_rate:.1%})",
                                metric_name="fill_efficiency",
                                current_value=fill_efficiency,
                                threshold=self.thresholds.min_fill_rate,
                                timestamp=current_time,
                                tags={"symbol": self.symbol},
                            )
                        )

                # Check spread capture rate
                spread_capture = metrics.get("spread_capture_rate", 0.0)
                if spread_capture < self.thresholds.min_spread_capture_rate:
                    if self._can_send_alert(MarketMakingAlert.POOR_SPREAD_CAPTURE):
                        alerts.append(
                            PerformanceAlert(
                                level=AlertLevel.WARNING,
                                message=f"Poor spread capture: {spread_capture:.1%} (threshold: {self.thresholds.min_spread_capture_rate:.1%})",
                                metric_name="spread_capture_rate",
                                current_value=spread_capture,
                                threshold=self.thresholds.min_spread_capture_rate,
                                timestamp=current_time,
                                tags={"symbol": self.symbol},
                            )
                        )

                # Check fee efficiency
                fee_ratio = metrics.get("fee_efficiency_ratio", 0.0)
                if fee_ratio > self.thresholds.max_fee_ratio:
                    if self._can_send_alert(MarketMakingAlert.HIGH_FEE_RATIO):
                        alerts.append(
                            PerformanceAlert(
                                level=AlertLevel.WARNING,
                                message=f"High fee ratio: {fee_ratio:.1%} (threshold: {self.thresholds.max_fee_ratio:.1%})",
                                metric_name="fee_efficiency_ratio",
                                current_value=fee_ratio,
                                threshold=self.thresholds.max_fee_ratio,
                                timestamp=current_time,
                                tags={"symbol": self.symbol},
                            )
                        )

                # Check inventory imbalance
                inventory_imbalance = abs(
                    metrics.get("current_inventory_imbalance", 0.0)
                )
                if (
                    inventory_imbalance > self.thresholds.max_inventory_imbalance * 100
                ):  # Convert to percentage
                    if self._can_send_alert(MarketMakingAlert.INVENTORY_IMBALANCE):
                        alerts.append(
                            PerformanceAlert(
                                level=AlertLevel.WARNING,
                                message=f"High inventory imbalance: {inventory_imbalance:.1f}% (threshold: {self.thresholds.max_inventory_imbalance * 100:.1f}%)",
                                metric_name="inventory_imbalance",
                                current_value=inventory_imbalance,
                                threshold=self.thresholds.max_inventory_imbalance * 100,
                                timestamp=current_time,
                                tags={"symbol": self.symbol},
                            )
                        )

                # Check signal effectiveness
                signal_success = metrics.get("signal_success_rate", 0.0)
                if (
                    signal_success < self.thresholds.min_signal_effectiveness
                    and signal_success > 0
                ):
                    if self._can_send_alert(MarketMakingAlert.SIGNAL_INEFFECTIVENESS):
                        alerts.append(
                            PerformanceAlert(
                                level=AlertLevel.WARNING,
                                message=f"Low signal effectiveness: {signal_success:.1%} (threshold: {self.thresholds.min_signal_effectiveness:.1%})",
                                metric_name="signal_success_rate",
                                current_value=signal_success,
                                threshold=self.thresholds.min_signal_effectiveness,
                                timestamp=current_time,
                                tags={"symbol": self.symbol},
                            )
                        )

                # Add alerts to tracking
                for alert in alerts:
                    self.recent_alerts.append(alert)
                    self.alert_cooldowns[alert.metric_name] = current_time

                return alerts

            except Exception as e:
                logger.exception("Error checking performance alerts: %s", e)
                return []

    def export_performance_data(
        self,
        time_window: timedelta = timedelta(hours=24),
        include_raw_data: bool = False,
    ) -> dict[str, Any]:
        """
        Export performance data for analysis and reporting.

        Args:
            time_window: Time window for data export
            include_raw_data: Whether to include raw trade/fill data

        Returns:
            Dictionary with exportable performance data
        """
        with self._lock:
            try:
                cutoff_time = datetime.now(UTC) - time_window

                export_data = {
                    "export_timestamp": datetime.now(UTC).isoformat(),
                    "symbol": self.symbol,
                    "time_window_hours": time_window.total_seconds() / 3600,
                    "session_start": self.session_start.isoformat(),
                    # Performance metrics
                    "performance_metrics": self.get_performance_metrics(time_window),
                    # Dashboard data
                    "dashboard_data": self.get_dashboard_data(),
                    # Alert summary
                    "alert_summary": {
                        "total_alerts": len(self.recent_alerts),
                        "alert_types": list(
                            set(alert.metric_name for alert in self.recent_alerts)
                        ),
                        "recent_alerts": [
                            {
                                "level": alert.level.value,
                                "message": alert.message,
                                "metric": alert.metric_name,
                                "timestamp": alert.timestamp.isoformat(),
                            }
                            for alert in self.recent_alerts
                            if alert.timestamp >= cutoff_time
                        ],
                    },
                    # Configuration
                    "thresholds": {
                        "min_fill_rate": self.thresholds.min_fill_rate,
                        "min_spread_capture_rate": self.thresholds.min_spread_capture_rate,
                        "max_fee_ratio": self.thresholds.max_fee_ratio,
                        "max_inventory_imbalance": self.thresholds.max_inventory_imbalance,
                        "min_signal_effectiveness": self.thresholds.min_signal_effectiveness,
                    },
                }

                # Include raw data if requested
                if include_raw_data:
                    export_data["raw_data"] = {
                        "completed_trades": [
                            {
                                "trade_id": t.trade_id,
                                "timestamp": t.timestamp.isoformat(),
                                "side": t.side,
                                "entry_price": str(t.entry_price),
                                "exit_price": str(t.exit_price),
                                "quantity": str(t.quantity),
                                "gross_pnl": str(t.gross_pnl),
                                "fees_paid": str(t.fees_paid),
                                "net_pnl": str(t.net_pnl),
                                "holding_time_seconds": t.holding_time_seconds,
                                "spread_captured": str(t.spread_captured),
                            }
                            for t in self.completed_trades
                            if t.timestamp >= cutoff_time
                        ],
                        "fill_events": [
                            {
                                "order_id": f["order_id"],
                                "side": f["side"],
                                "quantity": str(f["quantity"]),
                                "price": str(f["price"]),
                                "fees": str(f["fees"]),
                                "timestamp": f["timestamp"].isoformat(),
                                "fill_delay_seconds": f["fill_delay_seconds"],
                            }
                            for f in self.fill_events
                            if f["timestamp"] >= cutoff_time
                        ],
                        "signal_outcomes": [
                            {
                                "signal_id": s["signal_id"],
                                "bias_direction": s["bias_direction"],
                                "success": s["success"],
                                "pnl_result": str(s["pnl_result"]),
                                "duration_seconds": s["duration_seconds"],
                                "timestamp": s["outcome_timestamp"].isoformat(),
                            }
                            for s in self.signal_outcomes
                            if s["outcome_timestamp"] >= cutoff_time
                        ],
                    }

                return export_data

            except Exception as e:
                logger.exception("Error exporting performance data: %s", e)
                return {"error": str(e), "timestamp": datetime.now(UTC).isoformat()}

    # Private helper methods

    def _check_for_completed_trades(self, fill_event: dict[str, Any]) -> None:
        """Check if a fill completes any round-trip trades."""
        try:
            # Simple round-trip detection: look for opposite side fills at different levels
            # This is a simplified implementation - real market making might have more complex logic

            opposite_side = "SELL" if fill_event["side"] == "BUY" else "BUY"

            # Look for matching opposite position
            for pos_key, position in list(self.active_positions.items()):
                if (
                    position["side"] == opposite_side
                    and position["quantity"] <= fill_event["quantity"]
                ):
                    # Calculate trade P&L
                    if fill_event["side"] == "SELL":
                        # We're selling (closing a long position)
                        gross_pnl = (
                            fill_event["price"] - position["entry_price"]
                        ) * position["quantity"]
                    else:
                        # We're buying (closing a short position)
                        gross_pnl = (
                            position["entry_price"] - fill_event["price"]
                        ) * position["quantity"]

                    total_fees = position["fees_paid"] + fill_event["fees"]
                    net_pnl = gross_pnl - total_fees
                    holding_time = (
                        fill_event["timestamp"] - position["entry_time"]
                    ).total_seconds()

                    # Calculate spread captured
                    target_spread = position.get("target_spread", Decimal(0))
                    actual_spread = abs(fill_event["price"] - position["entry_price"])
                    spread_captured = (
                        min(actual_spread, target_spread)
                        if target_spread > 0
                        else actual_spread
                    )

                    # Create trade record
                    self.trade_counter += 1
                    trade = TradePnL(
                        trade_id=f"trade_{self.trade_counter}",
                        timestamp=fill_event["timestamp"],
                        side=position["side"],
                        entry_price=position["entry_price"],
                        exit_price=fill_event["price"],
                        quantity=position["quantity"],
                        gross_pnl=gross_pnl,
                        fees_paid=total_fees,
                        net_pnl=net_pnl,
                        holding_time_seconds=holding_time,
                        spread_captured=spread_captured,
                    )

                    self.completed_trades.append(trade)

                    # Remove completed position
                    del self.active_positions[pos_key]

                    # Update remaining quantity in fill if partial
                    if fill_event["quantity"] > position["quantity"]:
                        fill_event["quantity"] -= position["quantity"]
                    else:
                        break

                    logger.info(
                        "Completed trade %s: %s %.6f, P&L: %.6f (gross: %.6f, fees: %.6f)",
                        trade.trade_id,
                        trade.side,
                        float(trade.quantity),
                        float(trade.net_pnl),
                        float(trade.gross_pnl),
                        float(trade.fees_paid),
                    )

        except Exception as e:
            logger.exception("Error checking for completed trades: %s", e)

    def _calculate_fill_efficiency(self, recent_fills: list[dict[str, Any]]) -> float:
        """Calculate fill efficiency score based on fill times and rates."""
        if not recent_fills:
            return 0.0

        try:
            # Calculate average fill time
            avg_fill_time = statistics.mean(
                [f["fill_delay_seconds"] for f in recent_fills]
            )

            # Fill efficiency is inverse of fill time (normalized)
            # Assume target fill time is 5 seconds for full efficiency
            target_fill_time = 5.0
            time_efficiency = max(
                0.0, min(1.0, target_fill_time / max(avg_fill_time, target_fill_time))
            )

            # Factor in fill rate (fills per minute)
            fills_per_minute = (
                len(recent_fills) * 60.0 / 3600.0
            )  # Assuming 1-hour window
            rate_efficiency = min(
                1.0, fills_per_minute / 10.0
            )  # Target 10 fills per minute

            # Combined efficiency
            return (time_efficiency + rate_efficiency) / 2.0

        except Exception:
            return 0.0

    def _calculate_spread_capture_rate(
        self, recent_spreads: list[SpreadAnalysis]
    ) -> float:
        """Calculate average spread capture rate."""
        if not recent_spreads:
            return 0.0

        try:
            return statistics.mean([s.capture_rate for s in recent_spreads])
        except Exception:
            return 0.0

    def _calculate_fee_efficiency_ratio(self, recent_trades: list[TradePnL]) -> float:
        """Calculate the ratio of fees to gross profit."""
        if not recent_trades:
            return 0.0

        try:
            total_fees = sum(t.fees_paid for t in recent_trades)
            total_gross_profit = sum(
                max(t.gross_pnl, Decimal(0)) for t in recent_trades
            )

            if total_gross_profit == 0:
                return 1.0  # All fees, no profit

            return float(total_fees / total_gross_profit)

        except Exception:
            return 0.0

    def _calculate_inventory_turnover(self) -> float:
        """Calculate inventory turnover rate."""
        if len(self.inventory_snapshots) < 2:
            return 0.0

        try:
            # Simple turnover calculation based on position value changes
            recent_snapshot = self.inventory_snapshots[-1]
            previous_snapshot = self.inventory_snapshots[-2]

            time_diff_hours = (
                recent_snapshot.timestamp - previous_snapshot.timestamp
            ).total_seconds() / 3600.0
            if time_diff_hours == 0:
                return 0.0

            # Calculate position value change
            value_change = abs(
                recent_snapshot.position_value - previous_snapshot.position_value
            )
            avg_position_value = (
                recent_snapshot.position_value + previous_snapshot.position_value
            ) / 2

            if avg_position_value == 0:
                return 0.0

            # Turnover rate per hour
            hourly_turnover = float(value_change / avg_position_value)

            # Annualize (multiply by hours in a year)
            return hourly_turnover * 24 * 365

        except Exception:
            return 0.0

    def _calculate_max_drawdown(self, recent_trades: list[TradePnL]) -> float:
        """Calculate maximum drawdown from recent trades."""
        if not recent_trades:
            return 0.0

        try:
            # Calculate cumulative P&L
            cumulative_pnl = []
            running_total = Decimal(0)

            for trade in sorted(recent_trades, key=lambda t: t.timestamp):
                running_total += trade.net_pnl
                cumulative_pnl.append(float(running_total))

            # Find maximum drawdown
            peak = cumulative_pnl[0]
            max_drawdown = 0.0

            for pnl in cumulative_pnl:
                peak = max(peak, pnl)

                drawdown = peak - pnl
                max_drawdown = max(max_drawdown, drawdown)

            return max_drawdown

        except Exception:
            return 0.0

    def _calculate_sharpe_ratio(
        self, recent_trades: list[TradePnL], time_window: timedelta
    ) -> float:
        """Calculate Sharpe ratio for recent trades."""
        if len(recent_trades) < 2:
            return 0.0

        try:
            # Calculate returns
            returns = [float(t.net_pnl) for t in recent_trades]

            # Calculate mean and standard deviation
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)

            if std_return == 0:
                return 0.0

            # Annualize (assuming trades are representative of the time window)
            trades_per_year = (
                len(recent_trades) * (365 * 24) / (time_window.total_seconds() / 3600)
            )
            annualized_return = mean_return * trades_per_year
            annualized_std = std_return * (trades_per_year**0.5)

            return annualized_return / annualized_std if annualized_std > 0 else 0.0

        except Exception:
            return 0.0

    def _calculate_profit_factor(self, recent_trades: list[TradePnL]) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if not recent_trades:
            return 0.0

        try:
            gross_profit = sum(float(t.net_pnl) for t in recent_trades if t.net_pnl > 0)
            gross_loss = abs(
                sum(float(t.net_pnl) for t in recent_trades if t.net_pnl < 0)
            )

            if gross_loss == 0:
                return float("inf") if gross_profit > 0 else 0.0

            return gross_profit / gross_loss

        except Exception:
            return 0.0

    def _calculate_health_score(self, metrics: dict[str, Any]) -> float:
        """Calculate overall health score (0-100) based on performance metrics."""
        try:
            score = 100.0

            # Deduct for poor performance
            if metrics.get("win_rate", 0.0) < 0.5:
                score -= 15

            if metrics.get("fill_efficiency", 0.0) < self.thresholds.min_fill_rate:
                score -= 20

            if (
                metrics.get("spread_capture_rate", 0.0)
                < self.thresholds.min_spread_capture_rate
            ):
                score -= 15

            if metrics.get("fee_efficiency_ratio", 0.0) > self.thresholds.max_fee_ratio:
                score -= 10

            if (
                abs(metrics.get("current_inventory_imbalance", 0.0))
                > self.thresholds.max_inventory_imbalance * 100
            ):
                score -= 10

            if (
                metrics.get("signal_success_rate", 0.0)
                < self.thresholds.min_signal_effectiveness
            ):
                score -= 10

            # Deduct for negative P&L
            if metrics.get("net_pnl", 0.0) < 0:
                score -= 20

            return max(0.0, min(100.0, score))

        except Exception:
            return 50.0  # Default neutral score

    def _can_send_alert(self, alert_type: MarketMakingAlert) -> bool:
        """Check if an alert can be sent (not in cooldown)."""
        last_alert_time = self.alert_cooldowns.get(alert_type.value, datetime.min)
        cooldown_period = timedelta(minutes=5)  # 5-minute cooldown
        return datetime.now(UTC) - last_alert_time > cooldown_period

    async def start_monitoring(self) -> None:
        """Start the performance monitoring system."""
        await self.base_monitor.start_monitoring()
        logger.info("Market making performance monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop the performance monitoring system."""
        await self.base_monitor.stop_monitoring()
        logger.info("Market making performance monitoring stopped")
