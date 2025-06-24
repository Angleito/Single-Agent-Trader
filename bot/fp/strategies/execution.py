"""
Functional execution algorithms for order generation and smart routing.

This module provides pure functional implementations of various execution
algorithms including TWAP, VWAP, and Iceberg orders with slippage modeling.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from bot.fp.core.either import Either, Left, Right
from bot.fp.strategies.signals import Signal, SignalType


class OrderType(Enum):
    """Order type enumeration."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "BUY"
    SELL = "SELL"


class ExecutionAlgorithm(Enum):
    """Execution algorithm types."""

    TWAP = "TWAP"
    VWAP = "VWAP"
    ICEBERG = "ICEBERG"
    AGGRESSIVE = "AGGRESSIVE"
    PASSIVE = "PASSIVE"


@dataclass(frozen=True)
class Order:
    """Immutable order representation."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Decimal | None
    stop_price: Decimal | None
    time_in_force: str
    order_id: str
    timestamp: datetime
    metadata: dict


@dataclass(frozen=True)
class ExecutionParams:
    """Parameters for execution algorithms."""

    algorithm: ExecutionAlgorithm
    duration: timedelta
    slice_count: int
    max_participation_rate: Decimal
    urgency: Decimal  # 0.0 to 1.0
    min_order_size: Decimal
    max_order_size: Decimal


@dataclass(frozen=True)
class MarketContext:
    """Market context for execution decisions."""

    bid: Decimal
    ask: Decimal
    mid: Decimal
    spread: Decimal
    volume: Decimal
    volatility: Decimal
    liquidity_score: Decimal  # 0.0 to 1.0


@dataclass(frozen=True)
class SlippageModel:
    """Slippage model parameters."""

    linear_impact: Decimal  # basis points per unit size
    square_root_impact: Decimal  # basis points per sqrt(size)
    temporary_impact: Decimal  # temporary price impact
    permanent_impact: Decimal  # permanent price impact


@dataclass(frozen=True)
class Venue:
    """Trading venue representation."""

    name: str
    liquidity_score: Decimal
    fee_rate: Decimal
    latency_ms: int
    min_order_size: Decimal
    max_order_size: Decimal


@dataclass(frozen=True)
class RouteDecision:
    """Order routing decision."""

    venue: Venue
    order: Order
    expected_cost: Decimal
    expected_slippage: Decimal


@dataclass(frozen=True)
class ExecutionState:
    """Execution algorithm state."""

    original_order: Order
    executed_quantity: Decimal
    remaining_quantity: Decimal
    average_price: Decimal
    pending_orders: list[Order]
    completed_orders: list[Order]
    failed_orders: list[tuple[Order, str]]


@dataclass(frozen=True)
class ExecutionCostAnalysis:
    """Post-execution cost analysis."""

    total_cost_bps: Decimal
    spread_cost_bps: Decimal
    impact_cost_bps: Decimal
    timing_cost_bps: Decimal
    opportunity_cost_bps: Decimal


@dataclass(frozen=True)
class ExecutionConfig:
    """Configuration for functional execution engine."""

    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.TWAP
    default_urgency: Decimal = Decimal("0.5")
    default_slice_count: int = 10
    max_participation_rate: Decimal = Decimal("0.2")
    min_order_size: Decimal = Decimal(10)
    max_order_size: Decimal = Decimal(100000)
    enable_smart_routing: bool = True
    enable_cost_analysis: bool = True

    def __post_init__(self) -> None:
        """Validate execution configuration."""
        if not 0 <= self.default_urgency <= 1:
            raise ValueError(
                f"Default urgency must be between 0 and 1: {self.default_urgency}"
            )
        if self.default_slice_count < 1:
            raise ValueError(
                f"Slice count must be positive: {self.default_slice_count}"
            )
        if not 0 < self.max_participation_rate <= 1:
            raise ValueError(
                f"Max participation rate must be between 0 and 1: {self.max_participation_rate}"
            )
        if self.min_order_size <= 0:
            raise ValueError(f"Min order size must be positive: {self.min_order_size}")
        if self.max_order_size <= self.min_order_size:
            raise ValueError("Max order size must be greater than min order size")


@dataclass(frozen=True)
class ExecutionResult:
    """Result of functional execution."""

    success: bool
    execution_state: ExecutionState
    cost_analysis: ExecutionCostAnalysis | None
    routing_decisions: list[RouteDecision]
    execution_time_ms: float
    metadata: dict[str, Any]
    error_message: str | None = None

    @property
    def is_complete(self) -> bool:
        """Check if execution is complete."""
        return self.execution_state.remaining_quantity == 0

    @property
    def fill_rate(self) -> Decimal:
        """Calculate fill rate."""
        original_qty = self.execution_state.original_order.quantity
        executed_qty = self.execution_state.executed_quantity
        return executed_qty / original_qty if original_qty > 0 else Decimal(0)


# Order Generation Functions
# --------------------------


def signal_to_order(
    signal: Signal,
    position_size: Decimal,
    market_context: MarketContext,
    order_type: OrderType = OrderType.MARKET,
) -> Either[str, Order]:
    """
    Convert a trading signal to an order.

    Args:
        signal: Trading signal
        position_size: Position size to trade
        market_context: Current market context
        order_type: Type of order to generate

    Returns:
        Either an error message or the generated order
    """
    # Determine order side
    if signal.signal_type == SignalType.LONG:
        side = OrderSide.BUY
        price = market_context.ask if order_type == OrderType.LIMIT else None
    elif signal.signal_type == SignalType.SHORT:
        side = OrderSide.SELL
        price = market_context.bid if order_type == OrderType.LIMIT else None
    else:
        return Left("Cannot generate order for HOLD signal")

    # Generate order
    order = Order(
        symbol=signal.symbol,
        side=side,
        order_type=order_type,
        quantity=abs(position_size),
        price=price,
        stop_price=None,
        time_in_force="GTC",
        order_id=f"{signal.symbol}_{datetime.now().timestamp()}",
        timestamp=datetime.now(),
        metadata={
            "signal_strength": signal.strength.value,
            "signal_confidence": signal.confidence,
            "signal_source": signal.source,
        },
    )

    return Right(order)


def calculate_slippage(
    order: Order, market_context: MarketContext, model: SlippageModel
) -> Decimal:
    """Calculate expected slippage for an order."""
    # Simplified slippage calculation
    participation_rate = (
        order.quantity / market_context.volume
        if market_context.volume > 0
        else Decimal(0)
    )

    linear_impact = model.linear_impact * participation_rate
    sqrt_impact = model.square_root_impact * (participation_rate ** Decimal("0.5"))

    return linear_impact + sqrt_impact


def estimate_execution_price(
    order: Order, market_context: MarketContext, model: SlippageModel
) -> Decimal:
    """Estimate execution price including slippage."""
    slippage_bps = calculate_slippage(order, market_context, model)
    slippage_factor = slippage_bps / Decimal(10000)

    if order.side == OrderSide.BUY:
        return market_context.mid * (Decimal(1) + slippage_factor)
    return market_context.mid * (Decimal(1) - slippage_factor)


def update_execution_state(
    state: ExecutionState,
    filled_order: Order,
    fill_price: Decimal,
    fill_quantity: Decimal,
) -> ExecutionState:
    """Update execution state with a fill."""
    # Update quantities
    new_executed_quantity = state.executed_quantity + fill_quantity
    new_remaining_quantity = state.remaining_quantity - fill_quantity

    # Update average price
    if state.executed_quantity == 0:
        new_average_price = fill_price
    else:
        total_value = (
            state.average_price * state.executed_quantity + fill_price * fill_quantity
        )
        new_average_price = total_value / new_executed_quantity

    # Update order lists
    new_pending = [
        o for o in state.pending_orders if o.order_id != filled_order.order_id
    ]
    new_completed = state.completed_orders + [filled_order]

    return ExecutionState(
        original_order=state.original_order,
        executed_quantity=new_executed_quantity,
        remaining_quantity=new_remaining_quantity,
        average_price=new_average_price,
        pending_orders=new_pending,
        completed_orders=new_completed,
        failed_orders=state.failed_orders,
    )


def analyze_execution_costs(
    state: ExecutionState,
    arrival_price: Decimal,
    benchmark_price: Decimal,
    market_context: MarketContext,
) -> ExecutionCostAnalysis:
    """Analyze execution costs post-trade."""
    # Implementation shortfall
    if state.original_order.side == OrderSide.BUY:
        implementation_shortfall = (state.average_price - arrival_price) / arrival_price
    else:
        implementation_shortfall = (arrival_price - state.average_price) / arrival_price

    total_cost_bps = implementation_shortfall * Decimal(10000)

    # Spread cost
    spread_cost_bps = market_context.spread / market_context.mid * Decimal(10000) / 2

    # Market impact
    impact_cost_bps = total_cost_bps - spread_cost_bps

    # Timing cost (vs benchmark)
    if state.original_order.side == OrderSide.BUY:
        timing_cost = (state.average_price - benchmark_price) / benchmark_price
    else:
        timing_cost = (benchmark_price - state.average_price) / benchmark_price

    timing_cost_bps = timing_cost * Decimal(10000)

    # Opportunity cost (unfilled portion)
    if state.remaining_quantity > 0:
        opportunity_cost_bps = (
            state.remaining_quantity / state.original_order.quantity
        ) * Decimal(100)
    else:
        opportunity_cost_bps = Decimal(0)

    return ExecutionCostAnalysis(
        total_cost_bps=total_cost_bps,
        spread_cost_bps=spread_cost_bps,
        impact_cost_bps=impact_cost_bps,
        timing_cost_bps=timing_cost_bps,
        opportunity_cost_bps=opportunity_cost_bps,
    )


def select_execution_algorithm(
    order: Order, market_context: MarketContext, urgency: Decimal
) -> ExecutionAlgorithm:
    """Select appropriate execution algorithm based on order and market characteristics."""
    # Calculate order size relative to market
    order_value = order.quantity * market_context.mid
    market_volume_value = market_context.volume * market_context.mid
    relative_size = (
        order_value / market_volume_value if market_volume_value > 0 else Decimal(1)
    )

    # High urgency -> aggressive execution
    if urgency > Decimal("0.8"):
        return ExecutionAlgorithm.AGGRESSIVE

    # Large order -> VWAP or Iceberg
    if relative_size > Decimal("0.1"):
        if market_context.liquidity_score > Decimal("0.7"):
            return ExecutionAlgorithm.VWAP
        return ExecutionAlgorithm.ICEBERG

    # Medium urgency -> TWAP
    if urgency > Decimal("0.5"):
        return ExecutionAlgorithm.TWAP

    # Low urgency -> Passive
    return ExecutionAlgorithm.PASSIVE


def smart_order_route(
    order: Order,
    venues: list[Venue],
    market_contexts: dict[str, MarketContext],
    slippage_model: SlippageModel,
) -> Either[str, list[RouteDecision]]:
    """Determine optimal routing for an order across multiple venues."""
    if not venues:
        return Left("No venues available")

    # Score each venue
    venue_scores = []

    for venue in venues:
        # Get market context for venue
        context = market_contexts.get(venue.name)
        if not context:
            continue

        # Check order size constraints
        if order.quantity < venue.min_order_size:
            continue

        # Calculate expected costs
        slippage = calculate_slippage(order, context, slippage_model)
        fee_cost = venue.fee_rate * Decimal(10000)  # Convert to bps
        latency_cost = Decimal(str(venue.latency_ms)) / Decimal(1000)  # Latency penalty

        total_cost = slippage + fee_cost + latency_cost

        # Score based on liquidity and cost
        score = venue.liquidity_score / (Decimal(1) + total_cost / Decimal(10000))

        venue_scores.append((venue, score, slippage, total_cost))

    if not venue_scores:
        return Left("No suitable venues found")

    # Sort by score (descending)
    venue_scores.sort(key=lambda x: x[1], reverse=True)

    # Route to best venue (can be extended for multi-venue routing)
    best_venue, _, expected_slippage, total_cost = venue_scores[0]

    route_decision = RouteDecision(
        venue=best_venue,
        order=order,
        expected_cost=total_cost,
        expected_slippage=expected_slippage,
    )

    return Right([route_decision])


class FunctionalExecutionEngine:
    """Functional execution engine using pure functions."""

    def __init__(self, config: ExecutionConfig):
        """Initialize execution engine with configuration."""
        self.config = config
        self.default_slippage_model = SlippageModel(
            linear_impact=Decimal(5),  # 5 bps per unit participation
            square_root_impact=Decimal(10),  # 10 bps per sqrt(participation)
            temporary_impact=Decimal(2),  # 2 bps temporary impact
            permanent_impact=Decimal(1),  # 1 bps permanent impact
        )

    def execute_signal(
        self,
        signal: Signal,
        position_size: Decimal,
        market_context: MarketContext,
        venues: list[Venue] = None,
    ) -> Either[str, ExecutionResult]:
        """Execute a trading signal using functional execution algorithms."""
        try:
            import time

            start_time = time.time()

            # Convert signal to order
            order_result = signal_to_order(
                signal, position_size, market_context, OrderType.MARKET
            )

            if isinstance(order_result, Left):
                return Left(f"Order generation failed: {order_result.value}")

            order = order_result.value

            # Initialize execution state
            execution_state = ExecutionState(
                original_order=order,
                executed_quantity=Decimal(0),
                remaining_quantity=order.quantity,
                average_price=Decimal(0),
                pending_orders=[],
                completed_orders=[],
                failed_orders=[],
            )

            # For simplicity, simulate immediate execution at market price
            # In real implementation, this would use the configured algorithm
            fill_price = estimate_execution_price(
                order, market_context, self.default_slippage_model
            )

            # Update execution state with full fill
            final_state = update_execution_state(
                execution_state, order, fill_price, order.quantity
            )

            # Perform cost analysis if enabled
            cost_analysis = None
            if self.config.enable_cost_analysis:
                cost_analysis = analyze_execution_costs(
                    final_state,
                    market_context.mid,  # arrival price
                    market_context.mid,  # benchmark price
                    market_context,
                )

            # Smart routing if enabled
            routing_decisions = []
            if self.config.enable_smart_routing and venues:
                market_contexts = {venue.name: market_context for venue in venues}
                routing_result = smart_order_route(
                    order, venues, market_contexts, self.default_slippage_model
                )
                if isinstance(routing_result, Right):
                    routing_decisions = routing_result.value

            end_time = time.time()

            return Right(
                ExecutionResult(
                    success=True,
                    execution_state=final_state,
                    cost_analysis=cost_analysis,
                    routing_decisions=routing_decisions,
                    execution_time_ms=(end_time - start_time) * 1000,
                    metadata={
                        "urgency": float(self.config.default_urgency),
                        "slice_count": self.config.default_slice_count,
                        "smart_routing_enabled": self.config.enable_smart_routing,
                    },
                    error_message=None,
                )
            )

        except Exception as e:
            return Left(f"Execution failed: {e!s}")

    def get_algorithm_recommendation(
        self,
        order: Order,
        market_context: MarketContext,
        urgency: Decimal = Decimal("0.5"),
    ) -> ExecutionAlgorithm:
        """Get algorithm recommendation for an order."""
        return select_execution_algorithm(order, market_context, urgency)
