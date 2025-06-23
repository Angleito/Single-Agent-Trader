"""
Functional execution algorithms for order generation and smart routing.

This module provides pure functional implementations of various execution
algorithms including TWAP, VWAP, and Iceberg orders with slippage modeling.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

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


# Order Generation
# ----------------


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
        return Left("Cannot generate order for NEUTRAL signal")

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
            "signal_strength": float(signal.strength),
            "signal_confidence": float(signal.confidence),
            "signal_source": signal.metadata.get("source", "unknown"),
        },
    )

    return Right(order)


# Slippage Modeling
# -----------------


def calculate_slippage(
    order: Order, market_context: MarketContext, model: SlippageModel
) -> Decimal:
    """
    Calculate expected slippage for an order.

    Args:
        order: Order to execute
        market_context: Current market context
        model: Slippage model parameters

    Returns:
        Expected slippage in basis points
    """
    # Base slippage from spread
    spread_cost = market_context.spread / market_context.mid * Decimal(10000) / 2

    # Market impact
    participation_rate = order.quantity * market_context.mid / market_context.volume

    # Linear impact
    linear_impact = model.linear_impact * participation_rate

    # Square root impact (for larger orders)
    sqrt_impact = model.square_root_impact * (participation_rate ** Decimal("0.5"))

    # Urgency adjustment
    urgency_factor = (
        Decimal("1.5") if order.order_type == OrderType.MARKET else Decimal("1.0")
    )

    # Total slippage
    total_slippage = (spread_cost + linear_impact + sqrt_impact) * urgency_factor

    # Adjust for volatility
    volatility_adjustment = Decimal(1) + (market_context.volatility / Decimal(100))

    return total_slippage * volatility_adjustment


def estimate_execution_price(
    order: Order, market_context: MarketContext, model: SlippageModel
) -> Decimal:
    """
    Estimate the execution price including slippage.

    Args:
        order: Order to execute
        market_context: Current market context
        model: Slippage model

    Returns:
        Estimated execution price
    """
    # Get base price
    if order.side == OrderSide.BUY:
        base_price = market_context.ask
    else:
        base_price = market_context.bid

    # Calculate slippage
    slippage_bps = calculate_slippage(order, market_context, model)
    slippage_factor = slippage_bps / Decimal(10000)

    # Apply slippage
    if order.side == OrderSide.BUY:
        execution_price = base_price * (Decimal(1) + slippage_factor)
    else:
        execution_price = base_price * (Decimal(1) - slippage_factor)

    return execution_price


# Order Splitting
# ---------------


def split_order_uniform(
    order: Order, slice_count: int, min_size: Decimal
) -> Either[str, list[Order]]:
    """
    Split an order into uniform slices.

    Args:
        order: Order to split
        slice_count: Number of slices
        min_size: Minimum order size

    Returns:
        Either an error or list of child orders
    """
    if slice_count <= 0:
        return Left("Slice count must be positive")

    slice_size = order.quantity / Decimal(str(slice_count))

    if slice_size < min_size:
        return Left(f"Slice size {slice_size} below minimum {min_size}")

    # Create child orders
    child_orders = []
    remaining = order.quantity

    for i in range(slice_count):
        # Last slice gets any remainder
        if i == slice_count - 1:
            child_quantity = remaining
        else:
            child_quantity = slice_size
            remaining -= slice_size

        child_order = Order(
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=child_quantity,
            price=order.price,
            stop_price=order.stop_price,
            time_in_force=order.time_in_force,
            order_id=f"{order.order_id}_slice_{i}",
            timestamp=order.timestamp,
            metadata={
                **order.metadata,
                "parent_order_id": order.order_id,
                "slice_index": i,
                "total_slices": slice_count,
            },
        )
        child_orders.append(child_order)

    return Right(child_orders)


def split_order_vwap_weighted(
    order: Order, volume_profile: list[tuple[datetime, Decimal]], min_size: Decimal
) -> Either[str, list[tuple[datetime, Order]]]:
    """
    Split an order weighted by volume profile.

    Args:
        order: Order to split
        volume_profile: List of (time, volume) tuples
        min_size: Minimum order size

    Returns:
        Either an error or list of (time, order) tuples
    """
    if not volume_profile:
        return Left("Volume profile is empty")

    # Calculate total volume
    total_volume = sum(vol for _, vol in volume_profile)
    if total_volume == 0:
        return Left("Total volume is zero")

    # Create weighted orders
    scheduled_orders = []
    remaining = order.quantity

    for i, (time, volume) in enumerate(volume_profile):
        # Calculate proportion
        proportion = volume / total_volume

        # Last slice gets remainder
        if i == len(volume_profile) - 1:
            child_quantity = remaining
        else:
            child_quantity = order.quantity * proportion
            child_quantity = max(child_quantity, min_size)
            remaining -= child_quantity

        if child_quantity >= min_size:
            child_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=child_quantity,
                price=order.price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force,
                order_id=f"{order.order_id}_vwap_{i}",
                timestamp=time,
                metadata={
                    **order.metadata,
                    "parent_order_id": order.order_id,
                    "vwap_slice": i,
                    "volume_weight": float(proportion),
                },
            )
            scheduled_orders.append((time, child_order))

    return Right(scheduled_orders)


# Execution Algorithms
# --------------------


def twap_schedule(
    order: Order, params: ExecutionParams, start_time: datetime
) -> Either[str, list[tuple[datetime, Order]]]:
    """
    Generate TWAP (Time-Weighted Average Price) execution schedule.

    Args:
        order: Order to execute
        params: Execution parameters
        start_time: Start time for execution

    Returns:
        Either an error or scheduled orders
    """
    # Split order uniformly
    split_result = split_order_uniform(order, params.slice_count, params.min_order_size)

    if isinstance(split_result, Left):
        return split_result

    child_orders = split_result.value

    # Calculate time intervals
    interval = params.duration / params.slice_count

    # Schedule orders
    scheduled_orders = []
    current_time = start_time

    for child_order in child_orders:
        scheduled_orders.append((current_time, child_order))
        current_time += interval

    return Right(scheduled_orders)


def vwap_schedule(
    order: Order,
    params: ExecutionParams,
    volume_profile: list[tuple[datetime, Decimal]],
) -> Either[str, list[tuple[datetime, Order]]]:
    """
    Generate VWAP (Volume-Weighted Average Price) execution schedule.

    Args:
        order: Order to execute
        params: Execution parameters
        volume_profile: Historical volume profile

    Returns:
        Either an error or scheduled orders
    """
    return split_order_vwap_weighted(order, volume_profile, params.min_order_size)


def iceberg_orders(
    order: Order, params: ExecutionParams, visible_percent: Decimal = Decimal("0.1")
) -> Either[str, tuple[Order, list[Order]]]:
    """
    Generate iceberg order structure.

    Args:
        order: Order to execute
        params: Execution parameters
        visible_percent: Percentage of order to show

    Returns:
        Either an error or (visible_order, hidden_orders)
    """
    # Calculate visible and hidden quantities
    visible_quantity = order.quantity * visible_percent

    visible_quantity = max(visible_quantity, params.min_order_size)

    if visible_quantity >= order.quantity:
        return Left("Visible quantity exceeds total order")

    hidden_quantity = order.quantity - visible_quantity

    # Create visible order
    visible_order = Order(
        symbol=order.symbol,
        side=order.side,
        order_type=OrderType.LIMIT,
        quantity=visible_quantity,
        price=order.price,
        stop_price=order.stop_price,
        time_in_force="GTC",
        order_id=f"{order.order_id}_visible",
        timestamp=order.timestamp,
        metadata={**order.metadata, "iceberg": True, "order_part": "visible"},
    )

    # Split hidden quantity
    hidden_slices = int(hidden_quantity / visible_quantity) + 1
    split_result = split_order_uniform(
        Order(
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=hidden_quantity,
            price=order.price,
            stop_price=order.stop_price,
            time_in_force=order.time_in_force,
            order_id=f"{order.order_id}_hidden",
            timestamp=order.timestamp,
            metadata={**order.metadata, "iceberg": True, "order_part": "hidden"},
        ),
        hidden_slices,
        params.min_order_size,
    )

    if isinstance(split_result, Left):
        return split_result

    return Right((visible_order, split_result.value))


# Smart Order Routing
# -------------------


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


def smart_order_route(
    order: Order,
    venues: list[Venue],
    market_contexts: dict[str, MarketContext],
    slippage_model: SlippageModel,
) -> Either[str, list[RouteDecision]]:
    """
    Determine optimal routing for an order across multiple venues.

    Args:
        order: Order to route
        venues: Available trading venues
        market_contexts: Market context per venue
        slippage_model: Slippage model

    Returns:
        Either an error or routing decisions
    """
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


# Execution State Management
# --------------------------


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


def update_execution_state(
    state: ExecutionState,
    filled_order: Order,
    fill_price: Decimal,
    fill_quantity: Decimal,
) -> ExecutionState:
    """
    Update execution state with a fill.

    Args:
        state: Current execution state
        filled_order: Order that was filled
        fill_price: Execution price
        fill_quantity: Filled quantity

    Returns:
        Updated execution state
    """
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


# Execution Algorithm Selection
# -----------------------------


def select_execution_algorithm(
    order: Order, market_context: MarketContext, urgency: Decimal
) -> ExecutionAlgorithm:
    """
    Select appropriate execution algorithm based on order and market characteristics.

    Args:
        order: Order to execute
        market_context: Current market context
        urgency: Urgency score (0.0 to 1.0)

    Returns:
        Recommended execution algorithm
    """
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


# Execution Cost Analysis
# -----------------------


@dataclass(frozen=True)
class ExecutionCostAnalysis:
    """Post-execution cost analysis."""

    total_cost_bps: Decimal
    spread_cost_bps: Decimal
    impact_cost_bps: Decimal
    timing_cost_bps: Decimal
    opportunity_cost_bps: Decimal


def analyze_execution_costs(
    state: ExecutionState,
    arrival_price: Decimal,
    benchmark_price: Decimal,
    market_context: MarketContext,
) -> ExecutionCostAnalysis:
    """
    Analyze execution costs post-trade.

    Args:
        state: Final execution state
        arrival_price: Price at order arrival
        benchmark_price: Benchmark price (e.g., VWAP)
        market_context: Market context during execution

    Returns:
        Execution cost analysis
    """
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
