"""Market making strategy implementation."""

from bot.fp.core import MarketState, Signal, SignalType, Strategy
from bot.fp.indicators.volatility import bollinger_bands


def calculate_spread(
    volatility: float,
    base_spread: float,
    volatility_multiplier: float = 2.0,
    min_spread: float = 0.001,
    max_spread: float = 0.05,
) -> float:
    """Calculate spread based on volatility.

    Args:
        volatility: Current market volatility (standard deviation)
        base_spread: Base spread percentage
        volatility_multiplier: How much volatility affects spread
        min_spread: Minimum allowed spread
        max_spread: Maximum allowed spread

    Returns:
        Calculated spread percentage
    """
    spread = base_spread + (volatility * volatility_multiplier)
    return max(min_spread, min(spread, max_spread))


def calculate_inventory_skew(
    current_inventory: float, max_inventory: float, skew_factor: float = 0.5
) -> float:
    """Calculate price skew based on inventory position.

    Args:
        current_inventory: Current inventory position (-max to +max)
        max_inventory: Maximum allowed inventory (absolute value)
        skew_factor: How aggressively to skew prices (0-1)

    Returns:
        Skew adjustment factor (-1 to 1)
    """
    if max_inventory == 0:
        return 0.0

    # Normalize inventory to -1 to 1 range
    normalized_inventory = current_inventory / max_inventory

    # Apply skew factor (positive inventory = lower asks, higher bids)
    return -normalized_inventory * skew_factor


def calculate_order_book_imbalance(
    bid_volume: float, ask_volume: float, imbalance_threshold: float = 0.7
) -> float:
    """Calculate order book imbalance.

    Args:
        bid_volume: Total bid volume in order book
        ask_volume: Total ask volume in order book
        imbalance_threshold: Threshold for significant imbalance

    Returns:
        Imbalance ratio (-1 to 1, positive = more bids)
    """
    total_volume = bid_volume + ask_volume
    if total_volume == 0:
        return 0.0

    imbalance = (bid_volume - ask_volume) / total_volume

    # Apply threshold to filter noise
    if abs(imbalance) < imbalance_threshold:
        return 0.0

    return imbalance


def generate_quotes(
    mid_price: float,
    spread: float,
    inventory_skew: float,
    order_book_imbalance: float = 0.0,
    competitive_adjustment: float = 0.9,
) -> tuple[float, float]:
    """Generate bid and ask quotes.

    Args:
        mid_price: Current mid market price
        spread: Base spread to apply
        inventory_skew: Inventory-based price adjustment
        order_book_imbalance: Order book imbalance adjustment
        competitive_adjustment: Factor to tighten spread for competitiveness

    Returns:
        Tuple of (bid_price, ask_price)
    """
    # Apply competitive adjustment to spread
    adjusted_spread = spread * competitive_adjustment

    # Calculate half spread for each side
    half_spread = adjusted_spread / 2

    # Apply inventory skew (positive skew = higher prices)
    bid_adjustment = half_spread * (1 - inventory_skew)
    ask_adjustment = half_spread * (1 + inventory_skew)

    # Apply order book imbalance (positive = more bids, so widen ask)
    if order_book_imbalance > 0:
        ask_adjustment *= 1 + abs(order_book_imbalance) * 0.2
    else:
        bid_adjustment *= 1 + abs(order_book_imbalance) * 0.2

    bid_price = mid_price * (1 - bid_adjustment)
    ask_price = mid_price * (1 + ask_adjustment)

    return bid_price, ask_price


def check_inventory_limits(
    current_inventory: float, max_inventory: float, proposed_size: float, side: str
) -> bool:
    """Check if trade would exceed inventory limits.

    Args:
        current_inventory: Current inventory position
        max_inventory: Maximum allowed inventory
        proposed_size: Size of proposed trade
        side: 'buy' or 'sell'

    Returns:
        True if trade is within limits
    """
    if side == "buy":
        new_inventory = current_inventory + proposed_size
    else:
        new_inventory = current_inventory - proposed_size

    return abs(new_inventory) <= max_inventory


def market_maker_strategy(
    spread_factor: float = 0.002,
    inventory_limit: float = 10000.0,
    skew_factor: float = 0.3,
    volatility_lookback: int = 20,
    min_spread: float = 0.001,
    max_spread: float = 0.02,
    competitive_factor: float = 0.95,
    imbalance_threshold: float = 0.6,
    quote_size: float = 100.0,
) -> Strategy:
    """Create a market making strategy.

    Args:
        spread_factor: Base spread as fraction of price
        inventory_limit: Maximum inventory position (absolute)
        skew_factor: How aggressively to skew based on inventory (0-1)
        volatility_lookback: Periods for volatility calculation
        min_spread: Minimum allowed spread
        max_spread: Maximum allowed spread
        competitive_factor: Quote competitiveness adjustment (0-1)
        imbalance_threshold: Order book imbalance threshold
        quote_size: Default quote size

    Returns:
        Market making strategy function
    """

    def strategy(market: MarketState) -> Signal | None:
        """Market making strategy implementation."""
        candles = market.candles

        if len(candles) < volatility_lookback + 1:
            return None

        current_candle = candles[-1]
        mid_price = (current_candle.high + current_candle.low) / 2

        # Calculate volatility using Bollinger Bands
        bb_result = bollinger_bands(candles, period=volatility_lookback, num_std=2.0)

        if not bb_result:
            return None

        # Use band width as volatility measure
        volatility = (bb_result.upper[-1] - bb_result.lower[-1]) / bb_result.middle[-1]

        # Calculate dynamic spread
        spread = calculate_spread(
            volatility=volatility,
            base_spread=spread_factor,
            volatility_multiplier=2.0,
            min_spread=min_spread,
            max_spread=max_spread,
        )

        # Get current inventory from metadata
        metadata = market.metadata or {}
        current_inventory = metadata.get("inventory", 0.0)
        bid_volume = metadata.get("bid_volume", 1000.0)
        ask_volume = metadata.get("ask_volume", 1000.0)

        # Calculate adjustments
        inventory_skew = calculate_inventory_skew(
            current_inventory=current_inventory,
            max_inventory=inventory_limit,
            skew_factor=skew_factor,
        )

        order_book_imbalance = calculate_order_book_imbalance(
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            imbalance_threshold=imbalance_threshold,
        )

        # Generate quotes
        bid_price, ask_price = generate_quotes(
            mid_price=mid_price,
            spread=spread,
            inventory_skew=inventory_skew,
            order_book_imbalance=order_book_imbalance,
            competitive_adjustment=competitive_factor,
        )

        # Risk checks
        risk_metadata = {
            "current_inventory": current_inventory,
            "inventory_limit": inventory_limit,
            "inventory_utilization": (
                abs(current_inventory) / inventory_limit if inventory_limit > 0 else 0
            ),
            "spread": spread,
            "volatility": volatility,
            "inventory_skew": inventory_skew,
            "order_book_imbalance": order_book_imbalance,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "mid_price": mid_price,
            "quote_size": quote_size,
        }

        # Check if we should halt market making
        if abs(current_inventory) >= inventory_limit * 0.95:
            # Near inventory limit, only allow reducing positions
            if current_inventory > 0:
                # Long inventory, only allow sells
                return Signal(
                    type=SignalType.MARKET_MAKE,
                    strength=0.5,
                    entry=ask_price,
                    metadata={
                        **risk_metadata,
                        "mode": "inventory_reduction",
                        "side": "sell_only",
                    },
                )
            # Short inventory, only allow buys
            return Signal(
                type=SignalType.MARKET_MAKE,
                strength=0.5,
                entry=bid_price,
                metadata={
                    **risk_metadata,
                    "mode": "inventory_reduction",
                    "side": "buy_only",
                },
            )

        # Normal market making with both sides
        return Signal(
            type=SignalType.MARKET_MAKE,
            strength=1.0,
            entry=mid_price,  # Use mid price as reference
            metadata={
                **risk_metadata,
                "mode": "two_sided",
                "quotes": {
                    "bid": {"price": bid_price, "size": quote_size},
                    "ask": {"price": ask_price, "size": quote_size},
                },
            },
        )

    return strategy


def market_maker_with_stops(
    base_strategy: Strategy,
    stop_loss_pct: float = 0.02,
    daily_loss_limit: float = 1000.0,
    position_limit: float = 50000.0,
) -> Strategy:
    """Add risk management stops to market making strategy.

    Args:
        base_strategy: Base market making strategy
        stop_loss_pct: Stop loss percentage per position
        daily_loss_limit: Maximum daily loss allowed
        position_limit: Maximum position value allowed

    Returns:
        Enhanced strategy with risk stops
    """

    def strategy(market: MarketState) -> Signal | None:
        """Market making with risk management."""
        # Get base signal
        signal = base_strategy(market)

        if not signal or signal.type != SignalType.MARKET_MAKE:
            return signal

        # Get risk metrics from metadata
        metadata = market.metadata or {}
        daily_pnl = metadata.get("daily_pnl", 0.0)
        position_value = metadata.get("position_value", 0.0)

        # Check daily loss limit
        if daily_pnl <= -daily_loss_limit:
            return Signal(
                type=SignalType.HOLD,
                strength=1.0,
                entry=signal.entry,
                metadata={
                    "reason": "daily_loss_limit_reached",
                    "daily_pnl": daily_pnl,
                    "limit": daily_loss_limit,
                },
            )

        # Check position limit
        if abs(position_value) >= position_limit:
            # Modify signal to only reduce position
            signal_metadata = signal.metadata or {}
            current_inventory = signal_metadata.get("current_inventory", 0.0)

            if current_inventory > 0:
                signal_metadata["side"] = "sell_only"
            else:
                signal_metadata["side"] = "buy_only"

            signal_metadata["reason"] = "position_limit_reached"
            signal.metadata = signal_metadata

        # Add stop loss to metadata
        if signal.metadata:
            signal.metadata["stop_loss_pct"] = stop_loss_pct
            signal.metadata["risk_limits"] = {
                "daily_loss_limit": daily_loss_limit,
                "position_limit": position_limit,
                "current_daily_pnl": daily_pnl,
                "current_position_value": position_value,
            }

        return signal

    return strategy


def adaptive_market_maker(
    base_spread: float = 0.002,
    min_spread: float = 0.0005,
    max_spread: float = 0.05,
    adaptation_rate: float = 0.1,
) -> Strategy:
    """Create an adaptive market making strategy that learns from fill rates.

    Args:
        base_spread: Starting spread
        min_spread: Minimum allowed spread
        max_spread: Maximum allowed spread
        adaptation_rate: How quickly to adapt spread (0-1)

    Returns:
        Adaptive market making strategy
    """
    # State for adaptation
    current_spread = base_spread

    def strategy(market: MarketState) -> Signal | None:
        """Adaptive market making implementation."""
        nonlocal current_spread

        # Get fill rate from metadata
        metadata = market.metadata or {}
        fill_rate = metadata.get("fill_rate", 0.5)  # Default 50%
        target_fill_rate = metadata.get("target_fill_rate", 0.7)  # Target 70%

        # Adapt spread based on fill rate
        if fill_rate < target_fill_rate:
            # Too few fills, tighten spread
            current_spread *= 1 - adaptation_rate
        else:
            # Too many fills, widen spread
            current_spread *= 1 + adaptation_rate

        # Enforce limits
        current_spread = max(min_spread, min(current_spread, max_spread))

        # Create base strategy with adapted spread
        mm_strategy = market_maker_strategy(
            spread_factor=current_spread, inventory_limit=10000.0, skew_factor=0.3
        )

        # Get signal and add adaptation info
        signal = mm_strategy(market)

        if signal and signal.metadata:
            signal.metadata["adaptive_spread"] = current_spread
            signal.metadata["fill_rate"] = fill_rate
            signal.metadata["target_fill_rate"] = target_fill_rate

        return signal

    return strategy
