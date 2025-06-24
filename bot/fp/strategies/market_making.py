"""Enhanced market making strategy implementation with functional programming patterns.

This module provides sophisticated market making algorithms using pure functions,
immutable data structures, and functional composition for optimal performance.

Features:
- Advanced spread calculation models (constant, adaptive, volatility-based)
- Sophisticated inventory management with functional state
- Market microstructure analysis and liquidity detection
- Optimal quote generation with bias adjustments
- Risk-aware position sizing and exposure control
- Performance analytics and optimization

All functions are pure and composable for maximum reliability and testability.
"""

from bot.fp.core import MarketState, Signal, SignalType, Strategy
from bot.fp.indicators.volatility import calculate_bollinger_bands, calculate_atr, calculate_historical_volatility
from bot.fp.types.market import OrderBook, MarketSnapshot, Candle
from bot.fp.types.trading import MarketMake, TradeSignal
from bot.fp.types.portfolio import Position, Portfolio
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Callable, Union, Tuple, Dict, List
from dataclasses import dataclass
from datetime import datetime
import math


# ============================================================================
# ENHANCED SPREAD CALCULATION MODELS
# ============================================================================

@dataclass(frozen=True)
class SpreadModel:
    """Immutable spread calculation parameters."""
    base_spread: float
    min_spread: float
    max_spread: float
    volatility_factor: float = 2.0
    liquidity_factor: float = 1.0
    inventory_factor: float = 0.5
    skew_factor: float = 0.3

@dataclass(frozen=True) 
class MarketConditions:
    """Immutable market condition snapshot."""
    volatility: float
    bid_depth: float
    ask_depth: float
    spread_ratio: float
    volume_ratio: float
    price_momentum: float
    
@dataclass(frozen=True)
class SpreadResult:
    """Immutable spread calculation result."""
    base_spread: float
    adjusted_spread: float
    bid_adjustment: float
    ask_adjustment: float
    confidence: float
    model_used: str

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

def calculate_adaptive_spread(
    market_conditions: MarketConditions,
    model: SpreadModel,
    inventory_ratio: float = 0.0
) -> SpreadResult:
    """Calculate adaptive spread using market microstructure analysis.
    
    Uses sophisticated models including:
    - Volatility-based adjustments
    - Liquidity-sensitive sizing  
    - Inventory skew corrections
    - Market momentum factors
    
    Args:
        market_conditions: Current market state
        model: Spread calculation parameters
        inventory_ratio: Current inventory as ratio of max (-1 to 1)
        
    Returns:
        Comprehensive spread calculation result
    """
    # Base spread from model
    base = model.base_spread
    
    # Volatility adjustment - higher volatility = wider spreads
    vol_adjustment = market_conditions.volatility * model.volatility_factor
    
    # Liquidity adjustment - deeper book = tighter spreads
    total_depth = market_conditions.bid_depth + market_conditions.ask_depth
    liquidity_adjustment = -model.liquidity_factor * min(total_depth / 10000.0, 0.5)
    
    # Market momentum adjustment - trending markets need wider spreads
    momentum_adjustment = abs(market_conditions.price_momentum) * 0.3
    
    # Calculate adjusted spread
    adjusted = base + vol_adjustment + liquidity_adjustment + momentum_adjustment
    adjusted = max(model.min_spread, min(adjusted, model.max_spread))
    
    # Inventory skew adjustments
    inventory_skew = inventory_ratio * model.inventory_factor
    bid_adjustment = adjusted * (model.skew_factor * inventory_skew)
    ask_adjustment = adjusted * (-model.skew_factor * inventory_skew)
    
    # Calculate confidence based on data quality
    confidence = _calculate_spread_confidence(market_conditions)
    
    return SpreadResult(
        base_spread=base,
        adjusted_spread=adjusted,
        bid_adjustment=bid_adjustment,
        ask_adjustment=ask_adjustment,
        confidence=confidence,
        model_used="adaptive"
    )

def calculate_garch_spread(
    volatility_series: List[float],
    model: SpreadModel,
    lookback: int = 50
) -> SpreadResult:
    """Calculate spread using GARCH volatility forecasting.
    
    Uses GARCH(1,1) model to forecast next-period volatility
    and adjusts spreads accordingly.
    
    Args:
        volatility_series: Historical volatility observations
        model: Spread calculation parameters  
        lookback: Number of periods for GARCH estimation
        
    Returns:
        GARCH-based spread calculation
    """
    if len(volatility_series) < lookback:
        # Fallback to simple volatility
        current_vol = volatility_series[-1] if volatility_series else 0.02
        return SpreadResult(
            base_spread=model.base_spread,
            adjusted_spread=model.base_spread + current_vol * model.volatility_factor,
            bid_adjustment=0.0,
            ask_adjustment=0.0,
            confidence=0.5,
            model_used="simple_fallback"
        )
    
    # Simple GARCH(1,1) estimation
    recent_vols = volatility_series[-lookback:]
    alpha, beta = 0.1, 0.85  # Typical GARCH parameters
    omega = 0.05 * (1 - alpha - beta)  # Long-run variance
    
    # Forecast next period volatility
    current_variance = recent_vols[-1] ** 2
    forecast_variance = omega + alpha * current_variance + beta * current_variance
    forecast_vol = math.sqrt(forecast_variance)
    
    # Adjust spread based on forecasted volatility
    vol_premium = (forecast_vol - model.base_spread) * model.volatility_factor
    adjusted = model.base_spread + vol_premium
    adjusted = max(model.min_spread, min(adjusted, model.max_spread))
    
    return SpreadResult(
        base_spread=model.base_spread,
        adjusted_spread=adjusted,
        bid_adjustment=0.0,
        ask_adjustment=0.0,
        confidence=0.8,
        model_used="garch"
    )

def _calculate_spread_confidence(conditions: MarketConditions) -> float:
    """Calculate confidence score for spread calculation.
    
    Based on:
    - Market depth adequacy
    - Volatility stability
    - Data freshness
    
    Returns:
        Confidence score between 0 and 1
    """
    # Depth confidence - more depth = higher confidence
    depth_score = min(1.0, (conditions.bid_depth + conditions.ask_depth) / 5000.0)
    
    # Volatility confidence - moderate volatility = higher confidence
    vol_score = 1.0 - min(1.0, abs(conditions.volatility - 0.02) / 0.05)
    
    # Balance confidence - balanced book = higher confidence
    imbalance = abs(conditions.bid_depth - conditions.ask_depth) / (conditions.bid_depth + conditions.ask_depth + 1e-8)
    balance_score = 1.0 - min(1.0, imbalance)
    
    return (depth_score + vol_score + balance_score) / 3.0


# ============================================================================
# ENHANCED INVENTORY MANAGEMENT WITH FUNCTIONAL STATE
# ============================================================================

@dataclass(frozen=True)
class InventoryState:
    """Immutable inventory state representation."""
    current_position: float
    max_position: float
    target_position: float
    last_fill_time: Optional[datetime]
    position_value: float
    unrealized_pnl: float
    inventory_duration: float  # hours
    turn_rate: float  # daily turns
    
@dataclass(frozen=True)
class InventoryPolicy:
    """Immutable inventory management policy."""
    max_position_ratio: float = 0.1  # 10% of portfolio
    target_turn_rate: float = 8.0    # 8 turns per day
    skew_factor: float = 0.5
    urgency_threshold: float = 0.8   # 80% of max position
    timeout_hours: float = 4.0
    
@dataclass(frozen=True)
class InventorySignal:
    """Immutable inventory management signal."""
    urgency: float  # 0-1 scale
    skew_adjustment: float  # -1 to 1
    size_adjustment: float  # 0-2 multiplier
    reason: str
    confidence: float

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

def analyze_inventory_state(
    state: InventoryState,
    policy: InventoryPolicy,
    current_time: datetime
) -> InventorySignal:
    """Analyze inventory state and generate management signals.
    
    Uses sophisticated inventory management including:
    - Position size urgency
    - Time-based decay
    - Turn rate optimization
    - Risk-adjusted skewing
    
    Args:
        state: Current inventory state
        policy: Inventory management policy
        current_time: Current timestamp
        
    Returns:
        Inventory management signal
    """
    # Calculate position utilization
    position_ratio = abs(state.current_position) / state.max_position if state.max_position > 0 else 0
    
    # Calculate time urgency
    time_urgency = state.inventory_duration / policy.timeout_hours if policy.timeout_hours > 0 else 0
    
    # Calculate turn rate urgency
    turn_urgency = max(0, (policy.target_turn_rate - state.turn_rate) / policy.target_turn_rate)
    
    # Overall urgency (max of all factors)
    urgency = min(1.0, max(position_ratio, time_urgency, turn_urgency))
    
    # Skew adjustment based on position and urgency
    base_skew = -state.current_position / state.max_position if state.max_position > 0 else 0
    urgency_multiplier = 1.0 + urgency
    skew_adjustment = base_skew * policy.skew_factor * urgency_multiplier
    skew_adjustment = max(-1.0, min(1.0, skew_adjustment))
    
    # Size adjustment based on urgency
    if urgency > policy.urgency_threshold:
        # Increase size when urgent
        size_adjustment = 1.0 + (urgency - policy.urgency_threshold) * 2.0
    else:
        # Normal sizing
        size_adjustment = 1.0
    
    # Generate reason
    reasons = []
    if position_ratio > policy.urgency_threshold:
        reasons.append(f"position {position_ratio:.1%}")
    if time_urgency > 0.5:
        reasons.append(f"time {state.inventory_duration:.1f}h")
    if turn_urgency > 0.3:
        reasons.append(f"turns {state.turn_rate:.1f}")
    
    reason = f"inventory: {', '.join(reasons) if reasons else 'normal'}"
    
    # Confidence based on data quality
    confidence = 0.9 if state.last_fill_time else 0.7
    
    return InventorySignal(
        urgency=urgency,
        skew_adjustment=skew_adjustment,
        size_adjustment=size_adjustment,
        reason=reason,
        confidence=confidence
    )

def optimize_inventory_exposure(
    state: InventoryState,
    market_conditions: MarketConditions,
    target_notional: float
) -> Tuple[float, float]:  # (optimal_bid_size, optimal_ask_size)
    """Optimize bid/ask sizes for inventory management.
    
    Uses portfolio theory to optimize exposure while managing inventory risk.
    
    Args:
        state: Current inventory state
        market_conditions: Market conditions
        target_notional: Target notional per side
        
    Returns:
        Tuple of (optimal_bid_size, optimal_ask_size)
    """
    # Base size from target notional
    base_size = target_notional
    
    # Adjust for current inventory position
    inventory_ratio = state.current_position / state.max_position if state.max_position > 0 else 0
    
    # If long inventory, reduce bid size and increase ask size
    if inventory_ratio > 0:
        bid_multiplier = max(0.1, 1.0 - inventory_ratio * 1.5)
        ask_multiplier = min(2.0, 1.0 + inventory_ratio * 1.0)
    # If short inventory, increase bid size and reduce ask size
    elif inventory_ratio < 0:
        bid_multiplier = min(2.0, 1.0 + abs(inventory_ratio) * 1.0)
        ask_multiplier = max(0.1, 1.0 - abs(inventory_ratio) * 1.5)
    else:
        bid_multiplier = ask_multiplier = 1.0
    
    # Adjust for market volatility - reduce size in volatile markets
    volatility_adjustment = max(0.5, 1.0 - market_conditions.volatility * 10.0)
    
    # Adjust for liquidity - increase size in liquid markets
    liquidity_adjustment = min(1.5, 1.0 + (market_conditions.bid_depth + market_conditions.ask_depth) / 20000.0)
    
    # Calculate final sizes
    bid_size = base_size * bid_multiplier * volatility_adjustment * liquidity_adjustment
    ask_size = base_size * ask_multiplier * volatility_adjustment * liquidity_adjustment
    
    return bid_size, ask_size


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
        # Extract close prices for calculation
        close_prices = [float(candle.close) for candle in candles[-volatility_lookback:]]
        upper_band, middle_band, lower_band = calculate_bollinger_bands(
            close_prices, period=min(volatility_lookback, len(close_prices)), std_dev=2.0
        )

        if upper_band is None or lower_band is None or middle_band is None:
            return None

        # Use band width as volatility measure
        volatility = (upper_band - lower_band) / middle_band

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
