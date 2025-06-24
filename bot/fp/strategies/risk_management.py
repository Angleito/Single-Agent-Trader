"""Pure functional risk management strategies.

This module provides pure functions for risk management including:
- Position sizing algorithms (Kelly Criterion, Fixed Fractional, Volatility-based)
- Stop loss and take profit calculations
- Risk limit enforcement
- Portfolio heat and correlation adjustments
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    """Risk management configuration."""

    max_position_size: float = 0.25
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05
    max_drawdown_pct: float = 0.10
    kelly_fraction: float = 0.25
    win_probability: float = 0.55
    win_loss_ratio: float = 1.5


@dataclass(frozen=True)
class RiskResult:
    """Risk calculation result."""

    position_size: float
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_score: float = 0.0
    warnings: tuple[str, ...] = ()


class FunctionalRiskManager:
    """Functional risk manager implementation."""

    def __init__(self, config: RiskConfig):
        self.config = config

    def calculate_position_size(
        self,
        account_balance: float,
        current_price: float,
        volatility: float | None = None,
    ) -> RiskResult:
        """Calculate optimal position size using functional risk management."""
        # Use Kelly Criterion if we have win probability data
        kelly_size = calculate_kelly_criterion(
            self.config.win_probability,
            self.config.win_loss_ratio,
            self.config.kelly_fraction,
        )

        # Fixed fractional as backup
        fixed_size = calculate_fixed_fractional_size(
            account_balance, self.config.max_position_size * 100
        )

        # Use the more conservative estimate
        position_size = min(kelly_size, fixed_size, self.config.max_position_size)

        # Calculate stop loss and take profit levels
        stop_loss = current_price * (1 - self.config.stop_loss_pct)
        take_profit = current_price * (1 + self.config.take_profit_pct)

        # Calculate risk score based on various factors
        risk_score = self._calculate_risk_score(position_size, volatility)

        warnings = []
        if position_size >= self.config.max_position_size:
            warnings.append("Position size at maximum limit")
        if volatility and volatility > 0.05:
            warnings.append("High volatility detected")

        return RiskResult(
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_score=risk_score,
            warnings=tuple(warnings),
        )

    def _calculate_risk_score(
        self, position_size: float, volatility: float | None
    ) -> float:
        """Calculate a risk score from 0 (low risk) to 1 (high risk)."""
        base_risk = position_size / self.config.max_position_size

        if volatility:
            volatility_risk = min(volatility / 0.1, 1.0)  # Scale volatility to 0-1
            return min((base_risk + volatility_risk) / 2, 1.0)

        return base_risk


def calculate_kelly_criterion(
    win_probability: float, win_loss_ratio: float, kelly_fraction: float = 0.25
) -> float:
    """Calculate position size using Kelly Criterion.

    The Kelly Criterion formula: f = (p * b - q) / b
    where:
    - f = fraction of capital to bet
    - p = probability of winning
    - q = probability of losing (1 - p)
    - b = ratio of win to loss

    Args:
        win_probability: Probability of winning (0-1)
        win_loss_ratio: Ratio of average win to average loss
        kelly_fraction: Fraction of Kelly to use (default 0.25 for safety)

    Returns:
        Fraction of capital to risk (0-1)
    """
    if win_probability <= 0 or win_probability >= 1:
        return 0.0

    if win_loss_ratio <= 0:
        return 0.0

    lose_probability = 1 - win_probability
    kelly_percentage = (
        win_probability * win_loss_ratio - lose_probability
    ) / win_loss_ratio

    # Apply Kelly fraction for safety
    return max(0.0, min(kelly_percentage * kelly_fraction, 1.0))


def calculate_fixed_fractional_size(
    account_balance: float, risk_percentage: float = 2.0
) -> float:
    """Calculate position size using fixed fractional method.

    Args:
        account_balance: Total account balance
        risk_percentage: Percentage of account to risk per trade

    Returns:
        Dollar amount to risk
    """
    if account_balance <= 0:
        return 0.0

    return account_balance * (risk_percentage / 100.0)


def calculate_volatility_based_size(
    account_balance: float,
    volatility: float,
    target_volatility: float = 0.02,
    max_position_pct: float = 10.0,
) -> float:
    """Calculate position size based on volatility targeting.

    Position size is inversely proportional to volatility to maintain
    consistent risk across different market conditions.

    Args:
        account_balance: Total account balance
        volatility: Current market volatility (as decimal, e.g., 0.03 for 3%)
        target_volatility: Target portfolio volatility (default 2%)
        max_position_pct: Maximum position size as percentage of account

    Returns:
        Dollar amount for position
    """
    if account_balance <= 0 or volatility <= 0:
        return 0.0

    # Position size = (Target Vol / Market Vol) * Base Position
    base_position = account_balance * (max_position_pct / 100.0)
    volatility_adjustment = min(target_volatility / volatility, 2.0)  # Cap at 2x

    return base_position * volatility_adjustment


def calculate_atr_stop_loss(
    entry_price: float, atr: float, atr_multiplier: float = 2.0, is_long: bool = True
) -> float:
    """Calculate stop loss using Average True Range.

    Args:
        entry_price: Entry price of the position
        atr: Average True Range value
        atr_multiplier: Multiplier for ATR (default 2.0)
        is_long: True for long positions, False for short

    Returns:
        Stop loss price
    """
    if entry_price <= 0 or atr < 0:
        return 0.0

    stop_distance = atr * atr_multiplier

    if is_long:
        return max(0, entry_price - stop_distance)
    return entry_price + stop_distance


def calculate_percentage_stop_loss(
    entry_price: float, stop_percentage: float = 2.0, is_long: bool = True
) -> float:
    """Calculate stop loss using fixed percentage.

    Args:
        entry_price: Entry price of the position
        stop_percentage: Stop loss percentage (e.g., 2.0 for 2%)
        is_long: True for long positions, False for short

    Returns:
        Stop loss price
    """
    if entry_price <= 0:
        return 0.0

    stop_distance = entry_price * (stop_percentage / 100.0)

    if is_long:
        return max(0, entry_price - stop_distance)
    return entry_price + stop_distance


def calculate_risk_reward_take_profit(
    entry_price: float,
    stop_loss: float,
    risk_reward_ratio: float = 2.0,
    is_long: bool = True,
) -> float:
    """Calculate take profit based on risk/reward ratio.

    Args:
        entry_price: Entry price of the position
        stop_loss: Stop loss price
        risk_reward_ratio: Desired risk/reward ratio (default 2.0)
        is_long: True for long positions, False for short

    Returns:
        Take profit price
    """
    if entry_price <= 0 or stop_loss < 0:
        return 0.0

    risk_amount = abs(entry_price - stop_loss)
    reward_amount = risk_amount * risk_reward_ratio

    if is_long:
        return entry_price + reward_amount
    return max(0, entry_price - reward_amount)


def calculate_trailing_stop(
    current_price: float,
    highest_price: float,
    lowest_price: float,
    trailing_percentage: float = 2.0,
    is_long: bool = True,
) -> float:
    """Calculate trailing stop loss.

    Args:
        current_price: Current market price
        highest_price: Highest price since entry (for longs)
        lowest_price: Lowest price since entry (for shorts)
        trailing_percentage: Trailing stop percentage
        is_long: True for long positions, False for short

    Returns:
        Trailing stop price
    """
    if current_price <= 0:
        return 0.0

    trailing_distance_pct = trailing_percentage / 100.0

    if is_long:
        if highest_price <= 0:
            return 0.0
        return highest_price * (1 - trailing_distance_pct)
    if lowest_price <= 0:
        return float("inf")
    return lowest_price * (1 + trailing_distance_pct)


def calculate_portfolio_heat(
    positions: list[dict[str, float]], account_balance: float
) -> float:
    """Calculate total portfolio heat (risk exposure).

    Portfolio heat is the sum of all position risks as a percentage
    of account balance.

    Args:
        positions: List of position dictionaries with keys:
            - 'size': Position size in dollars
            - 'entry_price': Entry price
            - 'stop_loss': Stop loss price
            - 'is_long': Boolean for direction
        account_balance: Total account balance

    Returns:
        Portfolio heat as percentage (0-100)
    """
    if account_balance <= 0 or not positions:
        return 0.0

    total_risk = 0.0

    for position in positions:
        size = position.get("size", 0)
        entry_price = position.get("entry_price", 0)
        stop_loss = position.get("stop_loss", 0)
        is_long = position.get("is_long", True)

        if size <= 0 or entry_price <= 0 or stop_loss < 0:
            continue

        # Calculate risk per unit
        if is_long:
            risk_per_unit = max(0, entry_price - stop_loss)
        else:
            risk_per_unit = max(0, stop_loss - entry_price)

        # Calculate position risk
        units = size / entry_price if entry_price > 0 else 0
        position_risk = units * risk_per_unit
        total_risk += position_risk

    return (total_risk / account_balance) * 100.0


def enforce_risk_limits(
    proposed_size: float,
    current_positions: list[dict[str, float]],
    account_balance: float,
    max_position_size_pct: float = 10.0,
    max_portfolio_heat_pct: float = 6.0,
    max_correlated_exposure_pct: float = 15.0,
    correlation_threshold: float = 0.7,  # TODO: Implement correlation analysis
) -> tuple[float, str]:
    """Enforce risk limits on proposed position.

    Args:
        proposed_size: Proposed position size in dollars
        current_positions: List of current positions (same format as portfolio_heat)
        account_balance: Total account balance
        max_position_size_pct: Maximum single position as % of account
        max_portfolio_heat_pct: Maximum total portfolio heat
        max_correlated_exposure_pct: Maximum correlated exposure
        correlation_threshold: Correlation threshold for grouping

    Returns:
        Tuple of (adjusted_size, reason)
    """
    if account_balance <= 0:
        return (0.0, "Invalid account balance")

    # Check single position size limit
    max_position_size = account_balance * (max_position_size_pct / 100.0)
    if proposed_size > max_position_size:
        return (
            max_position_size,
            f"Reduced to max position size {max_position_size_pct}%",
        )

    # Check portfolio heat with new position
    # For simplicity, assume new position adds 2% heat
    current_heat = calculate_portfolio_heat(current_positions, account_balance)
    new_position_heat = 2.0  # Simplified assumption

    if current_heat + new_position_heat > max_portfolio_heat_pct:
        # Reduce position to fit within heat limit
        available_heat = max(0, max_portfolio_heat_pct - current_heat)
        reduction_factor = available_heat / new_position_heat
        adjusted_size = proposed_size * reduction_factor
        return (
            adjusted_size,
            f"Reduced to maintain portfolio heat under {max_portfolio_heat_pct}%",
        )

    # Check correlated exposure (simplified)
    total_exposure = sum(pos.get("size", 0) for pos in current_positions)
    total_exposure_pct = (total_exposure / account_balance) * 100.0

    if (
        total_exposure_pct + (proposed_size / account_balance * 100)
        > max_correlated_exposure_pct
    ):
        available_exposure = max(0, max_correlated_exposure_pct - total_exposure_pct)
        adjusted_size = account_balance * (available_exposure / 100.0)
        return (
            adjusted_size,
            f"Reduced to maintain correlated exposure under {max_correlated_exposure_pct}%",
        )

    return (proposed_size, "Position size approved")


def calculate_position_size_with_stop(
    account_balance: float,
    entry_price: float,
    stop_loss: float,
    risk_percentage: float = 2.0,
    is_long: bool = True,
) -> float:
    """Calculate position size based on stop loss and risk amount.

    This ensures that if the stop loss is hit, the loss will be
    exactly the specified risk percentage of the account.

    Args:
        account_balance: Total account balance
        entry_price: Entry price for the position
        stop_loss: Stop loss price
        risk_percentage: Percentage of account to risk
        is_long: True for long positions, False for short

    Returns:
        Position size in dollars
    """
    if account_balance <= 0 or entry_price <= 0 or stop_loss < 0:
        return 0.0

    # Calculate risk per unit
    if is_long:
        risk_per_unit = max(0, entry_price - stop_loss)
    else:
        risk_per_unit = max(0, stop_loss - entry_price)

    if risk_per_unit == 0:
        return 0.0

    # Calculate total risk amount
    risk_amount = account_balance * (risk_percentage / 100.0)

    # Calculate number of units
    units = risk_amount / risk_per_unit

    # Convert to position size
    return units * entry_price


def calculate_correlation_adjustment(
    correlations: dict[str, float],
    base_position_sizes: dict[str, float],
    correlation_penalty: float = 0.5,
) -> dict[str, float]:
    """Adjust position sizes based on correlations.

    Reduces position sizes for highly correlated assets to avoid
    concentration risk.

    Args:
        correlations: Dictionary of symbol pairs to correlation values
            e.g., {('BTC', 'ETH'): 0.8, ('BTC', 'SOL'): 0.6}
        base_position_sizes: Dictionary of symbol to base position size
        correlation_penalty: Penalty factor for correlated positions (0-1)

    Returns:
        Dictionary of adjusted position sizes
    """
    if not base_position_sizes:
        return {}

    adjusted_sizes = base_position_sizes.copy()
    symbols = list(base_position_sizes.keys())

    # Apply correlation penalties
    for i, symbol1 in enumerate(symbols):
        correlation_sum = 0.0

        for j, symbol2 in enumerate(symbols):
            if i == j:
                continue

            # Get correlation (check both orderings)
            corr = correlations.get(
                (symbol1, symbol2), correlations.get((symbol2, symbol1), 0.0)
            )

            if abs(corr) > 0.5:  # Only penalize significant correlations
                correlation_sum += abs(corr)

        # Apply adjustment based on total correlation
        if correlation_sum > 0:
            adjustment_factor = 1.0 - (correlation_penalty * min(correlation_sum, 1.0))
            adjusted_sizes[symbol1] = base_position_sizes[symbol1] * adjustment_factor

    return adjusted_sizes


def calculate_optimal_leverage(
    volatility: float,
    win_rate: float,
    risk_per_trade: float = 2.0,
    max_leverage: float = 10.0,
) -> float:
    """Calculate optimal leverage based on market conditions.

    Args:
        volatility: Current market volatility (as decimal)
        win_rate: Historical win rate (0-1)
        risk_per_trade: Risk per trade as percentage
        max_leverage: Maximum allowed leverage

    Returns:
        Optimal leverage to use
    """
    if volatility <= 0 or win_rate <= 0 or win_rate >= 1:
        return 1.0

    # Base leverage calculation
    # Higher win rate allows higher leverage
    # Higher volatility requires lower leverage
    base_leverage = (win_rate * 2.0) / (volatility * 100.0)

    # Adjust for risk per trade
    risk_adjustment = min(risk_per_trade / 2.0, 1.0)

    optimal_leverage = base_leverage * risk_adjustment

    # Apply constraints
    return max(1.0, min(optimal_leverage, max_leverage))


def calculate_drawdown_adjusted_size(
    base_size: float,
    current_drawdown_pct: float,
    max_drawdown_pct: float = 20.0,
    reduction_factor: float = 0.5,
) -> float:
    """Adjust position size based on current drawdown.

    Reduces position size during drawdowns to protect capital
    and allow for recovery.

    Args:
        base_size: Base position size
        current_drawdown_pct: Current drawdown as percentage (0-100)
        max_drawdown_pct: Maximum acceptable drawdown
        reduction_factor: How much to reduce size at max drawdown

    Returns:
        Adjusted position size
    """
    if base_size <= 0 or current_drawdown_pct < 0:
        return 0.0

    if current_drawdown_pct >= max_drawdown_pct:
        return base_size * reduction_factor

    # Linear reduction based on drawdown
    drawdown_ratio = current_drawdown_pct / max_drawdown_pct
    adjustment = 1.0 - (drawdown_ratio * (1.0 - reduction_factor))

    return base_size * adjustment
