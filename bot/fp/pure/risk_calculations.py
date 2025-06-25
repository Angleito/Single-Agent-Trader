"""Pure functions for risk calculations."""

from decimal import Decimal

from bot.fp.types.effects import Err, Ok, Result


def calculate_position_risk(
    position_size: Decimal,
    entry_price: Decimal,
    current_price: Decimal,
    stop_loss_pct: Decimal,
    max_risk_pct: Decimal = Decimal("2.0"),  # 2% max risk per trade
) -> Result[Decimal, str]:
    """
    Calculate position risk based on position size and stop loss.

    Args:
        position_size: Size of the position
        entry_price: Entry price of the position
        current_price: Current market price
        stop_loss_pct: Stop loss percentage
        max_risk_pct: Maximum risk percentage allowed

    Returns:
        Result containing calculated risk or error message
    """
    if position_size <= 0:
        return Err("Position size must be positive")

    if entry_price <= 0:
        return Err("Entry price must be positive")

    if current_price <= 0:
        return Err("Current price must be positive")

    if stop_loss_pct <= 0 or stop_loss_pct >= 100:
        return Err("Stop loss percentage must be between 0 and 100")

    # Calculate position value
    position_value = position_size * current_price

    # Calculate stop loss distance
    stop_loss_distance = current_price * (stop_loss_pct / Decimal("100.0"))

    # Calculate potential loss
    potential_loss = position_size * stop_loss_distance

    # Calculate risk as percentage of position value
    risk_percentage = (potential_loss / position_value) * Decimal("100.0")

    # Check if risk exceeds maximum allowed
    if risk_percentage > max_risk_pct:
        return Err(f"Risk {risk_percentage}% exceeds maximum allowed {max_risk_pct}%")

    return Ok(potential_loss)


def validate_position_size(
    position_size: Decimal,
    account_balance: Decimal,
    max_position_pct: Decimal = Decimal("25.0"),  # 25% max position size
) -> Result[Decimal, str]:
    """
    Validate position size against account balance and risk limits.

    Args:
        position_size: Size of the position to validate
        account_balance: Total account balance
        max_position_pct: Maximum position size as percentage of balance

    Returns:
        Result containing validated position size or error message
    """
    if position_size <= 0:
        return Err("Position size must be positive")

    if account_balance <= 0:
        return Err("Account balance must be positive")

    if max_position_pct <= 0 or max_position_pct > 100:
        return Err("Maximum position percentage must be between 0 and 100")

    # Calculate maximum allowed position size
    max_position_size = account_balance * (max_position_pct / Decimal("100.0"))

    if position_size > max_position_size:
        return Err(
            f"Position size {position_size} exceeds maximum allowed {max_position_size}"
        )

    return Ok(position_size)


def calculate_leverage_ratio(
    position_value: Decimal,
    margin_used: Decimal,
) -> Result[Decimal, str]:
    """
    Calculate effective leverage ratio.

    Args:
        position_value: Total value of the position
        margin_used: Margin used for the position

    Returns:
        Result containing leverage ratio or error message
    """
    if position_value <= 0:
        return Err("Position value must be positive")

    if margin_used <= 0:
        return Err("Margin used must be positive")

    leverage = position_value / margin_used
    return Ok(leverage)


def calculate_margin_utilization(
    used_margin: Decimal,
    total_margin: Decimal,
) -> Result[Decimal, str]:
    """
    Calculate margin utilization percentage.

    Args:
        used_margin: Currently used margin
        total_margin: Total available margin

    Returns:
        Result containing margin utilization percentage or error message
    """
    if used_margin < 0:
        return Err("Used margin cannot be negative")

    if total_margin <= 0:
        return Err("Total margin must be positive")

    if used_margin > total_margin:
        return Err("Used margin cannot exceed total margin")

    utilization = (used_margin / total_margin) * Decimal("100.0")
    return Ok(utilization)


def calculate_value_at_risk(
    portfolio_value: Decimal,
    volatility: Decimal,
    confidence_level: Decimal = Decimal("95.0"),  # 95% confidence
    time_horizon_days: int = 1,
) -> Result[Decimal, str]:
    """
    Calculate Value at Risk (VaR) for a portfolio.

    Args:
        portfolio_value: Total value of the portfolio
        volatility: Historical volatility (daily)
        confidence_level: Confidence level (e.g., 95.0 for 95%)
        time_horizon_days: Time horizon in days

    Returns:
        Result containing VaR or error message
    """
    if portfolio_value <= 0:
        return Err("Portfolio value must be positive")

    if volatility < 0:
        return Err("Volatility cannot be negative")

    if confidence_level <= 0 or confidence_level >= 100:
        return Err("Confidence level must be between 0 and 100")

    if time_horizon_days <= 0:
        return Err("Time horizon must be positive")

    # Z-score for confidence levels (approximate)
    z_scores = {
        Decimal("90.0"): Decimal("1.28"),
        Decimal("95.0"): Decimal("1.65"),
        Decimal("99.0"): Decimal("2.33"),
    }

    z_score = z_scores.get(confidence_level, Decimal("1.65"))  # Default to 95%

    # Calculate VaR
    time_factor = Decimal(time_horizon_days).sqrt()
    var = portfolio_value * volatility * z_score * time_factor

    return Ok(var)


def calculate_sharpe_ratio(
    portfolio_return: Decimal,
    risk_free_rate: Decimal,
    return_volatility: Decimal,
) -> Result[Decimal, str]:
    """
    Calculate Sharpe ratio for risk-adjusted returns.

    Args:
        portfolio_return: Portfolio return percentage
        risk_free_rate: Risk-free rate percentage
        return_volatility: Volatility of returns

    Returns:
        Result containing Sharpe ratio or error message
    """
    if return_volatility <= 0:
        return Err("Return volatility must be positive")

    excess_return = portfolio_return - risk_free_rate
    sharpe_ratio = excess_return / return_volatility

    return Ok(sharpe_ratio)


def calculate_maximum_drawdown(
    equity_curve: list[Decimal],
) -> Result[Decimal, str]:
    """
    Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: List of equity values over time

    Returns:
        Result containing maximum drawdown percentage or error message
    """
    if not equity_curve:
        return Err("Equity curve cannot be empty")

    if len(equity_curve) < 2:
        return Ok(Decimal("0.0"))  # No drawdown with single point

    max_drawdown = Decimal("0.0")
    peak = equity_curve[0]

    for equity in equity_curve:
        if equity > peak:
            peak = equity
        elif peak > 0:
            drawdown = (peak - equity) / peak * Decimal("100.0")
            max_drawdown = max(max_drawdown, drawdown)

    return Ok(max_drawdown)


def calculate_kelly_criterion(
    win_probability: Decimal,
    avg_win: Decimal,
    avg_loss: Decimal,
) -> Result[Decimal, str]:
    """
    Calculate optimal position size using Kelly Criterion.

    Args:
        win_probability: Probability of winning (0-1)
        avg_win: Average winning amount
        avg_loss: Average losing amount (positive value)

    Returns:
        Result containing optimal position size fraction or error message
    """
    if win_probability < 0 or win_probability > 1:
        return Err("Win probability must be between 0 and 1")

    if avg_win <= 0:
        return Err("Average win must be positive")

    if avg_loss <= 0:
        return Err("Average loss must be positive")

    # Kelly formula: f = (bp - q) / b
    # where b = avg_win/avg_loss, p = win_probability, q = 1-p
    b = avg_win / avg_loss
    p = win_probability
    q = Decimal("1.0") - p

    kelly_fraction = (b * p - q) / b

    # Cap at 25% to avoid excessive risk
    max_fraction = Decimal("0.25")
    if kelly_fraction > max_fraction:
        kelly_fraction = max_fraction
    elif kelly_fraction < 0:
        kelly_fraction = Decimal("0.0")  # Don't trade if negative

    return Ok(kelly_fraction)


def calculate_correlation_risk(
    positions: list[tuple[str, Decimal]],  # (symbol, position_value)
    correlations: dict[tuple[str, str], Decimal],  # correlation matrix
    max_correlation_exposure: Decimal = Decimal("50.0"),  # 50% max correlated exposure
) -> Result[Decimal, str]:
    """
    Calculate correlation risk across positions.

    Args:
        positions: List of (symbol, position_value) tuples
        correlations: Dictionary of correlation coefficients between symbols
        max_correlation_exposure: Maximum allowed correlated exposure percentage

    Returns:
        Result containing correlation risk assessment or error message
    """
    if not positions:
        return Ok(Decimal("0.0"))

    if len(positions) == 1:
        return Ok(Decimal("0.0"))  # No correlation risk with single position

    total_exposure = sum(pos_value for _, pos_value in positions)
    if total_exposure <= 0:
        return Err("Total exposure must be positive")

    # Calculate weighted correlation exposure
    correlated_exposure = Decimal("0.0")

    for i, (symbol1, value1) in enumerate(positions):
        for symbol2, value2 in positions[i + 1 :]:
            correlation_key = (
                (symbol1, symbol2) if symbol1 < symbol2 else (symbol2, symbol1)
            )
            correlation = correlations.get(correlation_key, Decimal("0.0"))

            # High correlation (>0.7) contributes to risk
            if correlation > Decimal("0.7"):
                pair_exposure = (value1 + value2) * correlation
                correlated_exposure += pair_exposure

    correlation_risk_pct = (correlated_exposure / total_exposure) * Decimal("100.0")

    if correlation_risk_pct > max_correlation_exposure:
        return Err(
            f"Correlation risk {correlation_risk_pct}% exceeds maximum {max_correlation_exposure}%"
        )

    return Ok(correlation_risk_pct)


def calculate_portfolio_beta(
    portfolio_returns: list[Decimal],
    market_returns: list[Decimal],
) -> Result[Decimal, str]:
    """
    Calculate portfolio beta relative to market.

    Args:
        portfolio_returns: List of portfolio returns
        market_returns: List of market returns

    Returns:
        Result containing portfolio beta or error message
    """
    if len(portfolio_returns) != len(market_returns):
        return Err("Portfolio and market return lists must have same length")

    if len(portfolio_returns) < 2:
        return Err("Need at least 2 data points to calculate beta")

    # Calculate means
    portfolio_mean = sum(portfolio_returns) / len(portfolio_returns)
    market_mean = sum(market_returns) / len(market_returns)

    # Calculate covariance and market variance
    covariance = Decimal("0.0")
    market_variance = Decimal("0.0")

    for p_ret, m_ret in zip(portfolio_returns, market_returns, strict=False):
        p_diff = p_ret - portfolio_mean
        m_diff = m_ret - market_mean
        covariance += p_diff * m_diff
        market_variance += m_diff * m_diff

    n = Decimal(len(portfolio_returns))
    covariance /= n - 1
    market_variance /= n - 1

    if market_variance == 0:
        return Err("Market variance is zero, cannot calculate beta")

    beta = covariance / market_variance
    return Ok(beta)
