"""
Risk management types for functional programming architecture.

This module defines immutable data structures and pure functions for risk management,
including risk parameters, limits, metrics, margin information, alerts, circuit breakers,
emergency stops, and comprehensive safety mechanisms.
"""

from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Union, Dict, List, Tuple, Optional

from bot.fp.types.effects import Maybe, Some, Nothing
from bot.fp.types.result import Result, Success, Failure


@dataclass(frozen=True)
class RiskParameters:
    """Risk management parameters for trading."""

    max_position_size: Decimal  # Maximum position size as percentage of portfolio
    max_leverage: Decimal  # Maximum allowed leverage
    stop_loss_pct: Decimal  # Stop loss percentage from entry
    take_profit_pct: Decimal  # Take profit percentage from entry


@dataclass(frozen=True)
class RiskLimits:
    """Hard limits for risk management."""

    daily_loss_limit: Decimal  # Maximum daily loss in USD
    position_limit: int  # Maximum number of concurrent positions
    margin_requirement: Decimal  # Minimum margin requirement percentage


@dataclass(frozen=True)
class RiskMetrics:
    """Current risk metrics and statistics."""

    current_exposure: Decimal  # Total exposure in USD
    var_95: Decimal  # Value at Risk at 95% confidence
    max_drawdown: Decimal  # Maximum drawdown percentage
    sharpe_ratio: Decimal  # Risk-adjusted return metric


@dataclass(frozen=True)
class MarginInfo:
    """Margin and balance information."""

    total_balance: Decimal  # Total account balance
    used_margin: Decimal  # Margin currently in use
    free_margin: Decimal  # Available margin
    margin_ratio: Decimal  # Used margin / total balance ratio


# Risk Alert Types (Sum Type)
class RiskAlertType(Enum):
    """Types of risk alerts."""

    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    MARGIN_CALL = "margin_call"
    DAILY_LOSS_LIMIT = "daily_loss_limit"


@dataclass(frozen=True)
class PositionLimitExceeded:
    """Alert when position limit is exceeded."""

    current_positions: int
    limit: int
    alert_type: RiskAlertType = RiskAlertType.POSITION_LIMIT_EXCEEDED


@dataclass(frozen=True)
class MarginCall:
    """Alert when margin ratio is too high."""

    margin_ratio: Decimal
    threshold: Decimal
    alert_type: RiskAlertType = RiskAlertType.MARGIN_CALL


@dataclass(frozen=True)
class DailyLossLimit:
    """Alert when daily loss limit is reached."""

    current_loss: Decimal
    limit: Decimal
    alert_type: RiskAlertType = RiskAlertType.DAILY_LOSS_LIMIT


# Union type for all risk alerts
RiskAlert = Union[PositionLimitExceeded, MarginCall, DailyLossLimit]


# Advanced Risk Management Types

@dataclass(frozen=True)
class FailureRecord:
    """Record of trading failures for circuit breaker."""
    
    timestamp: datetime
    failure_type: str
    error_message: str
    severity: str = "medium"  # low, medium, high, critical


@dataclass(frozen=True)
class CircuitBreakerState:
    """Circuit breaker state management."""
    
    state: str  # "CLOSED", "OPEN", "HALF_OPEN"
    failure_count: int
    failure_threshold: int
    timeout_seconds: int
    last_failure_time: Optional[datetime]
    consecutive_successes: int
    failure_history: Tuple[FailureRecord, ...]
    
    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == "OPEN"
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit breaker is closed."""
        return self.state == "CLOSED"
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open."""
        return self.state == "HALF_OPEN"
    
    @property
    def can_execute(self) -> bool:
        """Check if trades can be executed."""
        return self.state in ["CLOSED", "HALF_OPEN"]


@dataclass(frozen=True)
class EmergencyStopReason:
    """Reason for emergency stop activation."""
    
    reason_type: str
    description: str
    triggered_at: datetime
    severity: str = "critical"


@dataclass(frozen=True)
class EmergencyStopState:
    """Emergency stop state management."""
    
    is_stopped: bool
    stop_reason: Optional[EmergencyStopReason]
    stopped_at: Optional[datetime]
    manual_override: bool = False
    
    @property
    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        return not self.is_stopped or self.manual_override


@dataclass(frozen=True)
class APIProtectionState:
    """API failure protection state."""
    
    consecutive_failures: int
    max_retries: int
    base_delay: float
    last_failure_time: Optional[datetime]
    is_healthy: bool
    backoff_multiplier: float = 2.0
    
    @property
    def next_retry_delay(self) -> float:
        """Calculate next retry delay with exponential backoff."""
        return self.base_delay * (self.backoff_multiplier ** self.consecutive_failures)
    
    @property
    def can_retry(self) -> bool:
        """Check if API calls can be retried."""
        return self.consecutive_failures < self.max_retries


@dataclass(frozen=True)
class DailyPnL:
    """Daily P&L tracking."""
    
    date: date
    realized_pnl: Decimal = Decimal(0)
    unrealized_pnl: Decimal = Decimal(0)
    trades_count: int = 0
    max_drawdown: Decimal = Decimal(0)
    
    @property
    def total_pnl(self) -> Decimal:
        """Calculate total P&L for the day."""
        return self.realized_pnl + self.unrealized_pnl


@dataclass(frozen=True)
class RiskValidationResult:
    """Result of risk validation checks."""
    
    is_valid: bool
    reason: str
    severity: str  # "low", "medium", "high", "critical"
    validation_type: str
    timestamp: datetime
    additional_data: Dict[str, any] = None
    
    def __post_init__(self):
        if self.additional_data is None:
            object.__setattr__(self, 'additional_data', {})


@dataclass(frozen=True)
class PositionValidationResult:
    """Result of position validation checks."""
    
    is_valid: bool
    symbol: str
    reason: str
    severity: str
    validation_checks: Dict[str, bool]
    position_size: Decimal
    entry_price: Optional[Decimal]
    current_price: Optional[Decimal]
    timestamp: datetime


@dataclass(frozen=True)
class RiskLevelAssessment:
    """Overall risk level assessment."""
    
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    score: float  # 0-100 risk score
    contributing_factors: List[str]
    recommendations: List[str]
    timestamp: datetime
    
    @property
    def is_critical(self) -> bool:
        """Check if risk level is critical."""
        return self.risk_level == "CRITICAL"
    
    @property
    def is_high(self) -> bool:
        """Check if risk level is high or critical."""
        return self.risk_level in ["HIGH", "CRITICAL"]


@dataclass(frozen=True)
class PortfolioExposure:
    """Portfolio exposure analysis."""
    
    total_exposure: Decimal
    symbol_exposures: Dict[str, Decimal]
    sector_exposures: Dict[str, Decimal]
    correlation_risk: float
    concentration_risk: float
    max_single_position_pct: float
    portfolio_heat: float  # Total risk as percentage of account
    
    @property
    def is_overexposed(self) -> bool:
        """Check if portfolio is overexposed."""
        return self.portfolio_heat > 10.0  # 10% threshold


@dataclass(frozen=True)
class LeverageAnalysis:
    """Leverage analysis and optimization."""
    
    current_leverage: Decimal
    optimal_leverage: Decimal
    max_allowed_leverage: Decimal
    volatility_adjustment: float
    win_rate_adjustment: float
    risk_adjustment: float
    recommended_action: str  # "INCREASE", "DECREASE", "MAINTAIN"


@dataclass(frozen=True)
class CorrelationMatrix:
    """Correlation matrix for risk analysis."""
    
    symbols: Tuple[str, ...]
    correlations: Dict[Tuple[str, str], float]
    timestamp: datetime
    
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols."""
        key1 = (symbol1, symbol2)
        key2 = (symbol2, symbol1)
        return self.correlations.get(key1, self.correlations.get(key2, 0.0))


@dataclass(frozen=True)
class DrawdownAnalysis:
    """Drawdown analysis and tracking."""
    
    current_drawdown_pct: float
    max_drawdown_pct: float
    drawdown_duration_hours: float
    peak_balance: Decimal
    current_balance: Decimal
    is_in_drawdown: bool
    recovery_target: Decimal
    
    @property
    def is_severe_drawdown(self) -> bool:
        """Check if drawdown is severe (>15%)."""
        return self.current_drawdown_pct > 15.0


@dataclass(frozen=True)
class RiskMetricsSnapshot:
    """Comprehensive risk metrics snapshot."""
    
    timestamp: datetime
    account_balance: Decimal
    portfolio_exposure: PortfolioExposure
    leverage_analysis: LeverageAnalysis
    drawdown_analysis: DrawdownAnalysis
    daily_pnl: DailyPnL
    risk_assessment: RiskLevelAssessment
    position_count: int
    consecutive_losses: int
    margin_usage_pct: float
    
    @property
    def overall_risk_score(self) -> float:
        """Calculate overall risk score (0-100)."""
        # Weighted risk score calculation
        exposure_weight = 0.3
        leverage_weight = 0.2
        drawdown_weight = 0.25
        losses_weight = 0.15
        margin_weight = 0.1
        
        exposure_score = min(self.portfolio_exposure.portfolio_heat * 2, 100)
        leverage_score = min(float(self.leverage_analysis.current_leverage) * 5, 100)
        drawdown_score = min(self.drawdown_analysis.current_drawdown_pct * 5, 100)
        losses_score = min(self.consecutive_losses * 20, 100)
        margin_score = self.margin_usage_pct
        
        return (
            exposure_score * exposure_weight +
            leverage_score * leverage_weight +
            drawdown_score * drawdown_weight +
            losses_score * losses_weight +
            margin_score * margin_weight
        )


@dataclass(frozen=True)
class ComprehensiveRiskState:
    """Complete risk management state."""
    
    circuit_breaker: CircuitBreakerState
    emergency_stop: EmergencyStopState
    api_protection: APIProtectionState
    risk_metrics: RiskMetricsSnapshot
    correlation_matrix: CorrelationMatrix
    validation_history: Tuple[RiskValidationResult, ...]
    
    @property
    def can_trade(self) -> bool:
        """Check if trading is allowed based on all risk factors."""
        return (
            self.emergency_stop.can_trade and
            self.circuit_breaker.can_execute and
            self.api_protection.is_healthy and
            not self.risk_metrics.risk_assessment.is_critical
        )
    
    @property
    def trading_restrictions(self) -> List[str]:
        """Get list of current trading restrictions."""
        restrictions = []
        
        if not self.emergency_stop.can_trade:
            restrictions.append(f"Emergency stop: {self.emergency_stop.stop_reason.description if self.emergency_stop.stop_reason else 'Unknown'}")
        
        if not self.circuit_breaker.can_execute:
            restrictions.append(f"Circuit breaker {self.circuit_breaker.state}")
        
        if not self.api_protection.is_healthy:
            restrictions.append(f"API protection: {self.api_protection.consecutive_failures} failures")
        
        if self.risk_metrics.risk_assessment.is_critical:
            restrictions.append(f"Critical risk level: {self.risk_metrics.risk_assessment.risk_level}")
        
        return restrictions


# Enhanced Risk Alert Types

class AdvancedRiskAlertType(Enum):
    """Advanced risk alert types."""
    
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    EMERGENCY_STOP_TRIGGERED = "emergency_stop_triggered"
    API_FAILURES_EXCESSIVE = "api_failures_excessive"
    CORRELATION_RISK_HIGH = "correlation_risk_high"
    LEVERAGE_EXCESSIVE = "leverage_excessive"
    DRAWDOWN_SEVERE = "drawdown_severe"
    PORTFOLIO_OVEREXPOSED = "portfolio_overexposed"
    CONSECUTIVE_LOSSES_HIGH = "consecutive_losses_high"


@dataclass(frozen=True)
class AdvancedRiskAlert:
    """Advanced risk alert with detailed information."""
    
    alert_type: AdvancedRiskAlertType
    severity: str  # "low", "medium", "high", "critical"
    title: str
    description: str
    triggered_at: datetime
    metric_value: float
    threshold_value: float
    recommended_action: str
    auto_action_taken: bool = False


# Union type for all advanced risk alerts
AllRiskAlerts = Union[RiskAlert, AdvancedRiskAlert]


# Pure Risk Calculation Functions


def calculate_position_size(
    balance: Decimal, risk_per_trade: Decimal, stop_loss_pct: Decimal
) -> Decimal:
    """
    Calculate position size based on risk parameters.

    Args:
        balance: Total account balance
        risk_per_trade: Risk percentage per trade
        stop_loss_pct: Stop loss percentage

    Returns:
        Position size in base currency
    """
    if stop_loss_pct <= Decimal(0):
        return Decimal(0)

    risk_amount = balance * risk_per_trade / Decimal(100)
    position_size = risk_amount / (stop_loss_pct / Decimal(100))
    return position_size


def calculate_margin_ratio(used_margin: Decimal, total_balance: Decimal) -> Decimal:
    """
    Calculate margin ratio.

    Args:
        used_margin: Currently used margin
        total_balance: Total account balance

    Returns:
        Margin ratio as decimal
    """
    if total_balance <= Decimal(0):
        return Decimal(0)

    return used_margin / total_balance


def calculate_free_margin(total_balance: Decimal, used_margin: Decimal) -> Decimal:
    """
    Calculate available free margin.

    Args:
        total_balance: Total account balance
        used_margin: Currently used margin

    Returns:
        Free margin amount
    """
    return total_balance - used_margin


def calculate_max_position_value(
    balance: Decimal, max_position_size_pct: Decimal, leverage: Decimal
) -> Decimal:
    """
    Calculate maximum allowed position value.

    Args:
        balance: Total account balance
        max_position_size_pct: Maximum position size as percentage
        leverage: Trading leverage

    Returns:
        Maximum position value
    """
    max_position = balance * max_position_size_pct / Decimal(100)
    return max_position * leverage


def calculate_required_margin(
    position_size: Decimal, entry_price: Decimal, leverage: Decimal
) -> Decimal:
    """
    Calculate required margin for a position.

    Args:
        position_size: Size of the position
        entry_price: Entry price
        leverage: Trading leverage

    Returns:
        Required margin amount
    """
    if leverage <= Decimal(0):
        return Decimal(0)

    position_value = position_size * entry_price
    return position_value / leverage


def calculate_stop_loss_price(
    entry_price: Decimal, stop_loss_pct: Decimal, is_long: bool
) -> Decimal:
    """
    Calculate stop loss price.

    Args:
        entry_price: Position entry price
        stop_loss_pct: Stop loss percentage
        is_long: True for long position, False for short

    Returns:
        Stop loss price
    """
    stop_loss_amount = entry_price * stop_loss_pct / Decimal(100)

    if is_long:
        return entry_price - stop_loss_amount
    return entry_price + stop_loss_amount


def calculate_take_profit_price(
    entry_price: Decimal, take_profit_pct: Decimal, is_long: bool
) -> Decimal:
    """
    Calculate take profit price.

    Args:
        entry_price: Position entry price
        take_profit_pct: Take profit percentage
        is_long: True for long position, False for short

    Returns:
        Take profit price
    """
    take_profit_amount = entry_price * take_profit_pct / Decimal(100)

    if is_long:
        return entry_price + take_profit_amount
    return entry_price - take_profit_amount


def check_risk_alerts(
    margin_info: MarginInfo,
    limits: RiskLimits,
    current_positions: int,
    daily_pnl: Decimal,
) -> list[RiskAlert]:
    """
    Check for risk alerts based on current state.

    Args:
        margin_info: Current margin information
        limits: Risk limits
        current_positions: Number of open positions
        daily_pnl: Daily profit/loss

    Returns:
        List of active risk alerts
    """
    alerts: list[RiskAlert] = []

    # Check position limit
    if current_positions >= limits.position_limit:
        alerts.append(
            PositionLimitExceeded(
                current_positions=current_positions, limit=limits.position_limit
            )
        )

    # Check margin ratio
    margin_threshold = limits.margin_requirement / Decimal(100)
    if margin_info.margin_ratio > margin_threshold:
        alerts.append(
            MarginCall(
                margin_ratio=margin_info.margin_ratio, threshold=margin_threshold
            )
        )

    # Check daily loss limit
    if daily_pnl < Decimal(0) and abs(daily_pnl) >= limits.daily_loss_limit:
        alerts.append(
            DailyLossLimit(current_loss=abs(daily_pnl), limit=limits.daily_loss_limit)
        )

    return alerts


def calculate_position_risk(
    position_size: Decimal, entry_price: Decimal, stop_loss_price: Decimal
) -> Decimal:
    """
    Calculate risk amount for a position.

    Args:
        position_size: Size of the position
        entry_price: Entry price
        stop_loss_price: Stop loss price

    Returns:
        Risk amount in USD
    """
    price_diff = abs(entry_price - stop_loss_price)
    return position_size * price_diff


def is_within_risk_limits(
    proposed_risk: Decimal,
    current_exposure: Decimal,
    balance: Decimal,
    max_risk_pct: Decimal,
) -> bool:
    """
    Check if proposed trade is within risk limits.

    Args:
        proposed_risk: Risk of proposed trade
        current_exposure: Current total exposure
        balance: Account balance
        max_risk_pct: Maximum risk percentage

    Returns:
        True if within limits, False otherwise
    """
    total_risk = current_exposure + proposed_risk
    max_allowed_risk = balance * max_risk_pct / Decimal(100)
    return total_risk <= max_allowed_risk


# Advanced Risk Management Pure Functions


def create_circuit_breaker_state(
    failure_threshold: int = 5, timeout_seconds: int = 300
) -> CircuitBreakerState:
    """Create initial circuit breaker state."""
    return CircuitBreakerState(
        state="CLOSED",
        failure_count=0,
        failure_threshold=failure_threshold,
        timeout_seconds=timeout_seconds,
        last_failure_time=None,
        consecutive_successes=0,
        failure_history=(),
    )


def record_circuit_breaker_failure(
    state: CircuitBreakerState,
    failure_type: str,
    error_message: str,
    severity: str = "medium",
    timestamp: Optional[datetime] = None,
) -> CircuitBreakerState:
    """Record a failure in the circuit breaker."""
    if timestamp is None:
        timestamp = datetime.now()
    
    failure_record = FailureRecord(
        timestamp=timestamp,
        failure_type=failure_type,
        error_message=error_message,
        severity=severity,
    )
    
    new_failure_count = state.failure_count + 1
    new_state = "OPEN" if new_failure_count >= state.failure_threshold else state.state
    
    return CircuitBreakerState(
        state=new_state,
        failure_count=new_failure_count,
        failure_threshold=state.failure_threshold,
        timeout_seconds=state.timeout_seconds,
        last_failure_time=timestamp,
        consecutive_successes=0,
        failure_history=state.failure_history + (failure_record,),
    )


def record_circuit_breaker_success(state: CircuitBreakerState) -> CircuitBreakerState:
    """Record a success in the circuit breaker."""
    new_successes = state.consecutive_successes + 1
    
    # Transition to CLOSED if we have enough successes in HALF_OPEN state
    new_state = state.state
    if state.state == "HALF_OPEN" and new_successes >= 3:
        new_state = "CLOSED"
        new_successes = 0
    
    return CircuitBreakerState(
        state=new_state,
        failure_count=state.failure_count,
        failure_threshold=state.failure_threshold,
        timeout_seconds=state.timeout_seconds,
        last_failure_time=state.last_failure_time,
        consecutive_successes=new_successes,
        failure_history=state.failure_history,
    )


def update_circuit_breaker_state(
    state: CircuitBreakerState, current_time: Optional[datetime] = None
) -> CircuitBreakerState:
    """Update circuit breaker state based on time elapsed."""
    if current_time is None:
        current_time = datetime.now()
    
    # If OPEN and timeout has passed, transition to HALF_OPEN
    if (
        state.state == "OPEN"
        and state.last_failure_time
        and (current_time - state.last_failure_time).total_seconds() >= state.timeout_seconds
    ):
        return CircuitBreakerState(
            state="HALF_OPEN",
            failure_count=state.failure_count,
            failure_threshold=state.failure_threshold,
            timeout_seconds=state.timeout_seconds,
            last_failure_time=state.last_failure_time,
            consecutive_successes=0,
            failure_history=state.failure_history,
        )
    
    return state


def create_emergency_stop_state() -> EmergencyStopState:
    """Create initial emergency stop state."""
    return EmergencyStopState(
        is_stopped=False,
        stop_reason=None,
        stopped_at=None,
        manual_override=False,
    )


def trigger_emergency_stop(
    state: EmergencyStopState,
    reason_type: str,
    description: str,
    timestamp: Optional[datetime] = None,
) -> EmergencyStopState:
    """Trigger emergency stop with reason."""
    if timestamp is None:
        timestamp = datetime.now()
    
    stop_reason = EmergencyStopReason(
        reason_type=reason_type,
        description=description,
        triggered_at=timestamp,
        severity="critical",
    )
    
    return EmergencyStopState(
        is_stopped=True,
        stop_reason=stop_reason,
        stopped_at=timestamp,
        manual_override=state.manual_override,
    )


def clear_emergency_stop(
    state: EmergencyStopState, manual_override: bool = False
) -> EmergencyStopState:
    """Clear emergency stop state."""
    return EmergencyStopState(
        is_stopped=False,
        stop_reason=None,
        stopped_at=None,
        manual_override=manual_override,
    )


def create_api_protection_state(
    max_retries: int = 3, base_delay: float = 1.0
) -> APIProtectionState:
    """Create initial API protection state."""
    return APIProtectionState(
        consecutive_failures=0,
        max_retries=max_retries,
        base_delay=base_delay,
        last_failure_time=None,
        is_healthy=True,
        backoff_multiplier=2.0,
    )


def record_api_failure(
    state: APIProtectionState, timestamp: Optional[datetime] = None
) -> APIProtectionState:
    """Record an API failure."""
    if timestamp is None:
        timestamp = datetime.now()
    
    new_failures = state.consecutive_failures + 1
    
    return APIProtectionState(
        consecutive_failures=new_failures,
        max_retries=state.max_retries,
        base_delay=state.base_delay,
        last_failure_time=timestamp,
        is_healthy=new_failures < state.max_retries,
        backoff_multiplier=state.backoff_multiplier,
    )


def record_api_success(state: APIProtectionState) -> APIProtectionState:
    """Record an API success."""
    return APIProtectionState(
        consecutive_failures=0,
        max_retries=state.max_retries,
        base_delay=state.base_delay,
        last_failure_time=state.last_failure_time,
        is_healthy=True,
        backoff_multiplier=state.backoff_multiplier,
    )


def calculate_portfolio_exposure(
    positions: List[Dict[str, any]], account_balance: Decimal
) -> PortfolioExposure:
    """Calculate portfolio exposure metrics."""
    if not positions or account_balance <= 0:
        return PortfolioExposure(
            total_exposure=Decimal(0),
            symbol_exposures={},
            sector_exposures={},
            correlation_risk=0.0,
            concentration_risk=0.0,
            max_single_position_pct=0.0,
            portfolio_heat=0.0,
        )
    
    symbol_exposures = {}
    total_exposure = Decimal(0)
    max_position_value = Decimal(0)
    
    for position in positions:
        symbol = position.get("symbol", "")
        size = Decimal(str(position.get("size", 0)))
        price = Decimal(str(position.get("price", 0)))
        
        if size > 0 and price > 0:
            position_value = size * price
            symbol_exposures[symbol] = position_value
            total_exposure += position_value
            max_position_value = max(max_position_value, position_value)
    
    # Calculate metrics
    max_single_position_pct = float(max_position_value / account_balance * 100) if account_balance > 0 else 0.0
    portfolio_heat = float(total_exposure / account_balance * 100) if account_balance > 0 else 0.0
    
    # Simplified correlation and concentration risk
    num_positions = len([p for p in positions if p.get("size", 0) > 0])
    concentration_risk = max_single_position_pct / 100.0 if num_positions > 0 else 0.0
    correlation_risk = min(0.8, 1.0 / max(1, num_positions))  # Higher risk with fewer positions
    
    return PortfolioExposure(
        total_exposure=total_exposure,
        symbol_exposures=symbol_exposures,
        sector_exposures={},  # Could be enhanced with sector mapping
        correlation_risk=correlation_risk,
        concentration_risk=concentration_risk,
        max_single_position_pct=max_single_position_pct,
        portfolio_heat=portfolio_heat,
    )


def calculate_leverage_analysis(
    current_leverage: Decimal,
    volatility: float,
    win_rate: float,
    max_leverage: Decimal = Decimal(10),
) -> LeverageAnalysis:
    """Calculate leverage analysis and recommendations."""
    # Calculate optimal leverage based on Kelly Criterion and volatility
    if win_rate <= 0 or win_rate >= 1 or volatility <= 0:
        optimal_leverage = Decimal(1)
    else:
        # Simplified optimal leverage calculation
        kelly_fraction = (win_rate * 2 - (1 - win_rate)) / 2  # Simplified Kelly
        volatility_adjustment = max(0.1, 1.0 - volatility)  # Reduce leverage for high volatility
        optimal_leverage = min(
            max_leverage,
            Decimal(str(kelly_fraction * volatility_adjustment * 5))  # Scale up Kelly result
        )
        optimal_leverage = max(Decimal(1), optimal_leverage)  # Minimum 1x leverage
    
    # Determine recommended action
    if current_leverage > optimal_leverage * Decimal("1.2"):
        recommended_action = "DECREASE"
    elif current_leverage < optimal_leverage * Decimal("0.8"):
        recommended_action = "INCREASE"
    else:
        recommended_action = "MAINTAIN"
    
    return LeverageAnalysis(
        current_leverage=current_leverage,
        optimal_leverage=optimal_leverage,
        max_allowed_leverage=max_leverage,
        volatility_adjustment=1.0 - volatility,
        win_rate_adjustment=win_rate,
        risk_adjustment=1.0,  # Could be enhanced with additional risk factors
        recommended_action=recommended_action,
    )


def calculate_drawdown_analysis(
    current_balance: Decimal,
    peak_balance: Decimal,
    peak_time: Optional[datetime] = None,
    current_time: Optional[datetime] = None,
) -> DrawdownAnalysis:
    """Calculate drawdown analysis."""
    if current_time is None:
        current_time = datetime.now()
    
    if peak_balance <= 0:
        peak_balance = current_balance
    
    # Ensure peak is at least current balance
    actual_peak = max(peak_balance, current_balance)
    
    # Calculate drawdown
    if actual_peak > 0:
        current_drawdown_pct = float((actual_peak - current_balance) / actual_peak * 100)
    else:
        current_drawdown_pct = 0.0
    
    # Calculate duration
    if peak_time and current_drawdown_pct > 0:
        duration_hours = (current_time - peak_time).total_seconds() / 3600
    else:
        duration_hours = 0.0
    
    # Recovery target (back to peak)
    recovery_target = actual_peak
    
    return DrawdownAnalysis(
        current_drawdown_pct=current_drawdown_pct,
        max_drawdown_pct=current_drawdown_pct,  # Could be tracked over time
        drawdown_duration_hours=duration_hours,
        peak_balance=actual_peak,
        current_balance=current_balance,
        is_in_drawdown=current_drawdown_pct > 1.0,  # 1% threshold
        recovery_target=recovery_target,
    )


def assess_risk_level(
    portfolio_exposure: PortfolioExposure,
    leverage_analysis: LeverageAnalysis,
    drawdown_analysis: DrawdownAnalysis,
    consecutive_losses: int,
    margin_usage_pct: float,
) -> RiskLevelAssessment:
    """Assess overall risk level."""
    risk_factors = []
    score = 0.0
    
    # Portfolio heat risk
    if portfolio_exposure.portfolio_heat > 15.0:
        risk_factors.append(f"High portfolio heat: {portfolio_exposure.portfolio_heat:.1f}%")
        score += 25
    elif portfolio_exposure.portfolio_heat > 10.0:
        risk_factors.append(f"Elevated portfolio heat: {portfolio_exposure.portfolio_heat:.1f}%")
        score += 15
    
    # Leverage risk
    if leverage_analysis.current_leverage > leverage_analysis.max_allowed_leverage * Decimal("0.8"):
        risk_factors.append(f"High leverage: {leverage_analysis.current_leverage}x")
        score += 20
    
    # Drawdown risk
    if drawdown_analysis.is_severe_drawdown:
        risk_factors.append(f"Severe drawdown: {drawdown_analysis.current_drawdown_pct:.1f}%")
        score += 30
    elif drawdown_analysis.current_drawdown_pct > 10.0:
        risk_factors.append(f"Significant drawdown: {drawdown_analysis.current_drawdown_pct:.1f}%")
        score += 20
    
    # Consecutive losses risk
    if consecutive_losses >= 5:
        risk_factors.append(f"High consecutive losses: {consecutive_losses}")
        score += 25
    elif consecutive_losses >= 3:
        risk_factors.append(f"Multiple consecutive losses: {consecutive_losses}")
        score += 15
    
    # Margin usage risk
    if margin_usage_pct > 80.0:
        risk_factors.append(f"High margin usage: {margin_usage_pct:.1f}%")
        score += 20
    elif margin_usage_pct > 60.0:
        risk_factors.append(f"Elevated margin usage: {margin_usage_pct:.1f}%")
        score += 10
    
    # Determine risk level
    if score >= 70:
        risk_level = "CRITICAL"
        recommendations = ["Stop all trading", "Review risk management", "Reduce position sizes"]
    elif score >= 50:
        risk_level = "HIGH"
        recommendations = ["Reduce position sizes", "Lower leverage", "Increase stop losses"]
    elif score >= 25:
        risk_level = "MEDIUM"
        recommendations = ["Monitor closely", "Consider reducing exposure", "Review strategy"]
    else:
        risk_level = "LOW"
        recommendations = ["Continue normal operations", "Monitor market conditions"]
    
    return RiskLevelAssessment(
        risk_level=risk_level,
        score=score,
        contributing_factors=risk_factors,
        recommendations=recommendations,
        timestamp=datetime.now(),
    )


def create_comprehensive_risk_state(
    account_balance: Decimal,
    positions: List[Dict[str, any]],
    current_leverage: Decimal,
    volatility: float,
    win_rate: float,
    consecutive_losses: int,
    margin_usage_pct: float,
    peak_balance: Decimal,
) -> ComprehensiveRiskState:
    """Create comprehensive risk state from current conditions."""
    # Create individual components
    circuit_breaker = create_circuit_breaker_state()
    emergency_stop = create_emergency_stop_state()
    api_protection = create_api_protection_state()
    
    # Calculate portfolio exposure
    portfolio_exposure = calculate_portfolio_exposure(positions, account_balance)
    
    # Calculate leverage analysis
    leverage_analysis = calculate_leverage_analysis(
        current_leverage, volatility, win_rate
    )
    
    # Calculate drawdown analysis
    drawdown_analysis = calculate_drawdown_analysis(
        account_balance, peak_balance
    )
    
    # Create daily P&L (simplified)
    daily_pnl = DailyPnL(date=datetime.now().date())
    
    # Assess risk level
    risk_assessment = assess_risk_level(
        portfolio_exposure,
        leverage_analysis,
        drawdown_analysis,
        consecutive_losses,
        margin_usage_pct,
    )
    
    # Create risk metrics snapshot
    risk_metrics = RiskMetricsSnapshot(
        timestamp=datetime.now(),
        account_balance=account_balance,
        portfolio_exposure=portfolio_exposure,
        leverage_analysis=leverage_analysis,
        drawdown_analysis=drawdown_analysis,
        daily_pnl=daily_pnl,
        risk_assessment=risk_assessment,
        position_count=len([p for p in positions if p.get("size", 0) > 0]),
        consecutive_losses=consecutive_losses,
        margin_usage_pct=margin_usage_pct,
    )
    
    # Create correlation matrix (simplified)
    symbols = tuple(p.get("symbol", "") for p in positions if p.get("symbol"))
    correlation_matrix = CorrelationMatrix(
        symbols=symbols,
        correlations={},  # Could be populated with real correlation data
        timestamp=datetime.now(),
    )
    
    return ComprehensiveRiskState(
        circuit_breaker=circuit_breaker,
        emergency_stop=emergency_stop,
        api_protection=api_protection,
        risk_metrics=risk_metrics,
        correlation_matrix=correlation_matrix,
        validation_history=(),
    )


def check_advanced_risk_alerts(
    risk_state: ComprehensiveRiskState,
) -> List[AdvancedRiskAlert]:
    """Check for advanced risk alerts."""
    alerts = []
    current_time = datetime.now()
    
    # Circuit breaker alert
    if risk_state.circuit_breaker.is_open:
        alerts.append(AdvancedRiskAlert(
            alert_type=AdvancedRiskAlertType.CIRCUIT_BREAKER_OPEN,
            severity="high",
            title="Circuit Breaker Open",
            description=f"Circuit breaker is open with {risk_state.circuit_breaker.failure_count} failures",
            triggered_at=current_time,
            metric_value=float(risk_state.circuit_breaker.failure_count),
            threshold_value=float(risk_state.circuit_breaker.failure_threshold),
            recommended_action="Wait for timeout or fix underlying issues",
        ))
    
    # Emergency stop alert
    if risk_state.emergency_stop.is_stopped:
        alerts.append(AdvancedRiskAlert(
            alert_type=AdvancedRiskAlertType.EMERGENCY_STOP_TRIGGERED,
            severity="critical",
            title="Emergency Stop Active",
            description=risk_state.emergency_stop.stop_reason.description if risk_state.emergency_stop.stop_reason else "Emergency stop triggered",
            triggered_at=current_time,
            metric_value=1.0,
            threshold_value=0.0,
            recommended_action="Review emergency conditions and clear stop manually",
        ))
    
    # API protection alert
    if not risk_state.api_protection.is_healthy:
        alerts.append(AdvancedRiskAlert(
            alert_type=AdvancedRiskAlertType.API_FAILURES_EXCESSIVE,
            severity="medium",
            title="API Failures Excessive",
            description=f"API protection engaged due to {risk_state.api_protection.consecutive_failures} consecutive failures",
            triggered_at=current_time,
            metric_value=float(risk_state.api_protection.consecutive_failures),
            threshold_value=float(risk_state.api_protection.max_retries),
            recommended_action="Check API connectivity and reduce request frequency",
        ))
    
    # Leverage alert
    leverage_ratio = float(risk_state.risk_metrics.leverage_analysis.current_leverage / 
                          risk_state.risk_metrics.leverage_analysis.optimal_leverage)
    if leverage_ratio > 1.5:
        alerts.append(AdvancedRiskAlert(
            alert_type=AdvancedRiskAlertType.LEVERAGE_EXCESSIVE,
            severity="medium",
            title="Leverage Excessive",
            description=f"Current leverage {risk_state.risk_metrics.leverage_analysis.current_leverage}x exceeds optimal {risk_state.risk_metrics.leverage_analysis.optimal_leverage}x",
            triggered_at=current_time,
            metric_value=float(risk_state.risk_metrics.leverage_analysis.current_leverage),
            threshold_value=float(risk_state.risk_metrics.leverage_analysis.optimal_leverage),
            recommended_action="Reduce position sizes or close positions",
        ))
    
    # Drawdown alert
    if risk_state.risk_metrics.drawdown_analysis.is_severe_drawdown:
        alerts.append(AdvancedRiskAlert(
            alert_type=AdvancedRiskAlertType.DRAWDOWN_SEVERE,
            severity="high",
            title="Severe Drawdown",
            description=f"Current drawdown {risk_state.risk_metrics.drawdown_analysis.current_drawdown_pct:.1f}% is severe",
            triggered_at=current_time,
            metric_value=risk_state.risk_metrics.drawdown_analysis.current_drawdown_pct,
            threshold_value=15.0,
            recommended_action="Stop trading and review strategy",
        ))
    
    # Portfolio overexposure alert
    if risk_state.risk_metrics.portfolio_exposure.is_overexposed:
        alerts.append(AdvancedRiskAlert(
            alert_type=AdvancedRiskAlertType.PORTFOLIO_OVEREXPOSED,
            severity="medium",
            title="Portfolio Overexposed",
            description=f"Portfolio heat {risk_state.risk_metrics.portfolio_exposure.portfolio_heat:.1f}% exceeds safe levels",
            triggered_at=current_time,
            metric_value=risk_state.risk_metrics.portfolio_exposure.portfolio_heat,
            threshold_value=10.0,
            recommended_action="Reduce position sizes across portfolio",
        ))
    
    # Consecutive losses alert
    if risk_state.risk_metrics.consecutive_losses >= 5:
        alerts.append(AdvancedRiskAlert(
            alert_type=AdvancedRiskAlertType.CONSECUTIVE_LOSSES_HIGH,
            severity="high",
            title="High Consecutive Losses",
            description=f"{risk_state.risk_metrics.consecutive_losses} consecutive losses detected",
            triggered_at=current_time,
            metric_value=float(risk_state.risk_metrics.consecutive_losses),
            threshold_value=5.0,
            recommended_action="Stop trading and review strategy performance",
        ))
    
    return alerts
