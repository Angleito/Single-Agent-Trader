"""
Market Making Configuration System.

This module provides comprehensive configuration management for market making strategies,
including pydantic validation, environment variable integration, and support for
different trading profiles.
"""

import logging
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class MarketMakingStrategyConfig(BaseModel):
    """Core strategy parameters for market making operations."""

    # Base spread configuration
    base_spread_bps: int = Field(
        default=10,
        ge=1,
        le=500,
        description="Base spread in basis points (0.10% = 10 bps)",
    )
    min_spread_bps: int = Field(
        default=5, ge=1, le=100, description="Minimum allowed spread in basis points"
    )
    max_spread_bps: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum allowed spread in basis points",
    )

    # Order level configuration
    order_levels: int = Field(
        default=3, ge=1, le=10, description="Number of order levels per side"
    )
    level_multiplier: float = Field(
        default=1.5,
        ge=1.1,
        le=3.0,
        description="Multiplier for spacing between order levels",
    )

    # Position sizing
    max_position_pct: float = Field(
        default=25.0,
        ge=1.0,
        le=50.0,
        description="Maximum position size as percentage of account equity",
    )
    position_size_per_level: float = Field(
        default=5.0,
        ge=0.5,
        le=20.0,
        description="Position size per level as percentage of max position",
    )

    # Signal integration
    vumanchu_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight given to VuManChu signals in spread calculations",
    )
    signal_adjustment_factor: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Factor for adjusting spreads based on signal strength",
    )
    confidence_threshold: float = Field(
        default=0.6,
        ge=0.3,
        le=0.9,
        description="Minimum confidence threshold for signal-based adjustments",
    )

    @field_validator("min_spread_bps", "max_spread_bps")
    @classmethod
    def validate_spread_hierarchy(cls, v, info):
        """Ensure spread hierarchy is maintained."""
        if info.field_name == "min_spread_bps":
            return v
        if info.field_name == "max_spread_bps" and "base_spread_bps" in info.data:
            base_spread = info.data.get("base_spread_bps", 10)
            if v <= base_spread:
                raise ValueError(
                    f"max_spread_bps ({v}) must be greater than base_spread_bps ({base_spread})"
                )
        return v


class MarketMakingRiskConfig(BaseModel):
    """Risk management parameters for market making."""

    # Position limits
    max_position_value: Decimal = Field(
        default=Decimal(10000),
        ge=Decimal(100),
        le=Decimal(1000000),
        description="Maximum position value in base currency",
    )
    max_inventory_imbalance: float = Field(
        default=0.2,
        ge=0.05,
        le=0.5,
        description="Maximum allowed inventory imbalance (20% = 0.2)",
    )
    rebalancing_threshold: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="Inventory imbalance percentage to trigger rebalancing",
    )
    emergency_threshold: float = Field(
        default=15.0,
        ge=5.0,
        le=30.0,
        description="Emergency threshold to force position flattening",
    )

    # Time-based limits
    inventory_timeout_hours: float = Field(
        default=4.0,
        ge=0.5,
        le=24.0,
        description="Maximum hours to hold inventory position",
    )
    max_order_age_minutes: int = Field(
        default=30,
        ge=1,
        le=1440,
        description="Maximum age for open orders before cancellation",
    )

    # Stop loss configuration
    stop_loss_pct: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Stop loss percentage for emergency positions",
    )
    daily_loss_limit_pct: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="Daily loss limit as percentage of account equity",
    )

    # Volatility controls
    volatility_threshold: float = Field(
        default=0.3,
        ge=0.1,
        le=1.0,
        description="Volatility threshold for spread adjustments",
    )
    volatility_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Multiplier for spread increases during high volatility",
    )


class MarketMakingOrderConfig(BaseModel):
    """Order management configuration for market making."""

    # Order timing
    order_update_interval_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Interval between order updates in seconds",
    )
    order_timeout_seconds: int = Field(
        default=30, ge=5, le=300, description="Timeout for individual order operations"
    )

    # Price thresholds
    price_update_threshold_bps: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Price change threshold in bps to trigger order updates",
    )
    spread_update_threshold_bps: int = Field(
        default=3,
        ge=1,
        le=30,
        description="Spread change threshold in bps to trigger order updates",
    )

    # Order size adjustments
    volume_adjustment_factor: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Factor for adjusting order sizes based on volume",
    )
    liquidity_adjustment_factor: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Factor for adjusting order sizes based on liquidity",
    )

    # Slippage and fees
    max_slippage_bps: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum acceptable slippage in basis points",
    )
    fee_buffer_bps: int = Field(
        default=2, ge=1, le=20, description="Additional buffer for fees in basis points"
    )


class MarketMakingPerformanceConfig(BaseModel):
    """Performance monitoring and alerting configuration."""

    # Performance thresholds
    min_fill_rate: float = Field(
        default=0.3,
        ge=0.1,
        le=0.8,
        description="Minimum acceptable fill rate (30% = 0.3)",
    )
    min_spread_capture_rate: float = Field(
        default=0.6,
        ge=0.3,
        le=0.9,
        description="Minimum spread capture rate (60% = 0.6)",
    )
    max_fee_ratio: float = Field(
        default=0.5,
        ge=0.1,
        le=0.8,
        description="Maximum ratio of fees to profit (50% = 0.5)",
    )

    # Turnover and activity
    min_turnover_rate: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Minimum daily turnover rate (2x = 2.0)",
    )
    min_signal_effectiveness: float = Field(
        default=0.4,
        ge=0.2,
        le=0.8,
        description="Minimum signal effectiveness rate (40% = 0.4)",
    )

    # Risk metrics
    max_negative_pnl_streak: int = Field(
        default=5, ge=2, le=20, description="Maximum consecutive negative P&L periods"
    )
    max_drawdown_pct: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Maximum acceptable drawdown percentage",
    )

    # Reporting intervals
    performance_report_interval_minutes: int = Field(
        default=15,
        ge=1,
        le=1440,
        description="Interval for performance reports in minutes",
    )
    alert_cooldown_minutes: int = Field(
        default=5, ge=1, le=60, description="Cooldown period between similar alerts"
    )


class MarketMakingBluefinConfig(BaseModel):
    """Bluefin-specific configuration parameters."""

    # Network settings
    network: Literal["mainnet", "testnet"] = Field(
        default="mainnet", description="Bluefin network to use"
    )

    # Fee calculations
    maker_fee_rate: float = Field(
        default=0.0002,  # 0.02%
        ge=0.0,
        le=0.01,
        description="Maker fee rate for Bluefin trades",
    )
    taker_fee_rate: float = Field(
        default=0.0005,  # 0.05%
        ge=0.0,
        le=0.01,
        description="Taker fee rate for Bluefin trades",
    )

    # Gas and transaction costs
    gas_buffer_multiplier: float = Field(
        default=1.2,
        ge=1.0,
        le=2.0,
        description="Gas price buffer multiplier for transactions",
    )
    max_gas_price_gwei: int = Field(
        default=20, ge=1, le=100, description="Maximum acceptable gas price in Gwei"
    )

    # Order book depth
    orderbook_depth_levels: int = Field(
        default=10, ge=5, le=50, description="Number of orderbook levels to fetch"
    )
    min_liquidity_threshold: Decimal = Field(
        default=Decimal(1000),
        ge=Decimal(100),
        le=Decimal(100000),
        description="Minimum liquidity threshold for market making",
    )


class MarketMakingConfig(BaseModel):
    """Comprehensive market making configuration."""

    # Core components
    strategy: MarketMakingStrategyConfig = Field(
        default_factory=MarketMakingStrategyConfig,
        description="Core strategy parameters",
    )
    risk: MarketMakingRiskConfig = Field(
        default_factory=MarketMakingRiskConfig, description="Risk management parameters"
    )
    orders: MarketMakingOrderConfig = Field(
        default_factory=MarketMakingOrderConfig,
        description="Order management parameters",
    )
    performance: MarketMakingPerformanceConfig = Field(
        default_factory=MarketMakingPerformanceConfig,
        description="Performance monitoring parameters",
    )
    bluefin: MarketMakingBluefinConfig = Field(
        default_factory=MarketMakingBluefinConfig,
        description="Bluefin-specific parameters",
    )

    # Engine configuration
    enabled: bool = Field(default=False, description="Enable market making engine")
    symbol: str = Field(
        default="SUI-PERP", description="Trading symbol for market making"
    )
    cycle_interval_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Main engine cycle interval in seconds",
    )
    max_errors_per_hour: int = Field(
        default=50, ge=10, le=500, description="Maximum errors per hour before shutdown"
    )

    # Profile-based configurations
    profile: Literal["conservative", "moderate", "aggressive", "custom"] = Field(
        default="moderate",
        description="Trading profile for automatic parameter adjustment",
    )

    @model_validator(mode="after")
    def validate_configuration_consistency(self) -> "MarketMakingConfig":
        """Validate configuration consistency across components."""
        # Ensure spread hierarchy
        if self.strategy.min_spread_bps >= self.strategy.base_spread_bps:
            raise ValueError(
                f"min_spread_bps ({self.strategy.min_spread_bps}) must be less than "
                f"base_spread_bps ({self.strategy.base_spread_bps})"
            )

        if self.strategy.base_spread_bps >= self.strategy.max_spread_bps:
            raise ValueError(
                f"base_spread_bps ({self.strategy.base_spread_bps}) must be less than "
                f"max_spread_bps ({self.strategy.max_spread_bps})"
            )

        # Ensure risk thresholds are logical
        if self.risk.rebalancing_threshold >= self.risk.emergency_threshold:
            raise ValueError(
                f"rebalancing_threshold ({self.risk.rebalancing_threshold}) must be less than "
                f"emergency_threshold ({self.risk.emergency_threshold})"
            )

        # Ensure order timing is reasonable
        if self.orders.order_update_interval_seconds >= self.cycle_interval_seconds:
            logger.warning(
                "Order update interval (%ss) is >= cycle interval (%ss)",
                self.orders.order_update_interval_seconds,
                self.cycle_interval_seconds,
            )

        return self

    def apply_profile(self, profile: str) -> "MarketMakingConfig":
        """Apply profile-based configuration adjustments."""
        profile_configs = {
            "conservative": {
                "strategy": {
                    "base_spread_bps": 15,
                    "min_spread_bps": 8,
                    "max_spread_bps": 50,
                    "max_position_pct": 15.0,
                    "order_levels": 2,
                    "vumanchu_weight": 0.3,
                },
                "risk": {
                    "max_inventory_imbalance": 0.15,
                    "rebalancing_threshold": 3.0,
                    "emergency_threshold": 10.0,
                    "stop_loss_pct": 1.5,
                    "daily_loss_limit_pct": 2.0,
                },
                "performance": {
                    "min_fill_rate": 0.2,
                    "max_negative_pnl_streak": 3,
                    "max_drawdown_pct": 2.0,
                },
            },
            "moderate": {
                "strategy": {
                    "base_spread_bps": 10,
                    "min_spread_bps": 5,
                    "max_spread_bps": 100,
                    "max_position_pct": 25.0,
                    "order_levels": 3,
                    "vumanchu_weight": 0.6,
                },
                "risk": {
                    "max_inventory_imbalance": 0.2,
                    "rebalancing_threshold": 5.0,
                    "emergency_threshold": 15.0,
                    "stop_loss_pct": 2.0,
                    "daily_loss_limit_pct": 5.0,
                },
                "performance": {
                    "min_fill_rate": 0.3,
                    "max_negative_pnl_streak": 5,
                    "max_drawdown_pct": 3.0,
                },
            },
            "aggressive": {
                "strategy": {
                    "base_spread_bps": 8,
                    "min_spread_bps": 3,
                    "max_spread_bps": 150,
                    "max_position_pct": 40.0,
                    "order_levels": 5,
                    "vumanchu_weight": 0.8,
                },
                "risk": {
                    "max_inventory_imbalance": 0.3,
                    "rebalancing_threshold": 8.0,
                    "emergency_threshold": 20.0,
                    "stop_loss_pct": 3.0,
                    "daily_loss_limit_pct": 8.0,
                },
                "performance": {
                    "min_fill_rate": 0.4,
                    "max_negative_pnl_streak": 8,
                    "max_drawdown_pct": 5.0,
                },
            },
        }

        if profile not in profile_configs:
            logger.warning("Unknown profile '%s', using moderate defaults", profile)
            profile = "moderate"

        config_updates = profile_configs[profile]

        # Create new config with profile adjustments
        config_dict = self.model_dump()

        for section, updates in config_updates.items():
            if section in config_dict:
                config_dict[section].update(updates)

        return MarketMakingConfig.model_validate(config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary format."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "MarketMakingConfig":
        """Create configuration from dictionary."""
        return cls.model_validate(config_dict)

    def get_component_config(self, component: str) -> dict[str, Any]:
        """Get configuration dictionary for a specific component."""
        component_map = {
            "strategy": self.strategy,
            "risk": self.risk,
            "orders": self.orders,
            "performance": self.performance,
            "bluefin": self.bluefin,
        }

        if component not in component_map:
            raise ValueError(f"Unknown component: {component}")

        return component_map[component].model_dump()


def create_default_config(profile: str = "moderate") -> MarketMakingConfig:
    """Create a default market making configuration with the specified profile."""
    config = MarketMakingConfig()
    return config.apply_profile(profile)


def validate_config(config: dict[str, Any]) -> MarketMakingConfig:
    """Validate and create a market making configuration from a dictionary."""
    try:
        return MarketMakingConfig.model_validate(config)
    except Exception as e:
        logger.exception("Configuration validation failed: %s", e)
        raise ValueError(f"Invalid market making configuration: {e}")
