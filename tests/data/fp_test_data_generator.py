"""
FP Test Data Generator

Functional Programming compatible test data generator that creates immutable
test data structures for comprehensive testing of FP trading systems.
This generator works with FP types and provides realistic data scenarios.
"""

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Safe imports with fallbacks
try:
    from bot.fp.types.market import MarketSnapshot, OHLCV
    from bot.fp.types.portfolio import Portfolio, Position, TradeResult
    from bot.fp.types.trading import Long, Short, Hold, MarketMake, LimitOrder, MarketOrder
    from bot.fp.types.base import Money, Percentage, Symbol, TimeInterval
    from bot.fp.types.config import TradingConfig, RiskConfig
    from bot.fp.types.effects import Ok, Err, Some, Nothing, IO
    FP_TYPES_AVAILABLE = True
except ImportError:
    FP_TYPES_AVAILABLE = False
    # Create minimal fallback types for when FP types aren't available
    class MarketSnapshot:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class Portfolio:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class Position:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


# =============================================================================
# FP TEST DATA CONFIGURATION
# =============================================================================

@dataclass
class FPTestDataConfig:
    """Configuration for FP test data generation."""
    
    # Market data parameters
    base_price: float = 50000.0
    volatility: float = 0.02
    trend_strength: float = 0.0
    time_periods: int = 1000
    
    # Symbol configuration
    primary_symbol: str = "BTC-USD"
    secondary_symbols: List[str] = None
    
    # Portfolio parameters
    initial_balance: Decimal = Decimal("100000.00")
    max_position_size: float = 0.25
    
    # Risk parameters
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05
    
    # Scenario parameters
    scenario_type: str = "default"  # default, trending, ranging, volatile
    random_seed: Optional[int] = 42
    
    def __post_init__(self):
        if self.secondary_symbols is None:
            self.secondary_symbols = ["ETH-USD", "SOL-USD", "AVAX-USD"]


# =============================================================================
# FP MARKET DATA GENERATOR
# =============================================================================

class FPMarketDataGenerator:
    """Generator for FP-compatible market data."""
    
    def __init__(self, config: FPTestDataConfig = None):
        self.config = config or FPTestDataConfig()
        if self.config.random_seed:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        
        self.rng = np.random.default_rng(self.config.random_seed)
    
    def generate_market_snapshots(self, count: int = None) -> List[MarketSnapshot]:
        """Generate series of immutable market snapshots."""
        if not FP_TYPES_AVAILABLE:
            return []
        
        count = count or self.config.time_periods
        snapshots = []
        
        base_time = datetime.now(UTC)
        current_price = self.config.base_price
        
        for i in range(count):
            # Generate price movement
            price_change = self._generate_price_change(i, count)
            current_price *= (1 + price_change)
            current_price = max(current_price, 0.01)  # Ensure positive
            
            # Generate bid/ask spread
            spread_pct = self.rng.uniform(0.0005, 0.002)
            half_spread = current_price * spread_pct / 2
            
            bid = Decimal(str(current_price - half_spread))
            ask = Decimal(str(current_price + half_spread))
            price = (bid + ask) / 2
            
            # Generate volume with correlation to price movement
            base_volume = 100.0
            volume_multiplier = 1 + abs(price_change) * 10  # Higher volume on big moves
            volume = Decimal(str(base_volume * volume_multiplier * self.rng.uniform(0.5, 2.0)))
            
            snapshot = MarketSnapshot(
                timestamp=base_time + timedelta(minutes=i),
                symbol=self.config.primary_symbol,
                price=price,
                volume=volume,
                bid=bid,
                ask=ask,
            )
            snapshots.append(snapshot)
        
        return snapshots
    
    def generate_ohlcv_series(self, count: int = None, timeframe: str = "1m") -> List[OHLCV]:
        """Generate OHLCV candle series."""
        if not FP_TYPES_AVAILABLE:
            return []
        
        count = count or self.config.time_periods
        candles = []
        
        base_time = datetime.now(UTC)
        current_price = self.config.base_price
        
        # Calculate time delta based on timeframe
        time_delta = self._parse_timeframe(timeframe)
        
        for i in range(count):
            # Generate price movement for the period
            open_price = current_price
            price_change = self._generate_price_change(i, count)
            close_price = open_price * (1 + price_change)
            close_price = max(close_price, 0.01)
            
            # Generate high and low within the period
            period_volatility = abs(price_change) + self.rng.uniform(0.001, 0.005)
            high_price = max(open_price, close_price) * (1 + period_volatility * 0.5)
            low_price = min(open_price, close_price) * (1 - period_volatility * 0.5)
            low_price = max(low_price, 0.01)
            
            # Generate volume
            base_volume = 100.0
            volume_multiplier = 1 + abs(price_change) * 8
            volume = Decimal(str(base_volume * volume_multiplier * self.rng.uniform(0.7, 1.5)))
            
            candle = OHLCV(
                timestamp=base_time + (time_delta * i),
                open=Decimal(str(open_price)),
                high=Decimal(str(high_price)),
                low=Decimal(str(low_price)),
                close=Decimal(str(close_price)),
                volume=volume,
            )
            candles.append(candle)
            current_price = close_price
        
        return candles
    
    def generate_multi_symbol_data(self, symbols: List[str] = None) -> Dict[str, List[MarketSnapshot]]:
        """Generate market data for multiple symbols."""
        symbols = symbols or [self.config.primary_symbol] + self.config.secondary_symbols
        multi_data = {}
        
        for symbol in symbols:
            # Slightly different parameters for each symbol
            original_symbol = self.config.primary_symbol
            original_price = self.config.base_price
            
            self.config.primary_symbol = symbol
            if symbol != original_symbol:
                # Adjust base price for different symbols
                price_multipliers = {
                    "ETH-USD": 0.08,  # ~4000 vs 50000
                    "SOL-USD": 0.002,  # ~100 vs 50000
                    "AVAX-USD": 0.0008,  # ~40 vs 50000
                }
                multiplier = price_multipliers.get(symbol, 1.0)
                self.config.base_price = original_price * multiplier
            
            multi_data[symbol] = self.generate_market_snapshots()
            
            # Restore original config
            self.config.primary_symbol = original_symbol
            self.config.base_price = original_price
        
        return multi_data
    
    def _generate_price_change(self, step: int, total_steps: int) -> float:
        """Generate price change based on scenario type."""
        if self.config.scenario_type == "trending":
            # Add trend component
            trend_component = self.config.trend_strength
            random_component = self.rng.normal(0, self.config.volatility)
            return trend_component + random_component
        
        elif self.config.scenario_type == "ranging":
            # Oscillating pattern with mean reversion
            phase = (step / total_steps) * 4 * np.pi
            oscillation = np.sin(phase) * 0.001
            random_component = self.rng.normal(0, self.config.volatility * 0.5)
            
            # Mean reversion (pull back to center)
            progress = step / total_steps
            if progress > 0.1:  # After initial period
                mean_reversion = -oscillation * 0.5
            else:
                mean_reversion = 0
            
            return oscillation + random_component + mean_reversion
        
        elif self.config.scenario_type == "volatile":
            # High volatility with clustering
            base_change = self.rng.normal(0, self.config.volatility * 2)
            
            # Add volatility spikes occasionally
            if self.rng.random() < 0.05:  # 5% chance of spike
                spike_magnitude = self.rng.uniform(2, 5)
                spike_direction = self.rng.choice([-1, 1])
                base_change += spike_magnitude * spike_direction * self.config.volatility
            
            return base_change
        
        else:  # default
            return self.rng.normal(0, self.config.volatility)
    
    def _parse_timeframe(self, timeframe: str) -> timedelta:
        """Parse timeframe string to timedelta."""
        timeframe_map = {
            "1s": timedelta(seconds=1),
            "5s": timedelta(seconds=5),
            "15s": timedelta(seconds=15),
            "30s": timedelta(seconds=30),
            "1m": timedelta(minutes=1),
            "3m": timedelta(minutes=3),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1),
        }
        return timeframe_map.get(timeframe, timedelta(minutes=1))


# =============================================================================
# FP PORTFOLIO DATA GENERATOR
# =============================================================================

class FPPortfolioDataGenerator:
    """Generator for FP portfolio and trading data."""
    
    def __init__(self, config: FPTestDataConfig = None):
        self.config = config or FPTestDataConfig()
        if self.config.random_seed:
            random.seed(self.config.random_seed)
        
        self.rng = np.random.default_rng(self.config.random_seed)
    
    def generate_portfolio_timeline(self, market_data: List[MarketSnapshot], trade_frequency: int = 10) -> List[Portfolio]:
        """Generate portfolio state timeline based on market data."""
        if not FP_TYPES_AVAILABLE or not market_data:
            return []
        
        portfolios = []
        current_balance = self.config.initial_balance
        current_position = None
        
        for i, snapshot in enumerate(market_data):
            # Decide whether to trade
            if i % trade_frequency == 0 or (current_position and self._should_close_position(current_position, snapshot)):
                current_position, current_balance = self._execute_trade_decision(
                    current_position, current_balance, snapshot
                )
            
            # Update position current price if we have one
            if current_position:
                current_position = Position(
                    symbol=current_position.symbol,
                    side=current_position.side,
                    size=current_position.size,
                    entry_price=current_position.entry_price,
                    current_price=snapshot.price,
                )
            
            # Create portfolio snapshot
            positions = (current_position,) if current_position else ()
            portfolio = Portfolio(
                positions=positions,
                cash_balance=current_balance,
            )
            portfolios.append(portfolio)
        
        return portfolios
    
    def generate_trade_results(self, count: int = 50) -> List[TradeResult]:
        """Generate series of completed trade results."""
        if not FP_TYPES_AVAILABLE:
            return []
        
        results = []
        base_time = datetime.now(UTC)
        
        for i in range(count):
            # Random trade parameters
            is_profitable = self.rng.choice([True, False], p=[0.6, 0.4])  # 60% win rate
            side = self.rng.choice(["LONG", "SHORT"])
            
            entry_price = Decimal(str(self.rng.uniform(45000, 55000)))
            
            if is_profitable:
                profit_pct = self.rng.uniform(0.01, 0.08)  # 1-8% profit
                if side == "LONG":
                    exit_price = entry_price * (1 + profit_pct)
                else:
                    exit_price = entry_price * (1 - profit_pct)
            else:
                loss_pct = self.rng.uniform(0.005, 0.03)  # 0.5-3% loss
                if side == "LONG":
                    exit_price = entry_price * (1 - loss_pct)
                else:
                    exit_price = entry_price * (1 + loss_pct)
            
            exit_price = Decimal(str(float(exit_price)))
            
            # Trade timing
            entry_time = base_time + timedelta(hours=i)
            hold_duration = timedelta(minutes=self.rng.uniform(15, 240))
            exit_time = entry_time + hold_duration
            
            trade_result = TradeResult(
                trade_id=f"trade-{i:04d}",
                symbol=self.config.primary_symbol,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                size=Decimal(str(self.rng.uniform(0.01, 0.1))),
                entry_time=entry_time,
                exit_time=exit_time,
            )
            results.append(trade_result)
        
        return results
    
    def generate_positions(self, market_snapshots: List[MarketSnapshot]) -> List[Position]:
        """Generate position states based on market data."""
        if not FP_TYPES_AVAILABLE or not market_snapshots:
            return []
        
        positions = []
        
        for i, snapshot in enumerate(market_snapshots[::10]):  # Every 10th snapshot
            if i % 3 == 0:  # Sometimes no position
                continue
            
            side = self.rng.choice(["LONG", "SHORT"])
            entry_price = snapshot.price * Decimal(str(self.rng.uniform(0.98, 1.02)))
            size = Decimal(str(self.rng.uniform(0.01, self.config.max_position_size)))
            
            position = Position(
                symbol=snapshot.symbol,
                side=side,
                size=size,
                entry_price=entry_price,
                current_price=snapshot.price,
            )
            positions.append(position)
        
        return positions
    
    def _should_close_position(self, position: Position, snapshot: MarketSnapshot) -> bool:
        """Determine if position should be closed based on simple rules."""
        if not position or position.symbol != snapshot.symbol:
            return False
        
        # Calculate P&L percentage
        if position.side == "LONG":
            pnl_pct = float((snapshot.price - position.entry_price) / position.entry_price)
        else:
            pnl_pct = float((position.entry_price - snapshot.price) / position.entry_price)
        
        # Close on stop loss or take profit
        if pnl_pct <= -self.config.stop_loss_pct:
            return True
        if pnl_pct >= self.config.take_profit_pct:
            return True
        
        # Random close 5% of the time
        return self.rng.random() < 0.05
    
    def _execute_trade_decision(self, current_position: Optional[Position], balance: Decimal, snapshot: MarketSnapshot) -> Tuple[Optional[Position], Decimal]:
        """Execute trading decision and return new position and balance."""
        if current_position:
            # Close current position
            if current_position.side == "LONG":
                pnl = (snapshot.price - current_position.entry_price) * current_position.size
            else:
                pnl = (current_position.entry_price - snapshot.price) * current_position.size
            
            balance += current_position.entry_price * current_position.size + pnl
            current_position = None
        
        # Decide whether to open new position
        if self.rng.random() < 0.3:  # 30% chance to open new position
            side = self.rng.choice(["LONG", "SHORT"])
            position_value = balance * Decimal(str(self.rng.uniform(0.1, self.config.max_position_size)))
            size = position_value / snapshot.price
            
            if balance >= position_value:
                balance -= position_value
                current_position = Position(
                    symbol=snapshot.symbol,
                    side=side,
                    size=size,
                    entry_price=snapshot.price,
                    current_price=snapshot.price,
                )
        
        return current_position, balance


# =============================================================================
# FP TRADING SIGNAL GENERATOR
# =============================================================================

class FPTradingSignalGenerator:
    """Generator for FP trading signals and orders."""
    
    def __init__(self, config: FPTestDataConfig = None):
        self.config = config or FPTestDataConfig()
        if self.config.random_seed:
            random.seed(self.config.random_seed)
        
        self.rng = np.random.default_rng(self.config.random_seed)
    
    def generate_trade_signals(self, market_data: List[MarketSnapshot]) -> List[Union[Long, Short, Hold, MarketMake]]:
        """Generate sequence of trade signals based on market data."""
        if not FP_TYPES_AVAILABLE:
            return []
        
        signals = []
        
        for i, snapshot in enumerate(market_data):
            signal = self._generate_signal_for_snapshot(snapshot, i, len(market_data))
            signals.append(signal)
        
        return signals
    
    def generate_orders(self, signals: List[Union[Long, Short, Hold, MarketMake]], market_data: List[MarketSnapshot]) -> List[Union[LimitOrder, MarketOrder]]:
        """Generate orders from trade signals."""
        if not FP_TYPES_AVAILABLE or len(signals) != len(market_data):
            return []
        
        orders = []
        
        for signal, snapshot in zip(signals, market_data):
            if isinstance(signal, (Long, Short)):
                # Create market order for directional signals
                side = "buy" if isinstance(signal, Long) else "sell"
                order = MarketOrder(
                    symbol=snapshot.symbol,
                    side=side,
                    size=signal.size,
                )
                orders.append(order)
            
            elif isinstance(signal, MarketMake):
                # Create limit orders for market making
                bid_order = LimitOrder(
                    symbol=snapshot.symbol,
                    side="buy",
                    price=signal.bid_price,
                    size=signal.bid_size,
                )
                ask_order = LimitOrder(
                    symbol=snapshot.symbol,
                    side="sell", 
                    price=signal.ask_price,
                    size=signal.ask_size,
                )
                orders.extend([bid_order, ask_order])
        
        return orders
    
    def generate_market_making_signals(self, market_data: List[MarketSnapshot], spread_pct: float = 0.002) -> List[MarketMake]:
        """Generate market making signals."""
        if not FP_TYPES_AVAILABLE:
            return []
        
        signals = []
        
        for snapshot in market_data:
            half_spread = float(snapshot.price) * spread_pct / 2
            
            signal = MarketMake(
                bid_price=float(snapshot.price) - half_spread,
                ask_price=float(snapshot.price) + half_spread,
                bid_size=self.rng.uniform(0.05, 0.2),
                ask_size=self.rng.uniform(0.05, 0.2),
            )
            signals.append(signal)
        
        return signals
    
    def _generate_signal_for_snapshot(self, snapshot: MarketSnapshot, index: int, total: int) -> Union[Long, Short, Hold, MarketMake]:
        """Generate appropriate signal for market snapshot."""
        # Simple signal generation based on position in series and randomness
        signal_type = self.rng.choice(["long", "short", "hold", "market_make"], p=[0.3, 0.3, 0.2, 0.2])
        
        if signal_type == "long":
            return Long(
                confidence=self.rng.uniform(0.5, 0.9),
                size=self.rng.uniform(0.1, 0.5),
                reason=f"Bullish signal at step {index}",
            )
        elif signal_type == "short":
            return Short(
                confidence=self.rng.uniform(0.5, 0.9),
                size=self.rng.uniform(0.1, 0.5),
                reason=f"Bearish signal at step {index}",
            )
        elif signal_type == "market_make":
            spread_pct = self.rng.uniform(0.001, 0.003)
            half_spread = float(snapshot.price) * spread_pct / 2
            
            return MarketMake(
                bid_price=float(snapshot.price) - half_spread,
                ask_price=float(snapshot.price) + half_spread,
                bid_size=self.rng.uniform(0.05, 0.2),
                ask_size=self.rng.uniform(0.05, 0.2),
            )
        else:  # hold
            return Hold(reason=f"No clear signal at step {index}")


# =============================================================================
# FP TEST SCENARIO GENERATOR
# =============================================================================

class FPTestScenarioGenerator:
    """Generator for complete FP test scenarios."""
    
    def __init__(self, config: FPTestDataConfig = None):
        self.config = config or FPTestDataConfig()
        self.market_generator = FPMarketDataGenerator(config)
        self.portfolio_generator = FPPortfolioDataGenerator(config)
        self.signal_generator = FPTradingSignalGenerator(config)
    
    def generate_complete_scenario(self) -> Dict[str, any]:
        """Generate complete test scenario with all data types."""
        # Generate market data
        market_snapshots = self.market_generator.generate_market_snapshots()
        ohlcv_data = self.market_generator.generate_ohlcv_series()
        
        # Generate portfolio data
        portfolio_timeline = self.portfolio_generator.generate_portfolio_timeline(market_snapshots)
        trade_results = self.portfolio_generator.generate_trade_results()
        positions = self.portfolio_generator.generate_positions(market_snapshots)
        
        # Generate trading signals
        trade_signals = self.signal_generator.generate_trade_signals(market_snapshots)
        orders = self.signal_generator.generate_orders(trade_signals, market_snapshots)
        market_making_signals = self.signal_generator.generate_market_making_signals(market_snapshots)
        
        return {
            # Market data
            "market_snapshots": market_snapshots,
            "ohlcv_data": ohlcv_data,
            
            # Portfolio data
            "portfolio_timeline": portfolio_timeline,
            "trade_results": trade_results,
            "positions": positions,
            
            # Trading data
            "trade_signals": trade_signals,
            "orders": orders,
            "market_making_signals": market_making_signals,
            
            # Metadata
            "config": self.config,
            "scenario_type": self.config.scenario_type,
            "data_points": len(market_snapshots),
        }
    
    def generate_multi_scenario_suite(self) -> Dict[str, Dict[str, any]]:
        """Generate multiple scenarios for comprehensive testing."""
        scenarios = {}
        
        scenario_configs = [
            ("default", {"scenario_type": "default"}),
            ("trending_bull", {"scenario_type": "trending", "trend_strength": 0.001}),
            ("trending_bear", {"scenario_type": "trending", "trend_strength": -0.001}),
            ("ranging", {"scenario_type": "ranging", "volatility": 0.01}),
            ("volatile", {"scenario_type": "volatile", "volatility": 0.04}),
        ]
        
        for scenario_name, config_updates in scenario_configs:
            # Create scenario-specific config
            scenario_config = FPTestDataConfig(**{
                **self.config.__dict__,
                **config_updates,
                "random_seed": (self.config.random_seed or 42) + hash(scenario_name) % 1000,
            })
            
            # Generate scenario
            scenario_generator = FPTestScenarioGenerator(scenario_config)
            scenarios[scenario_name] = scenario_generator.generate_complete_scenario()
        
        return scenarios


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "FPTestDataConfig",
    
    # Generators
    "FPMarketDataGenerator",
    "FPPortfolioDataGenerator", 
    "FPTradingSignalGenerator",
    "FPTestScenarioGenerator",
    
    # Availability flag
    "FP_TYPES_AVAILABLE",
]