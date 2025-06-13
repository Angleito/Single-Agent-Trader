# AI Trading Bot - API Reference

*Version: 1.0.0 | Updated: 2025-06-11*

This comprehensive API reference covers all public interfaces, configuration options, data models, and integration examples for the AI Trading Bot.

## Table of Contents

1. [Core API Classes](#core-api-classes)
2. [Configuration Reference](#configuration-reference)
3. [Data Models](#data-models)
4. [Integration Examples](#integration-examples)
5. [Extension Points](#extension-points)
6. [Health and Monitoring APIs](#health-and-monitoring-apis)
7. [Error Handling](#error-handling)
8. [Performance Considerations](#performance-considerations)

## Core API Classes

### 1. Trading Engine

The main orchestrator class that coordinates all trading activities.

```python
class TradingEngine:
    """
    Main trading engine that orchestrates all components.
    
    Manages the complete trading loop including:
    - Market data ingestion
    - Technical indicator calculation
    - LLM-based decision making
    - Risk management validation
    - Trade execution
    - Position tracking and monitoring
    """
    
    def __init__(
        self,
        symbol: str = "BTC-USD",
        interval: str = "1m",
        config_file: Optional[str] = None,
        dry_run: bool = True
    ):
        """
        Initialize the trading engine.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD", "ETH-USD")
            interval: Candle interval for analysis ("1m", "5m", "15m", "1h", "4h", "1d")
            config_file: Optional configuration file path
            dry_run: Whether to run in dry-run mode (no real trades)
        """
    
    async def run(self) -> None:
        """
        Main trading loop entry point.
        
        Orchestrates the complete trading process with error handling
        and graceful shutdown capabilities.
        
        Raises:
            RuntimeError: If critical components fail to initialize
            Exception: For unexpected errors during operation
        """
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dict containing:
            - trade_count: Total number of trades executed
            - successful_trades: Number of profitable trades
            - success_rate: Percentage of successful trades
            - total_pnl: Total profit/loss
            - current_position: Current position information
            - uptime: Engine uptime in seconds
        """
    
    def get_risk_status(self) -> Dict[str, Any]:
        """
        Get current risk management status.
        
        Returns:
            Dict containing:
            - daily_pnl: Today's profit/loss
            - daily_loss_used: Percentage of daily loss limit used
            - position_count: Number of open positions
            - risk_score: Current risk assessment
        """
```

#### Usage Example:

```python
import asyncio
from bot.main import TradingEngine

async def run_trading_bot():
    # Create trading engine
    engine = TradingEngine(
        symbol="BTC-USD",
        interval="5m",
        dry_run=True  # Start safely
    )
    
    # Run the trading loop
    await engine.run()

# Start the bot
asyncio.run(run_trading_bot())
```

### 2. Market Data Provider

Handles real-time and historical market data from Coinbase.

```python
class MarketDataProvider:
    """
    Provides real-time and historical market data from Coinbase.
    
    Features:
    - Real-time WebSocket data streaming
    - Historical OHLCV data fetching
    - Data caching and validation
    - Multiple timeframe support
    """
    
    def __init__(self, symbol: str, interval: str):
        """
        Initialize market data provider.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC-USD")
            interval: Data interval ("1m", "5m", "15m", "1h", "4h", "1d")
        """
    
    async def connect(self) -> bool:
        """
        Connect to market data sources.
        
        Returns:
            True if connection successful, False otherwise
        """
    
    def get_latest_ohlcv(self, limit: int = 100) -> List[OHLCV]:
        """
        Get latest OHLCV data points.
        
        Args:
            limit: Maximum number of data points to return
            
        Returns:
            List of OHLCV data points, most recent last
            
        Raises:
            ValueError: If limit is invalid
            ConnectionError: If not connected to data source
        """
    
    def to_dataframe(self, limit: int = 200) -> pd.DataFrame:
        """
        Convert OHLCV data to pandas DataFrame.
        
        Args:
            limit: Number of recent candles to include
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
    
    def get_data_status(self) -> Dict[str, Any]:
        """
        Get current data connection and cache status.
        
        Returns:
            Dict containing:
            - connected: Boolean connection status
            - last_update: Timestamp of last data update
            - cached_candles: Number of cached data points
            - data_quality: Quality assessment of recent data
        """
    
    def subscribe_to_updates(self, callback: Callable[[OHLCV], None]) -> None:
        """
        Subscribe to real-time data updates.
        
        Args:
            callback: Function to call with new data points
        """
```

#### Usage Example:

```python
from bot.data.market import MarketDataProvider
import asyncio

async def monitor_market_data():
    # Create provider
    provider = MarketDataProvider("BTC-USD", "1m")
    
    # Connect to data source
    await provider.connect()
    
    # Get latest data
    data = provider.get_latest_ohlcv(50)
    print(f"Latest price: ${data[-1].close}")
    
    # Convert to DataFrame for analysis
    df = provider.to_dataframe(100)
    print(f"Data shape: {df.shape}")

asyncio.run(monitor_market_data())
```

### 3. Indicator Calculator

Calculates technical indicators including VuManChu Cipher A & B.

```python
class IndicatorCalculator:
    """
    Calculate technical indicators for market analysis.
    
    Includes:
    - VuManChu Cipher A (trend analysis)
    - VuManChu Cipher B (momentum analysis)
    - Standard indicators (EMA, RSI, VWAP, etc.)
    """
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators for the given DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional indicator columns
            
        Raises:
            ValueError: If DataFrame format is invalid
            InsufficientDataError: If not enough data for calculations
        """
    
    def calculate_cipher_a(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate VuManChu Cipher A indicator.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with Cipher A values (-1, 0, 1 for sell, neutral, buy)
        """
    
    def calculate_cipher_b(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VuManChu Cipher B indicator.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with columns: wave, money_flow, momentum
        """
    
    def get_latest_state(self, df_with_indicators: pd.DataFrame) -> Dict[str, float]:
        """
        Get the latest indicator state.
        
        Args:
            df_with_indicators: DataFrame with calculated indicators
            
        Returns:
            Dict with latest values for all indicators
        """
    
    def validate_data_sufficiency(self, df: pd.DataFrame) -> bool:
        """
        Check if DataFrame has sufficient data for indicator calculations.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if sufficient data, False otherwise
        """
```

#### Usage Example:

```python
from bot.indicators.vumanchu import IndicatorCalculator
from bot.data.market import MarketDataProvider

async def analyze_indicators():
    # Get market data
    provider = MarketDataProvider("BTC-USD", "1m")
    await provider.connect()
    df = provider.to_dataframe(200)
    
    # Calculate indicators
    calc = IndicatorCalculator()
    df_with_indicators = calc.calculate_all(df)
    
    # Get latest state
    state = calc.get_latest_state(df_with_indicators)
    print(f"Cipher A: {state.get('cipher_a')}")
    print(f"Cipher B Wave: {state.get('cipher_b_wave')}")
    print(f"RSI: {state.get('rsi')}")

asyncio.run(analyze_indicators())
```

### 4. LLM Agent

AI-powered trading decision engine using LangChain.

```python
class LLMAgent:
    """
    LangChain-powered AI agent for trading decisions.
    
    Uses large language models to analyze market conditions
    and generate trading actions based on technical indicators
    and market state.
    """
    
    def __init__(
        self,
        model_provider: str = "openai",
        model_name: str = "gpt-4o",
        temperature: float = 0.1
    ):
        """
        Initialize LLM agent.
        
        Args:
            model_provider: LLM provider ("openai", "anthropic", "ollama")
            model_name: Specific model name
            temperature: Model temperature for response creativity
        """
    
    async def analyze_market(self, market_state: MarketState) -> TradeAction:
        """
        Analyze market conditions and generate trading decision.
        
        Args:
            market_state: Current market state with price and indicators
            
        Returns:
            TradeAction with decision (LONG/SHORT/CLOSE/HOLD) and parameters
            
        Raises:
            LLMError: If LLM request fails
            ValidationError: If response format is invalid
        """
    
    def is_available(self) -> bool:
        """
        Check if LLM agent is available and functional.
        
        Returns:
            True if agent is ready, False otherwise
        """
    
    def test_connection(self) -> bool:
        """
        Test connection to LLM provider.
        
        Returns:
            True if connection successful, False otherwise
        """
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current LLM agent status.
        
        Returns:
            Dict containing:
            - llm_available: Boolean availability status
            - model_provider: Current provider name
            - model_name: Current model name
            - request_count: Number of requests made
            - error_count: Number of failed requests
            - cache_hit_rate: Percentage of cached responses used
        """
    
    def update_prompt_template(self, template: str) -> None:
        """
        Update the prompt template used for market analysis.
        
        Args:
            template: New prompt template string
        """
```

#### Usage Example:

```python
from bot.strategy.llm_agent import LLMAgent
from bot.types import MarketState, IndicatorData

async def get_trading_decision():
    # Create LLM agent
    agent = LLMAgent(model_provider="openai", model_name="gpt-4o")
    
    # Check availability
    if not agent.is_available():
        print("LLM agent not available")
        return
    
    # Create market state (normally from market data)
    market_state = MarketState(
        symbol="BTC-USD",
        interval="1m",
        timestamp=datetime.utcnow(),
        current_price=Decimal("50000"),
        ohlcv_data=[],  # Recent OHLCV data
        indicators=IndicatorData(
            cipher_a=1,
            cipher_b_wave=0.5,
            rsi=65,
            ema_20=49800
        ),
        current_position=None
    )
    
    # Get trading decision
    action = await agent.analyze_market(market_state)
    print(f"Recommended action: {action.action}")
    print(f"Position size: {action.size_pct}%")
    print(f"Rationale: {action.rationale}")

asyncio.run(get_trading_decision())
```

### 5. Risk Manager

Comprehensive risk management and position sizing.

```python
class RiskManager:
    """
    Comprehensive risk management system.
    
    Features:
    - Position sizing calculations
    - Daily/weekly/monthly loss limits
    - Stop-loss and take-profit validation
    - Risk-adjusted position management
    """
    
    def __init__(self, position_manager: Optional[PositionManager] = None):
        """
        Initialize risk manager.
        
        Args:
            position_manager: Optional position manager for tracking
        """
    
    def evaluate_risk(
        self,
        trade_action: TradeAction,
        current_position: Position,
        current_price: Decimal
    ) -> Tuple[bool, TradeAction, str]:
        """
        Evaluate and potentially modify trade action based on risk rules.
        
        Args:
            trade_action: Proposed trade action
            current_position: Current position state
            current_price: Current market price
            
        Returns:
            Tuple of:
            - approved: Whether trade is approved
            - modified_action: Potentially modified trade action
            - reason: Explanation of decision
        """
    
    def calculate_position_size(
        self,
        action: str,
        account_balance: Decimal,
        current_price: Decimal,
        risk_percent: float
    ) -> Decimal:
        """
        Calculate appropriate position size based on risk parameters.
        
        Args:
            action: Trade action ("LONG" or "SHORT")
            account_balance: Available account balance
            current_price: Current asset price
            risk_percent: Risk percentage for this trade
            
        Returns:
            Calculated position size in base currency units
        """
    
    def check_daily_limits(self) -> Tuple[bool, float]:
        """
        Check if daily loss limits have been exceeded.
        
        Returns:
            Tuple of:
            - within_limits: Whether within daily limits
            - usage_percent: Percentage of daily limit used
        """
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive risk metrics.
        
        Returns:
            Dict containing:
            - daily_pnl: Today's profit/loss
            - daily_loss_limit: Daily loss limit
            - daily_usage_pct: Percentage of daily limit used
            - weekly_pnl: Week's profit/loss
            - monthly_pnl: Month's profit/loss
            - position_count: Current open positions
            - max_positions: Maximum allowed positions
            - account_balance: Current account balance
            - risk_score: Overall risk assessment (LOW/MEDIUM/HIGH/CRITICAL)
        """
    
    def update_daily_pnl(self, realized_pnl: Decimal, unrealized_pnl: Decimal) -> None:
        """
        Update daily P&L tracking.
        
        Args:
            realized_pnl: Realized profit/loss from closed positions
            unrealized_pnl: Unrealized profit/loss from open positions
        """
```

#### Usage Example:

```python
from bot.risk import RiskManager
from bot.types import TradeAction, Position
from decimal import Decimal

def manage_trade_risk():
    # Create risk manager
    risk_manager = RiskManager()
    
    # Proposed trade action
    trade_action = TradeAction(
        action="LONG",
        size_pct=25,  # 25% of account
        take_profit_pct=4.0,
        stop_loss_pct=2.0,
        rationale="Strong bullish signals"
    )
    
    # Current position (empty)
    current_position = Position(
        symbol="BTC-USD",
        side="FLAT",
        size=Decimal('0'),
        timestamp=datetime.utcnow()
    )
    
    # Evaluate risk
    approved, modified_action, reason = risk_manager.evaluate_risk(
        trade_action, current_position, Decimal("50000")
    )
    
    if approved:
        print(f"Trade approved: {modified_action.action} {modified_action.size_pct}%")
    else:
        print(f"Trade rejected: {reason}")
    
    # Get risk metrics
    metrics = risk_manager.get_risk_metrics()
    print(f"Daily P&L usage: {metrics['daily_usage_pct']:.1f}%")
    print(f"Risk score: {metrics['risk_score']}")
```

### 6. Exchange Client

Coinbase Advanced Trade API integration.

```python
class CoinbaseClient:
    """
    Coinbase Advanced Trade API client.
    
    Handles:
    - Order placement and management
    - Account balance queries
    - Position tracking
    - Market data requests
    """
    
    def __init__(self):
        """Initialize Coinbase client with settings from configuration."""
    
    async def connect(self) -> bool:
        """
        Connect to Coinbase API and validate credentials.
        
        Returns:
            True if connection successful, False otherwise
        """
    
    async def execute_trade_action(
        self,
        trade_action: TradeAction,
        symbol: str,
        current_price: Decimal
    ) -> Optional[Order]:
        """
        Execute a trade action on Coinbase.
        
        Args:
            trade_action: Trade action to execute
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Order object if successful, None if failed
            
        Raises:
            ExchangeError: If order placement fails
            InsufficientFundsError: If account has insufficient funds
        """
    
    async def get_account_balance(self) -> Decimal:
        """
        Get current account balance.
        
        Returns:
            Available balance in USD
            
        Raises:
            ExchangeError: If balance request fails
        """
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get current open positions.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of current positions
        """
    
    async def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """
        Cancel all open orders for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with cancellation results
        """
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current connection status.
        
        Returns:
            Dict containing:
            - connected: Boolean connection status
            - sandbox: Whether using sandbox environment
            - rate_limit_remaining: Remaining API rate limit
            - last_request_time: Timestamp of last API request
        """
    
    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        size: Decimal,
        price: Decimal
    ) -> Optional[Order]:
        """
        Place a limit order.
        
        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            size: Order size
            price: Limit price
            
        Returns:
            Order object if successful, None if failed
        """
    
    async def place_market_order(
        self,
        symbol: str,
        side: str,
        size: Decimal
    ) -> Optional[Order]:
        """
        Place a market order.
        
        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell") 
            size: Order size
            
        Returns:
            Order object if successful, None if failed
        """
```

#### Usage Example:

```python
from bot.exchange.coinbase import CoinbaseClient
from bot.types import TradeAction
from decimal import Decimal

async def execute_trade():
    # Create client
    client = CoinbaseClient()
    
    # Connect to exchange
    connected = await client.connect()
    if not connected:
        print("Failed to connect to Coinbase")
        return
    
    # Check account balance
    balance = await client.get_account_balance()
    print(f"Account balance: ${balance}")
    
    # Create trade action
    trade_action = TradeAction(
        action="LONG",
        size_pct=10,
        take_profit_pct=3.0,
        stop_loss_pct=1.5,
        rationale="Test trade"
    )
    
    # Execute trade
    order = await client.execute_trade_action(
        trade_action, "BTC-USD", Decimal("50000")
    )
    
    if order:
        print(f"Order placed: {order.id}")
    else:
        print("Order failed")

asyncio.run(execute_trade())
```

## Configuration Reference

### 1. Environment Variables

#### System Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SYSTEM__ENVIRONMENT` | str | development | Environment type (development, staging, production) |
| `SYSTEM__DRY_RUN` | bool | true | Enable paper trading mode |
| `SYSTEM__LOG_LEVEL` | str | INFO | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `SYSTEM__LOG_FILE_PATH` | str | logs/bot.log | Log file path |
| `SYSTEM__UPDATE_FREQUENCY_SECONDS` | float | 60.0 | Main loop update frequency |
| `SYSTEM__ENABLE_MONITORING` | bool | true | Enable health monitoring |
| `SYSTEM__ALERT_WEBHOOK_URL` | str | None | Webhook URL for alerts |
| `SYSTEM__ALERT_EMAIL` | str | None | Email address for alerts |

#### Trading Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TRADING__SYMBOL` | str | BTC-USD | Trading pair symbol |
| `TRADING__INTERVAL` | str | 1m | Chart timeframe |
| `TRADING__LEVERAGE` | int | 5 | Trading leverage (1-20) |
| `TRADING__MAX_SIZE_PCT` | float | 20.0 | Maximum position size (% of equity) |
| `TRADING__ORDER_TIMEOUT_SECONDS` | int | 30 | Order timeout in seconds |
| `TRADING__SLIPPAGE_TOLERANCE_PCT` | float | 0.1 | Maximum slippage percentage |
| `TRADING__MIN_PROFIT_PCT` | float | 0.5 | Minimum profit target |

#### Risk Management

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RISK__MAX_DAILY_LOSS_PCT` | float | 5.0 | Maximum daily loss percentage |
| `RISK__MAX_WEEKLY_LOSS_PCT` | float | 15.0 | Maximum weekly loss percentage |
| `RISK__MAX_MONTHLY_LOSS_PCT` | float | 30.0 | Maximum monthly loss percentage |
| `RISK__MAX_CONCURRENT_TRADES` | int | 3 | Maximum concurrent positions |
| `RISK__DEFAULT_STOP_LOSS_PCT` | float | 2.0 | Default stop loss percentage |
| `RISK__DEFAULT_TAKE_PROFIT_PCT` | float | 4.0 | Default take profit percentage |
| `RISK__MIN_ACCOUNT_BALANCE` | decimal | 100 | Minimum account balance to trade |

#### LLM Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LLM__PROVIDER` | str | openai | LLM provider (openai, anthropic, ollama) |
| `LLM__MODEL_NAME` | str | gpt-4o | Model name |
| `LLM__TEMPERATURE` | float | 0.1 | Response creativity (0.0-2.0) |
| `LLM__MAX_TOKENS` | int | 1000 | Maximum tokens in response |
| `LLM__REQUEST_TIMEOUT` | int | 30 | Request timeout in seconds |
| `LLM__MAX_RETRIES` | int | 3 | Maximum retry attempts |
| `LLM__OPENAI_API_KEY` | str | None | OpenAI API key |
| `LLM__ANTHROPIC_API_KEY` | str | None | Anthropic API key |

#### Exchange Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EXCHANGE__CB_API_KEY` | str | None | Coinbase API key |
| `EXCHANGE__CB_API_SECRET` | str | None | Coinbase API secret |
| `EXCHANGE__CB_PASSPHRASE` | str | None | Coinbase passphrase |
| `EXCHANGE__CB_SANDBOX` | bool | true | Use sandbox environment |
| `EXCHANGE__API_TIMEOUT` | int | 10 | API request timeout |
| `EXCHANGE__RATE_LIMIT_REQUESTS` | int | 10 | Requests per minute limit |

### 2. Configuration Classes

#### Settings Class

```python
from bot.config import Settings, TradingProfile, Environment

# Create settings with defaults
settings = Settings()

# Create with specific profile
settings = Settings(profile=TradingProfile.CONSERVATIVE)

# Create with environment file
settings = Settings(_env_file=".env.production")

# Apply profile after creation
conservative_settings = settings.apply_profile(TradingProfile.CONSERVATIVE)

# Export for different environment
production_config = settings.export_for_environment(Environment.PRODUCTION)

# Validate trading environment
warnings = settings.validate_trading_environment()

# Save to file
settings.save_to_file("config/current_settings.json")

# Load from file
loaded_settings = Settings.load_from_file("config/current_settings.json")
```

#### Configuration Utilities

```python
from bot.config_utils import (
    setup_configuration,
    validate_configuration,
    ConfigManager,
    HealthMonitor
)

# Setup with validation
settings = setup_configuration(profile=TradingProfile.MODERATE)

# Comprehensive validation
result = validate_configuration(settings)
if result.is_valid:
    print("Configuration is valid")
else:
    print(f"Validation failed: {result.error}")

# Configuration management
manager = ConfigManager()

# Switch profiles with backup
new_settings = manager.switch_profile(
    settings, 
    TradingProfile.AGGRESSIVE,
    save_current=True
)

# Create backup
backup_path = manager.create_config_backup(settings)

# Export configuration
json_export = manager.export_configuration(settings, "json")
env_export = manager.export_configuration(settings, "env")
```

## Data Models

### 1. Core Trading Types

#### OHLCV Data

```python
from bot.types import OHLCV
from decimal import Decimal
from datetime import datetime

ohlcv = OHLCV(
    timestamp=datetime.utcnow(),
    open=Decimal("50000.00"),
    high=Decimal("50500.00"),
    low=Decimal("49800.00"),
    close=Decimal("50200.00"),
    volume=Decimal("125.5")
)

# Properties
price_change = ohlcv.price_change()  # Close - Open
price_change_pct = ohlcv.price_change_percent()  # % change
```

#### Trade Action

```python
from bot.types import TradeAction

action = TradeAction(
    action="LONG",  # LONG, SHORT, CLOSE, HOLD
    size_pct=15,    # Position size as % of account
    take_profit_pct=3.0,  # Take profit %
    stop_loss_pct=1.5,    # Stop loss %
    rationale="Strong bullish indicators on 5m chart"
)

# Validation
is_valid = action.is_valid()
validation_errors = action.validate()
```

#### Position

```python
from bot.types import Position
from decimal import Decimal

position = Position(
    symbol="BTC-USD",
    side="LONG",  # LONG, SHORT, FLAT
    size=Decimal("0.1"),  # Position size
    entry_price=Decimal("50000"),
    timestamp=datetime.utcnow(),
    unrealized_pnl=Decimal("150.00")
)

# Calculate P&L
current_price = Decimal("51500")
pnl = position.calculate_pnl(current_price)
pnl_percent = position.calculate_pnl_percent(current_price)
```

#### Market State

```python
from bot.types import MarketState, IndicatorData

market_state = MarketState(
    symbol="BTC-USD",
    interval="1m",
    timestamp=datetime.utcnow(),
    current_price=Decimal("50000"),
    ohlcv_data=[],  # List of recent OHLCV data
    indicators=IndicatorData(
        cipher_a=1,
        cipher_b_wave=0.5,
        rsi=65,
        ema_20=49800
    ),
    current_position=position
)
```

#### Order

```python
from bot.types import Order, OrderStatus

order = Order(
    id="order_123",
    symbol="BTC-USD",
    side="buy",
    size=Decimal("0.1"),
    price=Decimal("50000"),
    status=OrderStatus.FILLED,
    timestamp=datetime.utcnow()
)

# Order status checks
is_filled = order.is_filled()
is_active = order.is_active()
is_cancelled = order.is_cancelled()
```

### 2. Indicator Data Models

#### VuManChu Cipher Indicators

```python
from bot.types import CipherAResult, CipherBResult

# Cipher A Result
cipher_a = CipherAResult(
    signal=1,        # -1 (sell), 0 (neutral), 1 (buy)
    strength=0.8,    # Signal strength 0-1
    trend="bullish", # Trend direction
    confidence=0.75  # Confidence level 0-1
)

# Cipher B Result
cipher_b = CipherBResult(
    wave=0.5,        # Wave value
    money_flow=0.3,  # Money flow indicator
    momentum=0.6,    # Momentum value
    divergence=False # Divergence detected
)
```

#### Technical Indicators

```python
from bot.types import TechnicalIndicators

indicators = TechnicalIndicators(
    rsi=65.5,
    macd=0.15,
    macd_signal=0.12,
    ema_12=49950.0,
    ema_26=49800.0,
    bollinger_upper=51000.0,
    bollinger_lower=49000.0,
    volume_sma=150.0,
    atr=500.0
)

# Indicator analysis
is_overbought = indicators.is_rsi_overbought()  # RSI > 70
is_oversold = indicators.is_rsi_oversold()      # RSI < 30
macd_bullish = indicators.is_macd_bullish()     # MACD > Signal
```

### 3. Configuration Models

#### Trading Settings

```python
from bot.config import TradingSettings

trading_config = TradingSettings(
    symbol="ETH-USD",
    interval="5m",
    leverage=3,
    max_size_pct=25.0,
    order_timeout_seconds=45,
    slippage_tolerance_pct=0.15,
    min_profit_pct=0.75
)

# Validation
trading_config.validate_interval("1h")  # Validates interval format
```

#### Risk Settings

```python
from bot.config import RiskSettings
from decimal import Decimal

risk_config = RiskSettings(
    max_daily_loss_pct=3.0,
    max_weekly_loss_pct=10.0,
    max_monthly_loss_pct=20.0,
    max_concurrent_trades=2,
    default_stop_loss_pct=1.5,
    default_take_profit_pct=3.0,
    min_account_balance=Decimal("500"),
    emergency_stop_loss_pct=8.0
)
```

### 4. Health and Monitoring Models

#### Health Status

```python
from bot.health import HealthStatus, ComponentHealth

health = HealthStatus(
    status="healthy",  # healthy, warning, critical, error
    timestamp=datetime.utcnow(),
    uptime_seconds=3600,
    components={
        "system": ComponentHealth(
            status="healthy",
            memory_usage_mb=512.5,
            cpu_usage_percent=25.3,
            disk_usage_percent=45.2
        ),
        "apis": ComponentHealth(
            status="healthy",
            coinbase_status="connected",
            llm_status="available",
            response_time_ms=150
        )
    },
    metrics={
        "trades_today": 5,
        "success_rate": 0.8,
        "daily_pnl": 125.50
    }
)
```

## Integration Examples

### 1. Custom Strategy Implementation

```python
"""Custom trading strategy example."""

from bot.strategy.core import BaseStrategy
from bot.types import MarketState, TradeAction
from typing import Optional

class CustomMACDStrategy(BaseStrategy):
    """
    Custom MACD-based trading strategy.
    
    Enters long when MACD crosses above signal line
    and RSI is not overbought.
    """
    
    def __init__(self, rsi_threshold: float = 70):
        super().__init__()
        self.rsi_threshold = rsi_threshold
        self.name = "Custom MACD Strategy"
    
    def analyze(self, market_state: MarketState) -> TradeAction:
        """
        Analyze market and generate trading decision.
        
        Args:
            market_state: Current market conditions
            
        Returns:
            TradeAction with trading decision
        """
        indicators = market_state.indicators
        
        # Check for MACD bullish crossover
        macd_bullish = (
            indicators.macd > indicators.macd_signal and
            indicators.macd > 0
        )
        
        # Check RSI not overbought
        rsi_ok = indicators.rsi < self.rsi_threshold
        
        # Current position
        position = market_state.current_position
        
        if position and position.side != "FLAT":
            # Check exit conditions
            if position.side == "LONG":
                # Exit long if MACD turns bearish or RSI overbought
                if indicators.macd < indicators.macd_signal or indicators.rsi > 80:
                    return TradeAction(
                        action="CLOSE",
                        size_pct=0,
                        take_profit_pct=0,
                        stop_loss_pct=0,
                        rationale="MACD bearish or RSI overbought - closing long"
                    )
        else:
            # Check entry conditions
            if macd_bullish and rsi_ok:
                return TradeAction(
                    action="LONG",
                    size_pct=15,  # 15% position size
                    take_profit_pct=3.0,
                    stop_loss_pct=1.5,
                    rationale=f"MACD bullish crossover, RSI={indicators.rsi:.1f}"
                )
        
        # Default to hold
        return TradeAction(
            action="HOLD",
            size_pct=0,
            take_profit_pct=0,
            stop_loss_pct=0,
            rationale="No clear trading signals"
        )

# Usage
strategy = CustomMACDStrategy(rsi_threshold=75)
action = strategy.analyze(market_state)
```

### 2. Custom Indicator Implementation

```python
"""Custom indicator implementation example."""

import pandas as pd
import numpy as np
from bot.indicators.vumanchu import IndicatorCalculator

class CustomIndicatorCalculator(IndicatorCalculator):
    """Extended indicator calculator with custom indicators."""
    
    def calculate_custom_momentum(
        self, 
        df: pd.DataFrame, 
        fast_period: int = 12,
        slow_period: int = 26
    ) -> pd.Series:
        """
        Calculate custom momentum indicator.
        
        Args:
            df: DataFrame with OHLCV data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            
        Returns:
            Series with momentum values
        """
        # Calculate EMAs
        ema_fast = df['close'].ewm(span=fast_period).mean()
        ema_slow = df['close'].ewm(span=slow_period).mean()
        
        # Calculate momentum
        momentum = (ema_fast - ema_slow) / ema_slow * 100
        
        return momentum
    
    def calculate_volume_profile(
        self, 
        df: pd.DataFrame, 
        period: int = 20
    ) -> pd.DataFrame:
        """
        Calculate volume profile indicator.
        
        Args:
            df: DataFrame with OHLCV data
            period: Lookback period
            
        Returns:
            DataFrame with volume profile data
        """
        volume_sma = df['volume'].rolling(window=period).mean()
        volume_ratio = df['volume'] / volume_sma
        
        # Price-volume relationship
        price_change = df['close'].pct_change()
        volume_price_correlation = price_change.rolling(window=period).corr(
            df['volume'].pct_change()
        )
        
        return pd.DataFrame({
            'volume_ratio': volume_ratio,
            'volume_sma': volume_sma,
            'vp_correlation': volume_price_correlation
        })
    
    def calculate_all_extended(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators including custom ones."""
        # Calculate standard indicators
        df_with_indicators = self.calculate_all(df)
        
        # Add custom indicators
        df_with_indicators['custom_momentum'] = self.calculate_custom_momentum(df)
        
        volume_profile = self.calculate_volume_profile(df)
        for col in volume_profile.columns:
            df_with_indicators[col] = volume_profile[col]
        
        return df_with_indicators

# Usage
calc = CustomIndicatorCalculator()
df_with_custom = calc.calculate_all_extended(df)
```

### 3. Event-Driven Trading Bot

```python
"""Event-driven trading bot implementation."""

import asyncio
from typing import Dict, Any, Callable
from bot.main import TradingEngine
from bot.types import TradeAction, MarketState

class EventDrivenTradingBot:
    """
    Event-driven trading bot with custom event handlers.
    
    Supports custom event handling for:
    - Trade executions
    - Market condition changes
    - Risk limit breaches
    - System health events
    """
    
    def __init__(self):
        self.engine = TradingEngine()
        self.event_handlers: Dict[str, list] = {
            'trade_executed': [],
            'position_opened': [],
            'position_closed': [],
            'risk_limit_hit': [],
            'market_condition_change': [],
            'system_health_change': []
        }
    
    def on(self, event: str, handler: Callable):
        """
        Register event handler.
        
        Args:
            event: Event name
            handler: Callable event handler
        """
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(handler)
    
    def emit(self, event: str, data: Any):
        """
        Emit event to all registered handlers.
        
        Args:
            event: Event name
            data: Event data
        """
        handlers = self.event_handlers.get(event, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                print(f"Error in event handler: {e}")
    
    async def run_with_events(self):
        """Run trading bot with event emission."""
        # Override engine methods to emit events
        original_execute_trade = self.engine._execute_trade
        
        async def execute_trade_with_events(trade_action, current_price):
            # Execute original trade
            result = await original_execute_trade(trade_action, current_price)
            
            # Emit events
            self.emit('trade_executed', {
                'action': trade_action,
                'price': current_price,
                'result': result
            })
            
            if trade_action.action in ['LONG', 'SHORT']:
                self.emit('position_opened', {
                    'side': trade_action.action,
                    'size': trade_action.size_pct,
                    'price': current_price
                })
            elif trade_action.action == 'CLOSE':
                self.emit('position_closed', {
                    'price': current_price
                })
            
            return result
        
        # Replace method
        self.engine._execute_trade = execute_trade_with_events
        
        # Run engine
        await self.engine.run()

# Event handlers
def on_trade_executed(data):
    """Handle trade execution events."""
    action = data['action']
    price = data['price']
    print(f"🔥 Trade executed: {action.action} at ${price}")

def on_position_opened(data):
    """Handle position opening events."""
    side = data['side']
    size = data['size']
    print(f"📈 Position opened: {side} {size}%")

def on_risk_limit_hit(data):
    """Handle risk limit events."""
    limit_type = data['limit_type']
    current_value = data['current_value']
    limit_value = data['limit_value']
    print(f"⚠️ Risk limit hit: {limit_type} {current_value}/{limit_value}")

# Usage
bot = EventDrivenTradingBot()

# Register event handlers
bot.on('trade_executed', on_trade_executed)
bot.on('position_opened', on_position_opened)
bot.on('risk_limit_hit', on_risk_limit_hit)

# Run bot
asyncio.run(bot.run_with_events())
```

### 4. Multi-Symbol Portfolio Bot

```python
"""Multi-symbol portfolio trading bot."""

import asyncio
from typing import Dict, List
from bot.main import TradingEngine
from bot.types import TradeAction
from decimal import Decimal

class PortfolioTradingBot:
    """
    Multi-symbol portfolio trading bot.
    
    Manages trading across multiple symbols with:
    - Portfolio-level risk management
    - Position correlation analysis
    - Dynamic allocation adjustment
    """
    
    def __init__(self, symbols: List[str], base_allocation: Dict[str, float]):
        """
        Initialize portfolio bot.
        
        Args:
            symbols: List of trading symbols
            base_allocation: Base allocation percentages per symbol
        """
        self.symbols = symbols
        self.base_allocation = base_allocation
        self.engines: Dict[str, TradingEngine] = {}
        self.portfolio_balance = Decimal('0')
        
        # Initialize engines for each symbol
        for symbol in symbols:
            self.engines[symbol] = TradingEngine(
                symbol=symbol,
                dry_run=True  # Start safely
            )
    
    async def initialize_portfolio(self):
        """Initialize all trading engines."""
        for symbol, engine in self.engines.items():
            await engine._initialize_components()
            print(f"✅ Initialized {symbol} trading engine")
    
    def calculate_portfolio_risk(self) -> Dict[str, float]:
        """
        Calculate portfolio-level risk metrics.
        
        Returns:
            Dict with portfolio risk metrics
        """
        total_exposure = Decimal('0')
        symbol_exposures = {}
        
        for symbol, engine in self.engines.items():
            position = engine.current_position
            if position.side != "FLAT":
                exposure = abs(position.size * position.entry_price)
                symbol_exposures[symbol] = float(exposure)
                total_exposure += exposure
        
        return {
            'total_exposure': float(total_exposure),
            'symbol_exposures': symbol_exposures,
            'portfolio_utilization': float(total_exposure / max(self.portfolio_balance, 1))
        }
    
    def adjust_position_sizes(self, base_action: TradeAction, symbol: str) -> TradeAction:
        """
        Adjust position size based on portfolio constraints.
        
        Args:
            base_action: Original trade action
            symbol: Trading symbol
            
        Returns:
            Adjusted trade action
        """
        if base_action.action == "HOLD":
            return base_action
        
        # Get current portfolio risk
        portfolio_risk = self.calculate_portfolio_risk()
        
        # Calculate maximum allowed position size for this symbol
        max_allocation = self.base_allocation.get(symbol, 0.2)  # Default 20%
        current_utilization = portfolio_risk['portfolio_utilization']
        
        # Reduce size if portfolio is highly utilized
        if current_utilization > 0.8:  # 80% utilization
            adjustment_factor = 0.5  # Reduce by 50%
        elif current_utilization > 0.6:  # 60% utilization
            adjustment_factor = 0.75  # Reduce by 25%
        else:
            adjustment_factor = 1.0  # No adjustment
        
        adjusted_size = min(
            base_action.size_pct * adjustment_factor,
            max_allocation * 100  # Convert to percentage
        )
        
        return TradeAction(
            action=base_action.action,
            size_pct=adjusted_size,
            take_profit_pct=base_action.take_profit_pct,
            stop_loss_pct=base_action.stop_loss_pct,
            rationale=f"Portfolio-adjusted: {base_action.rationale}"
        )
    
    async def run_portfolio_trading(self):
        """Run portfolio trading across all symbols."""
        await self.initialize_portfolio()
        
        print(f"🎯 Starting portfolio trading with {len(self.symbols)} symbols")
        
        # Create tasks for all symbols
        tasks = []
        for symbol in self.symbols:
            task = asyncio.create_task(
                self.run_symbol_trading(symbol),
                name=f"trading_{symbol}"
            )
            tasks.append(task)
        
        # Run all trading engines concurrently
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            print(f"Portfolio trading error: {e}")
            # Cancel all tasks
            for task in tasks:
                task.cancel()
    
    async def run_symbol_trading(self, symbol: str):
        """Run trading for a specific symbol with portfolio constraints."""
        engine = self.engines[symbol]
        
        while True:
            try:
                # Get market analysis (simplified - normally would use full engine logic)
                # market_state = await engine.get_market_state()
                # base_action = await engine.llm_agent.analyze_market(market_state)
                
                # For demo, create dummy action
                base_action = TradeAction(
                    action="HOLD",
                    size_pct=15,
                    take_profit_pct=3.0,
                    stop_loss_pct=1.5,
                    rationale="Demo action"
                )
                
                # Adjust for portfolio constraints
                adjusted_action = self.adjust_position_sizes(base_action, symbol)
                
                if adjusted_action.action != "HOLD":
                    print(f"📊 {symbol}: {adjusted_action.action} {adjusted_action.size_pct}%")
                
                # Sleep before next iteration
                await asyncio.sleep(60)  # 1 minute intervals
                
            except Exception as e:
                print(f"Error trading {symbol}: {e}")
                await asyncio.sleep(30)  # Wait before retry

# Usage
portfolio_bot = PortfolioTradingBot(
    symbols=["BTC-USD", "ETH-USD", "SOL-USD"],
    base_allocation={
        "BTC-USD": 0.4,   # 40% allocation
        "ETH-USD": 0.35,  # 35% allocation
        "SOL-USD": 0.25   # 25% allocation
    }
)

# Run portfolio trading
asyncio.run(portfolio_bot.run_portfolio_trading())
```

## Extension Points

### 1. Custom Indicators

The bot provides extension points for adding custom technical indicators:

```python
"""Custom indicator extension example."""

from bot.indicators.vumanchu import IndicatorCalculator
import pandas as pd
import numpy as np

class ExtendedIndicatorCalculator(IndicatorCalculator):
    """Extended calculator with custom indicators."""
    
    def calculate_custom_rsi_divergence(
        self, 
        df: pd.DataFrame, 
        rsi_period: int = 14,
        lookback: int = 20
    ) -> pd.Series:
        """
        Detect RSI divergence patterns.
        
        Args:
            df: OHLCV DataFrame
            rsi_period: RSI calculation period
            lookback: Lookback period for divergence detection
            
        Returns:
            Series with divergence signals (-1, 0, 1)
        """
        # Calculate RSI
        rsi = self.calculate_rsi(df['close'], rsi_period)
        
        # Find price peaks and troughs
        price_peaks = df['high'].rolling(window=lookback, center=True).max() == df['high']
        price_troughs = df['low'].rolling(window=lookback, center=True).min() == df['low']
        
        # Find RSI peaks and troughs
        rsi_peaks = rsi.rolling(window=lookback, center=True).max() == rsi
        rsi_troughs = rsi.rolling(window=lookback, center=True).min() == rsi
        
        # Detect divergences
        divergence = pd.Series(0, index=df.index)
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        for i in range(lookback, len(df)):
            if price_troughs.iloc[i]:
                # Look for previous trough
                prev_troughs = price_troughs.iloc[i-lookback:i]
                if prev_troughs.any():
                    prev_idx = prev_troughs.idxmax()
                    if (df['low'].iloc[i] < df['low'].loc[prev_idx] and
                        rsi.iloc[i] > rsi.loc[prev_idx]):
                        divergence.iloc[i] = 1  # Bullish divergence
        
        return divergence
    
    def calculate_market_structure(
        self, 
        df: pd.DataFrame, 
        swing_period: int = 10
    ) -> pd.DataFrame:
        """
        Analyze market structure (higher highs, lower lows, etc.).
        
        Args:
            df: OHLCV DataFrame
            swing_period: Period for swing detection
            
        Returns:
            DataFrame with market structure analysis
        """
        # Detect swing highs and lows
        swing_highs = df['high'].rolling(window=swing_period*2+1, center=True).max() == df['high']
        swing_lows = df['low'].rolling(window=swing_period*2+1, center=True).min() == df['low']
        
        # Classify market structure
        structure = pd.DataFrame(index=df.index)
        structure['swing_high'] = swing_highs
        structure['swing_low'] = swing_lows
        structure['trend'] = 'sideways'
        
        # Determine trend based on swing points
        high_values = df.loc[swing_highs, 'high']
        low_values = df.loc[swing_lows, 'low']
        
        # Simple trend classification
        if len(high_values) >= 2 and len(low_values) >= 2:
            recent_highs = high_values.tail(2)
            recent_lows = low_values.tail(2)
            
            if recent_highs.iloc[-1] > recent_highs.iloc[-2] and recent_lows.iloc[-1] > recent_lows.iloc[-2]:
                structure['trend'] = 'uptrend'
            elif recent_highs.iloc[-1] < recent_highs.iloc[-2] and recent_lows.iloc[-1] < recent_lows.iloc[-2]:
                structure['trend'] = 'downtrend'
        
        return structure

# Register custom calculator
from bot.main import TradingEngine

class CustomTradingEngine(TradingEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace indicator calculator with extended version
        self.indicator_calc = ExtendedIndicatorCalculator()
```

### 2. Custom LLM Providers

Add support for custom LLM providers:

```python
"""Custom LLM provider implementation."""

from bot.strategy.llm_agent import LLMAgent
from bot.types import MarketState, TradeAction
import httpx
import json

class CustomLLMProvider:
    """Custom LLM provider implementation."""
    
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.client = httpx.AsyncClient()
    
    async def generate_response(self, prompt: str) -> str:
        """Generate response from custom LLM."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        response = await self.client.post(
            self.api_url,
            headers=headers,
            json=payload
        )
        
        response.raise_for_status()
        result = response.json()
        return result.get("text", "")

class CustomLLMAgent(LLMAgent):
    """LLM agent with custom provider."""
    
    def __init__(self, custom_provider: CustomLLMProvider):
        self.provider = custom_provider
        self.model_provider = "custom"
        self.model_name = "custom-model"
    
    async def analyze_market(self, market_state: MarketState) -> TradeAction:
        """Analyze market using custom LLM."""
        # Build prompt
        prompt = self._build_market_prompt(market_state)
        
        # Get LLM response
        response = await self.provider.generate_response(prompt)
        
        # Parse response to TradeAction
        action = self._parse_response(response)
        
        return action
    
    def _build_market_prompt(self, market_state: MarketState) -> str:
        """Build trading prompt for custom LLM."""
        return f"""
        Analyze the following market data and provide a trading decision:
        
        Symbol: {market_state.symbol}
        Current Price: ${market_state.current_price}
        RSI: {market_state.indicators.rsi}
        Cipher A: {market_state.indicators.cipher_a}
        
        Respond with JSON: {{"action": "LONG|SHORT|CLOSE|HOLD", "size_pct": 0-30, "rationale": "reason"}}
        """
    
    def _parse_response(self, response: str) -> TradeAction:
        """Parse LLM response to TradeAction."""
        try:
            data = json.loads(response)
            return TradeAction(
                action=data["action"],
                size_pct=data.get("size_pct", 10),
                take_profit_pct=data.get("take_profit_pct", 3.0),
                stop_loss_pct=data.get("stop_loss_pct", 1.5),
                rationale=data.get("rationale", "Custom LLM decision")
            )
        except Exception as e:
            # Fallback to HOLD
            return TradeAction(
                action="HOLD",
                size_pct=0,
                take_profit_pct=0,
                stop_loss_pct=0,
                rationale=f"Failed to parse LLM response: {e}"
            )

# Usage
custom_provider = CustomLLMProvider(
    api_url="https://api.custom-llm.com/v1/generate",
    api_key="your-custom-api-key"
)

custom_agent = CustomLLMAgent(custom_provider)
```

### 3. Custom Risk Management

Implement custom risk management strategies:

```python
"""Custom risk management implementation."""

from bot.risk import RiskManager
from bot.types import TradeAction, Position
from decimal import Decimal
from typing import Tuple
import numpy as np

class AdvancedRiskManager(RiskManager):
    """Advanced risk management with custom strategies."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.volatility_window = 20
        self.correlation_threshold = 0.7
        
    def calculate_position_size_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        account_balance: Decimal
    ) -> Decimal:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount  
            account_balance: Current account balance
            
        Returns:
            Optimal position size based on Kelly Criterion
        """
        if avg_loss <= 0:
            return Decimal('0')
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Cap Kelly fraction at 25% for safety
        kelly_fraction = min(kelly_fraction, 0.25)
        kelly_fraction = max(kelly_fraction, 0)  # No negative sizing
        
        return account_balance * Decimal(str(kelly_fraction))
    
    def calculate_volatility_adjusted_size(
        self,
        base_size: Decimal,
        current_volatility: float,
        baseline_volatility: float = 0.02
    ) -> Decimal:
        """
        Adjust position size based on market volatility.
        
        Args:
            base_size: Base position size
            current_volatility: Current market volatility
            baseline_volatility: Baseline volatility for comparison
            
        Returns:
            Volatility-adjusted position size
        """
        volatility_ratio = current_volatility / baseline_volatility
        
        # Reduce size as volatility increases
        if volatility_ratio > 2.0:
            adjustment = 0.5  # Halve position size
        elif volatility_ratio > 1.5:
            adjustment = 0.75  # Reduce by 25%
        elif volatility_ratio < 0.5:
            adjustment = 1.25  # Increase by 25%
        else:
            adjustment = 1.0
        
        return base_size * Decimal(str(adjustment))
    
    def evaluate_correlation_risk(
        self,
        new_symbol: str,
        existing_positions: list
    ) -> Tuple[bool, str]:
        """
        Evaluate correlation risk with existing positions.
        
        Args:
            new_symbol: Symbol for new position
            existing_positions: List of current positions
            
        Returns:
            Tuple of (allowed, reason)
        """
        if not existing_positions:
            return True, "No existing positions"
        
        # Simplified correlation check (in practice, would use historical data)
        high_correlation_pairs = {
            ("BTC-USD", "ETH-USD"): 0.8,
            ("ETH-USD", "SOL-USD"): 0.7,
        }
        
        for position in existing_positions:
            if position.side == "FLAT":
                continue
                
            # Check correlation
            pair = tuple(sorted([new_symbol, position.symbol]))
            correlation = high_correlation_pairs.get(pair, 0.3)  # Default low correlation
            
            if correlation > self.correlation_threshold:
                return False, f"High correlation ({correlation:.2f}) with {position.symbol}"
        
        return True, "Correlation risk acceptable"
    
    def dynamic_stop_loss(
        self,
        entry_price: Decimal,
        current_price: Decimal,
        volatility: float,
        time_in_trade: int
    ) -> Decimal:
        """
        Calculate dynamic stop loss based on market conditions.
        
        Args:
            entry_price: Position entry price
            current_price: Current market price
            volatility: Current market volatility
            time_in_trade: Time in trade (minutes)
            
        Returns:
            Dynamic stop loss price
        """
        # Base stop loss (2% default)
        base_stop_pct = 0.02
        
        # Adjust based on volatility
        volatility_adjustment = volatility * 2  # Scale volatility
        adjusted_stop_pct = base_stop_pct + volatility_adjustment
        
        # Adjust based on time in trade (trailing stop)
        if time_in_trade > 240:  # 4 hours
            # Tighten stop loss for long-held positions
            adjusted_stop_pct *= 0.8
        
        # Cap at reasonable levels
        adjusted_stop_pct = min(adjusted_stop_pct, 0.05)  # Max 5%
        adjusted_stop_pct = max(adjusted_stop_pct, 0.01)  # Min 1%
        
        stop_loss_price = entry_price * (1 - Decimal(str(adjusted_stop_pct)))
        
        return stop_loss_price
    
    def evaluate_risk_advanced(
        self,
        trade_action: TradeAction,
        current_position: Position,
        current_price: Decimal,
        market_volatility: float = 0.02
    ) -> Tuple[bool, TradeAction, str]:
        """
        Advanced risk evaluation with volatility and correlation checks.
        
        Args:
            trade_action: Proposed trade action
            current_position: Current position
            current_price: Current market price
            market_volatility: Current market volatility
            
        Returns:
            Tuple of (approved, modified_action, reason)
        """
        # Run standard risk evaluation first
        approved, modified_action, reason = self.evaluate_risk(
            trade_action, current_position, current_price
        )
        
        if not approved:
            return approved, modified_action, reason
        
        # Additional advanced checks
        
        # 1. Volatility adjustment
        if trade_action.action in ["LONG", "SHORT"]:
            original_size = Decimal(str(modified_action.size_pct))
            adjusted_size = self.calculate_volatility_adjusted_size(
                original_size, market_volatility
            )
            
            if adjusted_size != original_size:
                modified_action = TradeAction(
                    action=modified_action.action,
                    size_pct=float(adjusted_size),
                    take_profit_pct=modified_action.take_profit_pct,
                    stop_loss_pct=modified_action.stop_loss_pct,
                    rationale=f"Volatility-adjusted: {modified_action.rationale}"
                )
                reason += f" (volatility-adjusted from {original_size}%)"
        
        # 2. Time-based restrictions (example: no trading in first 30 min of market)
        from datetime import datetime
        current_hour = datetime.now().hour
        if current_hour == 9 and datetime.now().minute < 30:  # Market open
            return False, modified_action, "No trading in first 30 minutes"
        
        return True, modified_action, reason

# Usage in trading engine
class AdvancedTradingEngine(TradingEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace risk manager with advanced version
        self.risk_manager = AdvancedRiskManager()
```

## Health and Monitoring APIs

### 1. Health Check Endpoints

The bot provides REST API endpoints for health monitoring:

```python
from bot.health import create_health_endpoints
from bot.config import create_settings

# Create health endpoints
settings = create_settings()
endpoints = create_health_endpoints(settings)

# Basic health check
health = endpoints.get_health()
print(f"Status: {health['status']}")

# Detailed health check
detailed = endpoints.get_health_detailed()
print(f"Components: {detailed['components']}")

# Performance metrics
metrics = endpoints.get_metrics()
print(f"Memory usage: {metrics['memory_usage_mb']} MB")

# Configuration status
config_status = endpoints.get_configuration_status()
print(f"Config valid: {config_status['valid']}")

# Readiness check (for Kubernetes)
readiness = endpoints.get_readiness()
print(f"Ready: {readiness['ready']}")
```

#### Health Response Format

```json
{
  "status": "healthy",
  "timestamp": "2025-06-11T12:00:00Z",
  "uptime_seconds": 3600,
  "version": "1.0.0",
  "components": {
    "system": {
      "status": "healthy",
      "memory_usage_mb": 512.5,
      "cpu_usage_percent": 25.3,
      "disk_usage_percent": 45.2
    },
    "apis": {
      "status": "healthy", 
      "coinbase_status": "connected",
      "llm_status": "available",
      "response_time_ms": 150
    },
    "filesystem": {
      "status": "healthy",
      "logs_writable": true,
      "data_writable": true,
      "config_readable": true
    },
    "configuration": {
      "status": "healthy",
      "profile": "moderate",
      "dry_run": true,
      "warnings": []
    }
  },
  "metrics": {
    "trades_today": 5,
    "success_rate": 0.8,
    "daily_pnl": 125.50,
    "positions_open": 1,
    "api_requests_today": 1247,
    "error_rate": 0.02
  }
}
```

### 2. Prometheus Metrics

Export metrics in Prometheus format:

```python
from bot.health import create_monitoring_exporter

# Create exporter
exporter = create_monitoring_exporter(settings)

# Export Prometheus metrics
prometheus_metrics = exporter.export_prometheus_metrics()
print(prometheus_metrics)

# Save monitoring snapshot
snapshot_file = exporter.save_monitoring_snapshot()
print(f"Snapshot saved: {snapshot_file}")
```

#### Available Metrics

```prometheus
# System metrics
trading_bot_uptime_seconds{instance="bot-001"} 3600
trading_bot_memory_usage_bytes{instance="bot-001"} 537395200
trading_bot_cpu_usage_percent{instance="bot-001"} 25.3

# Trading metrics
trading_bot_trades_total{symbol="BTC-USD",action="LONG"} 15
trading_bot_trades_total{symbol="BTC-USD",action="SHORT"} 8
trading_bot_positions_active{symbol="BTC-USD"} 1
trading_bot_daily_pnl{symbol="BTC-USD"} 125.50

# API metrics
trading_bot_api_requests_total{provider="coinbase",endpoint="orders"} 145
trading_bot_api_requests_total{provider="openai",endpoint="chat"} 67
trading_bot_api_response_time_seconds{provider="coinbase"} 0.15
trading_bot_api_errors_total{provider="coinbase"} 2

# Health metrics
trading_bot_health_status{component="system"} 1
trading_bot_health_status{component="apis"} 1
trading_bot_health_status{component="configuration"} 1
```

### 3. Custom Monitoring Integration

```python
"""Custom monitoring integration example."""

import time
import json
from typing import Dict, Any
from bot.health import HealthCheckEndpoints

class CustomMonitoringIntegration:
    """Custom monitoring integration with external systems."""
    
    def __init__(self, health_endpoints: HealthCheckEndpoints):
        self.health_endpoints = health_endpoints
        self.metrics_buffer = []
        
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive metrics."""
        return {
            'timestamp': time.time(),
            'health': self.health_endpoints.get_health(),
            'detailed_health': self.health_endpoints.get_health_detailed(),
            'metrics': self.health_endpoints.get_metrics(),
            'config_status': self.health_endpoints.get_configuration_status()
        }
    
    def send_to_datadog(self, metrics: Dict[str, Any]):
        """Send metrics to Datadog."""
        # Implementation would use Datadog client
        print(f"Sending to Datadog: {metrics['timestamp']}")
    
    def send_to_newrelic(self, metrics: Dict[str, Any]):
        """Send metrics to New Relic."""
        # Implementation would use New Relic client
        print(f"Sending to New Relic: {metrics['timestamp']}")
    
    def send_to_custom_endpoint(self, metrics: Dict[str, Any], endpoint_url: str):
        """Send metrics to custom HTTP endpoint."""
        import requests
        
        try:
            response = requests.post(
                endpoint_url,
                json=metrics,
                timeout=10
            )
            response.raise_for_status()
            print(f"✅ Metrics sent to {endpoint_url}")
        except Exception as e:
            print(f"❌ Failed to send metrics: {e}")
    
    async def run_monitoring_loop(self, interval: int = 60):
        """Run continuous monitoring loop."""
        while True:
            try:
                # Collect metrics
                metrics = self.collect_metrics()
                
                # Send to various monitoring systems
                self.send_to_datadog(metrics)
                self.send_to_newrelic(metrics)
                self.send_to_custom_endpoint(
                    metrics, 
                    "https://monitoring.yourcompany.com/metrics"
                )
                
                # Buffer for analysis
                self.metrics_buffer.append(metrics)
                
                # Keep only last 100 measurements
                if len(self.metrics_buffer) > 100:
                    self.metrics_buffer.pop(0)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(30)  # Wait before retry

# Usage
monitoring = CustomMonitoringIntegration(health_endpoints)
asyncio.run(monitoring.run_monitoring_loop(interval=30))
```

## Error Handling

### 1. Exception Hierarchy

The bot defines a comprehensive exception hierarchy:

```python
"""Trading bot exception hierarchy."""

class TradingBotError(Exception):
    """Base exception for all trading bot errors."""
    pass

class ConfigurationError(TradingBotError):
    """Configuration-related errors."""
    pass

class ValidationError(TradingBotError):
    """Data validation errors."""
    pass

class APIError(TradingBotError):
    """API-related errors."""
    
    def __init__(self, message: str, status_code: int = None, response: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response

class ExchangeError(APIError):
    """Exchange API errors."""
    pass

class LLMError(APIError):
    """LLM provider errors."""
    pass

class RiskManagementError(TradingBotError):
    """Risk management constraint violations."""
    pass

class InsufficientFundsError(ExchangeError):
    """Insufficient account funds for trade."""
    pass

class InsufficientDataError(TradingBotError):
    """Insufficient market data for analysis."""
    pass

class PositionError(TradingBotError):
    """Position management errors."""
    pass

# Usage examples
try:
    # Trading operation
    pass
except InsufficientFundsError as e:
    logger.error(f"Insufficient funds: {e}")
    # Handle by reducing position size or waiting
except ExchangeError as e:
    logger.error(f"Exchange error {e.status_code}: {e}")
    # Handle by retrying or switching to dry-run
except LLMError as e:
    logger.error(f"LLM error: {e}")
    # Handle by using fallback strategy
except RiskManagementError as e:
    logger.warning(f"Risk constraint: {e}")
    # This is expected behavior - trade rejected for safety
```

### 2. Error Recovery Strategies

```python
"""Error recovery and retry mechanisms."""

import asyncio
import logging
from typing import Callable, Any, Optional
from functools import wraps

class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        retryable_exceptions: tuple = (APIError, ConnectionError)
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retryable_exceptions = retryable_exceptions

def with_retry(config: RetryConfig):
    """Decorator for automatic retry with exponential backoff."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                    
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        logging.error(f"Final attempt failed for {func.__name__}: {e}")
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.exponential_base ** (attempt - 1)),
                        config.max_delay
                    )
                    
                    logging.warning(
                        f"Attempt {attempt}/{config.max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    
                    await asyncio.sleep(delay)
                    
                except Exception as e:
                    # Non-retryable exception
                    logging.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator

# Usage examples
class RobustExchangeClient:
    """Exchange client with error recovery."""
    
    @with_retry(RetryConfig(max_attempts=3, base_delay=2.0))
    async def place_order_with_retry(self, order_params):
        """Place order with automatic retry."""
        # This will automatically retry on API errors
        return await self.exchange_client.place_order(order_params)
    
    @with_retry(RetryConfig(max_attempts=5, base_delay=1.0))
    async def get_account_balance_with_retry(self):
        """Get balance with retry for temporary network issues."""
        return await self.exchange_client.get_account_balance()
    
    async def execute_trade_with_fallback(self, trade_action):
        """Execute trade with comprehensive error handling."""
        try:
            # Try primary execution method
            return await self.place_order_with_retry(trade_action)
            
        except InsufficientFundsError:
            # Handle by reducing position size
            logging.warning("Insufficient funds, reducing position size")
            reduced_action = trade_action.copy()
            reduced_action.size_pct *= 0.5
            return await self.place_order_with_retry(reduced_action)
            
        except ExchangeError as e:
            if e.status_code == 429:  # Rate limit
                logging.warning("Rate limited, waiting before retry")
                await asyncio.sleep(60)
                return await self.place_order_with_retry(trade_action)
            else:
                raise
                
        except Exception as e:
            # Log error and switch to dry-run mode
            logging.error(f"Unexpected error, switching to dry-run: {e}")
            self.switch_to_dry_run()
            raise

class CircuitBreaker:
    """Circuit breaker pattern for API calls."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = APIError
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            
            # Success - reset circuit breaker
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logging.error(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise
```

## Performance Considerations

### 1. Optimization Guidelines

#### Memory Management

```python
"""Memory optimization techniques."""

import gc
import weakref
from typing import Dict, Any
import pandas as pd

class OptimizedDataManager:
    """Optimized data management for large datasets."""
    
    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self._cache: Dict[str, Any] = {}
        self._cache_order = []  # For LRU eviction
    
    def cache_data(self, key: str, data: Any):
        """Cache data with LRU eviction."""
        # Remove if already exists
        if key in self._cache:
            self._cache_order.remove(key)
        
        # Add to cache
        self._cache[key] = data
        self._cache_order.append(key)
        
        # Evict oldest if over limit
        while len(self._cache) > self.max_cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]
    
    def get_cached_data(self, key: str) -> Any:
        """Get cached data and update access order."""
        if key in self._cache:
            # Move to end (most recent)
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return self._cache[key]
        return None
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        # Convert object columns to category where appropriate
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        # Downcast numeric types
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df
    
    def periodic_cleanup(self):
        """Perform periodic memory cleanup."""
        # Force garbage collection
        gc.collect()
        
        # Clear old cache entries
        if len(self._cache) > self.max_cache_size * 0.8:
            keys_to_remove = self._cache_order[:len(self._cache_order)//4]
            for key in keys_to_remove:
                if key in self._cache:
                    del self._cache[key]
                self._cache_order.remove(key)

# Memory monitoring
class MemoryMonitor:
    """Monitor and manage memory usage."""
    
    def __init__(self, warning_threshold_mb: int = 800, critical_threshold_mb: int = 1500):
        self.warning_threshold = warning_threshold_mb * 1024 * 1024  # Convert to bytes
        self.critical_threshold = critical_threshold_mb * 1024 * 1024
    
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    
    def check_memory_usage(self) -> str:
        """Check memory usage and return status."""
        usage = self.get_memory_usage()
        
        if usage > self.critical_threshold:
            return "CRITICAL"
        elif usage > self.warning_threshold:
            return "WARNING"
        else:
            return "OK"
    
    def cleanup_if_needed(self):
        """Perform cleanup if memory usage is high."""
        status = self.check_memory_usage()
        
        if status in ["WARNING", "CRITICAL"]:
            logging.warning(f"High memory usage detected: {status}")
            gc.collect()
            
            if status == "CRITICAL":
                # More aggressive cleanup
                self._emergency_cleanup()
    
    def _emergency_cleanup(self):
        """Emergency memory cleanup procedures."""
        # Clear any large data structures
        # Restart components if necessary
        logging.critical("Performing emergency memory cleanup")
```

#### Async Performance

```python
"""Async performance optimization."""

import asyncio
import aiohttp
from typing import List, Coroutine, Any
import time

class AsyncOptimizer:
    """Optimize async operations for better performance."""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def batch_requests(
        self, 
        requests: List[Coroutine], 
        batch_size: int = 5
    ) -> List[Any]:
        """Execute requests in batches to avoid overwhelming APIs."""
        results = []
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            results.extend(batch_results)
            
            # Small delay between batches to be nice to APIs
            if i + batch_size < len(requests):
                await asyncio.sleep(0.1)
        
        return results
    
    async def limited_request(self, coro: Coroutine) -> Any:
        """Execute request with concurrency limiting."""
        async with self.semaphore:
            return await coro
    
    async def timeout_request(
        self, 
        coro: Coroutine, 
        timeout: float = 30.0
    ) -> Any:
        """Execute request with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise APIError(f"Request timed out after {timeout}s")

# Connection pooling
class OptimizedHTTPClient:
    """Optimized HTTP client with connection pooling."""
    
    def __init__(self):
        self.connector = aiohttp.TCPConnector(
            limit=100,              # Total connection pool size
            limit_per_host=30,      # Per-host limit
            ttl_dns_cache=300,      # DNS cache TTL
            use_dns_cache=True,     # Enable DNS caching
            keepalive_timeout=60,   # Keep connections alive
            enable_cleanup_closed=True
        )
        
        self.timeout = aiohttp.ClientTimeout(
            total=30,       # Total timeout
            connect=10,     # Connection timeout
            sock_read=20    # Socket read timeout
        )
        
        self.session = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=self.timeout
            )
        return self.session
    
    async def close(self):
        """Close HTTP session and cleanup."""
        if self.session and not self.session.closed:
            await self.session.close()
        
        # Close connector
        await self.connector.close()

# Performance monitoring
class PerformanceProfiler:
    """Profile performance of trading operations."""
    
    def __init__(self):
        self.metrics = {}
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return TimingContext(operation_name, self.metrics)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance metrics report."""
        report = {}
        for operation, times in self.metrics.items():
            report[operation] = {
                'count': len(times),
                'total_time': sum(times),
                'avg_time': sum(times) / len(times) if times else 0,
                'min_time': min(times) if times else 0,
                'max_time': max(times) if times else 0
            }
        return report

class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str, metrics: Dict):
        self.operation_name = operation_name
        self.metrics = metrics
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            if self.operation_name not in self.metrics:
                self.metrics[self.operation_name] = []
            self.metrics[self.operation_name].append(duration)

# Usage example
profiler = PerformanceProfiler()

async def optimized_trading_loop():
    """Trading loop with performance optimization."""
    optimizer = AsyncOptimizer(max_concurrent=5)
    http_client = OptimizedHTTPClient()
    
    try:
        while True:
            with profiler.time_operation('market_data_fetch'):
                # Fetch market data
                market_data = await fetch_market_data()
            
            with profiler.time_operation('indicator_calculation'):
                # Calculate indicators
                indicators = calculate_indicators(market_data)
            
            with profiler.time_operation('llm_analysis'):
                # Get LLM decision (with timeout)
                decision = await optimizer.timeout_request(
                    get_llm_decision(indicators),
                    timeout=10.0
                )
            
            # Trade execution with concurrency limit
            if decision.action != "HOLD":
                with profiler.time_operation('trade_execution'):
                    await optimizer.limited_request(
                        execute_trade(decision)
                    )
            
            await asyncio.sleep(60)  # Wait before next iteration
            
    finally:
        await http_client.close()
        
        # Print performance report
        report = profiler.get_performance_report()
        for operation, stats in report.items():
            print(f"{operation}: avg={stats['avg_time']:.3f}s, count={stats['count']}")
```

### 2. Scalability Patterns

#### Multi-Instance Management

```python
"""Multi-instance trading bot management."""

import asyncio
import json
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class BotInstance:
    """Represents a single bot instance."""
    id: str
    symbol: str
    interval: str
    status: str  # running, stopped, error
    pid: int
    last_heartbeat: float

class BotCluster:
    """Manage multiple trading bot instances."""
    
    def __init__(self):
        self.instances: Dict[str, BotInstance] = {}
        self.coordinator_running = False
    
    async def start_instance(
        self, 
        bot_id: str, 
        symbol: str, 
        interval: str = "1m"
    ) -> bool:
        """Start a new bot instance."""
        if bot_id in self.instances:
            return False
        
        # Start new process for bot instance
        process = await asyncio.create_subprocess_exec(
            "python", "-m", "bot.main", "live",
            "--symbol", symbol,
            "--interval", interval,
            "--instance-id", bot_id
        )
        
        # Track instance
        instance = BotInstance(
            id=bot_id,
            symbol=symbol,
            interval=interval,
            status="running",
            pid=process.pid,
            last_heartbeat=time.time()
        )
        
        self.instances[bot_id] = instance
        return True
    
    async def stop_instance(self, bot_id: str) -> bool:
        """Stop a bot instance."""
        if bot_id not in self.instances:
            return False
        
        instance = self.instances[bot_id]
        
        # Send graceful shutdown signal
        os.kill(instance.pid, signal.SIGTERM)
        
        # Wait for shutdown
        await asyncio.sleep(5)
        
        # Force kill if still running
        try:
            os.kill(instance.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # Process already terminated
        
        # Remove from tracking
        del self.instances[bot_id]
        return True
    
    async def monitor_instances(self):
        """Monitor all running instances."""
        while self.coordinator_running:
            for bot_id, instance in list(self.instances.items()):
                # Check if process is still running
                try:
                    os.kill(instance.pid, 0)  # Signal 0 just checks existence
                except ProcessLookupError:
                    # Process died
                    instance.status = "error"
                    print(f"Instance {bot_id} died unexpectedly")
                    
                    # Restart if configured
                    if should_restart_instance(instance):
                        await self.restart_instance(bot_id)
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def restart_instance(self, bot_id: str):
        """Restart a failed instance."""
        if bot_id not in self.instances:
            return
        
        instance = self.instances[bot_id]
        print(f"Restarting instance {bot_id}")
        
        # Stop old instance
        await self.stop_instance(bot_id)
        
        # Start new instance
        await self.start_instance(bot_id, instance.symbol, instance.interval)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get status of all instances in cluster."""
        return {
            "total_instances": len(self.instances),
            "running_instances": len([i for i in self.instances.values() if i.status == "running"]),
            "error_instances": len([i for i in self.instances.values() if i.status == "error"]),
            "instances": {
                bot_id: {
                    "symbol": instance.symbol,
                    "status": instance.status,
                    "pid": instance.pid
                }
                for bot_id, instance in self.instances.items()
            }
        }

# Load balancing for API requests
class LoadBalancer:
    """Load balance API requests across multiple endpoints."""
    
    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.current_index = 0
        self.endpoint_stats = {endpoint: {"requests": 0, "errors": 0} for endpoint in endpoints}
    
    def get_next_endpoint(self) -> str:
        """Get next endpoint using round-robin."""
        endpoint = self.endpoints[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.endpoints)
        return endpoint
    
    def get_best_endpoint(self) -> str:
        """Get endpoint with lowest error rate."""
        best_endpoint = None
        best_score = float('inf')
        
        for endpoint in self.endpoints:
            stats = self.endpoint_stats[endpoint]
            error_rate = stats["errors"] / max(stats["requests"], 1)
            
            if error_rate < best_score:
                best_score = error_rate
                best_endpoint = endpoint
        
        return best_endpoint or self.endpoints[0]
    
    def record_request(self, endpoint: str, success: bool):
        """Record request outcome for endpoint."""
        self.endpoint_stats[endpoint]["requests"] += 1
        if not success:
            self.endpoint_stats[endpoint]["errors"] += 1

# Usage example
async def run_multi_instance_cluster():
    """Run a cluster of trading bot instances."""
    cluster = BotCluster()
    
    # Start instances for different symbols
    await cluster.start_instance("btc-1m", "BTC-USD", "1m")
    await cluster.start_instance("eth-1m", "ETH-USD", "1m")
    await cluster.start_instance("btc-5m", "BTC-USD", "5m")
    
    # Start monitoring
    cluster.coordinator_running = True
    monitor_task = asyncio.create_task(cluster.monitor_instances())
    
    try:
        # Run indefinitely
        while True:
            status = cluster.get_cluster_status()
            print(f"Cluster status: {status['running_instances']}/{status['total_instances']} running")
            await asyncio.sleep(60)
            
    finally:
        cluster.coordinator_running = False
        await monitor_task
        
        # Stop all instances
        for bot_id in list(cluster.instances.keys()):
            await cluster.stop_instance(bot_id)
```

---

This comprehensive API reference provides detailed documentation for all major components, configuration options, data models, and integration patterns of the AI Trading Bot. It serves as both a reference for developers using the bot and a guide for extending its functionality.