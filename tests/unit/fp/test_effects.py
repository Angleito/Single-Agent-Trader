"""
Comprehensive tests for all effect modules in the functional trading bot.

This module tests all effect modules including IO monads, market data effects,
exchange effects, and all other functional components.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from bot.fp.effects.config import Config, ConfigSource, load_config
from bot.fp.effects.error import RetryPolicy, fallback
from bot.fp.effects.exchange import get_balance, place_order

# Import all effect modules
from bot.fp.effects.io import IO, IOEither, parallel, sequence
from bot.fp.effects.logging import LogLevel, debug, info
from bot.fp.effects.market_data import APIConfig, ConnectionConfig
from bot.fp.effects.monitoring import HealthStatus, health_check, increment_counter
from bot.fp.effects.persistence import State, load_state, save_state
from bot.fp.effects.time import delay, measure_time, now


class TestIOMonad:
    """Test IO monad laws and operations"""

    def test_io_identity_law(self):
        """Test left identity: IO.pure(a).flat_map(f) == f(a)"""

        def f(x):
            return IO.pure(x * 2)

        a = 5
        left_side = IO.pure(a).flat_map(f)
        right_side = f(a)

        assert left_side.run() == right_side.run()

    def test_io_right_identity_law(self):
        """Test right identity: m.flat_map(IO.pure) == m"""
        m = IO.pure(10)
        result = m.flat_map(IO.pure)

        assert result.run() == m.run()

    def test_io_associativity_law(self):
        """Test associativity: m.flat_map(f).flat_map(g) == m.flat_map(x => f(x).flat_map(g))"""

        def f(x):
            return IO.pure(x + 1)

        def g(x):
            return IO.pure(x * 2)

        m = IO.pure(5)

        left_side = m.flat_map(f).flat_map(g)
        right_side = m.flat_map(lambda x: f(x).flat_map(g))

        assert left_side.run() == right_side.run()

    def test_io_map(self):
        """Test IO map operation"""
        io = IO.pure(10)
        mapped = io.map(lambda x: x * 2)

        assert mapped.run() == 20

    def test_io_chain(self):
        """Test IO chain operation"""
        io1 = IO.pure(10)
        io2 = IO.pure(20)
        chained = io1.chain(io2)

        assert chained.run() == 20


class TestIOEither:
    """Test IOEither error handling"""

    def test_io_either_success(self):
        """Test successful IOEither operation"""
        success = IOEither.pure(42)
        result = success.run()

        assert result.is_right()
        assert result.value == 42

    def test_io_either_failure(self):
        """Test failed IOEither operation"""
        failure = IOEither.left("error")
        result = failure.run()

        assert result.is_left()
        assert result.value == "error"

    def test_io_either_map_success(self):
        """Test mapping over successful IOEither"""
        success = IOEither.pure(10)
        mapped = success.map(lambda x: x * 2)
        result = mapped.run()

        assert result.is_right()
        assert result.value == 20

    def test_io_either_map_failure(self):
        """Test mapping over failed IOEither"""
        failure = IOEither.left("error")
        mapped = failure.map(lambda x: x * 2)
        result = mapped.run()

        assert result.is_left()
        assert result.value == "error"

    def test_io_either_recover(self):
        """Test recovery from IOEither failure"""
        failure = IOEither.left("error")
        recovered = failure.recover(lambda e: f"recovered from {e}")

        assert recovered.run() == "recovered from error"


class TestEffectComposition:
    """Test effect composition and sequencing"""

    def test_sequence_ios(self):
        """Test sequencing multiple IOs"""
        ios = [IO.pure(1), IO.pure(2), IO.pure(3)]
        sequenced = sequence(ios)

        assert sequenced.run() == [1, 2, 3]

    def test_parallel_execution(self):
        """Test parallel execution of IOs"""

        def slow_computation(x):
            return IO(lambda: x * 2)

        ios = [slow_computation(i) for i in range(3)]
        result = parallel(ios, max_workers=2)

        assert result.run() == [0, 2, 4]


class TestMarketDataEffects:
    """Test market data effects"""

    def test_connection_config(self):
        """Test WebSocket connection configuration"""
        config = ConnectionConfig(
            url="wss://test.com", headers={"Authorization": "Bearer token"}
        )

        assert config.url == "wss://test.com"
        assert config.headers["Authorization"] == "Bearer token"
        assert config.heartbeat_interval == 30

    def test_api_config(self):
        """Test REST API configuration"""

        # Create a simple RateLimit for testing
        rate_limit = type("RateLimit", (), {"requests_per_second": 10})()

        config = APIConfig(
            base_url="https://api.test.com",
            headers={"API-Key": "test"},
            rate_limit=rate_limit,
        )

        assert config.base_url == "https://api.test.com"
        assert config.timeout == 30


class TestExchangeEffects:
    """Test exchange interaction effects"""

    def test_place_order_validation(self):
        """Test order placement validation"""

        # Create a mock order type for testing
        order = type(
            "Order",
            (),
            {"size": Decimal("0.1"), "price": Decimal(50000), "type": "limit"},
        )()

        result = place_order(order)
        order_result = result.run()

        assert order_result.is_right()

    def test_get_balance(self):
        """Test balance retrieval"""
        result = get_balance()
        balance = result.run()

        assert balance.is_right()
        assert balance.value.total_balance == Decimal(10000)


class TestLoggingEffects:
    """Test logging effects"""

    def test_info_logging(self):
        """Test info level logging"""
        log_effect = info("Test message", {"key": "value"})

        # Should not raise exception
        log_effect.run()

    def test_debug_logging(self):
        """Test debug level logging"""
        log_effect = debug("Debug message")

        # Should not raise exception
        log_effect.run()

    def test_log_levels(self):
        """Test log level enumeration"""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARN.value == "WARN"
        assert LogLevel.ERROR.value == "ERROR"


class TestPersistenceEffects:
    """Test persistence effects"""

    def test_save_and_load_state(self):
        """Test state persistence"""
        state = State(data={"balance": 1000}, version=1, updated_at=datetime.utcnow())

        save_effect = save_state("test_key", state)
        save_effect.run()

        load_effect = load_state("test_key")
        loaded = load_effect.run()

        # Should return None in this mock implementation
        assert loaded is None


class TestTimeEffects:
    """Test time-based effects"""

    def test_now_effect(self):
        """Test current time effect"""
        time_effect = now()
        current_time = time_effect.run()

        assert isinstance(current_time, datetime)

    def test_delay_effect(self):
        """Test delay effect"""
        delay_effect = delay(timedelta(milliseconds=1))

        # Should complete without error
        delay_effect.run()

    def test_measure_time(self):
        """Test time measurement"""
        test_effect = IO.pure(42)
        measured = measure_time(test_effect)
        result, duration = measured.run()

        assert result == 42
        assert isinstance(duration, timedelta)


class TestErrorHandling:
    """Test error handling effects"""

    def test_retry_policy(self):
        """Test retry policy creation"""
        policy = RetryPolicy(max_attempts=3, delay=0.1)

        assert policy.max_attempts == 3
        assert policy.delay == 0.1

    def test_fallback_effect(self):
        """Test fallback on error"""
        failing_effect = IO(lambda: 1 / 0)  # Division by zero
        safe_effect = fallback(42, failing_effect)

        result = safe_effect.run()
        assert result == 42


class TestMonitoringEffects:
    """Test monitoring and metrics effects"""

    def test_increment_counter(self):
        """Test counter increment"""
        counter_effect = increment_counter("test.counter", {"tag": "value"})

        # Should complete without error
        counter_effect.run()

    def test_health_check(self):
        """Test health check"""
        health_effect = health_check("test_service")
        status = health_effect.run()

        assert status == HealthStatus.HEALTHY


class TestConfigEffects:
    """Test configuration effects"""

    def test_load_config_from_env(self):
        """Test loading config from environment"""
        config_effect = load_config(ConfigSource.ENV)
        result = config_effect.run()

        assert result.is_right()
        assert result.value.source == ConfigSource.ENV

    def test_config_validation(self):
        """Test configuration validation"""
        from bot.fp.effects.config import validate_config

        config = Config(data={"key": "value"}, source=ConfigSource.ENV, version="1.0")

        validation_effect = validate_config(config)
        validated = validation_effect.run()

        assert validated.validated is True
        assert len(validated.errors) == 0


class TestPropertyBased:
    """Property-based tests using hypothesis-style testing"""

    def test_io_map_composition(self):
        """Test that map composition works correctly"""

        def f(x):
            return x + 1

        def g(x):
            return x * 2

        values = [1, 5, 10, 100]

        for value in values:
            io = IO.pure(value)

            # map(g . f) == map(g) . map(f)
            left = io.map(lambda x: g(f(x)))
            right = io.map(f).map(g)

            assert left.run() == right.run()

    def test_io_either_error_propagation(self):
        """Test that errors propagate correctly through IOEither chains"""
        error_msg = "test error"

        failed = IOEither.left(error_msg)

        # Multiple operations should preserve the error
        result = failed.map(lambda x: x + 1).map(lambda x: x * 2).run()

        assert result.is_left()
        assert result.value == error_msg


class TestIntegrationScenarios:
    """Integration tests for real-world trading scenarios"""

    def test_trading_pipeline(self):
        """Test a complete trading pipeline using effects"""
        # Simulate a trading decision pipeline
        market_data = IO.pure({"price": 50000, "volume": 100})

        # Apply indicators
        with_indicators = market_data.map(
            lambda data: {**data, "rsi": 65, "sma": 49500}
        )

        # Make trading decision
        decision = with_indicators.map(
            lambda data: (
                "BUY" if data["rsi"] > 60 and data["price"] > data["sma"] else "HOLD"
            )
        )

        result = decision.run()
        assert result == "BUY"

    def test_error_recovery_scenario(self):
        """Test error recovery in trading scenario"""

        # Simulate API failure with recovery
        def api_call():
            raise ConnectionError("API unavailable")

        failing_effect = IO(api_call)
        recovered = fallback({"price": 50000}, failing_effect)

        result = recovered.run()
        assert result == {"price": 50000}

    def test_concurrent_market_data(self):
        """Test concurrent market data processing"""
        symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

        # Create effects for fetching data for each symbol
        effects = [IO.pure(f"{symbol}_data") for symbol in symbols]

        # Process in parallel
        results = parallel(effects, max_workers=3)

        assert results.run() == ["BTC-USD_data", "ETH-USD_data", "SOL-USD_data"]


if __name__ == "__main__":
    pytest.main([__file__])
