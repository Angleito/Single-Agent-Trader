"""
Integration tests for the functional trading bot runtime components.

This module tests the integration between different runtime components
including the interpreter, scheduler, monitoring, and WebSocket management.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

# Import effects
from bot.fp.effects.io import IO
from bot.fp.effects.logging import LogLevel
from bot.fp.runtime.backtest import BacktestConfig, BacktestEngine
from bot.fp.runtime.cli import FunctionalCLI

# Import runtime components
from bot.fp.runtime.interpreter import EffectInterpreter, RuntimeConfig
from bot.fp.runtime.monitoring import MonitoringConfig, MonitoringRuntime
from bot.fp.runtime.scheduler import ScheduledTask, SchedulerConfig, TradingScheduler
from bot.fp.runtime.state import StateManager, create_state, get_state


class TestEffectInterpreter:
    """Test the effect interpreter"""

    def test_interpreter_creation(self):
        """Test creating an effect interpreter"""
        config = RuntimeConfig(
            max_concurrent_effects=50,
            effect_timeout=10.0,
            error_recovery=True,
            metrics_enabled=False,
        )

        interpreter = EffectInterpreter(config)

        assert interpreter.config.max_concurrent_effects == 50
        assert interpreter.config.effect_timeout == 10.0
        assert interpreter.context.active_effects == 0

    def test_run_simple_effect(self):
        """Test running a simple IO effect"""
        interpreter = EffectInterpreter(RuntimeConfig(metrics_enabled=False))

        effect = IO.pure(42)
        result = interpreter.run_effect(effect)

        assert result == 42

    def test_run_failing_effect(self):
        """Test running an effect that fails"""
        interpreter = EffectInterpreter(
            RuntimeConfig(error_recovery=False, metrics_enabled=False)
        )

        def failing_computation():
            raise ValueError("Test error")

        effect = IO(failing_computation)

        with pytest.raises(ValueError, match="Test error"):
            interpreter.run_effect(effect)

    def test_runtime_stats(self):
        """Test getting runtime statistics"""
        interpreter = EffectInterpreter(RuntimeConfig())

        stats = interpreter.get_runtime_stats()

        assert "active_effects" in stats
        assert "config" in stats
        assert stats["active_effects"] == 0


class TestTradingScheduler:
    """Test the trading scheduler"""

    def test_scheduler_creation(self):
        """Test creating a scheduler"""
        config = SchedulerConfig(
            trading_interval=timedelta(seconds=30), max_concurrent_tasks=5
        )

        scheduler = TradingScheduler(config)

        assert scheduler.config.trading_interval == timedelta(seconds=30)
        assert len(scheduler.tasks) == 0
        assert not scheduler.running

    def test_add_task(self):
        """Test adding a task to the scheduler"""
        scheduler = TradingScheduler(SchedulerConfig())

        task = ScheduledTask(
            name="test_task",
            effect=IO.pure("test result"),
            interval=timedelta(seconds=10),
        )

        scheduler.add_task(task)

        assert "test_task" in scheduler.tasks
        assert scheduler.tasks["test_task"].enabled

    def test_task_should_run(self):
        """Test task scheduling logic"""
        scheduler = TradingScheduler(SchedulerConfig())

        task = ScheduledTask(
            name="test_task",
            effect=IO.pure("test"),
            interval=timedelta(seconds=1),
            last_run=None,
        )

        # Should run if never run before
        assert scheduler.should_run_task(task)

        # Should not run if recently run
        task.last_run = datetime.utcnow()
        assert not scheduler.should_run_task(task)

        # Should run again after interval
        task.last_run = datetime.utcnow() - timedelta(seconds=2)
        assert scheduler.should_run_task(task)

    def test_disable_task(self):
        """Test disabling a task"""
        scheduler = TradingScheduler(SchedulerConfig())

        task = ScheduledTask(
            name="test_task", effect=IO.pure("test"), interval=timedelta(seconds=1)
        )

        scheduler.add_task(task)
        scheduler.disable_task("test_task")

        assert not scheduler.tasks["test_task"].enabled
        assert not scheduler.should_run_task(scheduler.tasks["test_task"])


class TestStateManager:
    """Test the state manager"""

    def test_create_state_ref(self):
        """Test creating a state reference"""
        manager = StateManager()

        ref = manager.create_ref("test_key", {"value": 42})

        assert ref is not None

        # Read the value
        value = ref.read().run()
        assert value == {"value": 42}

    def test_modify_state(self):
        """Test modifying state"""
        manager = StateManager()

        ref = manager.create_ref("counter", 0)

        # Modify the state
        new_value = ref.modify(lambda x: x + 1).run()

        assert new_value == 1

        # Read to confirm
        current_value = ref.read().run()
        assert current_value == 1

    def test_global_state_functions(self):
        """Test global state functions"""
        ref = create_state("global_test", {"balance": 1000})

        assert ref is not None

        # Get the same reference
        same_ref = get_state("global_test")
        assert same_ref is ref

        # Modify through reference
        new_value = ref.modify(lambda state: {**state, "balance": 2000}).run()
        assert new_value["balance"] == 2000


class TestMonitoringRuntime:
    """Test the monitoring runtime"""

    def test_monitoring_creation(self):
        """Test creating monitoring runtime"""
        config = MonitoringConfig(
            health_check_interval=timedelta(seconds=30),
            metrics_collection_interval=timedelta(seconds=10),
        )

        monitoring = MonitoringRuntime(config)

        assert monitoring.config.health_check_interval == timedelta(seconds=30)
        assert not monitoring.running
        assert len(monitoring.metrics_history) == 0

    def test_collect_metrics(self):
        """Test metrics collection"""
        monitoring = MonitoringRuntime(MonitoringConfig())

        metrics = monitoring.collect_system_metrics().run()

        assert metrics.timestamp is not None
        assert metrics.cpu_percent >= 0
        assert metrics.memory_mb >= 0

    def test_record_metrics(self):
        """Test recording metrics"""
        monitoring = MonitoringRuntime(MonitoringConfig())

        # Create fake metrics
        from bot.fp.runtime.monitoring import SystemMetrics

        metrics = SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=25.0,
            memory_mb=500.0,
            active_connections=5,
            error_rate=1.0,
            response_time_ms=50.0,
        )

        monitoring.record_metrics(metrics).run()

        assert len(monitoring.metrics_history) == 1
        assert monitoring.metrics_history[0] == metrics


class TestBacktestEngine:
    """Test the backtesting engine"""

    def test_backtest_config(self):
        """Test backtest configuration"""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            symbol="BTC-USD",
            initial_balance=Decimal(10000),
        )

        engine = BacktestEngine(config)

        assert engine.config.symbol == "BTC-USD"
        assert engine.balance == Decimal(10000)
        assert len(engine.positions) == 0

    def test_load_historical_data(self):
        """Test loading historical data"""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1, 0, 0),
            end_date=datetime(2024, 1, 1, 0, 5),  # 5 minutes
            symbol="BTC-USD",
        )

        engine = BacktestEngine(config)
        data = engine.load_historical_data().run()

        assert len(data) == 6  # 0, 1, 2, 3, 4, 5 minutes
        assert all("timestamp" in item for item in data)
        assert all("close" in item for item in data)

    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation"""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            initial_balance=Decimal(10000),
        )

        engine = BacktestEngine(config)

        # No positions - should equal balance
        value = engine.calculate_portfolio_value(Decimal(50000))
        assert value == Decimal(10000)

        # Add a position
        engine.positions.append(
            {
                "side": "long",
                "entry_price": Decimal(50000),
                "size": Decimal(100),
                "entry_time": datetime.utcnow(),
            }
        )

        # Calculate with different price
        value = engine.calculate_portfolio_value(Decimal(51000))
        assert value > Decimal(10000)  # Should be profitable


class TestCLI:
    """Test the CLI interface"""

    def test_parse_args(self):
        """Test argument parsing"""
        cli = FunctionalCLI()

        args = ["live", "--config", "test.json", "--debug"]
        config = cli.parse_args(args)

        assert config.command == "live"
        assert config.options["config"] == "test.json"
        assert config.options["debug"] is True
        assert config.log_level == LogLevel.DEBUG

    def test_parse_empty_args(self):
        """Test parsing empty arguments"""
        cli = FunctionalCLI()

        config = cli.parse_args([])

        assert config.command == "help"
        assert len(config.options) == 0


class TestIntegrationScenarios:
    """Integration tests for combined runtime components"""

    def test_scheduler_with_interpreter(self):
        """Test scheduler working with interpreter"""
        # Create components
        interpreter = EffectInterpreter(RuntimeConfig(metrics_enabled=False))
        scheduler = TradingScheduler(SchedulerConfig())

        # Create a counter in global state
        counter_ref = create_state("test_counter", 0)

        # Create task that increments counter
        def increment_counter():
            current = counter_ref.read().run()
            counter_ref.write(current + 1).run()
            return current + 1

        task = ScheduledTask(
            name="increment_task",
            effect=IO(increment_counter),
            interval=timedelta(milliseconds=1),
        )

        scheduler.add_task(task)

        # Run task manually
        scheduler.run_task(task)

        # Check counter was incremented
        final_value = counter_ref.read().run()
        assert final_value == 1

    def test_monitoring_with_state(self):
        """Test monitoring integration with state management"""
        # Create monitoring
        monitoring = MonitoringRuntime(MonitoringConfig())

        # Create state for tracking metrics
        metrics_ref = create_state("system_metrics", {})

        # Collect and store metrics
        metrics = monitoring.collect_system_metrics().run()
        metrics_ref.write(
            {
                "cpu": metrics.cpu_percent,
                "memory": metrics.memory_mb,
                "timestamp": metrics.timestamp.isoformat(),
            }
        ).run()

        # Verify metrics were stored
        stored_metrics = metrics_ref.read().run()
        assert "cpu" in stored_metrics
        assert "memory" in stored_metrics
        assert stored_metrics["cpu"] == metrics.cpu_percent

    @pytest.mark.asyncio
    async def test_concurrent_runtime_components(self):
        """Test multiple runtime components working together"""
        # Create components
        scheduler = TradingScheduler(SchedulerConfig())
        monitoring = MonitoringRuntime(MonitoringConfig())

        # Create shared state
        status_ref = create_state(
            "system_status",
            {
                "scheduler_running": False,
                "monitoring_running": False,
                "last_update": None,
            },
        )

        # Update status function
        def update_status(component: str, running: bool):
            current = status_ref.read().run()
            current[f"{component}_running"] = running
            current["last_update"] = datetime.utcnow().isoformat()
            status_ref.write(current).run()

        # Start components (simulate)
        update_status("scheduler", True)
        update_status("monitoring", True)

        # Check final status
        final_status = status_ref.read().run()
        assert final_status["scheduler_running"]
        assert final_status["monitoring_running"]
        assert final_status["last_update"] is not None


if __name__ == "__main__":
    pytest.main([__file__])
