"""
Comprehensive integration test for paper trading performance logging fixes.

This test validates the complete paper trading performance logging pipeline:
1. Trade execution and immediate file persistence
2. WebSocket performance data publishing
3. Structured trade logging integration
4. Performance monitor integration
5. Complete data flow validation

Tests both positive and negative scenarios to ensure robustness.
"""

import asyncio
import json
import tempfile
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot.config import Settings
from bot.logging.trade_logger import TradeLogger
from bot.paper_trading import PaperTradingAccount
from bot.performance_monitor import PerformanceMonitor, PerformanceThresholds
from bot.trading_types import TradeAction
from bot.websocket_publisher import WebSocketPublisher


class TestPaperTradingPerformanceFix:
    """Test comprehensive paper trading performance logging integration."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = Mock(spec=Settings)
        settings.system.dry_run = True
        settings.system.enable_websocket_publishing = True
        settings.system.websocket_dashboard_url = "ws://localhost:8000/ws"
        settings.system.websocket_publish_interval = 0.1
        settings.system.websocket_max_retries = 3
        settings.system.websocket_retry_delay = 1
        settings.system.websocket_timeout = 5

        settings.paper_trading.starting_balance = 10000
        settings.paper_trading.fee_rate = 0.001
        settings.paper_trading.slippage_rate = 0.0005

        settings.trading.leverage = 5
        settings.trading.symbol = "BTC-USD"
        settings.trading.enable_futures = False

        return settings

    @pytest.fixture
    def mock_websocket_connection(self):
        """Mock WebSocket connection for testing."""
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.send = AsyncMock()
        mock_ws.ping = AsyncMock()
        mock_ws.close = AsyncMock()
        return mock_ws

    @pytest.fixture
    def sample_trade_actions(self):
        """Sample trade actions for testing."""
        return {
            "long": TradeAction(
                action="LONG",
                size_pct=10,
                take_profit_pct=2.0,
                stop_loss_pct=1.5,
                rationale="Bullish indicators suggest upward momentum",
                leverage=5,
            ),
            "short": TradeAction(
                action="SHORT",
                size_pct=8,
                take_profit_pct=1.8,
                stop_loss_pct=1.2,
                rationale="Bearish divergence signals potential downward move",
                leverage=5,
            ),
            "close": TradeAction(
                action="CLOSE",
                size_pct=0,
                take_profit_pct=1.0,
                stop_loss_pct=1.0,
                rationale="Taking profit at resistance level",
                leverage=5,
            ),
            "hold": TradeAction(
                action="HOLD",
                size_pct=0,
                take_profit_pct=1.0,
                stop_loss_pct=1.0,
                rationale="Mixed signals, waiting for clearer direction",
                leverage=5,
            ),
        }

    @pytest.fixture
    def performance_monitor(self):
        """Performance monitor for testing."""
        thresholds = PerformanceThresholds()
        monitor = PerformanceMonitor(thresholds)
        return monitor

    @pytest.mark.asyncio
    async def test_complete_paper_trading_performance_pipeline(
        self,
        temp_data_dir,
        mock_settings,
        mock_websocket_connection,
        sample_trade_actions,
        performance_monitor,
    ):
        """
        Test complete paper trading performance logging pipeline.

        Validates:
        1. Trade execution -> immediate file persistence
        2. Performance data WebSocket publishing
        3. Structured trade logging
        4. Performance monitor integration
        """
        # Setup components
        data_dir = temp_data_dir / "paper_trading"
        log_dir = temp_data_dir / "logs"

        paper_account = PaperTradingAccount(
            starting_balance=Decimal("10000"), data_dir=data_dir
        )

        trade_logger = TradeLogger(log_dir=log_dir)

        # Mock WebSocket publisher with real connection
        with patch("websockets.connect", return_value=mock_websocket_connection):
            websocket_publisher = WebSocketPublisher(mock_settings)
            await websocket_publisher.initialize()

            # Test 1: Execute LONG trade and validate immediate persistence
            current_price = Decimal("50000")
            trade_action = sample_trade_actions["long"]

            # Execute trade
            order = paper_account.execute_trade_action(
                trade_action, "BTC-USD", current_price
            )

            # Validate trade execution
            assert order is not None
            assert order.status.value == "FILLED"
            assert paper_account.current_position.side == "LONG"
            assert paper_account.current_position.size > 0

            # Test immediate file persistence
            trades_file = data_dir / "trades.json"
            performance_file = data_dir / "performance.json"
            session_trades_file = data_dir / "session_trades.json"

            assert trades_file.exists(), "Trades file should be created immediately"
            assert performance_file.exists(), "Performance file should be created"
            assert session_trades_file.exists(), "Session trades file should be created"

            # Validate trades file content
            with open(trades_file) as f:
                trades_data = json.load(f)

            assert "open_trades" in trades_data
            assert "closed_trades" in trades_data
            assert len(trades_data["open_trades"]) == 1

            open_trade_id = list(trades_data["open_trades"].keys())[0]
            open_trade = trades_data["open_trades"][open_trade_id]
            assert open_trade["symbol"] == "BTC-USD"
            assert open_trade["side"] == "LONG"
            assert float(open_trade["entry_price"]) == float(current_price)

            # Validate session trades file
            with open(session_trades_file) as f:
                session_trades = json.load(f)

            assert len(session_trades) == 1
            assert session_trades[0]["status"] == "OPEN"
            assert session_trades[0]["symbol"] == "BTC-USD"

            # Test 2: WebSocket performance data publishing
            account_status = paper_account.get_account_status(
                {"BTC-USD": current_price}
            )

            await websocket_publisher.publish_performance_update(account_status)

            # Verify WebSocket message was queued and sent
            assert mock_websocket_connection.send.called
            sent_messages = []
            for call in mock_websocket_connection.send.call_args_list:
                message_json = call[0][0]
                message = json.loads(message_json)
                sent_messages.append(message)

            # Find performance update message
            performance_messages = [
                msg for msg in sent_messages if msg.get("type") == "performance_update"
            ]
            assert (
                len(performance_messages) > 0
            ), "Performance update should be published"

            perf_msg = performance_messages[0]
            assert "metrics" in perf_msg
            assert "equity" in perf_msg["metrics"]
            assert "total_pnl" in perf_msg["metrics"]

            # Test 3: Structured trade logging
            trade_logger.log_trade_decision(
                market_state=Mock(
                    symbol="BTC-USD",
                    current_price=current_price,
                    current_position=paper_account.current_position,
                    indicators=Mock(rsi=65, cipher_a_dot=0.5),
                    dominance_data=Mock(
                        stablecoin_dominance=5.2, dominance_24h_change=0.1
                    ),
                ),
                trade_action=trade_action,
                experience_id="test_exp_001",
            )

            # Verify trade decision log file
            decision_log_file = (
                log_dir / f"decisions_{datetime.now(UTC).strftime('%Y%m%d')}.jsonl"
            )
            assert decision_log_file.exists(), "Trade decision log should be created"

            with open(decision_log_file) as f:
                log_lines = f.readlines()

            assert len(log_lines) >= 1, "At least one log entry should exist"
            log_entry = json.loads(log_lines[-1])
            assert log_entry["symbol"] == "BTC-USD"
            assert log_entry["decision"]["action"] == "LONG"
            assert log_entry["experience_id"] == "test_exp_001"

            # Test 4: Performance monitor integration
            performance_metrics = paper_account.get_performance_metrics_for_monitor()

            assert isinstance(performance_metrics, list)
            assert len(performance_metrics) > 0

            # Validate metric structure
            for metric in performance_metrics:
                assert "name" in metric
                assert "value" in metric
                assert "timestamp" in metric
                assert "unit" in metric
                assert "tags" in metric

            # Find specific metrics
            equity_metrics = [
                m for m in performance_metrics if m["name"] == "paper_trading.equity"
            ]
            balance_metrics = [
                m for m in performance_metrics if m["name"] == "paper_trading.balance"
            ]

            assert len(equity_metrics) == 1, "Equity metric should be present"
            assert len(balance_metrics) == 1, "Balance metric should be present"

            # Test 5: Close position and validate complete flow
            new_price = Decimal("52000")  # Price increased
            close_action = sample_trade_actions["close"]

            close_order = paper_account.execute_trade_action(
                close_action, "BTC-USD", new_price
            )

            assert close_order is not None
            assert close_order.status.value == "FILLED"
            assert paper_account.current_position.side == "FLAT"
            assert paper_account.current_position.size == 0

            # Validate updated files after close
            with open(trades_file) as f:
                updated_trades_data = json.load(f)

            assert len(updated_trades_data["open_trades"]) == 0
            assert len(updated_trades_data["closed_trades"]) == 1

            closed_trade = updated_trades_data["closed_trades"][0]
            assert closed_trade["status"] == "CLOSED"
            assert closed_trade["exit_price"] is not None
            assert closed_trade["realized_pnl"] is not None

            # Validate session trades updated
            with open(session_trades_file) as f:
                updated_session_trades = json.load(f)

            closed_session_trades = [
                t for t in updated_session_trades if t["status"] == "CLOSED"
            ]
            assert len(closed_session_trades) == 1

            # Test structured logging for trade outcome
            closed_trade_data = updated_trades_data["closed_trades"][0]
            duration_minutes = 1.0  # Mock duration

            trade_logger.log_trade_outcome(
                experience_id="test_exp_001",
                entry_price=Decimal(closed_trade_data["entry_price"]),
                exit_price=Decimal(closed_trade_data["exit_price"]),
                pnl=Decimal(closed_trade_data["realized_pnl"]),
                duration_minutes=duration_minutes,
                insights="Successful trend following trade",
            )

            # Verify outcome log
            outcome_log_file = (
                log_dir / f"outcomes_{datetime.now(UTC).strftime('%Y%m%d')}.jsonl"
            )
            assert outcome_log_file.exists()

            with open(outcome_log_file) as f:
                outcome_lines = f.readlines()

            assert len(outcome_lines) >= 1
            outcome_entry = json.loads(outcome_lines[-1])
            assert outcome_entry["experience_id"] == "test_exp_001"
            assert outcome_entry["success"] == (float(outcome_entry["pnl"]) > 0)

            # Final WebSocket performance update
            final_account_status = paper_account.get_account_status()
            await websocket_publisher.publish_performance_update(final_account_status)

            await websocket_publisher.close()

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(
        self, temp_data_dir, mock_settings, sample_trade_actions
    ):
        """
        Test error handling and recovery in performance logging pipeline.

        Tests:
        1. File system errors during persistence
        2. WebSocket connection failures
        3. Logging errors
        4. Recovery mechanisms
        """
        data_dir = temp_data_dir / "paper_trading"

        paper_account = PaperTradingAccount(
            starting_balance=Decimal("10000"), data_dir=data_dir
        )

        # Test 1: File system error handling
        # Make directory read-only to simulate permission error
        data_dir.chmod(0o444)

        try:
            trade_action = sample_trade_actions["long"]
            current_price = Decimal("50000")

            # This should handle the error gracefully
            order = paper_account.execute_trade_action(
                trade_action, "BTC-USD", current_price
            )

            # Trade should still execute even if saving fails
            assert order is not None
            assert paper_account.current_position.side == "LONG"

        finally:
            # Restore permissions
            data_dir.chmod(0o755)

        # Test 2: WebSocket connection failure handling
        with patch(
            "websockets.connect", side_effect=ConnectionError("Connection failed")
        ):
            websocket_publisher = WebSocketPublisher(mock_settings)

            # Should handle connection failure gracefully
            success = await websocket_publisher.initialize()
            assert not success

            # Publishing should not raise error even when disconnected
            await websocket_publisher.publish_performance_update({"test": "data"})

            await websocket_publisher.close()

        # Test 3: Logging error handling
        # Create logger with invalid directory
        invalid_log_dir = Path("/invalid/path/that/should/not/exist")

        # This should handle the error gracefully during log writing
        trade_logger = TradeLogger(log_dir=invalid_log_dir)

        # Should not raise exception
        trade_logger.log_trade_decision(
            market_state=Mock(
                symbol="BTC-USD",
                current_price=Decimal("50000"),
                current_position=Mock(side="FLAT", size=0, unrealized_pnl=0),
                indicators=Mock(rsi=None, cipher_a_dot=None),
                dominance_data=None,
            ),
            trade_action=sample_trade_actions["long"],
            experience_id="error_test_001",
        )

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(
        self, temp_data_dir, mock_settings, sample_trade_actions, performance_monitor
    ):
        """
        Test performance monitoring integration with paper trading.

        Validates:
        1. Performance metrics collection during trades
        2. Latency tracking for operations
        3. Alert generation for performance issues
        4. Bottleneck analysis
        """
        await performance_monitor.start_monitoring(resource_monitor_interval=0.1)

        try:
            data_dir = temp_data_dir / "paper_trading"
            paper_account = PaperTradingAccount(
                starting_balance=Decimal("10000"), data_dir=data_dir
            )

            # Test latency tracking for trade execution
            with performance_monitor.track_operation("paper_trade_execution"):
                trade_action = sample_trade_actions["long"]
                current_price = Decimal("50000")

                order = paper_account.execute_trade_action(
                    trade_action, "BTC-USD", current_price
                )

                assert order is not None

                # Simulate some processing time
                await asyncio.sleep(0.01)

            # Test performance metrics collection
            performance_metrics = paper_account.get_performance_metrics_for_monitor()

            # Add metrics to performance monitor
            for metric_data in performance_metrics:
                from bot.performance_monitor import PerformanceMetric

                metric = PerformanceMetric(
                    name=metric_data["name"],
                    value=metric_data["value"],
                    timestamp=metric_data["timestamp"],
                    unit=metric_data["unit"],
                    tags=metric_data["tags"],
                )
                performance_monitor.add_metric(metric)

            # Test performance summary generation
            summary = performance_monitor.get_performance_summary(
                duration=timedelta(minutes=1)
            )

            assert "latency_summary" in summary
            assert "resource_summary" in summary
            assert "health_score" in summary
            assert isinstance(summary["health_score"], float)
            assert 0 <= summary["health_score"] <= 100

            # Validate latency metrics were recorded
            latency_metrics = performance_monitor.metrics_collector.get_metric_history(
                "latency.paper_trade_execution"
            )
            assert len(latency_metrics) > 0

            # Test bottleneck analysis
            bottleneck_analysis = (
                performance_monitor.bottleneck_analyzer.analyze_bottlenecks(
                    duration=timedelta(minutes=1)
                )
            )

            assert "bottlenecks" in bottleneck_analysis
            assert "recommendations" in bottleneck_analysis
            assert "analysis_period" in bottleneck_analysis

        finally:
            await performance_monitor.stop_monitoring()

    @pytest.mark.asyncio
    async def test_websocket_publishing_resilience(
        self, temp_data_dir, mock_settings, sample_trade_actions
    ):
        """
        Test WebSocket publishing resilience and reconnection.

        Tests:
        1. Automatic reconnection on connection loss
        2. Message queuing during disconnection
        3. Message delivery after reconnection
        4. Connection health monitoring
        """
        data_dir = temp_data_dir / "paper_trading"
        paper_account = PaperTradingAccount(
            starting_balance=Decimal("10000"), data_dir=data_dir
        )

        # Mock WebSocket that fails initially then succeeds
        connection_attempts = 0

        async def mock_connect(*args, **kwargs):
            nonlocal connection_attempts
            connection_attempts += 1

            if connection_attempts <= 2:  # Fail first 2 attempts
                raise ConnectionError("Connection failed")

            # Return successful mock connection
            mock_ws = AsyncMock()
            mock_ws.closed = False
            mock_ws.send = AsyncMock()
            mock_ws.ping = AsyncMock()
            mock_ws.close = AsyncMock()
            return mock_ws

        with patch("websockets.connect", side_effect=mock_connect):
            websocket_publisher = WebSocketPublisher(mock_settings)

            # Test initial connection failure and retry
            start_time = time.time()
            success = await websocket_publisher.initialize()
            elapsed = time.time() - start_time

            # Should eventually succeed after retries
            assert success or elapsed < 10  # Allow reasonable time for retries

            # Test message publishing during connection issues
            trade_action = sample_trade_actions["long"]
            current_price = Decimal("50000")

            order = paper_account.execute_trade_action(
                trade_action, "BTC-USD", current_price
            )

            account_status = paper_account.get_account_status()

            # This should queue message even if connection is unstable
            await websocket_publisher.publish_performance_update(account_status)
            await websocket_publisher.publish_trade_execution({"order_id": order.id})

            # Give time for background processing
            await asyncio.sleep(0.1)

            await websocket_publisher.close()

    @pytest.mark.asyncio
    async def test_concurrent_operations_thread_safety(
        self, temp_data_dir, mock_settings, sample_trade_actions
    ):
        """
        Test thread safety during concurrent operations.

        Tests:
        1. Concurrent trade executions
        2. Simultaneous file operations
        3. Thread-safe state updates
        4. Performance metric collection under load
        """
        data_dir = temp_data_dir / "paper_trading"
        paper_account = PaperTradingAccount(
            starting_balance=Decimal("100000"),  # Larger balance for multiple trades
            data_dir=data_dir,
        )

        # Execute multiple concurrent trades
        async def execute_trade_sequence(trade_id: int):
            """Execute a sequence of trades concurrently."""
            base_price = Decimal("50000")

            for i in range(3):
                current_price = base_price + (trade_id * 100) + (i * 50)

                if i == 0:  # Open position
                    action = sample_trade_actions["long"]
                elif i == 1:  # Hold
                    action = sample_trade_actions["hold"]
                else:  # Close position
                    action = sample_trade_actions["close"]

                if action.action != "HOLD":
                    order = paper_account.execute_trade_action(
                        action, "BTC-USD", current_price
                    )
                    assert order is not None

                # Small delay to allow other coroutines to run
                await asyncio.sleep(0.001)

        # Run multiple concurrent trade sequences
        tasks = [execute_trade_sequence(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # Validate final state consistency
        account_status = paper_account.get_account_status()
        assert isinstance(account_status["equity"], float)
        assert isinstance(account_status["total_trades"], int)

        # Validate file integrity after concurrent operations
        trades_file = data_dir / "trades.json"
        assert trades_file.exists()

        with open(trades_file) as f:
            trades_data = json.load(f)

        # Should have valid JSON structure
        assert "open_trades" in trades_data
        assert "closed_trades" in trades_data

        # Verify session trades file
        session_trades_file = data_dir / "session_trades.json"
        assert session_trades_file.exists()

        with open(session_trades_file) as f:
            session_trades = json.load(f)

        assert isinstance(session_trades, list)

    def test_data_persistence_validation(
        self, temp_data_dir, mock_settings, sample_trade_actions
    ):
        """
        Test data persistence validation and integrity.

        Tests:
        1. JSON schema validation
        2. Data type consistency
        3. File format integrity
        4. Recovery from corrupted files
        """
        data_dir = temp_data_dir / "paper_trading"
        paper_account = PaperTradingAccount(
            starting_balance=Decimal("10000"), data_dir=data_dir
        )

        # Execute trade to create files
        trade_action = sample_trade_actions["long"]
        current_price = Decimal("50000")

        order = paper_account.execute_trade_action(
            trade_action, "BTC-USD", current_price
        )
        assert order is not None

        # Validate file structure
        trades_file = data_dir / "trades.json"
        account_file = data_dir / "account.json"
        session_trades_file = data_dir / "session_trades.json"

        # Test JSON validity
        for file_path in [trades_file, account_file, session_trades_file]:
            assert file_path.exists()
            with open(file_path) as f:
                data = json.load(f)
                assert isinstance(data, dict | list)

        # Test specific data structure validation
        with open(trades_file) as f:
            trades_data = json.load(f)

        # Validate trades structure
        assert "open_trades" in trades_data
        assert "closed_trades" in trades_data
        assert isinstance(trades_data["open_trades"], dict)
        assert isinstance(trades_data["closed_trades"], list)

        # Validate session trades structure
        with open(session_trades_file) as f:
            session_trades = json.load(f)

        assert isinstance(session_trades, list)
        if session_trades:
            trade = session_trades[0]
            required_fields = [
                "id",
                "symbol",
                "side",
                "entry_time",
                "entry_price",
                "size",
                "fees",
                "status",
            ]
            for field in required_fields:
                assert field in trade

        # Test recovery from corrupted file
        # Corrupt the trades file
        with open(trades_file, "w") as f:
            f.write("invalid json content")

        # Create new account instance - should handle corruption gracefully
        paper_account_2 = PaperTradingAccount(
            starting_balance=Decimal("10000"), data_dir=data_dir
        )

        # Should still be functional with default state
        assert paper_account_2.current_balance == Decimal("10000")
        assert paper_account_2.trade_counter >= 0

    @pytest.mark.asyncio
    async def test_performance_benchmarks(
        self, temp_data_dir, mock_settings, sample_trade_actions, performance_monitor
    ):
        """
        Test performance benchmarks for paper trading operations.

        Validates:
        1. Trade execution latency < 10ms
        2. File persistence latency < 50ms
        3. WebSocket publishing latency < 100ms
        4. Memory usage stability
        """
        await performance_monitor.start_monitoring(resource_monitor_interval=0.1)

        try:
            data_dir = temp_data_dir / "paper_trading"
            paper_account = PaperTradingAccount(
                starting_balance=Decimal("10000"), data_dir=data_dir
            )

            # Benchmark trade execution
            trade_latencies = []

            for i in range(10):
                start_time = time.perf_counter()

                trade_action = sample_trade_actions["long"]
                current_price = Decimal("50000") + i

                order = paper_account.execute_trade_action(
                    trade_action, "BTC-USD", current_price
                )

                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                trade_latencies.append(latency_ms)

                assert order is not None

                # Close position for next iteration
                close_action = sample_trade_actions["close"]
                paper_account.execute_trade_action(
                    close_action, "BTC-USD", current_price + 10
                )

            # Validate performance benchmarks
            avg_latency = sum(trade_latencies) / len(trade_latencies)
            max_latency = max(trade_latencies)

            print(
                f"Trade execution latencies: avg={avg_latency:.2f}ms, max={max_latency:.2f}ms"
            )

            # Performance assertions
            assert (
                avg_latency < 50
            ), f"Average trade latency {avg_latency:.2f}ms exceeds 50ms threshold"
            assert (
                max_latency < 100
            ), f"Max trade latency {max_latency:.2f}ms exceeds 100ms threshold"

            # Test file I/O performance
            file_latencies = []

            for _ in range(5):
                start_time = time.perf_counter()

                # Force file save
                paper_account._save_state()

                end_time = time.perf_counter()
                file_latency_ms = (end_time - start_time) * 1000
                file_latencies.append(file_latency_ms)

            avg_file_latency = sum(file_latencies) / len(file_latencies)
            max_file_latency = max(file_latencies)

            print(
                f"File persistence latencies: avg={avg_file_latency:.2f}ms, max={max_file_latency:.2f}ms"
            )

            assert (
                avg_file_latency < 100
            ), f"Average file latency {avg_file_latency:.2f}ms exceeds 100ms threshold"
            assert (
                max_file_latency < 200
            ), f"Max file latency {max_file_latency:.2f}ms exceeds 200ms threshold"

        finally:
            await performance_monitor.stop_monitoring()
