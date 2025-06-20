"""
Structured trade logging for MCP memory integration.

Provides detailed logging of trading decisions, market context,
memory queries, and trade outcomes for analysis and debugging.
"""

import json
import logging
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from bot.trading_types import MarketState, TradeAction


class TradeLogger:
    """
    Specialized logger for structured trade and memory events.

    Logs trading activities in both human-readable and JSON formats
    for easy analysis and debugging of the memory-enhanced trading system.
    """

    def __init__(self, log_dir: Path | None = None):
        """Initialize the trade logger."""
        self.log_dir = log_dir or Path("logs/trades")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup loggers
        self.logger = logging.getLogger("bot.trades")
        self.memory_logger = logging.getLogger("bot.memory")

        # JSON log files
        self.decisions_file = (
            self.log_dir / f"decisions_{datetime.now(UTC).strftime('%Y%m%d')}.jsonl"
        )
        self.outcomes_file = (
            self.log_dir / f"outcomes_{datetime.now(UTC).strftime('%Y%m%d')}.jsonl"
        )
        self.memory_file = (
            self.log_dir / f"memory_{datetime.now(UTC).strftime('%Y%m%d')}.jsonl"
        )

    def log_trade_decision(
        self,
        market_state: MarketState,
        trade_action: TradeAction,
        experience_id: str | None = None,
        memory_context: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a trading decision with full context.

        Args:
            market_state: Current market conditions
            trade_action: Decision made by the system
            experience_id: ID for memory tracking
            memory_context: Past experiences used in decision
        """
        timestamp = datetime.now(UTC)

        # Extract key metrics
        indicators = {}
        if market_state.indicators:
            indicators = {
                "rsi": (
                    float(market_state.indicators.rsi)
                    if market_state.indicators.rsi
                    else None
                ),
                "cipher_a_dot": (
                    float(market_state.indicators.cipher_a_dot)
                    if market_state.indicators.cipher_a_dot
                    else None
                ),
                "cipher_b_wave": (
                    float(market_state.indicators.cipher_b_wave)
                    if market_state.indicators.cipher_b_wave
                    else None
                ),
                "cipher_b_money_flow": (
                    float(market_state.indicators.cipher_b_money_flow)
                    if market_state.indicators.cipher_b_money_flow
                    else None
                ),
                "ema_trend": (
                    "UP"
                    if (
                        market_state.indicators.ema_fast
                        and market_state.indicators.ema_slow
                        and market_state.indicators.ema_fast
                        > market_state.indicators.ema_slow
                    )
                    else "DOWN"
                ),
            }

        dominance = {}
        if market_state.dominance_data:
            dominance = {
                "stablecoin_dominance": float(
                    market_state.dominance_data.stablecoin_dominance
                ),
                "dominance_24h_change": float(
                    market_state.dominance_data.dominance_24h_change
                ),
            }

        # Create structured log entry
        decision_log = {
            "timestamp": timestamp.isoformat(),
            "experience_id": experience_id,
            "symbol": market_state.symbol,
            "price": float(market_state.current_price),
            "position": {
                "side": market_state.current_position.side,
                "size": float(market_state.current_position.size),
                "unrealized_pnl": (
                    float(market_state.current_position.unrealized_pnl)
                    if market_state.current_position.unrealized_pnl
                    else 0
                ),
            },
            "decision": {
                "action": trade_action.action,
                "size_pct": trade_action.size_pct,
                "rationale": trade_action.rationale,
                "leverage": trade_action.leverage,
            },
            "indicators": indicators,
            "dominance": dominance,
            "memory_used": memory_context is not None,
            "similar_experiences": (
                len(memory_context.get("experiences", [])) if memory_context else 0
            ),
        }

        # Log to file
        self._append_json_log(self.decisions_file, decision_log)

        # Human-readable log
        self.logger.info(
            f"Trade Decision: {trade_action.action} {market_state.symbol} @ ${market_state.current_price} | "
            f"RSI={indicators.get('rsi', 'N/A')} | "
            f"Wave={indicators.get('cipher_b_wave', 'N/A')} | "
            f"Memory={decision_log['similar_experiences']} experiences | "
            f"ID={experience_id}"
        )

        # Log detailed rationale if available
        if trade_action.rationale:
            self.logger.info(f"Rationale: {trade_action.rationale[:200]}...")

    def log_memory_query(
        self,
        query_params: dict[str, Any],
        results: list[dict[str, Any]],
        execution_time_ms: float,
    ) -> None:
        """
        Log memory query operations.

        Args:
            query_params: Parameters used for the query
            results: Similar experiences found
            execution_time_ms: Query execution time
        """
        memory_log = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": "query",
            "query": query_params,
            "results_count": len(results),
            "execution_time_ms": execution_time_ms,
            "top_similarities": [r.get("similarity", 0) for r in results[:3]],
        }

        self._append_json_log(self.memory_file, memory_log)

        self.memory_logger.debug(
            f"Memory query returned {len(results)} experiences in {execution_time_ms:.1f}ms"
        )

    def log_trade_outcome(
        self,
        experience_id: str,
        entry_price: Decimal,
        exit_price: Decimal,
        pnl: Decimal,
        duration_minutes: float,
        insights: str | None = None,
    ) -> None:
        """
        Log the outcome of a completed trade.

        Args:
            experience_id: Memory experience ID
            entry_price: Price at trade entry
            exit_price: Price at trade exit
            pnl: Realized profit/loss
            duration_minutes: How long position was held
            insights: Learned insights from the trade
        """
        outcome_log = {
            "timestamp": datetime.now(UTC).isoformat(),
            "experience_id": experience_id,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "pnl": float(pnl),
            "pnl_pct": float((exit_price - entry_price) / entry_price * 100),
            "duration_minutes": duration_minutes,
            "success": pnl > 0,
            "insights": insights,
        }

        self._append_json_log(self.outcomes_file, outcome_log)

        # Human-readable summary
        pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
        self.logger.info(
            f"Trade Completed: {experience_id} | "
            f"Entry=${entry_price} Exit=${exit_price} | "
            f"PnL={pnl_str} ({outcome_log['pnl_pct']:.1f}%) | "
            f"Duration={duration_minutes:.0f}min"
        )

        if insights:
            self.logger.info(f"Insights: {insights}")

    def log_pattern_statistics(self, pattern_stats: dict[str, dict[str, Any]]) -> None:
        """
        Log pattern performance statistics.

        Args:
            pattern_stats: Statistics for identified patterns
        """
        stats_log = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": "pattern_stats",
            "patterns": pattern_stats,
            "total_patterns": len(pattern_stats),
        }

        self._append_json_log(self.memory_file, stats_log)

        # Log top performing patterns
        sorted_patterns = sorted(
            pattern_stats.items(),
            key=lambda x: x[1].get("success_rate", 0) * x[1].get("count", 0),
            reverse=True,
        )

        for pattern, stats in sorted_patterns[:5]:
            self.memory_logger.info(
                f"Pattern '{pattern}': {stats['success_rate']:.1%} win rate "
                f"({stats['count']} trades, avg PnL=${stats['avg_pnl']:.2f})"
            )

    def log_position_update(
        self,
        trade_id: str,
        current_price: Decimal,
        unrealized_pnl: Decimal,
        max_favorable: Decimal,
        max_adverse: Decimal,
    ) -> None:
        """
        Log position progress updates.

        Args:
            trade_id: Active trade identifier
            current_price: Current market price
            unrealized_pnl: Current unrealized P&L
            max_favorable: Maximum profit seen
            max_adverse: Maximum loss seen
        """
        self.logger.debug(
            f"Position Update: {trade_id} | "
            f"Price=${current_price} | "
            f"Unrealized PnL=${unrealized_pnl:.2f} | "
            f"Max Profit=${max_favorable:.2f} | "
            f"Max Loss=${max_adverse:.2f}"
        )

    def log_memory_storage(
        self,
        experience_id: str,
        action: str,
        patterns: list[str],
        storage_location: str = "local",
    ) -> None:
        """
        Log memory storage operations.

        Args:
            experience_id: Experience being stored
            action: Trade action taken
            patterns: Identified market patterns
            storage_location: Where memory was stored
        """
        storage_log = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": "storage",
            "experience_id": experience_id,
            "action": action,
            "patterns": patterns,
            "location": storage_location,
        }

        self._append_json_log(self.memory_file, storage_log)

        self.memory_logger.debug(
            f"Stored experience {experience_id} with patterns: {', '.join(patterns)}"
        )

    def _append_json_log(self, file_path: Path, data: dict[str, Any]) -> None:
        """Append JSON data to log file."""
        try:
            with open(file_path, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            self.logger.exception(f"Failed to write JSON log: {e}")
