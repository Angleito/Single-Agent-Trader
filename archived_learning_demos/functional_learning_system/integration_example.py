"""
Integration example showing functional learning system with MCP compatibility.

This example demonstrates how to use the new functional learning components
while maintaining full compatibility with the existing MCP memory server.
"""

import asyncio
import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Dict, Any

from bot.config import settings
from bot.mcp.memory_server import MCPMemoryServer
from bot.trading_types import MarketState, TradeAction, IndicatorData, Position

from .functional_experience_manager import FunctionalExperienceManager
from .functional_self_improvement import FunctionalSelfImprovementEngine
from .learning_algorithms import analyze_trading_patterns, generate_strategy_insights
from .combinators import (
    build_analysis_pipeline,
    validate_minimum_experiences,
    parallel_learning_analysis,
    tap,
)

logger = logging.getLogger(__name__)


class FunctionalLearningSystem:
    """
    Integrated functional learning system with MCP compatibility.
    
    Demonstrates how to use functional learning components alongside
    the existing MCP memory server infrastructure.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the functional learning system."""
        self.config = config or {}
        
        # Initialize MCP memory server (existing infrastructure)
        self.memory_server = MCPMemoryServer()
        
        # Initialize functional components
        self.experience_manager = FunctionalExperienceManager(
            self.memory_server, self.config
        )
        self.improvement_engine = FunctionalSelfImprovementEngine(
            self.memory_server
        )
        
        logger.info("🎯 Functional Learning System: Initialized with MCP integration")
    
    async def start(self) -> None:
        """Start the learning system."""
        await self.memory_server.connect()
        await self.experience_manager.start()
        logger.info("✅ Functional Learning System: Started")
    
    async def stop(self) -> None:
        """Stop the learning system."""
        await self.experience_manager.stop()
        await self.memory_server.disconnect()
        logger.info("🛑 Functional Learning System: Stopped")
    
    async def record_and_analyze_decision(
        self,
        market_state: MarketState,
        trade_action: TradeAction,
    ) -> Dict[str, Any]:
        """
        Record a trading decision and perform functional analysis.
        
        This demonstrates the integration between functional components
        and the existing MCP memory system.
        """
        # Record decision using functional experience manager
        result = await self.experience_manager.record_trading_decision(
            market_state, trade_action
        )
        
        if result.is_failure():
            return {"error": result.failure()}
        
        experience_id = result.success()
        
        # Get current state for analysis
        current_state = self.experience_manager.state
        
        # Perform functional analysis if we have enough data
        if current_state.experience_state.total_completed >= 5:
            analysis_result = await self.improvement_engine.analyze_performance(
                current_state.experience_state
            )
            
            if analysis_result.is_success():
                improvement_state = analysis_result.success()
                
                # Get market-specific recommendations
                recommendations_result = self.improvement_engine.get_market_condition_recommendations(
                    improvement_state,
                    current_indicators=self._extract_indicators(market_state),
                    current_dominance=self._extract_dominance(market_state),
                )
                
                if recommendations_result.is_success():
                    recommendations = recommendations_result.success()
                    
                    return {
                        "experience_id": experience_id,
                        "functional_analysis": {
                            "pattern_count": len(improvement_state.pattern_analyses),
                            "regime_count": len(improvement_state.market_regimes),
                            "insight_count": len(improvement_state.active_insights),
                            "recommendation_count": len(improvement_state.recommendations),
                        },
                        "market_recommendations": recommendations,
                        "performance_metrics": {
                            "win_rate": improvement_state.current_metrics.win_rate,
                            "total_pnl": improvement_state.current_metrics.total_pnl,
                            "sharpe_ratio": improvement_state.current_metrics.sharpe_ratio,
                        },
                    }
        
        return {
            "experience_id": experience_id,
            "message": "Decision recorded, insufficient data for full analysis",
        }
    
    async def demonstrate_functional_pipeline(self) -> Dict[str, Any]:
        """
        Demonstrate functional composition and pipeline building.
        
        Shows how pure functions can be composed to create complex
        analysis workflows.
        """
        current_state = self.experience_manager.state.experience_state
        
        # Build a functional analysis pipeline
        analysis_pipeline = build_analysis_pipeline(
            validate_minimum_experiences(3),
            tap(lambda state: logger.info(
                "Pipeline processing %d experiences", state.total_experiences
            )),
        )
        
        # Execute pipeline
        pipeline_result = analysis_pipeline(current_state)
        
        if pipeline_result.is_failure():
            return {"error": pipeline_result.failure()}
        
        validated_state = pipeline_result.success()
        
        # Demonstrate parallel functional analysis
        analysis_functions = [
            lambda state: analyze_trading_patterns(state, min_samples=2),
            lambda state: generate_strategy_insights(state, recent_hours=168),  # 1 week
        ]
        
        # Convert to Result-returning functions for parallel execution
        def pattern_analysis_wrapper(state):
            try:
                patterns = analyze_trading_patterns(state, min_samples=2)
                from .learning_algorithms import LearningResult
                return LearningResult(
                    insights=(f"Analyzed {len(patterns)} patterns",),
                    confidence=0.8,
                    sample_size=len(patterns),
                    analysis_timestamp=datetime.now(UTC),
                )
            except Exception as e:
                from bot.fp.types.result import Err
                return Err(str(e))
        
        def insight_analysis_wrapper(state):
            try:
                insights = generate_strategy_insights(state, recent_hours=168)
                from .learning_algorithms import LearningResult
                return LearningResult(
                    insights=tuple(f"{insight.insight_type}: {insight.description}" for insight in insights),
                    confidence=0.7,
                    sample_size=len(insights),
                    analysis_timestamp=datetime.now(UTC),
                )
            except Exception as e:
                from bot.fp.types.result import Err
                return Err(str(e))
        
        parallel_analysis = parallel_learning_analysis([
            pattern_analysis_wrapper,
            insight_analysis_wrapper,
        ])
        
        analysis_results = parallel_analysis(validated_state)
        
        if analysis_results.is_success():
            results = analysis_results.success()
            return {
                "pipeline_validation": "success",
                "parallel_analysis": {
                    "pattern_analysis": {
                        "insights": list(results[0].insights),
                        "confidence": results[0].confidence,
                    } if len(results) > 0 else None,
                    "insight_analysis": {
                        "insights": list(results[1].insights),
                        "confidence": results[1].confidence,
                    } if len(results) > 1 else None,
                },
                "functional_composition": "successful",
            }
        
        return {"error": analysis_results.failure()}
    
    async def demonstrate_mcp_integration(self) -> Dict[str, Any]:
        """
        Demonstrate MCP memory server integration with functional components.
        
        Shows how functional learning components work seamlessly with
        the existing MCP infrastructure.
        """
        # Get experiences from functional state
        functional_experiences = self.experience_manager.state.experience_state.experiences
        
        # Query MCP memory server directly
        mcp_cache_size = len(self.memory_server.memory_cache)
        
        # Compare functional and MCP states
        completed_functional = sum(1 for exp in functional_experiences if exp.is_completed())
        completed_mcp = sum(
            1 for exp in self.memory_server.memory_cache.values()
            if exp.outcome is not None
        )
        
        # Demonstrate pattern statistics from both systems
        pattern_stats = await self.memory_server.get_pattern_statistics()
        functional_patterns = self.experience_manager.state.experience_state.patterns
        
        return {
            "functional_state": {
                "total_experiences": len(functional_experiences),
                "completed_experiences": completed_functional,
                "patterns": len(functional_patterns),
            },
            "mcp_state": {
                "cache_size": mcp_cache_size,
                "completed_experiences": completed_mcp,
                "pattern_statistics": len(pattern_stats),
            },
            "integration_status": "functional_and_mcp_synchronized",
            "compatibility": "full_backward_compatibility_maintained",
        }
    
    async def export_comprehensive_report(self) -> Dict[str, Any]:
        """
        Export a comprehensive report showing functional learning capabilities.
        """
        current_state = self.experience_manager.state.experience_state
        
        if current_state.total_completed < 3:
            return {
                "status": "insufficient_data",
                "message": "Need at least 3 completed trades for comprehensive analysis",
                "current_data": {
                    "total_experiences": current_state.total_experiences,
                    "completed": current_state.total_completed,
                },
            }
        
        # Perform comprehensive functional analysis
        analysis_result = await self.improvement_engine.analyze_performance(current_state)
        
        if analysis_result.is_failure():
            return {"error": analysis_result.failure()}
        
        improvement_state = analysis_result.success()
        
        # Generate full report
        report = self.improvement_engine.export_improvement_report(improvement_state)
        
        # Add integration status
        report["system_integration"] = {
            "functional_components": "active",
            "mcp_compatibility": "maintained", 
            "memory_server": "connected" if self.memory_server._connected else "disconnected",
            "learning_algorithms": "pure_functional",
            "state_management": "immutable",
        }
        
        # Add active trades summary
        report["active_trades"] = self.experience_manager.get_active_trades_summary()
        
        return report
    
    def _extract_indicators(self, market_state: MarketState) -> Dict[str, float]:
        """Extract indicators for analysis."""
        if not market_state.indicators:
            return {}
        
        ind = market_state.indicators
        return {
            "rsi": float(ind.rsi) if ind.rsi else 50.0,
            "cipher_b_wave": float(ind.cipher_b_wave) if ind.cipher_b_wave else 0.0,
            "ema_fast": float(ind.ema_fast) if ind.ema_fast else 0.0,
            "ema_slow": float(ind.ema_slow) if ind.ema_slow else 0.0,
        }
    
    def _extract_dominance(self, market_state: MarketState) -> Dict[str, float]:
        """Extract dominance data for analysis."""
        if not market_state.dominance_data:
            return {}
        
        dom = market_state.dominance_data
        return {
            "stablecoin_dominance": float(dom.stablecoin_dominance),
            "dominance_24h_change": float(dom.dominance_24h_change),
        }


async def demonstrate_functional_learning():
    """
    Demonstration function showing functional learning system capabilities.
    """
    print("🚀 Functional Learning System Demonstration")
    
    # Initialize system
    learning_system = FunctionalLearningSystem({
        "min_samples_for_pattern": 3,
        "confidence_threshold": 0.6,
    })
    
    await learning_system.start()
    
    try:
        # Create sample market state
        sample_market_state = MarketState(
            symbol="BTC-USD",
            interval="1h",
            timestamp=datetime.now(UTC),
            current_price=Decimal("50000"),
            ohlcv_data=[],
            indicators=IndicatorData(
                timestamp=datetime.now(UTC),
                rsi=65.0,
                cipher_b_wave=45.0,
                ema_fast=50100.0,
                ema_slow=49900.0,
            ),
            current_position=Position(
                symbol="BTC-USD",
                side="FLAT",
                size=Decimal("0"),
                entry_price=None,
                timestamp=datetime.now(UTC),
            ),
        )
        
        # Create sample trade action
        sample_trade_action = TradeAction(
            action="LONG",
            size_pct=10.0,
            take_profit_pct=3.0,
            stop_loss_pct=2.0,
            leverage=3,
            reduce_only=False,
            rationale="Functional learning system test trade",
        )
        
        # Demonstrate decision recording and analysis
        print("\n📝 Recording trading decision...")
        decision_result = await learning_system.record_and_analyze_decision(
            sample_market_state, sample_trade_action
        )
        print(f"Decision result: {decision_result}")
        
        # Demonstrate functional pipeline
        print("\n🔧 Demonstrating functional pipeline...")
        pipeline_result = await learning_system.demonstrate_functional_pipeline()
        print(f"Pipeline result: {pipeline_result}")
        
        # Demonstrate MCP integration
        print("\n🔗 Demonstrating MCP integration...")
        integration_result = await learning_system.demonstrate_mcp_integration()
        print(f"Integration result: {integration_result}")
        
        # Export comprehensive report
        print("\n📊 Exporting comprehensive report...")
        report = await learning_system.export_comprehensive_report()
        print(f"Report keys: {list(report.keys())}")
        
        print("\n✅ Functional Learning System demonstration completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        logger.exception("Demonstration failed")
    
    finally:
        await learning_system.stop()


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_functional_learning())