"""
RAG (Retrieval Augmented Generation) reader for integrating trading knowledge.

This module provides functionality to integrate external trading knowledge
and educational content to enhance LLM decision-making capabilities.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RAGReader:
    """
    RAG reader for integrating trading knowledge and educational content.

    Provides capabilities to load, index, and retrieve relevant trading
    knowledge to augment LLM decision-making with domain expertise.
    """

    def __init__(self, knowledge_dir: str = "knowledge"):
        """
        Initialize the RAG reader.

        Args:
            knowledge_dir: Directory containing trading knowledge files
        """
        self.knowledge_dir = Path(knowledge_dir)
        self.documents: list[dict[str, Any]] = []
        self.embeddings_cache: dict[str, Any] = {}

        # Create knowledge directory if it doesn't exist
        self.knowledge_dir.mkdir(exist_ok=True)

        logger.info(f"Initialized RAGReader with knowledge dir: {self.knowledge_dir}")

    async def load_knowledge_base(self) -> bool:
        """
        Load trading knowledge from files in the knowledge directory.

        Returns:
            True if knowledge base loaded successfully
        """
        try:
            self.documents = []

            # Load different types of knowledge files
            await self._load_trading_strategies()
            await self._load_market_analysis()
            await self._load_risk_guidelines()
            await self._load_indicator_docs()

            logger.info(f"Loaded {len(self.documents)} knowledge documents")
            return True

        except Exception as e:
            logger.exception(f"Failed to load knowledge base: {e}")
            return False

    async def _load_trading_strategies(self) -> None:
        """Load trading strategy documents."""
        strategies_file = self.knowledge_dir / "trading_strategies.json"

        if not strategies_file.exists():
            # Create default strategies file
            default_strategies = {
                "trend_following": {
                    "description": (
                        "Follow the primary trend direction using EMAs and "
                        "momentum indicators"
                    ),
                    "entry_criteria": [
                        "EMA fast > EMA slow for uptrend",
                        "RSI between 40-60 for momentum",
                        "Volume confirmation",
                    ],
                    "exit_criteria": [
                        "EMA crossover reversal",
                        "RSI extreme levels (>80 or <20)",
                        "Volume divergence",
                    ],
                    "risk_management": ("Use 1.5% stop loss, 2-3% take profit"),
                },
                "mean_reversion": {
                    "description": (
                        "Trade against extreme price movements expecting "
                        "return to mean"
                    ),
                    "entry_criteria": [
                        "RSI < 30 for oversold bounce",
                        "RSI > 70 for overbought pullback",
                        "Price near Bollinger Band extremes",
                    ],
                    "exit_criteria": [
                        "RSI returns to 50 level",
                        "Price reaches opposite Bollinger Band",
                        "Time-based exit (4-8 hours)",
                    ],
                    "risk_management": (
                        "Use tight stops (0.5-1%), quick profits (1-2%)"
                    ),
                },
            }

            with open(strategies_file, "w") as f:
                json.dump(default_strategies, f, indent=2)

        # Load strategies
        try:
            with open(strategies_file) as f:
                strategies = json.load(f)

            for name, strategy in strategies.items():
                self.documents.append(
                    {
                        "type": "strategy",
                        "title": name,
                        "content": strategy,
                        "tags": ["strategy", "trading", name],
                    }
                )

        except Exception as e:
            logger.warning(f"Failed to load trading strategies: {e}")

    async def _load_market_analysis(self) -> None:
        """Load market analysis guidelines."""
        analysis_content = {
            "market_structure": {
                "uptrend_characteristics": [
                    "Higher highs and higher lows",
                    "Rising EMAs in proper order",
                    "Increasing volume on advances",
                    "RSI maintaining above 40",
                ],
                "downtrend_characteristics": [
                    "Lower highs and lower lows",
                    "Declining EMAs in proper order",
                    "Increasing volume on declines",
                    "RSI maintaining below 60",
                ],
                "sideways_characteristics": [
                    "Equal highs and lows",
                    "Choppy EMA behavior",
                    "Decreasing volume",
                    "RSI oscillating around 50",
                ],
            },
            "support_resistance": {
                "identification": (
                    "Previous swing highs/lows, round numbers, VWAP levels"
                ),
                "trading_rules": "Buy near support, sell near resistance",
                "breakout_confirmation": "Volume increase + follow-through",
            },
        }

        self.documents.append(
            {
                "type": "analysis",
                "title": "market_analysis_guidelines",
                "content": analysis_content,
                "tags": ["analysis", "market", "structure"],
            }
        )

    async def _load_risk_guidelines(self) -> None:
        """Load risk management guidelines."""
        risk_content = {
            "position_sizing": {
                "max_risk_per_trade": "2% of account",
                "max_portfolio_risk": "6% of account",
                "correlation_limits": ("No more than 3 correlated positions"),
            },
            "stop_loss_rules": {
                "trend_trades": "1.5-2% stop loss",
                "counter_trend": "0.5-1% stop loss",
                "breakout_trades": "Below breakout level",
            },
            "take_profit_rules": {
                "minimum_rr": "1:1.5 risk-reward ratio",
                "trend_targets": "Previous swing levels",
                "partial_profits": "Take 50% at 1:1, let rest run",
            },
        }

        self.documents.append(
            {
                "type": "risk",
                "title": "risk_management_guidelines",
                "content": risk_content,
                "tags": ["risk", "management", "guidelines"],
            }
        )

    async def _load_indicator_docs(self) -> None:
        """Load technical indicator documentation."""
        indicator_content = {
            "cipher_indicators": {
                "cipher_a": {
                    "purpose": "Trend identification and momentum",
                    "signals": {
                        "bullish": ("Trend dot above zero, RSI recovery from oversold"),
                        "bearish": (
                            "Trend dot below zero, RSI decline from overbought"
                        ),
                    },
                    "best_timeframes": ["5m", "15m", "1h"],
                },
                "cipher_b": {
                    "purpose": "Money flow and wave analysis",
                    "signals": {
                        "bullish": "Wave crossing above zero, money flow > 60",
                        "bearish": "Wave crossing below zero, money flow < 40",
                    },
                    "divergence": "Watch for price/wave divergences",
                },
            },
            "supporting_indicators": {
                "rsi": {
                    "oversold": 30,
                    "overbought": 70,
                    "trend_zones": "40-60 in trending markets",
                },
                "ema": {
                    "fast": 9,
                    "slow": 21,
                    "trend": "Fast above slow = uptrend",
                },
                "vwap": {
                    "purpose": "Average price weighted by volume",
                    "usage": "Support/resistance, institutional levels",
                },
            },
        }

        self.documents.append(
            {
                "type": "indicators",
                "title": "technical_indicators_guide",
                "content": indicator_content,
                "tags": [
                    "indicators",
                    "technical",
                    "cipher",
                    "rsi",
                    "ema",
                ],
            }
        )

    def search_knowledge(
        self, query: str, max_results: int = 3
    ) -> list[dict[str, Any]]:
        """
        Search knowledge base for relevant information.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of relevant knowledge documents
        """
        if not self.documents:
            return []

        # Simple text-based search (could be enhanced with embeddings)
        query_lower = query.lower()
        scored_docs: list[tuple[int, dict[str, Any]]] = []

        for doc in self.documents:
            score = 0

            # Check title match
            if query_lower in doc["title"].lower():
                score += 10

            # Check tags match
            for tag in doc.get("tags", []):
                if query_lower in tag.lower():
                    score += 5

            # Check content match (simplified)
            content_str = json.dumps(doc["content"]).lower()
            if query_lower in content_str:
                score += 3

            if score > 0:
                scored_docs.append((score, doc))

        # Sort by score and return top results
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:max_results]]

    def get_context_for_decision(self, market_conditions: dict[str, Any]) -> str:
        """
        Get relevant context for trading decision based on market conditions.

        Args:
            market_conditions: Current market state and indicators

        Returns:
            Formatted context string for LLM
        """
        context_parts: list[str] = []

        # Determine market condition
        trend_direction = self._analyze_trend(market_conditions)

        # Get relevant strategy advice
        if trend_direction in ("uptrend", "downtrend"):
            strategies = self.search_knowledge("trend_following")
        else:
            strategies = self.search_knowledge("mean_reversion")

        # Get risk management guidance
        risk_docs = self.search_knowledge("risk_management")

        # Get indicator interpretation
        indicator_docs = self.search_knowledge("cipher_indicators")

        # Format context
        if strategies:
            context_parts.append("Strategy Guidance:")
            for doc in strategies[:1]:  # Take top strategy
                desc = doc["content"].get("description", "")
                context_parts.append(f"- {doc['title']}: {desc}")

        if risk_docs:
            context_parts.append("\\nRisk Management:")
            risk_content = risk_docs[0]["content"]
            max_risk = risk_content.get("position_sizing", {}).get(
                "max_risk_per_trade", "2%"
            )
            context_parts.append(f"- Max risk per trade: {max_risk}")

        if indicator_docs:
            context_parts.append("\\nIndicator Interpretation:")
            cipher_content = indicator_docs[0]["content"].get("cipher_indicators", {})
            cipher_a_purpose = cipher_content.get("cipher_a", {}).get("purpose", "")
            context_parts.append(f"- Cipher A: {cipher_a_purpose}")
            cipher_b_purpose = cipher_content.get("cipher_b", {}).get("purpose", "")
            context_parts.append(f"- Cipher B: {cipher_b_purpose}")

        return "\\n".join(context_parts)

    def _analyze_trend(self, market_conditions: dict[str, Any]) -> str:
        """
        Analyze trend direction from market conditions.

        Args:
            market_conditions: Market state data

        Returns:
            Trend direction: 'uptrend', 'downtrend', or 'sideways'
        """
        # Simple trend analysis based on EMAs
        ema_fast = market_conditions.get("ema_fast")
        ema_slow = market_conditions.get("ema_slow")

        if ema_fast and ema_slow:
            if ema_fast > ema_slow * 1.002:  # 0.2% threshold
                return "uptrend"
            elif ema_fast < ema_slow * 0.998:
                return "downtrend"

        return "sideways"

    def add_knowledge_document(
        self,
        doc_type: str,
        title: str,
        content: dict[str, Any],
        tags: list[str] | None = None,
    ) -> None:
        """
        Add a new knowledge document to the knowledge base.

        Args:
            doc_type: Type of document
            title: Document title
            content: Document content
            tags: Associated tags
        """
        document = {
            "type": doc_type,
            "title": title,
            "content": content,
            "tags": tags or [],
        }

        self.documents.append(document)
        logger.info(f"Added knowledge document: {title}")

    def get_knowledge_summary(self) -> dict[str, Any]:
        """
        Get summary of loaded knowledge base.

        Returns:
            Summary dictionary
        """
        doc_types: dict[str, int] = {}
        total_docs = len(self.documents)

        for doc in self.documents:
            doc_type = doc.get("type", "unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        return {
            "total_documents": total_docs,
            "document_types": doc_types,
            "knowledge_dir": str(self.knowledge_dir),
            "documents_loaded": total_docs > 0,
        }
