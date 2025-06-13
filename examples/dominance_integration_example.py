#!/usr/bin/env python3
"""
Example script demonstrating USDT/USDC dominance integration in the trading bot.

This script shows how the dominance data provider works and how it integrates
with the trading decision process.
"""

import asyncio
import logging
from datetime import datetime, UTC
from decimal import Decimal
from pathlib import Path
import sys

# Add parent directory to path to import bot modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.data.dominance import DominanceDataProvider
from bot.types import MarketState, IndicatorData, Position, MarketData
from bot.strategy.llm_agent import LLMAgent
from bot.config import create_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def display_dominance_data(provider: DominanceDataProvider):
    """Display current dominance data and market sentiment."""
    print("\n" + "="*60)
    print("STABLECOIN DOMINANCE DATA")
    print("="*60)
    
    # Get latest dominance data
    dominance = provider.get_latest_dominance()
    if dominance:
        print(f"\n📊 Current Metrics:")
        print(f"  • USDT Dominance: {dominance.usdt_dominance:.2f}%")
        print(f"  • USDC Dominance: {dominance.usdc_dominance:.2f}%")
        print(f"  • Total Stablecoin Dominance: {dominance.stablecoin_dominance:.2f}%")
        print(f"  • 24h Change: {dominance.dominance_24h_change:+.2f}%")
        
        if dominance.dominance_rsi:
            print(f"  • Dominance RSI: {dominance.dominance_rsi:.1f}")
        if dominance.stablecoin_velocity:
            print(f"  • Stablecoin Velocity: {dominance.stablecoin_velocity:.2f}")
        
        # Get market sentiment analysis
        sentiment = provider.get_market_sentiment()
        print(f"\n🎯 Market Sentiment Analysis:")
        print(f"  • Sentiment: {sentiment['sentiment']}")
        print(f"  • Confidence: {sentiment['confidence']:.0f}%")
        print(f"  • Score: {sentiment['score']:.1f}")
        
        print(f"\n📝 Factors:")
        for factor in sentiment['factors']:
            print(f"  • {factor}")
    else:
        print("❌ No dominance data available")


async def simulate_trading_decision(provider: DominanceDataProvider):
    """Simulate how dominance data affects trading decisions."""
    print("\n" + "="*60)
    print("TRADING DECISION SIMULATION")
    print("="*60)
    
    # Create mock market data
    current_price = Decimal("45000")
    mock_ohlcv = [
        MarketData(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            open=Decimal("44800"),
            high=Decimal("45200"),
            low=Decimal("44600"),
            close=current_price,
            volume=Decimal("1234.56")
        )
    ]
    
    # Get dominance data
    dominance = provider.get_latest_dominance()
    
    # Create indicator data with dominance metrics
    indicators = IndicatorData(
        timestamp=datetime.now(UTC),
        cipher_a_dot=0.5,
        cipher_b_wave=0.3,
        cipher_b_money_flow=0.2,
        rsi=55.0,
        ema_fast=44900.0,
        ema_slow=44700.0,
        vwap=44850.0,
        # Add dominance data
        usdt_dominance=dominance.usdt_dominance if dominance else None,
        usdc_dominance=dominance.usdc_dominance if dominance else None,
        stablecoin_dominance=dominance.stablecoin_dominance if dominance else None,
        dominance_trend=dominance.dominance_24h_change if dominance else None,
        dominance_rsi=dominance.dominance_rsi if dominance else None,
        stablecoin_velocity=dominance.stablecoin_velocity if dominance else None,
        market_sentiment=provider.get_market_sentiment()['sentiment'] if dominance else None
    )
    
    # Create market state
    market_state = MarketState(
        symbol="BTC-USD",
        interval="1m",
        timestamp=datetime.now(UTC),
        current_price=current_price,
        ohlcv_data=mock_ohlcv,
        indicators=indicators,
        current_position=Position(
            symbol="BTC-USD",
            side="FLAT",
            size=Decimal("0"),
            timestamp=datetime.now(UTC)
        ),
        dominance_data=dominance
    )
    
    print(f"\n📈 Market Context:")
    print(f"  • Current Price: ${current_price:,.2f}")
    print(f"  • RSI: {indicators.rsi}")
    print(f"  • Cipher A Dot: {indicators.cipher_a_dot}")
    
    if dominance:
        print(f"\n💰 Dominance Impact:")
        print(f"  • Stablecoin Dominance: {dominance.stablecoin_dominance:.2f}%")
        
        if dominance.stablecoin_dominance > 10:
            print("  • ⚠️  HIGH DOMINANCE - Risk-off sentiment, reduce position sizes")
        elif dominance.stablecoin_dominance > 7:
            print("  • ⚠️  ELEVATED DOMINANCE - Cautious market, normal position sizes")
        else:
            print("  • ✅ LOW DOMINANCE - Risk-on sentiment, normal to larger positions")
        
        if dominance.dominance_24h_change > 0.5:
            print("  • 📉 RISING DOMINANCE - Money flowing to stables (bearish)")
        elif dominance.dominance_24h_change < -0.5:
            print("  • 📈 FALLING DOMINANCE - Money flowing to crypto (bullish)")
        else:
            print("  • ➡️  STABLE DOMINANCE - No significant flow")
    
    # Try to get LLM decision if available
    try:
        settings = create_settings()
        llm_agent = LLMAgent()
        if llm_agent.is_available():
            print("\n🤖 Getting LLM trading decision...")
            decision = await llm_agent.analyze_market(market_state)
            print(f"  • Action: {decision.action}")
            print(f"  • Size: {decision.size_pct}%")
            print(f"  • Rationale: {decision.rationale}")
        else:
            print("\n⚡ Using fallback decision logic...")
            decision = llm_agent._get_fallback_decision(market_state)
            print(f"  • Action: {decision.action}")
            print(f"  • Size: {decision.size_pct}%")
            print(f"  • Rationale: {decision.rationale}")
    except Exception as e:
        print(f"\n❌ Could not generate trading decision: {e}")


async def display_historical_analysis(provider: DominanceDataProvider):
    """Display historical dominance analysis."""
    print("\n" + "="*60)
    print("HISTORICAL DOMINANCE ANALYSIS")
    print("="*60)
    
    # Get historical data
    history = provider.get_dominance_history(hours=24)
    
    if history:
        print(f"\n📊 24-Hour Statistics:")
        dominance_values = [h.stablecoin_dominance for h in history]
        
        if dominance_values:
            avg_dominance = sum(dominance_values) / len(dominance_values)
            max_dominance = max(dominance_values)
            min_dominance = min(dominance_values)
            
            print(f"  • Average Dominance: {avg_dominance:.2f}%")
            print(f"  • Maximum Dominance: {max_dominance:.2f}%")
            print(f"  • Minimum Dominance: {min_dominance:.2f}%")
            print(f"  • Range: {max_dominance - min_dominance:.2f}%")
            
            # Convert to DataFrame for better analysis
            df = provider.to_dataframe(hours=24)
            if not df.empty:
                print(f"\n📈 Trend Analysis:")
                latest_dom = df['stablecoin_dominance'].iloc[-1]
                first_dom = df['stablecoin_dominance'].iloc[0]
                change_24h = latest_dom - first_dom
                
                print(f"  • 24h Change: {change_24h:+.2f}%")
                print(f"  • Data Points: {len(df)}")
                
                if 'dominance_rsi' in df.columns:
                    latest_rsi = df['dominance_rsi'].dropna()
                    if not latest_rsi.empty:
                        print(f"  • Latest RSI: {latest_rsi.iloc[-1]:.1f}")
    else:
        print("❌ No historical data available")


async def main():
    """Main example function."""
    print("\n🚀 USDT/USDC Dominance Integration Example")
    print("=" * 60)
    
    # Create dominance provider
    provider = DominanceDataProvider(
        data_source="coingecko",
        update_interval=300  # 5 minutes
    )
    
    try:
        # Connect to data source
        print("\n📡 Connecting to dominance data provider...")
        await provider.connect()
        print("✅ Connected successfully!")
        
        # Wait a moment for initial data
        await asyncio.sleep(2)
        
        # Display current dominance data
        await display_dominance_data(provider)
        
        # Simulate trading decision with dominance
        await simulate_trading_decision(provider)
        
        # Show historical analysis
        await display_historical_analysis(provider)
        
        print("\n" + "="*60)
        print("💡 Key Insights:")
        print("="*60)
        print("""
1. High stablecoin dominance (>10%) indicates risk-off sentiment
2. Rising dominance suggests money flowing to safety (bearish)
3. Falling dominance suggests risk-on behavior (bullish)
4. Dominance RSI can signal potential reversals
5. The bot adjusts position sizes based on dominance levels
        """)
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        print(f"\n❌ Error: {e}")
    
    finally:
        # Cleanup
        print("\n🔌 Disconnecting...")
        await provider.disconnect()
        print("✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())