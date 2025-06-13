#!/usr/bin/env python3
"""
Test script for enhanced LLM logging system.

This script tests the new LLM completion logging infrastructure
without requiring a full trading bot setup.
"""

import asyncio
import logging
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

# Add bot module to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.config import settings
from bot.logging import create_llm_logger, create_langchain_callback
from bot.types import MarketState, IndicatorData, Position, TradeAction

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_completion_logging():
    """Test the completion logging system."""
    print("🧪 Testing Enhanced LLM Completion Logging System")
    print("=" * 60)
    
    # Create completion logger
    completion_logger = create_llm_logger(
        log_level="DEBUG",
        log_file="logs/test_llm_completions.log"
    )
    
    # Test 1: Basic request/response logging
    print("\n📝 Test 1: Basic Request/Response Logging")
    
    test_prompt = """You are an expert cryptocurrency trader. 
    Current market conditions:
    - BTC-USD: $45,230
    - RSI: 62.5
    - Trend: Bullish
    
    Should I buy, sell, or hold?"""
    
    market_context = {
        "symbol": "BTC-USD",
        "current_price": 45230.0,
        "rsi": 62.5,
        "trend": "bullish"
    }
    
    # Log a request
    request_id = completion_logger.log_completion_request(
        prompt=test_prompt,
        model="o3-mini",
        temperature=0.1,
        max_tokens=1000,
        market_context=market_context
    )
    
    print(f"  ✓ Logged completion request with ID: {request_id}")
    
    # Simulate response delay
    await asyncio.sleep(0.5)
    
    # Log a successful response
    mock_response = TradeAction(
        action="LONG",
        size_pct=15.0,
        take_profit_pct=3.0,
        stop_loss_pct=1.5,
        leverage=5,
        rationale="Bullish momentum with healthy RSI"
    )
    
    token_usage = {
        "prompt_tokens": 150,
        "completion_tokens": 75,
        "total_tokens": 225
    }
    
    completion_logger.log_completion_response(
        request_id=request_id,
        response=mock_response,
        response_time=0.5,
        token_usage=token_usage,
        success=True
    )
    
    print(f"  ✓ Logged successful completion response")
    
    # Test 2: Trading decision logging
    print("\n🎯 Test 2: Trading Decision Logging")
    
    # Create mock market state
    market_state = MarketState(
        symbol="BTC-USD",
        interval="3m",
        timestamp=datetime.now(UTC),
        current_price=Decimal("45230.50"),
        ohlcv_data=[],  # Empty for test
        indicators=IndicatorData(
            timestamp=datetime.now(UTC),
            cipher_a_dot=0.75,
            cipher_b_wave=0.32,
            cipher_b_money_flow=65.2,
            rsi=62.5,
            ema_fast=44800.0,
            ema_slow=44200.0
        ),
        current_position=Position(
            symbol="BTC-USD",
            side="FLAT",
            size=Decimal("0"),
            timestamp=datetime.now(UTC)
        )
    )
    
    completion_logger.log_trading_decision(
        request_id=request_id,
        trade_action=mock_response,
        market_state=market_state,
        validation_result="PASSED",
        risk_assessment="APPROVED - Within risk parameters"
    )
    
    print(f"  ✓ Logged trading decision with market context")
    
    # Test 3: Error handling
    print("\n❌ Test 3: Error Handling")
    
    error_request_id = completion_logger.log_completion_request(
        prompt="Test error scenario",
        model="o3-mini",
        temperature=0.1,
        max_tokens=1000
    )
    
    completion_logger.log_completion_response(
        request_id=error_request_id,
        response=None,
        response_time=2.0,
        success=False,
        error="API rate limit exceeded"
    )
    
    print(f"  ✓ Logged error response")
    
    # Test 4: Performance metrics
    print("\n📊 Test 4: Performance Metrics")
    
    # Simulate multiple completions for metrics
    for i in range(3):
        req_id = completion_logger.log_completion_request(
            prompt=f"Test prompt {i+1}",
            model="o3-mini",
            temperature=0.1,
            max_tokens=1000
        )
        
        completion_logger.log_completion_response(
            request_id=req_id,
            response=f"Test response {i+1}",
            response_time=0.3 + (i * 0.1),
            token_usage={
                "prompt_tokens": 100 + (i * 10),
                "completion_tokens": 50 + (i * 5),
                "total_tokens": 150 + (i * 15)
            },
            success=True
        )
    
    # Log performance metrics
    metrics = completion_logger.log_performance_metrics()
    
    print(f"  ✓ Performance Metrics:")
    print(f"    - Total Completions: {metrics['total_completions']}")
    print(f"    - Average Response Time: {metrics['avg_response_time_ms']}ms")
    print(f"    - Total Tokens: {metrics['total_tokens']}")
    print(f"    - Estimated Cost: ${metrics['total_cost_estimate_usd']:.6f}")
    print(f"    - Tokens per Second: {metrics['tokens_per_second']}")
    
    # Test 5: LangChain callback (if available)
    print("\n🔗 Test 5: LangChain Callback Handler")
    
    callback_handler = create_langchain_callback(completion_logger)
    if callback_handler:
        print(f"  ✓ LangChain callback handler created successfully")
        print(f"    - Type: {type(callback_handler).__name__}")
    else:
        print(f"  ⚠ LangChain not available - callback handler not created")
    
    # Test 6: Check log files
    print("\n📁 Test 6: Log File Verification")
    
    log_file = Path("logs/test_llm_completions.log")
    if log_file.exists():
        file_size = log_file.stat().st_size
        print(f"  ✓ Log file created: {log_file}")
        print(f"    - Size: {file_size} bytes")
        
        # Show last few lines
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                print(f"    - Total log entries: {len(lines)}")
                if lines:
                    print(f"    - Latest entry: {lines[-1].strip()}")
        except Exception as e:
            print(f"    - Error reading log file: {e}")
    else:
        print(f"  ❌ Log file not found: {log_file}")
    
    print("\n✅ Enhanced LLM Logging Test Complete!")
    print("=" * 60)
    
    return True


async def test_integration_with_settings():
    """Test integration with configuration settings."""
    print("\n🔧 Testing Configuration Integration")
    print("-" * 40)
    
    # Display current LLM logging settings
    print(f"Completion Logging Enabled: {settings.llm.enable_completion_logging}")
    print(f"Completion Log Level: {settings.llm.completion_log_level}")
    print(f"Completion Log File: {settings.llm.completion_log_file}")
    print(f"Performance Tracking: {settings.llm.enable_performance_tracking}")
    print(f"LangChain Callbacks: {settings.llm.enable_langchain_callbacks}")
    print(f"Market Context Logging: {settings.llm.log_market_context}")
    print(f"Performance Log Interval: {settings.llm.performance_log_interval}")
    
    # Test with configured settings
    if settings.llm.enable_completion_logging:
        config_logger = create_llm_logger()
        test_req_id = config_logger.log_completion_request(
            prompt="Testing with configured settings",
            model=settings.llm.model_name,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens
        )
        
        config_logger.log_completion_response(
            request_id=test_req_id,
            response="Configuration test successful",
            response_time=0.25,
            success=True
        )
        
        print(f"  ✓ Successfully logged with configured settings")
    else:
        print(f"  ⚠ Completion logging disabled in configuration")


if __name__ == "__main__":
    async def main():
        try:
            # Ensure logs directory exists
            Path("logs").mkdir(exist_ok=True)
            
            # Run tests
            await test_completion_logging()
            await test_integration_with_settings()
            
            print(f"\n🎉 All tests completed successfully!")
            
        except Exception as e:
            print(f"\n💥 Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        return 0
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)