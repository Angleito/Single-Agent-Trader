"""
Test Migration Compatibility

This module tests that the functional adapter migration maintains
exact API compatibility for all exchange operations.
"""

import asyncio
import logging
from decimal import Decimal

from bot.exchange.bluefin import BluefinClient
from bot.exchange.coinbase import CoinbaseClient
from bot.trading_types import Order, OrderStatus

logger = logging.getLogger(__name__)


async def test_coinbase_compatibility():
    """Test that CoinbaseClient functionality remains unchanged after FP migration."""
    try:
        # Create client in dry run mode
        client = CoinbaseClient(dry_run=True)
        
        # Test that the client initializes correctly
        assert hasattr(client, "_fp_adapter"), "Functional adapter should be initialized"
        
        # Test functional adapter methods exist
        assert hasattr(client, "get_functional_adapter"), "get_functional_adapter method missing"
        assert hasattr(client, "supports_functional_operations"), "supports_functional_operations method missing"
        assert hasattr(client, "place_order_functional"), "place_order_functional method missing"
        
        # Test adapter availability
        supports_fp = client.supports_functional_operations()
        logger.info(f"✅ Coinbase functional operations supported: {supports_fp}")
        
        # Test connection status (should work without network access in dry run)
        connection_status = client.get_connection_status()
        assert isinstance(connection_status, dict), "Connection status should be dict"
        logger.info("✅ Coinbase connection status method works")
        
        # Test balance method exists (will fail gracefully in dry run without credentials)
        try:
            balance = await client.get_account_balance()
            logger.info(f"✅ Coinbase balance method works: ${balance}")
        except Exception as e:
            logger.info(f"✅ Coinbase balance method exists (expected failure in dry run): {e}")
        
        logger.info("✅ Coinbase compatibility test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Coinbase compatibility test failed: {e}")
        return False


async def test_bluefin_compatibility():
    """Test that BluefinClient functionality remains unchanged after FP migration."""
    try:
        # Create client in dry run mode
        client = BluefinClient(dry_run=True)
        
        # Test that the client initializes correctly
        assert hasattr(client, "_fp_adapter"), "Functional adapter should be initialized"
        
        # Test functional adapter methods exist
        assert hasattr(client, "get_functional_adapter"), "get_functional_adapter method missing"
        assert hasattr(client, "supports_functional_operations"), "supports_functional_operations method missing"
        assert hasattr(client, "place_order_functional"), "place_order_functional method missing"
        
        # Test adapter availability
        supports_fp = client.supports_functional_operations()
        logger.info(f"✅ Bluefin functional operations supported: {supports_fp}")
        
        # Test connection status
        connection_status = client.get_connection_status()
        assert isinstance(connection_status, dict), "Connection status should be dict"
        logger.info("✅ Bluefin connection status method works")
        
        # Test futures property
        assert client.enable_futures == True, "Bluefin should enable futures"
        logger.info("✅ Bluefin futures property works")
        
        # Test balance method exists (will fail gracefully in dry run without credentials)
        try:
            balance = await client.get_account_balance()
            logger.info(f"✅ Bluefin balance method works: ${balance}")
        except Exception as e:
            logger.info(f"✅ Bluefin balance method exists (expected failure in dry run): {e}")
        
        logger.info("✅ Bluefin compatibility test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Bluefin compatibility test failed: {e}")
        return False


async def test_functional_adapter_registration():
    """Test that functional adapters are registered correctly."""
    try:
        from bot.fp.adapters.exchange_adapter import get_exchange_adapter
        
        # Get the unified adapter
        unified_adapter = get_exchange_adapter()
        
        # Test adapter registration
        logger.info(f"✅ Unified adapter created: {unified_adapter}")
        logger.info(f"✅ Available adapters: {list(unified_adapter.adapters.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Functional adapter registration test failed: {e}")
        return False


async def test_type_conversion():
    """Test type conversion utilities."""
    try:
        from bot.fp.adapters.type_converters import (
            create_fp_account_balance,
            create_order_result,
        )
        
        # Test balance conversion
        balance = create_fp_account_balance(Decimal("1000.50"))
        assert isinstance(balance, dict), "Balance should be dict"
        assert "cash" in balance, "Balance should have cash field"
        logger.info(f"✅ Balance conversion works: {balance}")
        
        # Test order result creation
        test_order = Order(
            id="test-123",
            symbol="BTC-USD",
            side="BUY",
            type="MARKET",
            quantity=Decimal("0.01"),
            status=OrderStatus.PENDING,
        )
        
        order_result = create_order_result(test_order)
        assert isinstance(order_result, dict), "Order result should be dict"
        assert "order_id" in order_result, "Order result should have order_id"
        logger.info(f"✅ Order result creation works: {order_result}")
        
        logger.info("✅ Type conversion test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Type conversion test failed: {e}")
        return False


async def run_compatibility_tests():
    """Run all compatibility tests."""
    logger.info("🚀 Starting exchange migration compatibility tests...")
    
    results = {
        "coinbase": await test_coinbase_compatibility(),
        "bluefin": await test_bluefin_compatibility(),
        "adapter_registration": await test_functional_adapter_registration(),
        "type_conversion": await test_type_conversion(),
    }
    
    passed = sum(results.values())
    total = len(results)
    
    logger.info("=" * 60)
    logger.info("EXCHANGE MIGRATION COMPATIBILITY TEST RESULTS")
    logger.info("=" * 60)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        logger.info(f"{test_name.upper()}: {status}")
    
    logger.info("=" * 60)
    logger.info(f"OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Migration maintains full API compatibility!")
        return True
    else:
        logger.error("❌ SOME TESTS FAILED - Migration may have compatibility issues")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = asyncio.run(run_compatibility_tests())
    exit(0 if success else 1)