#!/usr/bin/env python3
"""
Bluefin Authenticated WebSocket Demo

This example demonstrates how to use the new Bluefin WebSocket authentication
system to access private channels for account data, positions, and orders.

IMPORTANT: This example requires a valid Bluefin private key.
           Use testnet for testing to avoid risking real funds.
"""

import asyncio
import logging
import os
import signal
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class BluefinAuthenticatedWebSocketDemo:
    """
    Demo class showing how to use Bluefin WebSocket authentication.

    This demo:
    1. Connects to Bluefin WebSocket with authentication
    2. Subscribes to both public market data and private user data
    3. Handles account updates, position changes, and order status
    4. Demonstrates proper error handling and reconnection
    """

    def __init__(self):
        self.ws_client = None
        self.running = False

        # Get private key from environment
        self.private_key = os.getenv("BLUEFIN_PRIVATE_KEY")
        if not self.private_key:
            print("❌ BLUEFIN_PRIVATE_KEY environment variable not set")
            print("   Please set your ED25519 private key (64 hex characters)")
            print("   Example: export BLUEFIN_PRIVATE_KEY=your_64_char_hex_key")
            raise ValueError("Missing BLUEFIN_PRIVATE_KEY")

        # Use testnet by default for safety
        self.network = os.getenv("BLUEFIN_NETWORK", "testnet")
        self.symbol = os.getenv("BLUEFIN_SYMBOL", "SUI-PERP")

        print(f"🌐 Network: {self.network}")
        print(f"📊 Symbol: {self.symbol}")
        print(f"🔑 Private key: {self.private_key[:16]}...")

    async def on_market_data_update(self, candle_data):
        """Handle market data updates (public channel)."""
        print(
            f"📈 Market Data: {candle_data.symbol} "
            f"Close=${candle_data.close} "
            f"Volume={candle_data.volume} "
            f"@{candle_data.timestamp.strftime('%H:%M:%S')}"
        )

    async def on_account_update(self, account_data):
        """Handle account balance and data updates (private channel)."""
        print(f"💰 Account Update: {account_data}")

        # Extract useful information
        balance = account_data.get("balance", "unknown")
        free_collateral = account_data.get("freeCollateral", "unknown")

        if balance != "unknown":
            print(f"   💵 Balance: {balance}")
        if free_collateral != "unknown":
            print(f"   🆓 Free Collateral: {free_collateral}")

    async def on_position_update(self, position_data):
        """Handle position updates (private channel)."""
        print(f"📊 Position Update: {position_data}")

        # Extract position details
        symbol = position_data.get("symbol", "unknown")
        size = position_data.get("size", "unknown")
        side = position_data.get("side", "unknown")
        unrealized_pnl = position_data.get("unrealizedPnl", "unknown")

        if size != "unknown" and size != "0":
            print(f"   📍 Position: {side} {size} {symbol}")
            if unrealized_pnl != "unknown":
                pnl_color = "🟢" if float(unrealized_pnl) >= 0 else "🔴"
                print(f"   {pnl_color} Unrealized PnL: {unrealized_pnl}")
        else:
            print(f"   ✅ No position in {symbol}")

    async def on_order_update(self, order_data):
        """Handle order status updates (private channel)."""
        print(f"📋 Order Update: {order_data}")

        # Extract order details
        order_id = order_data.get("orderId", order_data.get("id", "unknown"))
        status = order_data.get("status", "unknown")
        symbol = order_data.get("symbol", "unknown")
        side = order_data.get("side", "unknown")
        quantity = order_data.get("quantity", "unknown")

        status_emoji = {
            "PENDING": "⏳",
            "OPEN": "🔵",
            "FILLED": "✅",
            "PARTIALLY_FILLED": "🟡",
            "CANCELLED": "❌",
            "REJECTED": "🚫",
        }.get(status, "❓")

        print(f"   {status_emoji} Order {order_id}: {status}")
        if symbol != "unknown":
            print(f"   📊 {side} {quantity} {symbol}")

    async def setup_websocket_client(self):
        """Set up the authenticated WebSocket client."""
        try:
            from bot.data.bluefin_websocket import BluefinWebSocketClient

            self.ws_client = BluefinWebSocketClient(
                symbol=self.symbol,
                interval="1m",
                network=self.network,
                private_key_hex=self.private_key,
                enable_private_channels=True,
                on_candle_update=self.on_market_data_update,
                on_account_update=self.on_account_update,
                on_position_update=self.on_position_update,
                on_order_update=self.on_order_update,
            )

            print("✅ WebSocket client created with authentication")

            # Check authentication status
            auth_status = self.ws_client.get_authentication_status()
            print(f"🔒 Authentication: {auth_status}")

            return True

        except ImportError as e:
            print(f"❌ Failed to import WebSocket client: {e}")
            return False
        except Exception as e:
            print(f"❌ Failed to create WebSocket client: {e}")
            return False

    async def connect_and_monitor(self):
        """Connect to WebSocket and monitor for updates."""
        try:
            print("🔌 Connecting to Bluefin WebSocket...")
            await self.ws_client.connect()

            print("✅ Connected to Bluefin WebSocket")
            print("🎯 Subscribing to market data and private channels...")

            # Wait a moment for subscriptions to complete
            await asyncio.sleep(2)

            # Show connection status
            status = self.ws_client.get_status()
            print("📊 Connection Status:")
            print(f"   🔗 Connected: {status['connected']}")
            print(f"   🔒 Authentication: {status['authentication_enabled']}")
            print(f"   🔐 Private Channels: {status['private_channels_enabled']}")
            print(f"   📺 Subscribed Channels: {status['subscribed_channels']}")

            self.running = True
            print("\n🚀 Demo running! Monitoring for updates...")
            print("   📈 Market data will appear from public channels")
            print("   💰 Account/position/order data will appear from private channels")
            print("   ⏹️  Press Ctrl+C to stop")

            # Keep running until stopped
            while self.running:
                await asyncio.sleep(1)

                # Periodically show authentication status
                if hasattr(self, "_last_status_check"):
                    if (
                        datetime.now() - self._last_status_check
                    ).seconds > 300:  # Every 5 minutes
                        self._show_status_update()
                        self._last_status_check = datetime.now()
                else:
                    self._last_status_check = datetime.now()

        except KeyboardInterrupt:
            print("\n⏹️  Received stop signal")
        except Exception as e:
            print(f"❌ Error during monitoring: {e}")
            import traceback

            traceback.print_exc()
        finally:
            await self.cleanup()

    def _show_status_update(self):
        """Show periodic status update."""
        if self.ws_client:
            status = self.ws_client.get_status()
            auth_status = self.ws_client.get_authentication_status()

            print("\n📊 Status Update:")
            print(f"   🔗 Connected: {status['connected']}")
            print(f"   📩 Messages: {status['message_count']}")
            print(f"   🔒 Authenticated: {auth_status['authenticated']}")
            print(f"   ⏰ Last Message: {status['last_message_time']}")

    async def cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up...")
        self.running = False

        if self.ws_client:
            try:
                await self.ws_client.disconnect()
                print("✅ WebSocket disconnected")
            except Exception as e:
                print(f"⚠️  Error during disconnect: {e}")

    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            print(f"\n📡 Received signal {signum}")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def run(self):
        """Run the demo."""
        print("🚀 Starting Bluefin Authenticated WebSocket Demo")
        print("=" * 60)

        # Set up signal handlers
        self.setup_signal_handlers()

        # Set up WebSocket client
        if not await self.setup_websocket_client():
            print("❌ Failed to set up WebSocket client")
            return

        # Run the monitoring loop
        await self.connect_and_monitor()

        print("👋 Demo completed")


async def main():
    """Main entry point."""
    try:
        demo = BluefinAuthenticatedWebSocketDemo()
        await demo.run()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\n💡 Setup Instructions:")
        print("1. Get your Bluefin private key (64 hex characters)")
        print("2. Set environment variable: export BLUEFIN_PRIVATE_KEY=your_key")
        print("3. Optionally set BLUEFIN_NETWORK=testnet (recommended)")
        print("4. Run this script again")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("🌟 Bluefin Authenticated WebSocket Demo")
    print("   This demo shows real-time market data + private account data")
    print("   Make sure you have BLUEFIN_PRIVATE_KEY set!")
    print()

    asyncio.run(main())
