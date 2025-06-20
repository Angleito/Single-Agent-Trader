#!/usr/bin/env python3
"""
Simplified Bluefin configuration testing utility.

This script provides quick validation and testing specifically for Bluefin DEX integration,
including private key validation, network connectivity, and endpoint accessibility.

Usage:
    python scripts/test_bluefin_config.py [options]

Options:
    --network NETWORK   Test specific network (mainnet/testnet)
    --test-api          Test API connectivity
    --test-rpc          Test Sui RPC connectivity
    --test-service      Test Bluefin service connectivity
    --validate-key      Validate private key format
    --quick             Run quick validation (no network tests)
    --verbose           Show detailed output
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import aiohttp

    from bot.config import Settings, create_settings
    from bot.exchange.bluefin_endpoints import BluefinEndpointConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(
        "Make sure you're running from the project root and dependencies are installed"
    )
    sys.exit(1)


class BluefinConfigTester:
    """Focused testing utility for Bluefin configuration."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.verbose = False
        self.results: dict[str, Any] = {
            "tests": [],
            "summary": {"passed": 0, "failed": 0, "warnings": 0},
        }

    def set_verbose(self, verbose: bool) -> None:
        """Enable verbose output."""
        self.verbose = verbose

    def log(self, message: str, level: str = "info") -> None:
        """Log message with appropriate formatting."""
        if level == "error":
            print(f"❌ {message}")
        elif level == "warning":
            print(f"⚠️  {message}")
        elif level == "success":
            print(f"✅ {message}")
        elif level == "info" and self.verbose:
            print(f"ℹ️  {message}")

    def add_result(
        self, test_name: str, status: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        """Add test result."""
        result = {
            "test": test_name,
            "status": status,
            "message": message,
            "timestamp": time.time(),
        }
        if details:
            result["details"] = details

        self.results["tests"].append(result)

        if status == "pass":
            self.results["summary"]["passed"] += 1
            self.log(f"{test_name}: {message}", "success")
        elif status == "fail":
            self.results["summary"]["failed"] += 1
            self.log(f"{test_name}: {message}", "error")
        elif status == "warning":
            self.results["summary"]["warnings"] += 1
            self.log(f"{test_name}: {message}", "warning")

    def validate_private_key(self) -> bool:
        """Validate Bluefin private key format."""
        self.log("Testing private key format validation...", "info")

        if not self.settings.exchange.bluefin_private_key:
            self.add_result(
                "private_key_validation", "fail", "No private key configured"
            )
            return False

        try:
            # Use the comprehensive validation from the config
            self.settings.exchange._validate_bluefin_private_key_comprehensive()

            # Determine key format
            key = self.settings.exchange.bluefin_private_key.get_secret_value().strip()
            words = key.split()

            if len(words) in [12, 24]:
                key_type = f"mnemonic ({len(words)} words)"
            elif key.startswith("suiprivkey"):
                key_type = "Sui Bech32"
            else:
                key_type = "hex"

            self.add_result(
                "private_key_validation",
                "pass",
                f"Valid {key_type} private key",
                {"format": key_type, "length": len(key)},
            )
            return True

        except ValueError as e:
            self.add_result(
                "private_key_validation", "fail", f"Invalid private key: {e!s}"
            )
            return False

    def validate_network_config(self, target_network: str | None = None) -> bool:
        """Validate network configuration."""
        network = target_network or self.settings.exchange.bluefin_network
        self.log(f"Testing network configuration for {network}...", "info")

        # Validate network value
        if network not in ["mainnet", "testnet"]:
            self.add_result("network_config", "fail", f"Invalid network: {network}")
            return False

        # Check environment consistency
        env = self.settings.system.environment.value
        dry_run = self.settings.system.dry_run

        warnings = []
        if env == "production" and network == "testnet":
            warnings.append("Production environment using testnet")
        if not dry_run and network == "testnet":
            warnings.append("Live trading enabled on testnet")
        if env == "development" and network == "mainnet" and not dry_run:
            self.add_result(
                "network_config",
                "fail",
                "Development environment should not use live mainnet",
            )
            return False

        if warnings:
            self.add_result(
                "network_config",
                "warning",
                f"Configuration warnings: {'; '.join(warnings)}",
            )
        else:
            self.add_result(
                "network_config", "pass", f"Network configuration valid ({network})"
            )

        return True

    def get_effective_endpoints(self, network: str) -> dict[str, str]:
        """Get effective endpoints for the network."""
        try:
            endpoints = BluefinEndpointConfig.get_endpoints(network)
            return {
                "rest_api": endpoints.rest_api,
                "websocket_api": endpoints.websocket_api,
                "websocket_notifications": endpoints.websocket_notifications,
            }
        except Exception as e:
            self.log(f"Failed to get endpoints: {e}", "error")
            return {}

    async def test_api_connectivity(self, network: str | None = None) -> bool:
        """Test Bluefin API connectivity."""
        network = network or self.settings.exchange.bluefin_network
        self.log(f"Testing Bluefin API connectivity ({network})...", "info")

        endpoints = self.get_effective_endpoints(network)
        if not endpoints:
            self.add_result(
                "api_connectivity", "fail", "Could not determine API endpoints"
            )
            return False

        rest_api = endpoints["rest_api"]
        test_url = f"{rest_api}/ticker24hr"

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            ) as session, session.get(test_url) as response:
                if response.status == 200:
                    data = await response.json()
                    ticker_count = len(data) if isinstance(data, list) else 1
                    self.add_result(
                        "api_connectivity",
                        "pass",
                        f"API accessible ({ticker_count} symbols)",
                        {"endpoint": rest_api, "status_code": response.status},
                    )
                    return True
                elif response.status == 429:
                    self.add_result(
                        "api_connectivity",
                        "warning",
                        "API rate limited (this is normal)",
                        {"endpoint": rest_api, "status_code": response.status},
                    )
                    return True
                else:
                    self.add_result(
                        "api_connectivity",
                        "fail",
                        f"API returned status {response.status}",
                        {"endpoint": rest_api, "status_code": response.status},
                    )
                    return False
        except Exception as e:
            self.add_result(
                "api_connectivity",
                "fail",
                f"Cannot reach API: {e!s}",
                {"endpoint": rest_api, "error": str(e)},
            )
            return False

    async def test_rpc_connectivity(self, network: str | None = None) -> bool:
        """Test Sui RPC connectivity."""
        network = network or self.settings.exchange.bluefin_network
        self.log(f"Testing Sui RPC connectivity ({network})...", "info")

        # Use custom RPC if provided, otherwise default
        if self.settings.exchange.bluefin_rpc_url:
            rpc_url = self.settings.exchange.bluefin_rpc_url
        elif network == "mainnet":
            rpc_url = "https://fullnode.mainnet.sui.io:443"
        else:
            rpc_url = "https://fullnode.testnet.sui.io:443"

        # Test with a simple RPC call
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sui_getLatestSuiSystemState",
            "params": [],
        }

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            ) as session, session.post(rpc_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if "result" in data:
                        epoch = data["result"].get("epoch", "unknown")
                        self.add_result(
                            "rpc_connectivity",
                            "pass",
                            f"RPC accessible (epoch: {epoch})",
                            {"endpoint": rpc_url, "epoch": epoch},
                        )
                        return True
                    else:
                        self.add_result(
                            "rpc_connectivity",
                            "fail",
                            f"Unexpected RPC response: {data}",
                            {"endpoint": rpc_url},
                        )
                        return False
                else:
                    self.add_result(
                        "rpc_connectivity",
                        "fail",
                        f"RPC returned status {response.status}",
                        {"endpoint": rpc_url, "status_code": response.status},
                    )
                    return False
        except Exception as e:
            self.add_result(
                "rpc_connectivity",
                "fail",
                f"Cannot reach RPC: {e!s}",
                {"endpoint": rpc_url, "error": str(e)},
            )
            return False

    async def test_service_connectivity(self) -> bool:
        """Test Bluefin service connectivity."""
        service_url = self.settings.exchange.bluefin_service_url
        self.log("Testing Bluefin service connectivity...", "info")

        try:
            # Test health endpoint
            health_url = f"{service_url.rstrip('/')}/health"
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session, session.get(health_url) as response:
                if response.status == 200:
                    data = await response.json()
                    self.add_result(
                        "service_connectivity",
                        "pass",
                        "Bluefin service accessible",
                        {"endpoint": service_url, "health_data": data},
                    )
                    return True
                else:
                    self.add_result(
                        "service_connectivity",
                        "warning",
                        f"Service returned status {response.status}",
                        {"endpoint": service_url, "status_code": response.status},
                    )
                    return False
        except Exception as e:
            self.add_result(
                "service_connectivity",
                "fail",
                f"Cannot reach service: {e!s}",
                {"endpoint": service_url, "error": str(e)},
            )

            # Provide helpful error messages
            if "Connection refused" in str(e):
                self.log("💡 Try running: docker-compose up bluefin-service", "info")
            elif "Name or service not known" in str(e):
                self.log("💡 Check EXCHANGE__BLUEFIN_SERVICE_URL in .env", "info")

            return False

    async def run_quick_validation(self) -> None:
        """Run quick validation without network tests."""
        print("🔍 Running quick Bluefin configuration validation...")

        # Validate private key
        self.validate_private_key()

        # Validate network config
        self.validate_network_config()

        # Validate service URL format
        service_url = self.settings.exchange.bluefin_service_url
        if service_url and ("://" in service_url):
            self.add_result("service_url_format", "pass", "Service URL format valid")
        else:
            self.add_result("service_url_format", "fail", "Invalid service URL format")

    async def run_full_validation(self, network: str | None = None) -> None:
        """Run full validation including network tests."""
        print("🔍 Running comprehensive Bluefin configuration validation...")

        # Run quick validation first
        await self.run_quick_validation()

        # Network tests
        if self.results["summary"]["failed"] == 0:  # Only if basic validation passed
            await self.test_api_connectivity(network)
            await self.test_rpc_connectivity(network)
            await self.test_service_connectivity()

    async def run_targeted_tests(
        self,
        test_api: bool = False,
        test_rpc: bool = False,
        test_service: bool = False,
        network: str | None = None,
    ) -> None:
        """Run specific targeted tests."""
        print("🔍 Running targeted Bluefin tests...")

        if test_api:
            await self.test_api_connectivity(network)

        if test_rpc:
            await self.test_rpc_connectivity(network)

        if test_service:
            await self.test_service_connectivity()

    def print_summary(self) -> None:
        """Print test summary."""
        summary = self.results["summary"]
        total_tests = summary["passed"] + summary["failed"] + summary["warnings"]

        print("\n" + "=" * 50)
        print("🔷 BLUEFIN CONFIGURATION TEST SUMMARY")
        print("=" * 50)

        if summary["failed"] == 0:
            if summary["warnings"] == 0:
                print("✅ All tests passed!")
            else:
                print(f"⚠️  Tests passed with {summary['warnings']} warning(s)")
        else:
            print(f"❌ {summary['failed']} test(s) failed")

        print(
            f"\nResults: {summary['passed']} passed, {summary['failed']} failed, {summary['warnings']} warnings"
        )
        print(f"Total tests: {total_tests}")

        # Show failed tests
        failed_tests = [t for t in self.results["tests"] if t["status"] == "fail"]
        if failed_tests:
            print("\n❌ Failed Tests:")
            for test in failed_tests:
                print(f"   • {test['test']}: {test['message']}")

        # Show warnings
        warning_tests = [t for t in self.results["tests"] if t["status"] == "warning"]
        if warning_tests:
            print("\n⚠️  Warnings:")
            for test in warning_tests:
                print(f"   • {test['test']}: {test['message']}")

        print("=" * 50)

    def export_results(self, filepath: str) -> None:
        """Export test results to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"📄 Results exported to: {filepath}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bluefin configuration testing utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/test_bluefin_config.py --quick
    python scripts/test_bluefin_config.py --test-api --test-rpc
    python scripts/test_bluefin_config.py --network testnet --verbose
    python scripts/test_bluefin_config.py --validate-key
        """,
    )

    parser.add_argument(
        "--network", choices=["mainnet", "testnet"], help="Test specific network"
    )
    parser.add_argument("--test-api", action="store_true", help="Test API connectivity")
    parser.add_argument(
        "--test-rpc", action="store_true", help="Test Sui RPC connectivity"
    )
    parser.add_argument(
        "--test-service", action="store_true", help="Test Bluefin service connectivity"
    )
    parser.add_argument(
        "--validate-key", action="store_true", help="Validate private key format only"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick validation (no network tests)"
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--export", type=str, help="Export results to JSON file")
    parser.add_argument("--env-file", type=str, help="Path to .env file")

    args = parser.parse_args()

    # Load settings
    try:
        settings = create_settings(env_file=args.env_file)

        if settings.exchange.exchange_type != "bluefin":
            print(
                f"❌ This tool is for Bluefin configuration. Current exchange: {settings.exchange.exchange_type}"
            )
            print("💡 Set EXCHANGE__EXCHANGE_TYPE=bluefin in your .env file")
            sys.exit(1)

        print("✅ Bluefin configuration loaded")
        print(f"   Network: {settings.exchange.bluefin_network}")
        print(f"   Dry Run: {settings.system.dry_run}")

    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)

    # Create tester
    tester = BluefinConfigTester(settings)
    tester.set_verbose(args.verbose)

    try:
        # Run appropriate tests
        if args.validate_key:
            tester.validate_private_key()
        elif args.quick:
            await tester.run_quick_validation()
        elif any([args.test_api, args.test_rpc, args.test_service]):
            await tester.run_targeted_tests(
                test_api=args.test_api,
                test_rpc=args.test_rpc,
                test_service=args.test_service,
                network=args.network,
            )
        else:
            # Default: run full validation
            await tester.run_full_validation(args.network)

        # Print summary
        tester.print_summary()

        # Export results if requested
        if args.export:
            tester.export_results(args.export)

        # Exit with error code if tests failed
        if tester.results["summary"]["failed"] > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n👋 Testing interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Testing failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
