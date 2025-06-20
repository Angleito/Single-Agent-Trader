#!/usr/bin/env python3
"""
Comprehensive configuration validation and testing utility for the AI Trading Bot.

This script provides advanced validation capabilities including:
- Network connectivity testing
- API endpoint validation
- Private key format validation
- Environment consistency checks
- Security validation
- Performance testing

Usage:
    python scripts/validate_config.py [options]

Options:
    --full              Run full validation including network tests
    --exchange-only     Test only exchange-specific configuration
    --bluefin-only      Test only Bluefin-specific configuration
    --export-report     Export validation report to file
    --monitor           Run continuous monitoring mode
    --fix-suggestions   Show automated fix suggestions
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

from bot.config import ConfigurationValidator, Settings, create_settings


class ConfigurationTester:
    """Advanced configuration testing and validation utility."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.validator = ConfigurationValidator(settings)
        self.test_results: dict[str, Any] = {}

    async def run_full_validation(self) -> dict[str, Any]:
        """Run comprehensive validation including network tests."""
        print("🔍 Running comprehensive configuration validation...")

        start_time = time.time()
        results = await self.validator.validate_all()
        duration = time.time() - start_time

        results["metadata"] = {
            "validation_duration": duration,
            "validator_version": "1.0",
            "settings_hash": self.settings.generate_config_hash(),
        }

        self.test_results = results
        return results

    def run_exchange_validation(self) -> dict[str, Any]:
        """Run exchange-specific validation tests."""
        print("🏦 Running exchange-specific validation...")

        if self.settings.exchange.exchange_type == "bluefin":
            return self.settings.test_bluefin_configuration()

        return {
            "status": "skipped",
            "reason": f"Exchange type {self.settings.exchange.exchange_type} not supported for advanced testing",
        }

    def run_bluefin_validation(self) -> dict[str, Any]:
        """Run Bluefin-specific validation tests."""
        if self.settings.exchange.exchange_type != "bluefin":
            return {"status": "skipped", "reason": "Not using Bluefin exchange"}

        print("🔷 Running Bluefin-specific validation...")
        return self.settings.test_bluefin_configuration()

    def generate_fix_suggestions(self) -> list[dict[str, Any]]:
        """Generate automated fix suggestions based on validation results."""
        suggestions = []

        if not self.test_results:
            return suggestions

        summary = self.test_results.get("summary", {})
        errors = summary.get("errors", [])
        warnings = summary.get("warnings", [])

        # Generate error fixes
        for error in errors:
            suggestion = self._generate_error_fix(error)
            if suggestion:
                suggestions.append(suggestion)

        # Generate warning fixes
        for warning in warnings:
            suggestion = self._generate_warning_fix(warning)
            if suggestion:
                suggestions.append(suggestion)

        return suggestions

    def _generate_error_fix(self, error: str) -> dict[str, Any] | None:
        """Generate fix suggestion for a specific error."""
        fixes = {
            "no internet connectivity": {
                "type": "network",
                "priority": "high",
                "description": "Internet connectivity is required for the trading bot",
                "suggestions": [
                    "Check your internet connection",
                    "Verify firewall settings",
                    "Check proxy configuration if applicable",
                    "Try running: ping 8.8.8.8",
                ],
            },
            "bluefin private key": {
                "type": "credential",
                "priority": "high",
                "description": "Private key format is invalid",
                "suggestions": [
                    "Ensure private key is 64 hex characters (with or without 0x prefix)",
                    "For mnemonic: use exactly 12 or 24 words",
                    "For Sui format: ensure it starts with 'suiprivkey'",
                    "Check for extra spaces or newlines in the key",
                ],
            },
            "cannot reach bluefin service": {
                "type": "service",
                "priority": "high",
                "description": "Bluefin service is unreachable",
                "suggestions": [
                    "Check if Docker is running and Bluefin service is started",
                    "Verify EXCHANGE__BLUEFIN_SERVICE_URL is correct",
                    "Run: docker-compose up bluefin-service",
                    "Check service logs: docker-compose logs bluefin-service",
                ],
            },
            "openai api key is invalid": {
                "type": "credential",
                "priority": "high",
                "description": "OpenAI API key is not working",
                "suggestions": [
                    "Verify your API key at https://platform.openai.com/api-keys",
                    "Check if the key has sufficient credits",
                    "Ensure the key starts with 'sk-'",
                    "Check for extra spaces in LLM__OPENAI_API_KEY",
                ],
            },
        }

        # Find matching fix
        for pattern, fix in fixes.items():
            if pattern.lower() in error.lower():
                return {"error": error, "fix": fix}

        return None

    def _generate_warning_fix(self, warning: str) -> dict[str, Any] | None:
        """Generate fix suggestion for a specific warning."""
        fixes = {
            "high leverage": {
                "type": "risk",
                "priority": "medium",
                "description": "High leverage increases risk",
                "suggestions": [
                    "Consider reducing TRADING__LEVERAGE to 5x or lower",
                    "Use conservative position sizing",
                    "Ensure adequate risk management",
                    "Test with paper trading first",
                ],
            },
            "production environment using testnet": {
                "type": "environment",
                "priority": "medium",
                "description": "Environment-network mismatch",
                "suggestions": [
                    "Set EXCHANGE__BLUEFIN_NETWORK=mainnet for production",
                    "Or change SYSTEM__ENVIRONMENT=development for testnet",
                    "Ensure consistency between environment and network",
                ],
            },
            "high llm temperature": {
                "type": "llm",
                "priority": "low",
                "description": "High temperature may cause inconsistent decisions",
                "suggestions": [
                    "Set LLM__TEMPERATURE to 0.1 or lower for trading",
                    "Higher temperatures are better for creative tasks",
                    "For trading, consistency is more important than creativity",
                ],
            },
        }

        # Find matching fix
        for pattern, fix in fixes.items():
            if pattern.lower() in warning.lower():
                return {"warning": warning, "fix": fix}

        return None

    def print_results(self, results: dict[str, Any]) -> None:
        """Print formatted validation results."""
        summary = results.get("summary", {})

        # Print header
        print("\n" + "=" * 60)
        print("🔍 CONFIGURATION VALIDATION RESULTS")
        print("=" * 60)

        # Print summary
        status = "✅ PASS" if summary.get("is_valid", False) else "❌ FAIL"
        print(f"\nOverall Status: {status}")
        print(f"Errors: {summary.get('total_errors', 0)}")
        print(f"Warnings: {summary.get('total_warnings', 0)}")

        if "metadata" in results:
            duration = results["metadata"].get("validation_duration", 0)
            print(f"Validation Time: {duration:.2f}s")

        # Print detailed results
        for section, section_results in results.items():
            if section in ["summary", "metadata"] or section_results is None:
                continue

            print(f"\n📋 {section.upper().replace('_', ' ')}")
            print("-" * 40)

            status = section_results.get("status", "unknown")
            status_icon = {"pass": "✅", "fail": "❌", "warning": "⚠️"}.get(status, "❓")
            print(f"Status: {status_icon} {status.upper()}")

            if "checks" in section_results:
                for check in section_results["checks"]:
                    check_icon = {
                        "pass": "✅",
                        "fail": "❌",
                        "warning": "⚠️",
                        "skip": "⏭️",
                    }.get(check["status"], "❓")
                    print(f"  {check_icon} {check['name']}: {check['message']}")

        # Print errors and warnings
        if summary.get("errors"):
            print(f"\n🚨 ERRORS ({len(summary['errors'])})")
            print("-" * 40)
            for error in summary["errors"]:
                print(f"  ❌ {error}")

        if summary.get("warnings"):
            print(f"\n⚠️  WARNINGS ({len(summary['warnings'])})")
            print("-" * 40)
            for warning in summary["warnings"]:
                print(f"  ⚠️  {warning}")

        print("\n" + "=" * 60)

    def print_fix_suggestions(self, suggestions: list[dict[str, Any]]) -> None:
        """Print fix suggestions in a formatted way."""
        if not suggestions:
            print("\n✨ No fix suggestions available")
            return

        print(f"\n🔧 FIX SUGGESTIONS ({len(suggestions)})")
        print("=" * 60)

        for i, suggestion in enumerate(suggestions, 1):
            fix = suggestion["fix"]
            issue = suggestion.get("error") or suggestion.get("warning")

            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                fix["priority"], "⚪"
            )

            print(
                f"\n{i}. {priority_icon} {fix['type'].upper()} - {fix['priority'].upper()} PRIORITY"
            )
            print(f"   Issue: {issue}")
            print(f"   Description: {fix['description']}")
            print("   Suggestions:")
            for sug in fix["suggestions"]:
                print(f"     • {sug}")

        print("\n" + "=" * 60)

    def export_report(self, filepath: str) -> None:
        """Export validation report to file."""
        if not self.test_results:
            print("❌ No test results to export. Run validation first.")
            return

        # Enhance results with additional metadata
        report = {
            "generated_at": time.time(),
            "settings_summary": self.settings.get_configuration_summary(),
            "validation_results": self.test_results,
            "fix_suggestions": self.generate_fix_suggestions(),
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with filepath.open("w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📄 Report exported to: {filepath}")

    async def run_monitoring_mode(self) -> None:
        """Run continuous configuration monitoring."""
        print("📊 Starting configuration monitoring mode...")
        print("Press Ctrl+C to stop")

        monitor = self.settings.create_configuration_monitor()

        def on_config_change(_settings, old_hash, new_hash):
            print(f"🔄 Configuration change detected: {old_hash} -> {new_hash}")

        monitor.register_change_callback(on_config_change)

        try:
            while True:
                # Check for changes
                if monitor.check_for_changes():
                    print("🔄 Running validation after configuration change...")
                    results = await self.run_full_validation()
                    self.print_results(results)

                # Print health status
                health = monitor.get_health_status()
                status_icon = {
                    "healthy": "💚",
                    "warning": "💛",
                    "degraded": "🧡",
                    "unhealthy": "❤️",
                }.get(health["overall_status"], "⚪")

                print(
                    f"{status_icon} Health: {health['overall_status']} | Hash: {health['config_hash'][:8]}"
                )

                await asyncio.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive configuration validation for AI Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/validate_config.py --full
    python scripts/validate_config.py --bluefin-only --fix-suggestions
    python scripts/validate_config.py --monitor
    python scripts/validate_config.py --export-report reports/config_validation.json
        """,
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full validation including network tests",
    )
    parser.add_argument(
        "--exchange-only",
        action="store_true",
        help="Test only exchange-specific configuration",
    )
    parser.add_argument(
        "--bluefin-only",
        action="store_true",
        help="Test only Bluefin-specific configuration",
    )
    parser.add_argument(
        "--export-report", type=str, help="Export validation report to file"
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Run continuous monitoring mode"
    )
    parser.add_argument(
        "--fix-suggestions", action="store_true", help="Show automated fix suggestions"
    )
    parser.add_argument(
        "--env-file", type=str, help="Path to .env file (default: .env)"
    )

    args = parser.parse_args()

    # Load settings
    try:
        settings = create_settings(env_file=args.env_file)
        print("✅ Configuration loaded successfully")
        print(f"   Exchange: {settings.exchange.exchange_type}")
        print(f"   Environment: {settings.system.environment.value}")
        print(f"   Dry Run: {settings.system.dry_run}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)

    # Create tester
    tester = ConfigurationTester(settings)

    try:
        # Run appropriate validation
        if args.monitor:
            await tester.run_monitoring_mode()
        elif args.full:
            results = await tester.run_full_validation()
            tester.print_results(results)
        elif args.exchange_only:
            results = tester.run_exchange_validation()
            tester.print_results(
                {
                    "exchange_validation": results,
                    "summary": {
                        "is_valid": results.get("status") == "pass",
                        "total_errors": 0,
                        "total_warnings": 0,
                    },
                }
            )
        elif args.bluefin_only:
            results = tester.run_bluefin_validation()
            tester.print_results(
                {
                    "bluefin_validation": results,
                    "summary": {
                        "is_valid": results.get("status") == "pass",
                        "total_errors": 0,
                        "total_warnings": 0,
                    },
                }
            )
        else:
            # Default: run basic validation
            results = tester.run_exchange_validation()
            tester.print_results(
                {
                    "exchange_validation": results,
                    "summary": {
                        "is_valid": results.get("status") == "pass",
                        "total_errors": 0,
                        "total_warnings": 0,
                    },
                }
            )

        # Show fix suggestions if requested
        if args.fix_suggestions and tester.test_results:
            suggestions = tester.generate_fix_suggestions()
            tester.print_fix_suggestions(suggestions)

        # Export report if requested
        if args.export_report:
            tester.export_report(args.export_report)

        # Exit with error code if validation failed
        if tester.test_results and not tester.test_results.get("summary", {}).get(
            "is_valid", True
        ):
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n👋 Validation interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
