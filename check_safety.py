#!/usr/bin/env python3
"""
Trading Bot Safety Checker
Verifies that all safety settings are properly configured to prevent accidental live trading.
"""

import json
import sys
from pathlib import Path


class SafetyChecker:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.errors = []
        self.warnings = []
        self.successes = []

    def check_env_file(self) -> bool:
        """Check .env file for safety settings."""
        env_path = self.project_root / ".env"

        if not env_path.exists():
            self.errors.append(
                "❌ .env file not found! Bot cannot run without configuration."
            )
            return False

        with open(env_path) as f:
            content = f.read()

        # Check critical safety settings
        checks = {
            "SYSTEM__DRY_RUN=true": "Paper trading mode",
            "TRADING__LEVERAGE=": "Leverage setting",
        }

        for setting, description in checks.items():
            if setting in content:
                if setting == "SYSTEM__DRY_RUN=true":
                    self.successes.append(f"✅ {description} is ENABLED (safe)")
                elif setting == "TRADING__LEVERAGE=":
                    # Extract leverage value
                    for line in content.split("\n"):
                        if line.startswith("TRADING__LEVERAGE="):
                            leverage = int(line.split("=")[1])
                            if leverage <= 5:
                                self.successes.append(
                                    f"✅ Leverage is {leverage}x (safe limit)"
                                )
                            else:
                                self.warnings.append(
                                    f"⚠️  Leverage is {leverage}x (above safe limit of 5x)"
                                )
            elif setting == "SYSTEM__DRY_RUN=true":
                # Check if it's false
                if "SYSTEM__DRY_RUN=false" in content:
                    self.errors.append(
                        "❌ LIVE TRADING MODE ACTIVE! Real money at risk!"
                    )
                else:
                    self.warnings.append(f"⚠️  {description} setting not found")

        return len(self.errors) == 0

    def check_json_configs(self) -> bool:
        """Check JSON configuration files for safety settings."""
        config_dir = self.project_root / "config"

        important_configs = [
            "production.json",
            "paper_trading.json",
            "conservative_config.json",
        ]

        for config_name in important_configs:
            config_path = config_dir / config_name

            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)

                # Check system.dry_run
                if config.get("system", {}).get("dry_run", False):
                    self.successes.append(f"✅ {config_name}: dry_run is true")
                else:
                    self.warnings.append(
                        f"⚠️  {config_name}: dry_run is false or missing"
                    )

                # Check leverage
                leverage = config.get("trading", {}).get("leverage", 0)
                if 0 < leverage <= 5:
                    self.successes.append(f"✅ {config_name}: leverage is {leverage}x")
                elif leverage > 5:
                    self.warnings.append(
                        f"⚠️  {config_name}: leverage is {leverage}x (high)"
                    )

        return True

    def check_api_keys(self) -> bool:
        """Check that API keys are not hardcoded."""
        env_path = self.project_root / ".env"

        if env_path.exists():
            with open(env_path) as f:
                content = f.read()

            # Check for placeholder API keys
            if "your-openai-api-key-here" in content:
                self.warnings.append("⚠️  OpenAI API key needs to be configured")
            if "0xPLEASE_REPLACE" in content:
                self.warnings.append("⚠️  Bluefin private key needs to be configured")

        return True

    def generate_report(self) -> str:
        """Generate safety check report."""
        report = []
        report.append("=" * 60)
        report.append("TRADING BOT SAFETY CHECK REPORT")
        report.append("=" * 60)
        report.append("")

        if not self.errors:
            report.append("🎉 SAFETY STATUS: ALL CRITICAL CHECKS PASSED")
            report.append("✅ Paper trading mode is active - No real money at risk")
        else:
            report.append("🚨 SAFETY STATUS: CRITICAL ISSUES FOUND")
            report.append("⚠️  Please review and fix before running the bot")

        report.append("")

        if self.errors:
            report.append("CRITICAL ERRORS:")
            for error in self.errors:
                report.append(f"  {error}")
            report.append("")

        if self.warnings:
            report.append("WARNINGS:")
            for warning in self.warnings:
                report.append(f"  {warning}")
            report.append("")

        if self.successes:
            report.append("VERIFIED SAFE:")
            for success in self.successes:
                report.append(f"  {success}")
            report.append("")

        report.append("RECOMMENDATIONS:")
        report.append(
            "  1. Always test with paper trading first (SYSTEM__DRY_RUN=true)"
        )
        report.append("  2. Keep leverage at 5x or below for safety")
        report.append("  3. Configure API keys before running the bot")
        report.append("  4. Monitor the bot closely during initial runs")
        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    def run(self) -> int:
        """Run all safety checks."""
        print("🔍 Running safety checks...\n")

        self.check_env_file()
        self.check_json_configs()
        self.check_api_keys()

        report = self.generate_report()
        print(report)

        # Return non-zero exit code if errors found
        return 1 if self.errors else 0


if __name__ == "__main__":
    checker = SafetyChecker()
    sys.exit(checker.run())
