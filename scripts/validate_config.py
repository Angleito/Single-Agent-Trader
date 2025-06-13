#!/usr/bin/env python3
"""
Configuration validation and health check script.

This script demonstrates the new environment setup and validation utilities.
Run this script to validate your bot configuration and check system health.
"""

import json
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import configuration utilities directly (avoid module-level settings import)
# We'll avoid importing from bot package to prevent automatic settings initialization
import os

# Temporarily unset environment variables that cause validation errors
env_backup = {}
problem_vars = [
    "ENVIRONMENT",
    "DRY_RUN",
    "LOG_LEVEL",
    "SYMBOL",
    "INTERVAL",
    "UPDATE_FREQUENCY_SECONDS",
]
for var in problem_vars:
    if var in os.environ:
        env_backup[var] = os.environ[var]
        del os.environ[var]

try:
    from bot.config import TradingProfile
    from bot.config_utils import (
        ConfigManager,
        create_startup_report,
        setup_configuration,
    )
finally:
    # Restore environment variables
    for var, value in env_backup.items():
        os.environ[var] = value


def main():
    """Main validation and health check function."""
    print("🤖 AI Trading Bot - Configuration Validation & Health Check")
    print("=" * 60)

    try:
        # Load configuration
        print("\n📋 Loading configuration...")
        settings = setup_configuration()
        print("✅ Configuration loaded successfully")
        print(f"   Environment: {settings.system.environment.value}")
        print(f"   Profile: {settings.profile.value}")
        print(f"   Dry Run: {settings.system.dry_run}")

        # Generate startup report
        print("\n🔍 Running comprehensive validation...")
        startup_report = create_startup_report(settings)

        # Display validation results
        validation = startup_report["startup_validation"]
        print("\n📊 Validation Results:")
        print(f"   Status: {'✅ Valid' if validation['valid'] else '❌ Invalid'}")

        if validation["critical_errors"]:
            print(f"   Critical Errors: {len(validation['critical_errors'])}")
            for error in validation["critical_errors"]:
                print(f"     ❌ {error}")

        if validation["warnings"]:
            print(f"   Warnings: {len(validation['warnings'])}")
            for warning in validation["warnings"][:5]:  # Show first 5 warnings
                print(f"     ⚠️  {warning}")
            if len(validation["warnings"]) > 5:
                print(f"     ... and {len(validation['warnings']) - 5} more warnings")

        # Display health check results
        health = startup_report["health_check"]
        print("\n🏥 Health Check Results:")
        print(
            f"   Overall Status: {get_status_emoji(health['overall_status'])} {health['overall_status'].title()}"
        )

        for component, details in health["components"].items():
            status_emoji = get_status_emoji(details["status"])
            print(f"   {component.title()}: {status_emoji} {details['status'].title()}")
            if details.get("issues"):
                for issue in details["issues"][:2]:  # Show first 2 issues per component
                    print(f"     • {issue}")

        # Display system metrics
        if health.get("metrics"):
            print("\n📈 Performance Metrics:")
            metrics = health["metrics"]
            if "uptime_formatted" in metrics:
                print(f"   Uptime: {metrics['uptime_formatted']}")
            if "memory_usage_mb" in metrics:
                print(f"   Memory Usage: {metrics['memory_usage_mb']:.1f} MB")
            if "cpu_percent" in metrics:
                print(f"   CPU Usage: {metrics['cpu_percent']:.1f}%")

        # Display recommendations
        if startup_report.get("recommendations"):
            print("\n💡 Recommendations:")
            for i, rec in enumerate(startup_report["recommendations"][:5], 1):
                print(f"   {i}. {rec}")

        # Demonstrate config management features
        print("\n🔧 Configuration Management Demo:")
        config_manager = ConfigManager()

        # List available profiles
        print(f"   Available Profiles: {[p.value for p in TradingProfile]}")

        # List backups
        backups = config_manager.list_config_backups()
        print(f"   Configuration Backups: {len(backups)} available")

        # Save detailed report
        report_path = Path("logs") / "startup_report.json"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(startup_report, f, indent=2, default=str)
        print(f"\n📄 Detailed report saved to: {report_path}")

        # Final status
        if validation["valid"] and health["overall_status"] in ["healthy", "warning"]:
            print("\n✅ System is ready for operation!")
            if settings.system.dry_run:
                print("   💡 Running in dry-run mode (safe for testing)")
            else:
                print("   ⚠️  Running in LIVE mode (real trading)")
        else:
            print("\n❌ System has issues that need to be resolved")
            return 1

        return 0

    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        return 1


def get_status_emoji(status: str) -> str:
    """Get emoji for status."""
    return {"healthy": "✅", "warning": "⚠️", "critical": "❌", "unknown": "❓"}.get(
        status.lower(), "❓"
    )


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
