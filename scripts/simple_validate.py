#!/usr/bin/env python3
"""
Simple configuration validation script.

This script validates the environment setup without triggering
full bot initialization, making it safe for initial setup validation.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_environment_variables():
    """Check for required environment variables."""
    print("🔍 Checking environment variables...")

    required_vars = {
        "LLM Provider": ["LLM__PROVIDER"],
        "Trading Symbol": ["TRADING__SYMBOL"],
        "System Config": ["SYSTEM__DRY_RUN", "SYSTEM__ENVIRONMENT"],
    }

    conditional_vars = {
        "OpenAI": ["LLM__OPENAI_API_KEY"],
        "Anthropic": ["LLM__ANTHROPIC_API_KEY"],
        "Coinbase (Live Trading)": [
            "EXCHANGE__CB_API_KEY",
            "EXCHANGE__CB_API_SECRET",
            "EXCHANGE__CB_PASSPHRASE",
        ],
    }

    issues = []
    warnings = []

    # Check basic required variables
    for category, vars_list in required_vars.items():
        for var in vars_list:
            if not os.getenv(var):
                issues.append(f"Missing {category}: {var}")

    # Check provider-specific variables
    llm_provider = os.getenv("LLM__PROVIDER", "openai").lower()
    if llm_provider == "openai" and not os.getenv("LLM__OPENAI_API_KEY"):
        issues.append("OpenAI API key required when using OpenAI provider")
    elif llm_provider == "anthropic" and not os.getenv("LLM__ANTHROPIC_API_KEY"):
        issues.append("Anthropic API key required when using Anthropic provider")

    # Check live trading requirements
    dry_run = os.getenv("SYSTEM__DRY_RUN", "true").lower() == "true"
    if not dry_run:
        for var in conditional_vars["Coinbase (Live Trading)"]:
            if not os.getenv(var):
                issues.append(f"Live trading requires: {var}")

    # Check for risky configurations
    leverage = float(os.getenv("TRADING__LEVERAGE", "5"))
    if leverage > 10:
        warnings.append(f"High leverage detected: {leverage}x")

    max_loss = float(os.getenv("RISK__MAX_DAILY_LOSS_PCT", "5"))
    if max_loss > 10:
        warnings.append(f"High daily loss limit: {max_loss}%")

    return issues, warnings


def check_file_structure():
    """Check that required files and directories exist."""
    print("📁 Checking file structure...")

    required_files = [
        ".env.example",
        "bot/config.py",
        "bot/config_utils.py",
        "bot/main.py",
    ]

    required_dirs = ["bot", "config", "logs", "data"]

    issues = []

    for file_path in required_files:
        if not Path(file_path).exists():
            issues.append(f"Missing file: {file_path}")

    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"  ✅ Created directory: {dir_path}")
            except Exception as e:
                issues.append(f"Cannot create directory {dir_path}: {e}")
        else:
            print(f"  ✅ Directory exists: {dir_path}")

    return issues


def check_python_requirements():
    """Check Python version and required packages."""
    print("🐍 Checking Python requirements...")

    issues = []

    # Check Python version
    if sys.version_info < (3, 8):
        issues.append(
            f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}"
        )
    else:
        print(
            f"  ✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )

    # Check required packages
    required_packages = ["pydantic", "pandas", "numpy", "requests"]
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ Package available: {package}")
        except ImportError:
            issues.append(f"Missing package: {package}")

    return issues


def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(".env")
    if env_file.exists():
        print("📄 Loading .env file...")
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        if key not in os.environ:  # Don't override existing env vars
                            os.environ[key] = value
            print("  ✅ Environment file loaded")
            return True
        except Exception as e:
            print(f"  ⚠️  Error loading .env file: {e}")
            return False
    else:
        print("  ⚠️  No .env file found (using .env.example as reference)")
        return False


def main():
    """Main validation function."""
    print("🤖 AI Trading Bot - Simple Configuration Validation")
    print("=" * 55)

    # Load environment file
    env_loaded = load_env_file()

    # Run all checks
    all_issues = []
    all_warnings = []

    # Check file structure
    file_issues = check_file_structure()
    all_issues.extend(file_issues)

    # Check Python requirements
    python_issues = check_python_requirements()
    all_issues.extend(python_issues)

    # Check environment variables
    env_issues, env_warnings = check_environment_variables()
    all_issues.extend(env_issues)
    all_warnings.extend(env_warnings)

    # Display results
    print("\n" + "=" * 55)
    print("📊 VALIDATION RESULTS")
    print("=" * 55)

    if all_issues:
        print(f"\n❌ Found {len(all_issues)} critical issues:")
        for issue in all_issues:
            print(f"   • {issue}")

    if all_warnings:
        print(f"\n⚠️  Found {len(all_warnings)} warnings:")
        for warning in all_warnings:
            print(f"   • {warning}")

    if not all_issues and not all_warnings:
        print("\n✅ All checks passed!")
    elif not all_issues:
        print("\n✅ No critical issues found (warnings can be addressed later)")

    # Provide next steps
    print("\n💡 NEXT STEPS:")

    if not env_loaded:
        print("   1. Copy .env.example to .env and configure your settings")
        print("      cp .env.example .env")

    if all_issues:
        print("   2. Address the critical issues listed above")
        print("   3. Re-run this validation script")
    else:
        print("   2. Run the full configuration validation:")
        print("      python -m bot.config_utils")
        print("   3. Start the bot in dry-run mode:")
        print("      python -m bot.main")

    print("\n📚 For detailed setup instructions, see:")
    print("   docs/Environment_Setup_Guide.md")

    return 1 if all_issues else 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
