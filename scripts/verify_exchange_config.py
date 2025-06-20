#!/usr/bin/env python3
"""
Exchange Configuration Verification Script

This script helps users verify their exchange configuration is properly set up
for the AI trading bot. It checks environment variables, validates settings,
and provides helpful feedback.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import bot modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from bot.config import ExchangeType, Settings
except ImportError as e:
    console = Console()
    console.print("[red]Error: Could not import bot configuration.[/red]")
    console.print(f"[yellow]Details: {e}[/yellow]")
    console.print(
        "\n[cyan]Please ensure you're running this from the project root:[/cyan]"
    )
    console.print("  python scripts/verify_exchange_config.py")
    sys.exit(1)

console = Console()


def check_env_file() -> bool:
    """Check if .env file exists."""
    env_path = Path(".env")
    if not env_path.exists():
        console.print("[red]✗ .env file not found![/red]")
        console.print(
            "[yellow]  Please copy .env.example to .env and configure it.[/yellow]"
        )
        return False
    console.print("[green]✓ .env file found[/green]")
    return True


def load_settings() -> Settings:
    """Load settings from environment."""
    try:
        # Load environment variables
        load_dotenv()
        return Settings()
    except Exception as e:
        console.print(f"[red]Error loading settings: {e}[/red]")
        sys.exit(1)


def check_required_vars(exchange_type: ExchangeType) -> dict[str, bool]:
    """Check if required environment variables are set for the selected exchange."""
    results = {}

    # Common required variables
    common_vars = {
        "LLM__OPENAI_API_KEY": os.getenv("LLM__OPENAI_API_KEY"),
        "SYSTEM__DRY_RUN": os.getenv("SYSTEM__DRY_RUN"),
        "TRADING__SYMBOL": os.getenv("TRADING__SYMBOL"),
    }

    for var, value in common_vars.items():
        results[var] = bool(value)

    # Exchange-specific variables
    if exchange_type == ExchangeType.COINBASE:
        coinbase_vars = {
            "EXCHANGE__CDP_API_KEY_NAME": os.getenv("EXCHANGE__CDP_API_KEY_NAME"),
            "EXCHANGE__CDP_PRIVATE_KEY": os.getenv("EXCHANGE__CDP_PRIVATE_KEY"),
        }
        for var, value in coinbase_vars.items():
            results[var] = bool(value)

    elif exchange_type == ExchangeType.BLUEFIN:
        bluefin_vars = {
            "EXCHANGE__BLUEFIN_PRIVATE_KEY": os.getenv("EXCHANGE__BLUEFIN_PRIVATE_KEY"),
            "EXCHANGE__BLUEFIN_NETWORK": os.getenv("EXCHANGE__BLUEFIN_NETWORK"),
        }
        for var, value in bluefin_vars.items():
            results[var] = bool(value)

    return results


def validate_symbol_format(
    symbol: str, exchange_type: ExchangeType
) -> tuple[bool, str]:
    """Validate trading symbol format for the exchange."""
    if exchange_type == ExchangeType.COINBASE:
        # Coinbase uses format like "BTC-USD", "ETH-USD"
        if "-" in symbol and len(symbol.split("-")) == 2:
            base, quote = symbol.split("-")
            if base.isalpha() and quote.isalpha():
                return True, "Valid Coinbase format (BASE-QUOTE)"
            else:
                return False, "Invalid format: Both parts should be alphabetic"
        else:
            return False, "Coinbase requires BASE-QUOTE format (e.g., BTC-USD)"

    elif exchange_type == ExchangeType.BLUEFIN:
        # Bluefin uses format like "BTC-PERP", "ETH-PERP"
        if "-" in symbol and symbol.endswith("-PERP"):
            base = symbol.replace("-PERP", "")
            if base.isalpha():
                return True, "Valid Bluefin format (BASE-PERP)"
            else:
                return False, "Invalid format: Base currency should be alphabetic"
        else:
            return False, "Bluefin requires BASE-PERP format (e.g., BTC-PERP)"

    return False, "Unknown exchange type"


def display_configuration(settings: Settings, env_check: dict[str, bool]) -> None:
    """Display the current configuration in a formatted table."""
    # Main configuration panel
    config_table = Table(title="Exchange Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")
    config_table.add_column("Status", style="green")

    # Exchange type
    exchange_display = str(settings.exchange.exchange_type.value).upper()
    config_table.add_row(
        "Exchange Type",
        exchange_display,
        "✓" if settings.exchange.exchange_type else "✗",
    )

    # Trading mode
    mode = "Paper Trading" if settings.system.dry_run else "LIVE TRADING"
    mode_style = "yellow" if settings.system.dry_run else "red bold"
    config_table.add_row("Trading Mode", f"[{mode_style}]{mode}[/{mode_style}]", "✓")

    # Trading symbol
    symbol_valid, symbol_msg = validate_symbol_format(
        settings.trading.symbol, settings.exchange.exchange_type
    )
    config_table.add_row(
        "Trading Symbol", settings.trading.symbol, "✓" if symbol_valid else "✗"
    )

    # Leverage
    config_table.add_row("Leverage", f"{settings.trading.leverage}x", "✓")

    # Network (Bluefin only)
    if settings.exchange.exchange_type == ExchangeType.BLUEFIN:
        network = settings.exchange.bluefin_network or "mainnet"
        config_table.add_row("Bluefin Network", network.upper(), "✓")

    console.print("\n")
    console.print(config_table)

    # Environment variables check
    env_table = Table(title="Environment Variables", box=box.ROUNDED)
    env_table.add_column("Variable", style="cyan")
    env_table.add_column("Status", style="white")

    for var, is_set in env_check.items():
        status = "[green]✓ Set[/green]" if is_set else "[red]✗ Not Set[/red]"
        env_table.add_row(var, status)

    console.print("\n")
    console.print(env_table)

    # Symbol validation message
    if not symbol_valid:
        console.print(f"\n[yellow]⚠ Symbol Format Issue: {symbol_msg}[/yellow]")


def display_recommendations(settings: Settings, env_check: dict[str, bool]) -> None:
    """Display recommendations based on the configuration."""
    recommendations = []

    # Check for missing environment variables
    missing_vars = [var for var, is_set in env_check.items() if not is_set]
    if missing_vars:
        recommendations.append(
            "[red]Missing Required Variables:[/red]\n"
            + "\n".join(f"  • {var}" for var in missing_vars)
        )

    # Symbol format recommendation
    symbol_valid, symbol_msg = validate_symbol_format(
        settings.trading.symbol, settings.exchange.exchange_type
    )
    if not symbol_valid:
        if settings.exchange.exchange_type == ExchangeType.COINBASE:
            recommendations.append(
                "[yellow]Symbol Format:[/yellow]\n"
                "  • Use BASE-QUOTE format (e.g., BTC-USD, ETH-USD)\n"
                "  • Common pairs: BTC-USD, ETH-USD, SOL-USD"
            )
        elif settings.exchange.exchange_type == ExchangeType.BLUEFIN:
            recommendations.append(
                "[yellow]Symbol Format:[/yellow]\n"
                "  • Use BASE-PERP format (e.g., BTC-PERP, ETH-PERP)\n"
                "  • Common pairs: BTC-PERP, ETH-PERP, SOL-PERP"
            )

    # Trading mode warning
    if not settings.system.dry_run:
        recommendations.append(
            "[red bold]⚠ LIVE TRADING MODE ENABLED ⚠[/red bold]\n"
            "  • Real money will be used!\n"
            "  • Ensure all settings are correct\n"
            "  • Consider testing with SYSTEM__DRY_RUN=true first"
        )

    # Exchange-specific recommendations
    if settings.exchange.exchange_type == ExchangeType.BLUEFIN:
        if settings.exchange.bluefin_network == "testnet":
            recommendations.append(
                "[cyan]Bluefin Testnet:[/cyan]\n"
                "  • Using testnet is good for testing\n"
                "  • Get testnet SUI from: https://docs.sui.io/guides/developer/getting-started/sui-token"
            )

    if recommendations:
        console.print("\n")
        console.print(
            Panel(
                "\n\n".join(recommendations),
                title="Recommendations",
                border_style="yellow",
            )
        )


def main():
    """Main function to verify exchange configuration."""
    console.print(
        Panel(
            "[bold cyan]Exchange Configuration Verifier[/bold cyan]\n"
            "This tool checks your trading bot configuration",
            box=box.ROUNDED,
        )
    )

    # Check for .env file
    if not check_env_file():
        return

    # Load settings
    console.print("\n[cyan]Loading configuration...[/cyan]")
    try:
        settings = load_settings()
        console.print("[green]✓ Configuration loaded successfully[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to load configuration: {e}[/red]")
        return

    # Check required environment variables
    env_check = check_required_vars(settings.exchange.exchange_type)

    # Display configuration
    display_configuration(settings, env_check)

    # Display recommendations
    display_recommendations(settings, env_check)

    # Overall status
    all_vars_set = all(env_check.values())
    symbol_valid, _ = validate_symbol_format(
        settings.trading.symbol, settings.exchange.exchange_type
    )

    if all_vars_set and symbol_valid:
        console.print("\n[green bold]✓ Configuration appears to be valid![/green bold]")
        if settings.system.dry_run:
            console.print("[cyan]Ready for paper trading.[/cyan]")
        else:
            console.print(
                "[yellow]Ready for live trading. Please double-check all settings![/yellow]"
            )
    else:
        console.print("\n[red bold]✗ Configuration issues detected![/red bold]")
        console.print("[yellow]Please address the recommendations above.[/yellow]")


if __name__ == "__main__":
    main()
