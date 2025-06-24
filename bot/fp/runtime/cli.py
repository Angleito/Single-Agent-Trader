"""
Functional CLI for Trading Bot

This module provides the command-line interface using functional effects.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import Any

from ..effects.config import ConfigSource, load_config, validate_config
from ..effects.io import IO
from ..effects.logging import LogConfig, LogLevel, configure_logging, error, info
from .interpreter import get_interpreter
from .scheduler import get_scheduler


@dataclass
class CLIConfig:
    """CLI configuration"""

    command: str
    options: dict[str, Any]
    config_path: str | None = None
    log_level: LogLevel = LogLevel.INFO

    # Enhanced options for compatibility with original CLI
    dry_run: bool | None = None
    symbol: str = "BTC-USD"
    interval: str = "1m"
    force: bool = False
    skip_health_check: bool = False
    market_making: bool | None = None
    mm_symbol: str | None = None
    mm_profile: str | None = None
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    initial_balance: float = 10000.0


class FunctionalCLI:
    """Functional command-line interface"""

    def __init__(self):
        self.interpreter = get_interpreter()
        self.scheduler = get_scheduler()

    def parse_args_from_dict(self, args_dict: dict[str, Any]) -> CLIConfig:
        """Parse arguments from a dictionary (for integration with Click)"""
        command = args_dict.get("command", "help")

        # Extract all the options and map them to CLIConfig fields
        config_options = CLIConfig(
            command=command,
            options=args_dict,
            config_path=args_dict.get("config"),
            log_level=LogLevel.DEBUG if args_dict.get("debug") else LogLevel.INFO,
            dry_run=args_dict.get("dry_run"),
            symbol=args_dict.get("symbol", "BTC-USD"),
            interval=args_dict.get("interval", "1m"),
            force=args_dict.get("force", False),
            skip_health_check=args_dict.get("skip_health_check", False),
            market_making=args_dict.get("market_making"),
            mm_symbol=args_dict.get("mm_symbol"),
            mm_profile=args_dict.get("mm_profile"),
            start_date=args_dict.get("start_date", "2024-01-01"),
            end_date=args_dict.get("end_date", "2024-12-31"),
            initial_balance=args_dict.get("initial_balance", 10000.0),
        )

        return config_options

    def parse_args(self, args: list[str]) -> CLIConfig:
        """Parse command line arguments"""
        if not args:
            return CLIConfig(command="help", options={})

        command = args[0]
        options = {}

        # Enhanced argument parsing with support for complex options
        i = 1
        while i < len(args):
            if args[i].startswith("--"):
                key = args[i][2:].replace("-", "_")  # Convert kebab-case to snake_case

                # Handle boolean flags and options with values
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    value = args[i + 1]
                    # Convert string values to appropriate types
                    if value.lower() in ("true", "false"):
                        value = value.lower() == "true"
                    elif value.replace(".", "").replace("-", "").isdigit():
                        value = float(value) if "." in value else int(value)

                    options[key] = value
                    i += 2
                else:
                    options[key] = True
                    i += 1
            else:
                i += 1

        # Handle special cases for backward compatibility
        options["command"] = command

        return self.parse_args_from_dict(options)

    def initialize_system(self, cli_config: CLIConfig) -> IO[None]:
        """Initialize the system"""

        def init():
            # Configure logging
            log_config = LogConfig(level=cli_config.log_level)
            configure_logging(log_config).run()

            info(
                "Initializing functional trading bot",
                {"command": cli_config.command, "options": cli_config.options},
            ).run()

            # Load configuration
            if cli_config.config_path:
                config_result = load_config(
                    ConfigSource.FILE, cli_config.config_path
                ).run()
                if config_result.is_right():
                    validated = validate_config(config_result.value).run()
                    if not validated.validated:
                        raise ValueError(f"Invalid config: {validated.errors}")

        return IO(init)

    def run_live_trading(self, cli_config: CLIConfig) -> IO[None]:
        """Run live trading mode"""

        def run():
            info(
                "Starting live trading mode",
                {
                    "symbol": cli_config.symbol,
                    "interval": cli_config.interval,
                    "dry_run": cli_config.dry_run,
                    "market_making": cli_config.market_making,
                    "mm_symbol": cli_config.mm_symbol,
                    "mm_profile": cli_config.mm_profile,
                },
            ).run()

            # TODO: Implement actual live trading logic using the functional runtime
            # For now, print the configuration that would be used
            print("Live trading configuration:")
            print(f"  Symbol: {cli_config.symbol}")
            print(f"  Interval: {cli_config.interval}")
            print(f"  Dry run: {cli_config.dry_run}")
            print(f"  Market making: {cli_config.market_making}")
            if cli_config.market_making:
                print(f"  MM Symbol: {cli_config.mm_symbol}")
                print(f"  MM Profile: {cli_config.mm_profile}")

            # Start the scheduler
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                loop.run_until_complete(self.scheduler.run_loop())
            except KeyboardInterrupt:
                info("Received interrupt signal, stopping...").run()
                self.scheduler.stop()
            finally:
                loop.close()

        return IO(run)

    def run_backtest(self, cli_config: CLIConfig) -> IO[None]:
        """Run backtesting mode"""

        def run():
            info(
                "Starting backtest mode",
                {
                    "symbol": cli_config.symbol,
                    "start_date": cli_config.start_date,
                    "end_date": cli_config.end_date,
                    "initial_balance": cli_config.initial_balance,
                    "market_making": cli_config.market_making,
                    "mm_symbol": cli_config.mm_symbol,
                    "mm_profile": cli_config.mm_profile,
                },
            ).run()

            # TODO: Implement actual backtesting logic using the functional runtime
            # For now, print the configuration that would be used
            print("Backtesting configuration:")
            print(f"  Symbol: {cli_config.symbol}")
            print(f"  Period: {cli_config.start_date} to {cli_config.end_date}")
            print(f"  Initial balance: ${cli_config.initial_balance:,.2f}")
            print(f"  Market making: {cli_config.market_making}")
            if cli_config.market_making:
                print(f"  MM Symbol: {cli_config.mm_symbol}")
                print(f"  MM Profile: {cli_config.mm_profile}")

        return IO(run)

    def show_help(self) -> IO[None]:
        """Show help information"""

        def show():
            help_text = """
Functional Trading Bot CLI

Usage:
    bot [command] [options]

Commands:
    live        Run live trading
    backtest    Run backtesting
    status      Show system status
    help        Show this help

Options:
    --config PATH       Configuration file path
    --debug             Enable debug logging
    --dry-run           Run in paper trading mode
    --symbol SYMBOL     Trading symbol (default: BTC-USD)
    --from DATE         Start date for backtest
    --to DATE           End date for backtest

Examples:
    bot live --config config.json
    bot backtest --from 2024-01-01 --to 2024-12-31
    bot status
"""
            print(help_text)

        return IO(show)

    def show_status(self) -> IO[None]:
        """Show system status"""

        def show():
            status = {
                "scheduler": self.scheduler.get_status(),
                "interpreter": self.interpreter.get_runtime_stats(),
            }

            print("System Status:")
            print(f"Scheduler running: {status['scheduler']['running']}")
            print(
                f"Active tasks: {len([t for t in status['scheduler']['tasks'].values() if t['enabled']])}"
            )
            print(f"Active effects: {status['interpreter']['active_effects']}")

        return IO(show)

    def execute_command(self, cli_config: CLIConfig) -> IO[int]:
        """Execute a CLI command"""

        def execute():
            try:
                # Initialize system
                self.initialize_system(cli_config).run()

                # Execute command
                if cli_config.command == "live":
                    self.run_live_trading(cli_config).run()
                elif cli_config.command == "backtest":
                    self.run_backtest(cli_config).run()
                elif cli_config.command == "status":
                    self.show_status().run()
                elif cli_config.command == "help":
                    self.show_help().run()
                else:
                    error(f"Unknown command: {cli_config.command}").run()
                    return 1

                return 0

            except Exception as e:
                error(f"Command execution failed: {e!s}").run()
                return 1

        return IO(execute)

    def execute_live_command(
        self,
        dry_run: bool | None = None,
        symbol: str = "BTC-USD",
        interval: str = "1m",
        config: str | None = None,
        force: bool = False,
        skip_health_check: bool = False,
        market_making: bool | None = None,
        mm_symbol: str | None = None,
        mm_profile: str | None = None,
    ) -> int:
        """Execute live trading command with all parameters"""
        cli_config = self.parse_args_from_dict(
            {
                "command": "live",
                "dry_run": dry_run,
                "symbol": symbol,
                "interval": interval,
                "config": config,
                "force": force,
                "skip_health_check": skip_health_check,
                "market_making": market_making,
                "mm_symbol": mm_symbol,
                "mm_profile": mm_profile,
            }
        )

        return self.execute_command(cli_config).run()

    def execute_backtest_command(
        self,
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        symbol: str = "BTC-USD",
        initial_balance: float = 10000.0,
        market_making: bool = False,
        mm_symbol: str | None = None,
        mm_profile: str = "moderate",
    ) -> int:
        """Execute backtest command with all parameters"""
        cli_config = self.parse_args_from_dict(
            {
                "command": "backtest",
                "start_date": start_date,
                "end_date": end_date,
                "symbol": symbol,
                "initial_balance": initial_balance,
                "market_making": market_making,
                "mm_symbol": mm_symbol,
                "mm_profile": mm_profile,
            }
        )

        return self.execute_command(cli_config).run()


def main(args: list[str] | None = None) -> int:
    """Main CLI entry point"""
    if args is None:
        args = sys.argv[1:]

    cli = FunctionalCLI()
    cli_config = cli.parse_args(args)

    return cli.execute_command(cli_config).run()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
