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


class FunctionalCLI:
    """Functional command-line interface"""

    def __init__(self):
        self.interpreter = get_interpreter()
        self.scheduler = get_scheduler()

    def parse_args(self, args: list[str]) -> CLIConfig:
        """Parse command line arguments"""
        if not args:
            return CLIConfig(command="help", options={})

        command = args[0]
        options = {}

        # Simple argument parsing
        i = 1
        while i < len(args):
            if args[i].startswith("--"):
                key = args[i][2:]
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    value = args[i + 1]
                    i += 2
                else:
                    value = True
                    i += 1
                options[key] = value
            else:
                i += 1

        return CLIConfig(
            command=command,
            options=options,
            config_path=options.get("config"),
            log_level=LogLevel.DEBUG if options.get("debug") else LogLevel.INFO,
        )

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

    def run_live_trading(self, options: dict[str, Any]) -> IO[None]:
        """Run live trading mode"""

        def run():
            info("Starting live trading mode", options).run()

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

    def run_backtest(self, options: dict[str, Any]) -> IO[None]:
        """Run backtesting mode"""

        def run():
            info("Starting backtest mode", options).run()

            # Implement backtesting logic
            from_date = options.get("from")
            to_date = options.get("to")
            symbol = options.get("symbol", "BTC-USD")

            print(f"Backtesting {symbol} from {from_date} to {to_date}")

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
                    self.run_live_trading(cli_config.options).run()
                elif cli_config.command == "backtest":
                    self.run_backtest(cli_config.options).run()
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
