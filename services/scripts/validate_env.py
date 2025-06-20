#!/usr/bin/env python3
"""
Environment validation script for AI Trading Bot.
Checks .env file existence, permissions, and validates all required variables.
"""

import re
import stat
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Constants for validation
MAX_LEVERAGE = 20  # Maximum leverage allowed
MIN_KEY_VALUE_LENGTH = 10  # Minimum length for key values


# Color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


class ValidationStatus(Enum):
    SUCCESS = "success"
    FAIL = "error"
    WARN = "warning"


@dataclass
class ValidationResult:
    status: ValidationStatus
    message: str
    fix: str | None = None


class EnvValidator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.env_path = self.project_root / ".env"
        self.env_example_path = self.project_root / ".env.example"
        self.errors: list[ValidationResult] = []
        self.warnings: list[ValidationResult] = []
        self.successes: list[ValidationResult] = []

    def run(self) -> bool:
        """Run all validation checks."""
        print(
            f"{Colors.BOLD}{Colors.BLUE}🔍 AI Trading Bot Environment Validator{Colors.END}\n"
        )

        # Check .env file existence
        if not self._check_env_exists():
            return False

        # Check file permissions
        self._check_file_permissions()

        # Load and parse .env file
        env_vars = self._load_env_file()
        if not env_vars:
            return False

        # Validate exchange configuration
        exchange_type = env_vars.get("EXCHANGE__EXCHANGE_TYPE", "").lower()
        self._validate_exchange_type(exchange_type, env_vars)

        # Validate common required variables
        self._validate_common_vars(env_vars)

        # Check for common mistakes
        self._check_common_mistakes(env_vars)

        # Display results
        self._display_results()

        return len(self.errors) == 0

    def _check_env_exists(self) -> bool:
        """Check if .env file exists."""
        if not self.env_path.exists():
            self.errors.append(
                ValidationResult(
                    status=ValidationStatus.FAIL,
                    message=".env file not found",
                    fix=f"Copy {self.env_example_path} to {self.env_path} and fill in your credentials:\n"
                    f"  cp {self.env_example_path} {self.env_path}",
                )
            )
            return False

        self.successes.append(
            ValidationResult(
                status=ValidationStatus.SUCCESS, message=".env file exists"
            )
        )
        return True

    def _check_file_permissions(self) -> None:
        """Check if .env file has secure permissions."""
        try:
            file_stat = self.env_path.stat()
            mode = file_stat.st_mode
            permissions = stat.filemode(mode)

            # Check if file is readable by group or others
            if mode & 0o044:
                self.warnings.append(
                    ValidationResult(
                        status=ValidationStatus.WARN,
                        message=f".env file permissions are too open: {permissions}",
                        fix=f"Secure your .env file with: chmod 600 {self.env_path}",
                    )
                )
            else:
                self.successes.append(
                    ValidationResult(
                        status=ValidationStatus.PASS,
                        message=f".env file has secure permissions: {permissions}",
                    )
                )
        except Exception as e:
            self.warnings.append(
                ValidationResult(
                    status=ValidationStatus.WARN,
                    message=f"Could not check file permissions: {e}",
                )
            )

    def _load_env_file(self) -> dict[str, str]:
        """Load and parse .env file."""
        env_vars = {}
        try:
            with self.env_path.open() as f:
                line_num = 0
                for file_line in f:
                    line_num += 1
                    line = file_line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue

                    # Check for valid KEY=VALUE format
                    if "=" not in line:
                        self.warnings.append(
                            ValidationResult(
                                status=ValidationStatus.WARN,
                                message=f"Line {line_num}: Invalid format (missing '='): {line[:50]}...",
                            )
                        )
                        continue

                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Remove quotes if present
                    if value and value[0] in "\"'":
                        if len(value) > 1 and value[0] == value[-1]:
                            value = value[1:-1]
                        else:
                            self.warnings.append(
                                ValidationResult(
                                    status=ValidationStatus.WARN,
                                    message=f"Line {line_num}: Unmatched quotes for {key}",
                                )
                            )

                    env_vars[key] = value

        except Exception as e:
            self.errors.append(
                ValidationResult(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to read .env file: {e}",
                )
            )
            return {}

        return env_vars

    def _validate_exchange_type(
        self, exchange_type: str, env_vars: dict[str, str]
    ) -> None:
        """Validate exchange type and related configuration."""
        valid_exchanges = ["coinbase", "bluefin"]

        if not exchange_type:
            self.errors.append(
                ValidationResult(
                    status=ValidationStatus.FAIL,
                    message="EXCHANGE__EXCHANGE_TYPE is not set",
                    fix="Add to .env: EXCHANGE__EXCHANGE_TYPE=coinbase  # or 'bluefin'",
                )
            )
            return

        if exchange_type not in valid_exchanges:
            self.errors.append(
                ValidationResult(
                    status=ValidationStatus.FAIL,
                    message=f"Invalid EXCHANGE__EXCHANGE_TYPE: '{exchange_type}'",
                    fix=f"Set EXCHANGE__EXCHANGE_TYPE to one of: {', '.join(valid_exchanges)}",
                )
            )
            return

        self.successes.append(
            ValidationResult(
                status=ValidationStatus.PASS,
                message=f"Exchange type is valid: {exchange_type}",
            )
        )

        # Validate exchange-specific variables
        if exchange_type == "coinbase":
            self._validate_coinbase_vars(env_vars)
        elif exchange_type == "bluefin":
            self._validate_bluefin_vars(env_vars)

    def _validate_coinbase_vars(self, env_vars: dict[str, str]) -> None:
        """Validate Coinbase-specific variables."""
        # CDP API Key Name
        api_key_name = env_vars.get("EXCHANGE__CDP_API_KEY_NAME", "")
        if not api_key_name:
            self.errors.append(
                ValidationResult(
                    status=ValidationStatus.FAIL,
                    message="EXCHANGE__CDP_API_KEY_NAME is not set",
                    fix="Add your Coinbase CDP API key name to .env",
                )
            )
        elif not re.match(r"^organizations/[^/]+/apiKeys/[^/]+$", api_key_name):
            self.warnings.append(
                ValidationResult(
                    status=ValidationStatus.WARN,
                    message="CDP API key name format looks unusual",
                    fix="Expected format: organizations/{org_id}/apiKeys/{key_id}",
                )
            )
        else:
            self.successes.append(
                ValidationResult(
                    status=ValidationStatus.PASS,
                    message="CDP API key name format is valid",
                )
            )

        # CDP Private Key
        private_key = env_vars.get("EXCHANGE__CDP_PRIVATE_KEY", "")
        if not private_key:
            self.errors.append(
                ValidationResult(
                    status=ValidationStatus.FAIL,
                    message="EXCHANGE__CDP_PRIVATE_KEY is not set",
                    fix="Add your Coinbase CDP private key (PEM format) to .env",
                )
            )
        elif not private_key.startswith("-----BEGIN EC PRIVATE KEY-----"):
            self.errors.append(
                ValidationResult(
                    status=ValidationStatus.FAIL,
                    message="CDP private key is not in PEM format",
                    fix="Ensure your private key starts with: -----BEGIN EC PRIVATE KEY-----",
                )
            )
        elif not private_key.endswith("-----END EC PRIVATE KEY-----"):
            self.errors.append(
                ValidationResult(
                    status=ValidationStatus.FAIL,
                    message="CDP private key is incomplete",
                    fix="Ensure your private key ends with: -----END EC PRIVATE KEY-----",
                )
            )
        else:
            self.successes.append(
                ValidationResult(
                    status=ValidationStatus.PASS,
                    message="CDP private key format is valid",
                )
            )

    def _validate_bluefin_vars(self, env_vars: dict[str, str]) -> None:
        """Validate Bluefin-specific variables with comprehensive validation."""
        # Split validation into smaller methods
        self._validate_bluefin_private_key(env_vars)
        self._validate_bluefin_network(env_vars)
        self._validate_bluefin_urls(env_vars)
        self._validate_bluefin_environment_consistency(env_vars)

    def _validate_bluefin_private_key(self, env_vars: dict[str, str]) -> None:
        """Validate Bluefin private key."""
        private_key = env_vars.get("EXCHANGE__BLUEFIN_PRIVATE_KEY", "")
        if not private_key:
            self.errors.append(
                ValidationResult(
                    status=ValidationStatus.FAIL,
                    message="EXCHANGE__BLUEFIN_PRIVATE_KEY is not set",
                    fix="Add your Sui wallet private key (hex, mnemonic, or Sui Bech32 format) to .env",
                )
            )
            return

        # Comprehensive private key validation
        validation_result = self._validate_bluefin_private_key_comprehensive(
            private_key
        )
        if validation_result:
            self.successes.append(validation_result)
        else:
            self.errors.append(
                ValidationResult(
                    status=ValidationStatus.FAIL,
                    message="Bluefin private key format is invalid",
                    fix="Use one of: 64-char hex (with/without 0x), 12/24-word mnemonic, or Sui Bech32 (suiprivkey...)",
                )
            )

    def _validate_bluefin_network(self, env_vars: dict[str, str]) -> None:
        """Validate Bluefin network setting."""
        network = env_vars.get("EXCHANGE__BLUEFIN_NETWORK", "mainnet").lower()
        if network not in ["mainnet", "testnet"]:
            self.warnings.append(
                ValidationResult(
                    status=ValidationStatus.WARN,
                    message=f"Invalid BLUEFIN_NETWORK: '{network}'",
                    fix="Set EXCHANGE__BLUEFIN_NETWORK to 'mainnet' or 'testnet'",
                )
            )
        else:
            self.successes.append(
                ValidationResult(
                    status=ValidationStatus.PASS,
                    message=f"Bluefin network is valid: {network}",
                )
            )

    def _validate_bluefin_urls(self, env_vars: dict[str, str]) -> None:
        """Validate Bluefin URL settings."""
        network = env_vars.get("EXCHANGE__BLUEFIN_NETWORK", "mainnet").lower()

        # Bluefin Service URL
        service_url = env_vars.get("EXCHANGE__BLUEFIN_SERVICE_URL", "")
        if service_url:
            if self._validate_url_format(service_url):
                self.successes.append(
                    ValidationResult(
                        status=ValidationStatus.PASS,
                        message="Bluefin service URL format is valid",
                    )
                )
            else:
                self.warnings.append(
                    ValidationResult(
                        status=ValidationStatus.WARN,
                        message="Bluefin service URL format looks invalid",
                        fix="Ensure URL includes protocol (http:// or https://)",
                    )
                )

        # Custom RPC URL (optional)
        rpc_url = env_vars.get("EXCHANGE__BLUEFIN_RPC_URL", "")
        if rpc_url:
            if self._validate_url_format(rpc_url):
                # Check network consistency
                if network == "mainnet" and "testnet" in rpc_url.lower():
                    self.warnings.append(
                        ValidationResult(
                            status=ValidationStatus.WARN,
                            message="Mainnet network with testnet RPC URL",
                            fix="Ensure RPC URL matches your network setting",
                        )
                    )
                elif network == "testnet" and "mainnet" in rpc_url.lower():
                    self.warnings.append(
                        ValidationResult(
                            status=ValidationStatus.WARN,
                            message="Testnet network with mainnet RPC URL",
                            fix="Ensure RPC URL matches your network setting",
                        )
                    )
                else:
                    self.successes.append(
                        ValidationResult(
                            status=ValidationStatus.PASS,
                            message="Custom RPC URL format is valid",
                        )
                    )
            else:
                self.warnings.append(
                    ValidationResult(
                        status=ValidationStatus.WARN,
                        message="Custom RPC URL format looks invalid",
                        fix="Ensure URL includes protocol and valid hostname",
                    )
                )

    def _validate_bluefin_environment_consistency(
        self, env_vars: dict[str, str]
    ) -> None:
        """Validate environment consistency for Bluefin."""
        network = env_vars.get("EXCHANGE__BLUEFIN_NETWORK", "mainnet").lower()
        environment = env_vars.get("SYSTEM__ENVIRONMENT", "development").lower()
        dry_run = env_vars.get("SYSTEM__DRY_RUN", "true").lower()

        if environment == "production" and network == "testnet":
            self.warnings.append(
                ValidationResult(
                    status=ValidationStatus.WARN,
                    message="Production environment using testnet network",
                    fix="Consider using mainnet for production or development for testnet",
                )
            )

        if dry_run == "false" and network == "testnet":
            self.warnings.append(
                ValidationResult(
                    status=ValidationStatus.WARN,
                    message="Live trading enabled on testnet network",
                    fix="Consider using paper trading (SYSTEM__DRY_RUN=true) for testnet",
                )
            )

    def _validate_bluefin_private_key_comprehensive(
        self, private_key: str
    ) -> ValidationResult | None:
        """Comprehensive validation of Bluefin private key formats."""
        private_key = private_key.strip()
        detected_formats = []

        # 1. Check for mnemonic phrase format (12 or 24 words)
        # Constants
        min_mnemonic_words = 12
        max_mnemonic_words = 24
        min_word_length = 2
        min_bech32_length = 50
        hex_key_length = 64
        bech32_prefix_length = 10  # Length of "suiprivkey" prefix

        words = private_key.split()
        if len(words) in [min_mnemonic_words, max_mnemonic_words]:
            if all(word.isalpha() and len(word) > min_word_length for word in words):
                detected_formats.append(f"mnemonic ({len(words)} words)")

                # Additional mnemonic validation
                if len(set(words)) < len(words) * 0.8:  # Too many repeated words
                    return None

        # 2. Check for Bech32-encoded Sui private key
        elif private_key.startswith("suiprivkey"):
            if len(private_key) >= min_bech32_length:  # Reasonable minimum length
                detected_formats.append("Sui Bech32")

                # Basic Bech32 character validation
                bech32_chars = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
                key_part = private_key[
                    bech32_prefix_length:
                ]  # Remove "suiprivkey" prefix
                if not all(c.lower() in bech32_chars for c in key_part):
                    return None

        # 3. Check for hex format (with or without 0x prefix)
        else:
            hex_key = private_key.removeprefix("0x")

            if len(hex_key) == hex_key_length and all(
                c in "0123456789abcdefABCDEF" for c in hex_key
            ):
                detected_formats.append("hex")

                # Check for obviously invalid keys
                if hex_key == "0" * 64 or hex_key.upper() == "F" * 64:
                    return None

        if detected_formats:
            return ValidationResult(
                status=ValidationStatus.PASS,
                message=f"Valid Bluefin private key ({', '.join(detected_formats)})",
            )

        return None

    def _validate_url_format(self, url: str) -> bool:
        """Validate URL format."""
        try:
            from urllib.parse import urlparse

            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def _validate_common_vars(self, env_vars: dict[str, str]) -> None:
        """Validate common required variables."""
        # OpenAI API Key
        openai_key = env_vars.get("LLM__OPENAI_API_KEY", "")
        if not openai_key:
            self.errors.append(
                ValidationResult(
                    status=ValidationStatus.FAIL,
                    message="LLM__OPENAI_API_KEY is not set",
                    fix="Add your OpenAI API key to .env",
                )
            )
        elif not openai_key.startswith("sk-"):
            self.warnings.append(
                ValidationResult(
                    status=ValidationStatus.WARN,
                    message="OpenAI API key format looks unusual",
                    fix="OpenAI keys typically start with 'sk-'",
                )
            )
        else:
            self.successes.append(
                ValidationResult(
                    status=ValidationStatus.PASS,
                    message="OpenAI API key format is valid",
                )
            )

        # Dry Run Mode
        dry_run = env_vars.get("SYSTEM__DRY_RUN", "").lower()
        if dry_run not in ["true", "false"]:
            self.errors.append(
                ValidationResult(
                    status=ValidationStatus.FAIL,
                    message=f"Invalid SYSTEM__DRY_RUN value: '{dry_run}'",
                    fix="Set SYSTEM__DRY_RUN to 'true' (paper trading) or 'false' (live trading)",
                )
            )
        elif dry_run == "false":
            self.warnings.append(
                ValidationResult(
                    status=ValidationStatus.WARN,
                    message="LIVE TRADING MODE is enabled! Real money will be used!",
                    fix="Set SYSTEM__DRY_RUN=true for paper trading",
                )
            )
        else:
            self.successes.append(
                ValidationResult(
                    status=ValidationStatus.PASS,
                    message="Paper trading mode is enabled (safe)",
                )
            )

        # Trading Symbol
        symbol = env_vars.get("TRADING__SYMBOL", "")
        if not symbol:
            self.warnings.append(
                ValidationResult(
                    status=ValidationStatus.WARN,
                    message="TRADING__SYMBOL is not set",
                    fix="Add TRADING__SYMBOL=BTC-USD (or your preferred pair) to .env",
                )
            )
        elif not re.match(r"^[A-Z]+-[A-Z]+$", symbol):
            self.warnings.append(
                ValidationResult(
                    status=ValidationStatus.WARN,
                    message=f"Trading symbol format looks unusual: '{symbol}'",
                    fix="Expected format: BASE-QUOTE (e.g., BTC-USD, ETH-USD)",
                )
            )
        else:
            self.successes.append(
                ValidationResult(
                    status=ValidationStatus.PASS,
                    message=f"Trading symbol is valid: {symbol}",
                )
            )

        # Trading Leverage (optional)
        leverage = env_vars.get("TRADING__LEVERAGE", "")
        if leverage:
            try:
                leverage_int = int(leverage)
                if leverage_int < 1 or leverage_int > MAX_LEVERAGE:
                    self.warnings.append(
                        ValidationResult(
                            status=ValidationStatus.WARN,
                            message=f"Leverage {leverage_int}x is outside typical range",
                            fix="Consider using leverage between 1x and 10x for safety",
                        )
                    )
            except ValueError:
                self.errors.append(
                    ValidationResult(
                        status=ValidationStatus.FAIL,
                        message=f"Invalid leverage value: '{leverage}'",
                        fix="TRADING__LEVERAGE must be a positive integer",
                    )
                )

    def _check_common_mistakes(self, env_vars: dict[str, str]) -> None:
        """Check for common configuration mistakes."""
        for key, value in env_vars.items():
            # Check for trailing spaces
            if value != value.strip():
                self.warnings.append(
                    ValidationResult(
                        status=ValidationStatus.WARN,
                        message=f"{key} has extra whitespace",
                        fix=f"Remove spaces from the beginning/end of {key}",
                    )
                )

            # Check for placeholder values
            if value in ["your-api-key-here", "YOUR_API_KEY", "<your-key>"]:
                self.errors.append(
                    ValidationResult(
                        status=ValidationStatus.FAIL,
                        message=f"{key} still has placeholder value",
                        fix=f"Replace the placeholder with your actual {key}",
                    )
                )

            # Check for exposed secrets in common mistake patterns
            if "key" in key.lower() and value and len(value) < MIN_KEY_VALUE_LENGTH:
                self.warnings.append(
                    ValidationResult(
                        status=ValidationStatus.WARN,
                        message=f"{key} seems too short",
                        fix="Double-check that you've entered the complete key",
                    )
                )

    def _display_results(self) -> None:
        """Display validation results in a user-friendly format."""
        print(f"{Colors.BOLD}Validation Results:{Colors.END}\n")

        # Display successes
        if self.successes:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ Passed Checks:{Colors.END}")
            for result in self.successes:
                print(
                    f"  {Colors.GREEN}{result.status.value}{Colors.END} {result.message}"
                )
            print()

        # Display warnings
        if self.warnings:
            print(f"{Colors.YELLOW}{Colors.BOLD}⚠ Warnings:{Colors.END}")
            for result in self.warnings:
                print(
                    f"  {Colors.YELLOW}{result.status.value}{Colors.END} {result.message}"
                )
                if result.fix:
                    print(f"    {Colors.BLUE}→ {result.fix}{Colors.END}")
            print()

        # Display errors
        if self.errors:
            print(f"{Colors.RED}{Colors.BOLD}✗ Errors:{Colors.END}")
            for result in self.errors:
                print(
                    f"  {Colors.RED}{result.status.value}{Colors.END} {result.message}"
                )
                if result.fix:
                    print(f"    {Colors.BLUE}→ {result.fix}{Colors.END}")
            print()

        # Summary
        total_checks = len(self.successes) + len(self.warnings) + len(self.errors)
        print(f"{Colors.BOLD}Summary:{Colors.END}")
        print(f"  Total checks: {total_checks}")
        print(f"  {Colors.GREEN}Passed: {len(self.successes)}{Colors.END}")
        print(f"  {Colors.YELLOW}Warnings: {len(self.warnings)}{Colors.END}")
        print(f"  {Colors.RED}Errors: {len(self.errors)}{Colors.END}")

        if self.errors:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ Validation FAILED{Colors.END}")
            print("Please fix the errors above before running the trading bot.")
            sys.exit(1)
        elif self.warnings:
            print(
                f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  Validation passed with warnings{Colors.END}"
            )
            print("The bot can run, but review the warnings for potential issues.")
        else:
            print(
                f"\n{Colors.GREEN}{Colors.BOLD}✅ All validations PASSED!{Colors.END}"
            )
            print("Your environment is properly configured.")


def main():
    """Main entry point."""
    validator = EnvValidator()
    validator.run()


if __name__ == "__main__":
    main()
