#!/usr/bin/env python3
"""Validate type checking configuration and demonstrate strict typing."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_mypy_config() -> bool:
    """Verify MyPy configuration is properly set up."""
    try:
        import tomli
    except ImportError:
        import tomllib as tomli

    config_path = project_root / "pyproject.toml"
    if not config_path.exists():
        print("❌ pyproject.toml not found")
        return False

    with open(config_path, "rb") as f:
        config = tomli.load(f)

    mypy_config = config.get("tool", {}).get("mypy", {})

    required_settings = {
        "python_version": "3.12",
        "warn_return_any": True,
        "warn_unused_configs": True,
        "disallow_untyped_defs": True,
        "disallow_incomplete_defs": True,
        "check_untyped_defs": True,
        "disallow_untyped_decorators": True,
        "no_implicit_optional": True,
        "warn_redundant_casts": True,
        "warn_unused_ignores": True,
        "warn_no_return": True,
        "warn_unreachable": True,
        "strict_equality": True,
    }

    print("🔍 Checking MyPy configuration:")
    all_good = True
    for setting, expected in required_settings.items():
        actual = mypy_config.get(setting)
        if actual == expected:
            print(f"  ✅ {setting}: {expected}")
        else:
            print(f"  ❌ {setting}: expected {expected}, got {actual}")
            all_good = False

    return all_good


def check_pyright_config() -> bool:
    """Verify Pyright configuration exists."""
    config_path = project_root / "pyrightconfig.json"
    if config_path.exists():
        print("✅ pyrightconfig.json found")
        import json

        with open(config_path) as f:
            config = json.load(f)
        if config.get("typeCheckingMode") == "strict":
            print("  ✅ Strict mode enabled")
            return True
        print("  ❌ Strict mode not enabled")
        return False
    print("❌ pyrightconfig.json not found")
    return False


def check_type_stubs() -> list[str]:
    """Check for type stub files."""
    stubs_dir = project_root / "bot" / "types" / "stubs"
    if not stubs_dir.exists():
        print("❌ Type stubs directory not found")
        return []

    stub_files = list(stubs_dir.glob("*.pyi"))
    print(f"📦 Found {len(stub_files)} type stub files:")

    stubs = []
    for stub in sorted(stub_files):
        if stub.name != "__init__.py":
            print(f"  ✅ {stub.name}")
            stubs.append(stub.stem)

    return stubs


def validate_import_types() -> None:
    """Validate that type imports work correctly."""
    print("\n🔍 Validating type imports:")

    try:
        from bot.types.base_types import OrderType, PositionSide

        print("  ✅ Successfully imported base types")
    except ImportError as e:
        print(f"  ❌ Failed to import base types: {e}")

    try:
        from bot.types.exceptions import TradingError, ValidationError

        print("  ✅ Successfully imported exception types")
    except ImportError as e:
        print(f"  ❌ Failed to import exception types: {e}")

    try:
        from bot.types.guards import is_valid_order, is_valid_price

        print("  ✅ Successfully imported type guards")
    except ImportError as e:
        print(f"  ❌ Failed to import type guards: {e}")


def demonstrate_strict_typing() -> None:
    """Demonstrate strict typing with examples."""
    print("\n📝 Demonstrating strict typing:")

    # Example 1: Properly typed function
    def calculate_position_size(
        balance: float,
        risk_percentage: float,
        stop_loss_distance: float,
    ) -> float:
        """Calculate position size with full type annotations."""
        if risk_percentage <= 0 or risk_percentage > 100:
            raise ValueError(f"Invalid risk percentage: {risk_percentage}")

        risk_amount = balance * (risk_percentage / 100)
        position_size = risk_amount / stop_loss_distance
        return round(position_size, 8)

    print("  ✅ Example function with complete type annotations")

    # Example 2: Generic types
    from typing import Generic, TypeVar

    T = TypeVar("T")

    class Result(Generic[T]):
        """Generic result wrapper."""

        def __init__(self, value: T, success: bool, error: str | None = None) -> None:
            self.value = value
            self.success = success
            self.error = error

        def unwrap(self) -> T:
            """Unwrap the result or raise an exception."""
            if not self.success:
                raise RuntimeError(f"Result error: {self.error}")
            return self.value

    print("  ✅ Generic type example defined")

    # Example 3: Protocol for duck typing
    from typing import Protocol

    class Tradeable(Protocol):
        """Protocol for tradeable assets."""

        symbol: str
        price: float
        volume: float

        def can_trade(self) -> bool: ...

    print("  ✅ Protocol type example defined")


def run_type_check_command() -> tuple[bool, str]:
    """Run MyPy type checking on the project."""
    import subprocess

    print("\n🔬 Running MyPy type check...")

    try:
        result = subprocess.run(
            ["poetry", "run", "mypy", "bot/", "--config-file", "pyproject.toml"],
            check=False,
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        if result.returncode == 0:
            return True, "Type checking passed successfully!"
        # Count errors
        error_count = result.stdout.count("error:")
        return False, f"Found {error_count} type errors"

    except Exception as e:
        return False, f"Failed to run type check: {e}"


def main() -> int:
    """Main validation function."""
    print("🎯 AI Trading Bot Type Checking Validation")
    print("=" * 50)

    # Check configurations
    mypy_ok = check_mypy_config()
    print()

    pyright_ok = check_pyright_config()
    print()

    # Check type stubs
    stubs = check_type_stubs()
    print()

    # Validate imports
    validate_import_types()

    # Demonstrate typing
    demonstrate_strict_typing()

    # Run actual type check
    check_passed, message = run_type_check_command()
    print(f"  {'✅' if check_passed else '❌'} {message}")

    # Summary
    print("\n📊 Summary:")
    print(f"  {'✅' if mypy_ok else '❌'} MyPy configuration")
    print(f"  {'✅' if pyright_ok else '❌'} Pyright configuration")
    print(f"  {'✅' if stubs else '❌'} Type stubs ({len(stubs)} found)")
    print(f"  {'✅' if check_passed else '❌'} Type checking")

    # Return appropriate exit code
    if mypy_ok and pyright_ok and stubs and check_passed:
        print("\n✅ All type checking validations passed!")
        return 0
    print("\n❌ Some type checking validations failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
