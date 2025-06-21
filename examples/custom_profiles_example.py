"""
Custom Market Making Profile Creation Examples.

This module demonstrates how to create custom market making profiles for different
trading scenarios, risk tolerances, and market conditions.

Features:
- Profile creation templates
- Risk parameter configuration
- Performance optimization profiles
- Market condition adaptive profiles
- Custom indicator integrations
- Profile validation and testing
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


class CustomProfileCreator:
    """
    Custom market making profile creation utility.

    Provides tools and templates for creating custom market making configurations
    tailored to specific trading requirements and market conditions.
    """

    def __init__(self):
        """Initialize the profile creator."""
        self.base_templates = self._load_base_templates()
        self.validation_rules = self._define_validation_rules()

    def _load_base_templates(self) -> dict[str, dict[str, Any]]:
        """Load base configuration templates."""
        return {
            "scalping": {
                "description": "Ultra-fast scalping profile for quick profits",
                "strategy": {
                    "base_spread_bps": 2,
                    "min_spread_bps": 1,
                    "max_spread_bps": 8,
                    "order_levels": 2,
                    "max_position_pct": 15,
                    "bias_adjustment_factor": 0.9,
                },
                "cycle_interval": 0.1,
                "risk": {
                    "max_position_value": 5000,
                    "stop_loss_pct": 0.5,
                    "daily_loss_limit_pct": 2.0,
                },
                "orders": {
                    "order_update_interval_seconds": 0.2,
                    "price_update_threshold_bps": 1,
                },
            },
            "swing_trading": {
                "description": "Longer-term swing trading with wider spreads",
                "strategy": {
                    "base_spread_bps": 50,
                    "min_spread_bps": 30,
                    "max_spread_bps": 200,
                    "order_levels": 4,
                    "max_position_pct": 35,
                    "bias_adjustment_factor": 0.4,
                },
                "cycle_interval": 5.0,
                "risk": {
                    "max_position_value": 20000,
                    "stop_loss_pct": 3.0,
                    "daily_loss_limit_pct": 8.0,
                },
                "orders": {
                    "order_update_interval_seconds": 10.0,
                    "price_update_threshold_bps": 10,
                },
            },
            "volatility_adaptive": {
                "description": "Adapts to market volatility automatically",
                "strategy": {
                    "base_spread_bps": 15,
                    "min_spread_bps": 5,
                    "max_spread_bps": 100,
                    "order_levels": 3,
                    "max_position_pct": 25,
                    "bias_adjustment_factor": 0.6,
                    "volatility_multiplier": 2.0,
                },
                "cycle_interval": 1.0,
                "risk": {
                    "max_position_value": 15000,
                    "volatility_threshold": 0.05,
                    "adaptive_position_sizing": True,
                },
            },
            "news_trader": {
                "description": "Optimized for news events and market announcements",
                "strategy": {
                    "base_spread_bps": 25,
                    "min_spread_bps": 10,
                    "max_spread_bps": 150,
                    "order_levels": 5,
                    "max_position_pct": 30,
                    "news_sensitivity": 0.8,
                },
                "cycle_interval": 0.5,
                "risk": {
                    "max_position_value": 12000,
                    "news_stop_trading": True,
                    "volatility_circuit_breaker": 0.1,
                },
            },
        }

    def _define_validation_rules(self) -> dict[str, Any]:
        """Define validation rules for profile parameters."""
        return {
            "strategy": {
                "base_spread_bps": {"min": 1, "max": 500, "type": "number"},
                "min_spread_bps": {"min": 0.5, "max": 100, "type": "number"},
                "max_spread_bps": {"min": 5, "max": 1000, "type": "number"},
                "order_levels": {"min": 1, "max": 10, "type": "integer"},
                "max_position_pct": {"min": 1, "max": 50, "type": "number"},
                "bias_adjustment_factor": {"min": 0.0, "max": 1.0, "type": "number"},
            },
            "risk": {
                "max_position_value": {"min": 100, "max": 100000, "type": "number"},
                "stop_loss_pct": {"min": 0.1, "max": 10.0, "type": "number"},
                "daily_loss_limit_pct": {"min": 1.0, "max": 20.0, "type": "number"},
            },
            "cycle_interval": {"min": 0.05, "max": 60.0, "type": "number"},
        }

    def create_custom_profile(
        self,
        name: str,
        base_template: str = "scalping",
        customizations: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a custom profile based on a template with customizations.

        Args:
            name: Name for the custom profile
            base_template: Base template to start from
            customizations: Custom parameters to override

        Returns:
            Complete custom profile configuration
        """
        if base_template not in self.base_templates:
            raise ValueError(f"Unknown base template: {base_template}")

        # Start with base template
        profile = self.base_templates[base_template].copy()
        profile["name"] = name
        profile["created_at"] = datetime.now(UTC).isoformat()
        profile["base_template"] = base_template

        # Apply customizations
        if customizations:
            profile = self._deep_merge_configs(profile, customizations)

        # Validate the profile
        validation_result = self.validate_profile(profile)
        if not validation_result["valid"]:
            raise ValueError(f"Invalid profile: {validation_result['errors']}")

        return profile

    def _deep_merge_configs(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge two configuration dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def validate_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        """
        Validate a profile configuration.

        Args:
            profile: Profile configuration to validate

        Returns:
            Validation result with success status and any errors
        """
        errors = []
        warnings = []

        # Validate required sections
        required_sections = ["strategy", "risk"]
        for section in required_sections:
            if section not in profile:
                errors.append(f"Missing required section: {section}")
                continue

            # Validate section parameters
            if section in self.validation_rules:
                section_errors = self._validate_section(
                    profile[section], self.validation_rules[section]
                )
                errors.extend([f"{section}.{error}" for error in section_errors])

        # Validate top-level parameters
        if "cycle_interval" in profile:
            if not self._validate_parameter(
                profile["cycle_interval"], self.validation_rules["cycle_interval"]
            ):
                errors.append("cycle_interval: Invalid value")

        # Business logic validations
        strategy = profile.get("strategy", {})
        if strategy.get("min_spread_bps", 0) >= strategy.get(
            "max_spread_bps", float("inf")
        ):
            errors.append("strategy: min_spread_bps must be less than max_spread_bps")

        if strategy.get("base_spread_bps", 0) < strategy.get("min_spread_bps", 0):
            warnings.append("strategy: base_spread_bps is less than min_spread_bps")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _validate_section(
        self, section: dict[str, Any], rules: dict[str, Any]
    ) -> list[str]:
        """Validate a configuration section against rules."""
        errors = []

        for param, value in section.items():
            if param in rules:
                if not self._validate_parameter(value, rules[param]):
                    errors.append(f"{param}: Invalid value {value}")

        return errors

    def _validate_parameter(self, value: Any, rule: dict[str, Any]) -> bool:
        """Validate a single parameter against its rule."""
        try:
            if rule["type"] == "number":
                val = float(value)
                return rule["min"] <= val <= rule["max"]
            if rule["type"] == "integer":
                val = int(value)
                return rule["min"] <= val <= rule["max"]
            return True
        except (ValueError, TypeError):
            return False

    def optimize_profile_for_market(
        self, base_profile: dict[str, Any], market_conditions: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Optimize a profile for specific market conditions.

        Args:
            base_profile: Base profile to optimize
            market_conditions: Current market conditions

        Returns:
            Optimized profile configuration
        """
        optimized = base_profile.copy()

        volatility = market_conditions.get("volatility", 0.02)
        volume = market_conditions.get("avg_volume", 1000000)
        trend_strength = market_conditions.get("trend_strength", 0.0)

        # Adjust spreads based on volatility
        if volatility > 0.05:  # High volatility
            multiplier = min(volatility / 0.02, 3.0)  # Cap at 3x
            optimized["strategy"]["base_spread_bps"] = int(
                optimized["strategy"]["base_spread_bps"] * multiplier
            )
            optimized["strategy"]["max_spread_bps"] = int(
                optimized["strategy"]["max_spread_bps"] * multiplier
            )
        elif volatility < 0.01:  # Low volatility
            multiplier = max(volatility / 0.02, 0.5)  # Floor at 0.5x
            optimized["strategy"]["base_spread_bps"] = int(
                optimized["strategy"]["base_spread_bps"] * multiplier
            )

        # Adjust position sizing based on volume
        if volume < 500000:  # Low volume
            optimized["strategy"]["max_position_pct"] *= 0.7
            optimized["strategy"]["order_levels"] = min(
                optimized["strategy"]["order_levels"], 2
            )
        elif volume > 5000000:  # High volume
            optimized["strategy"]["max_position_pct"] *= 1.3

        # Adjust bias factor based on trend strength
        if abs(trend_strength) > 0.7:  # Strong trend
            optimized["strategy"]["bias_adjustment_factor"] = min(
                optimized["strategy"]["bias_adjustment_factor"] * 1.5, 1.0
            )

        # Ensure values remain within valid ranges
        validation_result = self.validate_profile(optimized)
        if not validation_result["valid"]:
            logger.warning(
                f"Optimization resulted in invalid profile: {validation_result['errors']}"
            )
            return base_profile

        return optimized

    def generate_profile_examples(self) -> None:
        """Generate and display example custom profiles."""
        console.print(
            Panel.fit(
                "[bold green]Custom Market Making Profile Examples[/bold green]\n\n"
                "Demonstrating various custom profile configurations for different trading scenarios.",
                title="🎯 Custom Profiles",
            )
        )

        examples = [
            {
                "name": "crypto_scalper",
                "base": "scalping",
                "customizations": {
                    "strategy": {"base_spread_bps": 3, "max_position_pct": 20},
                    "risk": {"daily_loss_limit_pct": 1.5},
                },
                "description": "Optimized for cryptocurrency scalping",
            },
            {
                "name": "forex_swing",
                "base": "swing_trading",
                "customizations": {
                    "strategy": {"base_spread_bps": 30, "bias_adjustment_factor": 0.3},
                    "cycle_interval": 10.0,
                },
                "description": "Designed for forex swing trading",
            },
            {
                "name": "volatile_markets",
                "base": "volatility_adaptive",
                "customizations": {
                    "strategy": {"max_spread_bps": 200, "volatility_multiplier": 3.0},
                    "risk": {"volatility_threshold": 0.08},
                },
                "description": "Handles extreme market volatility",
            },
        ]

        for example in examples:
            try:
                profile = self.create_custom_profile(
                    name=example["name"],
                    base_template=example["base"],
                    customizations=example["customizations"],
                )

                self._display_profile_example(profile, example["description"])

            except Exception as e:
                console.print(f"[red]Error creating {example['name']}: {e}[/red]")

    def _display_profile_example(
        self, profile: dict[str, Any], description: str
    ) -> None:
        """Display a profile example."""
        console.print(f"\n[bold cyan]Profile: {profile['name']}[/bold cyan]")
        console.print(f"Description: {description}")
        console.print(f"Base Template: {profile['base_template']}")

        # Create summary table
        table = Table()
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Category", style="green")

        # Add key parameters
        strategy = profile.get("strategy", {})
        risk = profile.get("risk", {})

        table.add_row(
            "Base Spread", f"{strategy.get('base_spread_bps', 0)} bps", "Strategy"
        )
        table.add_row("Order Levels", str(strategy.get("order_levels", 0)), "Strategy")
        table.add_row(
            "Max Position", f"{strategy.get('max_position_pct', 0)}%", "Strategy"
        )
        table.add_row(
            "Cycle Interval", f"{profile.get('cycle_interval', 0)}s", "Performance"
        )
        table.add_row(
            "Max Position Value", f"${risk.get('max_position_value', 0):,}", "Risk"
        )
        table.add_row(
            "Daily Loss Limit", f"{risk.get('daily_loss_limit_pct', 0)}%", "Risk"
        )

        console.print(table)

    def save_profile(self, profile: dict[str, Any], filename: str | None = None) -> str:
        """
        Save a profile to a JSON file.

        Args:
            profile: Profile configuration to save
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = f"custom_profile_{profile.get('name', 'unnamed')}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"

        # Ensure the profiles directory exists
        profiles_dir = Path("config/profiles")
        profiles_dir.mkdir(parents=True, exist_ok=True)

        filepath = profiles_dir / filename

        # Save with pretty formatting
        with open(filepath, "w") as f:
            json.dump(profile, f, indent=2, default=str)

        console.print(f"✅ Profile saved to: {filepath}")
        return str(filepath)

    def load_profile(self, filepath: str) -> dict[str, Any]:
        """
        Load a profile from a JSON file.

        Args:
            filepath: Path to the profile file

        Returns:
            Loaded profile configuration
        """
        with open(filepath) as f:
            profile = json.load(f)

        # Validate loaded profile
        validation_result = self.validate_profile(profile)
        if not validation_result["valid"]:
            console.print(
                f"[yellow]Warning: Loaded profile has validation issues: {validation_result['errors']}[/yellow]"
            )

        return profile

    def compare_profiles(
        self, profile1: dict[str, Any], profile2: dict[str, Any]
    ) -> None:
        """
        Compare two profiles and show differences.

        Args:
            profile1: First profile to compare
            profile2: Second profile to compare
        """
        console.print(
            Panel.fit(
                f"[bold yellow]Profile Comparison[/bold yellow]\n\n"
                f"Comparing: {profile1.get('name', 'Profile 1')} vs {profile2.get('name', 'Profile 2')}",
                title="⚖️ Profile Comparison",
            )
        )

        # Create comparison table
        table = Table()
        table.add_column("Parameter", style="cyan")
        table.add_column(profile1.get("name", "Profile 1"), style="magenta")
        table.add_column(profile2.get("name", "Profile 2"), style="green")
        table.add_column("Difference", style="yellow")

        # Compare key parameters
        comparisons = [
            ("Base Spread (bps)", "strategy.base_spread_bps"),
            ("Order Levels", "strategy.order_levels"),
            ("Max Position (%)", "strategy.max_position_pct"),
            ("Cycle Interval (s)", "cycle_interval"),
            ("Max Position Value", "risk.max_position_value"),
            ("Daily Loss Limit (%)", "risk.daily_loss_limit_pct"),
        ]

        for param_name, param_path in comparisons:
            val1 = self._get_nested_value(profile1, param_path)
            val2 = self._get_nested_value(profile2, param_path)

            if val1 is not None and val2 is not None:
                try:
                    diff = float(val2) - float(val1)
                    diff_str = f"{diff:+.2f}" if diff != 0 else "Same"
                except (ValueError, TypeError):
                    diff_str = "Different" if val1 != val2 else "Same"
            else:
                diff_str = "N/A"

            table.add_row(param_name, str(val1), str(val2), diff_str)

        console.print(table)

    def _get_nested_value(self, config: dict[str, Any], path: str) -> Any:
        """Get a nested value from config using dot notation."""
        parts = path.split(".")
        value = config

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None

        return value


class ProfileOptimizer:
    """
    Profile optimization utility for performance tuning.

    Provides tools for optimizing profile parameters based on:
    - Historical performance data
    - Market conditions
    - Risk preferences
    - Performance targets
    """

    def __init__(self):
        """Initialize the profile optimizer."""
        self.optimization_metrics = [
            "profit_factor",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "fill_rate",
            "spread_capture_rate",
        ]

    def optimize_for_metric(
        self,
        base_profile: dict[str, Any],
        target_metric: str,
        historical_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Optimize a profile for a specific performance metric.

        Args:
            base_profile: Base profile to optimize
            target_metric: Metric to optimize for
            historical_data: Optional historical performance data

        Returns:
            Optimized profile configuration
        """
        if target_metric not in self.optimization_metrics:
            raise ValueError(f"Unknown metric: {target_metric}")

        optimized = base_profile.copy()

        # Optimization strategies by metric
        if target_metric == "profit_factor":
            # Tighten spreads for more captures
            optimized["strategy"]["base_spread_bps"] = max(
                int(optimized["strategy"]["base_spread_bps"] * 0.8),
                optimized["strategy"]["min_spread_bps"],
            )
            # Increase position size slightly
            optimized["strategy"]["max_position_pct"] = min(
                optimized["strategy"]["max_position_pct"] * 1.2, 50
            )

        elif target_metric == "sharpe_ratio":
            # Reduce position size for better risk-adjusted returns
            optimized["strategy"]["max_position_pct"] *= 0.8
            # Tighter stop losses
            optimized["risk"]["stop_loss_pct"] = min(
                optimized["risk"].get("stop_loss_pct", 2.0) * 0.8, 1.0
            )

        elif target_metric == "max_drawdown":
            # More conservative approach
            optimized["strategy"]["max_position_pct"] *= 0.6
            optimized["risk"]["daily_loss_limit_pct"] = min(
                optimized["risk"].get("daily_loss_limit_pct", 5.0) * 0.7, 3.0
            )
            # Wider spreads for safety
            optimized["strategy"]["base_spread_bps"] = int(
                optimized["strategy"]["base_spread_bps"] * 1.3
            )

        elif target_metric == "fill_rate":
            # More aggressive pricing
            optimized["strategy"]["base_spread_bps"] = max(
                int(optimized["strategy"]["base_spread_bps"] * 0.7),
                optimized["strategy"]["min_spread_bps"],
            )
            # More order levels
            optimized["strategy"]["order_levels"] = min(
                optimized["strategy"]["order_levels"] + 1, 8
            )

        return optimized

    def generate_optimization_report(
        self, original: dict[str, Any], optimized: dict[str, Any], target_metric: str
    ) -> None:
        """Generate an optimization report."""
        console.print(
            Panel.fit(
                f"[bold green]Profile Optimization Report[/bold green]\n\n"
                f"Target Metric: {target_metric}\n"
                f"Profile: {original.get('name', 'Unnamed')}",
                title="📊 Optimization Results",
            )
        )

        # Show key changes
        changes_table = Table(title="Key Changes")
        changes_table.add_column("Parameter", style="cyan")
        changes_table.add_column("Original", style="magenta")
        changes_table.add_column("Optimized", style="green")
        changes_table.add_column("Change", style="yellow")

        key_params = [
            ("Base Spread (bps)", "strategy.base_spread_bps"),
            ("Max Position (%)", "strategy.max_position_pct"),
            ("Order Levels", "strategy.order_levels"),
            ("Daily Loss Limit (%)", "risk.daily_loss_limit_pct"),
        ]

        for param_name, param_path in key_params:
            orig_val = self._get_nested_value(original, param_path)
            opt_val = self._get_nested_value(optimized, param_path)

            if orig_val is not None and opt_val is not None:
                try:
                    change_pct = (
                        (float(opt_val) - float(orig_val)) / float(orig_val)
                    ) * 100
                    change_str = f"{change_pct:+.1f}%"
                except (ValueError, TypeError, ZeroDivisionError):
                    change_str = "Changed"
            else:
                change_str = "N/A"

            changes_table.add_row(param_name, str(orig_val), str(opt_val), change_str)

        console.print(changes_table)

        # Show expected impact
        console.print(f"\n[bold blue]Expected Impact for {target_metric}:[/bold blue]")

        impact_descriptions = {
            "profit_factor": "• Tighter spreads should increase capture rate\n• Higher position size may increase profits\n• Risk: Lower margins per trade",
            "sharpe_ratio": "• Reduced position size improves risk-adjusted returns\n• Tighter stops reduce large losses\n• Risk: Lower absolute profits",
            "max_drawdown": "• Conservative sizing limits maximum losses\n• Wider spreads provide safety margins\n• Risk: Reduced profit opportunities",
            "fill_rate": "• Aggressive pricing increases fill probability\n• More levels provide depth\n• Risk: Thinner profit margins",
        }

        console.print(
            impact_descriptions.get(
                target_metric, "No specific impact description available"
            )
        )

    def _get_nested_value(self, config: dict[str, Any], path: str) -> Any:
        """Get a nested value from config using dot notation."""
        parts = path.split(".")
        value = config

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None

        return value


def main():
    """Main function to demonstrate custom profile creation."""
    try:
        creator = CustomProfileCreator()
        optimizer = ProfileOptimizer()

        # Generate example profiles
        creator.generate_profile_examples()

        # Demonstrate profile optimization
        console.print("\n" + "=" * 60)
        console.print("[bold magenta]Profile Optimization Examples[/bold magenta]")
        console.print("=" * 60)

        # Create a base profile
        base_profile = creator.create_custom_profile(
            name="test_profile",
            base_template="scalping",
            customizations={
                "strategy": {"base_spread_bps": 10, "max_position_pct": 25},
                "risk": {"daily_loss_limit_pct": 3.0},
            },
        )

        # Optimize for different metrics
        for metric in ["profit_factor", "sharpe_ratio", "max_drawdown"]:
            console.print(f"\n[cyan]Optimizing for {metric}...[/cyan]")
            optimized = optimizer.optimize_for_metric(base_profile, metric)
            optimizer.generate_optimization_report(base_profile, optimized, metric)

        # Save example profile
        saved_path = creator.save_profile(base_profile)
        console.print(f"\n✅ Example profile saved to: {saved_path}")

    except Exception as e:
        console.print(f"[red]Error in profile creation demo: {e}[/red]")
        logger.exception("Error in main")


if __name__ == "__main__":
    main()
