#!/usr/bin/env python3
"""
Comprehensive Configuration Validation Script

This script performs deep analysis of all configuration files, validates
consistency, checks for security issues, and provides recommendations.
"""

import json
import logging
import re

# Add bot to path for imports
import sys
from pathlib import Path
from typing import Any

import click
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ConfigurationValidator:
    """Comprehensive configuration validation and analysis."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.config_dir = base_path / "config"
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.recommendations: list[str] = []

    def validate_all(self) -> dict[str, Any]:
        """Run comprehensive validation of all configurations."""
        results = {
            "security_analysis": self._analyze_security(),
            "schema_validation": self._validate_schemas(),
            "consistency_check": self._check_consistency(),
            "field_analysis": self._analyze_fields(),
            "environment_validation": self._validate_environment(),
            "risk_analysis": self._analyze_risk_parameters(),
            "summary": self._generate_summary(),
        }

        return results

    def _analyze_security(self) -> dict[str, Any]:
        """Analyze security configuration and potential exposures."""
        logger.info("🔒 Analyzing security configuration...")

        security_issues = []
        env_files = []

        # Check for .env files
        for env_file in self.base_path.glob(".env*"):
            if env_file.name != ".env.example":
                env_files.append(str(env_file))

                # Check if contains real API keys
                if env_file.exists():
                    try:
                        content = env_file.read_text()
                        # Look for patterns that suggest real API keys
                        api_key_patterns = [
                            r"API_KEY=(?!your_|sk-proj-|your-).{20,}",
                            r"PRIVATE_KEY=(?!your_|0x123).{20,}",
                            r"SECRET=(?!your_|secret).{10,}",
                        ]

                        for pattern in api_key_patterns:
                            if re.search(pattern, content):
                                security_issues.append(
                                    f"Potential real API key found in {env_file.name}"
                                )
                    except Exception as e:
                        logger.warning(f"Could not read {env_file}: {e}")

        # Check gitignore effectiveness
        gitignore_path = self.base_path / ".gitignore"
        gitignore_effective = False
        if gitignore_path.exists():
            gitignore_content = gitignore_path.read_text()
            if ".env" in gitignore_content and ".env.*" in gitignore_content:
                gitignore_effective = True

        return {
            "env_files_found": env_files,
            "security_issues": security_issues,
            "gitignore_effective": gitignore_effective,
            "status": "warning" if security_issues else "pass",
        }

    def _validate_schemas(self) -> dict[str, Any]:
        """Validate all JSON config files against Pydantic schemas."""
        logger.info("📋 Validating configuration schemas...")

        validation_results = {}
        json_files = list(self.config_dir.glob("*.json"))
        json_files.extend(self.config_dir.glob("profiles/*.json"))

        for config_file in json_files:
            try:
                with open(config_file) as f:
                    config_data = json.load(f)

                # Try to validate with main Settings schema
                try:
                    Settings(**config_data)
                    validation_results[str(config_file.relative_to(self.base_path))] = {
                        "status": "valid",
                        "errors": [],
                    }
                except ValidationError as e:
                    validation_results[str(config_file.relative_to(self.base_path))] = {
                        "status": "invalid",
                        "errors": [str(error) for error in e.errors()],
                    }

            except json.JSONDecodeError as e:
                validation_results[str(config_file.relative_to(self.base_path))] = {
                    "status": "json_error",
                    "errors": [f"JSON decode error: {e}"],
                }
            except Exception as e:
                validation_results[str(config_file.relative_to(self.base_path))] = {
                    "status": "error",
                    "errors": [f"Validation error: {e}"],
                }

        return validation_results

    def _check_consistency(self) -> dict[str, Any]:
        """Check consistency across configuration files."""
        logger.info("🔍 Checking configuration consistency...")

        configs = {}
        inconsistencies = []

        # Load all configs
        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file) as f:
                    configs[config_file.name] = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load {config_file}: {e}")

        # Check field naming consistency
        field_variations = self._find_field_variations(configs)
        for field_purpose, variations in field_variations.items():
            if len(variations) > 1:
                inconsistencies.append(
                    {
                        "type": "field_naming",
                        "purpose": field_purpose,
                        "variations": list(variations),
                        "recommendation": f"Standardize to: {self._recommend_field_name(variations)}",
                    }
                )

        # Check value ranges
        value_ranges = self._analyze_value_ranges(configs)
        for field, range_info in value_ranges.items():
            if range_info["range"] > range_info["max"] * 0.5:  # High variation
                inconsistencies.append(
                    {
                        "type": "value_range",
                        "field": field,
                        "min": range_info["min"],
                        "max": range_info["max"],
                        "range": range_info["range"],
                        "recommendation": f"Consider standardizing range for {field}",
                    }
                )

        return {
            "inconsistencies": inconsistencies,
            "status": "warning" if inconsistencies else "pass",
        }

    def _analyze_fields(self) -> dict[str, Any]:
        """Analyze field coverage and missing requirements."""
        logger.info("📊 Analyzing field coverage...")

        required_sections = [
            "trading",
            "llm",
            "exchange",
            "risk",
            "data",
            "system",
            "paper_trading",
            "mcp",
            "omnisearch",
            "monitoring",
        ]

        configs = {}
        coverage_analysis = {}

        # Load all configs
        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file) as f:
                    config_data = json.load(f)
                    configs[config_file.name] = config_data

                    # Analyze section coverage
                    present_sections = set(config_data.keys())
                    missing_sections = set(required_sections) - present_sections
                    coverage_analysis[config_file.name] = {
                        "present": list(present_sections),
                        "missing": list(missing_sections),
                        "coverage_pct": len(present_sections)
                        / len(required_sections)
                        * 100,
                    }
            except Exception as e:
                logger.warning(f"Could not analyze {config_file}: {e}")

        return {
            "coverage_analysis": coverage_analysis,
            "required_sections": required_sections,
            "status": "pass",
        }

    def _validate_environment(self) -> dict[str, Any]:
        """Validate environment variable configuration."""
        logger.info("🌍 Validating environment configuration...")

        env_example_path = self.base_path / ".env.example"
        env_vars_documented = set()
        env_vars_used = set()

        # Parse .env.example
        if env_example_path.exists():
            content = env_example_path.read_text()
            for line in content.split("\n"):
                if "=" in line and not line.strip().startswith("#"):
                    var_name = line.split("=")[0].strip()
                    env_vars_documented.add(var_name)

        # Find environment variables used in Pydantic models
        # This is a simplified check - in reality, we'd need to parse the models
        config_py_path = self.base_path / "bot" / "config.py"
        if config_py_path.exists():
            content = config_py_path.read_text()
            # Look for environment variable patterns
            env_patterns = re.findall(r"[A-Z_]+__[A-Z_]+", content)
            env_vars_used.update(env_patterns)

        undocumented = env_vars_used - env_vars_documented
        unused = env_vars_documented - env_vars_used

        return {
            "documented_vars": list(env_vars_documented),
            "used_vars": list(env_vars_used),
            "undocumented": list(undocumented),
            "unused": list(unused),
            "status": "warning" if undocumented or unused else "pass",
        }

    def _analyze_risk_parameters(self) -> dict[str, Any]:
        """Analyze risk parameter consistency and safety."""
        logger.info("⚠️ Analyzing risk parameters...")

        risk_fields = [
            "max_daily_loss_pct",
            "max_weekly_loss_pct",
            "max_monthly_loss_pct",
            "leverage",
            "max_position_hold_hours",
            "default_stop_loss_pct",
            "default_take_profit_pct",
            "emergency_stop_loss_pct",
        ]

        configs = {}
        risk_analysis = {}

        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file) as f:
                    config_data = json.load(f)
                    configs[config_file.name] = config_data

                    # Extract risk parameters
                    risk_params = {}
                    for section in ["risk", "trading"]:
                        if section in config_data:
                            for field in risk_fields:
                                if field in config_data[section]:
                                    risk_params[field] = config_data[section][field]

                    risk_analysis[config_file.name] = risk_params
            except Exception as e:
                logger.warning(f"Could not analyze risk params in {config_file}: {e}")

        # Analyze parameter ranges and safety
        safety_issues = []
        for config_name, params in risk_analysis.items():
            # Check for dangerous combinations
            leverage = params.get("leverage", 1)
            daily_loss = params.get("max_daily_loss_pct", 0)

            if leverage > 10 and daily_loss > 5:
                safety_issues.append(
                    {
                        "config": config_name,
                        "issue": f"High leverage ({leverage}x) with high daily loss limit ({daily_loss}%)",
                        "severity": "high",
                    }
                )

            stop_loss = params.get("default_stop_loss_pct", 0)
            take_profit = params.get("default_take_profit_pct", 0)

            if stop_loss >= take_profit and stop_loss > 0 and take_profit > 0:
                safety_issues.append(
                    {
                        "config": config_name,
                        "issue": f"Stop loss ({stop_loss}%) >= take profit ({take_profit}%)",
                        "severity": "medium",
                    }
                )

        return {
            "risk_analysis": risk_analysis,
            "safety_issues": safety_issues,
            "status": (
                "error"
                if any(issue["severity"] == "high" for issue in safety_issues)
                else "warning" if safety_issues else "pass"
            ),
        }

    def _find_field_variations(self, configs: dict[str, Any]) -> dict[str, set[str]]:
        """Find field naming variations across configs."""
        field_variations = {}

        # Define semantic field groups
        semantic_groups = {
            "log_file": ["log_file", "log_file_path", "completion_log_file"],
            "timeout": ["timeout", "request_timeout", "api_timeout"],
            "max_tokens": ["max_tokens", "max_response_tokens"],
            "retry_delay": ["retry_delay", "retry_delay_seconds"],
        }

        for group_name, field_names in semantic_groups.items():
            found_variations = set()
            for config_name, config_data in configs.items():
                for field_name in field_names:
                    if self._field_exists_in_config(config_data, field_name):
                        found_variations.add(field_name)

            if found_variations:
                field_variations[group_name] = found_variations

        return field_variations

    def _field_exists_in_config(self, config: dict[str, Any], field_name: str) -> bool:
        """Check if a field exists anywhere in the config."""

        def search_dict(d, target):
            if isinstance(d, dict):
                if target in d:
                    return True
                for v in d.values():
                    if search_dict(v, target):
                        return True
            return False

        return search_dict(config, field_name)

    def _recommend_field_name(self, variations: set[str]) -> str:
        """Recommend the best field name from variations."""
        # Prefer more descriptive names
        priority_order = ["_path", "_file", "_seconds", "_timeout"]

        for priority in priority_order:
            for variation in variations:
                if priority in variation:
                    return variation

        # Return the longest name as default
        return max(variations, key=len)

    def _analyze_value_ranges(
        self, configs: dict[str, Any]
    ) -> dict[str, dict[str, float]]:
        """Analyze value ranges for numeric fields."""
        numeric_fields = {}

        for config_name, config_data in configs.items():
            self._extract_numeric_fields(config_data, numeric_fields, prefix="")

        # Calculate ranges
        ranges = {}
        for field, values in numeric_fields.items():
            if len(values) > 1:
                min_val, max_val = min(values), max(values)
                ranges[field] = {
                    "min": min_val,
                    "max": max_val,
                    "range": max_val - min_val,
                    "values": values,
                }

        return ranges

    def _extract_numeric_fields(
        self, config: Any, numeric_fields: dict[str, list[float]], prefix: str
    ):
        """Recursively extract numeric fields from config."""
        if isinstance(config, dict):
            for key, value in config.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                self._extract_numeric_fields(value, numeric_fields, new_prefix)
        elif isinstance(config, (int, float)) and not isinstance(config, bool):
            if prefix not in numeric_fields:
                numeric_fields[prefix] = []
            numeric_fields[prefix].append(float(config))

    def _generate_summary(self) -> dict[str, Any]:
        """Generate validation summary."""
        return {
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "total_recommendations": len(self.recommendations),
            "is_valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }

    def generate_report(self, results: dict[str, Any]) -> str:
        """Generate comprehensive validation report."""
        report_lines = [
            "# COMPREHENSIVE CONFIGURATION VALIDATION REPORT",
            f"Generated: {Path.cwd()}",
            "",
            "## EXECUTIVE SUMMARY",
        ]

        # Add sections based on results
        for section, data in results.items():
            if section == "summary":
                continue

            report_lines.extend(
                [
                    "",
                    f"## {section.replace('_', ' ').upper()}",
                    f"Status: {data.get('status', 'unknown').upper()}",
                    "",
                ]
            )

            # Add section-specific details
            if section == "security_analysis":
                if data.get("security_issues"):
                    report_lines.append("### Security Issues Found:")
                    for issue in data["security_issues"]:
                        report_lines.append(f"- ⚠️ {issue}")

            elif section == "consistency_check":
                if data.get("inconsistencies"):
                    report_lines.append("### Inconsistencies Found:")
                    for issue in data["inconsistencies"]:
                        report_lines.append(
                            f"- {issue['type']}: {issue.get('purpose', issue.get('field', 'unknown'))}"
                        )
                        report_lines.append(
                            f"  Recommendation: {issue['recommendation']}"
                        )

        return "\n".join(report_lines)


@click.command()
@click.option("--config-path", default=".", help="Path to configuration directory")
@click.option("--output", "-o", help="Output file for validation report")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def main(config_path: str, output: str, verbose: bool):
    """Run comprehensive configuration validation."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    base_path = Path(config_path).resolve()
    validator = ConfigurationValidator(base_path)

    click.echo("🔍 Running comprehensive configuration validation...")
    results = validator.validate_all()

    # Generate report
    report = validator.generate_report(results)

    if output:
        with open(output, "w") as f:
            f.write(report)
        click.echo(f"📊 Report saved to: {output}")
    else:
        click.echo(report)

    # Exit with error code if validation failed
    if not results["summary"]["is_valid"]:
        click.echo("❌ Configuration validation failed!")
        exit(1)
    else:
        click.echo("✅ Configuration validation passed!")


if __name__ == "__main__":
    main()
