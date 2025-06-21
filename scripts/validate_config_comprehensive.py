#!/usr/bin/env python3
"""
Comprehensive Configuration Validation Report for AI Trading Bot

This script validates all configuration files, settings, and environment variables
without requiring a valid .env file, providing a complete validation report.
"""

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Suppress warnings during validation
logging.disable(logging.WARNING)


@dataclass
class ValidationResult:
    category: str
    item: str
    status: str  # "pass", "fail", "warning", "skip"
    message: str
    details: str = ""


class ComprehensiveConfigValidator:
    """Comprehensive configuration validator that doesn't require valid .env"""

    def __init__(self):
        self.project_root = Path.cwd()
        self.config_dir = self.project_root / "config"
        self.results: list[ValidationResult] = []

    def validate_all(self) -> dict[str, Any]:
        """Run all validation checks"""
        print("🔍 Running Comprehensive Configuration Validation")
        print("=" * 60)

        # Validate JSON configuration files
        self._validate_json_configs()

        # Validate .env.example completeness
        self._validate_env_example()

        # Validate Docker configuration
        self._validate_docker_config()

        # Validate schema consistency
        self._validate_schema_consistency()

        # Validate configuration structure
        self._validate_config_structure()

        # Generate report
        return self._generate_report()

    def _validate_json_configs(self):
        """Validate all JSON configuration files"""
        print("\n📋 Validating JSON Configuration Files")
        print("-" * 40)

        json_files = list(self.config_dir.glob("**/*.json"))

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    config_data = json.load(f)

                # Basic structure validation
                self._validate_json_structure(json_file, config_data)

                self.results.append(
                    ValidationResult(
                        category="json_validation",
                        item=str(json_file.relative_to(self.config_dir)),
                        status="pass",
                        message="Valid JSON syntax and structure",
                    )
                )
                print(f"✅ {json_file.relative_to(self.config_dir)}")

            except json.JSONDecodeError as e:
                self.results.append(
                    ValidationResult(
                        category="json_validation",
                        item=str(json_file.relative_to(self.config_dir)),
                        status="fail",
                        message=f"JSON syntax error: {e}",
                        details=f"Line {e.lineno}, Column {e.colno}: {e.msg}",
                    )
                )
                print(f"❌ {json_file.relative_to(self.config_dir)}: {e}")

            except Exception as e:
                self.results.append(
                    ValidationResult(
                        category="json_validation",
                        item=str(json_file.relative_to(self.config_dir)),
                        status="fail",
                        message=f"File error: {e}",
                    )
                )
                print(f"❌ {json_file.relative_to(self.config_dir)}: {e}")

    def _validate_json_structure(self, json_file: Path, config_data: dict):
        """Validate the structure of individual JSON configs"""
        filename = json_file.name

        # Common required sections
        common_sections = {
            "trading": ["symbol", "interval", "leverage"],
            "llm": ["provider", "model_name"],
            "system": ["dry_run", "environment"],
            "risk": ["max_daily_loss_pct", "default_stop_loss_pct"],
        }

        # Skip schema.json and monitoring configs
        if filename in ["schema.json", "monitoring_config.json"]:
            return

        # Check for required sections
        for section, required_fields in common_sections.items():
            if section in config_data:
                section_data = config_data[section]
                for field in required_fields:
                    if field not in section_data:
                        self.results.append(
                            ValidationResult(
                                category="json_structure",
                                item=f"{filename}:{section}.{field}",
                                status="warning",
                                message=f"Missing recommended field: {section}.{field}",
                            )
                        )

    def _validate_env_example(self):
        """Validate .env.example completeness"""
        print("\n🔧 Validating .env.example Completeness")
        print("-" * 40)

        env_example_path = self.project_root / ".env.example"

        if not env_example_path.exists():
            self.results.append(
                ValidationResult(
                    category="env_validation",
                    item=".env.example",
                    status="fail",
                    message="Missing .env.example file",
                )
            )
            print("❌ .env.example file not found")
            return

        try:
            with open(env_example_path) as f:
                env_content = f.read()

            # Required environment variables
            required_vars = [
                "EXCHANGE__EXCHANGE_TYPE",
                "EXCHANGE__BLUEFIN_PRIVATE_KEY",
                "EXCHANGE__BLUEFIN_NETWORK",
                "LLM__OPENAI_API_KEY",
                "TRADING__SYMBOL",
                "TRADING__LEVERAGE",
                "SYSTEM__DRY_RUN",
                "SYSTEM__ENVIRONMENT",
                "RISK__MAX_DAILY_LOSS_PCT",
                "RISK__DEFAULT_STOP_LOSS_PCT",
                "BLUEFIN_SERVICE_API_KEY",
            ]

            missing_vars = []
            for var in required_vars:
                if var not in env_content:
                    missing_vars.append(var)

            if missing_vars:
                self.results.append(
                    ValidationResult(
                        category="env_validation",
                        item=".env.example completeness",
                        status="warning",
                        message=f"Missing {len(missing_vars)} environment variables",
                        details=f"Missing: {', '.join(missing_vars)}",
                    )
                )
                print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
            else:
                self.results.append(
                    ValidationResult(
                        category="env_validation",
                        item=".env.example completeness",
                        status="pass",
                        message="All required environment variables documented",
                    )
                )
                print("✅ All required environment variables documented")

            # Check for security best practices
            self._validate_env_security(env_content)

        except Exception as e:
            self.results.append(
                ValidationResult(
                    category="env_validation",
                    item=".env.example",
                    status="fail",
                    message=f"Error reading .env.example: {e}",
                )
            )
            print(f"❌ Error reading .env.example: {e}")

    def _validate_env_security(self, env_content: str):
        """Validate environment security practices"""
        security_checks = [
            {
                "pattern": r"(?i)password\s*=\s*[^#\n]+",
                "message": "Hardcoded password found",
                "severity": "fail",
            },
            {
                "pattern": r"(?i)secret\s*=\s*[^#\n]+",
                "message": "Hardcoded secret found",
                "severity": "warning",
            },
            {
                "pattern": r"sk-[a-zA-Z0-9]{48}",
                "message": "Actual OpenAI API key found",
                "severity": "fail",
            },
            {
                "pattern": r"0x[a-fA-F0-9]{64}",
                "message": "Actual private key found",
                "severity": "fail",
            },
        ]

        for check in security_checks:
            if re.search(check["pattern"], env_content):
                self.results.append(
                    ValidationResult(
                        category="env_security",
                        item=".env.example security",
                        status=check["severity"],
                        message=check["message"],
                    )
                )
                if check["severity"] == "fail":
                    print(f"❌ Security issue: {check['message']}")
                else:
                    print(f"⚠️  Security warning: {check['message']}")

    def _validate_docker_config(self):
        """Validate Docker configuration"""
        print("\n🐳 Validating Docker Configuration")
        print("-" * 40)

        docker_compose_path = self.project_root / "docker-compose.yml"

        if not docker_compose_path.exists():
            self.results.append(
                ValidationResult(
                    category="docker_validation",
                    item="docker-compose.yml",
                    status="fail",
                    message="Missing docker-compose.yml file",
                )
            )
            print("❌ docker-compose.yml not found")
            return

        try:
            with open(docker_compose_path) as f:
                docker_content = f.read()

            # Check for required services
            required_services = ["ai-trading-bot", "bluefin-service"]

            for service in required_services:
                if service in docker_content:
                    self.results.append(
                        ValidationResult(
                            category="docker_validation",
                            item=f"service_{service}",
                            status="pass",
                            message=f"Service {service} configured",
                        )
                    )
                    print(f"✅ Service {service} found")
                else:
                    self.results.append(
                        ValidationResult(
                            category="docker_validation",
                            item=f"service_{service}",
                            status="warning",
                            message=f"Service {service} not found",
                        )
                    )
                    print(f"⚠️  Service {service} not found")

            # Check for environment variable mapping
            if "${HOST_UID" in docker_content and "${HOST_GID" in docker_content:
                self.results.append(
                    ValidationResult(
                        category="docker_validation",
                        item="user_permissions",
                        status="pass",
                        message="User permission mapping configured",
                    )
                )
                print("✅ User permission mapping configured")
            else:
                self.results.append(
                    ValidationResult(
                        category="docker_validation",
                        item="user_permissions",
                        status="warning",
                        message="User permission mapping not configured",
                    )
                )
                print("⚠️  User permission mapping not configured")

        except Exception as e:
            self.results.append(
                ValidationResult(
                    category="docker_validation",
                    item="docker-compose.yml",
                    status="fail",
                    message=f"Error reading Docker config: {e}",
                )
            )
            print(f"❌ Error reading Docker config: {e}")

    def _validate_schema_consistency(self):
        """Validate schema consistency with JSON configs"""
        print("\n📐 Validating Schema Consistency")
        print("-" * 40)

        schema_path = self.config_dir / "schema.json"

        if not schema_path.exists():
            self.results.append(
                ValidationResult(
                    category="schema_validation",
                    item="schema.json",
                    status="warning",
                    message="Schema file not found",
                )
            )
            print("⚠️  schema.json not found")
            return

        try:
            with open(schema_path) as f:
                schema = json.load(f)

            # Validate schema structure
            if "$schema" in schema and "properties" in schema:
                self.results.append(
                    ValidationResult(
                        category="schema_validation",
                        item="schema_structure",
                        status="pass",
                        message="Schema has valid JSON Schema structure",
                    )
                )
                print("✅ Schema structure valid")
            else:
                self.results.append(
                    ValidationResult(
                        category="schema_validation",
                        item="schema_structure",
                        status="fail",
                        message="Invalid JSON Schema structure",
                    )
                )
                print("❌ Invalid JSON Schema structure")

            # Check required fields match common configs
            schema_required = set()
            if "required" in schema:
                schema_required.update(schema["required"])

            # Sample a few config files to check consistency
            sample_configs = [
                "development.json",
                "production.json",
                "paper_trading.json",
            ]

            for config_name in sample_configs:
                config_path = self.config_dir / config_name
                if config_path.exists():
                    with open(config_path) as f:
                        config_data = json.load(f)

                    config_sections = set(config_data.keys())
                    missing_in_config = schema_required - config_sections

                    if missing_in_config:
                        self.results.append(
                            ValidationResult(
                                category="schema_validation",
                                item=f"{config_name}_consistency",
                                status="warning",
                                message=f"Config missing schema-required sections: {', '.join(missing_in_config)}",
                            )
                        )
                        print(
                            f"⚠️  {config_name} missing sections: {', '.join(missing_in_config)}"
                        )
                    else:
                        self.results.append(
                            ValidationResult(
                                category="schema_validation",
                                item=f"{config_name}_consistency",
                                status="pass",
                                message="Config matches schema requirements",
                            )
                        )
                        print(f"✅ {config_name} matches schema")

        except Exception as e:
            self.results.append(
                ValidationResult(
                    category="schema_validation",
                    item="schema.json",
                    status="fail",
                    message=f"Error validating schema: {e}",
                )
            )
            print(f"❌ Error validating schema: {e}")

    def _validate_config_structure(self):
        """Validate overall configuration structure"""
        print("\n🏗️  Validating Configuration Structure")
        print("-" * 40)

        # Check for required directories
        required_dirs = ["config", "config/profiles", "logs", "data"]

        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                self.results.append(
                    ValidationResult(
                        category="structure_validation",
                        item=f"directory_{dir_path}",
                        status="pass",
                        message=f"Directory {dir_path} exists",
                    )
                )
                print(f"✅ Directory {dir_path} exists")
            else:
                self.results.append(
                    ValidationResult(
                        category="structure_validation",
                        item=f"directory_{dir_path}",
                        status="warning",
                        message=f"Directory {dir_path} missing",
                    )
                )
                print(f"⚠️  Directory {dir_path} missing")

        # Check configuration profiles
        profiles_dir = self.config_dir / "profiles"
        if profiles_dir.exists():
            profile_files = list(profiles_dir.glob("*.json"))
            if profile_files:
                self.results.append(
                    ValidationResult(
                        category="structure_validation",
                        item="configuration_profiles",
                        status="pass",
                        message=f"Found {len(profile_files)} configuration profiles",
                    )
                )
                print(f"✅ Found {len(profile_files)} configuration profiles")
            else:
                self.results.append(
                    ValidationResult(
                        category="structure_validation",
                        item="configuration_profiles",
                        status="warning",
                        message="No configuration profiles found",
                    )
                )
                print("⚠️  No configuration profiles found")

    def _generate_report(self) -> dict[str, Any]:
        """Generate comprehensive validation report"""
        print("\n📊 Generating Validation Report")
        print("-" * 40)

        # Categorize results
        categories = {}
        total_pass = 0
        total_fail = 0
        total_warning = 0

        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)

            if result.status == "pass":
                total_pass += 1
            elif result.status == "fail":
                total_fail += 1
            elif result.status == "warning":
                total_warning += 1

        # Generate summary
        overall_status = (
            "pass" if total_fail == 0 else "fail" if total_fail > 0 else "warning"
        )

        report = {
            "summary": {
                "overall_status": overall_status,
                "total_checks": len(self.results),
                "passed": total_pass,
                "failed": total_fail,
                "warnings": total_warning,
                "success_rate": (
                    f"{(total_pass / len(self.results) * 100):.1f}%"
                    if self.results
                    else "0%"
                ),
            },
            "categories": {},
        }

        # Add category details
        for category, results in categories.items():
            category_pass = sum(1 for r in results if r.status == "pass")
            category_fail = sum(1 for r in results if r.status == "fail")
            category_warning = sum(1 for r in results if r.status == "warning")

            report["categories"][category] = {
                "status": "pass" if category_fail == 0 else "fail",
                "total": len(results),
                "passed": category_pass,
                "failed": category_fail,
                "warnings": category_warning,
                "results": [
                    {
                        "item": r.item,
                        "status": r.status,
                        "message": r.message,
                        "details": r.details,
                    }
                    for r in results
                ],
            }

        return report

    def print_report(self, report: dict[str, Any]):
        """Print formatted validation report"""
        print("\n" + "=" * 60)
        print("📋 CONFIGURATION VALIDATION REPORT")
        print("=" * 60)

        summary = report["summary"]

        # Overall status
        status_icon = {"pass": "✅", "fail": "❌", "warning": "⚠️"}[
            summary["overall_status"]
        ]
        print(f"\n{status_icon} Overall Status: {summary['overall_status'].upper()}")
        print(f"📊 Success Rate: {summary['success_rate']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⚠️  Warnings: {summary['warnings']}")
        print(f"📝 Total Checks: {summary['total_checks']}")

        # Category breakdown
        print("\n📋 Category Breakdown:")
        print("-" * 40)

        for category, data in report["categories"].items():
            category_icon = {"pass": "✅", "fail": "❌"}[data["status"]]
            category_name = category.replace("_", " ").title()
            print(
                f"{category_icon} {category_name}: {data['passed']}/{data['total']} passed"
            )

            # Show failures and warnings
            failures = [r for r in data["results"] if r["status"] == "fail"]
            warnings = [r for r in data["results"] if r["status"] == "warning"]

            for failure in failures:
                print(f"    ❌ {failure['item']}: {failure['message']}")
                if failure["details"]:
                    print(f"       Details: {failure['details']}")

            for warning in warnings:
                print(f"    ⚠️  {warning['item']}: {warning['message']}")
                if warning["details"]:
                    print(f"       Details: {warning['details']}")

        print("\n" + "=" * 60)

        # Recommendations
        if summary["failed"] > 0 or summary["warnings"] > 0:
            print("\n🔧 RECOMMENDATIONS:")
            print("-" * 40)

            if summary["failed"] > 0:
                print("❌ CRITICAL ISSUES (must fix):")
                for category, data in report["categories"].items():
                    failures = [r for r in data["results"] if r["status"] == "fail"]
                    for failure in failures:
                        print(f"   • Fix {failure['item']}: {failure['message']}")

            if summary["warnings"] > 0:
                print("\n⚠️  WARNINGS (recommended to fix):")
                for category, data in report["categories"].items():
                    warnings = [r for r in data["results"] if r["status"] == "warning"]
                    for warning in warnings:
                        print(f"   • Check {warning['item']}: {warning['message']}")

            print("\n💡 NEXT STEPS:")
            print("   1. Fix all critical issues marked with ❌")
            print("   2. Address warnings marked with ⚠️")
            print(
                "   3. Re-run validation: python3 scripts/validate_config_comprehensive.py"
            )
            print("   4. Test with: python3 -m bot.main --validate-only")
        else:
            print("\n🎉 EXCELLENT! All validations passed.")
            print("   Your configuration is ready for use.")

        print("\n" + "=" * 60)


def main():
    """Main entry point"""
    validator = ComprehensiveConfigValidator()
    report = validator.validate_all()
    validator.print_report(report)

    # Save report to file
    report_path = Path("logs/config_validation_report.json")
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n📄 Detailed report saved to: {report_path}")

    # Exit with error code if validation failed
    if report["summary"]["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
