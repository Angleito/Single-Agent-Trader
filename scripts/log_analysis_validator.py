#!/usr/bin/env python3
"""
Comprehensive Log Analysis and Validation Tool
Analyzes Docker container logs for test execution, performance metrics, and error detection.
"""

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LogEntry:
    """Structured representation of a log entry"""

    timestamp: datetime
    level: str
    logger: str
    module: str
    function: str
    line: int
    message: str
    raw_line: str
    container_name: str | None = None
    test_name: str | None = None
    duration: float | None = None


@dataclass
class TestResult:
    """Test execution result"""

    test_name: str
    status: str  # PASSED, FAILED, SKIPPED, ERROR
    duration: float
    error_message: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None


@dataclass
class PerformanceMetric:
    """Performance metric data point"""

    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    container_name: str | None = None
    context: str | None = None


@dataclass
class ValidationResult:
    """Result of log validation"""

    total_entries: int
    error_count: int
    warning_count: int
    test_results: list[TestResult]
    performance_metrics: list[PerformanceMetric]
    coverage_score: float
    quality_issues: list[str]
    recommendations: list[str]


class LogPatternMatcher:
    """Handles pattern matching for different log formats"""

    def __init__(self):
        # Common log patterns
        self.patterns = {
            "detailed": re.compile(
                r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - "
                r"(?P<logger>[\w\.]+) - (?P<level>\w+) - "
                r"\[(?P<filename>[\w\.]+):(?P<line>\d+)\] - "
                r"(?P<function>\w+)\(\) - (?P<message>.*)"
            ),
            "simple": re.compile(
                r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - "
                r"(?P<level>\w+) - (?P<message>.*)"
            ),
            "json": re.compile(r"^\{.*\}$"),
            "performance": re.compile(
                r"PERF\|(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\|"
                r"(?P<level>\w+)\|(?P<logger>[\w\.]+)\|(?P<message>.*)"
            ),
            "alert": re.compile(
                r"ALERT\|(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\|"
                r"(?P<level>\w+)\|(?P<logger>[\w\.]+)\|(?P<message>.*)"
            ),
            "test_result": re.compile(
                r"(?P<test_name>test_\w+)\s+\.{3}\s+(?P<status>PASSED|FAILED|SKIPPED|ERROR)"
                r"(?:\s+\[(?P<duration>\d+\.\d+)s\])?"
            ),
            "pytest": re.compile(
                r"(?P<filename>\w+\.py)::(?P<test_name>test_\w+)\s+"
                r"(?P<status>PASSED|FAILED|SKIPPED|ERROR)(?:\s+\[(?P<duration>\d+\.\d+)s\])?"
            ),
            "docker_container": re.compile(
                r"(?P<container_name>[\w-]+)\s*\|\s*(?P<message>.*)"
            ),
        }

        # Performance metric patterns
        self.metric_patterns = {
            "cpu_usage": re.compile(r"CPU.*?(?P<value>\d+(?:\.\d+)?)%"),
            "memory_usage": re.compile(r"Memory.*?(?P<value>\d+(?:\.\d+)?)%"),
            "disk_usage": re.compile(r"Disk.*?(?P<value>\d+(?:\.\d+)?)%"),
            "response_time": re.compile(
                r"response.*?time.*?(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|s)"
            ),
            "throughput": re.compile(
                r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>rps|tps|ops)"
            ),
            "duration": re.compile(
                r"duration.*?(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|s|m)"
            ),
            "bytes_transferred": re.compile(
                r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>B|KB|MB|GB)"
            ),
        }


class LogAnalyzer:
    """Main log analysis and validation class"""

    def __init__(self, logs_directory: Path = None):
        self.logs_dir = logs_directory or Path("/app/logs")
        self.pattern_matcher = LogPatternMatcher()
        self.validation_rules = self._load_validation_rules()

    def _load_validation_rules(self) -> dict[str, Any]:
        """Load validation rules from configuration"""
        rules_file = Path(__file__).parent / "log_validation_rules.yaml"
        if rules_file.exists():
            with open(rules_file) as f:
                return yaml.safe_load(f)
        return self._get_default_validation_rules()

    def _get_default_validation_rules(self) -> dict[str, Any]:
        """Default validation rules"""
        return {
            "required_loggers": ["bot", "trading", "exchange", "llm"],
            "error_keywords": ["error", "exception", "failed", "timeout", "crash"],
            "performance_thresholds": {
                "cpu_usage": 85.0,
                "memory_usage": 90.0,
                "response_time_ms": 5000.0,
            },
            "test_coverage_threshold": 80.0,
            "max_log_gaps_minutes": 5,
            "required_test_patterns": ["test_", "setup", "teardown"],
        }

    def parse_log_entry(self, line: str, container_name: str = None) -> LogEntry | None:
        """Parse a single log line into structured format"""
        line = line.strip()
        if not line:
            return None

        # Try JSON format first
        if line.startswith("{"):
            try:
                data = json.loads(line)
                return LogEntry(
                    timestamp=datetime.fromisoformat(data.get("timestamp", "")),
                    level=data.get("level", "INFO"),
                    logger=data.get("logger", "unknown"),
                    module=data.get("module", ""),
                    function=data.get("function", ""),
                    line=data.get("line", 0),
                    message=data.get("message", ""),
                    raw_line=line,
                    container_name=container_name,
                )
            except (json.JSONDecodeError, ValueError):
                pass

        # Try other patterns
        for pattern_name, pattern in self.pattern_matcher.patterns.items():
            if pattern_name == "json":
                continue

            match = pattern.match(line)
            if match:
                groups = match.groupdict()
                try:
                    timestamp = datetime.strptime(
                        groups.get("timestamp", ""), "%Y-%m-%d %H:%M:%S"
                    )
                    return LogEntry(
                        timestamp=timestamp,
                        level=groups.get("level", "INFO"),
                        logger=groups.get("logger", "unknown"),
                        module=groups.get("filename", ""),
                        function=groups.get("function", ""),
                        line=int(groups.get("line", 0)) if groups.get("line") else 0,
                        message=groups.get("message", ""),
                        raw_line=line,
                        container_name=container_name,
                    )
                except ValueError:
                    continue

        # Fallback: treat as simple message
        return LogEntry(
            timestamp=datetime.now(),
            level="INFO",
            logger="unknown",
            module="",
            function="",
            line=0,
            message=line,
            raw_line=line,
            container_name=container_name,
        )

    def extract_test_results(self, log_entries: list[LogEntry]) -> list[TestResult]:
        """Extract test results from log entries"""
        test_results = []
        test_starts = {}

        for entry in log_entries:
            message = entry.message

            # Check for pytest format
            pytest_match = self.pattern_matcher.patterns["pytest"].search(message)
            if pytest_match:
                groups = pytest_match.groupdict()
                test_name = f"{groups['filename']}::{groups['test_name']}"
                duration = float(groups.get("duration", 0.0))

                test_results.append(
                    TestResult(
                        test_name=test_name,
                        status=groups["status"],
                        duration=duration,
                        start_time=entry.timestamp - timedelta(seconds=duration),
                        end_time=entry.timestamp,
                    )
                )
                continue

            # Check for simple test result format
            test_match = self.pattern_matcher.patterns["test_result"].search(message)
            if test_match:
                groups = test_match.groupdict()
                duration = float(groups.get("duration", 0.0))

                test_results.append(
                    TestResult(
                        test_name=groups["test_name"],
                        status=groups["status"],
                        duration=duration,
                        start_time=entry.timestamp - timedelta(seconds=duration),
                        end_time=entry.timestamp,
                    )
                )
                continue

            # Track test starts and ends
            if "test started" in message.lower() or "running test" in message.lower():
                test_name_match = re.search(r"test_\w+", message)
                if test_name_match:
                    test_starts[test_name_match.group()] = entry.timestamp

            elif (
                "test completed" in message.lower()
                or "test finished" in message.lower()
            ):
                test_name_match = re.search(r"test_\w+", message)
                if test_name_match:
                    test_name = test_name_match.group()
                    start_time = test_starts.get(test_name)
                    if start_time:
                        duration = (entry.timestamp - start_time).total_seconds()
                        status = "PASSED" if "passed" in message.lower() else "FAILED"

                        test_results.append(
                            TestResult(
                                test_name=test_name,
                                status=status,
                                duration=duration,
                                start_time=start_time,
                                end_time=entry.timestamp,
                            )
                        )

        return test_results

    def extract_performance_metrics(
        self, log_entries: list[LogEntry]
    ) -> list[PerformanceMetric]:
        """Extract performance metrics from log entries"""
        metrics = []

        for entry in log_entries:
            message = entry.message

            # Check each metric pattern
            for metric_name, pattern in self.pattern_matcher.metric_patterns.items():
                match = pattern.search(message)
                if match:
                    groups = match.groupdict()
                    value = float(groups["value"])
                    unit = groups.get("unit", "")

                    metrics.append(
                        PerformanceMetric(
                            timestamp=entry.timestamp,
                            metric_name=metric_name,
                            value=value,
                            unit=unit,
                            container_name=entry.container_name,
                            context=message,
                        )
                    )

        return metrics

    def validate_log_quality(
        self, log_entries: list[LogEntry]
    ) -> tuple[float, list[str]]:
        """Validate log quality and coverage"""
        issues = []
        coverage_score = 100.0

        if not log_entries:
            return 0.0, ["No log entries found"]

        # Check for required loggers
        present_loggers = set(entry.logger for entry in log_entries)
        required_loggers = set(self.validation_rules["required_loggers"])
        missing_loggers = required_loggers - present_loggers

        if missing_loggers:
            issues.append(f"Missing required loggers: {', '.join(missing_loggers)}")
            coverage_score -= len(missing_loggers) * 10

        # Check for error patterns
        error_entries = [
            entry
            for entry in log_entries
            if entry.level in ["ERROR", "CRITICAL"]
            or any(
                keyword in entry.message.lower()
                for keyword in self.validation_rules["error_keywords"]
            )
        ]

        if error_entries:
            issues.append(f"Found {len(error_entries)} error entries")
            coverage_score -= min(len(error_entries), 20)

        # Check for log gaps
        if len(log_entries) > 1:
            time_gaps = []
            for i in range(1, len(log_entries)):
                gap = (
                    log_entries[i].timestamp - log_entries[i - 1].timestamp
                ).total_seconds() / 60
                if gap > self.validation_rules["max_log_gaps_minutes"]:
                    time_gaps.append(gap)

            if time_gaps:
                issues.append(
                    f"Found {len(time_gaps)} log gaps > {self.validation_rules['max_log_gaps_minutes']} minutes"
                )
                coverage_score -= len(time_gaps) * 5

        # Check log level distribution
        level_counts = Counter(entry.level for entry in log_entries)
        total_entries = len(log_entries)

        # Too many errors or warnings might indicate issues
        error_ratio = (
            level_counts.get("ERROR", 0) + level_counts.get("CRITICAL", 0)
        ) / total_entries
        if error_ratio > 0.1:  # More than 10% errors
            issues.append(f"High error ratio: {error_ratio:.1%}")
            coverage_score -= error_ratio * 30

        warning_ratio = level_counts.get("WARNING", 0) / total_entries
        if warning_ratio > 0.2:  # More than 20% warnings
            issues.append(f"High warning ratio: {warning_ratio:.1%}")
            coverage_score -= warning_ratio * 20

        return max(0.0, coverage_score), issues

    def analyze_log_file(
        self, file_path: Path, container_name: str = None
    ) -> ValidationResult:
        """Analyze a single log file"""
        log_entries = []

        if not file_path.exists():
            return ValidationResult(
                total_entries=0,
                error_count=0,
                warning_count=0,
                test_results=[],
                performance_metrics=[],
                coverage_score=0.0,
                quality_issues=[f"Log file not found: {file_path}"],
                recommendations=["Ensure logging is properly configured"],
            )

        # Parse log entries
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = self.parse_log_entry(line, container_name)
                    if entry:
                        log_entries.append(entry)
                except Exception as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue

        # Extract structured data
        test_results = self.extract_test_results(log_entries)
        performance_metrics = self.extract_performance_metrics(log_entries)

        # Validate quality
        coverage_score, quality_issues = self.validate_log_quality(log_entries)

        # Count errors and warnings
        error_count = sum(
            1 for entry in log_entries if entry.level in ["ERROR", "CRITICAL"]
        )
        warning_count = sum(1 for entry in log_entries if entry.level == "WARNING")

        # Generate recommendations
        recommendations = self._generate_recommendations(
            log_entries, test_results, performance_metrics, quality_issues
        )

        return ValidationResult(
            total_entries=len(log_entries),
            error_count=error_count,
            warning_count=warning_count,
            test_results=test_results,
            performance_metrics=performance_metrics,
            coverage_score=coverage_score,
            quality_issues=quality_issues,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        log_entries: list[LogEntry],
        test_results: list[TestResult],
        performance_metrics: list[PerformanceMetric],
        quality_issues: list[str],
    ) -> list[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Test-related recommendations
        if test_results:
            failed_tests = [t for t in test_results if t.status == "FAILED"]
            if failed_tests:
                recommendations.append(
                    f"Investigate {len(failed_tests)} failed tests: "
                    f"{', '.join(t.test_name for t in failed_tests[:3])}{'...' if len(failed_tests) > 3 else ''}"
                )

            slow_tests = [t for t in test_results if t.duration > 30.0]
            if slow_tests:
                recommendations.append(
                    f"Optimize {len(slow_tests)} slow tests (>30s): "
                    f"{', '.join(t.test_name for t in slow_tests[:3])}{'...' if len(slow_tests) > 3 else ''}"
                )
        else:
            recommendations.append(
                "No test results found - ensure test logging is enabled"
            )

        # Performance-related recommendations
        if performance_metrics:
            # Check CPU usage
            cpu_metrics = [
                m for m in performance_metrics if m.metric_name == "cpu_usage"
            ]
            if cpu_metrics:
                high_cpu = [m for m in cpu_metrics if m.value > 80.0]
                if high_cpu:
                    recommendations.append(
                        f"High CPU usage detected ({len(high_cpu)} instances)"
                    )

            # Check memory usage
            memory_metrics = [
                m for m in performance_metrics if m.metric_name == "memory_usage"
            ]
            if memory_metrics:
                high_memory = [m for m in memory_metrics if m.value > 85.0]
                if high_memory:
                    recommendations.append(
                        f"High memory usage detected ({len(high_memory)} instances)"
                    )

        # Log quality recommendations
        if "Missing required loggers" in str(quality_issues):
            recommendations.append("Configure missing loggers in logging configuration")

        if "error entries" in str(quality_issues):
            recommendations.append("Review and address logged errors")

        if "log gaps" in str(quality_issues):
            recommendations.append(
                "Check for service interruptions or logging configuration issues"
            )

        return recommendations

    def analyze_docker_logs(
        self, container_names: list[str] = None, since: str = "1h"
    ) -> dict[str, ValidationResult]:
        """Analyze logs from Docker containers"""
        results = {}

        if not container_names:
            # Auto-detect running containers
            try:
                result = subprocess.run(
                    ["docker", "ps", "--format", "{{.Names}}"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                container_names = result.stdout.strip().split("\n")
            except subprocess.CalledProcessError:
                print("Failed to get container list")
                return results

        for container_name in container_names:
            if not container_name:
                continue

            try:
                # Get logs from Docker
                result = subprocess.run(
                    ["docker", "logs", "--since", since, container_name],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Parse logs
                log_entries = []
                for line in result.stdout.split("\n"):
                    if line.strip():
                        entry = self.parse_log_entry(line, container_name)
                        if entry:
                            log_entries.append(entry)

                # Extract data and validate
                test_results = self.extract_test_results(log_entries)
                performance_metrics = self.extract_performance_metrics(log_entries)
                coverage_score, quality_issues = self.validate_log_quality(log_entries)

                error_count = sum(
                    1 for entry in log_entries if entry.level in ["ERROR", "CRITICAL"]
                )
                warning_count = sum(
                    1 for entry in log_entries if entry.level == "WARNING"
                )

                recommendations = self._generate_recommendations(
                    log_entries, test_results, performance_metrics, quality_issues
                )

                results[container_name] = ValidationResult(
                    total_entries=len(log_entries),
                    error_count=error_count,
                    warning_count=warning_count,
                    test_results=test_results,
                    performance_metrics=performance_metrics,
                    coverage_score=coverage_score,
                    quality_issues=quality_issues,
                    recommendations=recommendations,
                )

            except subprocess.CalledProcessError as e:
                print(f"Failed to get logs for container {container_name}: {e}")
                results[container_name] = ValidationResult(
                    total_entries=0,
                    error_count=0,
                    warning_count=0,
                    test_results=[],
                    performance_metrics=[],
                    coverage_score=0.0,
                    quality_issues=[f"Failed to retrieve logs: {e}"],
                    recommendations=["Check if container is running and accessible"],
                )

        return results

    def generate_report(
        self, results: dict[str, ValidationResult], output_file: Path = None
    ) -> str:
        """Generate comprehensive analysis report"""
        report_lines = []

        # Header
        report_lines.extend(
            [
                "# Log Analysis and Validation Report",
                f"Generated: {datetime.now().isoformat()}",
                "",
                "## Executive Summary",
            ]
        )

        # Overall statistics
        total_entries = sum(r.total_entries for r in results.values())
        total_errors = sum(r.error_count for r in results.values())
        total_warnings = sum(r.warning_count for r in results.values())
        avg_coverage = (
            sum(r.coverage_score for r in results.values()) / len(results)
            if results
            else 0
        )

        report_lines.extend(
            [
                f"- Total log entries analyzed: {total_entries:,}",
                f"- Total errors found: {total_errors:,}",
                f"- Total warnings found: {total_warnings:,}",
                f"- Average coverage score: {avg_coverage:.1f}%",
                f"- Containers analyzed: {len(results)}",
                "",
            ]
        )

        # Container-specific results
        for container_name, result in results.items():
            report_lines.extend(
                [
                    f"## Container: {container_name}",
                    "",
                    "### Metrics",
                    f"- Log entries: {result.total_entries:,}",
                    f"- Errors: {result.error_count}",
                    f"- Warnings: {result.warning_count}",
                    f"- Coverage score: {result.coverage_score:.1f}%",
                    "",
                ]
            )

            # Test results
            if result.test_results:
                passed_tests = [t for t in result.test_results if t.status == "PASSED"]
                failed_tests = [t for t in result.test_results if t.status == "FAILED"]

                report_lines.extend(
                    [
                        "### Test Results",
                        f"- Total tests: {len(result.test_results)}",
                        f"- Passed: {len(passed_tests)}",
                        f"- Failed: {len(failed_tests)}",
                        "",
                    ]
                )

                if failed_tests:
                    report_lines.append("#### Failed Tests")
                    for test in failed_tests[:5]:  # Show first 5
                        report_lines.append(
                            f"- {test.test_name} ({test.duration:.2f}s)"
                        )
                    if len(failed_tests) > 5:
                        report_lines.append(f"... and {len(failed_tests) - 5} more")
                    report_lines.append("")

            # Performance metrics
            if result.performance_metrics:
                report_lines.extend(
                    [
                        "### Performance Metrics",
                        f"- Total metrics collected: {len(result.performance_metrics)}",
                    ]
                )

                # Group by metric type
                metric_groups = defaultdict(list)
                for metric in result.performance_metrics:
                    metric_groups[metric.metric_name].append(metric)

                for metric_name, metrics in metric_groups.items():
                    values = [m.value for m in metrics]
                    avg_value = sum(values) / len(values)
                    max_value = max(values)
                    report_lines.append(
                        f"- {metric_name}: avg={avg_value:.1f}, max={max_value:.1f}"
                    )

                report_lines.append("")

            # Quality issues
            if result.quality_issues:
                report_lines.extend(
                    [
                        "### Quality Issues",
                    ]
                )
                for issue in result.quality_issues:
                    report_lines.append(f"- {issue}")
                report_lines.append("")

            # Recommendations
            if result.recommendations:
                report_lines.extend(
                    [
                        "### Recommendations",
                    ]
                )
                for rec in result.recommendations:
                    report_lines.append(f"- {rec}")
                report_lines.append("")

            report_lines.append("---")
            report_lines.append("")

        # Generate summary recommendations
        all_recommendations = []
        for result in results.values():
            all_recommendations.extend(result.recommendations)

        if all_recommendations:
            rec_counts = Counter(all_recommendations)
            report_lines.extend(
                [
                    "## Top Recommendations",
                    "",
                ]
            )
            for rec, count in rec_counts.most_common(5):
                report_lines.append(f"- {rec} (affects {count} container(s))")
            report_lines.append("")

        report_content = "\n".join(report_lines)

        # Save to file if specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                f.write(report_content)
            print(f"Report saved to: {output_file}")

        return report_content


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Analyze and validate Docker container logs"
    )

    parser.add_argument(
        "--containers",
        "-c",
        nargs="+",
        help="Container names to analyze (default: auto-detect)",
    )

    parser.add_argument(
        "--since", "-s", default="1h", help="Time period for log analysis (default: 1h)"
    )

    parser.add_argument(
        "--logs-dir", "-d", type=Path, help="Directory containing log files"
    )

    parser.add_argument("--output", "-o", type=Path, help="Output file for report")

    parser.add_argument(
        "--format",
        "-f",
        choices=["text", "json", "yaml"],
        default="text",
        help="Output format",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Create analyzer
    analyzer = LogAnalyzer(args.logs_dir)

    # Analyze logs
    if args.logs_dir and args.logs_dir.exists():
        # Analyze log files
        results = {}
        for log_file in args.logs_dir.glob("*.log"):
            container_name = log_file.stem.replace("-monitor", "").replace(
                "-metrics", ""
            )
            result = analyzer.analyze_log_file(log_file, container_name)
            results[container_name] = result
    else:
        # Analyze Docker container logs
        results = analyzer.analyze_docker_logs(args.containers, args.since)

    # Generate output
    if args.format == "json":
        output = json.dumps(
            {name: asdict(result) for name, result in results.items()},
            indent=2,
            default=str,
        )
    elif args.format == "yaml":
        output = yaml.dump(
            {name: asdict(result) for name, result in results.items()},
            default_flow_style=False,
        )
    else:
        output = analyzer.generate_report(results, args.output)

    if not args.output or args.format != "text":
        print(output)

    # Exit with appropriate code
    total_errors = sum(r.error_count for r in results.values())
    avg_coverage = (
        sum(r.coverage_score for r in results.values()) / len(results) if results else 0
    )

    if total_errors > 0 or avg_coverage < 70:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
