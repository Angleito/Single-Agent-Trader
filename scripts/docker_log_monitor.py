#!/usr/bin/env python3
"""
Docker Log Monitor for Test Execution
Real-time monitoring of Docker container logs with test execution tracking.
"""

import asyncio
import json
import re
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import docker


@dataclass
class LogEvent:
    """Represents a single log event"""

    timestamp: datetime
    container: str
    level: str
    message: str
    source: str
    raw_line: str


@dataclass
class TestEvent:
    """Represents a test execution event"""

    test_name: str
    container: str
    event_type: str  # start, end, pass, fail, skip
    timestamp: datetime
    duration: float | None = None
    error_message: str | None = None


@dataclass
class ContainerStatus:
    """Container status information"""

    name: str
    state: str
    health: str
    start_time: datetime
    restart_count: int
    last_log_time: datetime | None = None


class DockerLogMonitor:
    """Real-time Docker log monitoring with test tracking"""

    def __init__(self, containers: list[str] = None, logs_dir: Path = None):
        self.client = docker.from_env()
        self.target_containers = containers or self._get_project_containers()
        self.logs_dir = logs_dir or Path("/app/logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Event tracking
        self.log_events = deque(maxlen=10000)
        self.test_events = deque(maxlen=1000)
        self.container_statuses = {}
        self.active_tests = {}

        # Monitoring state
        self.running = False
        self.log_streams = {}
        self.stats_collectors = {}

        # Pattern matching
        self.test_patterns = {
            "pytest_start": re.compile(r"(?P<file>\w+\.py)::(?P<test>test_\w+)\s+.*"),
            "pytest_result": re.compile(
                r"(?P<file>\w+\.py)::(?P<test>test_\w+)\s+(?P<result>PASSED|FAILED|SKIPPED|ERROR)"
                r"(?:\s+\[(?P<duration>\d+\.\d+)s\])?"
            ),
            "test_start": re.compile(
                r"(?i)(?:starting|running|executing)\s+test[:\s]+(?P<test>\w+)"
            ),
            "test_end": re.compile(
                r"(?i)test\s+(?P<test>\w+)\s+(?P<result>passed|failed|completed|finished)"
                r"(?:\s+in\s+(?P<duration>\d+\.\d+)s?)?"
            ),
            "docker_compose": re.compile(
                r"(?P<container>[\w-]+)\s*\|\s*(?P<message>.*)"
            ),
            "timestamp": re.compile(
                r"(?P<timestamp>\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)"
            ),
        }

    def _get_project_containers(self) -> list[str]:
        """Auto-detect project containers"""
        try:
            containers = self.client.containers.list(all=True)
            project_containers = []

            # Look for containers with project-related names
            project_names = [
                "ai-trading-bot",
                "dashboard",
                "mcp-memory",
                "postgres",
                "redis",
            ]

            for container in containers:
                container_name = container.name
                if any(name in container_name.lower() for name in project_names):
                    project_containers.append(container_name)

            return project_containers
        except Exception as e:
            print(f"Failed to auto-detect containers: {e}")
            return []

    def parse_log_line(self, container_name: str, line: str) -> LogEvent | None:
        """Parse a log line into structured format"""
        if not line.strip():
            return None

        # Extract timestamp
        timestamp = datetime.now()
        timestamp_match = self.test_patterns["timestamp"].search(line)
        if timestamp_match:
            try:
                ts_str = timestamp_match.group("timestamp")
                # Handle different timestamp formats
                for fmt in [
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S.%f",
                ]:
                    try:
                        timestamp = datetime.strptime(ts_str[:19], fmt[:19])
                        break
                    except ValueError:
                        continue
            except ValueError:
                pass

        # Extract log level
        level = "INFO"
        level_match = re.search(
            r"\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\b", line, re.IGNORECASE
        )
        if level_match:
            level = level_match.group(1).upper()
            if level == "WARN":
                level = "WARNING"

        # Handle docker-compose format
        compose_match = self.test_patterns["docker_compose"].match(line)
        if compose_match:
            actual_container = compose_match.group("container")
            message = compose_match.group("message")
        else:
            actual_container = container_name
            message = line

        return LogEvent(
            timestamp=timestamp,
            container=actual_container,
            level=level,
            message=message.strip(),
            source="docker",
            raw_line=line,
        )

    def detect_test_events(self, log_event: LogEvent) -> list[TestEvent]:
        """Detect test execution events from log messages"""
        events = []
        message = log_event.message

        # Check for pytest format
        pytest_result = self.test_patterns["pytest_result"].match(message)
        if pytest_result:
            groups = pytest_result.groupdict()
            test_name = f"{groups['file']}::{groups['test']}"
            result = groups["result"].lower()
            duration = float(groups["duration"]) if groups.get("duration") else None

            # Create end event
            event_type = {
                "passed": "pass",
                "failed": "fail",
                "skipped": "skip",
                "error": "fail",
            }.get(result, "end")

            events.append(
                TestEvent(
                    test_name=test_name,
                    container=log_event.container,
                    event_type=event_type,
                    timestamp=log_event.timestamp,
                    duration=duration,
                    error_message=message if event_type == "fail" else None,
                )
            )

            # Remove from active tests
            test_key = f"{log_event.container}:{test_name}"
            if test_key in self.active_tests:
                del self.active_tests[test_key]

            return events

        # Check for test start patterns
        test_start = self.test_patterns["test_start"].search(message)
        if test_start:
            test_name = test_start.group("test")
            test_key = f"{log_event.container}:{test_name}"

            events.append(
                TestEvent(
                    test_name=test_name,
                    container=log_event.container,
                    event_type="start",
                    timestamp=log_event.timestamp,
                )
            )

            # Track active test
            self.active_tests[test_key] = log_event.timestamp
            return events

        # Check for test end patterns
        test_end = self.test_patterns["test_end"].search(message)
        if test_end:
            groups = test_end.groupdict()
            test_name = groups["test"]
            result = groups["result"].lower()
            duration = float(groups["duration"]) if groups.get("duration") else None

            test_key = f"{log_event.container}:{test_name}"

            # Calculate duration if not provided
            if not duration and test_key in self.active_tests:
                start_time = self.active_tests[test_key]
                duration = (log_event.timestamp - start_time).total_seconds()

            event_type = (
                "pass" if result in ["passed", "completed", "finished"] else "fail"
            )

            events.append(
                TestEvent(
                    test_name=test_name,
                    container=log_event.container,
                    event_type=event_type,
                    timestamp=log_event.timestamp,
                    duration=duration,
                    error_message=message if event_type == "fail" else None,
                )
            )

            # Remove from active tests
            if test_key in self.active_tests:
                del self.active_tests[test_key]

        return events

    def update_container_status(self, container_name: str):
        """Update container status information"""
        try:
            container = self.client.containers.get(container_name)

            # Parse start time
            start_time = datetime.now()
            try:
                start_time_str = container.attrs["State"]["StartedAt"]
                start_time = datetime.fromisoformat(
                    start_time_str.replace("Z", "+00:00")
                )
            except (KeyError, ValueError):
                pass

            self.container_statuses[container_name] = ContainerStatus(
                name=container_name,
                state=container.status,
                health=container.attrs.get("State", {})
                .get("Health", {})
                .get("Status", "unknown"),
                start_time=start_time,
                restart_count=container.attrs.get("RestartCount", 0),
                last_log_time=datetime.now(),
            )
        except docker.errors.NotFound:
            # Container doesn't exist
            if container_name in self.container_statuses:
                self.container_statuses[container_name].state = "not_found"
        except Exception as e:
            print(f"Error updating status for {container_name}: {e}")

    async def stream_container_logs(self, container_name: str):
        """Stream logs from a single container"""
        try:
            container = self.client.containers.get(container_name)

            # Start streaming logs
            log_stream = container.logs(stream=True, follow=True, since=datetime.now())

            for line in log_stream:
                if not self.running:
                    break

                try:
                    line_str = line.decode("utf-8").strip()
                    if line_str:
                        # Parse log event
                        log_event = self.parse_log_line(container_name, line_str)
                        if log_event:
                            self.log_events.append(log_event)

                            # Detect test events
                            test_events = self.detect_test_events(log_event)
                            self.test_events.extend(test_events)

                            # Update last log time
                            if container_name in self.container_statuses:
                                self.container_statuses[
                                    container_name
                                ].last_log_time = log_event.timestamp

                            # Log to file
                            await self.save_log_event(log_event)

                            # Process test events
                            for test_event in test_events:
                                await self.save_test_event(test_event)

                except UnicodeDecodeError:
                    # Skip binary data
                    continue
                except Exception as e:
                    print(f"Error processing log line from {container_name}: {e}")
                    continue

        except docker.errors.NotFound:
            print(f"Container {container_name} not found")
        except Exception as e:
            print(f"Error streaming logs from {container_name}: {e}")

    async def save_log_event(self, event: LogEvent):
        """Save log event to file"""
        try:
            log_file = self.logs_dir / f"{event.container}-logs.jsonl"
            with open(log_file, "a") as f:
                json.dump(asdict(event), f, default=str)
                f.write("\n")
        except Exception as e:
            print(f"Error saving log event: {e}")

    async def save_test_event(self, event: TestEvent):
        """Save test event to file"""
        try:
            test_file = self.logs_dir / f"{event.container}-tests.jsonl"
            with open(test_file, "a") as f:
                json.dump(asdict(event), f, default=str)
                f.write("\n")
        except Exception as e:
            print(f"Error saving test event: {e}")

    def get_test_summary(self, container: str = None, hours: int = 1) -> dict:
        """Get test execution summary"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Filter events
        recent_events = [
            event
            for event in self.test_events
            if event.timestamp >= cutoff_time
            and (not container or event.container == container)
        ]

        # Group by test and container
        test_results = defaultdict(list)
        for event in recent_events:
            if event.event_type in ["pass", "fail", "skip"]:
                test_results[f"{event.container}:{event.test_name}"].append(event)

        # Calculate statistics
        stats = {
            "total_tests": len(test_results),
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "total_duration": 0.0,
            "avg_duration": 0.0,
            "containers": set(),
            "failed_tests": [],
            "slow_tests": [],
        }

        for test_key, events in test_results.items():
            latest_event = max(events, key=lambda e: e.timestamp)
            stats["containers"].add(latest_event.container)

            if latest_event.event_type == "pass":
                stats["passed"] += 1
            elif latest_event.event_type == "fail":
                stats["failed"] += 1
                stats["failed_tests"].append(
                    {
                        "name": latest_event.test_name,
                        "container": latest_event.container,
                        "error": latest_event.error_message,
                        "timestamp": latest_event.timestamp,
                    }
                )
            elif latest_event.event_type == "skip":
                stats["skipped"] += 1

            if latest_event.duration:
                stats["total_duration"] += latest_event.duration
                if latest_event.duration > 30.0:  # Slow test threshold
                    stats["slow_tests"].append(
                        {
                            "name": latest_event.test_name,
                            "container": latest_event.container,
                            "duration": latest_event.duration,
                        }
                    )

        if stats["total_tests"] > 0:
            stats["avg_duration"] = stats["total_duration"] / stats["total_tests"]

        stats["containers"] = list(stats["containers"])

        return stats

    def get_container_health(self) -> dict:
        """Get overall container health status"""
        health_status = {}

        for container_name, status in self.container_statuses.items():
            # Check log freshness
            log_freshness = "unknown"
            if status.last_log_time:
                age = (datetime.now() - status.last_log_time).total_seconds()
                if age < 60:
                    log_freshness = "fresh"
                elif age < 300:
                    log_freshness = "stale"
                else:
                    log_freshness = "old"

            health_status[container_name] = {
                "state": status.state,
                "health": status.health,
                "uptime": (datetime.now() - status.start_time).total_seconds(),
                "restart_count": status.restart_count,
                "log_freshness": log_freshness,
                "last_log": status.last_log_time,
            }

        return health_status

    def generate_status_report(self) -> str:
        """Generate a comprehensive status report"""
        lines = []

        # Header
        lines.extend(
            [
                "# Docker Log Monitor Status Report",
                f"Generated: {datetime.now().isoformat()}",
                "",
            ]
        )

        # Container status
        lines.append("## Container Status")
        health = self.get_container_health()
        for container_name, status in health.items():
            state_icon = "🟢" if status["state"] == "running" else "🔴"
            lines.extend(
                [
                    f"{state_icon} **{container_name}**",
                    f"  - State: {status['state']}",
                    f"  - Health: {status['health']}",
                    f"  - Uptime: {status['uptime']:.0f}s",
                    f"  - Restarts: {status['restart_count']}",
                    f"  - Log freshness: {status['log_freshness']}",
                    "",
                ]
            )

        # Test summary
        test_summary = self.get_test_summary()
        lines.extend(
            [
                "## Test Summary (Last Hour)",
                f"- Total tests: {test_summary['total_tests']}",
                f"- Passed: {test_summary['passed']} ✅",
                f"- Failed: {test_summary['failed']} ❌",
                f"- Skipped: {test_summary['skipped']} ⏭️",
                f"- Average duration: {test_summary['avg_duration']:.2f}s",
                "",
            ]
        )

        if test_summary["failed_tests"]:
            lines.append("### Failed Tests")
            for test in test_summary["failed_tests"][:5]:
                lines.append(f"- {test['container']}: {test['name']}")
            if len(test_summary["failed_tests"]) > 5:
                lines.append(f"... and {len(test_summary['failed_tests']) - 5} more")
            lines.append("")

        if test_summary["slow_tests"]:
            lines.append("### Slow Tests (>30s)")
            for test in test_summary["slow_tests"][:5]:
                lines.append(
                    f"- {test['container']}: {test['name']} ({test['duration']:.1f}s)"
                )
            lines.append("")

        # Recent log activity
        recent_logs = [
            event
            for event in self.log_events
            if (datetime.now() - event.timestamp).total_seconds() < 300
        ]

        if recent_logs:
            error_logs = [
                log for log in recent_logs if log.level in ["ERROR", "CRITICAL"]
            ]
            warning_logs = [log for log in recent_logs if log.level == "WARNING"]

            lines.extend(
                [
                    "## Recent Activity (Last 5 minutes)",
                    f"- Total log entries: {len(recent_logs)}",
                    f"- Errors: {len(error_logs)}",
                    f"- Warnings: {len(warning_logs)}",
                    "",
                ]
            )

            if error_logs:
                lines.append("### Recent Errors")
                for log in error_logs[-3:]:  # Last 3 errors
                    lines.append(f"- {log.container}: {log.message[:100]}...")
                lines.append("")

        return "\n".join(lines)

    async def start_monitoring(self):
        """Start monitoring all containers"""
        print(
            f"Starting Docker log monitor for containers: {', '.join(self.target_containers)}"
        )

        self.running = True

        # Update container statuses
        for container_name in self.target_containers:
            self.update_container_status(container_name)

        # Start log streaming tasks
        tasks = []
        for container_name in self.target_containers:
            task = asyncio.create_task(self.stream_container_logs(container_name))
            tasks.append(task)

        # Status update task
        async def status_updater():
            while self.running:
                for container_name in self.target_containers:
                    self.update_container_status(container_name)
                await asyncio.sleep(30)  # Update every 30 seconds

        status_task = asyncio.create_task(status_updater())
        tasks.append(status_task)

        # Periodic reporting task
        async def reporter():
            while self.running:
                await asyncio.sleep(300)  # Report every 5 minutes
                report = self.generate_status_report()

                # Save report
                report_file = (
                    self.logs_dir
                    / f"monitor-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
                )
                with open(report_file, "w") as f:
                    f.write(report)

                print(f"Status report saved to: {report_file}")

        report_task = asyncio.create_task(reporter())
        tasks.append(report_task)

        try:
            # Wait for all tasks
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("\nReceived interrupt signal, stopping...")
            self.running = False

            # Cancel all tasks
            for task in tasks:
                task.cancel()

            # Wait for cancellation
            await asyncio.gather(*tasks, return_exceptions=True)

    def stop(self):
        """Stop monitoring"""
        self.running = False


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Docker container logs")
    parser.add_argument(
        "--containers", "-c", nargs="+", help="Container names to monitor"
    )
    parser.add_argument("--logs-dir", "-d", type=Path, help="Logs output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    monitor = DockerLogMonitor(args.containers, args.logs_dir)

    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        monitor.stop()


if __name__ == "__main__":
    asyncio.run(main())
