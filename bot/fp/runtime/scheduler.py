"""
Trading Loop Scheduler for Functional Trading Bot

This module provides the main trading loop scheduler that orchestrates
market data collection, strategy execution, and trade management.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from bot.fp.effects.io import IO
from bot.fp.effects.logging import error, info
from bot.fp.effects.monitoring import health_check, increment_counter
from bot.fp.effects.time import now

from .interpreter import get_interpreter


@dataclass
class SchedulerConfig:
    """Scheduler configuration"""

    trading_interval: timedelta = timedelta(minutes=1)
    max_concurrent_tasks: int = 10
    health_check_interval: timedelta = timedelta(minutes=5)
    error_recovery_delay: timedelta = timedelta(seconds=30)


@dataclass
class ScheduledTask:
    """A scheduled task"""

    name: str
    effect: IO[Any]
    interval: timedelta
    last_run: datetime | None = None
    enabled: bool = True
    error_count: int = 0


class TradingScheduler:
    """Main trading scheduler"""

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.tasks: dict[str, ScheduledTask] = {}
        self.running = False
        self.interpreter = get_interpreter()

    def add_task(self, task: ScheduledTask) -> None:
        """Add a scheduled task"""
        self.tasks[task.name] = task
        info(
            f"Added scheduled task: {task.name}",
            {"interval": task.interval.total_seconds(), "enabled": task.enabled},
        ).run()

    def remove_task(self, name: str) -> None:
        """Remove a scheduled task"""
        if name in self.tasks:
            del self.tasks[name]
            info(f"Removed scheduled task: {name}").run()

    def enable_task(self, name: str) -> None:
        """Enable a scheduled task"""
        if name in self.tasks:
            self.tasks[name].enabled = True
            info(f"Enabled task: {name}").run()

    def disable_task(self, name: str) -> None:
        """Disable a scheduled task"""
        if name in self.tasks:
            self.tasks[name].enabled = False
            info(f"Disabled task: {name}").run()

    def should_run_task(self, task: ScheduledTask) -> bool:
        """Check if a task should run"""
        if not task.enabled:
            return False

        if task.last_run is None:
            return True

        current_time = now().run()
        return current_time >= task.last_run + task.interval

    def run_task(self, task: ScheduledTask) -> None:
        """Execute a single task"""
        try:
            info(f"Running task: {task.name}").run()

            # Execute the task effect
            self.interpreter.run_effect(task.effect)

            # Update last run time
            task.last_run = now().run()
            task.error_count = 0

            increment_counter("scheduler.tasks.success", {"task": task.name}).run()

        except Exception as e:
            task.error_count += 1

            error(
                f"Task {task.name} failed: {e!s}",
                {
                    "task": task.name,
                    "error_count": task.error_count,
                    "error_type": type(e).__name__,
                },
            ).run()

            increment_counter("scheduler.tasks.failed", {"task": task.name}).run()

            # Disable task if it fails too many times
            if task.error_count >= 5:
                task.enabled = False
                error(f"Disabled task {task.name} due to repeated failures").run()

    async def run_loop(self) -> None:
        """Main scheduler loop"""
        self.running = True
        info("Trading scheduler started").run()

        try:
            while self.running:
                current_time = now().run()

                # Check which tasks need to run
                tasks_to_run = [
                    task for task in self.tasks.values() if self.should_run_task(task)
                ]

                # Run tasks concurrently (up to max limit)
                if tasks_to_run:
                    info(
                        f"Running {len(tasks_to_run)} scheduled tasks",
                        {
                            "tasks": [task.name for task in tasks_to_run],
                            "current_time": current_time.isoformat(),
                        },
                    ).run()

                    # Limit concurrent tasks
                    for i in range(
                        0, len(tasks_to_run), self.config.max_concurrent_tasks
                    ):
                        batch = tasks_to_run[i : i + self.config.max_concurrent_tasks]

                        # Run batch of tasks concurrently
                        await asyncio.gather(
                            *[
                                asyncio.create_task(self.run_task_async(task))
                                for task in batch
                            ],
                            return_exceptions=True,
                        )

                # Health check
                if self.should_run_health_check():
                    await self.run_health_check()

                # Sleep until next check
                await asyncio.sleep(1.0)  # Check every second

        except Exception as e:
            error(f"Scheduler loop failed: {e!s}").run()
            raise

        finally:
            info("Trading scheduler stopped").run()

    async def run_task_async(self, task: ScheduledTask) -> None:
        """Run a task asynchronously"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.run_task, task)

        except Exception as e:
            error(f"Async task execution failed: {task.name} - {e!s}").run()

    def should_run_health_check(self) -> bool:
        """Check if health check should run"""
        # Simple implementation - could be more sophisticated
        return True

    async def run_health_check(self) -> None:
        """Run system health checks"""
        try:
            # Check scheduler health with proper check function
            def scheduler_check() -> bool:
                # Scheduler is healthy if it's running and has enabled tasks
                return self.running and any(
                    task.enabled for task in self.tasks.values()
                )

            status = health_check("scheduler", scheduler_check).run()

            # Check active tasks
            active_tasks = sum(1 for task in self.tasks.values() if task.enabled)

            info(
                "Health check completed",
                {
                    "status": status.status.value,
                    "active_tasks": active_tasks,
                    "total_tasks": len(self.tasks),
                },
            ).run()

        except Exception as e:
            error(f"Health check failed: {e!s}").run()

    def stop(self) -> None:
        """Stop the scheduler"""
        self.running = False
        info("Scheduler stop requested").run()

    def get_status(self) -> dict[str, Any]:
        """Get scheduler status"""
        return {
            "running": self.running,
            "tasks": {
                name: {
                    "enabled": task.enabled,
                    "last_run": task.last_run.isoformat() if task.last_run else None,
                    "error_count": task.error_count,
                    "interval_seconds": task.interval.total_seconds(),
                }
                for name, task in self.tasks.items()
            },
            "config": {
                "trading_interval": self.config.trading_interval.total_seconds(),
                "max_concurrent_tasks": self.config.max_concurrent_tasks,
            },
        }


# Default scheduler tasks
def create_default_tasks() -> list[ScheduledTask]:
    """Create default scheduler tasks"""
    return [
        ScheduledTask(
            name="market_data_collection",
            effect=IO(lambda: print("Collecting market data")),
            interval=timedelta(seconds=30),
        ),
        ScheduledTask(
            name="strategy_execution",
            effect=IO(lambda: print("Executing strategy")),
            interval=timedelta(minutes=1),
        ),
        ScheduledTask(
            name="risk_monitoring",
            effect=IO(lambda: print("Monitoring risk")),
            interval=timedelta(minutes=5),
        ),
        ScheduledTask(
            name="position_management",
            effect=IO(lambda: print("Managing positions")),
            interval=timedelta(minutes=1),
        ),
    ]


# Global scheduler instance
_scheduler: TradingScheduler | None = None


def get_scheduler() -> TradingScheduler:
    """Get the global scheduler instance"""
    global _scheduler
    if _scheduler is None:
        _scheduler = TradingScheduler(SchedulerConfig())

        # Add default tasks
        for task in create_default_tasks():
            _scheduler.add_task(task)

    return _scheduler
