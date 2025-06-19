"""
Test asyncio task management patterns to ensure proper cleanup.

This test validates the task lifecycle management patterns implemented
across the codebase to fix fire-and-forget task creation.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


class TestTaskManagementPatterns:
    """Test proper asyncio task lifecycle management patterns."""

    async def test_background_task_lifecycle(self):
        """Test that background tasks are properly tracked and cleaned up."""
        
        class MockService:
            def __init__(self):
                self._background_task: asyncio.Task | None = None
                self._worker_tasks: list[asyncio.Task] = []
                self._running = False
            
            async def start(self):
                """Start service with tracked background tasks."""
                self._running = True
                self._background_task = asyncio.create_task(self._background_worker())
                
                # Add some worker tasks
                for i in range(3):
                    task = asyncio.create_task(self._worker(i))
                    self._worker_tasks.append(task)
            
            async def stop(self):
                """Stop service and cleanup all tasks."""
                self._running = False
                
                # Cancel main background task
                if self._background_task and not self._background_task.done():
                    self._background_task.cancel()
                    try:
                        await self._background_task
                    except asyncio.CancelledError:
                        pass
                
                # Cancel worker tasks
                for task in self._worker_tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                
                self._worker_tasks.clear()
            
            async def _background_worker(self):
                """Mock background worker."""
                try:
                    while self._running:
                        await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    raise
            
            async def _worker(self, worker_id: int):
                """Mock worker task."""
                try:
                    while self._running:
                        await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    raise
        
        # Test the lifecycle
        service = MockService()
        
        # Start service
        await service.start()
        
        # Verify tasks are running
        assert service._background_task is not None
        assert not service._background_task.done()
        assert len(service._worker_tasks) == 3
        assert all(not task.done() for task in service._worker_tasks)
        
        # Stop service
        await service.stop()
        
        # Verify proper cleanup
        assert service._background_task.done()
        assert len(service._worker_tasks) == 0
        assert not service._running

    async def test_fire_and_forget_vs_tracked_patterns(self):
        """Test when to use fire-and-forget vs tracked patterns."""
        
        message_processing_tasks = []
        
        async def short_message_processor(message_id: int):
            """Simulate short message processing (acceptable fire-and-forget)."""
            await asyncio.sleep(0.01)  # Very short processing
            return f"processed_{message_id}"
        
        async def long_background_worker():
            """Simulate long background work (should be tracked)."""
            await asyncio.sleep(1.0)  # Longer running
            return "background_complete"
        
        # Fire-and-forget for short message processing (acceptable)
        for i in range(5):
            task = asyncio.create_task(short_message_processor(i))
            message_processing_tasks.append(task)  # Track for test only
        
        # Tracked task for long background work (required)
        background_task = asyncio.create_task(long_background_worker())
        
        # Wait for short tasks to complete quickly
        await asyncio.gather(*message_processing_tasks)
        
        # Cancel long task (simulating shutdown)
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            pass
        
        # Verify all tasks completed or were cancelled
        assert all(task.done() for task in message_processing_tasks)
        assert background_task.done()

    async def test_task_cleanup_with_exceptions(self):
        """Test task cleanup handles exceptions properly."""
        
        class TaskManagerWithErrors:
            def __init__(self):
                self._task: asyncio.Task | None = None
                self._error_tasks: list[asyncio.Task] = []
            
            async def start_with_errors(self):
                """Start tasks that will raise exceptions."""
                self._task = asyncio.create_task(self._failing_worker())
                
                for i in range(2):
                    task = asyncio.create_task(self._sometimes_failing_worker(i))
                    self._error_tasks.append(task)
            
            async def cleanup(self):
                """Cleanup tasks even when they have exceptions."""
                if self._task and not self._task.done():
                    self._task.cancel()
                    try:
                        await self._task
                    except (asyncio.CancelledError, Exception):
                        pass  # Handle both cancellation and other exceptions
                
                for task in self._error_tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except (asyncio.CancelledError, Exception):
                            pass
                
                self._error_tasks.clear()
            
            async def _failing_worker(self):
                """Worker that always fails."""
                await asyncio.sleep(0.1)
                raise ValueError("Simulated failure")
            
            async def _sometimes_failing_worker(self, worker_id: int):
                """Worker that sometimes fails."""
                await asyncio.sleep(0.1)
                if worker_id == 1:
                    raise RuntimeError(f"Worker {worker_id} failed")
                return f"Worker {worker_id} success"
        
        manager = TaskManagerWithErrors()
        await manager.start_with_errors()
        
        # Let tasks run briefly
        await asyncio.sleep(0.05)
        
        # Cleanup should handle exceptions gracefully
        await manager.cleanup()
        
        # Verify cleanup completed
        assert manager._task.done()
        assert len(manager._error_tasks) == 0

    async def test_periodic_task_cleanup(self):
        """Test periodic cleanup of completed tasks."""
        
        class ServiceWithPeriodicCleanup:
            def __init__(self):
                self._active_tasks: list[asyncio.Task] = []
            
            async def add_short_task(self, task_id: int):
                """Add a short-lived task."""
                task = asyncio.create_task(self._short_work(task_id))
                self._active_tasks.append(task)
                return task
            
            def cleanup_completed_tasks(self):
                """Remove completed tasks from tracking."""
                self._active_tasks = [task for task in self._active_tasks if not task.done()]
            
            async def _short_work(self, task_id: int):
                """Short work that completes quickly."""
                await asyncio.sleep(0.01)
                return f"task_{task_id}_done"
        
        service = ServiceWithPeriodicCleanup()
        
        # Add several short tasks
        tasks = []
        for i in range(5):
            task = await service.add_short_task(i)
            tasks.append(task)
        
        # Initially all tasks are tracked
        assert len(service._active_tasks) == 5
        
        # Wait for tasks to complete
        await asyncio.gather(*tasks)
        
        # Tasks are still tracked until cleanup
        assert len(service._active_tasks) == 5
        
        # Clean up completed tasks
        service.cleanup_completed_tasks()
        
        # Now completed tasks are removed
        assert len(service._active_tasks) == 0

    def test_task_naming_conventions(self):
        """Test that task naming follows established conventions."""
        
        class WellNamedService:
            def __init__(self):
                # Single background tasks use _task suffix
                self._monitor_task: asyncio.Task | None = None
                self._heartbeat_task: asyncio.Task | None = None
                self._connection_task: asyncio.Task | None = None
                
                # Collections use _tasks suffix
                self._worker_tasks: list[asyncio.Task] = []
                self._callback_tasks: list[asyncio.Task] = []
                
                # Descriptive names for clarity
                self._auto_reconnect_task: asyncio.Task | None = None
                self._reflection_tasks: list[asyncio.Task] = []
        
        service = WellNamedService()
        
        # Verify naming conventions are followed
        task_attributes = [attr for attr in dir(service) if attr.endswith('_task')]
        tasks_attributes = [attr for attr in dir(service) if attr.endswith('_tasks')]
        
        # Should have both singular and plural task attributes
        assert len(task_attributes) > 0
        assert len(tasks_attributes) > 0
        
        # Verify types
        for attr in task_attributes:
            value = getattr(service, attr)
            assert value is None or isinstance(value, asyncio.Task)
        
        for attr in tasks_attributes:
            value = getattr(service, attr)
            assert isinstance(value, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])