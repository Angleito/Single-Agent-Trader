# AsyncIO Task Management Fixes

## Summary of Changes

This document outlines the fixes applied to address "fire-and-forget" asyncio.create_task() patterns throughout the codebase to prevent resource leaks and improve shutdown reliability.

## Issues Fixed

### 1. WebSocket Publisher (`bot/websocket_publisher.py`)

**Problem**: Auto-reconnect manager task was created without storing reference
```python
# Before (problematic)
asyncio.create_task(self._auto_reconnect_manager())  # No reference stored!
```

**Fix**: Added instance variable tracking and proper cleanup
```python
# After (fixed)
self._auto_reconnect_task: asyncio.Task | None = None  # Instance variable
self._auto_reconnect_task = asyncio.create_task(self._auto_reconnect_manager())  # Store reference

# In close() method:
if self._auto_reconnect_task and not self._auto_reconnect_task.done():
    self._auto_reconnect_task.cancel()
    try:
        await self._auto_reconnect_task
    except asyncio.CancelledError:
        pass
```

### 2. Dashboard Backend (`dashboard/backend/main.py`)

**Problem**: Multiple fire-and-forget tasks in LogStreamer class
```python
# Before (problematic)
asyncio.create_task(self._stream_logs())  # No reference stored!
asyncio.create_task(self._watch_log_file(log_path))  # No reference stored!
asyncio.create_task(start_log_streaming_delayed())  # No reference stored!
```

**Fix**: Added comprehensive task tracking
```python
# After (fixed)
# Instance variables for tracking
self._stream_logs_task: asyncio.Task | None = None
self._file_watcher_tasks: list[asyncio.Task] = []
_delayed_startup_task: asyncio.Task | None = None  # Module level

# Store references when creating tasks
self._stream_logs_task = asyncio.create_task(self._stream_logs())
task = asyncio.create_task(self._watch_log_file(log_path))
self._file_watcher_tasks.append(task)
_delayed_startup_task = asyncio.create_task(start_log_streaming_delayed())

# Proper async cleanup in stop() method
async def stop(self):
    self.running = False

    # Cancel all tracked tasks
    if self._stream_logs_task and not self._stream_logs_task.done():
        self._stream_logs_task.cancel()
        try:
            await self._stream_logs_task
        except asyncio.CancelledError:
            pass

    for task in self._file_watcher_tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
```

### 3. Experience Manager (`bot/learning/experience_manager.py`)

**Problem**: Reflection scheduling tasks not tracked
```python
# Before (problematic)
asyncio.create_task(
    self._schedule_trade_reflection(active_trade, delay_minutes=...)
)  # No reference stored!
```

**Fix**: Added task list tracking with periodic cleanup
```python
# After (fixed)
# Instance variable for tracking
self._reflection_tasks: list[asyncio.Task] = []

# Store and track reflection tasks
reflection_task = asyncio.create_task(
    self._schedule_trade_reflection(active_trade, delay_minutes=...)
)
self._reflection_tasks.append(reflection_task)

# Clean up completed tasks periodically
self._reflection_tasks = [task for task in self._reflection_tasks if not task.done()]

# Proper cleanup in stop() method
for task in self._reflection_tasks:
    if not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
self._reflection_tasks.clear()
```

## Patterns NOT Changed (Acceptable Fire-and-Forget)

Some fire-and-forget patterns were kept because they are appropriate:

### 1. Message Processing Tasks
```python
# Acceptable: Short-lived message processing
asyncio.create_task(self._handle_websocket_message_async(message))
```
**Rationale**: These are short-lived tasks that process individual messages. Creating too many tracked references would cause memory overhead.

### 2. Callback Notification Tasks
```python
# Acceptable: Event notification callbacks
asyncio.create_task(self._safe_callback_async(callback, data))
```
**Rationale**: These are brief notification tasks that don't need lifecycle management.

### 3. Background Event Broadcasting
```python
# Acceptable: Event broadcasting in handlers
asyncio.create_task(manager.broadcast(formatted_event))
```
**Rationale**: These are fire-and-forget event notifications that should not block the caller.

## Best Practices Established

### 1. Task Instance Variable Naming Convention
- Use `_task` suffix for single background tasks: `self._monitor_task`
- Use `_tasks` suffix for collections: `self._reflection_tasks`
- Use descriptive names: `self._auto_reconnect_task`, `self._heartbeat_task`

### 2. Task Lifecycle Management
```python
class MyService:
    def __init__(self):
        self._background_task: asyncio.Task | None = None
        self._worker_tasks: list[asyncio.Task] = []

    async def start(self):
        self._background_task = asyncio.create_task(self._background_worker())

    async def stop(self):
        # Cancel single task
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

        # Cancel task list
        for task in self._worker_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._worker_tasks.clear()
```

### 3. When to Track vs Fire-and-Forget

**Track these tasks:**
- Long-running background workers
- Connection managers
- Monitoring loops
- Scheduled operations with significant delay
- Tasks that hold resources

**Fire-and-forget acceptable for:**
- Individual message processors
- Brief event notifications
- Short callback executions
- Error handling tasks

### 4. Error Handling for Background Tasks
```python
# For critical background tasks, add done callbacks
task = asyncio.create_task(self._critical_background_task())
task.add_done_callback(self._handle_task_error)

def _handle_task_error(self, task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception():
        logger.error(f"Background task failed: {task.exception()}")
```

## Testing Considerations

When writing tests for classes with background tasks:

```python
async def test_service_cleanup():
    service = MyService()
    await service.start()

    # Test that tasks are running
    assert service._background_task is not None
    assert not service._background_task.done()

    # Test proper cleanup
    await service.stop()
    assert service._background_task.done()
    assert len(service._worker_tasks) == 0
```

## Integration Test Patterns

For integration tests with multiple async tasks:

```python
async def test_multiple_tasks():
    tasks = []
    try:
        # Create tracked tasks
        tasks.append(asyncio.create_task(long_running_operation()))
        tasks.append(asyncio.create_task(another_operation()))

        # Test logic here
        results = await asyncio.gather(*tasks[:2])  # Wait for first 2 tasks

    finally:
        # Clean up all tasks
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
```

## Impact

These fixes ensure:
1. **Resource cleanup**: No more leaked background tasks on shutdown
2. **Graceful shutdown**: All background operations properly cancelled
3. **Memory management**: Task references cleaned up appropriately
4. **Debugging**: Background tasks can be monitored and inspected
5. **Reliability**: Consistent patterns across the codebase

## Files Modified

1. `/bot/websocket_publisher.py` - Fixed auto-reconnect task tracking
2. `/dashboard/backend/main.py` - Fixed LogStreamer task management
3. `/bot/learning/experience_manager.py` - Fixed reflection task tracking

All changes maintain backward compatibility while improving resource management.
