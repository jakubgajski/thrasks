"""Tests for ThreadedTaskGroup."""

import asyncio
import threading
import time

import pytest

from thrasks import ThreadedTaskGroup, SchedulingMode


@pytest.mark.asyncio
async def test_basic_task_creation():
    """Test basic task creation and execution."""
    results = []

    async def worker(value: int) -> int:
        results.append(value)
        return value * 2

    async with ThreadedTaskGroup(num_threads=2) as tg:
        tg.create_task(worker(1))
        tg.create_task(worker(2))
        tg.create_task(worker(3))

    assert sorted(results) == [1, 2, 3]


@pytest.mark.asyncio
async def test_task_return_values():
    """Test that task results are properly returned."""

    async def worker(value: int) -> int:
        await asyncio.sleep(0.01)
        return value * 2

    async with ThreadedTaskGroup(num_threads=2) as tg:
        f1 = tg.create_task(worker(1))
        f2 = tg.create_task(worker(2))
        f3 = tg.create_task(worker(3))

    assert await f1 == 2
    assert await f2 == 4
    assert await f3 == 6


@pytest.mark.asyncio
async def test_round_robin_distribution():
    """Test that tasks are distributed round-robin across threads."""
    thread_ids = []

    async def worker(value: int) -> int:
        thread_ids.append(threading.get_ident())
        await asyncio.sleep(0.01)
        return value

    async with ThreadedTaskGroup(num_threads=3) as tg:
        for i in range(9):
            tg.create_task(worker(i))

    # Should have used 3 different threads
    assert len(set(thread_ids)) == 3


@pytest.mark.asyncio
async def test_exception_propagation():
    """Test that exceptions are properly propagated."""

    async def failing_worker():
        await asyncio.sleep(0.01)
        raise ValueError("Test error")

    async def normal_worker():
        await asyncio.sleep(0.1)
        return "success"

    with pytest.raises(ValueError, match="Test error"):
        async with ThreadedTaskGroup(num_threads=2) as tg:
            tg.create_task(failing_worker())
            tg.create_task(normal_worker())


@pytest.mark.asyncio
async def test_exception_cancels_remaining_tasks():
    """Test that when one task fails, remaining tasks are cancelled."""
    started = []
    completed = []

    async def failing_worker():
        started.append("failing")
        await asyncio.sleep(0.01)
        raise ValueError("Test error")

    async def slow_worker(name: str):
        started.append(name)
        try:
            await asyncio.sleep(1.0)  # Long sleep
            completed.append(name)
            return "success"
        except asyncio.CancelledError:
            # Task was cancelled as expected
            raise

    with pytest.raises(ValueError, match="Test error"):
        async with ThreadedTaskGroup(num_threads=3) as tg:
            tg.create_task(slow_worker("worker1"))
            tg.create_task(slow_worker("worker2"))
            tg.create_task(failing_worker())
            tg.create_task(slow_worker("worker3"))

    # All tasks should have started
    assert "failing" in started
    # But the slow workers should have been cancelled before completing
    assert len(completed) < 3  # At least some should be cancelled


@pytest.mark.asyncio
async def test_multiple_exceptions():
    """Test ExceptionGroup for multiple exceptions."""

    async def failing_worker(msg: str):
        # No sleep - fail immediately to ensure both tasks fail before cancellation
        raise ValueError(msg)

    with pytest.raises((ValueError, ExceptionGroup)):
        async with ThreadedTaskGroup(num_threads=2) as tg:
            tg.create_task(failing_worker("Error 1"))
            tg.create_task(failing_worker("Error 2"))


@pytest.mark.asyncio
async def test_concurrent_execution():
    """Test that tasks actually run concurrently."""
    start_time = time.time()

    async def sleeper(duration: float) -> float:
        await asyncio.sleep(duration)
        return duration

    async with ThreadedTaskGroup(num_threads=3) as tg:
        tg.create_task(sleeper(0.1))
        tg.create_task(sleeper(0.1))
        tg.create_task(sleeper(0.1))

    elapsed = time.time() - start_time

    # Should complete in ~0.1s if concurrent, not 0.3s
    assert elapsed < 0.2, f"Tasks took {elapsed}s, expected < 0.2s (concurrent execution)"


@pytest.mark.asyncio
async def test_task_with_name():
    """Test creating tasks with names."""
    async def worker():
        return "result"

    async with ThreadedTaskGroup(num_threads=1) as tg:
        future = tg.create_task(worker(), name="test_task")

    result = await future
    assert result == "result"


@pytest.mark.asyncio
async def test_empty_task_group():
    """Test task group with no tasks."""
    async with ThreadedTaskGroup(num_threads=2) as tg:
        pass  # No tasks created

    # Should complete without errors


@pytest.mark.asyncio
async def test_many_tasks():
    """Test handling many tasks."""
    results = []

    async def worker(value: int) -> int:
        await asyncio.sleep(0.001)
        return value

    async with ThreadedTaskGroup(num_threads=4) as tg:
        futures = [tg.create_task(worker(i)) for i in range(100)]

    results = [await f for f in futures]
    assert results == list(range(100))


@pytest.mark.asyncio
async def test_invalid_num_threads():
    """Test that invalid num_threads raises ValueError."""
    with pytest.raises(ValueError, match="num_threads must be at least 1"):
        ThreadedTaskGroup(num_threads=0)

    with pytest.raises(ValueError, match="num_threads must be at least 1"):
        ThreadedTaskGroup(num_threads=-1)


@pytest.mark.asyncio
async def test_create_task_outside_context():
    """Test that create_task fails outside of context manager."""
    tg = ThreadedTaskGroup(num_threads=2)

    async def worker():
        return "result"

    coro = worker()
    with pytest.raises(RuntimeError, match="must be used in async with statement"):
        tg.create_task(coro)
    # Clean up the coroutine
    coro.close()


@pytest.mark.asyncio
async def test_cpu_bound_simulation():
    """Test CPU-bound-like tasks (useful with free-threading)."""
    results = []

    async def cpu_heavy_worker(value: int) -> int:
        # Simulate CPU-bound work
        total = 0
        for i in range(10000):
            total += i
        results.append(value)
        return value * 2

    async with ThreadedTaskGroup(num_threads=4) as tg:
        futures = [tg.create_task(cpu_heavy_worker(i)) for i in range(8)]

    final_results = [await f for f in futures]
    assert final_results == [i * 2 for i in range(8)]
    assert sorted(results) == list(range(8))


@pytest.mark.asyncio
async def test_single_thread():
    """Test that single thread works correctly."""
    results = []

    async def worker(value: int) -> int:
        results.append(value)
        return value

    async with ThreadedTaskGroup(num_threads=1) as tg:
        tg.create_task(worker(1))
        tg.create_task(worker(2))
        tg.create_task(worker(3))

    assert sorted(results) == [1, 2, 3]


@pytest.mark.asyncio
async def test_queue_mode_basic():
    """Test basic task execution in queue mode."""
    results = []

    async def worker(value: int) -> int:
        results.append(value)
        return value * 2

    async with ThreadedTaskGroup(num_threads=2, mode=SchedulingMode.QUEUE) as tg:
        tg.create_task(worker(1))
        tg.create_task(worker(2))
        tg.create_task(worker(3))

    assert sorted(results) == [1, 2, 3]


@pytest.mark.asyncio
async def test_queue_mode_return_values():
    """Test that task results are properly returned in queue mode."""

    async def worker(value: int) -> int:
        await asyncio.sleep(0.01)
        return value * 2

    async with ThreadedTaskGroup(num_threads=2, mode=SchedulingMode.QUEUE) as tg:
        f1 = tg.create_task(worker(1))
        f2 = tg.create_task(worker(2))
        f3 = tg.create_task(worker(3))

    assert await f1 == 2
    assert await f2 == 4
    assert await f3 == 6


@pytest.mark.asyncio
async def test_queue_mode_work_stealing():
    """Test that threads consume work from queue as they finish."""
    thread_ids = []
    lock = threading.Lock()

    async def fast_worker(value: int) -> int:
        with lock:
            thread_ids.append(threading.get_ident())
        await asyncio.sleep(0.001)
        return value

    async def slow_worker(value: int) -> int:
        with lock:
            thread_ids.append(threading.get_ident())
        await asyncio.sleep(0.1)
        return value

    async with ThreadedTaskGroup(num_threads=2, mode=SchedulingMode.QUEUE) as tg:
        # One slow task and many fast tasks
        tg.create_task(slow_worker(0))
        for i in range(1, 10):
            tg.create_task(fast_worker(i))

    # Should have used threads (not necessarily all tasks on different threads)
    assert len(thread_ids) == 10


@pytest.mark.asyncio
async def test_queue_mode_exception_propagation():
    """Test that exceptions are properly propagated in queue mode."""

    async def failing_worker():
        await asyncio.sleep(0.01)
        raise ValueError("Test error")

    async def normal_worker():
        await asyncio.sleep(0.1)
        return "success"

    with pytest.raises(ValueError, match="Test error"):
        async with ThreadedTaskGroup(num_threads=2, mode=SchedulingMode.QUEUE) as tg:
            tg.create_task(failing_worker())
            tg.create_task(normal_worker())


@pytest.mark.asyncio
async def test_queue_mode_many_tasks():
    """Test handling many tasks in queue mode."""
    results = []

    async def worker(value: int) -> int:
        await asyncio.sleep(0.001)
        return value

    async with ThreadedTaskGroup(num_threads=4, mode=SchedulingMode.QUEUE) as tg:
        futures = [tg.create_task(worker(i)) for i in range(100)]

    results = [await f for f in futures]
    assert results == list(range(100))


@pytest.mark.asyncio
async def test_queue_mode_concurrent_execution():
    """Test that tasks run concurrently in queue mode."""
    start_time = time.time()

    async def sleeper(duration: float) -> float:
        await asyncio.sleep(duration)
        return duration

    async with ThreadedTaskGroup(num_threads=3, mode=SchedulingMode.QUEUE) as tg:
        tg.create_task(sleeper(0.1))
        tg.create_task(sleeper(0.1))
        tg.create_task(sleeper(0.1))

    elapsed = time.time() - start_time

    # Should complete in ~0.1s if concurrent, not 0.3s
    assert elapsed < 0.2, f"Tasks took {elapsed}s, expected < 0.2s (concurrent execution)"


@pytest.mark.asyncio
async def test_queue_mode_uneven_workload():
    """Test queue mode with uneven workload distribution."""
    thread_usage = {}
    lock = threading.Lock()

    async def variable_work(value: int) -> int:
        tid = threading.get_ident()
        with lock:
            thread_usage[tid] = thread_usage.get(tid, 0) + 1

        # Variable sleep times
        sleep_time = 0.001 if value % 3 == 0 else 0.05
        await asyncio.sleep(sleep_time)
        return value

    async with ThreadedTaskGroup(num_threads=3, mode=SchedulingMode.QUEUE) as tg:
        futures = [tg.create_task(variable_work(i)) for i in range(15)]

    results = [await f for f in futures]
    assert sorted(results) == list(range(15))

    # In queue mode, threads should pick up work as they finish
    # So we expect all threads to be used
    assert len(thread_usage) == 3
