"""Tests for threaded_gather function."""

import asyncio
import threading
import time

import pytest

from thrasks import threaded_gather, SchedulingMode


@pytest.mark.asyncio
async def test_basic_gather():
    """Test basic gather functionality."""

    async def worker(value: int) -> int:
        await asyncio.sleep(0.01)
        return value * 2

    results = await threaded_gather(
        worker(1),
        worker(2),
        worker(3),
        num_threads=2,
    )

    assert results == [2, 4, 6]


@pytest.mark.asyncio
async def test_gather_empty():
    """Test gather with no arguments."""
    results = await threaded_gather()
    assert results == []


@pytest.mark.asyncio
async def test_gather_single_coroutine():
    """Test gather with single coroutine."""

    async def worker() -> str:
        return "result"

    results = await threaded_gather(worker(), num_threads=1)
    assert results == ["result"]


@pytest.mark.asyncio
async def test_gather_concurrent_execution():
    """Test that gather runs tasks concurrently."""
    start_time = time.time()

    async def sleeper(duration: float) -> float:
        await asyncio.sleep(duration)
        return duration

    results = await threaded_gather(
        sleeper(0.1),
        sleeper(0.1),
        sleeper(0.1),
        num_threads=3,
    )

    elapsed = time.time() - start_time

    # Should complete in ~0.1s if concurrent, not 0.3s
    assert elapsed < 0.2, f"Tasks took {elapsed}s, expected < 0.2s"
    assert results == [0.1, 0.1, 0.1]


@pytest.mark.asyncio
async def test_gather_with_exception():
    """Test gather with exception (default behavior)."""

    async def failing_worker():
        await asyncio.sleep(0.01)
        raise ValueError("Test error")

    async def normal_worker():
        await asyncio.sleep(0.02)
        return "success"

    with pytest.raises(ValueError, match="Test error"):
        await threaded_gather(
            failing_worker(),
            normal_worker(),
            num_threads=2,
        )


@pytest.mark.asyncio
async def test_gather_return_exceptions():
    """Test gather with return_exceptions=True."""

    async def failing_worker():
        await asyncio.sleep(0.01)
        raise ValueError("Test error")

    async def normal_worker():
        await asyncio.sleep(0.02)
        return "success"

    results = await threaded_gather(
        normal_worker(),
        failing_worker(),
        normal_worker(),
        num_threads=2,
        return_exceptions=True,
    )

    assert results[0] == "success"
    assert isinstance(results[1], ValueError)
    assert results[2] == "success"


@pytest.mark.asyncio
async def test_gather_with_tasks():
    """Test that gather works with existing tasks (falls back to asyncio.gather)."""

    async def worker(value: int) -> int:
        await asyncio.sleep(0.01)
        return value * 2

    # Create tasks in current event loop
    task1 = asyncio.create_task(worker(1))
    task2 = asyncio.create_task(worker(2))
    task3 = asyncio.create_task(worker(3))

    results = await threaded_gather(task1, task2, task3, num_threads=2)

    assert results == [2, 4, 6]


@pytest.mark.asyncio
async def test_gather_many_coroutines():
    """Test gather with many coroutines."""

    async def worker(value: int) -> int:
        await asyncio.sleep(0.001)
        return value

    coroutines = [worker(i) for i in range(50)]
    results = await threaded_gather(*coroutines, num_threads=4)

    assert results == list(range(50))


@pytest.mark.asyncio
async def test_gather_thread_distribution():
    """Test that coroutines are distributed across threads."""
    thread_ids = []

    async def worker(value: int) -> int:
        thread_ids.append(threading.get_ident())
        await asyncio.sleep(0.01)
        return value

    await threaded_gather(
        *[worker(i) for i in range(12)],
        num_threads=4,
    )

    # Should have used 4 different threads
    assert len(set(thread_ids)) == 4


@pytest.mark.asyncio
async def test_gather_different_return_types():
    """Test gather with different return types."""

    async def worker_int() -> int:
        return 42

    async def worker_str() -> str:
        return "hello"

    async def worker_list() -> list[int]:
        return [1, 2, 3]

    results = await threaded_gather(
        worker_int(),
        worker_str(),
        worker_list(),
        num_threads=2,
    )

    assert results == [42, "hello", [1, 2, 3]]


@pytest.mark.asyncio
async def test_gather_preserves_order():
    """Test that gather preserves the order of results."""

    async def worker(value: int, delay: float) -> int:
        await asyncio.sleep(delay)
        return value

    results = await threaded_gather(
        worker(1, 0.03),
        worker(2, 0.01),
        worker(3, 0.02),
        num_threads=3,
    )

    # Results should be in submission order, not completion order
    assert results == [1, 2, 3]


@pytest.mark.asyncio
async def test_gather_with_different_thread_counts():
    """Test gather with various thread counts."""

    async def worker(value: int) -> int:
        await asyncio.sleep(0.001)
        return value

    for num_threads in [1, 2, 4, 8]:
        results = await threaded_gather(
            *[worker(i) for i in range(10)],
            num_threads=num_threads,
        )
        assert results == list(range(10))


@pytest.mark.asyncio
async def test_gather_exception_in_multiple_tasks():
    """Test gather with multiple exceptions and return_exceptions=True."""

    async def failing_worker(msg: str):
        await asyncio.sleep(0.01)
        raise ValueError(msg)

    async def normal_worker(value: int) -> int:
        await asyncio.sleep(0.01)
        return value

    results = await threaded_gather(
        normal_worker(1),
        failing_worker("Error 1"),
        normal_worker(2),
        failing_worker("Error 2"),
        num_threads=2,
        return_exceptions=True,
    )

    assert results[0] == 1
    assert isinstance(results[1], ValueError)
    assert results[2] == 2
    assert isinstance(results[3], ValueError)


@pytest.mark.asyncio
async def test_gather_cpu_bound_simulation():
    """Test gather with CPU-bound-like tasks."""

    async def cpu_heavy_worker(value: int) -> int:
        # Simulate CPU-bound work
        total = 0
        for i in range(10000):
            total += i
        return value * 2

    start_time = time.time()
    results = await threaded_gather(
        *[cpu_heavy_worker(i) for i in range(8)],
        num_threads=4,
    )
    elapsed = time.time() - start_time

    assert results == [i * 2 for i in range(8)]
    # With threading, this could potentially be faster than sequential
    # (especially with free-threading enabled)


@pytest.mark.asyncio
async def test_gather_api_compatibility():
    """Test that threaded_gather API is compatible with asyncio.gather."""

    async def worker(value: int) -> int:
        return value * 2

    # Test that it accepts *args like asyncio.gather
    results = await threaded_gather(
        worker(1),
        worker(2),
        worker(3),
    )
    assert results == [2, 4, 6]

    # Test with return_exceptions parameter
    async def failer():
        raise ValueError("error")

    results_with_exc = await threaded_gather(
        worker(1),
        failer(),
        return_exceptions=True,
    )
    assert results_with_exc[0] == 2
    assert isinstance(results_with_exc[1], ValueError)


@pytest.mark.asyncio
async def test_gather_queue_mode_basic():
    """Test basic gather with queue mode."""

    async def worker(value: int) -> int:
        await asyncio.sleep(0.01)
        return value * 2

    results = await threaded_gather(
        worker(1),
        worker(2),
        worker(3),
        num_threads=2,
        mode=SchedulingMode.QUEUE,
    )

    assert results == [2, 4, 6]


@pytest.mark.asyncio
async def test_gather_queue_mode_concurrent():
    """Test that queue mode runs tasks concurrently."""
    start_time = time.time()

    async def sleeper(duration: float) -> float:
        await asyncio.sleep(duration)
        return duration

    results = await threaded_gather(
        sleeper(0.1),
        sleeper(0.1),
        sleeper(0.1),
        num_threads=3,
        mode=SchedulingMode.QUEUE,
    )

    elapsed = time.time() - start_time

    # Should complete in ~0.1s if concurrent, not 0.3s
    assert elapsed < 0.2, f"Tasks took {elapsed}s, expected < 0.2s"
    assert results == [0.1, 0.1, 0.1]


@pytest.mark.asyncio
async def test_gather_queue_mode_uneven_workload():
    """Test queue mode with uneven workload."""
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

    results = await threaded_gather(
        *[variable_work(i) for i in range(15)],
        num_threads=3,
        mode=SchedulingMode.QUEUE,
    )

    assert results == list(range(15))

    # In queue mode, threads should pick up work as they finish
    assert len(thread_usage) == 3


@pytest.mark.asyncio
async def test_gather_queue_mode_with_exception():
    """Test queue mode with exception."""

    async def failing_worker():
        await asyncio.sleep(0.01)
        raise ValueError("Test error")

    async def normal_worker():
        await asyncio.sleep(0.02)
        return "success"

    with pytest.raises(ValueError, match="Test error"):
        await threaded_gather(
            failing_worker(),
            normal_worker(),
            num_threads=2,
            mode=SchedulingMode.QUEUE,
        )


@pytest.mark.asyncio
async def test_gather_queue_mode_return_exceptions():
    """Test queue mode with return_exceptions=True."""

    async def failing_worker():
        await asyncio.sleep(0.01)
        raise ValueError("Test error")

    async def normal_worker():
        await asyncio.sleep(0.02)
        return "success"

    results = await threaded_gather(
        normal_worker(),
        failing_worker(),
        normal_worker(),
        num_threads=2,
        mode=SchedulingMode.QUEUE,
        return_exceptions=True,
    )

    assert results[0] == "success"
    assert isinstance(results[1], ValueError)
    assert results[2] == "success"


@pytest.mark.asyncio
async def test_gather_queue_mode_many_tasks():
    """Test queue mode with many tasks."""

    async def worker(value: int) -> int:
        await asyncio.sleep(0.001)
        return value

    results = await threaded_gather(
        *[worker(i) for i in range(100)],
        num_threads=4,
        mode=SchedulingMode.QUEUE,
    )

    assert results == list(range(100))
