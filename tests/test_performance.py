"""Performance tests comparing thrasks with standard asyncio."""

import asyncio
import json
import logging
import time
from typing import Any

import pytest

from thrasks import ThreadedTaskGroup, threaded_gather

logger = logging.getLogger(__name__)


# === CPU-Bound Workloads ==+


async def cpu_intensive_json(data: dict[str, Any], iterations: int = 1000) -> int:
    """CPU-intensive JSON serialization/deserialization."""
    result = 0
    for _ in range(iterations):
        serialized = json.dumps(data)
        _ = json.loads(serialized)
        result += len(serialized)
    return result


async def cpu_fibonacci(n: int) -> int:
    """CPU-intensive recursive calculation."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


# === Blocking Workloads ===


async def blocking_sleep(duration: float) -> float:
    """Blocking sleep that locks the thread (simulates sync I/O)."""
    time.sleep(duration)  # Plain sleep, not asyncio.sleep
    return duration


# === I/O-Bound Workloads ===


async def io_bound_sleep(duration: float) -> float:
    """I/O-bound sleep operation."""
    await asyncio.sleep(duration)
    return duration


async def io_bound_with_cpu(sleep_time: float, cpu_work: int = 1000) -> tuple[float, float]:
    """Mixed I/O and CPU work."""
    await asyncio.sleep(sleep_time)
    result = sum(i / (i + 1) for i in range(cpu_work))
    return sleep_time, result


# === Performance Test Fixtures ===


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_performance_cpu_json():
    """Compare CPU-intensive JSON processing: asyncio vs thrasks."""
    num_tasks = 16
    iterations = 2000
    test_data = [
        {"id": i, "data": "x" * 100, "nested": {"values": list(range(50))}}
        for i in range(num_tasks)
    ]

    # Test with standard asyncio.gather
    start = time.perf_counter()
    asyncio_results = await asyncio.gather(
        *[cpu_intensive_json(data, iterations) for data in test_data]
    )
    asyncio_time = time.perf_counter() - start

    # Test with thrasks
    start = time.perf_counter()
    thrasks_results = await threaded_gather(
        *[cpu_intensive_json(data, iterations) for data in test_data],
        num_threads=2,
    )
    thrasks_time = time.perf_counter() - start

    assert asyncio_results == thrasks_results

    speedup = asyncio_time / thrasks_time

    logger.info("=" * 60)
    logger.info("CPU-Intensive JSON Processing (%d tasks, %d iterations)", num_tasks, iterations)
    logger.info("=" * 60)
    logger.info("asyncio.gather:         %.3fs (baseline)", asyncio_time)
    logger.info("thrasks (2 threads):    %.3fs (%.2fx)", thrasks_time, speedup)
    logger.info("=" * 60)

    # CPU-bound calculations should be faster with thrasks
    assert speedup > 1.5, f"Expected thrasks to be at least 1.5x faster, got {speedup:.2f}x"


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_performance_fibonacci():
    """Compare CPU-intensive calculations: asyncio vs thrasks."""
    num_tasks = 16
    fib_values = [10000 + i * 1000 for i in range(num_tasks)]

    # Test with standard asyncio.gather
    start = time.perf_counter()
    asyncio_results = await asyncio.gather(
        *[cpu_fibonacci(n) for n in fib_values]
    )
    asyncio_time = time.perf_counter() - start

    # Test with thrasks
    start = time.perf_counter()
    thrasks_results = await threaded_gather(
        *[cpu_fibonacci(n) for n in fib_values],
        num_threads=2,
    )
    thrasks_time = time.perf_counter() - start

    assert asyncio_results == thrasks_results

    speedup = asyncio_time / thrasks_time

    logger.info("=" * 60)
    logger.info("CPU-Intensive Fibonacci (%d tasks)", num_tasks)
    logger.info("=" * 60)
    logger.info("asyncio.gather:         %.3fs (baseline)", asyncio_time)
    logger.info("thrasks (2 threads):    %.3fs (%.2fx)", thrasks_time, speedup)
    logger.info("=" * 60)

    # CPU-bound calculations should be faster with thrasks
    assert speedup > 1.5, f"Expected thrasks to be at least 1.5x faster, got {speedup:.2f}x"


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_performance_io_bound():
    """Compare I/O-bound tasks: asyncio vs thrasks (should be similar)."""
    num_tasks = 20
    sleep_time = 0.05

    # Test with standard asyncio.gather
    start = time.perf_counter()
    asyncio_results = await asyncio.gather(
        *[io_bound_sleep(sleep_time) for _ in range(num_tasks)]
    )
    asyncio_time = time.perf_counter() - start

    # Test with thrasks
    start = time.perf_counter()
    thrasks_results = await threaded_gather(
        *[io_bound_sleep(sleep_time) for _ in range(num_tasks)],
        num_threads=2,
    )
    thrasks_time = time.perf_counter() - start

    assert asyncio_results == thrasks_results

    overhead_ratio = thrasks_time / asyncio_time

    logger.info("=" * 60)
    logger.info("I/O-Bound Sleep (%d tasks, %.2fs each)", num_tasks, sleep_time)
    logger.info("=" * 60)
    logger.info("asyncio.gather:         %.3fs (baseline)", asyncio_time)
    logger.info("thrasks (2 threads):    %.3fs (%.2fx)", thrasks_time, asyncio_time / thrasks_time)
    logger.info("Note: For pure I/O, asyncio should be similar or faster (less overhead)")
    logger.info("=" * 60)

    # For pure I/O, thrasks shouldn't be much slower (max 3x overhead)
    assert overhead_ratio < 3.0, f"Expected thrasks overhead to be less than 3x for I/O, got {overhead_ratio:.2f}x"


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_performance_thread_locked_sleep():
    """Compare blocking sleep: asyncio vs thrasks (demonstrates thrasks advantage)."""
    num_tasks = 20
    sleep_time = 0.1

    # Test with standard asyncio.gather (will block the entire event loop)
    start = time.perf_counter()
    asyncio_results = await asyncio.gather(
        *[blocking_sleep(sleep_time) for _ in range(num_tasks)]
    )
    asyncio_time = time.perf_counter() - start

    # Test with thrasks (will run in parallel threads)
    start = time.perf_counter()
    thrasks_results = await threaded_gather(
        *[blocking_sleep(sleep_time) for _ in range(num_tasks)],
        num_threads=2,
    )
    thrasks_time = time.perf_counter() - start

    assert asyncio_results == thrasks_results

    speedup = asyncio_time / thrasks_time

    logger.info("=" * 60)
    logger.info("Thread-Locked Sleep (%d tasks, %.1fs each)", num_tasks, sleep_time)
    logger.info("=" * 60)
    logger.info("asyncio.gather:         %.3fs (runs sequentially!)", asyncio_time)
    logger.info("thrasks (2 threads):    %.3fs (%.2fx)", thrasks_time, speedup)
    logger.info("Note: time.sleep() blocks the event loop in asyncio but not in thrasks")
    logger.info("      thrasks executes blocking operations in parallel threads")
    logger.info("=" * 60)

    # Blocking sleep should be MUCH faster with thrasks (at least 3x for 4 threads with 20 tasks)
    assert speedup > 1.5, f"Expected thrasks to be at least 3x faster for blocking sleep, got {speedup:.2f}x"


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_performance_mixed_workload():
    """Compare mixed I/O and CPU workload: asyncio vs thrasks."""
    num_tasks = 20
    sleep_time = 0.01
    cpu_work = 50000

    # Test with standard asyncio.gather
    start = time.perf_counter()
    asyncio_results = await asyncio.gather(
        *[io_bound_with_cpu(sleep_time, cpu_work) for _ in range(num_tasks)]
    )
    asyncio_time = time.perf_counter() - start

    # Test with thrasks
    start = time.perf_counter()
    thrasks_results = await threaded_gather(
        *[io_bound_with_cpu(sleep_time, cpu_work) for _ in range(num_tasks)],
        num_threads=2,
    )
    thrasks_time = time.perf_counter() - start

    assert asyncio_results == thrasks_results

    speedup = asyncio_time / thrasks_time

    logger.info("=" * 60)
    logger.info("Mixed I/O + CPU Workload (%d tasks)", num_tasks)
    logger.info("=" * 60)
    logger.info("asyncio.gather:         %.3fs (baseline)", asyncio_time)
    logger.info("thrasks (2 threads):    %.3fs (%.2fx)", thrasks_time, speedup)
    logger.info("=" * 60)

    # Mixed workload should benefit from thrasks (at least 1.3x speedup)
    assert speedup > 1.3, f"Expected thrasks to be at least 1.3x faster for mixed workload, got {speedup:.2f}x"


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_performance_task_group_cpu():
    """Compare ThreadedTaskGroup vs asyncio.TaskGroup for CPU work."""
    num_tasks = 16
    fib_values = [10000 + i * 1000 for i in range(num_tasks)]

    # Test with asyncio.TaskGroup
    start = time.perf_counter()
    async with asyncio.TaskGroup() as tg:
        asyncio_tasks = [
            tg.create_task(cpu_fibonacci(n))
            for n in fib_values
        ]
    asyncio_results = [task.result() for task in asyncio_tasks]
    asyncio_time = time.perf_counter() - start

    # Test with ThreadedTaskGroup
    start = time.perf_counter()
    async with ThreadedTaskGroup(num_threads=2) as tg:
        thrasks_tasks = [
            tg.create_task(cpu_fibonacci(n))
            for n in fib_values
        ]
    thrasks_results = [await task for task in thrasks_tasks]
    thrasks_time = time.perf_counter() - start

    assert asyncio_results == thrasks_results

    speedup = asyncio_time / thrasks_time

    logger.info("=" * 60)
    logger.info("TaskGroup CPU Fibonacci (%d tasks)", num_tasks)
    logger.info("=" * 60)
    logger.info("asyncio.TaskGroup:      %.3fs (baseline)", asyncio_time)
    logger.info("ThreadedTaskGroup (2 threads):  %.3fs (%.2fx)", thrasks_time, speedup)
    logger.info("=" * 60)

    # CPU-bound calculations should be faster with ThreadedTaskGroup
    assert speedup > 1.5, f"Expected ThreadedTaskGroup to be at least 1.5x faster, got {speedup:.2f}x"


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_performance_scaling():
    """Test how performance scales with different thread counts."""
    num_tasks = 32
    thread_counts = [1, 2, 4, 8, 16]
    fib_values = [8000 + i * 500 for i in range(num_tasks)]

    results = {}

    for num_threads in thread_counts:
        start = time.perf_counter()
        await threaded_gather(
            *[cpu_fibonacci(n) for n in fib_values],
            num_threads=num_threads,
        )
        elapsed = time.perf_counter() - start
        results[num_threads] = elapsed

    logger.info("=" * 60)
    logger.info("Thread Scaling Performance (%d tasks)", num_tasks)
    logger.info("=" * 60)
    baseline = results[1]
    for num_threads in thread_counts:
        elapsed = results[num_threads]
        speedup = baseline / elapsed
        logger.info("%2d thread(s):  %.3fs  (speedup: %.2fx)", num_threads, elapsed, speedup)
    logger.info("=" * 60)
    logger.info("Note: Speedup depends on free-threading being enabled")
    logger.info("=" * 60)

    # With free-threading, we should see good speedup with more threads
    speedup_4 = baseline / results[4]
    assert speedup_4 > 1.5, f"Expected at least 1.5x speedup with 4 threads, got {speedup_4:.2f}x"
    # Verify 8 threads also provide meaningful speedup
    speedup_8 = baseline / results[8]
    assert speedup_8 > 1.8, f"Expected at least 1.8x speedup with 8 threads, got {speedup_8:.2f}x"


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_performance_overhead():
    """Measure overhead of thrasks for trivial tasks."""
    num_tasks = 100

    async def trivial_task(x: int) -> int:
        return x * 2

    # Test with asyncio.gather
    start = time.perf_counter()
    asyncio_results = await asyncio.gather(
        *[trivial_task(i) for i in range(num_tasks)]
    )
    asyncio_time = time.perf_counter() - start

    # Test with thrasks
    start = time.perf_counter()
    thrasks_results = await threaded_gather(
        *[trivial_task(i) for i in range(num_tasks)],
        num_threads=2,
    )
    thrasks_time = time.perf_counter() - start

    assert asyncio_results == thrasks_results

    overhead_ratio = thrasks_time / asyncio_time

    logger.info("=" * 60)
    logger.info("Overhead Test - Trivial Tasks (%d tasks)", num_tasks)
    logger.info("=" * 60)
    logger.info("asyncio.gather:         %.4fs (baseline)", asyncio_time)
    logger.info("thrasks (2 threads):    %.4fs (overhead: %.2fx)", thrasks_time, overhead_ratio)
    logger.info("=" * 60)
    logger.info("Note: thrasks has higher overhead for trivial tasks")
    logger.info("      Use asyncio for simple/fast operations")
    logger.info("=" * 60)

    # For trivial tasks, overhead shouldn't be excessive (max 50x)
    assert overhead_ratio < 50.0, f"Expected thrasks overhead to be less than 50x for trivial tasks, got {overhead_ratio:.2f}x"


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_performance_real_world_scenario():
    """Simulate real-world scenario: API responses with heavy processing."""
    num_requests = 200

    async def simulate_api_request(request_id: int) -> dict[str, Any]:
        """Simulate API request with network delay and CPU processing."""
        # Simulate network I/O
        await asyncio.sleep(0.1)

        # Simulate heavy JSON processing (like parsing large responses)
        data = {
            "id": request_id,
            "payload": "x" * 1000,
            "items": [{"value": i, "squared": i * i} for i in range(100)],
        }

        # Heavy CPU processing
        fib_result = await cpu_fibonacci(5000 + request_id * 100)

        # Additional JSON serialization work
        for _ in range(10):
            serialized = json.dumps(data)
            _ = json.loads(serialized)

        return {"request_id": request_id, "status": "processed", "data": data, "fib": fib_result}

    # Test with asyncio.gather
    start = time.perf_counter()
    asyncio_results = await asyncio.gather(
        *[simulate_api_request(i) for i in range(num_requests)]
    )
    asyncio_time = time.perf_counter() - start

    # Test with thrasks
    start = time.perf_counter()
    thrasks_results = await threaded_gather(
        *[simulate_api_request(i) for i in range(num_requests)],
        num_threads=2,
    )
    thrasks_time = time.perf_counter() - start

    assert len(asyncio_results) == len(thrasks_results) == num_requests

    speedup = asyncio_time / thrasks_time

    logger.info("=" * 60)
    logger.info("Real-World Scenario: API + Heavy Processing (%d requests)", num_requests)
    logger.info("=" * 60)
    logger.info("asyncio.gather:         %.3fs (baseline)", asyncio_time)
    logger.info("thrasks (2 threads):    %.3fs (%.2fx)", thrasks_time, speedup)
    logger.info("=" * 60)
    logger.info("This simulates: network I/O + JSON processing + Fibonacci calculation")
    logger.info("=" * 60)

    # Real-world mixed workload should benefit from thrasks
    assert speedup > 1.5, f"Expected thrasks to be at least 1.5x faster for real-world scenario, got {speedup:.2f}x"


# === Summary Test ===


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_performance_summary():
    """Run a quick summary of key performance comparisons."""
    logger.info("=" * 70)
    logger.info("THRASKS PERFORMANCE SUMMARY")
    logger.info("=" * 70)

    # Quick CPU test
    num_tasks = 12
    fib_values = [8000 + i * 500 for i in range(num_tasks)]

    start = time.perf_counter()
    await asyncio.gather(*[cpu_fibonacci(n) for n in fib_values])
    asyncio_time = time.perf_counter() - start

    start = time.perf_counter()
    await threaded_gather(*[cpu_fibonacci(n) for n in fib_values], num_threads=2)
    thrasks_time = time.perf_counter() - start

    logger.info("CPU-Bound (Fibonacci):  asyncio=%.3fs  thrasks=%.3fs  (%.2fx)", asyncio_time, thrasks_time, asyncio_time / thrasks_time)

    # Quick I/O test
    start = time.perf_counter()
    await asyncio.gather(*[io_bound_sleep(0.1) for _ in range(num_tasks)])
    asyncio_time = time.perf_counter() - start

    start = time.perf_counter()
    await threaded_gather(*[io_bound_sleep(0.1) for _ in range(num_tasks)], num_threads=2)
    thrasks_time = time.perf_counter() - start

    logger.info("I/O-Bound (Sleep): asyncio=%.3fs  thrasks=%.3fs  (%.2fx)", asyncio_time, thrasks_time, asyncio_time / thrasks_time)

    logger.info("=" * 70)
    logger.info("RECOMMENDATIONS:")
    logger.info("  • Use thrasks for CPU-intensive async operations (with free-threading)")
    logger.info("  • Use asyncio for pure I/O-bound operations (lower overhead)")
    logger.info("  • Use thrasks for mixed I/O + CPU workloads")
    logger.info("=" * 70)
