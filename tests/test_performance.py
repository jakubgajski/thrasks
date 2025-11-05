"""Performance tests comparing thrasks with standard asyncio."""

import asyncio
import hashlib
import json
import time
from typing import Any

import pytest

from thrasks import ThreadedTaskGroup, threaded_gather


# === CPU-Bound Workloads ===


async def cpu_intensive_hash(data: bytes, iterations: int = 1000) -> str:
    """CPU-intensive hashing operation."""
    result = data
    for _ in range(iterations):
        result = hashlib.sha256(result).digest()
    return result.hex()


async def cpu_intensive_json(data: dict[str, Any], iterations: int = 100) -> int:
    """CPU-intensive JSON serialization/deserialization."""
    result = 0
    for _ in range(iterations):
        serialized = json.dumps(data)
        deserialized = json.loads(serialized)
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


# === I/O-Bound Workloads ===


async def io_bound_sleep(duration: float) -> float:
    """I/O-bound sleep operation."""
    await asyncio.sleep(duration)
    return duration


async def io_bound_with_cpu(sleep_time: float, cpu_work: int = 100) -> tuple[float, int]:
    """Mixed I/O and CPU work."""
    await asyncio.sleep(sleep_time)
    result = sum(i * i for i in range(cpu_work))
    return sleep_time, result


# === Performance Test Fixtures ===


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_performance_cpu_hashing():
    """Compare CPU-intensive hashing: asyncio vs thrasks."""
    num_tasks = 20
    iterations = 500
    data_chunks = [f"chunk-{i}".encode() for i in range(num_tasks)]

    # Test with standard asyncio.gather
    start = time.perf_counter()
    asyncio_results = await asyncio.gather(
        *[cpu_intensive_hash(chunk, iterations) for chunk in data_chunks]
    )
    asyncio_time = time.perf_counter() - start

    # Test with thrasks (4 threads)
    start = time.perf_counter()
    thrasks_results_4 = await threaded_gather(
        *[cpu_intensive_hash(chunk, iterations) for chunk in data_chunks],
        num_threads=4,
    )
    thrasks_time_4 = time.perf_counter() - start

    # Test with thrasks (8 threads)
    start = time.perf_counter()
    thrasks_results_8 = await threaded_gather(
        *[cpu_intensive_hash(chunk, iterations) for chunk in data_chunks],
        num_threads=8,
    )
    thrasks_time_8 = time.perf_counter() - start

    # Verify results are identical
    assert asyncio_results == thrasks_results_4 == thrasks_results_8

    # Report performance
    print(f"\n{'=' * 60}")
    print(f"CPU-Intensive Hashing Performance ({num_tasks} tasks, {iterations} iterations)")
    print(f"{'=' * 60}")
    print(f"asyncio.gather:         {asyncio_time:.3f}s (baseline)")
    print(f"thrasks (4 threads):    {thrasks_time_4:.3f}s ({asyncio_time / thrasks_time_4:.2f}x)")
    print(f"thrasks (8 threads):    {thrasks_time_8:.3f}s ({asyncio_time / thrasks_time_8:.2f}x)")
    print(f"{'=' * 60}")

    # Note: With free-threading, thrasks should be faster
    # Without free-threading (GIL), times will be similar


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_performance_cpu_json():
    """Compare CPU-intensive JSON processing: asyncio vs thrasks."""
    num_tasks = 16
    iterations = 50
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
        num_threads=4,
    )
    thrasks_time = time.perf_counter() - start

    assert asyncio_results == thrasks_results

    print(f"\n{'=' * 60}")
    print(f"CPU-Intensive JSON Processing ({num_tasks} tasks, {iterations} iterations)")
    print(f"{'=' * 60}")
    print(f"asyncio.gather:         {asyncio_time:.3f}s (baseline)")
    print(f"thrasks (4 threads):    {thrasks_time:.3f}s ({asyncio_time / thrasks_time:.2f}x)")
    print(f"{'=' * 60}")


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
        num_threads=4,
    )
    thrasks_time = time.perf_counter() - start

    assert asyncio_results == thrasks_results

    print(f"\n{'=' * 60}")
    print(f"CPU-Intensive Fibonacci ({num_tasks} tasks)")
    print(f"{'=' * 60}")
    print(f"asyncio.gather:         {asyncio_time:.3f}s (baseline)")
    print(f"thrasks (4 threads):    {thrasks_time:.3f}s ({asyncio_time / thrasks_time:.2f}x)")
    print(f"{'=' * 60}")


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
        num_threads=4,
    )
    thrasks_time = time.perf_counter() - start

    assert asyncio_results == thrasks_results

    print(f"\n{'=' * 60}")
    print(f"I/O-Bound Sleep ({num_tasks} tasks, {sleep_time}s each)")
    print(f"{'=' * 60}")
    print(f"asyncio.gather:         {asyncio_time:.3f}s (baseline)")
    print(f"thrasks (4 threads):    {thrasks_time:.3f}s ({asyncio_time / thrasks_time:.2f}x)")
    print(f"Note: For pure I/O, asyncio should be similar or faster (less overhead)")
    print(f"{'=' * 60}")


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_performance_mixed_workload():
    """Compare mixed I/O and CPU workload: asyncio vs thrasks."""
    num_tasks = 20
    sleep_time = 0.01
    cpu_work = 10000

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
        num_threads=4,
    )
    thrasks_time = time.perf_counter() - start

    assert asyncio_results == thrasks_results

    print(f"\n{'=' * 60}")
    print(f"Mixed I/O + CPU Workload ({num_tasks} tasks)")
    print(f"{'=' * 60}")
    print(f"asyncio.gather:         {asyncio_time:.3f}s (baseline)")
    print(f"thrasks (4 threads):    {thrasks_time:.3f}s ({asyncio_time / thrasks_time:.2f}x)")
    print(f"{'=' * 60}")


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_performance_task_group_cpu():
    """Compare ThreadedTaskGroup vs asyncio.TaskGroup for CPU work."""
    num_tasks = 16
    iterations = 300
    data_chunks = [f"chunk-{i}".encode() for i in range(num_tasks)]

    # Test with asyncio.TaskGroup
    start = time.perf_counter()
    async with asyncio.TaskGroup() as tg:
        asyncio_tasks = [
            tg.create_task(cpu_intensive_hash(chunk, iterations))
            for chunk in data_chunks
        ]
    asyncio_results = [task.result() for task in asyncio_tasks]
    asyncio_time = time.perf_counter() - start

    # Test with ThreadedTaskGroup
    start = time.perf_counter()
    async with ThreadedTaskGroup(num_threads=4) as tg:
        thrasks_tasks = [
            tg.create_task(cpu_intensive_hash(chunk, iterations))
            for chunk in data_chunks
        ]
    thrasks_results = [await task for task in thrasks_tasks]
    thrasks_time = time.perf_counter() - start

    assert asyncio_results == thrasks_results

    print(f"\n{'=' * 60}")
    print(f"TaskGroup CPU Hashing ({num_tasks} tasks, {iterations} iterations)")
    print(f"{'=' * 60}")
    print(f"asyncio.TaskGroup:      {asyncio_time:.3f}s (baseline)")
    print(f"ThreadedTaskGroup (4):  {thrasks_time:.3f}s ({asyncio_time / thrasks_time:.2f}x)")
    print(f"{'=' * 60}")


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_performance_scaling():
    """Test how performance scales with different thread counts."""
    num_tasks = 32
    iterations = 200
    thread_counts = [1, 2, 4, 8, 16]
    data_chunks = [f"chunk-{i}".encode() for i in range(num_tasks)]

    results = {}

    for num_threads in thread_counts:
        start = time.perf_counter()
        await threaded_gather(
            *[cpu_intensive_hash(chunk, iterations) for chunk in data_chunks],
            num_threads=num_threads,
        )
        elapsed = time.perf_counter() - start
        results[num_threads] = elapsed

    print(f"\n{'=' * 60}")
    print(f"Thread Scaling Performance ({num_tasks} tasks, {iterations} iterations)")
    print(f"{'=' * 60}")
    baseline = results[1]
    for num_threads in thread_counts:
        elapsed = results[num_threads]
        speedup = baseline / elapsed
        print(f"{num_threads:2d} thread(s):  {elapsed:.3f}s  (speedup: {speedup:.2f}x)")
    print(f"{'=' * 60}")
    print("Note: Speedup depends on free-threading being enabled")
    print(f"{'=' * 60}")


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
        num_threads=4,
    )
    thrasks_time = time.perf_counter() - start

    assert asyncio_results == thrasks_results

    print(f"\n{'=' * 60}")
    print(f"Overhead Test - Trivial Tasks ({num_tasks} tasks)")
    print(f"{'=' * 60}")
    print(f"asyncio.gather:         {asyncio_time:.4f}s (baseline)")
    print(f"thrasks (4 threads):    {thrasks_time:.4f}s (overhead: {thrasks_time / asyncio_time:.2f}x)")
    print(f"{'=' * 60}")
    print("Note: thrasks has higher overhead for trivial tasks")
    print("      Use asyncio for simple/fast operations")
    print(f"{'=' * 60}")


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_performance_real_world_scenario():
    """Simulate real-world scenario: API responses with heavy processing."""
    num_requests = 20

    async def simulate_api_request(request_id: int) -> dict[str, Any]:
        """Simulate API request with network delay and CPU processing."""
        # Simulate network I/O
        await asyncio.sleep(0.02)

        # Simulate heavy JSON processing (like parsing large responses)
        data = {
            "id": request_id,
            "payload": "x" * 1000,
            "items": [{"value": i, "squared": i * i} for i in range(100)],
        }

        # Heavy processing
        for _ in range(50):
            serialized = json.dumps(data)
            parsed = json.loads(serialized)
            _ = hashlib.sha256(serialized.encode()).hexdigest()

        return {"request_id": request_id, "status": "processed", "data": data}

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
        num_threads=4,
    )
    thrasks_time = time.perf_counter() - start

    assert len(asyncio_results) == len(thrasks_results) == num_requests

    print(f"\n{'=' * 60}")
    print(f"Real-World Scenario: API + Heavy Processing ({num_requests} requests)")
    print(f"{'=' * 60}")
    print(f"asyncio.gather:         {asyncio_time:.3f}s (baseline)")
    print(f"thrasks (4 threads):    {thrasks_time:.3f}s ({asyncio_time / thrasks_time:.2f}x)")
    print(f"{'=' * 60}")
    print("This simulates: network I/O + JSON processing + hashing")
    print(f"{'=' * 60}")


# === Summary Test ===


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_performance_summary():
    """Run a quick summary of key performance comparisons."""
    print(f"\n{'=' * 70}")
    print("THRASKS PERFORMANCE SUMMARY")
    print(f"{'=' * 70}")

    # Quick CPU test
    num_tasks = 12
    data = [f"test-{i}".encode() for i in range(num_tasks)]

    start = time.perf_counter()
    await asyncio.gather(*[cpu_intensive_hash(d, 300) for d in data])
    asyncio_time = time.perf_counter() - start

    start = time.perf_counter()
    await threaded_gather(*[cpu_intensive_hash(d, 300) for d in data], num_threads=4)
    thrasks_time = time.perf_counter() - start

    print(f"\nCPU-Bound (Hash):  asyncio={asyncio_time:.3f}s  thrasks={thrasks_time:.3f}s  ({asyncio_time/thrasks_time:.2f}x)")

    # Quick I/O test
    start = time.perf_counter()
    await asyncio.gather(*[io_bound_sleep(0.05) for _ in range(num_tasks)])
    asyncio_time = time.perf_counter() - start

    start = time.perf_counter()
    await threaded_gather(*[io_bound_sleep(0.05) for _ in range(num_tasks)], num_threads=4)
    thrasks_time = time.perf_counter() - start

    print(f"I/O-Bound (Sleep): asyncio={asyncio_time:.3f}s  thrasks={thrasks_time:.3f}s  ({asyncio_time/thrasks_time:.2f}x)")

    print(f"\n{'=' * 70}")
    print("RECOMMENDATIONS:")
    print("  • Use thrasks for CPU-intensive async operations (with free-threading)")
    print("  • Use asyncio for pure I/O-bound operations (lower overhead)")
    print("  • Use thrasks for mixed I/O + CPU workloads")
    print(f"{'=' * 70}\n")
