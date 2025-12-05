# thrasks

**Threaded async tasks for free-threaded Python 3.14**

`thrasks` is a Python library that allows you to distribute async tasks across multiple threads, each with its own event loop, enabling true parallel execution of coroutines in free-threaded Python 3.14+.

## Features

- **ThreadedTaskGroup**: An async context manager that distributes tasks across a pool of threads
- **threaded_gather**: A drop-in replacement for `asyncio.gather` that executes coroutines in parallel across threads
- **Two Scheduling Modes**:
  - **ROUND_ROBIN** (default): Tasks assigned to threads in round-robin fashion
  - **QUEUE**: Work-stealing queue where threads pick up tasks as they finish
- **API Compatibility**: Maintains full compatibility with asyncio's TaskGroup and gather APIs
- **Free-threading Ready**: Designed to leverage Python 3.14's free-threaded mode for true parallelism

## Installation

```bash
pip install thrasks
```

Or with uv:

```bash
uv add thrasks
```

## Requirements

- Python 3.14+freethreading or later

## Does it work?

Below you can find the output of performance benchmarks (run on Linux):

```
 uv run pytest tests --log-cli-level=INFO -k "benchmark"
========== test session starts ==========
platform linux -- Python 3.14.0rc2, pytest-8.4.2, pluggy-1.6.0
rootdir: /home/jakub/Documents/programowanie/python/thrasks
configfile: pytest.ini
plugins: asyncio-1.2.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collected 53 items / 42 deselected / 11 selected                                                                                                                                                 

tests/test_performance.py::test_performance_cpu_json 
---------- live log call -----------
INFO     tests.test_performance:test_performance.py:97 ==========
INFO     tests.test_performance:test_performance.py:98 CPU-Intensive JSON Processing (16 tasks, 2000 iterations)
INFO     tests.test_performance:test_performance.py:99 ==========
INFO     tests.test_performance:test_performance.py:100 asyncio.gather:         0.163s (baseline)
INFO     tests.test_performance:test_performance.py:101 thrasks (2 threads):    0.086s (1.88x)
INFO     tests.test_performance:test_performance.py:102 ==========
PASSED                                                                                                                                                                                     [  9%]
tests/test_performance.py::test_performance_fibonacci 
---------- live log call -----------
INFO     tests.test_performance:test_performance.py:134 ==========
INFO     tests.test_performance:test_performance.py:135 CPU-Intensive Fibonacci (16 tasks)
INFO     tests.test_performance:test_performance.py:136 ==========
INFO     tests.test_performance:test_performance.py:137 asyncio.gather:         0.044s (baseline)
INFO     tests.test_performance:test_performance.py:138 thrasks (2 threads):    0.025s (1.77x)
INFO     tests.test_performance:test_performance.py:139 ==========
PASSED                                                                                                                                                                                     [ 18%]
tests/test_performance.py::test_performance_io_bound 
---------- live log call -----------
INFO     tests.test_performance:test_performance.py:171 ==========
INFO     tests.test_performance:test_performance.py:172 I/O-Bound Sleep (20 tasks, 0.05s each)
INFO     tests.test_performance:test_performance.py:173 ==========
INFO     tests.test_performance:test_performance.py:174 asyncio.gather:         0.050s (baseline)
INFO     tests.test_performance:test_performance.py:175 thrasks (2 threads):    0.053s (0.95x)
INFO     tests.test_performance:test_performance.py:176 Note: For pure I/O, asyncio should be similar or faster (less overhead)
INFO     tests.test_performance:test_performance.py:177 ==========
PASSED                                                                                                                                                                                     [ 27%]
tests/test_performance.py::test_performance_thread_locked_sleep 
---------- live log call -----------
INFO     tests.test_performance:test_performance.py:209 ==========
INFO     tests.test_performance:test_performance.py:210 Thread-Locked Sleep (20 tasks, 0.1s each)
INFO     tests.test_performance:test_performance.py:211 ==========
INFO     tests.test_performance:test_performance.py:212 asyncio.gather:         2.004s (runs sequentially!)
INFO     tests.test_performance:test_performance.py:213 thrasks (2 threads):    1.004s (2.00x)
INFO     tests.test_performance:test_performance.py:214 Note: time.sleep() blocks the event loop in asyncio but not in thrasks
INFO     tests.test_performance:test_performance.py:215       thrasks executes blocking operations in parallel threads
INFO     tests.test_performance:test_performance.py:216 ==========
PASSED                                                                                                                                                                                     [ 36%]
tests/test_performance.py::test_performance_mixed_workload 
---------- live log call -----------
INFO     tests.test_performance:test_performance.py:249 ==========
INFO     tests.test_performance:test_performance.py:250 Mixed I/O + CPU Workload (20 tasks)
INFO     tests.test_performance:test_performance.py:251 ==========
INFO     tests.test_performance:test_performance.py:252 asyncio.gather:         0.045s (baseline)
INFO     tests.test_performance:test_performance.py:253 thrasks (2 threads):    0.030s (1.51x)
INFO     tests.test_performance:test_performance.py:254 ==========
PASSED                                                                                                                                                                                     [ 45%]
tests/test_performance.py::test_performance_task_group_cpu 
---------- live log call -----------
INFO     tests.test_performance:test_performance.py:291 ==========
INFO     tests.test_performance:test_performance.py:292 TaskGroup CPU Fibonacci (16 tasks)
INFO     tests.test_performance:test_performance.py:293 ==========
INFO     tests.test_performance:test_performance.py:294 asyncio.TaskGroup:      0.045s (baseline)
INFO     tests.test_performance:test_performance.py:295 ThreadedTaskGroup (2 threads):  0.025s (1.77x)
INFO     tests.test_performance:test_performance.py:296 ==========
PASSED                                                                                                                                                                                     [ 54%]
tests/test_performance.py::test_performance_scaling 
---------- live log call -----------
INFO     tests.test_performance:test_performance.py:321 ==========
INFO     tests.test_performance:test_performance.py:322 Thread Scaling Performance (32 tasks)
INFO     tests.test_performance:test_performance.py:323 ==========
INFO     tests.test_performance:test_performance.py:328  1 thread(s):  0.076s  (speedup: 1.00x)
INFO     tests.test_performance:test_performance.py:328  2 thread(s):  0.039s  (speedup: 1.92x)
INFO     tests.test_performance:test_performance.py:328  4 thread(s):  0.024s  (speedup: 3.21x)
INFO     tests.test_performance:test_performance.py:328  8 thread(s):  0.021s  (speedup: 3.62x)
INFO     tests.test_performance:test_performance.py:328 16 thread(s):  0.022s  (speedup: 3.44x)
INFO     tests.test_performance:test_performance.py:329 ==========
INFO     tests.test_performance:test_performance.py:330 Note: Speedup depends on free-threading being enabled
INFO     tests.test_performance:test_performance.py:331 ==========
PASSED                                                                                                                                                                                     [ 63%]
tests/test_performance.py::test_performance_overhead 
---------- live log call -----------
INFO     tests.test_performance:test_performance.py:369 ==========
INFO     tests.test_performance:test_performance.py:370 Overhead Test - Trivial Tasks (100 tasks)
INFO     tests.test_performance:test_performance.py:371 ==========
INFO     tests.test_performance:test_performance.py:372 asyncio.gather:         0.0002s (baseline)
INFO     tests.test_performance:test_performance.py:373 thrasks (2 threads):    0.0025s (overhead: 10.82x)
INFO     tests.test_performance:test_performance.py:374 ==========
INFO     tests.test_performance:test_performance.py:375 Note: thrasks has higher overhead for trivial tasks
INFO     tests.test_performance:test_performance.py:376       Use asyncio for simple/fast operations
INFO     tests.test_performance:test_performance.py:377 ==========
PASSED                                                                                                                                                                                     [ 72%]
tests/test_performance.py::test_performance_real_world_scenario 
---------- live log call -----------
INFO     tests.test_performance:test_performance.py:430 ==========
INFO     tests.test_performance:test_performance.py:431 Real-World Scenario: API + Heavy Processing (200 requests)
INFO     tests.test_performance:test_performance.py:432 ==========
INFO     tests.test_performance:test_performance.py:433 asyncio.gather:         0.616s (baseline)
INFO     tests.test_performance:test_performance.py:434 thrasks (2 threads):    0.338s (1.82x)
INFO     tests.test_performance:test_performance.py:435 ==========
INFO     tests.test_performance:test_performance.py:436 This simulates: network I/O + JSON processing + Fibonacci calculation
INFO     tests.test_performance:test_performance.py:437 ==========
PASSED                                                                                                                                                                                     [ 81%]
tests/test_performance.py::test_performance_queue_vs_round_robin 
---------- live log call -----------
INFO     tests.test_performance:test_performance.py:487 ==========
INFO     tests.test_performance:test_performance.py:488 Scheduling Mode Comparison: Uneven Workload (30 tasks, 4 threads)
INFO     tests.test_performance:test_performance.py:489 ==========
INFO     tests.test_performance:test_performance.py:490 ROUND_ROBIN mode:       0.013s (baseline)
INFO     tests.test_performance:test_performance.py:491 QUEUE mode:             0.011s (1.20x)
INFO     tests.test_performance:test_performance.py:492 ==========
INFO     tests.test_performance:test_performance.py:493 Note: QUEUE mode should be faster for uneven workloads
INFO     tests.test_performance:test_performance.py:494       as threads pick up new work as soon as they finish
INFO     tests.test_performance:test_performance.py:495 ==========
PASSED                                                                                                                                                                                     [ 90%]
tests/test_performance.py::test_performance_summary 
---------- live log call -----------
INFO     tests.test_performance:test_performance.py:506 ==========
INFO     tests.test_performance:test_performance.py:507 THRASKS PERFORMANCE SUMMARY
INFO     tests.test_performance:test_performance.py:508 ==========
INFO     tests.test_performance:test_performance.py:522 CPU-Bound (Fibonacci):  asyncio=0.012s  thrasks=0.009s  (1.39x)
INFO     tests.test_performance:test_performance.py:533 I/O-Bound (Sleep): asyncio=0.101s  thrasks=0.104s  (0.97x)
INFO     tests.test_performance:test_performance.py:535 ==========
INFO     tests.test_performance:test_performance.py:536 RECOMMENDATIONS:
INFO     tests.test_performance:test_performance.py:537   • Use thrasks for CPU-intensive async operations (with free-threading)
INFO     tests.test_performance:test_performance.py:538   • Use asyncio for pure I/O-bound operations (lower overhead)
INFO     tests.test_performance:test_performance.py:539   • Use thrasks for mixed I/O + CPU workloads
INFO     tests.test_performance:test_performance.py:540 ==========
PASSED                                                                                                                                                                                     [100%]

========== 11 passed, 42 deselected in 4.99s ==========
```

## Quick Start

### Using ThreadedTaskGroup

Similar to `asyncio.TaskGroup`, but runs tasks across multiple threads:

```python
import asyncio
from thrasks import ThreadedTaskGroup, SchedulingMode


async def compute_heavy_task(n: int) -> int:
    """Simulate CPU-bound work."""
    total = sum(i * i for i in range(n))
    return total


async def main():
    # Default: ROUND_ROBIN mode
    async with ThreadedTaskGroup(num_threads=4) as tg:
        # Tasks are distributed round-robin across 4 threads
        task1 = tg.create_task(compute_heavy_task(1000000))
        task2 = tg.create_task(compute_heavy_task(2000000))
        task3 = tg.create_task(compute_heavy_task(3000000))
        task4 = tg.create_task(compute_heavy_task(4000000))

    # All tasks completed - retrieve results
    print(f"Results: {await task1}, {await task2}, {await task3}, {await task4}")

    # QUEUE mode: better for uneven workloads
    async with ThreadedTaskGroup(num_threads=4, mode=SchedulingMode.QUEUE) as tg:
        # Threads pick up tasks as they become available
        tasks = [tg.create_task(compute_heavy_task(i * 1000000)) for i in range(1, 9)]

    results = [await t for t in tasks]
    print(f"Queue mode results: {results}")


asyncio.run(main())
```

### Using threaded_gather

A drop-in replacement for `asyncio.gather` with threading support:

```python
import asyncio
from thrasks import threaded_gather, SchedulingMode


async def fetch_data(url: str) -> str:
    """Simulate fetching data."""
    await asyncio.sleep(0.1)
    return f"Data from {url}"


async def main():
    # Default: ROUND_ROBIN mode
    results = await threaded_gather(
        fetch_data("https://api1.example.com"),
        fetch_data("https://api2.example.com"),
        fetch_data("https://api3.example.com"),
        fetch_data("https://api4.example.com"),
        num_threads=3,
    )

    print(results)

    # QUEUE mode: optimal for variable task durations
    results = await threaded_gather(
        fetch_data("https://slow-api.example.com"),
        fetch_data("https://fast-api.example.com"),
        fetch_data("https://medium-api.example.com"),
        num_threads=2,
        mode=SchedulingMode.QUEUE,
    )

    print(results)


asyncio.run(main())
```

## Scheduling Modes

`thrasks` supports two scheduling strategies for distributing tasks across threads:

### ROUND_ROBIN (Default)

Tasks are assigned to threads in a predictable round-robin fashion. Each new task goes to the next thread in sequence.

**Best for:**
- Predictable workloads where tasks have similar durations
- When you want deterministic task distribution
- Lower overhead (no queue synchronization)

```python
async with ThreadedTaskGroup(num_threads=4, mode=SchedulingMode.ROUND_ROBIN) as tg:
    # Task 1 -> Thread 0, Task 2 -> Thread 1, Task 3 -> Thread 2, Task 4 -> Thread 3
    # Task 5 -> Thread 0, Task 6 -> Thread 1, ...
    for i in range(100):
        tg.create_task(process_item(i))
```

### QUEUE (Work-Stealing)

Tasks are placed in a shared queue. Threads consume tasks as soon as they finish their current work, automatically picking up the next available task.

**Best for:**
- Uneven workloads where task durations vary significantly
- When some tasks are much slower than others
- Dynamic load balancing across threads

```python
async with ThreadedTaskGroup(num_threads=4, mode=SchedulingMode.QUEUE) as tg:
    # Threads automatically pick up tasks as they become available
    # Fast threads will process more tasks than slow threads
    for i in range(100):
        tg.create_task(process_item(i))  # Variable duration tasks
```

**Example with uneven workload:**

```python
import asyncio
from thrasks import ThreadedTaskGroup, SchedulingMode


async def variable_task(task_id: int) -> int:
    """Task with variable duration."""
    if task_id % 10 == 0:
        # Every 10th task is slow
        await asyncio.sleep(0.5)
    else:
        await asyncio.sleep(0.05)
    return task_id


async def main():
    # With ROUND_ROBIN, slow tasks may bottleneck their assigned thread
    async with ThreadedTaskGroup(num_threads=4, mode=SchedulingMode.ROUND_ROBIN) as tg:
        tasks = [tg.create_task(variable_task(i)) for i in range(40)]

    # With QUEUE, threads pick up new work immediately, balancing the load
    async with ThreadedTaskGroup(num_threads=4, mode=SchedulingMode.QUEUE) as tg:
        tasks = [tg.create_task(variable_task(i)) for i in range(40)]


asyncio.run(main())
```

## API Reference

### ThreadedTaskGroup

```python
class ThreadedTaskGroup:
    """Async context manager for managing tasks across multiple threads."""

    def __init__(
        self,
        num_threads: int = 4,
        *,
        mode: SchedulingMode = SchedulingMode.ROUND_ROBIN,
    ) -> None:
        """
        Initialize the threaded task group.

        Args:
            num_threads: Number of threads to use for running tasks.
            mode: Scheduling mode (ROUND_ROBIN or QUEUE).
        """

    def create_task(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        name: str | None = None,
        context: Context | None = None,
    ) -> asyncio.Future[T]:
        """
        Create a task from a coroutine and submit it to a thread.

        In ROUND_ROBIN mode: tasks distributed round-robin across threads.
        In QUEUE mode: tasks placed in queue for threads to consume.

        Args:
            coro: The coroutine to run
            name: Optional name for the task
            context: Optional context for the task

        Returns:
            A Future representing the task
        """
```

**Key behaviors:**
- Automatically awaits all tasks when exiting the context
- Cancels remaining tasks if any task raises an exception
- Raises `ExceptionGroup` if multiple tasks fail
- Compatible with `asyncio.TaskGroup` API

**Example:**

```python
async with ThreadedTaskGroup(num_threads=4) as tg:
    future1 = tg.create_task(my_coroutine(), name="task1")
    future2 = tg.create_task(another_coroutine())
    # Tasks are automatically awaited on context exit

# Retrieve results after context exit
result1 = await future1
result2 = await future2
```

### threaded_gather

```python
async def threaded_gather(
    *aws: Coroutine[Any, Any, Any] | asyncio.Task[Any],
    num_threads: int = 4,
    return_exceptions: bool = False,
    mode: SchedulingMode = SchedulingMode.ROUND_ROBIN,
) -> list[Any]:
    """
    Run awaitables concurrently across multiple threads.

    If passed coroutines, distributes them across threads using specified mode.
    If passed tasks, behaves like asyncio.gather.

    Args:
        *aws: Awaitables (coroutines or tasks) to run
        num_threads: Number of threads to use (only for coroutines)
        return_exceptions: If True, exceptions are returned as results
        mode: Scheduling mode (ROUND_ROBIN or QUEUE)

    Returns:
        List of results from all awaitables
    """
```

**Key behaviors:**
- Maintains order of results (matches input order)
- With `return_exceptions=False` (default): raises first exception
- With `return_exceptions=True`: returns exceptions as part of results
- Falls back to `asyncio.gather` when passed existing tasks
- Compatible with `asyncio.gather` API

**Example:**

```python
# Basic usage
results = await threaded_gather(
    coro1(),
    coro2(),
    coro3(),
    num_threads=2,
)

# With exception handling
results = await threaded_gather(
    safe_coro(),
    might_fail_coro(),
    another_coro(),
    num_threads=3,
    return_exceptions=True,  # Exceptions returned in results
)
```

## Use Cases

### CPU-Bound Async Operations

Perfect for CPU-intensive operations within async code:

```python
import asyncio
import json
from thrasks import threaded_gather


async def process_json(data: dict, iterations: int = 1000) -> int:
    """CPU-intensive JSON serialization/deserialization."""
    result = 0
    for _ in range(iterations):
        serialized = json.dumps(data)
        _ = json.loads(serialized)
        result += len(serialized)
    return result


async def main():
    data_chunks = [
        {"id": i, "data": "x" * 100, "nested": {"values": list(range(50))}}
        for i in range(4)
    ]

    # Process all chunks in parallel across threads
    results = await threaded_gather(
        *[process_json(chunk) for chunk in data_chunks],
        num_threads=4,
    )

    print(f"Processed {len(results)} chunks")


asyncio.run(main())
```

### Parallel API Requests with Heavy Processing

```python
import asyncio
import json
from thrasks import ThreadedTaskGroup


async def fibonacci(n: int) -> int:
    """CPU-intensive calculation."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


async def fetch_and_process(request_id: int) -> dict:
    """Fetch data and do heavy processing."""
    # Simulate network fetch
    await asyncio.sleep(0.1)

    # Heavy CPU work (benefits from threading)
    data = {
        "id": request_id,
        "payload": "x" * 1000,
        "items": [{"value": i, "squared": i * i} for i in range(100)],
    }
    fib_result = await fibonacci(5000)

    # Additional JSON processing
    for _ in range(10):
        serialized = json.dumps(data)
        _ = json.loads(serialized)

    return {"request_id": request_id, "status": "processed", "fib": fib_result}


async def main():
    async with ThreadedTaskGroup(num_threads=4) as tg:
        futures = [tg.create_task(fetch_and_process(i)) for i in range(20)]

    results = [await f for f in futures]
    print(f"Processed {len(results)} requests")


asyncio.run(main())
```

### Mixed Workloads

```python
import asyncio
from thrasks import ThreadedTaskGroup


async def io_bound_task(n: int) -> str:
    """I/O-bound task."""
    await asyncio.sleep(0.1)
    return f"IO-{n}"


async def cpu_bound_task(n: int) -> int:
    """CPU-bound task."""
    result = sum(i * i for i in range(n * 100000))
    return result


async def main():
    # Mix of I/O and CPU-bound tasks across threads
    async with ThreadedTaskGroup(num_threads=4) as tg:
        io_futures = [tg.create_task(io_bound_task(i)) for i in range(10)]
        cpu_futures = [tg.create_task(cpu_bound_task(i)) for i in range(4)]

    io_results = [await f for f in io_futures]
    cpu_results = [await f for f in cpu_futures]

    print(f"IO results: {io_results}")
    print(f"CPU results: {cpu_results}")


asyncio.run(main())
```

## Exception Handling

### ThreadedTaskGroup

By default, if any task raises an exception, remaining tasks are cancelled:

```python
async def failing_task():
    raise ValueError("Something went wrong")


async def normal_task():
    await asyncio.sleep(1)
    return "success"


try:
    async with ThreadedTaskGroup(num_threads=2) as tg:
        tg.create_task(failing_task())
        tg.create_task(normal_task())
except ValueError as e:
    print(f"Task failed: {e}")
```

Multiple exceptions are collected into an `ExceptionGroup`:

```python
try:
    async with ThreadedTaskGroup(num_threads=2) as tg:
        tg.create_task(failing_task_1())
        tg.create_task(failing_task_2())
except ExceptionGroup as eg:
    print(f"Multiple tasks failed: {len(eg.exceptions)} exceptions")
    for exc in eg.exceptions:
        print(f"  - {type(exc).__name__}: {exc}")
```

### threaded_gather

Default behavior (raise first exception):

```python
try:
    results = await threaded_gather(
        safe_coro(),
        failing_coro(),
        num_threads=2,
    )
except ValueError as e:
    print(f"Gather failed: {e}")
```

Return exceptions as results:

```python
results = await threaded_gather(
    safe_coro(),
    failing_coro(),
    another_safe_coro(),
    num_threads=2,
    return_exceptions=True,
)

for i, result in enumerate(results):
    if isinstance(result, Exception):
        print(f"Task {i} failed: {result}")
    else:
        print(f"Task {i} succeeded: {result}")
```

## Performance Considerations

### Free-Threading Mode

For optimal performance with CPU-bound tasks, use Python's free-threaded build:

```bash
# Install free-threaded Python 3.14
python3.14t --version

# Run your script
python3.14t my_script.py
```

### Thread Count Selection

- **I/O-bound tasks**: Use more threads (e.g., 10-50) since threads will mostly wait
- **CPU-bound tasks**: Match thread count to CPU cores (e.g., 4-8)
- **Mixed workloads**: Start with 2x CPU cores and tune based on profiling

```python
import os

# Auto-detect CPU count
num_threads = os.cpu_count() or 4

async with ThreadedTaskGroup(num_threads=num_threads) as tg:
    # Your tasks here
    pass
```

### When to Use thrasks

**Good use cases:**
- CPU-intensive operations in async context (JSON processing, Fibonacci calculations, heavy computation)
- Mixed I/O and CPU workloads (API requests with heavy processing)
- Blocking operations that lock threads (e.g., `time.sleep()` instead of `asyncio.sleep()`)
- When you need TaskGroup-like API across threads
- Free-threaded Python 3.14+ environments

**Not recommended:**
- Pure I/O-bound tasks with `asyncio.sleep()` (use regular asyncio for lower overhead)
- Simple, fast coroutines (overhead not worth it)
- Environments without free-threading support

## Comparison with asyncio

| Feature           | asyncio.TaskGroup | ThreadedTaskGroup            |
|-------------------|-------------------|------------------------------|
| Thread model      | Single thread     | Multiple threads             |
| Parallelism       | Concurrent (I/O)  | Parallel (CPU + I/O)         |
| GIL impact        | Blocked by GIL    | Bypassed with free-threading |
| Overhead          | Minimal           | Thread creation/coordination |
| API compatibility | ✓                 | ✓                            |

| Feature           | asyncio.gather   | threaded_gather                  |
|-------------------|------------------|----------------------------------|
| Thread model      | Single thread    | Multiple threads                 |
| Parallelism       | Concurrent (I/O) | Parallel (CPU + I/O)             |
| Task support      | ✓                | ✓ (falls back to asyncio.gather) |
| API compatibility | ✓                | ✓                                |

## Testing

Run the test suite:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests (excluding performance benchmarks)
pytest -m "not benchmark"

# Run with coverage
pytest -m "not benchmark" --cov=thrasks --cov-report=term-missing

# Run performance benchmarks
pytest -m benchmark -v -s

# Run specific performance test
pytest tests/test_performance.py::test_performance_summary -v -s
```

### Performance Benchmarks

The library includes comprehensive performance tests comparing thrasks with standard asyncio:

- **CPU-intensive workloads**: JSON processing, Fibonacci calculations
- **I/O-bound workloads**: asyncio.sleep operations
- **Blocking workloads**: time.sleep (thread-locking operations)
- **Mixed workloads**: Combined I/O and CPU operations
- **Scaling tests**: Performance across different thread counts
- **Real-world scenarios**: API request processing with heavy computation (network I/O + JSON + Fibonacci)

Run all benchmarks:

```bash
pytest -m benchmark -v -s
```

**Note**: Performance benefits are most visible when running with Python's free-threaded mode (`python3.14t`). Without free-threading, the GIL limits true parallelism for CPU-bound tasks.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Credits

Inspired by Kotlin's coroutine dispatchers and the concept of running coroutines across thread pools.

## Changelog

### 0.1.0 (2025-11-05)

- Initial release
- ThreadedTaskGroup implementation
- threaded_gather implementation
- Full asyncio API compatibility
- Comprehensive test suite
