# thrasks

**Threaded async tasks for free-threaded Python 3.14**

`thrasks` is a Python library that brings Kotlin-style coroutine threading to Python's asyncio. It allows you to distribute async tasks across multiple threads, each with its own event loop, enabling true parallel execution of coroutines in free-threaded Python 3.14+.

## Features

- **ThreadedTaskGroup**: An async context manager that distributes tasks across a pool of threads (round-robin)
- **threaded_gather**: A drop-in replacement for `asyncio.gather` that executes coroutines in parallel across threads
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

## Quick Start

### Using ThreadedTaskGroup

Similar to `asyncio.TaskGroup`, but runs tasks across multiple threads:

```python
import asyncio
from thrasks import ThreadedTaskGroup


async def compute_heavy_task(n: int) -> int:
    """Simulate CPU-bound work."""
    total = sum(i * i for i in range(n))
    return total


async def main():
    async with ThreadedTaskGroup(num_threads=4) as tg:
        # Tasks are distributed round-robin across 4 threads
        task1 = tg.create_task(compute_heavy_task(1000000))
        task2 = tg.create_task(compute_heavy_task(2000000))
        task3 = tg.create_task(compute_heavy_task(3000000))
        task4 = tg.create_task(compute_heavy_task(4000000))

    # All tasks completed - retrieve results
    print(f"Results: {await task1}, {await task2}, {await task3}, {await task4}")


asyncio.run(main())
```

### Using threaded_gather

A drop-in replacement for `asyncio.gather` with threading support:

```python
import asyncio
from thrasks import threaded_gather


async def fetch_data(url: str) -> str:
    """Simulate fetching data."""
    await asyncio.sleep(0.1)
    return f"Data from {url}"


async def main():
    # Distribute coroutines across 3 threads
    results = await threaded_gather(
        fetch_data("https://api1.example.com"),
        fetch_data("https://api2.example.com"),
        fetch_data("https://api3.example.com"),
        fetch_data("https://api4.example.com"),
        num_threads=3,
    )

    print(results)


asyncio.run(main())
```

## API Reference

### ThreadedTaskGroup

```python
class ThreadedTaskGroup:
    """Async context manager for managing tasks across multiple threads."""

    def __init__(self, num_threads: int = 4) -> None:
        """
        Initialize the threaded task group.

        Args:
            num_threads: Number of threads to use for running tasks.
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

        Tasks are distributed round-robin across available threads.

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
) -> list[Any]:
    """
    Run awaitables concurrently across multiple threads.

    If passed coroutines, distributes them round-robin across threads.
    If passed tasks, behaves like asyncio.gather.

    Args:
        *aws: Awaitables (coroutines or tasks) to run
        num_threads: Number of threads to use (only for coroutines)
        return_exceptions: If True, exceptions are returned as results

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
import hashlib
from thrasks import threaded_gather


async def hash_data(data: bytes) -> str:
    """CPU-intensive hashing."""
    # With free-threading, this can run in parallel
    for _ in range(10000):
        data = hashlib.sha256(data).digest()
    await asyncio.sleep(0.1)
    return data.hex()


async def main():
    data_chunks = [b"chunk1", b"chunk2", b"chunk3", b"chunk4"]

    # Hash all chunks in parallel across threads
    hashes = await threaded_gather(
        *[hash_data(chunk) for chunk in data_chunks],
        num_threads=4,
    )

    print(f"Computed {len(hashes)} hashes")


asyncio.run(main())
```

### Parallel API Requests with Heavy Processing

```python
import asyncio
import json
from thrasks import ThreadedTaskGroup


async def fetch_and_process(url: str) -> dict:
    """Fetch data and do heavy JSON processing."""
    # Simulate network fetch
    await asyncio.sleep(0.1)
    data = {"large": "json" * 10000}

    # Heavy processing (benefits from threading)
    processed = json.dumps(json.loads(json.dumps(data)))
    return {"url": url, "size": len(processed)}


async def main():
    urls = [f"https://api.example.com/data/{i}" for i in range(20)]

    async with ThreadedTaskGroup(num_threads=5) as tg:
        futures = [tg.create_task(fetch_and_process(url)) for url in urls]

    results = [await f for f in futures]
    print(f"Processed {len(results)} URLs")


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
- CPU-intensive operations in async context (hashing, compression, parsing)
- Mixed I/O and CPU workloads
- When you need TaskGroup-like API across threads
- Free-threaded Python 3.14+ environments

**Not recommended:**
- Pure I/O-bound tasks (use regular asyncio)
- Simple, fast coroutines (overhead not worth it)
- Environments without free-threading support

## Comparison with asyncio

| Feature | asyncio.TaskGroup | ThreadedTaskGroup |
|---------|-------------------|-------------------|
| Thread model | Single thread | Multiple threads |
| Parallelism | Concurrent (I/O) | Parallel (CPU + I/O) |
| GIL impact | Blocked by GIL | Bypassed with free-threading |
| Overhead | Minimal | Thread creation/coordination |
| API compatibility |  |  |

| Feature | asyncio.gather | threaded_gather |
|---------|----------------|-----------------|
| Thread model | Single thread | Multiple threads |
| Parallelism | Concurrent (I/O) | Parallel (CPU + I/O) |
| Task support |  |  (falls back to asyncio.gather) |
| API compatibility |  |  |

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

- **CPU-intensive workloads**: Hashing, JSON processing, calculations
- **I/O-bound workloads**: Sleep operations, network simulation
- **Mixed workloads**: Combined I/O and CPU operations
- **Scaling tests**: Performance across different thread counts
- **Real-world scenarios**: API request processing with heavy computation

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
