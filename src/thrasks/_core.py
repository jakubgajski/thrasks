"""Core implementation of thrasks - threaded task groups and gather."""

import asyncio
import threading
from collections.abc import Coroutine
from contextvars import Context
from typing import Any, TypeVar

__all__ = ["ThreadedTaskGroup", "threaded_gather"]

T = TypeVar("T")


class _ThreadEventLoop:
    """Manages an event loop running in a dedicated thread."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._tasks: list[asyncio.Task[Any]] = []

    def start(self) -> None:
        """Start the event loop in a new thread."""
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self._ready_event.wait()  # Wait until loop is ready

    def _run_loop(self) -> None:
        """Run the event loop (executed in thread)."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._ready_event.set()

        # Keep the loop running until stop is requested
        try:
            self.loop.run_forever()
        finally:
            # Clean up all tasks
            try:
                tasks = asyncio.all_tasks(self.loop)
                for task in tasks:
                    task.cancel()
                # Run loop once more to process cancellations
                if tasks:
                    self.loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            finally:
                self.loop.close()

    def stop(self) -> None:
        """Stop the event loop and join the thread."""
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)

    def cancel_all_tasks(self) -> None:
        """Cancel all tasks running in this thread's event loop."""
        if not self.loop:
            return

        def _cancel() -> None:
            for task in self._tasks:
                if not task.done():
                    task.cancel()

        self.loop.call_soon_threadsafe(_cancel)

    def submit_coroutine(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        name: str | None = None,
        context: Context | None = None,
    ) -> asyncio.Future[T]:
        """Submit a coroutine to run in this thread's event loop."""
        if not self.loop:
            raise RuntimeError("Event loop not started")

        # Create future in the calling thread's event loop
        try:
            caller_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create unbound future
            caller_loop = None

        if caller_loop:
            future: asyncio.Future[T] = caller_loop.create_future()
        else:
            future = asyncio.Future()

        def _create_task() -> None:
            try:
                # Create task in the thread's event loop
                task = self.loop.create_task(coro, name=name, context=context)
                self._tasks.append(task)

                # Chain the task result to the future
                def _done_callback(t: asyncio.Task[T]) -> None:
                    try:
                        if t.cancelled():
                            if caller_loop:
                                def _cancel():
                                    if not future.done():
                                        future.cancel()
                                caller_loop.call_soon_threadsafe(_cancel)
                            else:
                                if not future.done():
                                    future.cancel()
                        elif t.exception() is not None:
                            exc = t.exception()
                            if caller_loop:
                                def _set_exc():
                                    if not future.done():
                                        future.set_exception(exc)
                                caller_loop.call_soon_threadsafe(_set_exc)
                            else:
                                if not future.done():
                                    future.set_exception(exc)
                        else:
                            result = t.result()
                            if caller_loop:
                                def _set_result():
                                    if not future.done():
                                        future.set_result(result)
                                caller_loop.call_soon_threadsafe(_set_result)
                            else:
                                if not future.done():
                                    future.set_result(result)
                    except Exception:
                        pass  # Future might already be done or other error

                task.add_done_callback(_done_callback)
            except Exception as e:
                if not future.done():
                    if caller_loop:
                        caller_loop.call_soon_threadsafe(future.set_exception, e)
                    else:
                        future.set_exception(e)

        self.loop.call_soon_threadsafe(_create_task)
        return future


class ThreadedTaskGroup:
    """
    Async context manager that distributes tasks across multiple threads.

    Similar to asyncio.TaskGroup but runs tasks in a pool of threads,
    each with its own event loop. Tasks are submitted round-robin fashion.

    Example:
        async with ThreadedTaskGroup(num_threads=4) as tg:
            tg.create_task(my_coroutine())
            tg.create_task(another_coroutine())
    """

    def __init__(self, num_threads: int = 4, *, _cancel_on_error: bool = True) -> None:
        """
        Initialize the threaded task group.

        Args:
            num_threads: Number of threads to use for running tasks
            _cancel_on_error: Internal parameter to control cancellation behavior
        """
        if num_threads < 1:
            raise ValueError("num_threads must be at least 1")

        self._num_threads = num_threads
        self._threads: list[_ThreadEventLoop] = []
        self._tasks: list[asyncio.Future[Any]] = []
        self._next_thread_idx = 0
        self._entered = False
        self._cancel_on_error = _cancel_on_error

    async def __aenter__(self) -> "ThreadedTaskGroup":
        """Enter the context manager and start threads."""
        self._entered = True

        # Start all thread event loops
        for _ in range(self._num_threads):
            thread_loop = _ThreadEventLoop()
            thread_loop.start()
            self._threads.append(thread_loop)

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the context manager, wait for tasks, and cleanup threads."""
        try:
            # Wait for all tasks to complete
            if self._tasks:
                if self._cancel_on_error:
                    # Cancel remaining tasks if any task fails (TaskGroup behavior)
                    pending = set(self._tasks)
                    exceptions: list[BaseException] = []
                    cancellation_triggered = False

                    while pending:
                        # Wait for at least one task to complete
                        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                        # Collect exceptions from completed tasks
                        for task in done:
                            try:
                                await task
                            except asyncio.CancelledError:
                                # Ignore cancellations from our own cancellation
                                pass
                            except BaseException as e:
                                exceptions.append(e)

                        # If we found exceptions and haven't cancelled yet, cancel remaining tasks
                        if exceptions and not cancellation_triggered:
                            # Before cancelling, check if any pending tasks are also done
                            # This handles the race condition where multiple tasks fail simultaneously
                            still_pending = set()
                            for task in pending:
                                if task.done():
                                    # Task completed, collect its result
                                    try:
                                        await task
                                    except asyncio.CancelledError:
                                        pass
                                    except BaseException as e:
                                        exceptions.append(e)
                                else:
                                    still_pending.add(task)

                            pending = still_pending
                            cancellation_triggered = True

                            # Cancel tasks in thread event loops
                            for thread in self._threads:
                                thread.cancel_all_tasks()
                            # Cancel pending futures
                            for task in pending:
                                if not task.done():
                                    task.cancel()

                    # Raise exceptions
                    if exceptions:
                        if len(exceptions) == 1:
                            raise exceptions[0]
                        else:
                            raise ExceptionGroup("Multiple task exceptions", exceptions)
                else:
                    # Don't cancel on error - just wait for all tasks
                    # Don't raise exceptions in this mode; let caller handle them
                    await asyncio.gather(*self._tasks, return_exceptions=True)
        finally:
            # Stop all thread event loops
            for thread in self._threads:
                thread.stop()

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
        if not self._entered:
            raise RuntimeError("ThreadedTaskGroup must be used in async with statement")

        # Select next thread (round-robin)
        thread = self._threads[self._next_thread_idx]
        self._next_thread_idx = (self._next_thread_idx + 1) % self._num_threads

        # Submit coroutine to the selected thread
        future = thread.submit_coroutine(coro, name=name, context=context)
        self._tasks.append(future)

        return future


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

    Example:
        results = await threaded_gather(
            coro1(), coro2(), coro3(),
            num_threads=2
        )
    """
    if not aws:
        return []

    # Check if all awaitables are tasks
    all_tasks = all(isinstance(aw, asyncio.Task) for aw in aws)

    if all_tasks:
        # All are tasks - use regular gather
        return await asyncio.gather(*aws, return_exceptions=return_exceptions)

    # At least some are coroutines - use threaded approach
    all_coroutines = all(asyncio.iscoroutine(aw) for aw in aws)

    if not all_coroutines and not all_tasks:
        # Mixed - need to handle carefully
        # Convert tasks to awaitables that we can gather
        async def _wrap_task(task: asyncio.Task[Any]) -> Any:
            return await task

        # For mixed case, just use regular gather
        return await asyncio.gather(*aws, return_exceptions=return_exceptions)

    # All are coroutines - use threaded task group
    # Use ThreadedTaskGroup which handles all the waiting and exception collection
    # When return_exceptions=True, don't cancel tasks on error
    async with ThreadedTaskGroup(num_threads=num_threads, _cancel_on_error=not return_exceptions) as tg:
        futures = [tg.create_task(coro) for coro in aws]

    # Collect results from futures
    return await asyncio.gather(*futures, return_exceptions=return_exceptions)
