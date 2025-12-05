"""Core implementation of thrasks - threaded task groups and gather."""

import asyncio
import queue
import threading
from collections.abc import Coroutine
from contextvars import Context
from enum import Enum
from typing import Any, TypeVar

__all__ = ["ThreadedTaskGroup", "threaded_gather", "SchedulingMode"]

T = TypeVar("T")


class SchedulingMode(Enum):
    """Scheduling mode for distributing tasks across threads."""

    ROUND_ROBIN = "round_robin"
    QUEUE = "queue"


class _ThreadEventLoop:
    """Manages an event loop running in a dedicated thread."""

    def __init__(
        self,
        *,
        work_queue: queue.Queue[tuple[Coroutine[Any, Any, Any], asyncio.Future[Any], asyncio.AbstractEventLoop | None, str | None, Context | None] | None] | None = None,
    ) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._tasks: list[asyncio.Task[Any]] = []
        self._work_queue = work_queue
        self._queue_consumer_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the event loop in a new thread."""
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self._ready_event.wait()  # Wait until loop is ready

    def _cleanup_tasks(self) -> None:
        """Cancel all tasks and close the loop."""
        if not self.loop:
            return

        tasks = asyncio.all_tasks(self.loop)
        for task in tasks:
            task.cancel()

        # Run loop once more to process cancellations
        if tasks:
            self.loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

        self.loop.close()

    async def _consume_work_queue(self) -> None:
        """Consume tasks from work queue until None is received."""
        while True:
            item = await asyncio.get_event_loop().run_in_executor(
                None, self._work_queue.get
            )
            if item is None:
                break

            coro, future, caller_loop, name, context = item
            self._schedule_task(coro, future, caller_loop, name, context)

    def _run_loop(self) -> None:
        """Run the event loop (executed in thread)."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._ready_event.set()

        # Start queue consumer if we have a work queue
        if self._work_queue is not None:
            self._queue_consumer_task = self.loop.create_task(self._consume_work_queue())

        # Keep the loop running until stop is requested
        try:
            self.loop.run_forever()
        finally:
            self._cleanup_tasks()

    def stop(self) -> None:
        """Stop the event loop and join the thread."""
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)

    def _cancel_tasks_internal(self) -> None:
        """Internal method to cancel tasks (runs in event loop thread)."""
        for task in self._tasks:
            if not task.done():
                task.cancel()

    def cancel_all_tasks(self) -> None:
        """Cancel all tasks running in this thread's event loop."""
        if not self.loop:
            return

        self.loop.call_soon_threadsafe(self._cancel_tasks_internal)

    def _set_future_cancelled(
        self, future: asyncio.Future[Any], caller_loop: asyncio.AbstractEventLoop | None
    ) -> None:
        """Set future as cancelled, handling thread-safety."""
        if future.done():
            return

        if caller_loop:
            caller_loop.call_soon_threadsafe(future.cancel)
        else:
            future.cancel()

    def _set_exception_threadsafe(
        self, future: asyncio.Future[Any], exc: BaseException
    ) -> None:
        """Set exception if future not done (runs in caller's thread)."""
        if not future.done():
            future.set_exception(exc)

    def _set_future_exception(
        self,
        future: asyncio.Future[Any],
        exc: BaseException,
        caller_loop: asyncio.AbstractEventLoop | None,
    ) -> None:
        """Set future exception, handling thread-safety."""
        if future.done():
            return

        if caller_loop:
            caller_loop.call_soon_threadsafe(
                self._set_exception_threadsafe, future, exc
            )
        else:
            future.set_exception(exc)

    def _set_result_threadsafe(
        self, future: asyncio.Future[T], result: T
    ) -> None:
        """Set result if future not done (runs in caller's thread)."""
        if not future.done():
            future.set_result(result)

    def _set_future_result(
        self,
        future: asyncio.Future[T],
        result: T,
        caller_loop: asyncio.AbstractEventLoop | None,
    ) -> None:
        """Set future result, handling thread-safety."""
        if future.done():
            return

        if caller_loop:
            caller_loop.call_soon_threadsafe(
                self._set_result_threadsafe, future, result
            )
        else:
            future.set_result(result)

    def _handle_task_completion(
        self,
        task: asyncio.Task[T],
        future: asyncio.Future[T],
        caller_loop: asyncio.AbstractEventLoop | None,
    ) -> None:
        """Handle task completion by setting the future's result."""
        try:
            if task.cancelled():
                self._set_future_cancelled(future, caller_loop)
                return

            if task.exception() is not None:
                self._set_future_exception(future, task.exception(), caller_loop)
                return

            self._set_future_result(future, task.result(), caller_loop)
        except Exception:
            pass  # Future might already be done or other error

    def _schedule_task(
        self,
        coro: Coroutine[Any, Any, T],
        future: asyncio.Future[T],
        caller_loop: asyncio.AbstractEventLoop | None,
        name: str | None,
        context: Context | None,
    ) -> None:
        """Create and schedule task in this thread's event loop."""
        try:
            task = self.loop.create_task(coro, name=name, context=context)
            self._tasks.append(task)
            task.add_done_callback(
                lambda t: self._handle_task_completion(t, future, caller_loop)
            )
        except Exception as e:
            self._set_future_exception(future, e, caller_loop)

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
            caller_loop = None

        future: asyncio.Future[T] = (
            caller_loop.create_future() if caller_loop else asyncio.Future()
        )

        self.loop.call_soon_threadsafe(
            self._schedule_task, coro, future, caller_loop, name, context
        )
        return future


class ThreadedTaskGroup:
    """
    Async context manager that distributes tasks across multiple threads.

    Similar to asyncio.TaskGroup but runs tasks in a pool of threads,
    each with its own event loop. Supports two scheduling modes:
    - ROUND_ROBIN: Tasks assigned to threads in round-robin fashion (default)
    - QUEUE: Tasks placed in a queue consumed by threads as they finish work

    Example:
        async with ThreadedTaskGroup(num_threads=4) as tg:
            tg.create_task(my_coroutine())
            tg.create_task(another_coroutine())
    """

    def __init__(
        self,
        num_threads: int = 4,
        *,
        mode: SchedulingMode = SchedulingMode.ROUND_ROBIN,
        _cancel_on_error: bool = True,
    ) -> None:
        """
        Initialize the threaded task group.

        Args:
            num_threads: Number of threads to use for running tasks
            mode: Scheduling mode (ROUND_ROBIN or QUEUE)
            _cancel_on_error: Internal parameter to control cancellation behavior
        """
        if num_threads < 1:
            raise ValueError("num_threads must be at least 1")

        self._num_threads = num_threads
        self._mode = mode
        self._threads: list[_ThreadEventLoop] = []
        self._tasks: list[asyncio.Future[Any]] = []
        self._next_thread_idx = 0
        self._entered = False
        self._cancel_on_error = _cancel_on_error
        self._work_queue: queue.Queue[tuple[Coroutine[Any, Any, Any], asyncio.Future[Any], asyncio.AbstractEventLoop | None, str | None, Context | None] | None] | None = None

    async def __aenter__(self) -> "ThreadedTaskGroup":
        """Enter the context manager and start threads."""
        self._entered = True

        # Create work queue if using QUEUE mode
        if self._mode == SchedulingMode.QUEUE:
            self._work_queue = queue.Queue()

        # Start all thread event loops
        for _ in range(self._num_threads):
            thread_loop = _ThreadEventLoop(work_queue=self._work_queue)
            thread_loop.start()
            self._threads.append(thread_loop)

        return self

    async def _collect_task_exceptions(self, done_tasks: set[asyncio.Future[Any]]) -> list[BaseException]:
        """Collect exceptions from completed tasks."""
        exceptions: list[BaseException] = []
        for task in done_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except BaseException as e:
                exceptions.append(e)
        return exceptions

    def _cancel_remaining_tasks(self, pending: set[asyncio.Future[Any]]) -> None:
        """Cancel all pending tasks in thread event loops."""
        for thread in self._threads:
            thread.cancel_all_tasks()
        for task in pending:
            if not task.done():
                task.cancel()

    async def _process_completed_task(self, task: asyncio.Future[Any]) -> BaseException | None:
        """Process a completed task and return its exception if any."""
        try:
            await task
            return None
        except asyncio.CancelledError:
            return None
        except BaseException as e:
            return e

    async def _check_pending_tasks(
        self, pending: set[asyncio.Future[Any]]
    ) -> tuple[set[asyncio.Future[Any]], list[BaseException]]:
        """Check pending tasks for any that completed, collect their exceptions."""
        still_pending = set()
        exceptions: list[BaseException] = []

        for task in pending:
            if task.done():
                exc = await self._process_completed_task(task)
                if exc is not None:
                    exceptions.append(exc)
            else:
                still_pending.add(task)

        return still_pending, exceptions

    def _raise_collected_exceptions(self, exceptions: list[BaseException]) -> None:
        """Raise collected exceptions as single exception or ExceptionGroup."""
        if not exceptions:
            return

        if len(exceptions) == 1:
            raise exceptions[0]
        else:
            raise ExceptionGroup("Multiple task exceptions", exceptions)

    async def _wait_for_tasks_with_cancellation(self) -> None:
        """Wait for all tasks, cancelling remaining if any fails."""
        pending = set(self._tasks)
        exceptions: list[BaseException] = []
        cancellation_triggered = False

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            new_exceptions = await self._collect_task_exceptions(done)
            exceptions.extend(new_exceptions)

            # Cancel remaining tasks if we found exceptions
            if exceptions and not cancellation_triggered:
                pending, more_exceptions = await self._check_pending_tasks(pending)
                exceptions.extend(more_exceptions)
                cancellation_triggered = True
                self._cancel_remaining_tasks(pending)

        self._raise_collected_exceptions(exceptions)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the context manager, wait for tasks, and cleanup threads."""
        try:
            if not self._tasks:
                return

            if self._cancel_on_error:
                await self._wait_for_tasks_with_cancellation()
            else:
                await asyncio.gather(*self._tasks, return_exceptions=True)
        finally:
            # Signal queue consumers to stop
            if self._work_queue is not None:
                for _ in range(self._num_threads):
                    self._work_queue.put(None)

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

        In ROUND_ROBIN mode: tasks distributed round-robin across threads
        In QUEUE mode: tasks placed in queue for threads to consume

        Args:
            coro: The coroutine to run
            name: Optional name for the task
            context: Optional context for the task

        Returns:
            A Future representing the task
        """
        if not self._entered:
            raise RuntimeError("ThreadedTaskGroup must be used in async with statement")

        # Create future in the calling thread's event loop
        try:
            caller_loop = asyncio.get_running_loop()
        except RuntimeError:
            caller_loop = None

        future: asyncio.Future[T] = (
            caller_loop.create_future() if caller_loop else asyncio.Future()
        )
        self._tasks.append(future)

        if self._mode == SchedulingMode.QUEUE:
            # Add to work queue
            self._work_queue.put((coro, future, caller_loop, name, context))
        else:
            # Round-robin scheduling
            thread = self._threads[self._next_thread_idx]
            self._next_thread_idx = (self._next_thread_idx + 1) % self._num_threads
            future = thread.submit_coroutine(coro, name=name, context=context)
            self._tasks[-1] = future

        return future


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

    Example:
        results = await threaded_gather(
            coro1(), coro2(), coro3(),
            num_threads=2,
            mode=SchedulingMode.QUEUE
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
    async with ThreadedTaskGroup(num_threads=num_threads, mode=mode, _cancel_on_error=not return_exceptions) as tg:
        futures = [tg.create_task(coro) for coro in aws]

    # Collect results from futures
    return await asyncio.gather(*futures, return_exceptions=return_exceptions)
