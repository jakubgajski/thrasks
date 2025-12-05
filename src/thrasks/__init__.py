"""
thrasks - Threaded async tasks for free-threaded Python 3.14.

Provides tools to run asyncio coroutines across multiple threads,
inspired by Kotlin's coroutine dispatchers.
"""

from thrasks._core import SchedulingMode, ThreadedTaskGroup, threaded_gather

__version__ = "0.1.0"
__all__ = ["ThreadedTaskGroup", "threaded_gather", "SchedulingMode"]
