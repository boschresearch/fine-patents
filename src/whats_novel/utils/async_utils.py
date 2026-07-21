# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: MIT

import asyncio
import contextlib
import inspect
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
    Iterable,
    Literal,
    TypeVar,
    overload,
)

# Cached per-process event loop for running async callables under ProcessPoolExecutor
_PROCESS_LOOP: asyncio.AbstractEventLoop | None = None


def _get_or_create_process_loop() -> asyncio.AbstractEventLoop:
    global _PROCESS_LOOP
    if _PROCESS_LOOP is None or _PROCESS_LOOP.is_closed():
        _PROCESS_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_PROCESS_LOOP)
    return _PROCESS_LOOP


def _call_func_with_kwargs(func, kwargs):
    """Helper for multiprocessing: runs func(**kwargs) in a subprocess.
    If func is async, reuse a cached per-process event loop instead of
    creating/closing a new one on each invocation."""
    if inspect.iscoroutinefunction(func):
        loop = _get_or_create_process_loop()
        return loop.run_until_complete(func(**kwargs))
    else:
        return func(**kwargs)


async def worker(
    work_queue: asyncio.Queue,
    results_queue: asyncio.Queue,
    func: Callable,
    use_processes: bool,
    executor: ProcessPoolExecutor | None,
    return_exceptions: bool,
    capture_traceback: bool,
):
    """Worker coroutine pulling (index, kwargs) from work_queue and executing func.
    Publishes (index, kwargs, result, error) where error is either None, a string, or an Exception.
    A sentinel of (index, None) signals shutdown."""
    loop = asyncio.get_running_loop()
    while True:
        index, inputs = await work_queue.get()
        if inputs is None:  # sentinel
            work_queue.task_done()
            break
        try:
            if use_processes:
                results = await loop.run_in_executor(
                    executor, _call_func_with_kwargs, func, inputs
                )
            else:
                if inspect.iscoroutinefunction(func):
                    results = await func(**inputs)
                else:
                    results = await asyncio.to_thread(func, **inputs)
            await results_queue.put((index, inputs, results, None))
        except Exception as e:  # pragma: no cover - error path
            if return_exceptions:
                err: Any = e
            else:
                tb = traceback.format_exc() if capture_traceback else ""
                suffix = f"\n{tb}" if tb else ""
                err = f"ERROR/{repr(e)}{suffix}"
            await results_queue.put((index, inputs, None, err))
        finally:
            work_queue.task_done()


async def aenumerate(it: AsyncIterable[dict[str, Any]], start: int = 0):
    idx = start
    async for item in it:
        yield idx, item
        idx += 1


T = TypeVar("T")

AsyncFunc = Callable[..., Awaitable[T]]
SyncFunc = Callable[..., T]


# async function + return_exceptions=True
@overload
def run_async(
    inputs: Iterable[dict[str, Any]] | AsyncIterable[dict[str, Any]],
    func: AsyncFunc[T],
    n_workers: int,
    *,
    use_processes: bool = ...,
    ordered: bool = ...,
    return_exceptions: Literal[True] = ...,
    capture_traceback: bool = ...,
    max_queue_size: int | None = ...,
) -> AsyncGenerator[
    tuple[dict[str, Any], T, None] | tuple[dict[str, Any], None, Exception], None
]: ...


# async function + return_exceptions=False
@overload
def run_async(
    inputs: Iterable[dict[str, Any]] | AsyncIterable[dict[str, Any]],
    func: AsyncFunc[T],
    n_workers: int,
    *,
    use_processes: bool = ...,
    ordered: bool = ...,
    return_exceptions: Literal[False],
    capture_traceback: bool = ...,
    max_queue_size: int | None = ...,
) -> AsyncGenerator[
    tuple[dict[str, Any], T, None] | tuple[dict[str, Any], None, str], None
]: ...


# sync function + return_exceptions=True
@overload
def run_async(
    inputs: Iterable[dict[str, Any]] | AsyncIterable[dict[str, Any]],
    func: SyncFunc[T],
    n_workers: int,
    *,
    use_processes: bool = ...,
    ordered: bool = ...,
    return_exceptions: Literal[True] = ...,
    capture_traceback: bool = ...,
    max_queue_size: int | None = ...,
) -> AsyncGenerator[
    tuple[dict[str, Any], T, None] | tuple[dict[str, Any], None, Exception], None
]: ...


# sync function + return_exceptions=False
@overload
def run_async(
    inputs: Iterable[dict[str, Any]] | AsyncIterable[dict[str, Any]],
    func: SyncFunc[T],
    n_workers: int,
    *,
    use_processes: bool = ...,
    ordered: bool = ...,
    return_exceptions: Literal[False],
    capture_traceback: bool = ...,
    max_queue_size: int | None = ...,
) -> AsyncGenerator[
    tuple[dict[str, Any], T, None] | tuple[dict[str, Any], None, str], None
]: ...


async def run_async(
    inputs: Iterable[dict[str, Any]] | AsyncIterable[dict[str, Any]],
    func: AsyncFunc[T] | SyncFunc[T],
    n_workers: int,
    *,
    use_processes: bool = False,
    ordered: bool = True,
    return_exceptions: bool = True,
    capture_traceback: bool = False,
    max_queue_size: int | None = None,
) -> AsyncGenerator[
    tuple[dict[str, Any], T, None] | tuple[dict[str, Any], None, Exception | str], None
]:
    """Run a function (sync or async) concurrently over an iterable of input dicts.

    Parameters
    ----------
    inputs : Iterable[dict[str, Any]]
        Each dict is expanded as keyword arguments to func.
    func : Callable
        Either an async function or a synchronous function.
    n_workers : int
        Number of concurrent workers (processes or coroutines/threads).
    use_processes : bool, default False
        Whether to use a `ProcessPoolExecutor` (CPU-bound) or asyncio + threads.
    ordered : bool, default True
        If True, yields results in the same order as inputs. If False, yields as soon as completed.
    return_exceptions : bool, default False
        If True, errors are yielded as Exception objects instead of strings.
    capture_traceback : bool, default False
        If True and return_exceptions is False, error strings include formatted traceback.
    max_queue_size : int | None, default None
        If set, bounds the internal work queue to apply backpressure when producing inputs.

    Yields
    ------
    tuple(input_dict, result, error)
        error is None on success, else string or Exception depending on settings.
    """
    executor = None
    if use_processes:
        executor = ProcessPoolExecutor(max_workers=n_workers)

    work_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size or 0)
    results_queue: asyncio.Queue = asyncio.Queue()

    total_items = 0

    async def feeder():
        nonlocal total_items
        if hasattr(inputs, "__aiter__"):
            async for idx, input_dict in aenumerate(inputs):  # type: ignore
                await work_queue.put((idx, input_dict))
                total_items += 1
        else:
            for idx, input_dict in enumerate(inputs):  # type: ignore
                await work_queue.put((idx, input_dict))
                total_items += 1
        for _ in range(n_workers):
            await work_queue.put((-1, None))

    feeder_task = asyncio.create_task(feeder())

    workers = [
        asyncio.create_task(
            worker(
                work_queue,
                results_queue,
                func,
                use_processes,
                executor,
                return_exceptions,
                capture_traceback,
            )
        )
        for _ in range(n_workers)
    ]

    try:
        if ordered:
            buffer: dict[int, tuple[dict[str, Any], Any, Any]] = {}
            next_index = 0
            yielded = 0
            while True:
                if yielded >= total_items and feeder_task.done():
                    break
                index, input_dict, results, error = await results_queue.get()
                if index < 0:
                    continue
                buffer[index] = (input_dict, results, error)
                while next_index in buffer:
                    input_d, res_d, err_d = buffer.pop(next_index)
                    yield input_d, res_d, err_d
                    next_index += 1
                    yielded += 1
        else:
            finished = 0
            while True:
                if finished >= total_items and feeder_task.done():
                    break
                index, input_dict, results, error = await results_queue.get()
                if index < 0:
                    continue
                finished += 1
                yield input_dict, results, error
    finally:
        # Cleanup on normal completion or early consumer break
        if not feeder_task.done():
            feeder_task.cancel()
            with contextlib.suppress(Exception):
                await feeder_task

        # Ensure sentinels exist in case feeder didn’t run to completion
        try:
            pending_sentinels = n_workers
            # Try to insert enough sentinels non-blocking
            while pending_sentinels > 0:
                try:
                    work_queue.put_nowait((-1, None))
                    pending_sentinels -= 1
                except asyncio.QueueFull:
                    # If full, give workers a chance to consume
                    await asyncio.sleep(0)
                    break

        except Exception:
            pass

        # Wait for workers to exit
        await asyncio.gather(*workers, return_exceptions=True)

        if executor:
            # Hard shutdown of process pool
            executor.shutdown(wait=True)


async def run_sequential(
    inputs: Iterable[dict[str, Any]],
    func: Callable[..., Awaitable[T]],
    n_workers: int,
    *,
    use_processes: bool = False,
    ordered: bool = True,
    return_exceptions: bool = True,
    capture_traceback: bool = False,
    max_queue_size: int | None = None,
) -> AsyncGenerator[tuple[dict[str, Any], T | None, Exception | str | None], None]:
    for input in inputs:
        try:
            result = await func(**input)
            yield input, result, None
        except Exception as e:
            if return_exceptions:
                err: Any = e
            else:
                tb = traceback.format_exc() if capture_traceback else ""
                suffix = f"\n{tb}" if tb else ""
                err = f"ERROR/{repr(e)}{suffix}"
            yield input, None, err


if __name__ == "__main__":

    async def async_cpu_bound_task(x: int) -> int:
        result = sum(i * i for i in range(10000))
        return x * 2 + (result % 10)

    def cpu_bound_task(x: int) -> int:
        result = sum(i * i for i in range(10000))
        return x * 2 + (result % 10)

    async def test_process_workers(f, inputs, use_processes, ordered):
        results = []

        async for input_dict, result, error in run_async(
            inputs, f, n_workers=16, use_processes=use_processes, ordered=ordered
        ):
            results.append(result)
            if error:
                print(f"Error processing {input_dict}: {error}")

        return results

    async def main():
        inputs = [{"x": i} for i in range(1000)]
        expected_results = [cpu_bound_task(input_dict["x"]) for input_dict in inputs]

        for f, use_processes, ordered in (
            (cpu_bound_task, True, True),
            (cpu_bound_task, True, False),
            (async_cpu_bound_task, True, True),
            (async_cpu_bound_task, True, False),
            (async_cpu_bound_task, False, True),
            (async_cpu_bound_task, False, False),
        ):
            print(
                f"\n\n{'#'*10} {f.__name__} use_processes={use_processes} ordered={ordered}"
            )
            for _ in range(10):
                results = await test_process_workers(f, inputs, use_processes, ordered)

                if not ordered:
                    if set(results) != set(expected_results):
                        print(
                            f"Unordered results do not match expected results for {f.__name__} with use_processes={use_processes}"
                        )
                else:
                    if results != expected_results:
                        print(
                            f"{results} != {expected_results}. {f.__name__} with use_processes={use_processes} failed correctness check"
                        )
        print("DONE")

        # test invocations to see if type hints work
        async for r in run_async(inputs, cpu_bound_task, n_workers=16):
            break

        async for r in run_async(
            inputs, cpu_bound_task, n_workers=16, return_exceptions=True
        ):
            break

        async for r in run_async(
            inputs, cpu_bound_task, n_workers=16, return_exceptions=False
        ):
            break

        async for r in run_async(inputs, async_cpu_bound_task, n_workers=16):
            break

        async for r in run_async(
            inputs, async_cpu_bound_task, n_workers=16, return_exceptions=True
        ):
            break

        async for r in run_async(
            inputs, async_cpu_bound_task, n_workers=16, return_exceptions=False
        ):
            break

    asyncio.run(main())
