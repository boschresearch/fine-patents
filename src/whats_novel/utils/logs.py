import asyncio
import copy
import logging
import re
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any, Optional, TextIO

import pandas as pd
import rich.logging
from rich.console import Console
from rich.table import Table


class TerminalOnlyConsole(Console):
    """Console that adds a tag to prevent logging to file and uses dynamic sys.stdout."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.only_terminal_active = False
        self.not_counting_lines = False
        self.other_lines_printed = 0

    def _render_buffer(self, *args, **kwargs) -> str:
        rendered = super()._render_buffer(*args, **kwargs)
        if not self.not_counting_lines:
            self.other_lines_printed += rendered.count("\n")
        return (
            TeeStream.NOT_TO_FILE_TAG if self.only_terminal_active else ""
        ) + rendered

    def only_terminal(self):
        """Context manager to output only to terminal, not to file."""

        class OnlyTerminalContextManager:
            def __init__(self, console: TerminalOnlyConsole):
                self.console = console

            def __enter__(self):
                self.console.only_terminal_active = True

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.console.only_terminal_active = False

        return OnlyTerminalContextManager(self)

    def dont_count_lines(self):
        """Context manager to prevent counting lines printed to console."""

        class DontCountLinesContextManager:
            def __init__(self, console: TerminalOnlyConsole):
                self.console = console

            def __enter__(self):
                self.console.not_counting_lines = True

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.console.not_counting_lines = False

        return DontCountLinesContextManager(self)


console = TerminalOnlyConsole()


def get_logger(name: str = __name__) -> logging.Logger:
    """Get a rich logger with the specified name."""
    logging.basicConfig(
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich.logging.RichHandler(console=console, rich_tracebacks=True)],
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


class TeeStream:
    """Stream wrapper that writes to both original stream and file."""

    NOT_TO_FILE_TAG = "<<<NOT_TO_FILE>>>"
    hyperlink_pattern = re.compile(r"\033\]8;.*?;.*?\033\\(.*?)\033]8;;\033\\")

    def __init__(self, original_stream: TextIO, file_handle: TextIO) -> None:
        self.original_stream = original_stream
        self.file_handle = file_handle

    def write(self, data: str) -> int:
        if self.NOT_TO_FILE_TAG in data:
            data = data.replace(self.NOT_TO_FILE_TAG, "")
            file = False
        else:
            file = True

        self.original_stream.write(data)
        if file:
            self.file_handle.write(self.remove_hyperlinks(data))
        return len(data)

    def flush(self) -> None:
        self.original_stream.flush()
        self.file_handle.flush()

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.original_stream, attr)

    def remove_hyperlinks(self, text: str) -> str:
        return self.hyperlink_pattern.sub(r"\1", text)


class OutputRedirector:
    """Context manager that redirects stdout, stderr, and rich.Console to both console and file.

    Example:
        with OutputRedirector('output.log'):
            print("Goes to both console and file")
            Console().print("[bold]Rich output[/bold] also captured")
    """

    def __init__(
        self,
        log_file: str | Path,
        mode: str = "w",
        encoding: str = "utf-8",
        redirect_stdout: bool = True,
        redirect_stderr: bool = True,
    ) -> None:
        self.log_file = Path(log_file)
        self.mode = mode
        self.encoding = encoding
        self.redirect_stdout = redirect_stdout
        self.redirect_stderr = redirect_stderr
        self.file_handle: Optional[TextIO] = None
        self.original_stdout: Optional[TextIO] = None
        self.original_stderr: Optional[TextIO] = None

    def __enter__(self) -> "OutputRedirector":
        self.file_handle = open(self.log_file, self.mode, encoding=self.encoding)  # type: ignore
        if self.redirect_stdout:
            self.original_stdout = sys.stdout
            sys.stdout = TeeStream(self.original_stdout, self.file_handle)  # type: ignore
        if self.redirect_stderr:
            self.original_stderr = sys.stderr
            sys.stderr = TeeStream(self.original_stderr, self.file_handle)  # type: ignore
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.original_stdout:
            sys.stdout = self.original_stdout
        if self.original_stderr:
            sys.stderr = self.original_stderr
        if self.file_handle:
            self.file_handle.close()


def print_dataframe_rich(
    df: pd.DataFrame,
    title: Optional[str] = None,
    max_rows: Optional[int] = None,
    max_cols: Optional[int] = None,
    show_lines: bool = True,
    show_header: bool = True,
    show_row_number: bool = True,
    decimal_places: int = 3,
    use_full_width: bool = True,
) -> None:
    """
    Print a pandas DataFrame using rich tables for better formatting.

    Args:
        df: The pandas DataFrame to display
        title: Optional title for the table
        max_rows: Maximum number of rows to display (None for all)
        max_cols: Maximum number of columns to display (None for all)
        show_lines: Whether to show grid lines
        show_header: Whether to show column headers
        show_row_number: Whether to show row numbers
        decimal_places: Number of decimal places to round floats to
        use_full_width: Whether to use the full terminal width
    """

    console = Console()
    table = Table(
        title=title,
        show_lines=show_lines,
        show_header=show_header,
        expand=use_full_width,
    )

    display_df = df.copy()
    if max_rows is not None:
        display_df = display_df.head(max_rows)
    if max_cols is not None:
        display_df = display_df.iloc[:, :max_cols]

    if show_row_number:
        table.add_column("Row", style="dim", width=6)

    for col in display_df.columns:
        table.add_column(str(col), overflow="fold")

    for idx, row in display_df.iterrows():
        row_data = []
        if show_row_number:
            row_data.append(str(idx))

        for value in row:
            if pd.isna(value):
                row_data.append("[dim]NaN[/dim]")
            elif isinstance(value, float):
                row_data.append(f"{value:.{decimal_places}f}")
            else:
                row_data.append(str(value))

        table.add_row(*row_data)

    console.print(table)

    if max_rows is not None and len(df) > max_rows:
        console.print(f"[dim]... showing {max_rows} of {len(df)} rows[/dim]")
    if max_cols is not None and len(df.columns) > max_cols:
        console.print(f"[dim]... showing {max_cols} of {len(df.columns)} columns[/dim]")


class RichTableProgress:
    """Progress tracker that displays statistics in a rich table format."""

    def __init__(
        self,
        total: int | None = None,
        print_every: int = 1,
        dynamic: bool = True,
        persistent_every: int | None = None,
    ) -> None:
        self.total = total
        self.print_every = print_every
        self.current_count = 0
        self.current_count_empty = 0
        self.start_time = pd.Timestamp.now()
        self.console = console
        self.dynamic = dynamic
        self.persistent_every = persistent_every
        self.table = None
        self._closed = False
        self.semaphore = asyncio.Semaphore(1)
        self._last_line_count = 0
        self._dynamic_started = False

    def update(
        self,
        increment: int = 1,
        increment_empty: int = 0,
        data: dict | None = None,
        sort: bool = False,
        add_defaults: bool = True,
        default_group_name: str = "default",
    ) -> None:
        """Update progress counter and display statistics."""
        self.current_count += increment
        self.current_count_empty += increment_empty
        data = copy.deepcopy(data or {})
        elapsed_seconds = (pd.Timestamp.now() - self.start_time).total_seconds()

        for key, value in data.items():
            if isinstance(value, RichTableProgress.AvgPerSec):
                data[key] = value.resolve(data, elapsed_seconds)

        if sort:
            data = dict(sorted(data.items()))
        if add_defaults:
            data = {
                **self._build_default_stats(elapsed_seconds, default_group_name),
                **data,
            }

        if increment == 0 or self.current_count % self.print_every == 0:
            self.table = self._build_table(self._group_by_prefix(data))
            self._render_table()
            if (
                self.persistent_every
                and self.current_count > 0
                and self.current_count % self.persistent_every == 0
            ):
                self.print()
            if self.total is not None and self.current_count >= self.total:
                self.close()

    async def async_update(self, *args, **kwargs) -> None:
        """Asynchronous version of update method."""
        async with self.semaphore:
            self.update(*args, **kwargs)

    def _build_default_stats(self, elapsed_seconds: float, group_name: str) -> dict:
        """Build default statistics dictionary."""
        now = pd.Timestamp.now()
        non_empty_count = self.current_count - self.current_count_empty
        stats = {
            f"{group_name}/now": now.strftime("%Y-%m-%d %H:%M:%S"),
            f"{group_name}/progress": (
                f"{self.current_count} / {self.total} ({self.current_count / self.total * 100:.1f}%)"
                if self.total
                else str(self.current_count)
            ),
            f"{group_name}/time": (
                self.format_time(elapsed_seconds)
                if self.total
                else f"{elapsed_seconds:.2f} sec"
            ),
            f"{group_name}/speed": f"{(non_empty_count) / elapsed_seconds:.2f} it/s",
        }
        if self.total and non_empty_count > 0:
            total_seconds = (elapsed_seconds / non_empty_count) * self.total
            stats[f"{group_name}/total_time"] = self.format_time(total_seconds)
            stats[f"{group_name}/remaining_time"] = self.format_time(
                total_seconds - elapsed_seconds
            )
        elif self.total:
            stats[f"{group_name}/total_time"] = stats[
                f"{group_name}/remaining_time"
            ] = "N/A"
        return stats

    @staticmethod
    def _group_by_prefix(data: dict) -> dict[str, list[tuple[str, str]]]:
        """Group data by prefix (before '/' separator)."""
        groups: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
        for key, value in data.items():
            prefix, suffix = key.split("/", 1) if "/" in key else ("", key)
            groups[prefix].append((suffix, value))
        return dict(groups)

    def _build_table(self, grouped_data: dict[str, list[tuple[str, str]]]) -> Table:
        """Build a rich Table from grouped data."""
        table = Table()
        if len(grouped_data) > 1:
            table.add_column("Group")
            table.add_column("Key")
            table.add_column("Value")
            for group_name, rows in grouped_data.items():
                middle_index = len(rows) // 2
                for row_index, (key, value) in enumerate(rows):
                    table.add_row(
                        group_name if row_index == middle_index else "",
                        str(key),
                        str(value),
                    )
                table.add_section()
        else:
            table.add_column("Key")
            table.add_column("Value")
            for rows in grouped_data.values():
                for key, value in rows:
                    table.add_row(str(key), str(value))
        return table

    def print(self) -> None:
        """Print the current table to console."""
        if self.table:
            self.console.print(self.table)

    def _render_table(self) -> None:
        """Render the table either dynamically or statically."""
        if not self.dynamic or not self.table:
            return

        with self.console.only_terminal():
            # Check if other lines were printed since last render
            other_lines = self.console.other_lines_printed
            self.console.other_lines_printed = 0  # Reset the counter

            # Move cursor up and clear previous table if this isn't the first render
            if self._dynamic_started and self._last_line_count > 0:
                # Move cursor up by the number of lines in the previous table plus any intermediate prints
                total_lines_up = self._last_line_count + other_lines
                sys.stdout.write(f"\033[{total_lines_up}A")
                # Clear only the lines occupied by the last table
                sys.stdout.write(f"\033[{self._last_line_count}M")  # Clear current line
                # Move cursor back to where we started (after the cleared table lines)
                if other_lines > 0:
                    sys.stdout.write(f"\033[{other_lines}B")

            # Render the new table to see how many lines it takes
            buffer = StringIO()
            temp_console = Console(file=buffer, force_terminal=True)
            temp_console.print(self.table)
            output = buffer.getvalue()
            # Count lines for next update
            self._last_line_count = output.count("\n")
            self._dynamic_started = True

            # Print the table
            with self.console.dont_count_lines():
                self.console.print(self.table)
                self.console.file.flush()

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds into a human-readable time string."""
        return (
            f"{seconds * 1000:.0f}ms"
            if seconds < 1
            else f"{seconds:.1f}s"
            if seconds < 60
            else f"{seconds // 60:.0f}min {seconds % 60:.0f}s"
            if seconds < 3600
            else f"{seconds // 3600:.0f}h {seconds % 3600 // 60:.0f}min"
            if seconds < 86400
            else f"{seconds // 86400:.0f}days {seconds % 86400 // 3600:.0f}h"
        )

    class AvgPerSec:
        """Helper class to compute average per second of a metric."""

        def __init__(self, key: str, unit: str) -> None:
            self.key, self.unit = key, unit

        def resolve(self, data: dict, elapsed_time: float) -> str:
            """Compute average per second from total value."""
            return (
                f"{data[self.key] / elapsed_time:.2f} {self.unit}"
                if self.key in data
                else "N/A"
            )

    def close(self) -> None:
        """Close the dynamic display if active."""
        if self.dynamic and self._dynamic_started and not self._closed:
            # Print a final newline to move past the dynamic display
            with self.console.only_terminal():
                sys.stdout.write("\n")
                sys.stdout.flush()
            self._closed = True

    def __enter__(self) -> "RichTableProgress":
        self.update(0)
        return self

    async def __aenter__(self) -> "RichTableProgress":
        self.update(0)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.print()
        self.close()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.print()
        self.close()


if __name__ == "__main__":
    import tempfile
    import time

    print("\n=== Testing OutputRedirector with RichTableProgress ===")

    # Create a temporary log file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False) as tmpfile:
        log_path = tmpfile.name

    print(f"Logging to: {log_path}")

    # Use OutputRedirector to capture both console output and progress to file
    with OutputRedirector(log_path, mode="w"):
        print("Starting data processing pipeline...")
        console.print(
            "[bold cyan]Processing 10 items with progress tracking[/bold cyan]"
        )

        # Use RichTableProgress inside the redirected context
        with RichTableProgress(
            total=101, print_every=1, persistent_every=10
        ) as progress:
            for i in range(101):
                progress.update(
                    increment=1,
                    data={
                        "processing/items": i + 1,
                        "processing/errors": 0 if i < 8 else 1,
                        "stats/tokens_processed": (i + 1) * 1000,
                        "stats/tokens_per_sec": RichTableProgress.AvgPerSec(
                            "stats/tokens_processed", "tokens/s"
                        ),
                    },
                    sort=True,
                )
                console.print(f"Processed item {i + 1}")
                time.sleep(0.1)

        console.print("[bold green]✓ Processing complete![/bold green]")

    # Read and display the log file
    print("\n=== Log file contents ===")
    with open(log_path, "r") as f:
        print(f.read())
