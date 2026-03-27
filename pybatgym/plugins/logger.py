"""CSV Logger plugin for PyBatGym.

Logs step-level metrics to a CSV file for post-episode analysis.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional, TextIO

from pybatgym.plugins.registry import Plugin


class CSVLoggerPlugin(Plugin):
    """Logs each step to a CSV file."""

    COLUMNS = [
        "episode", "step", "action", "reward",
        "sim_time", "pending_jobs", "utilization", "done",
    ]

    def __init__(self, output_dir: str = "logs", prefix: str = "pybatgym") -> None:
        self._output_dir = Path(output_dir)
        self._prefix = prefix
        self._episode = 0
        self._step = 0
        self._file: Optional[TextIO] = None
        self._writer: Optional[csv.writer] = None

    @property
    def name(self) -> str:
        return "csv_logger"

    def on_reset(self, state: dict[str, Any]) -> None:
        self._close_file()
        self._episode += 1
        self._step = 0

        self._output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self._output_dir / f"{self._prefix}_ep{self._episode:04d}.csv"
        self._file = open(filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)

    def on_step(
        self,
        action: int,
        reward: float,
        state: dict[str, Any],
        done: bool,
    ) -> None:
        if self._writer is None:
            return

        self._step += 1
        resource = state.get("resource")
        utilization = resource.utilization if resource else 0.0

        self._writer.writerow([
            self._episode,
            self._step,
            action,
            f"{reward:.6f}",
            f"{state.get('current_time', 0.0):.1f}",
            len(state.get("pending_jobs", [])),
            f"{utilization:.4f}",
            int(done),
        ])

        if done and self._file:
            self._file.flush()

    def on_close(self) -> None:
        self._close_file()

    def _close_file(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()
        self._file = None
        self._writer = None
