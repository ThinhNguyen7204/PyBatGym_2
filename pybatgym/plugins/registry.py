"""Plugin registry for PyBatGym.

Plugins hook into the environment lifecycle (reset, step, close)
to provide logging, monitoring, and benchmarking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Plugin(ABC):
    """Base class for all PyBatGym plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique plugin name."""

    def on_reset(self, state: dict[str, Any]) -> None:
        """Called after env.reset()."""

    def on_step(
        self,
        action: int,
        reward: float,
        state: dict[str, Any],
        done: bool,
    ) -> None:
        """Called after env.step()."""

    def on_close(self) -> None:
        """Called on env.close()."""


class PluginRegistry:
    """Manages plugin instances for an environment."""

    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}

    def register(self, plugin: Plugin) -> None:
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin '{plugin.name}' already registered")
        self._plugins[plugin.name] = plugin

    def unregister(self, name: str) -> None:
        self._plugins.pop(name, None)

    def get(self, name: str) -> Plugin | None:
        return self._plugins.get(name)

    @property
    def all(self) -> list[Plugin]:
        return list(self._plugins.values())

    def broadcast_reset(self, state: dict[str, Any]) -> None:
        for plugin in self._plugins.values():
            plugin.on_reset(state)

    def broadcast_step(
        self,
        action: int,
        reward: float,
        state: dict[str, Any],
        done: bool,
    ) -> None:
        for plugin in self._plugins.values():
            plugin.on_step(action, reward, state, done)

    def broadcast_close(self) -> None:
        for plugin in self._plugins.values():
            plugin.on_close()
