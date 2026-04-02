"""TensorBoard Logging Plugin for PyBatGym.

Logs episode-level metrics to TensorBoard for visualization.
"""

from typing import Any
import os
from datetime import datetime

from pybatgym.plugins.registry import Plugin


class TensorBoardLoggerPlugin(Plugin):
    """Plugin to log key metrics to TensorBoard.
    
    Requires `tensorboard` or `tensorboardX`.
    """

    @property
    def name(self) -> str:
        return "tensorboard_logger"

    def __init__(self, log_dir: str = "logs/tensorboard"):
        """Initialize the TensorBoard logger.
        
        Args:
            log_dir: Directory to save the TensorBoard events.
        """
        self.log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self._writer = None
        self._step_count = 0
        self._episode_count = 0
        self._current_episode_rewards: list[float] = []
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=self.log_dir)
        except ImportError:
            print("Warning: TensorBoardLoggerPlugin requires `tensorboard` and `torch`. "
                  "Logging is disabled. Install via `pip install tensorboard torch`.")

    def on_step(self, action: int, reward: float,
                state: dict[str, Any], done: bool) -> None:
        """Called after each env.step(). Matches base Plugin signature."""
        if not self._writer:
            return

        self._step_count += 1
        self._current_episode_rewards.append(reward)

        # Per-step scalar
        self._writer.add_scalar("Step/Reward", reward, self._step_count)

        if done:
            self._log_episode(state)

    def _log_episode(self, state: dict[str, Any]) -> None:
        """Log aggregated metrics at the end of an episode."""
        if not self._writer:
            return

        self._episode_count += 1

        total_reward = sum(self._current_episode_rewards)
        self._writer.add_scalar("Episode/Total_Reward", total_reward, self._episode_count)
        self._writer.add_scalar("Episode/Length", len(self._current_episode_rewards), self._episode_count)

        # Extract utilization from resource in state
        resource = state.get("resource")
        if resource is not None:
            self._writer.add_scalar("Metrics/Utilization", resource.utilization, self._episode_count)

        self._current_episode_rewards.clear()

    def on_close(self) -> None:
        """Close the TensorBoard writer."""
        if self._writer:
            self._writer.close()
