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

    def on_step(self, action: int, observation: Any, reward: float, 
                terminated: bool, truncated: bool, info: dict[str, Any]) -> None:
        """Handle step event to accumulate rewards."""
        if not self._writer:
            return
            
        self._step_count += 1
        self._current_episode_rewards.append(reward)
        
        if terminated or truncated:
            self._log_episode(info)

    def _log_episode(self, info: dict[str, Any]) -> None:
        """Log aggregated metrics at the end of an episode."""
        if not self._writer:
            return
            
        self._episode_count += 1
        
        # Log aggregated rewards
        total_reward = sum(self._current_episode_rewards)
        self._writer.add_scalar("Episode/Total_Reward", total_reward, self._episode_count)
        
        # Log metrics from the environment's info dict
        if "utilization" in info:
            self._writer.add_scalar("Metrics/Utilization", info["utilization"], self._episode_count)
        if "throughput" in info:
            self._writer.add_scalar("Metrics/Throughput", info["throughput"], self._episode_count)
            
        if "avg_waiting_time" in info:
            self._writer.add_scalar("Metrics/Average_Waiting_Time", info["avg_waiting_time"], self._episode_count)
        if "avg_bounded_slowdown" in info:
            self._writer.add_scalar("Metrics/Average_Slowdown", info["avg_bounded_slowdown"], self._episode_count)

        self._current_episode_rewards.clear()

    def on_close(self) -> None:
        """Close the TensorBoard writer."""
        if self._writer:
            self._writer.close()
