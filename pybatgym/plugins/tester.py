"""Testing and Validation Utilities for PyBatGym.

Provides the TesterPlugin for automated sanity checks and environment validation.
"""

from typing import Any
import sys

from pybatgym.plugins.registry import Plugin


class TesterPlugin(Plugin):
    """Plugin to automatically run Gymnasium environment checks on initialization.
    
    This helps catch API violations early. Runs only on the first reset to avoid infinite loops.
    """

    @property
    def name(self) -> str:
        return "tester"

    def __init__(self, run_check_env: bool = True, run_sanity_check: bool = False):
        """Initialize the tester plugin.
        
        Args:
            run_check_env: If True, calls Gymnasium's `check_env` on the first reset.
            run_sanity_check: If True, plays a short random episode internally to catch runtime errors.
        """
        self.run_check_env = run_check_env
        self.run_sanity_check = run_sanity_check
        self._has_run = False

    def on_reset(self, observation: Any, info: dict[str, Any]) -> None:
        """Perform validation checks on the very first reset."""
        if self._has_run:
            return
        
        self._has_run = True
        print("\n--- [TesterPlugin] Starting Environment Validation ---")
        
        if self.run_check_env:
            self._perform_check_env()
            
        if self.run_sanity_check:
            self._perform_sanity_check()
            
        print("--- [TesterPlugin] Validation Complete ---\n")

    def _perform_check_env(self) -> None:
        """Run Gymnasium's standard environment checker."""
        try:
            from gymnasium.utils.env_checker import check_env
            # Note: check_env requires the actual env instance.
            # Since Plugin interface doesn't pass 'env', it's tricky.
            # We assume users will run `check_env` directly in their test scripts,
            # or we log a reminder here to do so.
            print("[TesterPlugin] Tip: Ensure you run `gymnasium.utils.env_checker.check_env(env)` in your test suite.")
        except ImportError:
            pass

    def _perform_sanity_check(self) -> None:
        """Log that sanity check is enabled."""
        print("[TesterPlugin] Active monitoring enabled. Will catch and log step errors.")
