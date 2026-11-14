from abc import ABC, abstractmethod
from typing import Any

import numpy as np

State = Any
Action = Any


class BoardGame(ABC):
    @abstractmethod
    def reset(self) -> State:
        ...

    @abstractmethod
    def legal_actions(self, state: State) -> list[Action]:
        ...

    @abstractmethod
    def step(self, state: State, action: Action) -> tuple[State, float, bool]:
        """Returns (next_state, reward, done).
        reward is from the perspective of the player who just moved.
        """
        ...

    @abstractmethod
    def current_player(self, state: State) -> int:
        ...

    # ------------------------------------------------------------------
    # Evaluation helpers (used by metrics.py and RL environments)
    # ------------------------------------------------------------------

    def win_prob(self, state: State) -> float:
        """Win probability [0,1] for the current player. Default: 0.5."""
        return 0.5

    def greedy_action(self, state: State, legal_actions: list[Action]) -> Action:
        """The 'obvious' greedy action. Used by metric_decision. Default: first legal action."""
        return legal_actions[0]

    # ------------------------------------------------------------------
    # RL interface (required for Stage5/6 PPO environments)
    # ------------------------------------------------------------------

    def obs_vector(self, state: State) -> np.ndarray:
        """Flat float32 observation vector for RL environments."""
        raise NotImplementedError(f"{type(self).__name__} must implement obs_vector")

    def obs_dim(self) -> int:
        """Observation vector dimension."""
        return int(self.obs_vector(self.reset()).shape[0])

    def action_count(self) -> int:
        """Total number of discrete actions."""
        raise NotImplementedError(f"{type(self).__name__} must implement action_count")

    def encode_action(self, action: Action) -> int:
        """Encode action to integer index for RL action space."""
        raise NotImplementedError(f"{type(self).__name__} must implement encode_action")

    def decode_action(self, idx: int) -> Action:
        """Decode integer index back to game action."""
        raise NotImplementedError(f"{type(self).__name__} must implement decode_action")

    def fun_metric_norms(self) -> np.ndarray:
        """Per-metric normalization denominators for RL reward computation."""
        return np.array([1.0, 0.45, 6.0, 1.0, 5.0])

    # ------------------------------------------------------------------
    # Game-specific ("extra") fun metrics (G2: the 5 generic metrics miss
    # game-specific sources of fun — e.g. Othello flip magnitude). Opt-in via
    # include_extra so the 5-metric RL reward / weight files stay unchanged.
    # ------------------------------------------------------------------

    def extra_metric_names(self) -> list[str]:
        """Names of game-specific metrics (default: none)."""
        return []

    def extra_metric_norms(self) -> np.ndarray:
        """Normalization denominators for the extra metrics."""
        return np.array([])

    def extra_game_metrics(self, turns_data) -> np.ndarray:
        """Extra metrics for ONE game. turns_data = list of (state_before, action, legal)."""
        return np.array([])

    def extra_feature_vector(self, logs) -> np.ndarray:
        """Extra metrics aggregated over a matchup (list of GameLog)."""
        return np.array([])

    def move_reward(self, state_before, action) -> float:
        """Dense per-move reward for intervention experiments (default 0). A sparse
        terminal metric reward gives PPO little gradient; a per-move signal (e.g.
        Othello flip count) lets the policy actually maximize the metric."""
        return 0.0

    # ------------------------------------------------------------------
    # Formatting (used by Stage4 LLM prompts)
    # ------------------------------------------------------------------

    def format_state(self, state: State) -> str:
        """Human-readable state string."""
        return str(state)

    def format_turn(self, player: int, action: Action,
                    state_before: State, state_after: State) -> str:
        """Human-readable description of one turn."""
        return f"  P{player}: {action}"

    def game_name(self) -> str:
        """Game name for prompts."""
        return type(self).__name__


class Agent(ABC):
    @abstractmethod
    def select_action(self, state: State, legal_actions: list[Action]) -> Action:
        ...
