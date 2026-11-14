"""
Stage3: ProfiledMCTSAgent

Simulates a human player profile by adjusting MCTS parameters:
  game_experience : 0=beginner, 1=expert  → iterations
  risk_tolerance  : 0=avoid, 1=seek       → UCB exploration constant c
  planning_depth  : short/long            → iterations (multiplier)
  attention_span  : 0=full, 1=low         → error probability in late game
"""
import random
import math
from agents.mcts_agent import MCTSAgent, _Node
from base import BoardGame, State, Action


class ProfiledMCTSAgent(MCTSAgent):
    def __init__(
        self,
        game: BoardGame,
        game_experience: float = 0.5,   # 0=beginner, 1=expert
        risk_tolerance: float = 0.5,    # 0=avoid risk, 1=seek risk
        planning_depth: str = "medium", # short / medium / long
        attention_span: float = 1.0,    # 1=full attention, 0=distracted
        seed: int | None = None,
    ):
        base_iter = {
            "short":  100,
            "medium": 400,
            "long":  1200,
        }[planning_depth]

        iterations = int(base_iter * (0.3 + 0.7 * game_experience))
        # risk_tolerance → UCB c: high risk = more exploration
        c = 0.5 + 2.0 * risk_tolerance

        super().__init__(game, iterations=iterations, c=c, seed=seed)
        self.attention_span = attention_span
        self._turn_count = 0
        self._rng = random.Random(seed)

    def select_action(self, state: State, legal_actions: list[Action]) -> Action:
        self._turn_count += 1

        # Attention degrades late-game: with prob (1-attention)*turn_factor, pick randomly
        fatigue = (1.0 - self.attention_span) * min(1.0, self._turn_count / 10)
        if self._rng.random() < fatigue:
            return self._rng.choice(legal_actions)

        return super().select_action(state, legal_actions)

    def reset_turn_count(self):
        self._turn_count = 0
