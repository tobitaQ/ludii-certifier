"""
Nim implementation matching nim.gdl (default 3 heaps: 3, 4, 5, normal play).

State: (heap0, ..., heapN-1, current_player)  -- the heap count is the instance's
  current_player: 0 = white, 1 = black
Action: (heap_index, n_to_remove)
"""
import numpy as np
from base import BoardGame, State, Action

NimState = tuple[int, ...]            # (h0, ..., hN-1, player)
NimAction = tuple[int, int]           # (heap_idx, amount)

INITIAL_HEAPS = (3, 4, 5)
_MAX_HEAP = max(INITIAL_HEAPS)
_N_HEAPS = len(INITIAL_HEAPS)


class Nim(BoardGame):
    """`heaps` overrides the opening position: any number of heaps, normal play.

    Nim is solved, so the opponent's strength is not what decides the game: the
    player to move wins with perfect play iff the nim-sum is non-zero. The whole
    of Stage 7 seats the participant first, so the opening position -- not the
    agent -- sets who is theoretically winning, and a second game from the same
    heaps is a rerun of the first. The default is the pipeline's (3, 4, 5)."""

    def __init__(self, heaps: tuple[int, ...] | None = None):
        # read INITIAL_HEAPS at construction, not at import: screen_nim_scale.py
        # scales the game by patching the module constant before building a Nim()
        heaps = INITIAL_HEAPS if heaps is None else heaps
        if not heaps or any(int(h) < 0 for h in heaps):
            raise ValueError(f"Nim needs at least one non-negative heap, got {heaps}")
        self.heaps = tuple(int(h) for h in heaps)
        self.n_heaps = len(self.heaps)
        self.max_heap = max(self.heaps)

    def reset(self) -> NimState:
        return (*self.heaps, 0)

    def legal_actions(self, state: NimState) -> list[NimAction]:
        heaps = state[:-1]              # everything but the player-to-move
        actions = []
        for i, size in enumerate(heaps):
            for n in range(1, size + 1):
                actions.append((i, n))
        return actions

    def step(self, state: NimState, action: NimAction) -> tuple[NimState, float, bool]:
        heaps = list(state[:-1])
        player = state[-1]
        heap_idx, amount = action

        heaps[heap_idx] -= amount
        next_player = 1 - player
        next_state = (*heaps, next_player)

        done = all(h == 0 for h in heaps)
        reward = 1.0 if done else 0.0
        return next_state, reward, done

    def current_player(self, state: NimState) -> int:
        return state[-1]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def win_prob(self, state: NimState) -> float:
        nim_sum = 0
        for h in state[:-1]:
            nim_sum ^= h
        return 0.9 if nim_sum != 0 else 0.1

    def greedy_action(self, state: NimState, legal_actions: list[NimAction]) -> NimAction:
        return max(legal_actions, key=lambda a: (a[1], state[a[0]]))

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------

    # The action encoding is sized from this instance's largest heap, so a custom
    # opening keeps a collision-free index (default (3,4,5) -> 15, as before).
    def obs_vector(self, state: NimState) -> np.ndarray:
        return np.array(
            [h / self.max_heap for h in state[:-1]] + [float(state[-1])],
            dtype=np.float32,
        )

    def action_count(self) -> int:
        return self.n_heaps * self.max_heap  # 15 for the default heaps

    def encode_action(self, action: NimAction) -> int:
        heap_idx, n_take = action
        return heap_idx * self.max_heap + (n_take - 1)

    def decode_action(self, idx: int) -> NimAction:
        return idx // self.max_heap, (idx % self.max_heap) + 1

    def fun_metric_norms(self) -> np.ndarray:
        return np.array([1.0, 0.45, 6.0, 1.0, 5.0])

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_state(self, state: NimState) -> str:
        return "(" + "-".join(str(h) for h in state[:-1]) + ")"

    def format_turn(self, player: int, action: NimAction,
                    state_before: NimState, state_after: NimState) -> str:
        heap_idx, n_take = action
        return (f"  P{player}: 山{heap_idx + 1}から{n_take}本取る "
                f"→ {self.format_state(state_after)}")

    def game_name(self) -> str:
        return "Nim"
