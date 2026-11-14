"""
Kalah (Mancala variant, standard rules).

Board layout (14 positions):
  0..5  : Player 0's pits (left to right)
  6     : Player 0's store (Kalah)
  7..12 : Player 1's pits (left to right, so pit 7 faces pit 5, pit 12 faces pit 0)
  13    : Player 1's store (Kalah)
  14    : current player

Sowing: counterclockwise (increasing index mod 14), skipping opponent's store.

Rules:
  - Last seed in own store → extra turn
  - Last seed in own empty pit AND opposite pit has seeds → capture both into own store
  - Game ends when one side's pits are all empty; remaining seeds go to opposite player's store
  - Winner: player with more seeds in store

Action: int 0–5 (pit index relative to current player; decoded to absolute index in step)
"""
import os
import numpy as np
from base import BoardGame, State, Action

N_PITS   = 6
N_SEEDS  = 4  # seeds per pit at start (default; override with env KALAH_SEEDS for scale-up)
TOTAL    = N_PITS * N_SEEDS * 2  # 48


def _seeds() -> int:
    """Initial seeds per pit; KALAH_SEEDS env enables Kalah(6,k) with no code change."""
    return int(os.environ.get("KALAH_SEEDS", N_SEEDS))


def _own_store(player: int) -> int:
    return 6 if player == 0 else 13


def _opp_store(player: int) -> int:
    return 13 if player == 0 else 6


def _own_pits(player: int) -> range:
    return range(0, 6) if player == 0 else range(7, 13)


def _abs_pit(action: int, player: int) -> int:
    """Convert relative action (0-5) to absolute board index."""
    return action if player == 0 else action + 7


def _opposite(pit: int) -> int:
    """Opposite pit across the board: always 12 - pit."""
    return 12 - pit


def _sow(board: list[int], pit: int, player: int) -> tuple[list[int], bool]:
    """
    Sow seeds from pit. Returns (new_board, extra_turn).
    Applies capture rule in-place.
    """
    seeds = board[pit]
    board[pit] = 0
    skip = _opp_store(player)
    own_store = _own_store(player)
    own_pits = _own_pits(player)

    pos = pit
    for _ in range(seeds):
        while True:
            pos = (pos + 1) % 14
            if pos != skip:
                break
        board[pos] += 1

    extra_turn = (pos == own_store)

    # Capture: last seed landed in own empty pit (now has exactly 1)
    if pos in own_pits and board[pos] == 1:
        opp_pos = _opposite(pos)
        if board[opp_pos] > 0:
            board[own_store] += 1 + board[opp_pos]
            board[pos] = 0
            board[opp_pos] = 0

    return board, extra_turn


# ---------------------------------------------------------------------------
# Kalah game
# ---------------------------------------------------------------------------

class Kalah(BoardGame):
    def reset(self) -> tuple:
        s = _seeds()
        board = [s] * 6 + [0] + [s] * 6 + [0]
        return tuple(board) + (0,)  # player 0 goes first

    def current_player(self, state: tuple) -> int:
        return state[14]

    def legal_actions(self, state: tuple) -> list[int]:
        board = state[:14]
        player = state[14]
        offset = 0 if player == 0 else 7
        return [i for i in range(N_PITS) if board[i + offset] > 0]

    def step(self, state: tuple, action: int) -> tuple[tuple, float, bool]:
        board = list(state[:14])
        player = state[14]

        abs_pit = _abs_pit(int(action), player)
        board, extra_turn = _sow(board, abs_pit, player)

        # Determine who moves next
        next_player = player if extra_turn else 1 - player
        next_pits = range(0, 6) if next_player == 0 else range(7, 13)

        # Game ends when next player has no moves (Ludii: "no Moves Mover" at turn start)
        # Note: only checking next player, NOT current player — when current player empties
        # their own pits, the opponent may still move (and can sow seeds back).
        if not any(board[i] > 0 for i in next_pits):
            # Sweep remaining seeds to respective stores
            board[6]  += sum(board[0:6])
            board[13] += sum(board[7:13])
            for i in list(range(6)) + list(range(7, 13)):
                board[i] = 0
            my_store = _own_store(player)
            opp_store = _opp_store(player)
            reward = 1.0 if board[my_store] > board[opp_store] else (-1.0 if board[my_store] < board[opp_store] else 0.0)
            return tuple(board) + (next_player,), reward, True

        return tuple(board) + (next_player,), 0.0, False

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def win_prob(self, state: tuple) -> float:
        board = state[:14]
        player = state[14]
        my_store  = board[_own_store(player)]
        opp_store = board[_opp_store(player)]
        # Include pit seeds at half value
        my_pits  = sum(board[i] for i in _own_pits(player))
        opp_pits = sum(board[i] for i in _own_pits(1 - player))
        my_score  = my_store  + 0.5 * my_pits
        opp_score = opp_store + 0.5 * opp_pits
        total = my_score + opp_score
        if total < 1:
            return 0.5
        return my_score / total

    def greedy_action(self, state: tuple, legal_actions: list[int]) -> int:
        board = list(state[:14])
        player = state[14]

        # 1. Prefer action that gives extra turn (last seed in own store)
        own_store = _own_store(player)
        for a in legal_actions:
            abs_pit = _abs_pit(a, player)
            seeds = board[abs_pit]
            skip = _opp_store(player)
            pos = abs_pit
            for _ in range(seeds):
                while True:
                    pos = (pos + 1) % 14
                    if pos != skip:
                        break
            if pos == own_store:
                return a

        # 2. Prefer action that creates a capture
        own_pits = set(_own_pits(player))
        for a in legal_actions:
            abs_pit = _abs_pit(a, player)
            seeds = board[abs_pit]
            skip = _opp_store(player)
            pos = abs_pit
            for _ in range(seeds):
                while True:
                    pos = (pos + 1) % 14
                    if pos != skip:
                        break
            if pos in own_pits and board[pos] == 0:  # would land in empty own pit
                opp_pos = _opposite(pos)
                if board[opp_pos] > 0:
                    return a

        # 3. Pick pit with most seeds
        offset = 0 if player == 0 else 7
        return max(legal_actions, key=lambda a: board[a + offset])

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------

    def obs_vector(self, state: tuple) -> np.ndarray:
        board = state[:14]
        player = state[14]
        obs = np.array(board, dtype=np.float32) / (_seeds() * N_PITS * 2)
        return np.append(obs, float(player))

    def action_count(self) -> int:
        return N_PITS  # 6

    def encode_action(self, action: int) -> int:
        return int(action)

    def decode_action(self, idx: int) -> int:
        return int(idx)

    def fun_metric_norms(self) -> np.ndarray:
        # Kalah: fewer lead changes than Othello, moderate state diversity
        return np.array([1.0, 0.30, 8.0, 1.0, 8.0])

    # ------------------------------------------------------------------
    # Game-specific fun metric (G2): extra-turn rate. Kalah's fun is dominated
    # by landing the last seed in your store to move again ("連続手番" in 97% of
    # Sonnet's reasons). A move grants an extra turn iff the mover stays to move.
    # ------------------------------------------------------------------

    def _grants_extra_turn(self, state_before: tuple, action) -> bool:
        after, _, _ = self.step(state_before, action)
        return self.current_player(after) == self.current_player(state_before)

    def extra_metric_names(self) -> list[str]:
        return ["extra_turn_rate"]

    def extra_metric_norms(self) -> np.ndarray:
        return np.array([1.0])  # already a 0–1 rate

    def extra_game_metrics(self, turns_data) -> np.ndarray:
        if not turns_data:
            return np.array([0.0])
        c = sum(self._grants_extra_turn(sb, a) for sb, a, _ in turns_data)
        return np.array([c / len(turns_data)])

    def extra_feature_vector(self, logs) -> np.ndarray:
        moves = extra = 0
        for gl in logs:
            for t in gl.turns:
                moves += 1
                if self.current_player(t.state_after) == self.current_player(t.state_before):
                    extra += 1
        return np.array([extra / max(moves, 1)])

    def move_reward(self, state_before, action) -> float:
        return 1.0 if self._grants_extra_turn(state_before, action) else 0.0

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_state(self, state: tuple) -> str:
        b = state[:14]
        # Standard Mancala board (viewed from P0's side):
        #   P1 store | P1 pits right-to-left | P0 store
        #              P0 pits left-to-right
        p1_pits = [b[12], b[11], b[10], b[9], b[8], b[7]]
        p0_pits = [b[0], b[1], b[2], b[3], b[4], b[5]]
        p0_store, p1_store = b[6], b[13]
        row1 = "  ".join(f"{v:2d}" for v in p1_pits)
        row2 = "  ".join(f"{v:2d}" for v in p0_pits)
        return (
            f"P1[{p1_store:2d}]  {row1}  [  ]\n"
            f"P0[  ]  {row2}  [{p0_store:2d}]"
        )

    def format_turn(self, player: int, action: int,
                    state_before: tuple, state_after: tuple) -> str:
        abs_pit = _abs_pit(int(action), player)
        seeds = state_before[abs_pit]
        extra = " (連続手番)" if state_after[14] == player else ""
        return f"  P{player}: 穴{action}から{seeds}粒撒く{extra}"

    def game_name(self) -> str:
        return "Kalah"
