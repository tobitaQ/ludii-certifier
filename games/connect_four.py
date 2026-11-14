"""
Connect Four (6×7 board, normal play: 4 in a row wins).

State: tuple of 43 ints — board (42, row-major) + current_player (1)
  board[r * 7 + c]: 0=empty, 1=player0, 2=player1
  row 0 = top, row 5 = bottom (gravity pulls down)
Action: int — column index 0–6
"""
import math
import numpy as np
from base import BoardGame, State, Action

ROWS = 6
COLS = 7
WIN  = 4


def _idx(r: int, c: int) -> int:
    return r * COLS + c


# ---------------------------------------------------------------------------
# Win check (from the cell where the last piece was placed)
# ---------------------------------------------------------------------------

def _check_win(board: list[int], row: int, col: int, piece: int) -> bool:
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < ROWS and 0 <= c < COLS and board[_idx(r, c)] == piece:
                count += 1
                r += sign * dr
                c += sign * dc
        if count >= WIN:
            return True
    return False


# ---------------------------------------------------------------------------
# Positional evaluation (for win_prob and greedy agent)
# ---------------------------------------------------------------------------

def _score_window(window: list[int], piece: int, opp: int) -> int:
    p, e = window.count(piece), window.count(0)
    if window.count(opp) > 0:
        return 0  # mixed window
    if p == 4: return 100
    if p == 3 and e == 1: return 5
    if p == 2 and e == 2: return 2
    return 0


def _score_board(board: list[int], piece: int) -> int:
    opp = 3 - piece
    score = 0
    # Center column preference
    score += sum(3 for r in range(ROWS) if board[_idx(r, 3)] == piece)
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - WIN + 1):
            w = [board[_idx(r, c + i)] for i in range(WIN)]
            score += _score_window(w, piece, opp)
    # Vertical
    for c in range(COLS):
        for r in range(ROWS - WIN + 1):
            w = [board[_idx(r + i, c)] for i in range(WIN)]
            score += _score_window(w, piece, opp)
    # Diagonal ↘
    for r in range(ROWS - WIN + 1):
        for c in range(COLS - WIN + 1):
            w = [board[_idx(r + i, c + i)] for i in range(WIN)]
            score += _score_window(w, piece, opp)
    # Diagonal ↗
    for r in range(WIN - 1, ROWS):
        for c in range(COLS - WIN + 1):
            w = [board[_idx(r - i, c + i)] for i in range(WIN)]
            score += _score_window(w, piece, opp)
    return score


# ---------------------------------------------------------------------------
# ConnectFour game
# ---------------------------------------------------------------------------

class ConnectFour(BoardGame):
    def reset(self) -> tuple:
        return (0,) * (ROWS * COLS) + (0,)  # all empty, player 0

    def current_player(self, state: tuple) -> int:
        return state[-1]

    def legal_actions(self, state: tuple) -> list[int]:
        board = state[:-1]
        return [c for c in range(COLS) if board[_idx(0, c)] == 0]

    def step(self, state: tuple, action: int) -> tuple[tuple, float, bool]:
        board = list(state[:-1])
        player = state[-1]
        piece = player + 1  # 0→1, 1→2

        # Gravity: find lowest empty row in column
        col = int(action)
        row = next(r for r in range(ROWS - 1, -1, -1) if board[_idx(r, col)] == 0)
        board[_idx(row, col)] = piece

        next_state = tuple(board) + (1 - player,)

        if _check_win(board, row, col, piece):
            return next_state, 1.0, True

        done = not self.legal_actions(next_state)  # draw: board full
        return next_state, 0.0, done

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def win_prob(self, state: tuple) -> float:
        board = list(state[:-1])
        player = state[-1]
        piece = player + 1
        my_score  = _score_board(board, piece)
        opp_score = _score_board(board, 3 - piece)
        return 1.0 / (1.0 + math.exp(-(my_score - opp_score) / 25.0))

    def greedy_action(self, state: tuple, legal_actions: list[int]) -> int:
        board = list(state[:-1])
        player = state[-1]
        piece = player + 1
        opp   = 3 - piece

        # 1. Take winning move
        for col in legal_actions:
            row = next(r for r in range(ROWS - 1, -1, -1) if board[_idx(r, col)] == 0)
            board[_idx(row, col)] = piece
            if _check_win(board, row, col, piece):
                board[_idx(row, col)] = 0
                return col
            board[_idx(row, col)] = 0

        # 2. Block opponent win
        for col in legal_actions:
            row = next(r for r in range(ROWS - 1, -1, -1) if board[_idx(r, col)] == 0)
            board[_idx(row, col)] = opp
            if _check_win(board, row, col, opp):
                board[_idx(row, col)] = 0
                return col
            board[_idx(row, col)] = 0

        # 3. Prefer center columns
        center_order = [3, 2, 4, 1, 5, 0, 6]
        for col in center_order:
            if col in legal_actions:
                return col
        return legal_actions[0]

    def tactical_action(self, state: tuple, legal_actions: list[int]):
        """Immediate winning move, else immediate block, else None. Lets a weak
        MCTS opponent avoid missing an obvious win/block (which reads as a bug)
        without changing its overall strength on non-forcing positions."""
        board = list(state[:-1]); player = state[-1]
        piece = player + 1; opp = 3 - piece
        for p in (piece, opp):                       # take a win first, then block
            for col in legal_actions:
                row = next(r for r in range(ROWS - 1, -1, -1) if board[_idx(r, col)] == 0)
                board[_idx(row, col)] = p
                won = _check_win(board, row, col, p)
                board[_idx(row, col)] = 0
                if won:
                    return col
        return None

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------

    def obs_vector(self, state: tuple) -> np.ndarray:
        board = state[:-1]
        player = state[-1]
        piece = player + 1
        obs = np.zeros(ROWS * COLS + 1, dtype=np.float32)
        for i, cell in enumerate(board):
            if cell == piece:
                obs[i] = 1.0
            elif cell != 0:
                obs[i] = -1.0
        obs[-1] = float(player)
        return obs

    def action_count(self) -> int:
        return COLS  # 7

    def encode_action(self, action: int) -> int:
        return int(action)

    def decode_action(self, idx: int) -> int:
        return int(idx)

    def fun_metric_norms(self) -> np.ndarray:
        # Approximate upper bounds for Connect Four metrics
        return np.array([1.0, 0.35, 10.0, 1.0, 8.0])

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_state(self, state: tuple) -> str:
        board = state[:-1]
        symbols = {0: ".", 1: "X", 2: "O"}
        rows = []
        for r in range(ROWS):
            rows.append(" ".join(symbols[board[_idx(r, c)]] for c in range(COLS)))
        rows.append("0 1 2 3 4 5 6")
        return "\n".join(rows)

    def format_turn(self, player: int, action: int,
                    state_before: tuple, state_after: tuple) -> str:
        symbol = "X" if player == 0 else "O"
        return f"  P{player}({symbol}): {action}列目に置く"

    def game_name(self) -> str:
        return "Connect Four"
