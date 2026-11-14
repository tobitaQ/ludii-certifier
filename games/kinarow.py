"""
k-in-a-row on an m×n gravity board (Connect Four generalized).

Depth is tuned purely by the win length k on a fixed 6×7 board:
  k=3 shallow (nearly trivial) / k=4 = standard Connect Four / k=5 deep.
State: tuple of rows*cols ints (row-major, 0=empty/1/2) + current_player.
Action: column index 0..cols-1 (gravity).
"""
import math
import numpy as np
from base import BoardGame


class KInARow(BoardGame):
    def __init__(self, rows: int = 6, cols: int = 7, k: int = 4):
        self.rows, self.cols, self.k = rows, cols, k

    def _idx(self, r, c):
        return r * self.cols + c

    # --- core ---------------------------------------------------------------
    def reset(self):
        return (0,) * (self.rows * self.cols) + (0,)

    def current_player(self, state):
        return state[-1]

    def legal_actions(self, state):
        b = state[:-1]
        return [c for c in range(self.cols) if b[self._idx(0, c)] == 0]

    def _check_win(self, board, row, col, piece):
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.rows and 0 <= c < self.cols and board[self._idx(r, c)] == piece:
                    count += 1; r += sign * dr; c += sign * dc
            if count >= self.k:
                return True
        return False

    def step(self, state, action):
        board = list(state[:-1]); player = state[-1]; piece = player + 1
        col = int(action)
        row = next(r for r in range(self.rows - 1, -1, -1) if board[self._idx(r, col)] == 0)
        board[self._idx(row, col)] = piece
        nxt = tuple(board) + (1 - player,)
        if self._check_win(board, row, col, piece):
            return nxt, 1.0, True
        return nxt, 0.0, not self.legal_actions(nxt)

    # --- evaluation ---------------------------------------------------------
    def _score_window(self, window, piece, opp):
        p, e = window.count(piece), window.count(0)
        if window.count(opp) > 0:
            return 0
        if p == self.k:       return 100
        if p == self.k - 1 and e == 1: return 5
        if p == self.k - 2 and e == 2: return 2
        return 0

    def _score_board(self, board, piece):
        opp = 3 - piece; k = self.k; sc = 0
        cc = self.cols // 2
        sc += sum(3 for r in range(self.rows) if board[self._idx(r, cc)] == piece)
        for r in range(self.rows):
            for c in range(self.cols - k + 1):
                sc += self._score_window([board[self._idx(r, c + i)] for i in range(k)], piece, opp)
        for c in range(self.cols):
            for r in range(self.rows - k + 1):
                sc += self._score_window([board[self._idx(r + i, c)] for i in range(k)], piece, opp)
        for r in range(self.rows - k + 1):
            for c in range(self.cols - k + 1):
                sc += self._score_window([board[self._idx(r + i, c + i)] for i in range(k)], piece, opp)
        for r in range(k - 1, self.rows):
            for c in range(self.cols - k + 1):
                sc += self._score_window([board[self._idx(r - i, c + i)] for i in range(k)], piece, opp)
        return sc

    def win_prob(self, state):
        b = list(state[:-1]); piece = state[-1] + 1
        d = self._score_board(b, piece) - self._score_board(b, 3 - piece)
        return 1.0 / (1.0 + math.exp(-d / 25.0))

    def greedy_action(self, state, legal_actions):
        board = list(state[:-1]); piece = state[-1] + 1; opp = 3 - piece
        for tgt in (piece, opp):  # win, then block
            for col in legal_actions:
                row = next(r for r in range(self.rows - 1, -1, -1) if board[self._idx(r, col)] == 0)
                board[self._idx(row, col)] = tgt
                won = self._check_win(board, row, col, tgt)
                board[self._idx(row, col)] = 0
                if won:
                    return col
        order = sorted(legal_actions, key=lambda c: abs(c - self.cols // 2))
        return order[0]

    # --- RL interface -------------------------------------------------------
    def obs_vector(self, state):
        board = state[:-1]; piece = state[-1] + 1
        obs = np.zeros(self.rows * self.cols + 1, dtype=np.float32)
        for i, cell in enumerate(board):
            obs[i] = 1.0 if cell == piece else (-1.0 if cell != 0 else 0.0)
        obs[-1] = float(state[-1])
        return obs

    def action_count(self):
        return self.cols

    def encode_action(self, action):
        return int(action)

    def decode_action(self, idx):
        return int(idx)

    def fun_metric_norms(self):
        return np.array([1.0, 0.35, 10.0, 1.0, 8.0])

    # --- formatting ---------------------------------------------------------
    def format_state(self, state):
        board = state[:-1]; sym = {0: ".", 1: "X", 2: "O"}
        rows = [" ".join(sym[board[self._idx(r, c)]] for c in range(self.cols)) for r in range(self.rows)]
        rows.append(" ".join(str(c) for c in range(self.cols)))
        return "\n".join(rows)

    def format_turn(self, player, action, state_before, state_after):
        return f"  P{player}({'X' if player == 0 else 'O'}): {int(action)}列目に置く"

    def game_name(self):
        return f"Connect{self.k}"
