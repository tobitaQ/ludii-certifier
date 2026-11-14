"""
Othello on an N×N board (standard 8×8 generalized). Depth = board size N
(4×4 nearly-solved / 6×6 / 8×8 standard). Same rules, flip mechanic, corners.
Includes the flip_mean game-specific metric (Othello's fun driver) so it can be
tracked across board sizes. The fixed-8×8 `Othello` class is left untouched.
"""
import numpy as np
from base import BoardGame

_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class OthelloN(BoardGame):
    def __init__(self, size: int = 8):
        self.size = size

    def _idx(self, r, c):
        return r * self.size + c

    def _flips(self, board, r, c, piece):
        opp = 3 - piece; out = []
        for dr, dc in _DIRS:
            line = []; nr, nc = r + dr, c + dc
            while 0 <= nr < self.size and 0 <= nc < self.size and board[self._idx(nr, nc)] == opp:
                line.append(self._idx(nr, nc)); nr += dr; nc += dc
            if line and 0 <= nr < self.size and 0 <= nc < self.size and board[self._idx(nr, nc)] == piece:
                out.extend(line)
        return out

    def _legal_for(self, board, piece):
        return [self._idx(r, c) for r in range(self.size) for c in range(self.size)
                if board[self._idx(r, c)] == 0 and self._flips(board, r, c, piece)]

    def reset(self):
        n = self.size; m = n // 2
        b = [0] * (n * n)
        b[self._idx(m - 1, m - 1)] = 2; b[self._idx(m, m)] = 2
        b[self._idx(m - 1, m)] = 1;     b[self._idx(m, m - 1)] = 1
        return tuple(b) + (0,)

    def current_player(self, state):
        return state[-1]

    def legal_actions(self, state):
        return self._legal_for(list(state[:-1]), state[-1] + 1)

    def step(self, state, action):
        board = list(state[:-1]); player = state[-1]; piece = player + 1
        a = int(action); board[a] = piece
        for idx in self._flips(board, a // self.size, a % self.size, piece):
            board[idx] = piece
        nxt = 1 - player
        if self._legal_for(board, nxt + 1):
            return tuple(board) + (nxt,), 0.0, False
        if self._legal_for(board, piece):
            return tuple(board) + (player,), 0.0, False
        my, op = board.count(piece), board.count(nxt + 1)
        return tuple(board) + (nxt,), (1.0 if my > op else (-1.0 if my < op else 0.0)), True

    # --- evaluation ---------------------------------------------------------
    def _corners(self):
        n = self.size
        return [self._idx(0, 0), self._idx(0, n - 1), self._idx(n - 1, 0), self._idx(n - 1, n - 1)]

    def win_prob(self, state):
        board = list(state[:-1]); piece = state[-1] + 1; opp = 3 - piece
        my, op = board.count(piece), board.count(opp); tot = my + op
        if tot == 0:
            return 0.5
        ps = (my - op) / tot
        cs = sum(1 if board[i] == piece else (-1 if board[i] == opp else 0) for i in self._corners()) / 4.0
        return (0.6 * ps + 0.4 * cs + 1.0) / 2.0

    def greedy_action(self, state, legal_actions):
        board = list(state[:-1]); piece = state[-1] + 1
        cset = set(self._corners())
        for a in legal_actions:
            if a in cset:
                return a
        n = self.size
        def is_x(i):
            r, c = divmod(i, n)
            return (r in (0, 1, n - 2, n - 1) and c in (0, 1, n - 2, n - 1) and i not in cset)
        safe = [a for a in legal_actions if not is_x(a)] or legal_actions
        return max(safe, key=lambda a: len(self._flips(board, a // n, a % n, piece)))

    # --- RL interface -------------------------------------------------------
    def obs_vector(self, state):
        board = state[:-1]; piece = state[-1] + 1
        obs = np.zeros(self.size * self.size + 1, dtype=np.float32)
        for i, cell in enumerate(board):
            obs[i] = 1.0 if cell == piece else (-1.0 if cell != 0 else 0.0)
        obs[-1] = float(state[-1])
        return obs

    def action_count(self):
        return self.size * self.size

    def encode_action(self, a):
        return int(a)

    def decode_action(self, i):
        return int(i)

    def fun_metric_norms(self):
        return np.array([1.0, 0.35, 12.0, 1.0, 35.0])

    # --- game-specific: flip_mean ------------------------------------------
    def _flip_count(self, state_before, action):
        r, c = divmod(int(action), self.size)
        return len(self._flips(list(state_before[:-1]), r, c, state_before[-1] + 1))

    def extra_metric_names(self):
        return ["flip_mean"]

    def extra_metric_norms(self):
        return np.array([6.0])

    def extra_game_metrics(self, turns_data):
        fl = [self._flip_count(sb, a) for sb, a, _ in turns_data]
        return np.array([float(np.mean(fl)) if fl else 0.0])

    def extra_feature_vector(self, logs):
        fl = [self._flip_count(t.state_before, t.chosen_action) for g in logs for t in g.turns]
        return np.array([float(np.mean(fl)) if fl else 0.0])

    def move_reward(self, state_before, action):
        return self._flip_count(state_before, action) / 6.0

    # --- formatting ---------------------------------------------------------
    def format_state(self, state):
        board = state[:-1]; sym = {0: ".", 1: "X", 2: "O"}; n = self.size
        return "\n".join(" ".join(sym[board[self._idx(r, c)]] for c in range(n)) for r in range(n))

    def format_turn(self, player, action, state_before, state_after):
        r, c = divmod(int(action), self.size)
        flipped = len(self._flips(list(state_before[:-1]), r, c, player + 1))
        return f"  P{player}({'X' if player == 0 else 'O'}): {'abcdefgh'[c]}{r+1} （{flipped}枚返す）"

    def game_name(self):
        return f"Othello{self.size}x{self.size}"
