"""
English Checkers on an N×N board (standard 8×8 generalized). Depth = board size:
  6×6 mini (2 rows of men each) / 8×8 standard (3 rows) / 10×10 large (4 rows).
Same rules as `checkers.py` (mandatory capture, multi-jump, king promotion), with
every SIZE-dependent constant parametrized by self.size. The fixed-8×8 `Checkers`
class is left untouched.

Includes the capture_mean game-specific metric (a capture game's analogue to
Othello's flip_mean) so it can be tracked across board sizes.

State: tuple of size*size board ints + current_player + no_capture_count.
Action: (from_pos, to_pos), pos = r*size+c. Encoding: from_pos*(size*size)+to_pos.

`optional_capture=True` is the AGENCY lever (Stage 8): the player may decline a
capture, which is the same rule change `checkers.py` exposes through
CHECKERS_OPTIONAL_CAPTURE for the 8x8 board (there it is a module-level flag read
at import; here it is per-instance so both variants can live in one process).
Measured on 6x6 under random self-play: forced-ply fraction 0.286 -> 0.021 and
the mandatory-capture prune is 0.714 -- i.e. the lever moves the agency axis on
the small board at least as much as it does on 8x8 (0.224 -> 0.006, prune 0.782).
"""
import numpy as np
from base import BoardGame

ALL_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


class CheckersN(BoardGame):
    def __init__(self, size: int = 8, optional_capture: bool = False):
        self.size = size
        self.ncell = size * size
        self.rows_per = size // 2 - 1          # rows of men per player
        self.draw_limit = max(40, size * 10)   # scales with board
        self.optional_capture = optional_capture   # agency lever (see module docstring)

    # --- helpers ------------------------------------------------------------
    def _pos(self, r, c):
        return r * self.size + c

    def _rc(self, pos):
        return divmod(pos, self.size)

    @staticmethod
    def _is_dark(r, c):
        return (r + c) % 2 == 1

    @staticmethod
    def _man_move_dirs(player):
        return [(-1, -1), (-1, 1)] if player == 0 else [(1, -1), (1, 1)]

    def _crown_row(self, player):
        return 0 if player == 0 else self.size - 1

    def _get_jumps(self, board, r, c, player, is_king, captured=frozenset()):
        """Recursively enumerate complete jump sequences from (r,c).
        Returns list of (to_r, to_c, frozenset_of_captured_positions)."""
        n = self.size
        opp_man = 3 if player == 0 else 1
        opp_king = 4 if player == 0 else 2
        results = []
        for dr, dc in ALL_DIRS:
            mid_r, mid_c = r + dr, c + dc
            to_r, to_c = r + 2 * dr, c + 2 * dc
            if not (0 <= to_r < n and 0 <= to_c < n):
                continue
            mid_pos = self._pos(mid_r, mid_c)
            if mid_pos in captured:
                continue
            if board[mid_pos] not in (opp_man, opp_king):
                continue
            if board[self._pos(to_r, to_c)] != 0:
                continue
            new_cap = captured | {mid_pos}
            newly_crowned = (not is_king) and to_r == self._crown_row(player)
            if newly_crowned:
                results.append((to_r, to_c, new_cap))
            else:
                further = self._get_jumps(board, to_r, to_c, player, is_king, new_cap)
                results.extend(further if further else [(to_r, to_c, new_cap)])
        return results

    def _legal_moves(self, board, player):
        """List of (from_pos, to_pos, frozenset_captured). Mandatory capture unless
        self.optional_capture. A (from,to) pair never collides between the two kinds:
        every jump displaces +-2 per hop, a plain move +-1, so parity separates them."""
        my_man = 1 if player == 0 else 3
        my_king = 2 if player == 0 else 4
        captures, non_caps = [], []
        n = self.size
        for r in range(n):
            for c in range(n):
                if not self._is_dark(r, c):
                    continue
                val = board[self._pos(r, c)]
                if val not in (my_man, my_king):
                    continue
                is_king = (val == my_king)
                for to_r, to_c, cap in self._get_jumps(board, r, c, player, is_king):
                    captures.append((self._pos(r, c), self._pos(to_r, to_c), cap))
                move_dirs = ALL_DIRS if is_king else self._man_move_dirs(player)
                for dr, dc in move_dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and board[self._pos(nr, nc)] == 0:
                        non_caps.append((self._pos(r, c), self._pos(nr, nc), frozenset()))
        if self.optional_capture:
            return captures + non_caps          # agency variant: capture is optional
        return captures if captures else non_caps

    def _apply_move(self, board, from_pos, to_pos, captured, player):
        piece = board[from_pos]
        board[to_pos] = piece
        board[from_pos] = 0
        for cp in captured:
            board[cp] = 0
        to_r, _ = self._rc(to_pos)
        if player == 0 and to_r == 0 and piece == 1:
            board[to_pos] = 2
        elif player == 1 and to_r == self.size - 1 and piece == 3:
            board[to_pos] = 4
        return board

    # --- core ---------------------------------------------------------------
    def reset(self):
        n = self.size
        board = [0] * self.ncell
        for r in range(n - self.rows_per, n):          # player 0 (Black), bottom
            for c in range(n):
                if self._is_dark(r, c):
                    board[self._pos(r, c)] = 1
        for r in range(self.rows_per):                 # player 1 (Red), top
            for c in range(n):
                if self._is_dark(r, c):
                    board[self._pos(r, c)] = 3
        return tuple(board) + (0, 0)

    def current_player(self, state):
        return state[self.ncell]

    def legal_actions(self, state):
        board = list(state[:self.ncell])
        moves = self._legal_moves(board, state[self.ncell])
        seen, result = set(), []
        for fp, tp, _ in moves:
            if (fp, tp) not in seen:
                seen.add((fp, tp)); result.append((fp, tp))
        return result

    def step(self, state, action):
        nc_i = self.ncell
        from_pos, to_pos = int(action[0]), int(action[1])
        board = list(state[:nc_i]); player = state[nc_i]; no_cap = state[nc_i + 1]
        moves = self._legal_moves(board, player)
        matching = [(fp, tp, cap) for fp, tp, cap in moves if fp == from_pos and tp == to_pos]
        if not matching:
            if moves:
                from_pos, to_pos, captured = moves[0]
            else:
                return state, 0.0, True
        else:
            from_pos, to_pos, captured = matching[0]
        is_capture = len(captured) > 0
        board = self._apply_move(board, from_pos, to_pos, captured, player)
        new_no_cap = 0 if is_capture else no_cap + 1
        nxt = 1 - player
        if new_no_cap >= self.draw_limit:
            return tuple(board) + (nxt, new_no_cap), 0.0, True
        if not self._legal_moves(board, nxt):
            return tuple(board) + (nxt, new_no_cap), 1.0, True
        return tuple(board) + (nxt, new_no_cap), 0.0, False

    # --- evaluation ---------------------------------------------------------
    def win_prob(self, state):
        board = state[:self.ncell]; player = state[self.ncell]
        my_man = 1 if player == 0 else 3; my_king = 2 if player == 0 else 4
        op_man = 3 if player == 0 else 1; op_king = 4 if player == 0 else 2
        my = sum(1.0 if v == my_man else (1.5 if v == my_king else 0.0) for v in board)
        opp = sum(1.0 if v == op_man else (1.5 if v == op_king else 0.0) for v in board)
        return 0.5 if my + opp < 0.1 else my / (my + opp)

    def greedy_action(self, state, legal_actions):
        board = list(state[:self.ncell]); player = state[self.ncell]
        moves = self._legal_moves(board, player)
        seen = {}
        for fp, tp, cap in moves:
            k = (fp, tp)
            if k not in seen or len(cap) > len(seen[k]):
                seen[k] = cap
        legal_set = set(map(tuple, legal_actions))
        cand = [(k, v) for k, v in seen.items() if k in legal_set]
        if not cand:
            return legal_actions[0]
        max_cap = max(len(v) for _, v in cand)
        adv = lambda m: (-self._rc(m[0][1])[0] if player == 0 else self._rc(m[0][1])[0])
        if max_cap > 0:
            return max([m for m in cand if len(m[1]) == max_cap], key=adv)[0]
        return max(cand, key=adv)[0]

    # --- RL interface -------------------------------------------------------
    def obs_vector(self, state):
        board = state[:self.ncell]; player = state[self.ncell]
        my_man = 1 if player == 0 else 3; my_king = 2 if player == 0 else 4
        op_man = 3 if player == 0 else 1; op_king = 4 if player == 0 else 2
        obs = np.zeros(self.ncell + 1, dtype=np.float32)
        for i, cell in enumerate(board):
            if cell == my_man: obs[i] = 0.5
            elif cell == my_king: obs[i] = 1.0
            elif cell == op_man: obs[i] = -0.5
            elif cell == op_king: obs[i] = -1.0
        obs[-1] = float(player)
        return obs

    def action_count(self):
        return self.ncell * self.ncell

    def encode_action(self, action):
        return int(action[0]) * self.ncell + int(action[1])

    def decode_action(self, idx):
        return (idx // self.ncell, idx % self.ncell)

    def fun_metric_norms(self):
        return np.array([1.0, 0.35, 14.0, 1.0, 12.0])

    # --- game-specific: capture_mean ---------------------------------------
    def _capture_count(self, state_before, action):
        board = list(state_before[:self.ncell]); player = state_before[self.ncell]
        fp, tp = int(action[0]), int(action[1])
        for f, t, cap in self._legal_moves(board, player):
            if f == fp and t == tp:
                return len(cap)
        return 0

    def extra_metric_names(self):
        return ["capture_mean"]

    def extra_metric_norms(self):
        return np.array([0.6])

    def extra_game_metrics(self, turns_data):
        cs = [self._capture_count(sb, a) for sb, a, _ in turns_data]
        return np.array([float(np.mean(cs)) if cs else 0.0])

    def extra_feature_vector(self, logs):
        cs = [self._capture_count(t.state_before, t.chosen_action) for g in logs for t in g.turns]
        return np.array([float(np.mean(cs)) if cs else 0.0])

    def move_reward(self, state_before, action):
        return self._capture_count(state_before, action)

    # --- formatting ---------------------------------------------------------
    def format_state(self, state):
        board = state[:self.ncell]; sym = {0: ".", 1: "b", 2: "B", 3: "r", 4: "R"}; n = self.size
        col = "abcdefghij"[:n]
        rows = ["  " + " ".join(col)]
        for r in range(n):
            rows.append(f"{r} " + " ".join(sym[board[self._pos(r, c)]] for c in range(n)))
        return "\n".join(rows)

    def format_turn(self, player, action, state_before, state_after):
        fr, fc = self._rc(int(action[0])); tr, tc = self._rc(int(action[1]))
        col = "abcdefghij"[:self.size]
        captured = sum(1 for i in range(self.ncell)
                       if state_before[i] != 0 and state_after[i] == 0 and i != int(action[0]))
        cap_str = f" ×{captured}" if captured else ""
        return f"  P{player}({'b/B' if player == 0 else 'r/R'}): {col[fc]}{fr}→{col[tc]}{tr}{cap_str}"

    def game_name(self):
        v = "-optional" if self.optional_capture else ""
        return f"Checkers{self.size}x{self.size}{v}"
