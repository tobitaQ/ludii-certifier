"""
English Checkers (8×8 board, standard American rules).

Board: 8×8, pieces only on dark squares where (r+c) % 2 == 1
  r=0 = top row, r=7 = bottom row
  Player 0 (Black): starts rows 5,6,7; moves upward (r decreases); promotes at row 0
  Player 1 (Red):   starts rows 0,1,2; moves downward (r increases); promotes at row 7

Cell values:
  0 = empty
  1 = Player 0 man (Black)
  2 = Player 0 king (Black King)
  3 = Player 1 man (Red)
  4 = Player 1 king (Red King)

State: tuple of 66 ints
  state[0..63] : board cells
  state[64]    : current player (0 or 1)
  state[65]    : moves since last capture (draw detection; >= 80 → draw)

Action: (from_pos, to_pos) tuple where pos = r*8+c
Encoding: from_pos * 64 + to_pos  (action_count = 4096)

Rules:
  - Men capture in all 4 diagonal directions (not just forward)
  - Mandatory capture: if any capture exists, must capture
  - Multi-jump: if landing square offers further capture, must continue
  - King promotion: man reaching last row stops and is crowned (cannot continue)
  - Draw: 80 moves without a capture
"""
import os
import numpy as np
from base import BoardGame, State, Action

# Rule variant for the agency manipulation (Stage7): when set, captures are NOT
# mandatory (the player may decline a capture). Read once at import; the GUI sets
# CHECKERS_OPTIONAL_CAPTURE=1 before importing the engine (cf. othello OTHELLO_SIZE).
_OPTIONAL_CAPTURE = os.environ.get("CHECKERS_OPTIONAL_CAPTURE") == "1"

SIZE = 8
ALL_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
DRAW_LIMIT = 80


def _man_move_dirs(player: int) -> list[tuple[int, int]]:
    return [(-1, -1), (-1, 1)] if player == 0 else [(1, -1), (1, 1)]


def _is_dark(r: int, c: int) -> bool:
    return (r + c) % 2 == 1


def _pos(r: int, c: int) -> int:
    return r * SIZE + c


def _rc(pos: int) -> tuple[int, int]:
    return divmod(pos, SIZE)


def _get_jumps(
    board: list[int], r: int, c: int, player: int, is_king: bool,
    captured: frozenset = frozenset(),
) -> list[tuple[int, int, frozenset]]:
    """
    Recursively enumerate all complete jump sequences from (r,c).
    Returns list of (to_r, to_c, frozenset_of_captured_positions).
    """
    opp_man  = 3 if player == 0 else 1
    opp_king = 4 if player == 0 else 2

    results = []
    for dr, dc in ALL_DIRS:
        mid_r, mid_c = r + dr, c + dc
        to_r,  to_c  = r + 2*dr, c + 2*dc

        if not (0 <= to_r < SIZE and 0 <= to_c < SIZE):
            continue
        mid_pos = _pos(mid_r, mid_c)
        if mid_pos in captured:
            continue  # can't jump same piece twice

        mid_val  = board[mid_pos]
        land_val = board[_pos(to_r, to_c)]

        if mid_val not in (opp_man, opp_king):
            continue
        if land_val != 0:
            continue

        new_cap = captured | {mid_pos}

        # Check if piece is newly crowned at landing
        newly_crowned = (not is_king) and (
            (player == 0 and to_r == 0) or (player == 1 and to_r == 7)
        )

        if newly_crowned:
            # Stop here; king promotion ends the jump sequence
            results.append((to_r, to_c, new_cap))
        else:
            further = _get_jumps(board, to_r, to_c, player, is_king, new_cap)
            if further:
                results.extend(further)
            else:
                results.append((to_r, to_c, new_cap))

    return results


def _legal_moves(board: list[int], player: int) -> list[tuple[int, int, frozenset]]:
    """
    Returns list of (from_pos, to_pos, frozenset_captured).
    Mandatory capture: returns only capture moves if any exist.
    """
    my_man  = 1 if player == 0 else 3
    my_king = 2 if player == 0 else 4

    captures = []
    non_caps = []

    for r in range(SIZE):
        for c in range(SIZE):
            if not _is_dark(r, c):
                continue
            val = board[_pos(r, c)]
            if val not in (my_man, my_king):
                continue
            is_king = (val == my_king)

            # Capture moves
            for to_r, to_c, cap in _get_jumps(board, r, c, player, is_king):
                captures.append((_pos(r, c), _pos(to_r, to_c), cap))

            # Non-capture moves
            move_dirs = ALL_DIRS if is_king else _man_move_dirs(player)
            for dr, dc in move_dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < SIZE and 0 <= nc < SIZE and board[_pos(nr, nc)] == 0:
                    non_caps.append((_pos(r, c), _pos(nr, nc), frozenset()))

    if _OPTIONAL_CAPTURE:
        return captures + non_caps          # agency variant: capture is optional
    return captures if captures else non_caps


def _apply_move(
    board: list[int], from_pos: int, to_pos: int,
    captured: frozenset, player: int,
) -> list[int]:
    """Apply a move to board in-place and return board."""
    piece = board[from_pos]
    board[to_pos] = piece
    board[from_pos] = 0
    for cp in captured:
        board[cp] = 0
    # Promotion
    to_r, _ = _rc(to_pos)
    if player == 0 and to_r == 0 and piece == 1:
        board[to_pos] = 2
    elif player == 1 and to_r == 7 and piece == 3:
        board[to_pos] = 4
    return board


# ---------------------------------------------------------------------------
# Checkers game
# ---------------------------------------------------------------------------

class Checkers(BoardGame):
    # MCTSAgent truncates random playouts to this many plies and scores the leaf
    # with heuristic() below. Checkers playouts are long (~60, and ~167 under
    # optional capture), making its MCTS move ~1s natively -- too slow for WASM.
    mcts_rollout_cap = 16

    def heuristic(self, state: tuple) -> float:
        """Cheap leaf eval for a depth-capped rollout: P(player 0 / Black wins) in
        [0,1] from piece material (kings weighted 1.6) and advancement.

        Material alone leaves a sparse endgame flat -- every quiet shuffle scores
        the same -- so advancement at least keeps men walking toward promotion.
        Anti-draw behaviour deliberately does NOT live here: in a zero-sum eval a
        draw is genuinely good for whoever is losing, so a stalling penalty just
        tells the trailing side to run. Wanting a fight regardless is a property of
        the opponent we choose to field, not of the position (web/main.py)."""
        b = state[:SIZE * SIZE]
        p0 = b.count(1) + 1.6 * b.count(2)
        p1 = b.count(3) + 1.6 * b.count(4)
        if p0 + p1 == 0:
            return 0.5
        mat = p0 / (p0 + p1)
        # advancement: player 0 promotes at row 0, player 1 at row SIZE-1
        a0 = sum(SIZE - 1 - (i // SIZE) for i in range(SIZE * SIZE) if b[i] == 1)
        a1 = sum(i // SIZE for i in range(SIZE * SIZE) if b[i] == 3)
        adv = 0.5 if a0 + a1 == 0 else a0 / (a0 + a1)
        # Advancement only earns its weight as the board empties. It exists to stop
        # a sparse endgame from being flat; applied to a full board it gives the
        # opening a gradient to climb, and the opponent marches out in perfect
        # order with no tempo to spare -- correct-looking and no fun to play.
        n = b.count(1) + b.count(2) + b.count(3) + b.count(4)
        w = 0.15 * max(0.0, 1.0 - n / 16)
        return max(0.0, min(1.0, (1 - w) * mat + w * adv))

    def reset(self) -> tuple:
        board = [0] * (SIZE * SIZE)
        # Player 0 (Black): rows 5,6,7
        for r in [5, 6, 7]:
            for c in range(SIZE):
                if _is_dark(r, c):
                    board[_pos(r, c)] = 1
        # Player 1 (Red): rows 0,1,2
        for r in [0, 1, 2]:
            for c in range(SIZE):
                if _is_dark(r, c):
                    board[_pos(r, c)] = 3
        return tuple(board) + (0, 0)  # player 0, no_capture_count=0

    def current_player(self, state: tuple) -> int:
        return state[64]

    def legal_actions(self, state: tuple) -> list[tuple[int, int]]:
        board = list(state[:64])
        player = state[64]
        moves = _legal_moves(board, player)
        # Deduplicate by (from, to); keep first occurrence
        seen: set = set()
        result = []
        for fp, tp, _ in moves:
            key = (fp, tp)
            if key not in seen:
                seen.add(key)
                result.append(key)
        return result

    def step(self, state: tuple, action: tuple[int, int]) -> tuple[tuple, float, bool]:
        from_pos, to_pos = int(action[0]), int(action[1])
        board = list(state[:64])
        player = state[64]
        no_cap = state[65]

        moves = _legal_moves(board, player)
        matching = [(fp, tp, cap) for fp, tp, cap in moves if fp == from_pos and tp == to_pos]

        if not matching:
            # Invalid move: pick first legal move as fallback
            if moves:
                from_pos, to_pos, captured = moves[0]
            else:
                return state, 0.0, True  # no moves (shouldn't happen)
        else:
            from_pos, to_pos, captured = matching[0]

        is_capture = len(captured) > 0
        board = _apply_move(board, from_pos, to_pos, captured, player)

        new_no_cap = 0 if is_capture else no_cap + 1
        next_player = 1 - player

        # Draw: too many moves without capture
        if new_no_cap >= DRAW_LIMIT:
            return tuple(board) + (next_player, new_no_cap), 0.0, True

        # Check if next player has moves
        next_moves = _legal_moves(board, next_player)
        if not next_moves:
            return tuple(board) + (next_player, new_no_cap), 1.0, True  # current player wins

        return tuple(board) + (next_player, new_no_cap), 0.0, False

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def win_prob(self, state: tuple) -> float:
        board = state[:64]
        player = state[64]
        my_man  = 1 if player == 0 else 3
        my_king = 2 if player == 0 else 4
        op_man  = 3 if player == 0 else 1
        op_king = 4 if player == 0 else 2
        my  = sum(1.0 if board[i] == my_man  else (1.5 if board[i] == my_king  else 0.0) for i in range(64))
        opp = sum(1.0 if board[i] == op_man  else (1.5 if board[i] == op_king  else 0.0) for i in range(64))
        total = my + opp
        if total < 0.1:
            return 0.5
        return my / total

    def greedy_action(self, state: tuple, legal_actions: list) -> tuple[int, int]:
        board = list(state[:64])
        player = state[64]
        moves = _legal_moves(board, player)

        # Deduplicate and get capture counts
        seen: dict = {}
        for fp, tp, cap in moves:
            key = (fp, tp)
            if key not in seen or len(cap) > len(seen[key]):
                seen[key] = cap

        # Filter to only legal actions
        legal_set = set(map(tuple, legal_actions))
        candidates = [(k, v) for k, v in seen.items() if k in legal_set]
        if not candidates:
            return legal_actions[0]

        # Prefer more captures
        max_cap = max(len(v) for _, v in candidates)
        capture_moves = [(k, v) for k, v in candidates if len(v) == max_cap]

        if max_cap > 0:
            # Among max-capture moves, prefer advancing toward promotion
            my_man = 1 if player == 0 else 3
            def advance_score(m):
                (fp, tp), cap = m
                to_r, _ = _rc(tp)
                return -to_r if player == 0 else to_r
            return max(capture_moves, key=advance_score)[0]

        # Non-capture: prefer advancing
        def advance_score_nc(m):
            (fp, tp), _ = m
            to_r, _ = _rc(tp)
            return -to_r if player == 0 else to_r
        return max(candidates, key=advance_score_nc)[0]

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------

    def obs_vector(self, state: tuple) -> np.ndarray:
        board = state[:64]
        player = state[64]
        my_man  = 1 if player == 0 else 3
        my_king = 2 if player == 0 else 4
        op_man  = 3 if player == 0 else 1
        op_king = 4 if player == 0 else 2
        obs = np.zeros(65, dtype=np.float32)
        for i, cell in enumerate(board):
            if   cell == my_man:   obs[i] =  0.5
            elif cell == my_king:  obs[i] =  1.0
            elif cell == op_man:   obs[i] = -0.5
            elif cell == op_king:  obs[i] = -1.0
        obs[64] = float(player)
        return obs

    def action_count(self) -> int:
        return SIZE * SIZE * SIZE * SIZE  # 4096

    def encode_action(self, action: tuple[int, int]) -> int:
        return int(action[0]) * 64 + int(action[1])

    def decode_action(self, idx: int) -> tuple[int, int]:
        return (idx // 64, idx % 64)

    def fun_metric_norms(self) -> np.ndarray:
        # Checkers: long games, moderate lead changes, high state diversity
        return np.array([1.0, 0.35, 14.0, 1.0, 12.0])

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_state(self, state: tuple) -> str:
        board = state[:64]
        symbols = {0: ".", 1: "b", 2: "B", 3: "r", 4: "R"}
        rows = ["  a b c d e f g h"]
        for r in range(SIZE):
            row_str = " ".join(symbols[board[_pos(r, c)]] for c in range(SIZE))
            rows.append(f"{r} {row_str}")
        return "\n".join(rows)

    def format_turn(self, player: int, action: tuple[int, int],
                    state_before: tuple, state_after: tuple) -> str:
        from_pos, to_pos = action
        fr, fc = _rc(from_pos)
        tr, tc = _rc(to_pos)
        col = "abcdefgh"
        piece = "b/B" if player == 0 else "r/R"
        captured = sum(1 for i in range(64) if state_before[i] != 0 and state_after[i] == 0 and i != from_pos)
        cap_str = f" ×{captured}" if captured else ""
        return f"  P{player}({piece}): {col[fc]}{fr}→{col[tc]}{tr}{cap_str}"

    def game_name(self) -> str:
        return "Checkers"
