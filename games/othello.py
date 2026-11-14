"""
Othello/Reversi (8×8 board, standard rules).

State: tuple of 65 ints — board (64, row-major) + current_player (1)
  board[r*8+c]: 0=empty, 1=player0(Black), 2=player1(White)
  Initial: center 4 pieces placed (d4=W, e4=B, d5=B, e5=W)
Action: int — cell index r*8+c (0–63); only legal cells are valid

Pass: when the current player has no legal moves, they pass automatically.
End:  neither player has legal moves OR board is full.
Win:  player with more pieces.
"""
import os
import numpy as np
from base import BoardGame, State, Action

# Board side; OTHELLO_SIZE env enables a smaller, beginner-friendlier board (e.g. 6x6).
# Read at import: set the env BEFORE importing this module / eval_improvement.
SIZE = int(os.environ.get("OTHELLO_SIZE", 8))
DIRS = [(-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),           ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)]


def _idx(r: int, c: int) -> int:
    return r * SIZE + c


def _flips(board: list[int], r: int, c: int, piece: int) -> list[int]:
    """Return list of cell indices that would flip if piece is placed at (r, c)."""
    opp = 3 - piece
    result = []
    for dr, dc in DIRS:
        line = []
        nr, nc = r + dr, c + dc
        while 0 <= nr < SIZE and 0 <= nc < SIZE and board[_idx(nr, nc)] == opp:
            line.append(_idx(nr, nc))
            nr += dr
            nc += dc
        if line and 0 <= nr < SIZE and 0 <= nc < SIZE and board[_idx(nr, nc)] == piece:
            result.extend(line)
    return result


def _legal_for(board: list[int], piece: int) -> list[int]:
    """Return list of cell indices that are legal moves for piece."""
    moves = []
    for r in range(SIZE):
        for c in range(SIZE):
            if board[_idx(r, c)] == 0 and _flips(board, r, c, piece):
                moves.append(_idx(r, c))
    return moves


def _count(board: list[int], piece: int) -> int:
    return board.count(piece)


# ---------------------------------------------------------------------------
# Othello game
# ---------------------------------------------------------------------------

class Othello(BoardGame):
    # MCTSAgent truncates random playouts to this many plies and scores the leaf
    # with heuristic() below -- Othello's full rollouts (~60 plies) are the reason
    # its MCTS move is ~1.8s natively (too slow for the WASM/web build).
    mcts_rollout_cap = 16

    def heuristic(self, state: tuple) -> float:
        """Cheap leaf eval for a depth-capped rollout: P(player 0 / Black wins) in
        [0,1] from disc material, corner control, and the squares that give corners
        away.

        The X-square term is what stops the capped-rollout opponent from handing
        over corners: a player who sits on the diagonal neighbour of an *empty*
        corner is usually about to lose that corner, and material alone cannot see
        it within 16 plies. Corners are also worth more than material this early --
        disc count famously anti-correlates with winning until the endgame."""
        b = state[:SIZE * SIZE]
        n1, n2 = b.count(1), b.count(2)
        last = SIZE - 1
        # (corner, its diagonal X-square, its two edge C-squares)
        risky = (
            (_idx(0, 0), _idx(1, 1), _idx(0, 1), _idx(1, 0)),
            (_idx(0, last), _idx(1, last - 1), _idx(0, last - 1), _idx(1, last)),
            (_idx(last, 0), _idx(last - 1, 1), _idx(last, 1), _idx(last - 1, 0)),
            (_idx(last, last), _idx(last - 1, last - 1), _idx(last, last - 1),
             _idx(last - 1, last)),
        )
        c1 = sum(1 for c, _, _, _ in risky if b[c] == 1)
        c2 = sum(1 for c, _, _, _ in risky if b[c] == 2)
        x1 = x2 = 0.0
        for corner, xsq, c_a, c_b in risky:
            if b[corner]:
                continue                       # corner settled: its guards no longer matter
            for sq, w in ((xsq, 1.0), (c_a, 0.5), (c_b, 0.5)):
                if b[sq] == 1:
                    x1 += w
                elif b[sq] == 2:
                    x2 += w
        mat = 0.5 if n1 + n2 == 0 else n1 / (n1 + n2)
        cor = 0.5 if c1 + c2 == 0 else c1 / (c1 + c2)
        risk = 0.5 if x1 + x2 == 0 else x2 / (x1 + x2)     # high = the OPPONENT is exposed
        return max(0.0, min(1.0, 0.35 * mat + 0.45 * cor + 0.20 * risk))

    def reset(self) -> tuple:
        board = [0] * (SIZE * SIZE)
        m = SIZE // 2  # center 4 (works for any even SIZE)
        board[_idx(m - 1, m - 1)] = 2  # White
        board[_idx(m - 1, m)] = 1      # Black
        board[_idx(m, m - 1)] = 1      # Black
        board[_idx(m, m)] = 2          # White
        return tuple(board) + (0,)  # player 0 (Black) goes first

    def current_player(self, state: tuple) -> int:
        return state[-1]

    def legal_actions(self, state: tuple) -> list[int]:
        board = list(state[:-1])
        player = state[-1]
        piece = player + 1
        return _legal_for(board, piece)

    def step(self, state: tuple, action: int) -> tuple[tuple, float, bool]:
        board = list(state[:-1])
        player = state[-1]
        piece = player + 1

        # Place piece and flip
        board[int(action)] = piece
        for idx in _flips(board, action // SIZE, action % SIZE, piece):
            board[idx] = piece

        next_player = 1 - player
        next_piece = next_player + 1

        # Check if next player can move
        if _legal_for(board, next_piece):
            next_state = tuple(board) + (next_player,)
            return next_state, 0.0, False

        # Next player must pass — check if current player can move again
        if _legal_for(board, piece):
            next_state = tuple(board) + (player,)
            return next_state, 0.0, False

        # Neither player can move → game over
        next_state = tuple(board) + (next_player,)
        my_count  = _count(board, piece)
        opp_count = _count(board, next_piece)
        if my_count > opp_count:
            reward = 1.0
        elif my_count < opp_count:
            reward = -1.0
        else:
            reward = 0.0
        return next_state, reward, True

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def win_prob(self, state: tuple) -> float:
        """Heuristic: piece count + corner bonus."""
        board = list(state[:-1])
        player = state[-1]
        piece = player + 1
        opp   = 3 - piece

        my  = _count(board, piece)
        opp_cnt = _count(board, opp)
        total = my + opp_cnt
        if total == 0:
            return 0.5
        piece_score = (my - opp_cnt) / total  # -1..1

        # Corner bonus
        corners = [_idx(0,0), _idx(0,SIZE-1), _idx(SIZE-1,0), _idx(SIZE-1,SIZE-1)]
        corner_score = sum(
            1 if board[i] == piece else (-1 if board[i] == opp else 0)
            for i in corners
        ) / 4.0  # -1..1

        raw = 0.6 * piece_score + 0.4 * corner_score  # -1..1
        return (raw + 1.0) / 2.0  # 0..1

    def greedy_action(self, state: tuple, legal_actions: list[int]) -> int:
        """Greedy: corners first, then maximize flips."""
        board = list(state[:-1])
        player = state[-1]
        piece = player + 1

        corners = {_idx(0,0), _idx(0,SIZE-1), _idx(SIZE-1,0), _idx(SIZE-1,SIZE-1)}
        for a in legal_actions:
            if a in corners:
                return a

        # Avoid moves adjacent to empty corners (X-squares)
        def _is_x_square(idx: int) -> bool:
            r, c = divmod(idx, SIZE)
            return (r in (0, 1, 6, 7) and c in (0, 1, 6, 7) and idx not in corners)

        safe = [a for a in legal_actions if not _is_x_square(a)]
        candidates = safe if safe else legal_actions

        return max(candidates, key=lambda a: len(_flips(board, a // SIZE, a % SIZE, piece)))

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------

    def obs_vector(self, state: tuple) -> np.ndarray:
        board = state[:-1]
        player = state[-1]
        piece = player + 1
        obs = np.zeros(SIZE * SIZE + 1, dtype=np.float32)
        for i, cell in enumerate(board):
            if cell == piece:
                obs[i] = 1.0
            elif cell != 0:
                obs[i] = -1.0
        obs[-1] = float(player)
        return obs

    def action_count(self) -> int:
        return SIZE * SIZE  # 64

    def encode_action(self, action: int) -> int:
        return int(action)

    def decode_action(self, idx: int) -> int:
        return int(idx)

    def fun_metric_norms(self) -> np.ndarray:
        # Othello: 60 possible moves, frequent lead changes due to mass flips
        # comeback can reach 30+ in Othello; set norm high to avoid clipping
        return np.array([1.0, 0.35, 12.0, 1.0, 35.0])

    # ------------------------------------------------------------------
    # Game-specific fun metric (G2): flip magnitude. Validated as a positive
    # predictor of LLM/human-judged Othello fun (Spearman +0.36, p=0.048) —
    # "一度にたくさん変わると嬉しい". The 5 generic metrics don't measure it.
    # ------------------------------------------------------------------

    def _flip_count(self, state_before: tuple, action) -> int:
        board = list(state_before[:-1])
        r, c = divmod(int(action), SIZE)
        return len(_flips(board, r, c, state_before[-1] + 1))

    def extra_metric_names(self) -> list[str]:
        return ["flip_mean"]

    def extra_metric_norms(self) -> np.ndarray:
        return np.array([6.0])  # ~max mean flips/move

    def extra_game_metrics(self, turns_data) -> np.ndarray:
        fl = [self._flip_count(sb, a) for sb, a, _ in turns_data]
        return np.array([float(np.mean(fl)) if fl else 0.0])

    def move_reward(self, state_before, action) -> float:
        """Dense flip reward: this move's flip count, normalized — drives the policy
        to actually maximize flips (terminal mean-flip reward is too sparse)."""
        return self._flip_count(state_before, action) / 6.0

    def extra_feature_vector(self, logs) -> np.ndarray:
        fl = [self._flip_count(t.state_before, t.chosen_action) for g in logs for t in g.turns]
        return np.array([float(np.mean(fl)) if fl else 0.0])

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_state(self, state: tuple) -> str:
        board = state[:-1]
        symbols = {0: ".", 1: "X", 2: "O"}
        rows = ["  a b c d e f g h"]
        for r in range(SIZE):
            row_str = " ".join(symbols[board[_idx(r, c)]] for c in range(SIZE))
            rows.append(f"{r+1} {row_str}")
        return "\n".join(rows)

    def format_turn(self, player: int, action: int,
                    state_before: tuple, state_after: tuple) -> str:
        r, c = divmod(int(action), SIZE)
        col_label = "abcdefgh"[c]
        symbol = "X" if player == 0 else "O"
        board_before = list(state_before[:-1])
        piece = player + 1
        flipped = len(_flips(board_before, r, c, piece))
        return f"  P{player}({symbol}): {col_label}{r+1} （{flipped}枚ひっくり返す）"

    def game_name(self) -> str:
        return "Othello"
