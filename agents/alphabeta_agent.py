"""Terminal-aware alpha-beta agent with iterative deepening and a time budget.

Handles games where a move can keep the same player to move (Kalah extra turns,
Othello passes) by NOT flipping the value for same-player children, so it stays
correct beyond strictly-alternating games. Uses the exact terminal reward (from
the mover's view, in {-1,0,1}) at leaves and game.win_prob as the depth-horizon
heuristic. Mirrors gui_shell._ab (kept independent so the pipeline stays
pygame-free).

Stronger than MCTS in small-branching games like Kalah, so it can contest an
expert human where MCTS saturates (Kalah has no MCTS certification ladder,
tau=0). Calibrate DOWN with time_budget / max_depth for weaker tiers.
"""
import time
import random

from base import Agent


class _Timeout(Exception):
    pass


class AlphaBetaAgent(Agent):
    def __init__(self, game, time_budget: float = 0.30, max_depth: int = 64,
                 seed: int | None = None, order_key=None):
        self.game = game
        self.time_budget = time_budget
        self.max_depth = max_depth
        self.order_key = order_key
        self._rng = random.Random(seed)

    def select_action(self, state, legal_actions):
        if len(legal_actions) <= 1:
            return legal_actions[0]
        deadline = time.monotonic() + self.time_budget
        best_a = legal_actions[0]
        for depth in range(1, self.max_depth + 1):          # iterative deepening
            try:
                a = self._root(state, depth, deadline)
            except _Timeout:
                break
            if a is not None:
                best_a = a
        return best_a

    def _root(self, state, depth, deadline):
        game = self.game
        mover = game.current_player(state)
        legal = game.legal_actions(state)
        if self.order_key is not None:
            legal = sorted(legal, key=self.order_key)
        best, best_a, alpha, beta = -2.0, None, -2.0, 2.0
        for a in legal:
            ns, r, done = game.step(state, a)
            if done:
                v = float(r)
            elif game.current_player(ns) == mover:          # same player moves again
                v = self._ab(ns, depth - 1, alpha, beta, deadline)
            else:                                           # normal turn change: negamax flip
                v = -self._ab(ns, depth - 1, -beta, -alpha, deadline)
            if v > best:
                best, best_a = v, a
            if best > alpha:
                alpha = best
        return best_a

    def _ab(self, state, depth, alpha, beta, deadline):
        game = self.game
        if time.monotonic() > deadline:
            raise _Timeout
        legal = game.legal_actions(state)
        if not legal or depth == 0:
            return 2.0 * game.win_prob(state) - 1.0
        mover = game.current_player(state)
        if self.order_key is not None:
            legal = sorted(legal, key=self.order_key)
        best = -2.0
        for a in legal:
            ns, r, done = game.step(state, a)
            if done:
                v = float(r)
            elif game.current_player(ns) == mover:
                v = self._ab(ns, depth - 1, alpha, beta, deadline)
            else:
                v = -self._ab(ns, depth - 1, -beta, -alpha, deadline)
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best
