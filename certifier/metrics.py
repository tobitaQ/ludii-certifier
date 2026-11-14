"""
Fun metrics from game logs.

5 indicators mapped to prior literature (CLAUDE.md):
  flow       - Csikszentmihalyi: closeness to 50% win rate
  challenge  - Malone: std of win-probability proxy across turns
  pattern    - Koster: entropy of state distribution (unique states)
  decision   - Sweetser/GameFlow: ratio of turns where action mattered
  comeback   - Nature 2020: reversal count per game

All metric functions accept an optional `game` parameter.
When provided, game.win_prob() and game.greedy_action() are used.
When omitted, legacy Nim-specific fallbacks are used.
"""
import math
from collections import Counter

import numpy as np
from runner import GameLog, TurnLog


# ---------------------------------------------------------------------------
# Per-game helpers
# ---------------------------------------------------------------------------

def _nim_sum_eval(state) -> float:
    """Legacy Nim-specific win-prob fallback."""
    heaps = state[:3]
    nim_sum = heaps[0] ^ heaps[1] ^ heaps[2]
    total = sum(heaps)
    if total == 0:
        return 0.5
    return 0.9 if nim_sum != 0 else 0.1


def _win_prob(state, player: int, game=None) -> float:
    """Win probability for player 0, using game.win_prob() when available."""
    if game is not None:
        p = game.win_prob(state)
        return p if player == 0 else 1.0 - p
    # Legacy Nim fallback
    p = _nim_sum_eval(state)
    return p if player == 0 else 1.0 - p


def _win_prob_series(game_log: GameLog, game=None) -> list[float]:
    return [_win_prob(t.state_before, t.player, game) for t in game_log.turns]


def _reversals(prob_series: list[float], threshold: float = 0.5) -> int:
    count = 0
    side = prob_series[0] > threshold if prob_series else True
    for p in prob_series[1:]:
        new_side = p > threshold
        if new_side != side:
            count += 1
            side = new_side
    return count


def _policy_entropy_turn(turn: TurnLog) -> float:
    if hasattr(turn, "visit_counts") and turn.visit_counts:
        counts = np.array(list(turn.visit_counts.values()), dtype=float)
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log(probs + 1e-12)))
    n = len(turn.legal_actions)
    return math.log(n) if n > 1 else 0.0


# ---------------------------------------------------------------------------
# 5 fun indicators (per matchup / set of logs)
# ---------------------------------------------------------------------------

def metric_flow(logs: list[GameLog], game=None) -> float:
    """Closeness to 50% win rate."""
    if not logs:
        return 0.0
    wins0 = sum(1 for g in logs if g.winner == 0) / len(logs)
    wins1 = 1.0 - wins0
    return 1.0 - (abs(wins0 - 0.5) + abs(wins1 - 0.5)) / 2 * 2


def metric_challenge(logs: list[GameLog], game=None) -> float:
    """Std of win-probability proxy across turns."""
    if not logs:
        return 0.0
    stds = []
    for g in logs:
        series = _win_prob_series(g, game)
        if len(series) > 1:
            stds.append(float(np.std(series)))
    return float(np.mean(stds)) if stds else 0.0


def metric_pattern(logs: list[GameLog], game=None) -> float:
    """Entropy of visited state distribution."""
    if not logs:
        return 0.0
    counter: Counter = Counter()
    for g in logs:
        for t in g.turns:
            counter[t.state_before] += 1
    total = sum(counter.values())
    probs = np.array(list(counter.values()), dtype=float) / total
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def metric_decision(logs: list[GameLog], game=None) -> float:
    """
    Fraction of turns where chosen action differed from the greedy action.
    Uses game.greedy_action() when game is provided (game-agnostic).
    Falls back to Nim-specific heuristic when game=None.
    """
    if not logs:
        return 0.0
    total, non_trivial = 0, 0
    for g in logs:
        for t in g.turns:
            total += 1
            if not t.legal_actions:
                continue
            if game is not None:
                greedy = game.greedy_action(t.state_before, t.legal_actions)
            else:
                # Legacy Nim fallback
                try:
                    greedy = max(t.legal_actions, key=lambda a: a[1])
                except (TypeError, IndexError):
                    greedy = max(t.legal_actions)
            if t.chosen_action != greedy:
                non_trivial += 1
    return non_trivial / total if total else 0.0


def metric_comeback(logs: list[GameLog], game=None) -> float:
    """Average number of lead reversals per game."""
    if not logs:
        return 0.0
    reversals = []
    for g in logs:
        series = _win_prob_series(g, game)
        reversals.append(_reversals(series))
    return float(np.mean(reversals))


def metric_policy_entropy(logs: list[GameLog], game=None) -> float:
    """Average policy entropy per turn."""
    if not logs:
        return 0.0
    vals = [_policy_entropy_turn(t) for g in logs for t in g.turns]
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Combined feature vector (per matchup)
# ---------------------------------------------------------------------------

METRIC_NAMES = ["flow", "challenge", "pattern", "decision", "comeback"]


def metric_names(game=None, include_extra=False) -> list[str]:
    """The 5 generic metric names, plus the game's extra metrics if requested."""
    names = list(METRIC_NAMES)
    if include_extra and game is not None:
        names += list(game.extra_metric_names())
    return names


def compute_feature_vector(logs: list[GameLog], game=None, include_extra=False) -> np.ndarray:
    """Fun metrics over a matchup (METRIC_NAMES order); with include_extra=True the
    game's extra metrics (e.g. Othello flip_mean) are appended."""
    base = np.array([
        metric_flow(logs, game),
        metric_challenge(logs, game),
        metric_pattern(logs, game),
        metric_decision(logs, game),
        metric_comeback(logs, game),
    ])
    if include_extra and game is not None:
        extra = game.extra_feature_vector(logs)
        if len(extra):
            base = np.concatenate([base, extra])
    return base


def fun_score(logs: list[GameLog], weights: np.ndarray, game=None) -> float:
    """Weighted sum of fun metrics."""
    return float(weights @ compute_feature_vector(logs, game))


# ---------------------------------------------------------------------------
# Summary dict (for display)
# ---------------------------------------------------------------------------

def summary(logs: list[GameLog], game=None) -> dict:
    return {
        "flow":           metric_flow(logs, game),
        "challenge":      metric_challenge(logs, game),
        "pattern":        metric_pattern(logs, game),
        "decision":       metric_decision(logs, game),
        "comeback":       metric_comeback(logs, game),
        "policy_entropy": metric_policy_entropy(logs, game),
        "avg_length":     sum(g.n_turns for g in logs) / len(logs) if logs else 0.0,
        "unique_states":  len({t.state_before for g in logs for t in g.turns}),
    }
