"""Record-grain metric vector shared by the Stage 4/5 scoring path and the RL reward.

Why this module exists (2026-09-10, feature audit): `stage5_rl.compute_single_game_metrics`
used `game.win_prob(state)` -- the win probability *for the side to move* -- without
re-orienting it to one player, so at record grain its flow / challenge / comeback were
computed on a series that alternates perspective every ply (Table 3 note of the CHI
paper).  `metrics.compute_feature_vector([log])` uses the one-player series.  This
module is the single implementation both paths now call; `perspective="mover"` keeps
the historical behaviour for reproducing the published Stage 4 numbers.

Contract (one-player, record grain):
  perspective   first mover (player 0); each ply's win probability is re-oriented
                from that ply's recorded player (or game.current_player(state) when
                no player list is given), never from ply parity -- extra turns
                (Kalah) and multi-jumps are therefore handled.
  flow          outcome balance of the log set, 1 - 2|w - 0.5| with w the fraction of
                records won by player 0 (a draw is a non-win).  For ONE record w is 0
                or 1, so flow is identically 0 at this grain (not estimable).
  challenge     SD of the one-player per-ply series.
  pattern       entropy of the visited-state distribution (= log length when no
                state recurs).
  decision      fraction of plies whose chosen move differs from the greedy move.
  comeback      number of sign changes of (p - 0.5) along the one-player series.
`dev/test_feature_consistency.py` asserts this function equals
`metrics.compute_feature_vector([log], game)` on the same record.
"""
from collections import Counter

import numpy as np


def _default_game():
    from games.nim import Nim
    return Nim()


def compute_single_game_metrics(turns_data, game=None, include_extra=False,
                                perspective="one-player", players=None, winner=None):
    """
    turns_data: list of (state_before, chosen_action, legal_actions)
    game:       BoardGame instance (game.win_prob / game.greedy_action / game.current_player)
    perspective:"one-player" (Table 3 definition, default) or "mover" (historical
                Stage 4/5 path: side-to-move series, flow = 1-2|mean p - 0.5|).
    players:    optional list of the player to move on each ply (e.g. GameLog turn
                .player); when None, game.current_player(state_before) is used.
    winner:     unused at record grain (flow is 0 for one record either way); kept
                so callers can pass it explicitly.
    Returns np.ndarray [flow, challenge, pattern, decision, comeback] (+ extras).
    """
    if game is None:
        game = _default_game()
    n_extra = len(game.extra_metric_names()) if include_extra else 0
    if not turns_data:
        return np.zeros(5 + n_extra)
    if perspective not in ("one-player", "mover"):
        raise ValueError(perspective)

    win_probs, states, chosen_actions, greedy_actions = [], [], [], []
    for i, (state_before, chosen, legal) in enumerate(turns_data):
        p = game.win_prob(state_before)
        if perspective == "one-player":
            pl = players[i] if players is not None else game.current_player(state_before)
            p = p if pl == 0 else 1.0 - p
        win_probs.append(p)
        states.append(state_before if isinstance(state_before, tuple)
                      else tuple(int(x) for x in state_before))
        greedy_actions.append(game.greedy_action(state_before, legal) if legal else chosen)
        chosen_actions.append(chosen)

    if perspective == "mover":
        flow = 1.0 - abs(float(np.mean(win_probs)) - 0.5) * 2
    else:
        flow = 0.0          # one record: w in {0,1} -> outcome balance is 0 (constant)
    challenge = float(np.std(win_probs)) if len(win_probs) > 1 else 0.0

    cnt = Counter(states)
    probs = np.array(list(cnt.values()), dtype=float) / len(states)
    pattern = float(-np.sum(probs * np.log(probs + 1e-12)))

    non_trivial = sum(1 for c, g in zip(chosen_actions, greedy_actions) if c != g)
    decision = non_trivial / len(chosen_actions)

    side, reversals = win_probs[0] > 0.5, 0
    for p in win_probs[1:]:
        ns = p > 0.5
        if ns != side:
            reversals += 1
            side = ns

    base = np.array([flow, challenge, pattern, decision, float(reversals)])
    if include_extra and game.extra_metric_names():
        base = np.concatenate([base, game.extra_game_metrics(turns_data)])
    return base
