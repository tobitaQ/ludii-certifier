"""Generic RL interface for converted games that lack obs_vector / action_count /
encode_action / decode_action (Stage 5 needs them; 5 of the 12 Stage-5 games
do not implement them).

adapt(game) probes the game with random play and patches the INSTANCE:
  * observation = tanh(flat_state / scale), scale = max |value| seen in probing,
    fixed length (padded / truncated to the probed maximum);
  * action universe = every action seen in probing; int actions are widened to
    the contiguous range [min, max] so unseen cells are still encodable.
Games that already implement the interface are returned unchanged.
"""
import random
import numpy as np


def _flat(state):
    out = []
    stack = [state]
    while stack:
        x = stack.pop()
        if isinstance(x, (tuple, list)):
            stack.extend(reversed(x))
        elif isinstance(x, np.ndarray):
            out.extend(float(v) for v in x.ravel())
        else:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                out.append(0.0)
    return out


def adapt(game, probe_games=200, max_plies=600, seed=0):
    need = [m for m in ("obs_vector", "action_count", "encode_action", "decode_action")
            if not _implemented(game, m)]
    if not need:
        return game
    rng = random.Random(seed)
    acts, maxlen, scale = set(), 0, 1.0
    for _ in range(probe_games):
        s = game.reset()
        for _ in range(max_plies):
            f = _flat(s)
            maxlen = max(maxlen, len(f))
            scale = max(scale, max((abs(v) for v in f), default=1.0))
            la = game.legal_actions(s)
            if not la:
                break
            acts.update(la)
            s, _, d = game.step(s, rng.choice(la))
            if d:
                break
    if all(isinstance(a, (int, np.integer)) for a in acts):
        lo, hi = min(acts), max(acts)
        universe = list(range(int(lo), int(hi) + 1))
    else:
        universe = sorted(acts, key=repr)
    index = {a: i for i, a in enumerate(universe)}
    # Random probing misses rare actions (37/305 games crashed on an unseen
    # (from,to) tuple). Reserve headroom and register unseen actions lazily;
    # the Discrete space is the capacity, masks only ever mark known actions.
    capacity = min(max(2 * len(universe), len(universe) + 256), 8192)
    n_obs, sc = maxlen, float(scale)

    def obs_vector(state):
        f = _flat(state)[:n_obs]
        f += [0.0] * (n_obs - len(f))
        return np.tanh(np.asarray(f, dtype=np.float32) / sc)

    def action_count():
        return capacity

    def encode_action(a):
        if a in index:
            return index[a]
        if len(universe) >= capacity:
            raise KeyError(f"action {a!r}: adapter capacity {capacity} exhausted")
        index[a] = len(universe)
        universe.append(a)
        return index[a]

    def decode_action(i):
        i = int(i)
        return universe[i] if i < len(universe) else universe[0]

    if "obs_vector" in need:
        game.obs_vector = obs_vector
    if "action_count" in need:
        game.action_count = action_count
    if "encode_action" in need:
        game.encode_action = encode_action
    if "decode_action" in need:
        game.decode_action = decode_action
    game._rl_adapter = {"n_obs": n_obs, "scale": sc, "n_actions": capacity,
                        "probed_actions": len(acts), "lazy": True}
    return game


def _implemented(game, name):
    """True if the game's class overrides the base method (base raises NotImplementedError).
    Operator wrappers (games.ops.*) delegate every method to the wrapped game, so the static
    check is confirmed by a probe call: NotImplementedError means 'not implemented'."""
    fn = getattr(type(game), name, None)
    if fn is None:
        return False
    code = getattr(fn, "__code__", None)
    src_ok = code is not None and "base.py" not in code.co_filename
    if not src_ok:
        return False
    try:
        st = game.reset()
        if name == "obs_vector":
            game.obs_vector(st)
        elif name == "action_count":
            game.action_count()
        elif name == "encode_action":
            la = game.legal_actions(st)
            if la:
                game.encode_action(la[0])
        elif name == "decode_action":
            game.decode_action(0)
    except NotImplementedError:
        return False
    except Exception:
        pass
    return True
