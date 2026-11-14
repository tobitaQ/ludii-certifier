"""Play-validation of EVERY runnable converted game (certified or not).

For each game: varied play (random, greedy both seats, MCTS) + win-reward
MaskablePPO self-play, recording what breaks and whether the game "converges
as a game":

  status   OK | CRASH (exception) | NONTERMINATING (ply cap hit in random play)
           | DEADLOCK (no legal move while not done) | TIMEOUT (budget)
  degenerate  first-mover decisive share outside [0.15, 0.85] under random play,
              or draw rate >= 0.5, or mean length < 6
  learnable   trained policy beats random (both seats) > 0.6 AND the final
              checkpoint beats the 1/3 checkpoint > 0.55   (skill exists, is learnable)
  rl_balance  first-mover share under trained self-play (does learning collapse it?)

No API calls. Resumable; one JSON per game under results/validate_play/.
Usage: .venv/bin/python pipeline/validate_play.py [--workers 3] [--timesteps 100000]
         [--budget 600] [game ...]
"""
import hashlib
import sys, os, json, time, random, argparse, signal, traceback, glob
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline"))
OUT = ROOT / "results" / "validate_play"
PLY_CAP = 1500


def _alarm(signum, frame):
    raise TimeoutError("budget exceeded")


def source_hash(key):
    """sha256 of the translation and its ops wrapper, so a record says WHICH translation it describes.

    Axis B had no such stamp, and nothing re-ran it when a repair landed. On 2026-11-01 a spot check
    found stored rows describing translations that no longer exist: Spaji recorded at 72.5% draws where
    the current file draws none, Conspirateurs at 0% where it now caps out every game. The ladder has a
    rule that a change is followed by a full re-certification; axis B did not, and this is the half of
    that rule a file can enforce.
    """
    h = hashlib.sha256()
    for sub in ("converted", "ops"):
        f = ROOT / "src" / "games" / sub / f"{key}.py"
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _load(key):
    from stage5_rl import _make_game
    from rl_adapter import adapt
    return adapt(_make_game(f"converted:{key}"))


def play(game, agents, seed, cap=PLY_CAP):
    """Returns (winner: 0/1/-1, plies, flag) flag in {None,'deadlock','cap'}."""
    st = game.reset(); n = 0
    while n < cap:
        p = game.current_player(st)
        la = game.legal_actions(st)
        if not la:
            return -1, n, "deadlock"
        a = agents[p](st, la)
        st, rw, d = game.step(st, a)
        n += 1
        if d:
            return (p if rw > 0 else (1 - p if rw < 0 else -1)), n, None
    return -1, n, "cap"


def rnd_agent(seed):
    r = random.Random(seed)
    return lambda st, la: la[r.randrange(len(la))]


def stats(results):
    w = {0: 0, 1: 0, -1: 0}; lens = []; flags = {"deadlock": 0, "cap": 0}
    for win, n, f in results:
        w[win] += 1; lens.append(n)
        if f:
            flags[f] += 1
    n = len(results); dec = w[0] + w[1]
    return {"p0": w[0] / n, "p1": w[1] / n, "draw": w[-1] / n,
            "share": (w[0] / dec) if dec else 0.5, "len": sum(lens) / n, **flags}


class WinEnv:
    """ONE-SIDED env: the agent controls one seat (alternating per episode),
    the opponent is a fixed policy (random, or a frozen checkpoint). Reward is
    from the AGENT's perspective (+1 win / -1 loss / 0 draw) at the terminal
    step. A single-policy both-sides self-play env is wrong here: without
    sign-flipping per turn the value function rewards the opponent's wins too
    and the policy can learn to end games by losing (observed on 15 games)."""
    def __init__(self, game, opponent=None, seed=0):
        from stage5_rl import GameFunEnv
        import numpy as np
        self.inner = GameFunEnv(game, np.ones(5, dtype=np.float32))
        self.game = game
        self.observation_space = self.inner.observation_space
        self.action_space = self.inner.action_space
        self.metadata = {}
        self.opponent = opponent            # callable(state, legal) -> action, or None = random
        self.rng = np.random.default_rng(seed)
        self.seat = 0

    def _opp_move(self):
        st = self.inner.state
        la = self.game.legal_actions(st)
        if not la:
            return None, 0.0, True
        a = self.opponent(st, la) if self.opponent else la[self.rng.integers(len(la))]
        nxt, rw, done = self.game.step(st, a)
        self.inner.state = nxt
        return nxt, rw, done

    def reset(self, seed=None, options=None):
        obs, info = self.inner.reset(seed=seed)
        self.seat = int(self.rng.integers(2))
        if self.game.current_player(self.inner.state) != self.seat:
            nxt, rw, done = self._opp_move()
            if done:                        # degenerate: opponent's first move ends the game
                return self.game.obs_vector(self.inner.state), info
            obs = self.game.obs_vector(nxt)
        return obs, info

    def step(self, action):
        st = self.inner.state
        legal = self.game.legal_actions(st)
        if not legal:                       # the agent's seat has no legal move: terminal, no reward
            return self.game.obs_vector(st), 0.0, True, False, {}
        dec = self.game.decode_action(int(action))
        if dec not in legal:
            dec = legal[self.inner.np_random.integers(len(legal))]
        nxt, rw, done = self.game.step(st, dec)      # rw: mover (= agent) perspective
        self.inner.state = nxt
        if done:
            return self.game.obs_vector(nxt), float(rw), True, False, {}
        # opponent replies until it is the agent's turn again (extra turns, passes)
        while self.game.current_player(self.inner.state) != self.seat:
            nxt, orw, done = self._opp_move()
            if nxt is None:
                return self.game.obs_vector(self.inner.state), 0.0, True, False, {}
            if done:
                return self.game.obs_vector(nxt), float(-orw), True, False, {}
        if not self.game.legal_actions(self.inner.state):
            return self.game.obs_vector(self.inner.state), 0.0, True, False, {}
        return self.game.obs_vector(self.inner.state), 0.0, False, False, {}

    def action_masks(self):
        return self.inner.action_masks()


def train(game, timesteps, n_envs, seed=1):
    import torch
    torch.set_num_threads(2)
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sb3_contrib import MaskablePPO
    import gymnasium as gym

    class _G(gym.Env):
        def __init__(self, i=0):
            self.w = WinEnv(game, seed=seed * 1000 + i)
            self.observation_space = self.w.observation_space
            self.action_space = self.w.action_space
        def reset(self, seed=None, options=None):
            return self.w.reset(seed=seed)
        def step(self, a):
            return self.w.step(a)
        def action_masks(self):
            return self.w.action_masks()

    env = DummyVecEnv([(lambda i=i: _G(i)) for i in range(n_envs)])
    model = MaskablePPO("MlpPolicy", env, device="cpu", verbose=0, n_steps=256, batch_size=256,
                        n_epochs=4, learning_rate=3e-4, gamma=0.99, ent_coef=0.01, seed=seed)
    ckpts = []
    third = max(1, timesteps // 3)
    for i in range(3):
        model.learn(total_timesteps=third, reset_num_timesteps=False)
        ckpts.append({k: v.detach().clone() for k, v in model.policy.state_dict().items()} if i < 2 else None)
    return model, ckpts


def policy_agent(model, game, seed):
    import numpy as np
    r = np.random.default_rng(seed)
    def act(st, la):
        obs = game.obs_vector(st)
        mask = np.zeros(game.action_count(), dtype=bool)
        for a in la:
            mask[game.encode_action(a)] = True
        a, _ = model.predict(obs, action_masks=mask, deterministic=False)
        dec = game.decode_action(int(a))
        return dec if dec in la else la[r.integers(len(la))]
    return act


def validate_one(args):
    # `off` shifts every episode seed of the random screen, so the same four conditions can be
    # measured on a DIFFERENT 40 games per implementation. That turns "how stable is the
    # plays/does-not-play partition under its own sample" from an interval argument into a
    # measurement -- the same move the ladder makes with held-out seeds (2026-11-04).
    key, timesteps, n_envs, budget, off = (args + (0,))[:5] if isinstance(args, tuple) else args
    t0 = time.time()
    out = {"game": key, "status": "OK", "source": source_hash(key), "sample_offset": off,
           "checks": {}}
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(budget)
    try:
        game = _load(key)
        # 1. random play
        rr = [play(game, [rnd_agent(1000 + off + i), rnd_agent(2000 + off + i)], off + i)
              for i in range(40)]
        s = stats(rr); out["checks"]["random"] = s
        if s["cap"] > 0:
            out["status"] = "NONTERMINATING"
        if s["deadlock"] > 0:
            out["status"] = "DEADLOCK"
        out["degenerate"] = not (0.15 <= s["share"] <= 0.85) or s["draw"] >= 0.5 or s["len"] < 6
        # 2. greedy vs random both seats
        from pipeline import make_greedy_agent
        g = make_greedy_agent(game)
        ga = lambda st, la: g.select_action(st, la)
        gr = [play(game, [ga, rnd_agent(3000 + i)], i) for i in range(20)]
        rg = [play(game, [rnd_agent(4000 + i), ga], i) for i in range(20)]
        out["checks"]["greedy_vs_random"] = (stats(gr)["p0"] + stats(rg)["p1"]) / 2
        # 3. win-reward RL self-play (skipped in quick mode: timesteps <= 0)
        if timesteps <= 0:
            out["learnable"] = None; out["rl_degenerate"] = None
            out["seconds"] = round(time.time() - t0)
            signal.alarm(0)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"{key}.json").write_text(json.dumps(out, indent=1))
            return out
        model, ckpts = train(game, timesteps, n_envs)
        pa = policy_agent(model, game, 7)
        pr = [play(game, [pa, rnd_agent(5000 + i)], i) for i in range(20)]
        rp = [play(game, [rnd_agent(6000 + i), pa], i) for i in range(20)]
        vs_random = (stats(pr)["p0"] + stats(rp)["p1"]) / 2
        # final vs 1/3 checkpoint
        import copy
        early = copy.deepcopy(model); early.policy.load_state_dict(ckpts[0])
        ea = policy_agent(early, game, 8)
        fe = [play(game, [pa, ea], i) for i in range(20)]
        ef = [play(game, [ea, pa], i) for i in range(20)]
        vs_early = (stats(fe)["p0"] + stats(ef)["p1"]) / 2
        ss = [play(game, [pa, policy_agent(model, game, 9)], i) for i in range(30)]
        st_ = stats(ss)
        out["checks"]["rl"] = {"vs_random": vs_random, "final_vs_early": vs_early,
                               "selfplay_share": st_["share"], "selfplay_draw": st_["draw"],
                               "selfplay_len": st_["len"], "timesteps": timesteps}
        out["learnable"] = bool(vs_random > 0.6 and vs_early > 0.55)
        out["rl_degenerate"] = not (0.15 <= st_["share"] <= 0.85) or st_["draw"] >= 0.5
    except TimeoutError:
        out["status"] = "TIMEOUT"
    except Exception as e:
        out["status"] = "CRASH"; out["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        out["trace"] = traceback.format_exc()[-800:]
    finally:
        signal.alarm(0)
    out["seconds"] = round(time.time() - t0)
    (OUT / f"{key}.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--out-dir", default=None, help="results subdir (default validate_play)")
    ap.add_argument("--levels", default=None, help="comma list of ladder levels to include, e.g. L4,L5")
    ap.add_argument("--sample-offset", type=int, default=0,
                    help="shift every episode seed of the random screen, to measure the same four "
                         "conditions on a different 40 games (use with --out-dir)")
    ap.add_argument("games", nargs="*")
    a = ap.parse_args()
    global OUT
    if a.out_dir:
        OUT = ROOT / "results" / a.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    conv = json.load(open(ROOT / "results" / "conversion_results.json"))
    keys = sorted({c["module"].split(".")[-1] for c in conv
                   if c.get("smoke_ok") and c.get("module", "").startswith("games.converted.")})
    if a.levels:
        lad = {r["game"]: r["level"] for r in json.load(open(ROOT / "results" / "ladder_certification_K30_effect.json"))}
        want = set(a.levels.split(","))
        keys = [k for k in keys if lad.get(k) in want]
    if a.games:
        keys = [k for k in keys if k in set(a.games)]
    keys = [k for k in keys if not (OUT / f"{k}.json").exists()]
    print(f"Play-validation: {len(keys)} games, {a.workers} workers, RL {a.timesteps} steps, "
          f"budget {a.budget}s\n", flush=True)
    tasks = [(k, a.timesteps, a.n_envs, a.budget, a.sample_offset) for k in keys]
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(validate_one, tasks)):
            rl = r.get("checks", {}).get("rl", {})
            print(f"  [{i+1:3d}/{len(keys)}] {r['game']:24s} {r['status']:14s} "
                  f"deg={r.get('degenerate')!s:5s} learn={r.get('learnable')!s:5s} "
                  f"vsRnd={rl.get('vs_random', float('nan')):.2f} fin/early={rl.get('final_vs_early', float('nan')):.2f} "
                  f"({r['seconds']}s) {r.get('error', '')[:50]}", flush=True)
    from collections import Counter
    allr = [json.load(open(p)) for p in glob.glob(str(OUT / "*.json"))]
    print("\n=== status:", dict(Counter(r["status"] for r in allr)))
    print("    degenerate:", sum(1 for r in allr if r.get("degenerate")),
          " learnable:", sum(1 for r in allr if r.get("learnable")), " of", len(allr))


if __name__ == "__main__":
    main()
