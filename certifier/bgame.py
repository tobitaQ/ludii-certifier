"""bgame — one entry point for the certified board-game library.

Python API
----------
    import bgame
    g  = bgame.load("oware")            # BoardGame (with its repair operators, if any)
    s  = g.reset(); a = g.legal_actions(s)[0]; s2, reward, done = g.step(s, a)
    bgame.games(level="L5")             # keys at a certification level
    bgame.select(level="L5", playable=True, players=2)   # research selection over the catalog (see select.__doc__)
    bgame.catalog()["baghchal"]         # the full record of one game (fidelity / play / exact / analysis / reference / residual / variants)
    bgame.load("hnefatafl", variant="fetlar_latin")      # a derived (option-certified / balanced) version; canonical untouched
    bgame.variants("awari")             # derived versions on record
    bgame.info("oware")                 # ladder record, operators, game-property check
    bgame.certify("oware", K=30)        # fidelity ladder vs Ludii (L0..L5) -> dict
    bgame.validate("oware")             # game-property certificate (B): terminates, non-degenerate, learnable
    bgame.repair("oware", "add_forced_pass")   # apply a repair operator, re-certify
    bgame.agent("oware", model="local") # run the rule agent (local | opus); promotes only if the ladder says so
    bgame.gym("oware")                  # Gymnasium env class with action_masks() (MaskablePPO-ready)
    bgame.play("oware", policy="random", seed=1)   # one rollout -> trajectory

Command line (same verbs)
-------------------------
    bgame list [--level L5]             bgame info <game>
    bgame play <game> [--seed 1]        bgame certify <game> [-K 30]
    bgame validate <game>               bgame repair <game> <operator> [key=value ...]
    bgame agent <game> [--model local|opus] [--turns 40]
    bgame gym <game>                    (writes exports/env_<game>.py)
    bgame ui [--port 8765]              (Ladder Workbench in the browser)
"""
import sys, json, importlib, subprocess, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
R = ROOT / "results"
PY = str(ROOT / ".venv" / "bin" / "python")


def _json(name, default):
    p = R / name
    return json.load(open(p)) if p.exists() else default


def _ladder():
    return {r["game"]: r for r in _json("ladder_certification_K30_effect.json", [])}


def _ludii_name(key):
    for c in _json("conversion_results.json", []):
        if c.get("module", "").endswith("." + key) and c.get("lud_path"):
            return Path(c["lud_path"]).name
    raise KeyError(f"no Ludii reference recorded for {key!r}")


# ---------------------------------------------------------------- library
def games(level=None):
    """Keys of the corpus, optionally only those certified at `level` (L2-, L3, L4, L5).

    The corpus is what results/corpus_manifest.json names, NOT what is in the directory. Those were the
    same thing until the control group's forty translations were written next to it and a full
    re-certification silently produced 345 rows; the directory also holds seven residual translations and
    a working copy, which were never the corpus either. The manifest refuses a run whose membership has
    changed without being re-blessed.
    """
    lad = _ladder()
    import corpus_manifest
    m = corpus_manifest.load()
    if m:
        corpus_manifest.require("bgame.games()")
        keys = sorted(m["games"])
    else:
        keys = sorted(p.stem for p in (ROOT / "src" / "games" / "converted").glob("*.py")
                      if not p.stem.startswith("_") and "_agent" not in p.stem)
    if level is None:
        return keys
    return [k for k in keys if lad.get(k, {}).get("level") == level]


def module_name(key, variant=None):
    if variant:
        for x in _json("variants.json", {}).get(key, []):
            if x["variant"] == variant:
                return x["module"]           # a variant may be the canonical module itself (hnefatafl fetlar_latin)
        return f"games.ops.{key}__{variant}"
    ops = _json("operators_applied.json", {})
    return f"games.ops.{key}" if key in ops else f"games.converted.{key}"


def load(key, variant=None):
    """Instantiate the game. Default = the canonical, Ludii-certified artifact (translation + its recorded repair
    operators). `variant` = a derived version from results/variants.json (an option-certified or balanced variant);
    variants never replace the canonical game."""
    from ludii_oracle import find_boardgame_class
    mod = importlib.import_module(module_name(key, variant))
    return find_boardgame_class(mod)()


def catalog():
    """The research catalog (results/catalog.json, built by pipeline/build_catalog.py): one record per game with
    fidelity, game-property, imbalance, exact value, analysis, reference issues, residual status and variants."""
    return {x["key"]: x for x in _json("catalog.json", {"games": []})["games"]}


def select(level=None, playable=None, players=None, family=None, row=None, imbalance=None, exact=None,
           reference_issue=None, operators=None, min_plies=None, max_plies=None):
    """Keys of the games matching every given condition (None = no constraint). Examples
        select(level="L5", playable=True, players=2)            # faithful, plays: RL benchmark set (169)
        select(row="degenerate", imbalance="asymmetric-roles")  # hunt-type games, one side needs skill
        select(exact="LOSS")                                    # solved second-player wins
        select(family="hunt", level="L5")                       # a mechanically homogeneous family
        select(reference_issue=True)                            # Ludii's own defects / simplifications
    """
    out = []
    for k, x in catalog().items():
        if level is not None and x["fidelity"]["level"] != level: continue
        if playable is not None and x["flags"]["playable"] != playable: continue
        if players is not None and x["players"] != players: continue
        if family is not None and x["family"] != family: continue
        if row is not None and x["play"]["row"] != row: continue
        if imbalance is not None and x["play"]["imbalance"] != imbalance: continue
        if exact is not None and (x["exact"] or {}).get("value_first_mover") != exact: continue
        if reference_issue is not None and x["flags"]["reference_issue"] != reference_issue: continue
        if operators is not None and (bool(x["operators"]) != operators if isinstance(operators, bool) else operators not in x["operators"]): continue
        if min_plies is not None and (x["fidelity"]["max_plies"] or 0) < min_plies: continue
        if max_plies is not None and (x["fidelity"]["max_plies"] or 0) > max_plies: continue
        out.append(k)
    return out


def variants(key=None, designed=True):
    """Derived versions: {key: [{slug, options, level, plays}]} (or the list for one key).

    Two kinds, and the difference matters: a CERTIFIED variant is a real Ludii ruleset (an option the `.lud`
    already carries), so it has a level on the fidelity ladder; a DESIGNED variant (stage 3, from
    results/balance_stage3.json) changes a parameter the reference does not offer, so no `.lud` exists to
    certify it against and its record carries `verified: False, level: None`. Pass designed=False for only
    the certified ones."""
    v = {k: list(x) for k, x in _json("variants.json", {}).items()}
    if designed:
        for name, d in _json("balance_stage3.json", {}).items():
            if "module" not in d:
                continue
            v.setdefault(d["game"], []).append(
                {"variant": d["tag"], "options": None, "level": None, "verified": False,
                 "module": d["module"], "designed": True, "change": d.get("change"),
                 "band": d.get("band"), "mcts_band": d.get("mcts_band")})
    return v if key is None else v.get(key, [])


def info(key):
    lad = _ladder().get(key, {})
    vp = {}
    for d in ("validate_play_500k", "validate_play"):
        p = R / d / f"{key}.json"
        if p.exists():
            vp = json.load(open(p)); break
    return {"game": key, "module": module_name(key), "level": lad.get("level"),
            "ladder": {k: lad.get(k) for k in ("l3", "l4", "l5", "mapping", "first_fail_episode")},
            "operators": _json("operators_applied.json", {}).get(key, []),
            "game_property": {k: vp.get(k) for k in ("status", "checks")} if vp else None,
            "ludii": lad.get("ludii") or _ludii_name(key)}


def certify(key, K=30, options=None, max_steps=600):
    """Fidelity ladder against Ludii: returns {'level', 'l3', 'l4', 'l5', 'counterexample', ...}."""
    from ludii_oracle import certify as _c
    return _c(module_name(key), _ludii_name(key), episodes=K, max_steps=max_steps, options=options)


def validate(key, timesteps=100_000, n_envs=32, budget=600):
    """Game-property certificate (B): random/greedy play terminates, non-degenerate, RL-learnable."""
    from validate_play import validate_one
    return validate_one((key, timesteps, n_envs, budget))


def repair(key, operator, K=30, **args):
    """Apply a repair operator (see pipeline/operators.py: add_forced_pass, override_end, override_winner,
    remove_meta_move) to the game, record it in the ledger and re-certify."""
    from apply_operators import write_module
    ops = _json("operators_applied.json", {})
    cur = list(ops.get(key, []))
    cur.append({"operator": operator, "args": args} if args else operator)
    write_module(key, cur)
    ops[key] = cur
    (R / "operators_applied.json").write_text(json.dumps(ops, indent=1))
    for m in list(sys.modules):
        if m.startswith(f"games.ops.{key}"):
            del sys.modules[m]
    return certify(key, K=K)


def agent(key, model="local", K=3, turns=40, timeout=2700):
    """Run the rule agent in a sandbox; the ladder (K=30) decides whether its fix is promoted."""
    if model == "local":
        cmd = [PY, "pipeline/rule_agent_local.py", "--K", str(K), "--max-turns", str(turns), key]
        out = R / "rule_agent_local" / f"{key}.json"
    else:
        cmd = [PY, "pipeline/rule_agent.py", "--K", str(K), "--max-turns", str(turns), "--timeout", str(timeout), key]
        out = R / "rule_agent" / f"{key}.json"
    subprocess.run(cmd, cwd=ROOT)
    return json.load(open(out)) if out.exists() else None


def gym(key):
    """Gymnasium env class (one-sided; action_masks() for sb3_contrib.MaskablePPO)."""
    import numpy as np, gymnasium as G
    from rl_adapter import adapt
    from stage5_rl import GameFunEnv
    game = adapt(load(key))

    class Env(G.Env):
        def __init__(self):
            self.game = game
            self.inner = GameFunEnv(self.game, np.ones(5, dtype=np.float32))
            self.observation_space = self.inner.observation_space
            self.action_space = self.inner.action_space
        def reset(self, seed=None, options=None):
            return self.inner.reset(seed=seed)
        def step(self, a):
            st = self.inner.state
            legal = self.game.legal_actions(st)
            dec = self.game.decode_action(int(a))
            if dec not in legal:
                dec = legal[self.inner.np_random.integers(len(legal))]
            nxt, rw, done = self.game.step(st, dec)
            self.inner.state = nxt
            return self.game.obs_vector(nxt), float(rw), bool(done), False, {}
        def action_masks(self):
            return self.inner.action_masks()
    Env.__name__ = key.title().replace("_", "") + "Env"
    return Env


def play(key, policy="random", seed=1, max_steps=600):
    """One rollout. policy: 'random' | 'greedy' (uses the game's greedy_action if present).
    Returns {'history': [...], 'reward', 'done', 'steps', 'final_state'}."""
    g = load(key); rng = random.Random(seed)
    s = g.reset(); hist = []; rw = 0.0; done = False
    for _ in range(max_steps):
        legal = g.legal_actions(s)
        if not legal:
            break
        a = g.greedy_action(s) if policy == "greedy" and hasattr(g, "greedy_action") else rng.choice(legal)
        s, rw, done = g.step(s, a); hist.append(a)
        if done:
            break
    return {"history": hist, "reward": rw, "done": done, "steps": len(hist), "final_state": s}


# ---------------------------------------------------------------- CLI
def _cli(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="bgame", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("--level")
    p = sub.add_parser("info"); p.add_argument("game")
    p = sub.add_parser("play"); p.add_argument("game"); p.add_argument("--policy", default="random"); p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("certify"); p.add_argument("game"); p.add_argument("-K", type=int, default=30); p.add_argument("--options", nargs="*")
    p = sub.add_parser("validate"); p.add_argument("game"); p.add_argument("--timesteps", type=int, default=100_000)
    p = sub.add_parser("repair"); p.add_argument("game"); p.add_argument("operator"); p.add_argument("args", nargs="*"); p.add_argument("-K", type=int, default=30)
    p = sub.add_parser("agent"); p.add_argument("game"); p.add_argument("--model", default="local"); p.add_argument("--turns", type=int, default=40); p.add_argument("-K", type=int, default=3)
    p = sub.add_parser("gym"); p.add_argument("game")
    p = sub.add_parser("ui"); p.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        lad = _ladder()
        for k in games(a.level):
            print(f"{k:28s} {lad.get(k, {}).get('level', '?')}")
    elif a.cmd == "info":
        print(json.dumps(info(a.game), indent=1, ensure_ascii=False, default=str))
    elif a.cmd == "play":
        r = play(a.game, a.policy, a.seed)
        print(f"{a.game}: {r['steps']} plies, done={r['done']}, reward={r['reward']}")
        print("history:", r["history"])
    elif a.cmd == "certify":
        r = certify(a.game, K=a.K, options=a.options)
        print(f"{a.game}: {r['level']}  l3={r['l3']} l4={r['l4']} l5={r['l5']}  (K={a.K})")
        if r.get("counterexample"):
            print("counterexample:", json.dumps(r["counterexample"], default=str)[:600])
    elif a.cmd == "validate":
        r = validate(a.game, timesteps=a.timesteps)
        print(json.dumps(r, indent=1, default=str))
    elif a.cmd == "repair":
        kv = dict(x.split("=", 1) for x in a.args)
        kv = {k: (int(v) if v.lstrip("-").isdigit() else v) for k, v in kv.items()}
        r = repair(a.game, a.operator, K=a.K, **kv)
        print(f"{a.game} + {a.operator}{kv or ''}: {r['level']}  l3={r['l3']} l4={r['l4']} l5={r['l5']}")
    elif a.cmd == "agent":
        r = agent(a.game, model=a.model, K=a.K, turns=a.turns)
        print(json.dumps(r, indent=1, ensure_ascii=False) if r else "agent produced no record")
    elif a.cmd == "gym":
        (ROOT / "exports").mkdir(exist_ok=True)
        code = (f'"""Gymnasium env for {a.game} (bgame)."""\nimport sys; from pathlib import Path\n'
                f'sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))\nimport bgame\n'
                f'{a.game.title().replace("_", "")}Env = bgame.gym("{a.game}")\n')
        p = ROOT / "exports" / f"env_{a.game}.py"; p.write_text(code)
        print(f"wrote {p.relative_to(ROOT)}  (class {a.game.title().replace('_', '')}Env, use with sb3_contrib.MaskablePPO)")
    elif a.cmd == "ui":
        print(f"Ladder Workbench: http://127.0.0.1:{a.port}  (remote: ssh -N -L {a.port}:127.0.0.1:{a.port} <host>)")
        subprocess.run([str(ROOT / ".venv" / "bin" / "uvicorn"), "pipeline.workbench:app", "--host", "127.0.0.1",
                        "--port", str(a.port), "--app-dir", str(ROOT)], cwd=ROOT)


if __name__ == "__main__":
    _cli(sys.argv[1:])
