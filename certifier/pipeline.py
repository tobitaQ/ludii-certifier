"""
Stage1-3 pipeline.

Nim (default): run with  python src/pipeline.py
Connect Four:  run with  python src/pipeline.py --game connect_four

Stage1: Random vs Random, Greedy vs Greedy
Stage2: MCTS vs MCTS (balanced)
Stage3: 8 human profiles (profiled MCTS agents) vs each other

For each stage, compute the 5-dim feature vector and
track how fun weights evolve (PCA → equal-weight → convergence check).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from games.nim import Nim
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.mcts_agent import MCTSAgent
from agents.profiled_mcts_agent import ProfiledMCTSAgent
from runner import run_matches
from metrics import compute_feature_vector, fun_score, summary, METRIC_NAMES

N_GAMES = 500
game = Nim()

# ---------------------------------------------------------------------------
# Generic greedy agent factory
# ---------------------------------------------------------------------------

def make_greedy_agent(g):
    """Return the appropriate greedy agent for the given game."""
    from games.nim import Nim as _Nim
    if isinstance(g, _Nim):
        return GreedyAgent()
    if hasattr(g, "greedy_action"):
        from agents.connect_four_greedy_agent import ConnectFourGreedyAgent
        return ConnectFourGreedyAgent(g)
    from agents.generic_greedy_agent import GenericGreedyAgent
    return GenericGreedyAgent(g, seed=99)      # 2026-09-21: was RandomAgent(seed=99) for every translation


# ---------------------------------------------------------------------------
# Stage definitions (Nim — kept for backward compatibility)
# ---------------------------------------------------------------------------

def stage1_logs():
    logs = {}
    logs["rnd_vs_rnd"] = run_matches(game, [RandomAgent(0), RandomAgent(1)], N_GAMES)
    logs["grd_vs_rnd"] = run_matches(game, [GreedyAgent(), RandomAgent(2)], N_GAMES)
    logs["grd_vs_grd"] = run_matches(game, [GreedyAgent(), GreedyAgent()], N_GAMES)
    return logs


def stage2_logs():
    logs = {}
    mcts_bal = lambda s: MCTSAgent(game, iterations=600, c=1.414, seed=s)
    logs["mcts_vs_mcts"]  = run_matches(game, [mcts_bal(0), mcts_bal(1)], N_GAMES)
    logs["mcts_vs_rnd"]   = run_matches(game, [mcts_bal(2), RandomAgent(3)], N_GAMES)
    logs["mcts_vs_grd"]   = run_matches(game, [mcts_bal(4), GreedyAgent()], N_GAMES)
    return logs


PROFILES = {
    "P1": dict(game_experience=0.0, risk_tolerance=0.0, planning_depth="short",  attention_span=0.6),
    "P2": dict(game_experience=0.0, risk_tolerance=1.0, planning_depth="short",  attention_span=0.6),
    "P3": dict(game_experience=0.0, risk_tolerance=0.5, planning_depth="medium", attention_span=0.8),
    "P4": dict(game_experience=0.5, risk_tolerance=0.0, planning_depth="medium", attention_span=0.9),
    "P5": dict(game_experience=0.5, risk_tolerance=1.0, planning_depth="medium", attention_span=0.9),
    "P6": dict(game_experience=1.0, risk_tolerance=0.0, planning_depth="long",   attention_span=1.0),
    "P7": dict(game_experience=1.0, risk_tolerance=1.0, planning_depth="long",   attention_span=1.0),
    "P8": dict(game_experience=1.0, risk_tolerance=0.5, planning_depth="long",   attention_span=1.0),
}


def stage3_logs():
    logs = {}
    profile_items = list(PROFILES.items())
    for i, (name_a, params_a) in enumerate(profile_items):
        for name_b, params_b in profile_items[i+1:]:
            key = f"{name_a}_vs_{name_b}"
            agent_a = ProfiledMCTSAgent(game, **params_a, seed=i*10)
            agent_b = ProfiledMCTSAgent(game, **params_b, seed=i*10+1)
            logs[key] = run_matches(game, [agent_a, agent_b], N_GAMES)
    return logs


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix(all_logs: dict) -> tuple[np.ndarray, list[str]]:
    keys = list(all_logs.keys())
    X = np.array([compute_feature_vector(all_logs[k]) for k in keys])
    return X, keys


# ---------------------------------------------------------------------------
# Weight extraction via PCA
# ---------------------------------------------------------------------------

def extract_weights_pca(X: np.ndarray) -> np.ndarray:
    """
    Normalize features, run PCA, return PC1 as weight vector.
    PC1 captures the direction of maximum variance across matchups.
    Weights are absolute-valued and normalized to sum to 1.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(5, X.shape[0]))
    pca.fit(X_scaled)
    pc1 = np.abs(pca.components_[0])
    return pc1 / pc1.sum()


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_weights(weights: np.ndarray, stage: str, matchup_keys: list[str], feature_matrix: np.ndarray):
    import json, os
    os.makedirs("results", exist_ok=True)
    path = f"results/weights_{stage}.json"
    data = {
        "stage": stage,
        "weights": {n: float(v) for n, v in zip(METRIC_NAMES, weights)},
        "matchups": {
            k: {n: float(v) for n, v in zip(METRIC_NAMES, fv)}
            for k, fv in zip(matchup_keys, feature_matrix)
        }
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  → saved to {path}")


# ---------------------------------------------------------------------------
# Generic pipeline runner (game-agnostic Stage1-3)
# ---------------------------------------------------------------------------

def run_pipeline(
    g, game_label: str, n_games: int = N_GAMES,
    result_prefix: str = "", mcts_iters: int = 600,
    skip_stage3: bool = False, n_stage3_games: int | None = None,
):
    """
    Run Stage1-3 for any BoardGame g and save per-stage weights.
    result_prefix:  e.g. "c4_" → saves results/weights_c4_stage1.json etc.
    mcts_iters:     MCTS iterations for Stage2 agents; Stage3 ProfiledMCTS is
                    scaled proportionally.
    skip_stage3:    If True, skip Stage3 (fast mode for complex games).
    n_stage3_games: Override n_games for Stage3 only (e.g. 20 while n_games=50).
    """
    grd = make_greedy_agent(g)
    iters_scale = mcts_iters / 600.0  # scale ProfiledMCTS base iterations
    s3_games = n_stage3_games if n_stage3_games is not None else n_games

    def _stage1():
        logs = {}
        logs["rnd_vs_rnd"] = run_matches(g, [RandomAgent(0),  RandomAgent(1)],  n_games)
        logs["grd_vs_rnd"] = run_matches(g, [grd,              RandomAgent(2)],  n_games)
        logs["grd_vs_grd"] = run_matches(g, [grd,              make_greedy_agent(g)], n_games)
        return logs

    def _stage2():
        mk = lambda s: MCTSAgent(g, iterations=mcts_iters, c=1.414, seed=s)
        return {
            "mcts_vs_mcts": run_matches(g, [mk(0), mk(1)],        n_games),
            "mcts_vs_rnd":  run_matches(g, [mk(2), RandomAgent(3)], n_games),
            "mcts_vs_grd":  run_matches(g, [mk(4), make_greedy_agent(g)], n_games),
        }

    def _scale_profile(params: dict) -> dict:
        """Scale planning_depth base iterations by iters_scale."""
        p = dict(params)
        # Convert planning_depth to scaled iterations via a custom approach:
        # just pass params as-is; ProfiledMCTSAgent uses its own base_iter mapping.
        # We monkey-patch by converting long→medium or medium→short when scaling down.
        if iters_scale <= 0.34 and p.get("planning_depth") == "long":
            p["planning_depth"] = "short"
        elif iters_scale <= 0.67 and p.get("planning_depth") == "long":
            p["planning_depth"] = "medium"
        elif iters_scale <= 0.34 and p.get("planning_depth") == "medium":
            p["planning_depth"] = "short"
        return p

    def _stage3():
        logs = {}
        items = list(PROFILES.items())
        for i, (na, pa) in enumerate(items):
            for nb, pb in items[i + 1:]:
                a = ProfiledMCTSAgent(g, **_scale_profile(pa), seed=i * 10)
                b = ProfiledMCTSAgent(g, **_scale_profile(pb), seed=i * 10 + 1)
                logs[f"{na}_vs_{nb}"] = run_matches(g, [a, b], s3_games)
        return logs

    def _fv(logs):
        return {k: compute_feature_vector(v, g) for k, v in logs.items()}

    s3_label = (f"SKIPPED (--skip-stage3)" if skip_stage3
                else f"N={s3_games}/matchup")
    print("=" * 60)
    print(f"Stage1-3 pipeline: {game_label}")
    print(f"  Stage1+2: N={n_games}/matchup   Stage3: {s3_label}")
    print("=" * 60)

    s1 = _stage1()
    fv1 = _fv(s1)
    X1, k1 = np.array(list(fv1.values())), list(fv1.keys())
    w1 = extract_weights_pca(X1)
    print(f"\nStage1 weights: { {n: f'{v:.3f}' for n, v in zip(METRIC_NAMES, w1)} }")
    save_weights(w1, f"{result_prefix}stage1", k1, X1)

    s2 = _stage2()
    fv2 = _fv(s2)
    X2, k2 = np.array(list(fv2.values())), list(fv2.keys())
    X12 = np.vstack([X1, X2])
    w2 = extract_weights_pca(X12)
    print(f"Stage2 weights: { {n: f'{v:.3f}' for n, v in zip(METRIC_NAMES, w2)} }")
    print(f"  Stage1→2 cosine: {cosine_similarity(w1, w2):.4f}")
    save_weights(w2, f"{result_prefix}stage2", k1 + k2, X12)

    if skip_stage3:
        print(f"\nStage3: skipped  → using Stage2 weights as final weights")
        return w2

    s3 = _stage3()
    fv3 = _fv(s3)
    X3, k3 = np.array(list(fv3.values())), list(fv3.keys())
    X123 = np.vstack([X1, X2, X3])
    w3 = extract_weights_pca(X123)
    print(f"Stage3 weights: { {n: f'{v:.3f}' for n, v in zip(METRIC_NAMES, w3)} }")
    print(f"  Stage2→3 cosine: {cosine_similarity(w2, w3):.4f}")
    save_weights(w3, f"{result_prefix}stage3", k1 + k2 + k3, X123)

    sim_12 = cosine_similarity(w1, w2)
    sim_23 = cosine_similarity(w2, w3)
    if sim_12 > 0.95 and sim_23 > 0.95:
        print("  → [Pattern A] Converging")
    elif sim_23 < sim_12:
        print("  → [Pattern D] Later stages diverge")
    else:
        print("  → [Pattern B/C] Not yet converged")

    return w3


def _snake_to_camel(name: str) -> str:
    """'english_draughts' → 'EnglishDraughts'"""
    return "".join(w.capitalize() for w in name.split("_"))


def load_converted_game(module_name: str):
    """Load a game class from src/games/converted/<module_name>.py."""
    import importlib
    mod = importlib.import_module(f"games.converted.{module_name}")
    class_name = _snake_to_camel(module_name)
    cls = getattr(mod, class_name)
    return cls()


def list_converted_games() -> list[str]:
    """Return module names of all successfully converted games."""
    import glob
    converted_dir = os.path.join(os.path.dirname(__file__), "games", "converted")
    paths = glob.glob(os.path.join(converted_dir, "*.py"))
    names = []
    for p in sorted(paths):
        stem = os.path.basename(p)[:-3]
        if stem != "__init__":
            names.append(stem)
    return names


if __name__ == "__main__":
    import argparse

    BUILTIN_GAMES = ["nim", "connect_four", "othello", "kalah", "checkers"]
    converted_games = list_converted_games()

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="nim",
                        help="Game name: nim/connect_four/othello/kalah/checkers or any converted game (e.g. hex, pente, y)")
    parser.add_argument("--list-games",     action="store_true", help="List available converted games and exit")
    parser.add_argument("--n-games",        type=int, default=N_GAMES)
    parser.add_argument("--mcts-iters",     type=int, default=600,
                        help="MCTS iterations for Stage2/3 agents (default 600; use 50-200 for complex games)")
    parser.add_argument("--skip-stage3",    action="store_true",
                        help="Skip Stage3 (fast mode for complex games; saves Stage1+2 only)")
    parser.add_argument("--n-stage3-games", type=int, default=None,
                        help="Override n_games for Stage3 only (e.g. --n-games 100 --n-stage3-games 20)")
    cliargs = parser.parse_args()

    if cliargs.list_games:
        print("Built-in games:", ", ".join(BUILTIN_GAMES))
        print("Converted games:", ", ".join(converted_games))
        sys.exit(0)

    game_map = {
        "connect_four": ("Connect Four", "c4_",  lambda: __import__("games.connect_four", fromlist=["ConnectFour"]).ConnectFour()),
        "othello":      ("Othello",      "oth_", lambda: __import__("games.othello",      fromlist=["Othello"]).Othello()),
        "kalah":        ("Kalah",        "kal_", lambda: __import__("games.kalah",         fromlist=["Kalah"]).Kalah()),
        "checkers":     ("Checkers",     "chk_", lambda: __import__("games.checkers",      fromlist=["Checkers"]).Checkers()),
    }
    if cliargs.game in game_map:
        label, prefix, factory = game_map[cliargs.game]
        run_pipeline(factory(), label,
                     n_games=cliargs.n_games, result_prefix=prefix,
                     mcts_iters=cliargs.mcts_iters,
                     skip_stage3=cliargs.skip_stage3,
                     n_stage3_games=cliargs.n_stage3_games)
        sys.exit(0)

    # Converted game: load dynamically from src/games/converted/
    if cliargs.game in converted_games:
        g = load_converted_game(cliargs.game)
        label = _snake_to_camel(cliargs.game)
        prefix = cliargs.game + "_"
        run_pipeline(g, label,
                     n_games=cliargs.n_games, result_prefix=prefix,
                     mcts_iters=cliargs.mcts_iters,
                     skip_stage3=cliargs.skip_stage3,
                     n_stage3_games=cliargs.n_stage3_games)
        sys.exit(0)
    print("=" * 60)
    print("Stage1: Random / Greedy baselines")
    print("=" * 60)
    s1 = stage1_logs()
    X1, keys1 = build_feature_matrix(s1)
    w1 = extract_weights_pca(X1)

    for k, fv in zip(keys1, X1):
        print(f"  {k:<22}", "  ".join(f"{METRIC_NAMES[i]}={fv[i]:.3f}" for i in range(5)))
    print(f"\n  PCA weights: { {n: f'{v:.3f}' for n, v in zip(METRIC_NAMES, w1)} }")
    save_weights(w1, "stage1", keys1, X1)

    print()
    print("=" * 60)
    print("Stage2: MCTS agents")
    print("=" * 60)
    s2 = stage2_logs()
    X2, keys2 = build_feature_matrix(s2)
    X12 = np.vstack([X1, X2])
    w2 = extract_weights_pca(X12)

    for k, fv in zip(keys2, X2):
        print(f"  {k:<22}", "  ".join(f"{METRIC_NAMES[i]}={fv[i]:.3f}" for i in range(5)))
    print(f"\n  PCA weights (Stage1+2): { {n: f'{v:.3f}' for n, v in zip(METRIC_NAMES, w2)} }")
    print(f"  Cosine similarity to Stage1: {cosine_similarity(w1, w2):.4f}")
    save_weights(w2, "stage2", keys1 + keys2, X12)

    print()
    print("=" * 60)
    print("Stage3: Profiled MCTS (8 human profiles)")
    print("=" * 60)
    s3 = stage3_logs()
    X3, keys3 = build_feature_matrix(s3)
    X123 = np.vstack([X1, X2, X3])
    w3 = extract_weights_pca(X123)

    print(f"  {len(keys3)} matchups across 8 profiles")
    for k, fv in zip(keys3, X3):
        print(f"  {k:<22}", "  ".join(f"{METRIC_NAMES[i]}={fv[i]:.3f}" for i in range(5)))
    print(f"\n  PCA weights (Stage1+2+3): { {n: f'{v:.3f}' for n, v in zip(METRIC_NAMES, w3)} }")
    print(f"  Cosine similarity Stage1→Stage2: {cosine_similarity(w1, w2):.4f}")
    print(f"  Cosine similarity Stage2→Stage3: {cosine_similarity(w2, w3):.4f}")
    save_weights(w3, "stage3", keys1 + keys2 + keys3, X123)

    print()
    print("=" * 60)
    print("Convergence summary")
    print("=" * 60)
    stage_weights = [("Stage1", w1), ("Stage1+2", w2), ("Stage1+2+3", w3)]
    print(f"  {'Stage':<15}", "  ".join(f"{n:>9}" for n in METRIC_NAMES))
    for label, w in stage_weights:
        print(f"  {label:<15}", "  ".join(f"{v:>9.3f}" for v in w))

    sim_12 = cosine_similarity(w1, w2)
    sim_23 = cosine_similarity(w2, w3)
    print(f"\n  Stage1→2 similarity: {sim_12:.4f}")
    print(f"  Stage2→3 similarity: {sim_23:.4f}")
    if sim_12 > 0.95 and sim_23 > 0.95:
        print("  → [Pattern A] Converging — weights are stable across stages")
    elif sim_23 < sim_12:
        print("  → [Pattern D] Later stages diverge — possible overfitting to profile structure")
    else:
        print("  → [Pattern B/C] Not yet converged — more stages needed")
