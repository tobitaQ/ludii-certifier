"""
Stage5: PPO reinforcement learning for fun optimization.

Workflow:
  1. GameFunEnv: self-play with Stage4 fun-weighted terminal reward (any BoardGame)
  2. Train PPO (128 parallel envs, 500K timesteps by default)
  3. Evaluate trained agent (500 games) vs. random baseline (500 games)
  4. For each metric: compute Cohen's d (effect size of RL vs. random)
  5. Update weights toward metrics the RL agent reliably improved
  6. Save results/weights_stage5.json

Run:
  python src/stage5_rl.py
  python src/stage5_rl.py --timesteps 200000 --n-envs 64  # faster
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from collections import Counter
from scipy.stats import mannwhitneyu

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from pipeline import cosine_similarity
from metrics import METRIC_NAMES

# Legacy constant kept for backward compatibility (stage6_profiles.py imports it)
MAX_HEAP = 5


def _default_game():
    from games.nim import Nim
    return Nim()


# ---------------------------------------------------------------------------
# Per-game metric computation (single-game, game-agnostic)
# ---------------------------------------------------------------------------

def compute_single_game_metrics(turns_data, game=None, include_extra=False,
                                perspective=None, players=None):
    """Record-grain metric vector.  Since 2026-09-10 this delegates to
    record_metrics.compute_single_game_metrics (one implementation shared with the
    Stage 8 predictor path, one-player series by default).  The historical
    side-to-move behaviour is perspective="mover" (env RECORD_PERSPECTIVE=mover)."""
    from record_metrics import compute_single_game_metrics as _impl
    if perspective is None:
        perspective = os.environ.get("RECORD_PERSPECTIVE", "one-player")
    return _impl(turns_data, game, include_extra=include_extra,
                 perspective=perspective, players=players)

def compute_fun_reward(turns_data, weights, game=None, reward_metric=None, include_extra=False):
    """Terminal reward over the normalized metric vector.

    reward_metric is None  → weighted fun score   (Σ wᵢ·mᵢ, the Stage4-weight objective)
    reward_metric is an int → that single metric   (mᵢ alone, for P1 intervention runs)
    """
    if game is None:
        game = _default_game()
    m          = compute_single_game_metrics(turns_data, game, include_extra=include_extra)
    norms      = game.fun_metric_norms()
    if include_extra and len(game.extra_metric_norms()):
        norms = np.concatenate([norms, game.extra_metric_norms()])
    normalized = np.clip(m / norms, 0.0, 1.0)
    if reward_metric is not None:
        return float(normalized[reward_metric])
    return float(weights @ normalized)


# ---------------------------------------------------------------------------
# Gymnasium Environment (game-agnostic)
# ---------------------------------------------------------------------------

class GameFunEnv(gym.Env):
    """
    Self-play env for any BoardGame. One policy plays both sides.
    Terminal reward = fun_score(game_trajectory, weights).
    """

    metadata = {}

    def __init__(self, game, weights, reward_metric=None, include_extra=False, dense=False):
        super().__init__()
        self.game          = game
        self.weights       = np.array(weights, dtype=np.float32)
        self.reward_metric = reward_metric  # None → fun score; int → single-metric (P1)
        self.include_extra = include_extra  # True → metric vector includes game-specific metrics
        self.dense         = dense          # True → per-move move_reward (dense intervention)

        obs_dim = game.obs_dim()
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32),
            high= np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(game.action_count())

        self.state      = game.reset()
        self.turns_data = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state      = self.game.reset()
        self.turns_data = []
        return self.game.obs_vector(self.state), {}

    def step(self, action):
        legal   = self.game.legal_actions(self.state)
        decoded = self.game.decode_action(int(action))

        if decoded not in legal:
            decoded = legal[self.np_random.integers(len(legal))]

        self.turns_data.append((self.state, decoded, legal))
        # Dense intervention: reward this move's metric contribution every step.
        step_r = self.game.move_reward(self.state, decoded) if self.dense else 0.0
        next_state, _, done = self.game.step(self.state, decoded)
        self.state = next_state

        if done:
            reward = step_r if self.dense else compute_fun_reward(
                self.turns_data, self.weights, self.game,
                reward_metric=self.reward_metric, include_extra=self.include_extra)
            return self.game.obs_vector(self.state), reward, True, False, {}

        return self.game.obs_vector(self.state), step_r, False, False, {}

    def action_masks(self) -> np.ndarray:
        """Boolean mask of legal actions for MaskablePPO — without it the policy
        wastes ~92% of moves on illegal actions (→ random fallback → ~random play)."""
        mask = np.zeros(self.game.action_count(), dtype=bool)
        for a in self.game.legal_actions(self.state):
            mask[self.game.encode_action(a)] = True
        return mask


def NimFunEnv(weights):
    """Backward-compatible factory: returns GameFunEnv(Nim(), weights)."""
    return GameFunEnv(_default_game(), weights)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class _ProgressCB(BaseCallback):
    def __init__(self, total):
        super().__init__()
        self.total = total
        self._last = 0

    def _on_step(self):
        if self.num_timesteps - self._last >= max(self.total // 10, 1):
            self._last = self.num_timesteps
            pct = 100 * self.num_timesteps / self.total
            print(f"    {self.num_timesteps:>8}/{self.total}  ({pct:.0f}%)")
        return True


def train_ppo(weights, n_envs=128, timesteps=500_000, device="auto", game=None,
              reward_metric=None, include_extra=False, dense=False, use_masking=False,
              seed=None):
    if game is None:
        game = _default_game()

    def _make(seed):
        def _init():
            e = GameFunEnv(game, weights, reward_metric=reward_metric,
                           include_extra=include_extra, dense=dense)
            e.reset(seed=seed)
            return e
        return _init

    base = 0 if seed is None else 1000 * seed
    vec_env = DummyVecEnv([_make(base + i) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env)

    ppo_kwargs = dict(device=device, verbose=0, n_steps=512, batch_size=256,
                      n_epochs=4, learning_rate=3e-4, gamma=0.99, ent_coef=0.01,
                      seed=seed)
    if use_masking:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO("MlpPolicy", vec_env, **ppo_kwargs)
    else:
        model = PPO("MlpPolicy", vec_env, **ppo_kwargs)

    print(f"  Training {'MaskablePPO' if use_masking else 'PPO'}  "
          f"envs={n_envs}  steps={timesteps}  device={device}")
    model.learn(total_timesteps=timesteps, callback=_ProgressCB(timesteps))
    vec_env.close()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def collect_games(policy, weights, n_games=500, deterministic=False, game=None,
                  reward_metric=None, include_extra=False):
    """
    Run n_games, return metrics_array (n_games, 5) and rewards_array (n_games,).
    policy=None → random agent.
    rewards_array reflects the training objective (single metric if reward_metric set).
    """
    if game is None:
        game = _default_game()
    env = GameFunEnv(game, weights, reward_metric=reward_metric, include_extra=include_extra)
    metrics_list, rewards_list = [], []

    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        while not done:
            if policy is None:
                legal = game.legal_actions(env.state)
                legal_enc = [game.encode_action(a) for a in legal]
                action = int(env.np_random.choice(legal_enc))
            else:
                action, _ = policy.predict(obs, deterministic=deterministic)
                action = int(action)
            obs, _, done, _, _ = env.step(action)

        metrics_list.append(compute_single_game_metrics(env.turns_data, game))
        rewards_list.append(compute_fun_reward(env.turns_data, weights, game,
                                               reward_metric=reward_metric,
                                               include_extra=include_extra))

    return np.array(metrics_list), np.array(rewards_list)


# ---------------------------------------------------------------------------
# Weight update
# ---------------------------------------------------------------------------

def compute_weights_from_comparison(rl_metrics, baseline_metrics, stage4_weights, alpha=0.3):
    """
    Update weights toward metrics the RL agent reliably improved over random.
    Uses Cohen's d (effect size) and Mann-Whitney U (significance).
    """
    effect_sizes = []
    stat_results = {}

    for i, name in enumerate(METRIC_NAMES):
        rl_m = rl_metrics[:, i]
        base_m = baseline_metrics[:, i]

        pooled_std = np.sqrt((np.var(rl_m) + np.var(base_m)) / 2.0)
        cohen_d = abs(float(np.mean(rl_m)) - float(np.mean(base_m))) / (pooled_std + 1e-8)

        stat, pval = mannwhitneyu(rl_m, base_m, alternative="two-sided")

        effect_sizes.append(cohen_d)
        stat_results[name] = {
            "rl_mean":  float(np.mean(rl_m)),
            "base_mean": float(np.mean(base_m)),
            "cohen_d":  float(cohen_d),
            "pval":     float(pval),
        }
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "n.s."))
        print(f"    {name:<12}: RL={np.mean(rl_m):.3f}  Base={np.mean(base_m):.3f}"
              f"  d={cohen_d:.3f}  p={pval:.4f} {sig}")

    effect_sizes = np.array(effect_sizes)
    total = effect_sizes.sum()
    improvement_weights = effect_sizes / total if total > 1e-8 else np.ones(5) / 5.0

    # Blend Stage4 weights (70%) with RL-discovered weights (30%)
    new_weights = (1.0 - alpha) * stage4_weights + alpha * improvement_weights
    new_weights /= new_weights.sum()

    return new_weights, stat_results


# ---------------------------------------------------------------------------
# Device selection helper
# ---------------------------------------------------------------------------

# SB3 MLP policies run faster on CPU for small observation spaces due to
# GPU data-transfer overhead. Threshold: obs_dim <= 32 → prefer CPU.
_MLP_CPU_THRESHOLD = 32


def resolve_device(requested: str, obs_dim: int) -> tuple[str, str]:
    """
    Resolve actual device from --device arg and observation space size.
    Returns (device, reason_str).
    'auto' picks CPU for small obs spaces, CUDA for large ones.
    """
    if requested == "cpu":
        return "cpu", "explicitly set to cpu"
    if requested == "cuda":
        if obs_dim <= _MLP_CPU_THRESHOLD:
            note = (
                f"WARNING: obs_dim={obs_dim} is small. "
                f"MLP policies are often FASTER on CPU (SB3 #1245). "
                f"Consider --device cpu or --device auto."
            )
        else:
            note = f"obs_dim={obs_dim} — GPU suited for large obs space"
        return "cuda", note
    # auto
    if obs_dim <= _MLP_CPU_THRESHOLD:
        return "cpu", (
            f"auto: obs_dim={obs_dim} <= {_MLP_CPU_THRESHOLD} "
            f"→ CPU chosen (MLP faster on CPU for small obs; use --device cuda to override)"
        )
    return "cuda", (
        f"auto: obs_dim={obs_dim} > {_MLP_CPU_THRESHOLD} "
        f"→ CUDA chosen (large obs space benefits from GPU; use --device cpu to override)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _make_game(game_name: str):
    if game_name == "connect_four":
        from games.connect_four import ConnectFour
        return ConnectFour()
    if game_name == "othello":
        from games.othello import Othello
        return Othello()
    if game_name == "kalah":
        from games.kalah import Kalah
        return Kalah()
    if game_name == "checkers":
        from games.checkers import Checkers
        return Checkers()
    if game_name.startswith("converted:"):
        # UIST: any gate-1-certified converted game, e.g. "converted:hex"
        import importlib, inspect, json
        from pathlib import Path as _P
        key = game_name.split(':', 1)[1]
        # honour the repair-operator ledger: the playable version of a game is its ops-wrapped module
        ledger_p = _P(__file__).resolve().parents[1] / "results" / "operators_applied.json"
        ops = json.load(open(ledger_p)) if ledger_p.exists() else {}
        ops_file = _P(__file__).resolve().parent / "games" / "ops" / f"{key}.py"
        mod = importlib.import_module(f"games.ops.{key}" if (key in ops or ops_file.exists()) else f"games.converted.{key}")
        for n, o in inspect.getmembers(mod, inspect.isclass):
            if n.startswith("_") or n == "BoardGame" or getattr(o, "__abstractmethods__", None):
                continue
            if all(hasattr(o, m) for m in ("reset", "legal_actions", "step", "current_player")):
                return o()
        raise ValueError(f"no BoardGame class in {game_name}")
    return _default_game()


def _result_prefix(game_name: str) -> str:
    prefixes = {"connect_four": "c4_", "othello": "oth_", "kalah": "kal_", "checkers": "chk_"}
    if game_name.startswith("converted:"):
        return "conv_"
    return prefixes.get(game_name, "")


def main():
    parser = argparse.ArgumentParser(description="Stage5 PPO fun optimization")
    parser.add_argument("--game",      choices=["nim", "connect_four", "othello", "kalah", "checkers"], default="nim")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs",    type=int, default=128)
    parser.add_argument("--n-eval",    type=int, default=500,
                        help="Games for evaluation/baseline (each)")
    parser.add_argument("--device",    type=str, default="auto",
                        help="'cpu' | 'cuda' | 'auto' (default). "
                             "auto picks CPU for small obs spaces (MLP is faster), "
                             "CUDA for large obs spaces.")
    parser.add_argument("--alpha",     type=float, default=0.3,
                        help="Blend factor toward RL-discovered weights (0–1)")
    parser.add_argument("--reward-metric", choices=METRIC_NAMES, default=None,
                        help="P1 intervention mode: reward = this single metric mᵢ(τ) "
                             "instead of the Stage4-weighted fun score. "
                             "Omit for the default fun-score reward.")
    args = parser.parse_args()

    game   = _make_game(args.game)
    prefix = _result_prefix(args.game)

    # P1: single-metric intervention. None → default fun-weighted reward.
    reward_metric = METRIC_NAMES.index(args.reward_metric) if args.reward_metric else None

    obs_dim = game.obs_dim()
    actual_device, device_reason = resolve_device(args.device, obs_dim)

    print("=" * 60)
    print(f"Stage5: PPO fun optimization  [{game.game_name()}]")
    print(f"  timesteps : {args.timesteps}")
    print(f"  n_envs    : {args.n_envs}")
    print(f"  n_eval    : {args.n_eval}")
    print(f"  device    : {actual_device}  [{device_reason}]")
    print(f"  alpha     : {args.alpha}")
    print(f"  reward    : {args.reward_metric or 'fun-score (Σ wᵢ·mᵢ)'}")
    print("=" * 60)

    # Load Stage4 weights
    s4_path = f"results/weights_{prefix}stage4.json"
    with open(s4_path) as f:
        s4 = json.load(f)
    w4 = np.array([s4["weights"][n] for n in METRIC_NAMES])
    print(f"\nLoaded Stage4 weights from {s4_path}")
    for n, v in zip(METRIC_NAMES, w4):
        print(f"  {n:<12}: {v:.4f}")

    # Train PPO
    print("\n[1] Training PPO agent...")
    model = train_ppo(w4, n_envs=args.n_envs, timesteps=args.timesteps,
                      device=actual_device, game=game, reward_metric=reward_metric)

    # Evaluate
    print(f"\n[2] Evaluating trained agent ({args.n_eval} games)...")
    rl_metrics, rl_rewards = collect_games(model, w4, n_games=args.n_eval,
                                           deterministic=False, game=game,
                                           reward_metric=reward_metric)
    print(f"  RL agent  — avg reward: {rl_rewards.mean():.4f} ± {rl_rewards.std():.4f}")

    print(f"\n[3] Running random baseline ({args.n_eval} games)...")
    base_metrics, base_rewards = collect_games(None, w4, n_games=args.n_eval, game=game,
                                               reward_metric=reward_metric)
    print(f"  Random    — avg reward: {base_rewards.mean():.4f} ± {base_rewards.std():.4f}")

    # Weight update
    print("\n[4] Computing weight update (Cohen's d: RL vs. random):")
    w5, stat_results = compute_weights_from_comparison(rl_metrics, base_metrics, w4, alpha=args.alpha)

    print(f"\n  Stage5 weights:")
    for name, v in zip(METRIC_NAMES, w5):
        print(f"    {name:<12}: {v:.4f}")

    # Compare with Stage4 (weight-convergence framing — meaningful for fun-score mode)
    sim = cosine_similarity(w4, w5)
    print(f"\n  Cosine similarity Stage4→Stage5: {sim:.4f}")
    if sim > 0.95:
        print("  → [Pattern A] Converging — RL confirms LLM weights")
    elif sim < 0.80:
        print("  → [Pattern D] Diverging — RL emphasizes different metrics")
    else:
        print("  → [Pattern B/C] Partial agreement")

    # P1 intervention readout: did training on the targeted metric actually raise it?
    if args.reward_metric:
        tgt = stat_results[args.reward_metric]
        print(f"\n  [P1 intervention] target='{args.reward_metric}'  "
              f"d={tgt['cohen_d']:.3f}  p={tgt['pval']:.4f}  "
              f"(RL={tgt['rl_mean']:.3f} vs Base={tgt['base_mean']:.3f})")

    # Save — one file per (game × reward) so the 25 P1 conditions don't overwrite
    os.makedirs("results", exist_ok=True)
    suffix = f"_rm-{args.reward_metric}" if args.reward_metric else ""
    out = {
        "stage":    f"{prefix}stage5",
        "game":     game.game_name(),
        "reward_metric": args.reward_metric,   # None = fun-weighted score
        "timesteps": args.timesteps,
        "n_envs":   args.n_envs,
        "n_eval":   args.n_eval,
        "alpha":    args.alpha,
        "weights":  {n: float(v) for n, v in zip(METRIC_NAMES, w5)},
        "rl_vs_random": stat_results,
        "rl_reward_mean":   float(rl_rewards.mean()),
        "base_reward_mean": float(base_rewards.mean()),
        "cosine_sim_to_stage4": float(sim),
    }

    path = f"results/weights_{prefix}stage5{suffix}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  → saved to {path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
