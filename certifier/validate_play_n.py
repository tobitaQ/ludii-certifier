"""The game-property certificate for N seats, not two.

Axis A never cared how many players a game has -- the ladder compares legal-move sets, boards, termination
and the winner, and the five games in the corpus with three or four seats all certify at L5. Axis B did
care, for one reason: its band is written on the FIRST SEAT's share of decided games, which is a
two-player sentence. So the second certificate returned N/A for those games, the cross table carried an
N/A column, and the paper had to say "outside the two-player check" -- an asymmetry in the design, not a
fact about the games. (Ludii has 41 games with three or more seats; the corpus caught five of them, and one,
shred, slipped through the N/A guard and was read with a two-player harness for a year.)

The generalisation is the obvious one and it reduces to the current rule at N=2. Neutral is 1/N rather
than 1/2, and the half-width scales so that two seats reproduce the band we have always used:

    every seat's share of decided games within  1/N +- 0.35/(N-1)      (N=2: 0.15 .. 0.85, unchanged)

The other three conditions never mentioned seats and are kept verbatim: every game terminates inside the
ply cap, fewer than half are drawn, games last at least six plies.

The band is not being tuned here. 0.35 is the number axis B has used since the first iteration; the only
new decision is how it scales, and it is fixed before the games are measured rather than after.

Usage:  .venv/bin/python pipeline/validate_play_n.py [--games 40] [--cap 1500] [game ...]
        (no game list: every corpus game with more than two seats)
Output: results/validate_play_nplayer.json
"""
import sys, json, random, argparse, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "pipeline"))
R = ROOT / "results"
HALF_WIDTH_AT_2 = 0.35          # the band axis B has used from the start; see validate_play.py


def band(n_players):
    """(low, high) for one seat's share of decided games. N=2 gives the historical 0.15--0.85."""
    # Rounded: the band is a declared threshold and must not turn on binary floating point.
    # 1/2 - 0.35 is 0.15000000000000002, so a share of exactly 0.15 -- the stated edge -- failed
    # `low <= share`. No game in this corpus sits on the edge, but one in Ludii's catalogue does.
    neutral = 1.0 / n_players
    half = HALF_WIDTH_AT_2 / (n_players - 1)
    return round(neutral - half, 12), round(neutral + half, 12)


def play_py(game, seed, cap):
    """One random game in the translation. Returns (plies, outcome) where outcome is 'mover-won',
    'draw', 'mover-lost', 'deadlock' or 'cap'.

    The translation cannot name the winner when the mover loses. `step` returns one scalar reward and
    base.py defines it from the mover's side, so on three or more seats a -1 says only "somebody else
    won" -- which is the same limit the ladder's L5 check runs into (`_winner_verdict` passes a mover
    loss iff Ludii's winner is *another* player, without checking which). Read as a two-player result it
    turns every such game into a draw: that is how ashere came back "100% drawn" when in fact all 40
    games were decided. So the plies and the termination come from here, and the winner does not."""
    rng = random.Random(seed)
    st = game.reset()
    for n in range(cap):
        la = game.legal_actions(st)
        if not la:
            return n, "deadlock"
        st, rw, done = game.step(st, la[rng.randrange(len(la))])
        if done:
            return n + 1, "mover-won" if rw > 0 else "draw" if rw == 0 else "mover-lost"
    return cap, "cap"


def ludii_game(key):
    import ludii_oracle as lo
    conv = {c["module"].split(".")[-1]: c for c in json.load(open(R / "conversion_results.json"))
            if c.get("module") and c.get("lud_path")}
    return lo.load_game(Path(conv[key]["lud_path"]).name)


def play_ludii(game, seed, cap):
    """The same random game in the reference, which does name the winner (trial.ranking()).

    Legitimate because the winner is the one quantity the translation withholds and because all five
    multi-seat games certify at L5 -- the two engines agree on legal moves, on the board after every
    move, on the terminating ply and on the winner for K=30 seeds, so the distribution being measured
    is the same object. It is the reference, not the translation, that is sampled here, and the paper
    says so. Returns (winner seat 0-based, or -1 for a draw, plies, flag)."""
    import ludii_oracle as lo
    trial = lo.Trial(game); ctx = lo.Context(game, trial); game.start(ctx)
    rng = random.Random(seed)
    for n in range(cap):
        if trial.over():
            return lo._ludii_winner(game, trial) - 1, n, None
        mv = game.moves(ctx).moves()
        if mv.isEmpty():
            return -1, n, "no-moves"
        game.apply(ctx, mv.get(rng.randrange(mv.size())))
    return -1, cap, "cap"


def seats():
    """Seat counts as the reference reports them (pipeline/player_counts.py). Not the .lud text: the count can
    live in a named-seat list or in an option, and reading it out of the file with a regex put shred at
    three seats when its default ruleset has two."""
    return {k: v for k, v in json.load(open(R / "player_counts.json")).items() if isinstance(v, int)}


def row_of(r):
    """The cross table's row, by the same precedence the two-player certificate uses: a game that does
    not finish is nonterminating whatever its shares look like, then the band."""
    if r["capped"] or r["no_moves"]:
        return "nonterminating" if r["capped"] else "deadlock"
    if r["mean_plies"] < 6:
        return "degenerate"
    return "plays" if r["plays"] else "degenerate"


def measure(key, games, cap):
    import bgame
    n = seats()[key]
    g = bgame.load(key)
    lg = ludii_game(key)
    wins = [0] * n
    draws = capped = nomoves = 0
    lens = []
    py_lens, py_capped, py_term = [], 0, 0
    t0 = time.time()
    for s in range(games):
        w, plies, f = play_ludii(lg, 1000 + s, cap)
        lens.append(plies)
        if f == "cap":
            capped += 1
        elif f == "no-moves":
            nomoves += 1
        if w < 0:
            draws += 1
        else:
            wins[w] += 1
        # the translation is run on the same seeds as a cross-check on the two quantities it can report:
        # does the game end, and after how many plies
        pl, outc = play_py(g, 1000 + s, cap)
        py_lens.append(pl)
        py_capped += outc == "cap"
        py_term += outc in ("mover-won", "draw", "mover-lost")
    dec = sum(wins)
    lo, hi = band(n)
    shares = [round(w / dec, 3) if dec else None for w in wins]
    ok_band = bool(dec) and all(lo <= s <= hi for s in shares)
    out = {"players": n, "games": games, "winner_from": "ludii-ranking", "wins": wins, "shares": shares,
           "band": [round(lo, 3), round(hi, 3)], "in_band": ok_band,
           "draw_rate": round(draws / games, 3), "mean_plies": round(sum(lens) / games, 1),
           "capped": capped, "no_moves": nomoves,
           "python_mean_plies": round(sum(py_lens) / games, 1), "python_capped": py_capped,
           "python_terminated": py_term, "seconds": round(time.time() - t0, 1)}
    out["plays"] = bool(ok_band and capped == 0 and nomoves == 0
                        and out["draw_rate"] < 0.5 and out["mean_plies"] >= 6)
    out["row"] = row_of(out)
    # the two engines are certified to agree; if their lengths or their termination counts diverge here,
    # the sample is not measuring one game and the record says so rather than averaging the two
    out["engines_agree"] = bool(py_capped == capped and abs(out["python_mean_plies"] - out["mean_plies"])
                                <= 0.25 * max(1.0, out["mean_plies"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("keys", nargs="*")
    a = ap.parse_args()
    sea = seats()
    keys = a.keys or sorted(k for k, v in sea.items() if v > 2)
    print(f"{len(keys)} games with more than two seats; {a.games} random games each, {a.cap}-ply cap")
    out = json.load(open(R / "validate_play_nplayer.json")) if (R / "validate_play_nplayer.json").exists() else {}
    out = {k: v for k, v in out.items() if sea.get(k, 0) > 2}      # drop anything the reference calls two-seat
    for k in keys:
        try:
            out[k] = measure(k, a.games, a.cap)
        except Exception as e:
            out[k] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        r = out[k]
        print(f"  {k:20s} N={r.get('players')} shares={r.get('shares')} band={r.get('band')} "
              f"draws={r.get('draw_rate')} len={r.get('mean_plies')}/py{r.get('python_mean_plies')} "
              f"capped={r.get('capped')}/py{r.get('python_capped')} agree={r.get('engines_agree')} "
              f"-> {r.get('row')} plays={r.get('plays')} {r.get('error','')}", flush=True)
        (R / "validate_play_nplayer.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    good = sum(1 for v in out.values() if v.get("plays"))
    print(f"\nplays: {good} of {len(out)}")


if __name__ == "__main__":
    main()
