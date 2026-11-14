"""If a translation were wrong, would the ladder say so? The false-positive side, measured.

Every certifier defect found so far has been a FALSE NEGATIVE: a correct translation the ladder could not
compare (a hidden piece's visibility, a string field, the wrong ruleset, a tie-break that picked without
verifying). Those do not threaten the paper, which states plainly that the certifier is a lower bound.
What would threaten it is a false positive -- an \\lvl5 awarded to a translation that does not reproduce
the reference -- and that rate has never been measured. A reviewer is entitled to ask for it before
believing 294 of 305.

So: take games the ladder certifies at \\lvl5, break them on purpose, one rung at a time, and see whether
the ladder notices. The mutations are wrappers, not edits, so the translation is untouched and each
mutant is exactly one defect away from a certified game:

  drop_move     \\lvl3   one legal action is withheld whenever there are at least two
  perturb_board \\lvl4   the move is played, then two board cells are exchanged
  early_end     \\lvl5   the game is declared over at a fixed ply, with the mover winning
  flip_winner   \\lvl5   the terminal reward is negated (pipeline/operators.py's override_winner)

A mutant that still certifies at \\lvl5 is a hole in the certifier, named and reproducible. The existing
mutation test covers one declaration (swap the declared fields and require the certification to fail);
this generalises it to the rungs themselves.

Usage:  .venv/bin/python pipeline/mutation_test.py [--games 30] [--episodes 30] [--time-limit 120]
Output: results/mutation_test.json
"""
import sys, json, random, time, argparse, importlib, subprocess, signal
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "pipeline"))
R = ROOT / "results"
OUT = R / "mutation_test.json"

# A mutation that never fires is not a hole in the certifier, it is a mutation that was never applied:
# early_end@25 on tic-tac-toe, whose games are over by ply 9, leaves the translation untouched and the
# certification is rightly \lvl5. Each wrapper therefore counts the occasions on which it actually
# changed what the game did, and a mutant with a count of zero is reported as NOT APPLICABLE -- outside
# the denominator -- rather than as an undetected defect.
FIRED = [0]


def drop_move(cls):
    from base import BoardGame

    class M(BoardGame):
        __mutation__ = "drop_move"
        __wrapped_class__ = cls

        def __init__(self):
            self.inner = cls()

        def __getattr__(self, n):
            if n == "inner":
                raise AttributeError(n)
            return getattr(self.inner, n)

        def reset(self):
            return self.inner.reset()

        def current_player(self, s):
            return self.inner.current_player(s)

        def legal_actions(self, s):
            la = list(self.inner.legal_actions(s))
            if len(la) > 1:
                FIRED[0] += 1
                return la[:-1]
            return la

        def step(self, s, a):
            return self.inner.step(s, a)
    M.__name__ = cls.__name__
    return M


def _swap_in_board(st):
    """Return `st` with two cells of its board exchanged, one occupied and one empty, or None.

    The board is not always the state. Most translations carry (board, side-to-move, ...) or something
    deeper, so a mutation that only understood a flat tuple of ints fired on one of the first six games
    tried and called the other five not applicable -- which would have left the \\lvl4 rung unmeasured
    while looking like a clean result. So: find the longest tuple of ints anywhere in the state and swap
    inside that, rebuilding the surrounding structure unchanged."""
    best = [None, 0]        # path to the longest int-tuple, and its length

    def walk(node, path):
        if isinstance(node, tuple):
            if node and all(isinstance(x, int) and not isinstance(x, bool) for x in node):
                if len(node) > best[1]:
                    best[0], best[1] = path, len(node)
                return
            for i, x in enumerate(node):
                walk(x, path + (i,))
    walk(st, ())
    if best[0] is None or best[1] < 4:
        return None

    def get(node, path):
        for i in path:
            node = node[i]
        return node

    def put(node, path, val):
        if not path:
            return val
        l = list(node)
        l[path[0]] = put(node[path[0]], path[1:], val)
        return tuple(l)

    board = get(st, best[0])
    occ = [i for i, v in enumerate(board) if v]
    emp = [i for i, v in enumerate(board) if not v]
    if not occ or not emp:
        return None
    l = list(board)
    i, j = occ[0], emp[0]
    l[i], l[j] = l[j], l[i]
    return put(st, best[0], tuple(l))


def perturb_board(cls):
    """Play the move, then exchange one occupied and one empty cell of the board inside the state."""
    from base import BoardGame

    class M(BoardGame):
        __mutation__ = "perturb_board"
        __wrapped_class__ = cls

        def __init__(self):
            self.inner = cls()

        def __getattr__(self, n):
            if n == "inner":
                raise AttributeError(n)
            return getattr(self.inner, n)

        def reset(self):
            return self.inner.reset()

        def current_player(self, s):
            return self.inner.current_player(s)

        def legal_actions(self, s):
            return self.inner.legal_actions(s)

        def step(self, s, a):
            nxt, rw, done = self.inner.step(s, a)
            alt = _swap_in_board(nxt)
            if alt is not None:
                FIRED[0] += 1
                nxt = alt
            return nxt, rw, done
    M.__name__ = cls.__name__
    return M


def early_end(cls, at=25):
    from base import BoardGame

    class M(BoardGame):
        __mutation__ = f"early_end@{at}"
        __wrapped_class__ = cls

        def __init__(self):
            self.inner = cls()

        def __getattr__(self, n):
            if n == "inner":
                raise AttributeError(n)
            return getattr(self.inner, n)

        def reset(self):
            return self.inner.reset() + (0,)

        def current_player(self, s):
            return self.inner.current_player(s[:-1])

        def legal_actions(self, s):
            return self.inner.legal_actions(s[:-1])

        def step(self, s, a):
            nxt, rw, done = self.inner.step(s[:-1], a)
            n = s[-1] + 1
            if not done and n >= at:
                FIRED[0] += 1
                return nxt + (n,), 1.0, True
            return nxt + (n,), rw, done
    M.__name__ = cls.__name__
    return M


def flip_winner(cls):
    """Negate the terminal reward, counting the rewards actually negated.

    This used to call operators.override_winner(mode="flip"). The disclosure decision of 2026-10-20
    releases the ladder and withholds the repair operators, so a harness a reader is meant to run cannot
    depend on a file they will not have -- and the mutation is four lines. A drawn game flips to itself,
    so only a non-zero terminal reward counts as the mutation firing."""
    from base import BoardGame

    class M(BoardGame):
        __mutation__ = "flip_winner"
        __wrapped_class__ = cls

        def __init__(self):
            self.inner = cls()

        def __getattr__(self, n):
            if n == "inner":
                raise AttributeError(n)
            return getattr(self.inner, n)

        def reset(self):
            return self.inner.reset()

        def current_player(self, s):
            return self.inner.current_player(s)

        def legal_actions(self, s):
            return self.inner.legal_actions(s)

        def step(self, s, a):
            nxt, rw, done = self.inner.step(s, a)
            if done and rw:
                FIRED[0] += 1
                return nxt, -rw, done
            return nxt, rw, done
    M.__name__ = cls.__name__
    return M


def end_at(key):
    """Where early_end should cut: half the mean length the game-property run observed, so the cut lands
    inside a real game rather than past its end."""
    try:
        cat = {x["key"]: x for x in json.load(open(R / "catalog.json"))["games"]}
        return max(3, int((cat[key]["play"]["random"]["len"] or 20) * 0.5))
    except Exception:
        return 25


def run_one(key, name, episodes, time_limit, cap):
    """One mutant, one certification, in this process. Printed as JSON for the parent."""
    import types
    import bgame
    from ludii_oracle import certify, find_boardgame_class
    table = {"drop_move": drop_move, "perturb_board": perturb_board,
             "early_end": lambda c: early_end(c, at=end_at(key)), "flip_winner": flip_winner}
    base = find_boardgame_class(importlib.import_module(bgame.module_name(key)))
    m = types.ModuleType(f"games.mut.{key}__{name}")
    m.__dict__[base.__name__] = table[name](base)
    sys.modules[m.__name__] = m
    r = certify(m.__name__, bgame._ludii_name(key), episodes=episodes,
                time_limit=time_limit, deadline=cap)
    # A mutant that falls short of \lvl5 because its BUDGET ran out has not been detected -- the ladder
    # never got to look. Counting that as a detection would inflate the measured sensitivity exactly where
    # it is weakest, on the slow games, so a level decided by a cut or an UNREACHED timeout is recorded as
    # not measured. It has not yet happened (0 of the first 111 detections), which is why it is worth a
    # guard rather than a correction.
    cut = bool(r.get("budget_cut")) or any("UNREACHED" in str(r.get(k) or "") for k in ("l3", "l4", "l5"))
    print(json.dumps({"level": r.get("level"), "l3": r.get("l3"), "l4": r.get("l4"), "l5": r.get("l5"),
                      "budget_cut": bool(r.get("budget_cut")), "fired": FIRED[0],
                      "K": episodes, "time_limit": time_limit,
                      "detected": None if (cut or not FIRED[0]) else (r.get("level") != "L5"),
                      "not_applicable": FIRED[0] == 0,
                      "error": "not measured: the budget decided the level, not the mutation" if cut else None,
                      "cut_at": end_at(key) if name == "early_end" else None}))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        _, _, key, name, ep, tl = sys.argv[:6]
        run_one(key, name, int(ep), float(tl), float(tl) * 8)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=30)
    # Stage 2 of the certifier is 240 s/episode with a 1200 s deadline, and four games in this sample
    # (Breakthru, Duidoku, Gala Buginese, Vigilance) certify only there -- Breakthru is \lvl4 at stage 1
    # with l5=UNREACHED:timeout and \lvl5 only at stage 2. Measuring their mutants under a smaller budget
    # than the unmutated game needs cannot answer whether the ladder notices the mutation, so the default
    # here is the certifier's own stage-2 budget rather than a number chosen for this harness.
    ap.add_argument("--time-limit", type=float, default=240.0)
    ap.add_argument("--retry-capped", action="store_true",
                    help="re-run rows that hit the cap, under the cap given now")
    ap.add_argument("--cap", type=float, default=420.0,
                    help="hard wall-clock seconds per mutant; beyond it the row is not measured")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("keys", nargs="*")
    a = ap.parse_args()
    import bgame
    lad = {r["game"]: r for r in json.load(open(R / "ladder_certification_K30_effect.json"))}
    pool = sorted(g for g, r in lad.items() if r["level"] == "L5")
    keys = a.keys or random.Random(a.seed).sample(pool, min(a.games, len(pool)))
    muts = ["drop_move", "perturb_board", "early_end", "flip_winner"]
    print(f"{len(keys)} certified games x {len(muts)} mutations, K={a.episodes}, "
          f"{a.time_limit}s/episode\nsample seed {a.seed}: {keys}\n", flush=True)
    out = json.load(open(OUT)) if OUT.exists() else {}
    for i, k in enumerate(keys):
        rec = out.setdefault(k, {})
        for name in muts:
            # A row that hit the cap is settled, not pending: the game is slow by construction and will
            # cap again, so retrying it costs the cap once more and tells us nothing. Other errors are
            # retried, since those can be contention.
            # A row measured under a different K OR a different per-episode budget is a different
            # measurement and cannot sit in the same table -- the rule the ladder itself runs under.
            # No leniency for a row that does not say what it was measured under. The first version
            # accepted a missing budget as "same", which left seventeen rows from the run before the
            # stamping was added sitting in a table whose other 103 rows were measured differently --
            # an unknown budget is the strongest reason to re-measure, not a reason to keep.
            same = (rec.get(name, {}).get("K") == a.episodes
                    and rec.get(name, {}).get("time_limit") == a.time_limit)
            done_ = (name in rec and same
                     and (not rec[name].get("error")
                          or ("cap" in rec[name].get("error", "") and not a.retry_capped)))
            if done_:
                continue
            if name in rec and rec[name].get("K") not in (a.episodes, None):
                print(f"  re-running {k} {name}: stored at K={rec[name]['K']}, this run is K={a.episodes}",
                      flush=True)
            t0 = time.time()
            # Each certification runs in its own process under a hard wall-clock cap. Some translations
            # are slow by construction -- duidoku recomputes its legal set inside step(), so applying all
            # 729 of them once per ply is half a million validity checks -- and the certifier's own
            # per-episode limit is only consulted between plies, so a single ply can outlast it. A game
            # that does not finish inside the cap is recorded as UNMEASURED, never as detected or missed:
            # the denominator shrinks and says so, which is the honest way to keep the sample random
            # rather than quietly drawing only the fast games.
            try:
                r = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), str(Path(__file__)),
                                "--one", k, name, str(a.episodes), str(a.time_limit)],
                               cwd=ROOT, capture_output=True, text=True, timeout=a.cap)
            except subprocess.TimeoutExpired:
                rec[name] = {"detected": None, "error": f"not measured: exceeded the {a.cap:.0f}s cap"}
                rec[name]["seconds"] = round(time.time() - t0, 1)
                OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
                continue
            try:
                rec[name] = json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][-1])
            except Exception:
                rec[name] = {"detected": None,
                             "error": (r.stderr or r.stdout or "no output")[-120:]}
            rec[name]["seconds"] = round(time.time() - t0, 1)
            OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print(f"  [{i+1:3d}/{len(keys)}] {k:24s} " +
              " ".join(f"{n}={'Y' if rec[n].get('detected') else ('n/a' if rec[n].get('not_applicable') else ('?' if rec[n].get('detected') is None else 'MISSED'))}"
                       for n in muts), flush=True)
    print()
    for name in muts:
        v = [rec[name] for rec in out.values() if name in rec]
        ok = [x for x in v if x.get("detected")]
        miss = sorted(g for g, rec in out.items() if rec.get(name, {}).get("detected") is False)
        na = [x for x in v if x.get("not_applicable")]
        err = [x for x in v if x.get("error")]
        print(f"{name:14s} detected {len(ok)}/{len(v) - len(na) - len(err)}" +
              (f"   MISSED: {miss}" if miss else "") +
              (f"   not applicable: {len(na)}" if na else "") +
              (f"   not measured: {len(err)}" if err else ""))
    print("\nan undetected mutant is a hole in the certifier: it is named above and reproducible")
    print("-> results/mutation_test.json")


if __name__ == "__main__":
    main()
