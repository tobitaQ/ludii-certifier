"""Compare two full certification runs and say which rows may be compared at all.

Every certifier change is followed by a full re-certification, and the decision to promote the new run to
canonical rests on one sentence: "no level dropped". That sentence is only true of rows that were actually
measured the same way. A row whose run hit the per-game deadline stopped early, and its level is then the
level of the episodes it managed -- gonnect flipped L2- -> L5 in the 2026-09-25 v6 run purely because a
1200 s deadline cut it after 7 episodes, before the seed (26) that fails. The records carry `budget_cut`;
the first comparison of that run ignored the field, which is the mistake this tool exists to prevent.

  comparable   both runs measured the game, not the budget
  incomparable one side was cut by its per-game deadline, or decided a level by an UNREACHED timeout
               -> listed for a re-run at the stage-3 budget, never counted as a rise or a drop
Usage:  .venv/bin/python pipeline/ladder_diff.py OLD.json NEW.json
Output: prints the table; writes results/ladder_diff_<new stem>.json
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"
RANK = {"L0": 0, "L2-": 2, "L3": 3, "L4": 4, "L5": 5}


def load(p):
    p = Path(p)
    if not p.exists(): p = R / p
    d = json.load(open(p))
    return {r["game"]: r for r in d}, p


def main():
    if len(sys.argv) < 3:
        print(__doc__); return 2
    old, po = load(sys.argv[1]); new, pn = load(sys.argv[2])
    up, down, same, cut = [], [], [], []
    for g, b in new.items():
        a = old.get(g)
        if a is None:
            cut.append((g, "not in the old run", None, b["level"])); continue
        # a row whose level was decided by a TIMEOUT is not a measurement of the game, it is a measurement of
        # the budget: omny read L5 at 900 s per episode and L4 ("l5: UNREACHED:timeout") at 240 s, and the
        # first version of this tool called that a regression because it only looked at the per-game deadline.
        def unreached(x):
            return any(str(x.get(k, "")).startswith("UNREACHED") for k in ("l3", "l4", "l5"))

        def err(x):
            # A row whose verdict is an ERROR is not a measurement either. v14 lost its JVM when the
            # parent hit its wall-clock cap and seven games came back `SKIP:error Java Virtual Machine is
            # not running`: compared naively that is seven L5 games dropping to L2-, the exact shape of a
            # catastrophic regression and entirely an artefact of the harness dying (2026-10-17).
            for k in ("l3", "l4", "l5"):
                v = str(x.get(k) or "")
                if v.startswith("SKIP:error") or v.startswith("ERROR"):
                    return v[:70]
            return None
        if err(a) or err(b) or a.get("budget_cut") or b.get("budget_cut") or unreached(a) or unreached(b):
            why = ("the run errored: " + str(err(b) or err(a))) if (err(a) or err(b)) else (
                b.get("budget_cut") or a.get("budget_cut")
                or ("new run timed out: " + str(b.get("l5") or b.get("l4") or b.get("l3"))
                    if unreached(b) else "old run timed out"))
            cut.append((g, why, a["level"], b["level"])); continue
        if a["level"] == b["level"]: same.append(g)
        elif RANK[b["level"]] > RANK[a["level"]]: up.append((g, a["level"], b["level"], str(b.get("mapping"))))
        else: down.append((g, a["level"], b["level"], str(b.get("l3"))[:20], str(b.get("l4"))[:20], str(b.get("l5"))[:24]))
    print(f"{po.name} -> {pn.name}: {len(new)} of {len(old)} rows present")
    print(f"  comparable: {len(same)+len(up)+len(down)}   same {len(same)} | up {len(up)} | DOWN {len(down)}")
    for g, a, b, m in up: print(f"    UP   {g:26s} {a} -> {b}   ({m})")
    for row in down: print(f"    DOWN {row[0]:26s} {row[1]} -> {row[2]}   l3={row[3]} l4={row[4]} l5={row[5]}")
    print(f"  NOT comparable (a deadline cut one side): {len(cut)}")
    for g, why, a, b in cut: print(f"    {g:26s} {a} -> {b}   {str(why)[:70]}")
    out = {"old": po.name, "new": pn.name, "same": same, "up": up, "down": down,
           "incomparable": [{"game": g, "why": str(w), "old": a, "new": b} for g, w, a, b in cut],
           "promotable": not down and not cut,
           "note": ("A run is promotable to canonical only when nothing dropped AND no row is incomparable. "
                    "Re-run the incomparable rows at the stage-3 budget (--time-limit 900, no deadline) first.")}
    p = R / f"ladder_diff_{pn.stem}.json"; p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"  -> {p.name}; promotable: {out['promotable']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
