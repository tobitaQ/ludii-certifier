"""What the corpus IS, written down, so that a run cannot silently certify a different one.

On 2026-10-17 a full re-certification produced 345 rows. The corpus is 305. Forty control-group
translations had been written into `src/games/converted/` -- the right place for them to be importable, the
wrong place for them to be counted -- and every count in the paper was briefly computed over a set that
included them. Nothing in the pipeline objected, because nothing in the pipeline had ever been told which
305 games the corpus contains: `bgame.games()` returns whatever `.py` files are in the directory.

So the membership is written down here, once, and checked. Two different things are recorded and they are
NOT treated alike:

  membership  which 305 keys are the corpus. This is an invariant. A key appearing or disappearing is a
              different corpus, every published count is about a different set, and the run is refused.
  content     the sha256 of each translation's source. This is provenance, not an invariant: declarations
              (LUDII_SCORE_MAP, LUDII_OPTIONS) are edited in these files as part of the work, so a changed
              hash is reported and recorded, never refused.

Re-blessing is deliberate and leaves a record: `--bless` rewrites the manifest and prints what moved, so
the commit that changes the corpus says so.

Usage:  .venv/bin/python pipeline/corpus_manifest.py [--bless] [--json]
Output: results/corpus_manifest.json
"""
import sys, json, hashlib, argparse, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "corpus_manifest.json"
CONV = ROOT / "src" / "games" / "converted"
LADDER = R / "ladder_certification_K30_effect.json"

# Modules that live in the converted directory and are NOT the corpus. Each needs a reason, because an
# unexplained file there is how 345 games got certified: the directory is the import path, not the
# definition of the corpus. Blessing refuses any on-disk module that is neither in the corpus nor here.
EXCLUDED = {
    "adidada_chk": "working copy of adidada kept while its count-only board was being fixed",
    "aime": "residual: translated but outside the 305 selected for the corpus",
    "alea_evangelii": "residual: translated but outside the 305 selected for the corpus",
    "castello": "residual: translated but outside the 305 selected for the corpus",
    "chinese_checkers": "residual: translated but outside the 305 selected for the corpus",
    "kensington": "residual: translated but outside the 305 selected for the corpus",
    "shobu": "residual: translated but outside the 305 selected for the corpus",
    "suffragetto": "residual: translated but outside the 305 selected for the corpus",
}


def on_disk():
    """key -> sha256 of the translation source, for every module in the converted directory."""
    return {p.stem: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(CONV.glob("*.py"))
            if not p.stem.startswith("_") and "_agent" not in p.stem}


def load():
    return json.load(open(OUT)) if OUT.exists() else None


def verify():
    """(missing, extra, changed) against the manifest; empty lists if there is no manifest yet.

    `extra` counts only modules that are neither the corpus nor a listed exclusion -- a new file nobody
    accounted for, which is the case that has to stop a run.
    """
    m = load()
    if not m:
        return [], [], []
    have, want = on_disk(), m["games"]
    known = set(want) | set(m.get("excluded", EXCLUDED))
    return (sorted(set(want) - set(have)), sorted(set(have) - known),
            sorted(k for k in set(have) & set(want) if have[k] != want[k]))


def require(who="this run"):
    """Refuse to proceed if the corpus on disk is not the corpus the manifest names.

    Content changes do not refuse: a translation's declarations are edited as part of the work. Only
    membership does, because membership is what every count in the paper is over.
    """
    missing, extra, _ = verify()
    if missing or extra:
        raise SystemExit(
            f"{who} would run over a different corpus than results/corpus_manifest.json names: "
            f"{len(missing)} missing {missing[:6]}, {len(extra)} extra {extra[:6]}.\n"
            f"If the corpus really changed, re-bless it deliberately:\n"
            f"    .venv/bin/python pipeline/corpus_manifest.py --bless")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bless", action="store_true", help="rewrite the manifest from what is on disk")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    have = on_disk()
    missing, extra, changed = verify()
    old = load()
    if a.bless:
        # The corpus is the set the canonical ladder certifies, not the set of files in the directory.
        corpus = sorted(r["game"] for r in json.load(open(LADDER)))
        absent = [k for k in corpus if k not in have]
        unexplained = sorted(set(have) - set(corpus) - set(EXCLUDED))
        if absent:
            raise SystemExit(f"the ladder certifies {len(absent)} games with no module on disk: {absent[:8]}")
        if unexplained:
            raise SystemExit(
                f"{len(unexplained)} modules in {CONV.relative_to(ROOT)} are neither the corpus nor a "
                f"listed exclusion: {unexplained}\nAdd each to EXCLUDED in this file with a reason, or "
                f"move it out of the directory, before blessing.")
        OUT.write_text(json.dumps({
            "blessed": datetime.date.today().isoformat(), "n": len(corpus),
            "games": {k: have[k] for k in corpus},
            "excluded": {k: EXCLUDED[k] for k in sorted(EXCLUDED) if k in have},
            "note": "membership is an invariant; content hashes are provenance",
        }, indent=1, ensure_ascii=False))
        print(f"blessed {len(corpus)} translations as the corpus, "
              f"{sum(1 for k in EXCLUDED if k in have)} modules excluded by name")
        return 0
    if a.json:
        print(json.dumps({"n_disk": len(have), "missing": missing, "extra": extra, "changed": changed},
                         indent=1, ensure_ascii=False))
        return 0
    if not old:
        print(f"no manifest yet; {len(have)} translations on disk.\n"
              f"    .venv/bin/python pipeline/corpus_manifest.py --bless")
        return 1
    print(f"manifest blessed {old['blessed']}: {old['n']} games;  modules on disk: {len(have)} "
          f"({len(old.get('excluded', {}))} excluded by name)")
    print(f"  membership  missing {len(missing)} {missing[:8]}   extra {len(extra)} {extra[:8]}")
    print(f"  content     {len(changed)} translations edited since the manifest {changed[:8]}")
    print("\nthe corpus is the one the manifest names" if not (missing or extra)
          else "\nTHE CORPUS ON DISK IS NOT THE CORPUS THE MANIFEST NAMES -- every count would be over a "
               "different set")
    return 1 if (missing or extra) else 0


if __name__ == "__main__":
    sys.exit(main())
