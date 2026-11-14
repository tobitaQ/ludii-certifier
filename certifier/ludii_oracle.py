"""Canonical move-level differential oracle (single source of truth).

Semantics of "faithful": there exists ONE consistent board mapping (action
encoding + a cell permutation built from Ludii's own (row, col) geometry under
a rigid symmetry, or a reading-order rank for non-rectangular boards, or the
legacy square-grid symmetry) under which the Python implementation's set of
legal moves equals Ludii's at every replayed step. Candidates are enumerated at
the initial state; since an empty board cannot disambiguate symmetries,
certify_run() replays under every consistent candidate and keeps the best
ladder level (existential quantification over the mapping). oracle_run() keeps
the legacy single-candidate behaviour for the L3 tools.

Ludii numbers hex-diamond boards along anti-diagonals, not row-major; assuming
row-major made the whole Hex family look wrong at L5 while the Python was right.

All tools (audit, repair, promotion gate) must call oracle_run() so the verdict
is identical everywhere — a looser per-tool oracle previously promoted a
regressed game.
"""
import math, random, importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (this script lives in pipeline/)
SRC = ROOT / "src"
JAR = ROOT / "ludii" / "Ludii-1.3.14.jar"

import jpype, jpype.imports
if not jpype.isJVMStarted():
    import sys
    sys.path.insert(0, str(SRC))
    jpype.startJVM(classpath=[str(JAR.resolve())])
from other import GameLoader            # type: ignore
from other.context import Context       # type: ignore
from other.trial import Trial           # type: ignore


def load_game(name, options=None):
    """GameLoader.loadGameFromName with an optional list of Ludii option
    strings such as ["Board Size/9x9"] (a translation may implement a
    non-default ruleset; certification must then target that variant)."""
    if options:
        from java.util import ArrayList
        o = ArrayList()
        for x in options:
            o.add(x)
        return GameLoader.loadGameFromName(name, o)
    return GameLoader.loadGameFromName(name)


def _norm(f, t):
    return (None, t) if f == t else (f, t)


class UnparsableAction(ValueError):
    """The syntactic (from,to) reading does not apply to this action encoding;
    callers fall back to the effect-based path."""


def pkey(a, ncells=0):
    """Python action -> normalized (from, to) key (syntactic reading)."""
    try:
        if isinstance(a, (tuple, list)) and len(a) >= 2:
            return _norm(int(a[0]), int(a[1]))
        a = int(a)
    except (TypeError, ValueError):
        raise UnparsableAction(repr(a))
    return _norm(a // ncells, a % ncells) if ncells > 0 else (None, a)


def _sym(W, name):
    def conv(cell):
        if cell is None or cell < 0:      # meta-moves (pass/swap = -1) are not board cells
            return cell
        r, c = cell // W, cell % W
        return {"identity": r*W+c, "vflip": (W-1-r)*W+c, "hflip": r*W+(W-1-c),
                "rot180": (W-1-r)*W+(W-1-c), "transpose": c*W+r}[name]
    return conv


_SYMS = ["identity", "vflip", "hflip", "rot180", "transpose"]


def _by_move_state(cand, mv):
    """Among python actions sharing one (from, to) key, keep the one whose third element is the move's state.

    Ludii's (from, to) is not always a move's identity. The L game offers up to 17 moves at a ply that reduce
    to 10 (from, to) pairs: the L piece is one tile anchored on one site and the rest of its shape lives in the
    move's `state` (its orientation), which the translation carries as the third element of its own action
    (from, to, orientation) and verified against Ludii's numbering when it was written. Taking cand[0] picks the
    wrong orientation about half the time; the two engines then hold different footprints and the divergence
    surfaces plies later in the NEUTRAL-DOT moves (l_game, FAIL@15, seed 1).

    This narrows, never widens: it only applies when more than one python action matches and exactly one of them
    carries the move's state, so a game whose third element means something else (sher_bakr's hopped-over cell,
    say) is untouched, and so is every game whose (from, to) key is already injective.
    """
    if len(cand) <= 1:
        return cand
    try:
        s = int(mv.state())
    except Exception:
        return cand
    hit = [a for a in cand if isinstance(a, (tuple, list)) and len(a) >= 3 and int(a[2]) == s]
    return hit if len(hit) == 1 else cand


def _by_state(hits, action):
    """The same narrowing in the other direction: one scripted python action, several Ludii moves."""
    if len(hits) <= 1 or not (isinstance(action, (tuple, list)) and len(action) >= 3):
        return hits
    try:
        s = int(action[2])
        hit = [m for m in hits if int(m.state()) == s]
    except Exception:
        return hits
    return hit if len(hit) == 1 else hits


def _pset(pylegal, ncells, T):
    out = set()
    for a in pylegal:
        f, t = pkey(a, ncells)
        out.add((None if f is None else T(f), None if t is None else T(t)))
    return out


def ludii_geometry(game, stype="Cell"):
    """{(row, col): ludii_site_index} for the board's default site type.
    Ludii numbers hex-diamond boards along anti-diagonals and some boards
    column-wise, so the site index is NOT row-major in general."""
    topo = game.board().topology()
    sites = {"Cell": topo.cells, "Vertex": topo.vertices, "Edge": topo.edges}[stype]()
    geo = {}
    for i in range(sites.size()):
        s = sites.get(i)
        geo[(int(s.row()), int(s.col()))] = i
    return geo


def _geo_sym(geo, H, W, name):
    """Python row-major (H x W) cell -> Ludii site index via Ludii's own
    (row, col) geometry, under a rigid symmetry applied to (row, col)."""
    def conv(cell):
        if cell is None or cell < 0:      # meta-moves (pass/swap = -1) are not board cells
            return cell
        r, c = cell // W, cell % W
        rc = {"identity": (r, c), "vflip": (H-1-r, c), "hflip": (r, W-1-c),
              "rot180": (H-1-r, W-1-c), "transpose": (c, r),
              "rot90": (c, H-1-r), "rot270": (W-1-c, r),
              "antitranspose": (W-1-c, H-1-r)}[name]
        return geo.get(rc, -1)
    return conv


_GEO_SYMS = ["identity", "vflip", "hflip", "rot180", "transpose", "rot90", "rot270",
             "antitranspose"]


def candidate_mappings(pylegal, lset, geo=None):
    """All (ncells, name, W, T) under which the Python legal set equals the
    Ludii legal set at this state, geometry-based candidates first (they follow
    Ludii's real site numbering), then the legacy square-grid search.
    Placement games cannot disambiguate symmetries at the empty board, so the
    caller must try candidates until one survives the whole replay.
    Returns [] when the actions have no syntactic (from,to) reading."""
    try:
        return _candidate_mappings(pylegal, lset, geo)
    except UnparsableAction:
        return []


def _candidate_mappings(pylegal, lset, geo=None):
    is_int = bool(pylegal) and isinstance(pylegal[0], int)
    ncands = [0] + ([W*W for W in range(3, 22)] if is_int else [0])
    cells = [c for k in lset for c in k if c is not None]
    out, seen = [], set()

    def add(ncells, name, W, T):
        sig = (ncells, tuple(T(i) for i in range(W * W)))
        if sig not in seen:
            seen.add(sig); out.append((ncells, name, W, T))

    if geo:
        H = max(r for r, _ in geo) + 1
        Wg = max(c for _, c in geo) + 1
        for ncells in dict.fromkeys(ncands):
            for s in _GEO_SYMS:
                if s in ("transpose", "rot90", "rot270", "antitranspose") and H != Wg:
                    continue
                T = _geo_sym(geo, H, Wg, s)
                if _pset(pylegal, ncells, T) == lset:
                    add(ncells, f"geo:{s}", Wg, T)
        # Non-rectangular boards (triangles, hex-hex, crosses): Python usually
        # numbers cells densely in reading order; rank Ludii's sites by (row,col)
        # under each orientation and map python index i -> i-th site.
        n = len(geo)
        for oname, keyf in (("rank:rc", lambda rc: (rc[0], rc[1])),
                            ("rank:Rc", lambda rc: (-rc[0], rc[1])),
                            ("rank:rC", lambda rc: (rc[0], -rc[1])),
                            ("rank:RC", lambda rc: (-rc[0], -rc[1])),
                            ("rank:cr", lambda rc: (rc[1], rc[0])),
                            ("rank:Cr", lambda rc: (-rc[1], rc[0])),
                            ("rank:cR", lambda rc: (rc[1], -rc[0])),
                            ("rank:CR", lambda rc: (-rc[1], -rc[0]))):
            order = [geo[rc] for rc in sorted(geo, key=keyf)]
            T = (lambda order: lambda cell: cell if (cell is None or cell < 0)
                 else (order[cell] if 0 <= cell < len(order) else -1))(order)
            Wr = max(n, 1)
            for ncells in dict.fromkeys(ncands):
                if _pset(pylegal, ncells, T) == lset:
                    add(ncells, oname, Wr, T)
    for ncells in dict.fromkeys(ncands):
        pcells = [c for k in (pkey(a, ncells) for a in pylegal) for c in k if c is not None]
        hi = max(cells + pcells, default=0) + 1
        for W in range(max(1, int(math.sqrt(hi))), 22):
            if W * W < hi:
                continue
            for s in _SYMS:
                T = _sym(W, s)
                if _pset(pylegal, ncells, T) == lset:
                    add(ncells, s, W, T)
    return out


def detect_mapping(pylegal, lset, geo=None):
    """First consistent mapping (see candidate_mappings) or None."""
    c = candidate_mappings(pylegal, lset, geo)
    return c[0] if c else None


def find_boardgame_class(mod):
    import inspect
    for n, o in inspect.getmembers(mod, inspect.isclass):
        if n.startswith("_") or n == "BoardGame":
            continue
        if getattr(o, "__abstractmethods__", None):
            continue
        if all(hasattr(o, m) for m in ("reset", "legal_actions", "step", "current_player")):
            return o
    return None


def oracle_run(module, ludii_name, max_steps=12, seed=1):
    """Replay Ludii's moves under one consistent mapping; return a verdict dict:
      {faithful, step, mapping, ludii_only, python_only, note}
    faithful=True means sets matched for all checked steps (game may end earlier).
    """
    game = GameLoader.loadGameFromName(ludii_name)
    geo = ludii_geometry(game, ludii_site_info(game)[0])
    trial = Trial(game); ctx = Context(game, trial); game.start(ctx)
    mod = importlib.import_module(module)
    cls = find_boardgame_class(mod)
    if cls is None:
        return {"faithful": False, "note": "no BoardGame class"}
    g = cls(); st = g.reset(); random.seed(seed)
    mapping = None
    for step in range(max_steps):
        if trial.over():
            return {"faithful": True, "step": step, "mapping": _mapstr(mapping),
                    "ludii_only": [], "python_only": [], "note": "ludii-over"}
        legal = game.moves(ctx).moves()
        lmoves = [legal.get(i) for i in range(legal.size())]
        lset = {_norm(int(m.fromNonDecision()), int(m.toNonDecision())) for m in lmoves}
        pylegal = g.legal_actions(st)
        if mapping is None:
            mapping = detect_mapping(pylegal, lset, geo)
            if mapping is None:
                # report under identity for inspection
                raw = _pset(pylegal, 0, lambda c: c)
                return {"faithful": False, "step": step, "mapping": None,
                        "ludii_only": sorted(lset - raw)[:12],
                        "python_only": sorted(raw - lset)[:12],
                        "note": "no consistent mapping"}
        ncells, s, W, T = mapping
        ps = _pset(pylegal, ncells, T)
        if ps != lset:
            return {"faithful": False, "step": step, "mapping": _mapstr(mapping),
                    "ludii_only": sorted(lset - ps)[:12],
                    "python_only": sorted(ps - lset)[:12], "note": "diverge"}
        # advance: pick a Ludii move, apply it in both
        mv = lmoves[random.randrange(len(lmoves))]
        key = _norm(int(mv.fromNonDecision()), int(mv.toNonDecision()))
        game.apply(ctx, mv)
        cand = [a for a in pylegal
                if (lambda k: (None if k[0] is None else T(k[0]),
                               None if k[1] is None else T(k[1])))(pkey(a, ncells)) == key]
        if not cand:
            return {"faithful": False, "step": step, "mapping": _mapstr(mapping),
                    "ludii_only": [key], "python_only": [], "note": "ludii move absent"}
        st, _, done = g.step(st, cand[0])
        if done:
            return {"faithful": True, "step": step + 1, "mapping": _mapstr(mapping),
                    "ludii_only": [], "python_only": [], "note": "python-over"}
    return {"faithful": True, "step": max_steps, "mapping": _mapstr(mapping),
            "ludii_only": [], "python_only": [], "note": "max-steps"}


def _mapstr(m):
    if m is None:
        return None
    ncells, s, W, _ = m
    return f"ncells{ncells}/{s}@W{W}"


# ---------------- L4/L5 certification (fidelity ladder, PI plan 2026-09-11) ----
# L3 = legal-move sets match on the replayed trajectory (oracle_run's criterion).
# L4 = the board state after every applied move matches Ludii's per-site
#      who/count under the SAME coordinate transform T, an offset into the
#      Python state vector, and a consistent piece-label map. Games whose state
#      is not a flat numeric vector are UNTESTABLE (reported, not failed).
# L5 = both engines terminate on the same ply and agree on the winner
#      (Python reward is mover-relative per base.py; Ludii via trial.ranking()).

def ludii_site_info(game):
    """(site_type, n_sites) for the board's default site type."""
    topo = game.board().topology()
    ds = str(game.board().defaultSite())
    if ds == "Vertex":
        return "Vertex", int(topo.vertices().size())
    if ds == "Edge":
        return "Edge", int(topo.edges().size())
    return "Cell", int(topo.cells().size())


def ludii_what(ctx, stype, n):
    """Per-site component id (piece TYPE, not owner) for container 0."""
    cs = ctx.state().containerStates()[0]
    what = getattr(cs, f"what{stype}")
    return [int(what(i)) for i in range(n)]


def tile_footprint(game, stype, geo, inv, site, state, what):
    """Cells a TILE placement covers, by walking the component's own shape.

    Ludii writes only the anchor into who/count/what; the rest of the tile is implied by its `walk` -- a
    turtle path of StepTypes (F step forward, R/L turn) -- rotated by the placement's `state`. Reading the
    path is what makes this general: cram's tile is [F] (a domino), domineering's are [R, F] and [F], and
    pentomino's are paths of six to twelve steps. A single-step approximation certified cram but broke
    domineering and pentomino (2026-09-30), which is why the walk is followed literally here.
    """
    try:
        comp = game.equipment().components()[what]
        if not bool(comp.isTile()):
            return {site}
        walks = list(comp.walk())
        if not walks:
            return {site}
    except Exception:
        return {site}
    rc = inv.get(site)
    if rc is None:
        return {site}
    DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]          # the rotation order Ludii's `state` indexes
    out = {site}
    for row in walks:
        r, c = rc
        facing = state % 4
        for stp in list(row):
            t = str(stp).upper()
            if t.startswith("R"):
                facing = (facing + 1) % 4
            elif t.startswith("L"):
                facing = (facing - 1) % 4
            elif t.startswith("F"):
                dr, dc = DIRS[facing]
                r, c = r + dr, c + dc
                p = geo.get((r, c))
                if p is None:
                    break
                out.add(p)
    return out


def ludii_hands(ctx):
    """Piece totals of the off-board containers (hands / pools = containers 1..), in container order."""
    game = ctx.game(); conts = game.equipment().containers()
    if len(conts) <= 1:
        return []
    cs = ctx.state().containerStates(); sf = list(ctx.sitesFrom()); stacking = bool(game.isStacking())
    out = []
    for k in range(1, len(conts)):
        tot = 0
        for i in range(int(conts[k].numSites())):
            site = sf[k] + i
            tot += int(cs[k].sizeStackCell(site)) if stacking else int(cs[k].countCell(site))
        out.append(tot)
    return out


def ludii_scores(ctx):
    """Per-player score as the engine holds it (1..N). A score is state the board does not show: azteka's
    (set Score P1 (count Pieces Mover in:(sites Board))) freezes the round-1 piece count, which survives the
    board reset and decides the winner 36 plies later."""
    n = int(ctx.game().players().count())
    try:
        return [int(ctx.score(p)) for p in range(1, n + 1)]
    except Exception:
        return []


def declared_score_map(cls):
    """LUDII_SCORE_MAP declared on the game class or its module: list where entry k is the index, in the
    FLATTENED state, of the field holding player k+1's Ludii score, or -1 if untracked. Same contract as
    LUDII_HAND_MAP -- declared, never learned."""
    import sys as _sys
    m = None; c = cls
    while c is not None and m is None:
        m = getattr(c, "LUDII_SCORE_MAP", None)
        if m is None:
            m = getattr(_sys.modules.get(c.__module__), "LUDII_SCORE_MAP", None)
        c = getattr(c, "__wrapped_class__", None)
    if m is None:
        return None
    try:
        return [int(x) for x in m]
    except Exception:
        return None


def declared_start_sync(cls):
    """`ludii_start(who, count, what)` declared on the translation, or None.

    Four games in the corpus begin from a position Ludii DRAWS AT RANDOM ((place Random ...)), so the two
    engines start from different boards and the ladder cannot begin at all: all four sit at L2- with
    `FAIL@0` and no mapping, never having been compared. They are the only rows whose reason is a gap in
    the certifier rather than anything about the translation.

    The hook closes it: the certifier starts Ludii, reads the position Ludii drew, and asks the translation
    to begin from that position instead of drawing its own. The lists are indexed by LUDII SITE and the
    translation is responsible for the correspondence -- the same contract as the other declarations, and
    for the same reason: it is a fact about the translation that only the translation can state. The
    mapping search then runs as usual, with an unusually easy job, because a randomly drawn position pins
    the symmetry far better than an empty board does.

    Declared, never inferred, and self-checking: a translation whose sync is wrong starts from a different
    position and fails at ply 0, exactly as it does today.
    """
    f = getattr(cls, "ludii_start", None)
    return f if callable(f) else None


def _start_state(g, ctx, stype, nsites):
    """The translation's start position: its own, or Ludii's where the game declares the sync hook."""
    f = declared_start_sync(type(g))
    if f is None:
        return g.reset()
    who, cnt = ludii_board(ctx, stype, nsites)
    what = ludii_what(ctx, stype, nsites)
    return f(g, list(who), list(cnt), list(what))


def declared_rotation_map(cls):
    """LUDII_ROTATION_MAP declared on the game class or its module: list where entry i is the index, in the
    FLATTENED state, of the field holding the ROTATION of board cell i (the translation's own cell numbering,
    the one LUDII_SITE_MAP maps), or -1 where there is none.

    The third declaration, and the first whose VALUES are not required to agree. Trax's two tiles sit on one
    cell each, so there is no footprint to expand (walk() is None) and the move carries no orientation
    (Move.state() is -1); the only place the difference between two placements on the same cell exists is
    Ludii's `rotationCell` after the move, and the translation's own `rots` block. But the translation counts
    quarter turns 0..3 while Ludii indexes eighths, 0, 2, 4, 6 -- so the declaration can only say WHERE the
    rotation is, and the correspondence between the two numberings is learned during the replay, the way the
    player correspondence already is."""
    import sys as _sys
    m = None; c = cls
    while c is not None and m is None:
        m = getattr(c, "LUDII_ROTATION_MAP", None)
        if m is None:
            m = getattr(_sys.modules.get(c.__module__), "LUDII_ROTATION_MAP", None)
        c = getattr(c, "__wrapped_class__", None)
    if m is None:
        return None
    try:
        return [int(x) for x in m]
    except Exception:
        return None


def ludii_rotations(ctx, stype, nsites):
    cs = ctx.state().containerStates()[0]
    f = getattr(cs, f"rotation{stype}")
    out = []
    for j in range(nsites):
        try:
            out.append(int(f(j)))
        except Exception:
            out.append(0)
    return out


def declared_hand_map(cls):
    """LUDII_HAND_MAP declared on the game class or its module: list where entry k is the index, in the FLATTENED
    state, of the field that holds the piece total of Ludii container k+1 (Hand1, Hand2, ...), or -1 if untracked.
    Declared, never learned (learning was under-determined: spire 09-22)."""
    import sys as _sys
    m = None; c = cls
    while c is not None and m is None:
        m = getattr(c, "LUDII_HAND_MAP", None)
        if m is None:
            m = getattr(_sys.modules.get(c.__module__), "LUDII_HAND_MAP", None)
        c = getattr(c, "__wrapped_class__", None)
    if m is None:
        return None
    try:
        return [int(x) for x in m]
    except Exception:
        return None


def ludii_board(ctx, stype, n):
    """Per-site (who, count) lists for container 0. On stacked boards `who` is the top
    piece's owner and `count` is the STACK SIZE (Ludii's count is 1 for any stack), so a move
    that takes a piece off one stack and puts it on another changes both sites."""
    cs = ctx.state().containerStates()[0]
    who = getattr(cs, f"who{stype}")
    stacking = bool(ctx.game().isStacking())
    cnt = getattr(cs, f"sizeStack{stype}") if stacking else getattr(cs, f"count{stype}")
    return [int(who(i)) for i in range(n)], [int(cnt(i)) for i in range(n)]


def _flatten_state(state):
    """Python state -> flat list of ints, or None if not representable.

    A leaf that is not a number used to abandon the whole state, and with it the board: Conspirateurs
    keeps its phase as the STRING "Placement"/"Movement" and its multi-hop path as a frozenset, so the
    certifier could not read its 289-cell board at all, found no mapping, and the row has carried the
    reason code `time-cap` since September -- a budget that is not the reason (the mapping search takes
    six seconds and the episode one). Non-numeric leaves are therefore encoded instead of fatal: a string
    by a stable small integer, a set by its sorted members. Games that flattened before flatten to exactly
    the same list, because this only adds handling for the leaves that used to return None.
    """
    import numpy as _np
    if isinstance(state, _np.ndarray):
        try:
            return [int(x) for x in state.ravel().tolist()]
        except (TypeError, ValueError):
            return None
    if isinstance(state, (str, bytes)):
        return [_str_code(state)]
    if isinstance(state, (set, frozenset)):
        out = []
        for x in sorted(state, key=repr):
            if isinstance(x, (tuple, list, set, frozenset, str, bytes)):
                sub = _flatten_state(x)
                if sub is None:
                    return None
                out.extend(sub)
            else:
                try:
                    out.append(int(x))
                except (TypeError, ValueError):
                    return None
        return out
    if isinstance(state, (tuple, list)):
        out = []
        for x in state:
            if isinstance(x, (tuple, list, set, frozenset, _np.ndarray, str, bytes)):
                sub = _flatten_state(x)
                if sub is None:
                    return None
                out.extend(sub)
            else:
                try:
                    out.append(int(x))
                except (TypeError, ValueError):
                    return None
        return out
    return None


_STR_CODES = {}


def _str_code(s):
    """A stable small integer for a string leaf (a phase name, a piece label).

    Stable within a run and order-independent: the code is the string's own hash folded into a small
    range, not a counter, so two runs that visit the states in a different order agree. Only used to
    make the state comparable to itself over time -- nothing is compared against Ludii through it.
    """
    import hashlib
    k = s if isinstance(s, str) else s.decode("utf-8", "replace")
    if k not in _STR_CODES:
        _STR_CODES[k] = -1000 - (int(hashlib.md5(k.encode()).hexdigest()[:6], 16) % 1000)
    return _STR_CODES[k]


def _l4_candidates(flat_len, nsites):
    """Board-window hypotheses: (offset, mode). mode 'who' compares occupancy
    via a functional piece-label map; mode 'count' compares site counts."""
    offs = {0}
    if flat_len - nsites > 0:
        offs.add(flat_len - nsites)
    # a state can hold SEVERAL board-sized blocks (cram: an anchor block and a coverage block, 36 each in a
    # 73-field state). Only 0 and flat-nsites were offered, so the second block at offset 36 was unreachable.
    offs.update(k * nsites for k in range(1, 1 + (flat_len // nsites)) if k * nsites + nsites <= flat_len)
    return [{"off": o, "mode": m, "vmap": {}}
            for o in sorted(offs) for m in ("who", "count")]


def _l4_check(cands, flat, nsites, T, who, cnt):
    """Filter candidate hypotheses against one observed board. Mutates vmaps."""
    keep = []
    for c in cands:
        win = flat[c["off"]:c["off"] + nsites]
        if len(win) < nsites:
            continue
        ok = True
        for i, v in enumerate(win):
            j = T(i)
            if j is None or not (0 <= j < nsites):
                ok = False
                break
            if c["mode"] == "count":
                if v != cnt[j]:
                    ok = False
                    break
            else:
                w = who[j]
                if c["vmap"].setdefault(v, w) != w:
                    ok = False
                    break
        if ok:
            keep.append(c)
    return keep


def settle_forced_passes(game, ctx, trial, max_passes=3):
    """If Ludii's only legal moves are Pass moves, apply them (up to
    max_passes) and return trial.over(). Ludii ends a no-moves game only after
    both players have passed; the board itself is unchanged by passes."""
    for _ in range(max_passes):
        if trial.over():
            return True
        legal = game.moves(ctx).moves()
        n = int(legal.size())
        if n == 0:
            return bool(trial.over())
        moves = [legal.get(i) for i in range(n)]
        if not all(bool(m.isPass()) for m in moves):
            return False          # Ludii still has real moves: genuine mismatch
        game.apply(ctx, moves[0])
    return bool(trial.over())


def _level(r):
    l3, l4, l5 = (r.get(k) or "" for k in ("l3", "l4", "l5"))
    if not l3.startswith("PASS"):
        return "L2-"
    if not l4.startswith("PASS"):
        return "L3"
    if not l5.startswith("PASS"):
        return "L4"
    return "L5"


_RANK = {"L2-": 2, "L3": 3, "L4": 4, "L5": 5}
# ---------------- Effect-based L3 (encoding-independent) ---------------------
# The syntactic L3 compares (from, to) keys, which presupposes that the
# translation encodes moves like Ludii does. 173/305 translations get no
# mapping at ply 0; most of them encode moves differently (destination-only
# ints, tagged tuples, jump paths) or use pit/stack boards. Here a move is
# identified by its EFFECT: the set of board sites whose owner or count changes
# when it is applied. Effects are comparable across encodings once a site
# mapping T is fixed, and they cover placement, movement, captures, multi-jumps
# and sowing alike. Ludii effects come from applying each legal move to a
# copied Context; Python effects from step() on the immutable state.

_COUNT_ONLY_CACHE = {}


def count_only_board(game):
    """Is this board's position fully described by the COUNT vector alone?

    True when the board carries exactly one real component and that component belongs to the Shared
    player. Then there is one piece type and no ownership, so a position is just "how many seeds in each
    pit" -- `who` and `what` add nothing.

    They do worse than nothing. On such a board Ludii's `who` is not ownership at all: it is a record of
    whether a move has ever written the cell. `(set Count 10 to:(sites Track))` at the start leaves who=0
    on pits that plainly hold ten seeds, and who flips to the shared owner the first time a piece is moved
    in. Measured over 200 plies on each of the ten corpus games in this class: 2,024 cell-observations with
    count>0 and who=0 (adjiboto 453, ako_okwe 381, adi 130, adidada 130, altiev 110, awari 110, oware 110,
    and three games with none), and NOT ONE with count=0 and who!=0. Ako Okwe's recorded L3 failure is
    exactly this artefact -- sowing a single seed back into its own pit leaves the count at 1 while who
    changes, so the reference has a move a count-only translation cannot express, and no rule in that
    translation is wrong.

    Dropping `who` here is not a loosening. A key that is a COMPLETE description of the position cannot
    merge two moves that lead to different positions; it can only stop separating moves that lead to the
    same one, which is correct. Scope is derived from the equipment, never declared per game: pentomino and
    trax are Shared too but carry 12 and 2 component types, so `what` still distinguishes their pieces and
    they are excluded.
    """
    # NEVER key this on id(game): a Java proxy that has been collected lets the next game land on the same
    # id, and the cache then answers for the wrong game. Caught 2026-10-08 with the two answers exactly
    # swapped -- Hex (two Markers) read True and Awari (one Shared Seed) read False -- which would have
    # corrupted every row of the next full re-certification.
    try:
        key = str(game.name())
    except Exception:
        key = None
    if key is not None and key in _COUNT_ONLY_CACHE:
        return _COUNT_ONLY_CACHE[key]
    ok = False
    try:
        comps = game.equipment().components()
        real = []
        for i in range(len(comps)):
            c = comps[i]
            if c is None or str(c.name()) == "Disc":      # the default placeholder
                continue
            real.append(c)
        ok = len(real) == 1 and int(real[0].owner()) == int(game.players().count()) + 1
    except Exception:
        ok = False
    if key is not None:
        _COUNT_ONLY_CACHE[key] = ok
    return ok


def ludii_visibility(game, ctx, stype, nsites):
    """Per-site visibility as a bitmask over players, or None when the game hides nothing.

    Banqi deals all 32 pieces face down and the first move of every game is a reveal. Ludii's context
    holds the true identity of a face-down piece -- the hiding is a per-player VIEW, kept in separate
    flags -- so `who`, `count` and `what` are unchanged by a reveal and all 32 of Ludii's opening moves
    came out with the same (empty) effect, while the translation, which encodes face-down in the cell
    itself, produced 32 distinct ones. The game could not be compared at ply 0 and had sat at L2- with
    no mapping since the corpus was first certified, under the reason code `stochastic-start`; the random
    deal is real but it is not what was blocking the comparison.

    This is the same kind of gap as the stack size and the tile rotation: a state component the reference
    exposes and the certifier was not reading. `None` for every game whose `hiddenInformation()` is false,
    so the effect signature of the other 303 is bit-for-bit what it was.
    """
    if not bool(game.hiddenInformation()):
        return None
    import jpype
    st = getattr(jpype.JClass("game.types.board.SiteType"), stype)
    cs = ctx.state().containerStates()[0]
    np_ = int(game.players().count())
    out = []
    for i in range(nsites):
        v = 0
        for p in range(1, np_ + 1):
            try:
                if cs.isHiddenWhat(p, i, 0, st):
                    v |= 1 << (p - 1)
            except Exception:
                return None
        out.append(v)
    return out


def declared_options(module):
    """`LUDII_OPTIONS` declared by the translation: the Ludii ruleset it actually implements, or None.

    Hermit's `.lud` offers eleven boards and defaults to a 61-vertex triangular hex; the translation is a
    5x5 grid, i.e. it implements `Board Type/Square 5`. Certifying it against the default compared two
    different games, and the row sat at L2- with no mapping for a month under a reason code that turned
    out to be wrong twice over. Which ruleset a translation implements is a fact only the translation can
    state, exactly like its site, hand, score and rotation maps, so it states it -- and then every path
    (the ladder, the sweep, the position dump) compares against the right reference without being told.

    This is not a variant: a variant is a second artifact beside a canonical one that certifies on its
    own. This is a correction to what the row is compared against, of the same kind as the board mapping.
    """
    import importlib, sys as _sys
    try:
        m = importlib.import_module(module) if isinstance(module, str) else module
    except Exception:
        return None
    v = getattr(m, "LUDII_OPTIONS", None)
    if not v:
        # follow the operator wrappers down to the translation, like every other declaration: a game
        # carrying repair operators is `games.ops.<key>`, whose class delegates to the original
        c = find_boardgame_class(m)
        while c is not None and not v:
            v = getattr(c, "LUDII_OPTIONS", None) or getattr(_sys.modules.get(c.__module__), "LUDII_OPTIONS", None)
            c = getattr(c, "__wrapped_class__", None)
    return [str(x) for x in v] if v else None


def declared_hidden_values(cls):
    """`LUDII_HIDDEN_VALUES` declared on the translation: the cell values that mean "face down", or None.

    Occupancy is all the agreement check can compare without knowing the translation's numbering, and on
    a game with face-down pieces that is not enough: Banqi's cell integer carries owner, piece type AND
    whether the piece is turned up, so the two engines can hold the same pieces on the same squares and
    still disagree about which of them are visible -- a difference that shows up only as a move-set
    difference several plies later, which is exactly the shape of the four diagnoses withdrawn on
    2026-10-06. Which values mean face down is a fact about the translation's encoding, so the
    translation states it, like the site, hand, score and rotation maps.
    """
    import sys as _sys
    v = None; c = cls
    while c is not None and v is None:
        v = getattr(c, "LUDII_HIDDEN_VALUES", None)
        if v is None:
            v = getattr(_sys.modules.get(c.__module__), "LUDII_HIDDEN_VALUES", None)
        c = getattr(c, "__wrapped_class__", None)
    return frozenset(int(x) for x in v) if v else None


def _ludii_effects(game, ctx, stype, nsites, lmoves, geo=None, inv=None):
    who0, cnt0 = ludii_board(ctx, stype, nsites)
    cnt_only = count_only_board(game)
    vis0 = ludii_visibility(game, ctx, stype, nsites)
    out = []
    for m in lmoves:
        c2 = Context(ctx)
        game.apply(c2, m)
        who1, cnt1 = ludii_board(c2, stype, nsites)
        eff = ({j for j in range(nsites) if cnt0[j] != cnt1[j]} if cnt_only else
               {j for j in range(nsites) if who0[j] != who1[j] or cnt0[j] != cnt1[j]})
        if vis0 is not None:
            vis1 = ludii_visibility(game, c2, stype, nsites)
            if vis1 is not None:
                eff |= {j for j in range(nsites) if vis0[j] != vis1[j]}
        if geo is not None:                     # expand tile placements to the cells they cover
            cs = c2.state().containerStates()[0]
            for j in list(eff):
                try:
                    w = int(getattr(cs, f"what{stype}")(j)); st = int(getattr(cs, f"state{stype}")(j))
                except Exception:
                    continue
                if w:
                    eff |= tile_footprint(game, stype, geo, inv, j, st, w)
        out.append(frozenset(eff))
    return out


def _python_effects(g, st, pylegal, nsites, off):
    f0 = _flatten_state(st)
    if f0 is None or len(f0) < off + nsites:
        return None
    w0 = f0[off:off + nsites]
    out = []
    for a in pylegal:
        try:
            s1, _, _ = g.step(st, a)
        except Exception:
            return None
        f1 = _flatten_state(s1)
        if f1 is None or len(f1) < off + nsites:
            return None
        w1 = f1[off:off + nsites]
        out.append(frozenset(i for i in range(nsites) if w0[i] != w1[i]))
    return out


def _geo_candidates_T(geo, nsites):
    """All site-mapping candidates T (python index -> ludii site) from geometry,
    without the syntactic legal-set test; named like candidate_mappings."""
    cands = []
    H = max(r for r, _ in geo) + 1
    Wg = max(c for _, c in geo) + 1
    for sname in _GEO_SYMS:
        if sname in ("transpose", "rot90", "rot270", "antitranspose") and H != Wg:
            continue
        cands.append((f"geo:{sname}", Wg, _geo_sym(geo, H, Wg, sname)))
    n = len(geo)
    for oname, keyf in (("rank:rc", lambda rc: (rc[0], rc[1])), ("rank:Rc", lambda rc: (-rc[0], rc[1])),
                        ("rank:rC", lambda rc: (rc[0], -rc[1])), ("rank:RC", lambda rc: (-rc[0], -rc[1])),
                        ("rank:cr", lambda rc: (rc[1], rc[0])), ("rank:Cr", lambda rc: (-rc[1], rc[0])),
                        ("rank:cR", lambda rc: (rc[1], -rc[0])), ("rank:CR", lambda rc: (-rc[1], -rc[0]))):
        order = [geo[rc] for rc in sorted(geo, key=keyf)]
        T = (lambda order: lambda cell: cell if (cell is None or cell < 0)
             else (order[cell] if 0 <= cell < len(order) else -1))(order)
        cands.append((oname, max(n, 1), T))
    return cands


def _map_sig(sig, T):
    return frozenset(T(i) for i in sig)


def _infer_bijection(peffs, leffs, nsites, max_nodes=200000, T0=None):
    """Learn a site bijection T (python index -> ludii site) such that every
    Python effect set maps onto a Ludii effect set (hypergraph matching by
    backtracking). Only used for small boards with non-singleton effects
    (mancala rings, sowing, multi-site moves), where the constraint is tight.
    Returns a dict or None."""
    if nsites > 40 or not peffs or all(len(e) <= 1 for e in peffs):
        return None
    P = sorted(set(peffs), key=lambda e: (-len(e), sorted(e)))
    Lby = {}
    for e in set(leffs):
        Lby.setdefault(len(e), []).append(e)
    if sorted(len(e) for e in P) != sorted(len(e) for e in set(leffs)):
        return None
    T = dict(T0 or {}); used = set(T.values()); nodes = [0]

    def assign(pset, lset):
        """try to extend T so that pset -> lset; return list of newly assigned keys or None"""
        new = []
        free_p = [i for i in pset if i not in T]
        mapped = {T[i] for i in pset if i in T}
        if not mapped <= lset:
            return None
        free_l = [j for j in lset if j not in mapped]
        if len(free_p) != len(free_l):
            return None
        # try all matchings of free_p onto free_l (small)
        import itertools
        for perm in itertools.permutations(free_l):
            ok = True; added = []
            for i, j in zip(free_p, perm):
                if j in used:
                    ok = False; break
                T[i] = j; used.add(j); added.append(i)
            if ok:
                return added
            for i in added:
                used.discard(T.pop(i))
        return None

    def rec(k):
        nodes[0] += 1
        if nodes[0] > max_nodes:
            return False
        if k == len(P):
            return True
        pset = P[k]
        for lset in Lby.get(len(pset), []):
            # cheap prune: already-mapped members must land inside lset
            if any(i in T and T[i] not in lset for i in pset):
                continue
            free_p = [i for i in pset if i not in T]
            if len(free_p) > 6:      # permutation blow-up guard
                continue
            added = assign(pset, lset)
            if added is None:
                continue
            if rec(k + 1):
                return True
            for i in added:
                used.discard(T.pop(i))
        return False

    return dict(T) if rec(0) else None




def _learn_bijection_by_play(game, stype, nsites, cls, off, plies=24, seed=3):
    """Grow a site bijection over several plies: at each ply learn from the
    current effect sets (constrained by what is known), then apply a pair of
    moves whose effects already agree under the partial mapping."""
    trial = Trial(game); ctx = Context(game, trial); game.start(ctx)
    g = cls(); st = _start_state(g, ctx, stype, nsites); rng = random.Random(seed)
    geo = ludii_geometry(game, stype)          # needed for the tile-footprint expansion
    T = {}
    for _ in range(plies):
        if trial.over():
            break
        legal = game.moves(ctx).moves()
        lmoves = [legal.get(i) for i in range(legal.size())]
        leff = _ludii_effects(game, ctx, stype, nsites, lmoves, geo, {v: k for k, v in (geo or {}).items()})
        pylegal = g.legal_actions(st)
        peff = _python_effects(g, st, pylegal, nsites, off)
        if peff is None or not pylegal or not lmoves:
            break
        grown = _infer_bijection(peff, leff, nsites, T0=T)
        if grown:
            T = grown
        if len(T) >= nsites:
            break
        # pick an aligned pair to advance
        pairs = [(pi, li) for pi, pe in enumerate(peff) if all(i in T for i in pe)
                 for li, le in enumerate(leff) if frozenset(T[i] for i in pe) == le]
        if not pairs:
            break
        pi, li = pairs[rng.randrange(len(pairs))]
        game.apply(ctx, lmoves[li])
        st, _, done = g.step(st, pylegal[pi])
        if done:
            break
    return T or None



def declared_site_map(cls):
    """LUDII_SITE_MAP declared on the game class or its module: list/tuple where entry i is the
    Ludii site index of the translation's cell i (-1 for cells with no Ludii counterpart)."""
    import sys as _sys
    m = None
    c = cls
    while c is not None and m is None:          # look through operator wrappers to the translation
        m = getattr(c, "LUDII_SITE_MAP", None)
        if m is None:
            m = getattr(_sys.modules.get(c.__module__), "LUDII_SITE_MAP", None)
        c = getattr(c, "__wrapped_class__", None)
    if m is None:
        return None
    try:
        return [int(x) for x in m]
    except Exception:
        return None


def effect_candidates(game, stype, nsites, geo, cls):
    """[(name, off, T)] consistent at the initial state under effect equality."""
    trial = Trial(game); ctx = Context(game, trial); game.start(ctx)
    legal = game.moves(ctx).moves()
    lmoves = [legal.get(i) for i in range(legal.size())]
    leff = set(_ludii_effects(game, ctx, stype, nsites, lmoves, geo, {v: k for k, v in (geo or {}).items()}))
    g = cls(); st = _start_state(g, ctx, stype, nsites); pylegal = g.legal_actions(st)
    flat = _flatten_state(st)
    if flat is None or not pylegal:
        return []
    # board window offsets: the front, the back, and the back minus 1..3 trailing fields that a
    # repair-operator wrapper may append (forced-pass counter, swap flag, turn counter)
    offs = sorted({0} | {max(0, len(flat) - nsites - k) for k in range(0, 4)})
    out = []
    decl = declared_site_map(cls)
    for off in offs:
        peff = _python_effects(g, st, pylegal, nsites, off)
        if peff is None:
            continue
        if decl is not None:
            # a translation may DECLARE its cell -> Ludii-site table (LUDII_SITE_MAP); it is a
            # candidate like any other and still has to survive the whole replay
            T = (lambda d: lambda cell: cell if (cell is None or cell < 0) else (d[cell] if 0 <= cell < len(d) else -1))(decl)
            if {_map_sig(sg, T) for sg in peff} == leff:
                out.append((f"effect:declared/off{off}", off, T))
        for name, W, T in _geo_candidates_T(geo, nsites):
            if {_map_sig(sg, T) for sg in peff} == leff:
                out.append((f"effect:{name}/off{off}", off, T))
        if not out:
            learned = _learn_bijection_by_play(game, stype, nsites, cls, off)
            if learned:
                # complete the partial bijection with unused sites in index order
                free = [j for j in range(nsites) if j not in learned.values()]
                full = dict(learned)
                for i in range(nsites):
                    if i not in full:
                        full[i] = free.pop(0) if free else -1
                T = (lambda d: lambda cell: None if cell is None else d.get(cell, -1))(full)
                if {_map_sig(sg, T) for sg in peff} == leff:
                    out.append((f"effect:learned/off{off}", off, T))
    return out


def _looks_pass(a):
    r = repr(a).lower()
    return a == -1 or r.startswith("-1") or r.startswith("(-1") or "pass" in r


def _ludii_winner(game, trial):
    """Winner as a Ludii player id (1..N) from trial.ranking(); 0 for a draw. N-player aware:
    the winner is the player with the best (lowest) rank, a draw when the best rank is shared."""
    n = int(game.players().count())
    rk = trial.ranking()
    ranks = {p: float(rk[p]) for p in range(1, n + 1)}
    best = min(ranks.values())
    tops = [p for p, r in ranks.items() if r == best]
    return 0 if len(tops) != 1 else tops[0]


def _winner_verdict(game, trial, reward, mover_ludii, pmap):
    """Compare Python's mover-relative terminal reward with Ludii's winner.
    reward > 0: mover won; == 0: draw; < 0: mover lost (for N > 2 the winner is then only
    known to be someone else, so PASS iff Ludii's winner is another player)."""
    n = int(game.players().count())
    lw = _ludii_winner(game, trial)
    if reward > 0:
        pw = mover_ludii
    elif reward == 0:
        pw = 0
    elif n == 2:
        others = [v for v in pmap.values() if v != mover_ludii]
        pw = others[0] if others else ({1, 2} - {mover_ludii}).pop()
    else:
        return ("PASS" if (lw != 0 and lw != mover_ludii) else f"FAIL:winner(py=not-{mover_ludii},ludii={lw})"), None, lw
    return ("PASS" if pw == lw else f"FAIL:winner(py={pw},ludii={lw})"), pw, lw


def _replay_effect(game, stype, nsites, cls, cand, max_steps, seed, time_limit, script=None, trace=None):
    """Full-game replay with effect-equality at every ply (L3), then the usual
    L4 (board) and L5 (termination/winner) checks. With `script` (a list of Python
    actions, e.g. a logged RL game) the Python side plays the script instead of a
    random Ludii move, so any trajectory can be differentially checked.

    `trace`: a dict to fill with BOTH engines' full position at the step an L3 check fails. Every hand-written
    replay of an ambiguous game re-creates the ambiguity it is meant to diagnose -- pairing moves by their
    changed cells alone put trax on a different board and produced a wrong diagnosis that had to be withdrawn
    (2026-10-06). The only admissible evidence is the position the CERTIFIER is in, so it dumps it itself."""
    import time
    t0 = time.time()
    name, off, T = cand
    res = {"l3": "PASS", "l4": None, "l5": None, "steps": 0, "mapping": name, "note": "",
           "counterexample": None}
    trial = Trial(game); ctx = Context(game, trial); game.start(ctx)
    g = cls(); st = _start_state(g, ctx, stype, nsites); rng = random.Random(seed)
    geo = ludii_geometry(game, stype)          # needed for the tile-footprint expansion
    l4c, l4_fail, pmap, pmap_ok, history = None, None, {}, True, []
    Tinv = {}
    # Off-board state by DECLARATION (LUDII_HAND_MAP): same-effect moves (add-from-hand vs release-from-pool) are told
    # apart by comparing the declared hand fields after the move with Ludii's container totals. Declared, never learned
    # (learning was under-determined and regressed spire, 2026-09-22). Merged from the dev oracle 2026-09-25 with PI
    # approval of the full re-certification it requires.
    hand_map = declared_hand_map(cls)
    score_map = declared_score_map(cls)

    def _score_score(s2_, c2_):
        """same shape as the hand term: reward a candidate whose declared score fields match the engine's"""
        if not score_map:
            return 0
        sc = ludii_scores(c2_); f2 = _flatten_state(s2_)
        if f2 is None:
            return 0
        out = 0
        for k, j in enumerate(score_map):
            if 0 <= j < len(f2) and k < len(sc):
                out += 4 if f2[j] == sc[k] else -4
        return out

    def _hand_score(s2_, c2_):
        if not hand_map:
            return 0
        h2 = ludii_hands(c2_); f2 = _flatten_state(s2_)
        if f2 is None:
            return 0
        sc = 0
        for k, j in enumerate(hand_map):
            if 0 <= j < len(f2) and k < len(h2):
                sc += 4 if f2[j] == h2[k] else -4
        return sc

    # Python cell value -> Ludii component id, learned from positions both engines agree on
    # (initial position included). Used to tell apart same-effect moves that differ only in
    # the piece TYPE placed (two marker kinds of one player look identical in who/count).
    from collections import Counter as _Counter
    wmap = {}

    def _learn_wmap(st_, ctx_):
        f = _flatten_state(st_)
        if f is None or len(f) < off + nsites:
            return
        what = ludii_what(ctx_, stype, nsites)
        for i in range(nsites):
            j = T(i)
            if j is None or not (0 <= j < nsites):
                continue
            v, wv = f[off + i], what[j]
            if v and wv:
                wmap.setdefault(v, _Counter())[wv] += 1

    def _wmap_of(v):
        c = wmap.get(v)
        return c.most_common(1)[0][0] if c else None

    # Declared per-site rotation (LUDII_ROTATION_MAP). The values cannot be compared directly -- trax counts
    # quarter turns 0..3 where Ludii indexes eighths 0, 2, 4, 6 -- and they cannot be LEARNED either: every
    # trax ply is ambiguous, so there is never an unambiguous position to learn the correspondence from (the
    # first attempt, 2026-10-03, left the map empty for the whole game and changed nothing). What does not
    # depend on either engine's numbering is the ORDER: list the moves of one ambiguity group by (piece type,
    # rotation) on both sides and the k-th is the k-th, for any correspondence that preserves order -- which
    # 0,1,2,3 -> 0,2,4,6 does. The declaration says WHERE the rotation is; the order does the rest.
    rot_map = declared_rotation_map(cls)

    # TRIED AND REVERTED (2026-10-06): also replaying with the rotation order REVERSED, in case the two
    # numberings run opposite ways. It changed nothing (trax identical at K=5 and K=30, same failing seeds),
    # so it is not carried -- a candidate that fixes nothing measured does not earn its cost.
    _rot_why = []
    # python rotation value -> ludii rotation value, learned from the BOARD, not from moves. The first
    # attempt learned it from unambiguous moves and stayed empty, because trax has almost none (14 of 15
    # plies are ambiguous). Cells both engines agree are occupied are a different matter: there are plenty
    # from ply 0, and they pin the correspondence (trax: 0->0, 1->2, 2->4, 3->6, settled by ply 4).
    rpair = {}
    # The ladder is consecutive: L4 (the board after every move) is only evaluated when L3 passes for the
    # whole game, so a run that stops at an L3 failure never reports that the boards had already diverged.
    # On a count-only board they can part many plies earlier and the move sets only notice later --
    # adidada's pit held 1 seed against Ludii's 13 at the failing ply, and the difference in the legal
    # moves there is a consequence, not a cause. The first board divergence is therefore recorded as it
    # happens, so the dump names the cause and not only the symptom.
    first_board_div = {}

    def _note_board(step_, st_, ctx_):
        if first_board_div or not count_only_board(game):
            return
        f_ = _flatten_state(st_)
        if not f_:
            return
        _w, c_ = ludii_board(ctx_, stype, nsites)
        for i_ in range(nsites):
            j_ = T(i_)
            if j_ is None or not (0 <= j_ < nsites) or off + i_ >= len(f_):
                continue
            if int(f_[off + i_]) != int(c_[j_]):
                first_board_div.update({"step": step_, "site": j_, "py_index": i_,
                                        "python": int(f_[off + i_]), "ludii_count": int(c_[j_])})
                return

    def _learn_rpair(st_, ctx_):
        if not rot_map:
            return
        f_ = _flatten_state(st_) or []
        rot_ = ludii_rotations(ctx_, stype, nsites)
        w_, _c = ludii_board(ctx_, stype, nsites)
        for i_ in range(min(nsites, len(rot_map))):
            j_ = T(i_); idx = rot_map[i_]
            if j_ is None or not (0 <= j_ < nsites) or not (0 <= idx < len(f_)):
                continue
            if not (off + i_ < len(f_) and f_[off + i_]) or not w_[j_]:
                continue
            rpair.setdefault(f_[idx], rot_[j_])

    def _agreement(st_, ctx_, cells, T_, Tinv_, off_):
        """Do the two engines still agree? Board first, then every declared off-board field.

        A difference in the legal MOVES is evidence about the rules only if both engines are standing in the
        same position. Twice on 2026-10-06 a move-set difference was written up as a rule defect when the
        boards had already diverged (adidada: Ludii 13 seeds where python had 1; ex_nihilo: 55 of 61 sites),
        and both had to be withdrawn. Board agreement is necessary but not sufficient: spice's colliding
        moves leave the SAME board and differ only in the scores stored before gravity, so the declared
        fields are checked too."""
        f_ = _flatten_state(st_) or []
        who_, cnt_ = ludii_board(ctx_, stype, nsites)
        what_ = ludii_what(ctx_, stype, nsites)
        cnt_only = count_only_board(game)
        dis = []
        for j in (cells or range(nsites)):
            i_ = Tinv_.get(j)
            if i_ is None or not (0 <= off_ + i_ < len(f_)):
                continue
            pv = f_[off_ + i_]
            # the translation's board window says "is this cell occupied, and by what", in its own numbering;
            # compare only what is comparable without knowing that numbering -- occupancy.
            #
            # EXCEPT on a count-only board, where the cell value IS the count and the two numbers are the
            # same quantity. Occupancy alone is far too weak there: at adidada's failing ply python held 1
            # seed in a pit where Ludii held 13, both non-zero, so this check reported agreement and the
            # move-set difference downstream (python captures, Ludii does not, because 1+1 is a capturing
            # count and 13+1 is not) read as a rule difference. It is not: the boards had diverged earlier.
            # That is the same error the module's docstring lists four times, committed inside the tool
            # written to prevent it (2026-10-16).
            if cnt_only:
                if int(pv) != int(cnt_[j]):
                    dis.append({"site": j, "py_index": i_, "python": pv, "ludii_count": cnt_[j],
                                "note": "count-only board: the cell value and Ludii's count are the "
                                        "same quantity"})
            elif bool(pv) != bool(who_[j] or cnt_[j]):
                dis.append({"site": j, "py_index": i_, "python": pv,
                            "ludii_who": who_[j], "ludii_count": cnt_[j], "ludii_what": what_[j]})
        out = {"boards_agree_on_occupancy": not dis, "board_disagreements": dis[:12],
               "board_comparison": "counts (count-only board)" if cnt_only else "occupancy"}
        hidden_vals = declared_hidden_values(type(g))
        vis_ = ludii_visibility(game, ctx_, stype, nsites) if hidden_vals else None
        if vis_ is not None:
            vdis = []
            for j in (cells or range(nsites)):
                i_ = Tinv_.get(j)
                if i_ is None or not (0 <= off_ + i_ < len(f_)):
                    continue
                pv = f_[off_ + i_]
                if bool(pv) and (pv in hidden_vals) != bool(vis_[j]):
                    vdis.append({"site": j, "py_index": i_, "python": pv,
                                 "python_face_down": pv in hidden_vals, "ludii_hidden_what": vis_[j]})
            out["engines_agree_on_visibility"] = not vdis
            out["visibility_disagreements"] = vdis[:12]
        hid = {}
        if hand_map:
            h = ludii_hands(ctx_)
            hid["hand"] = {"python": [f_[x] if 0 <= x < len(f_) else None for x in hand_map],
                           "ludii": list(h)[:len(hand_map)]}
        if score_map:
            sc = ludii_scores(ctx_)
            hid["score"] = {"python": [f_[x] if 0 <= x < len(f_) else None for x in score_map],
                            "ludii": list(sc)[:len(score_map)]}
        if rot_map:
            rot_ = ludii_rotations(ctx_, stype, nsites)
            pairs = []
            for j in (cells or range(nsites)):
                i_ = Tinv_.get(j)
                if i_ is None or i_ >= len(rot_map):
                    continue
                idx = rot_map[i_]
                pairs.append({"site": j, "py_index": i_,
                              "python_rotation": f_[idx] if 0 <= idx < len(f_) else None,
                              "ludii_rotation": rot_[j]})
            hid["rotation"] = pairs[:16]
        out["declared_hidden_state"] = hid or None
        out["encoding_note"] = (
            "`python` is flat[off + cell] of the FLATTENED state -- the board window the mapping selected, in "
            "the translation's own numbering. It is NOT necessarily the internal value the translation's own "
            "code works with: trax's window holds the tile TYPE while its functions take a packed "
            "1 + type*4 + rotation, and reading one as the other produced a withdrawn diagnosis on "
            "2026-10-07. Compare occupancy, or the declared fields, and read the translation before "
            "interpreting a raw value.")
        return out

    def _rot_rank_pick(sig_, cand_a_, lmoves_, leff_, mv_):
        """The python action matching the drawn Ludii move inside its ambiguity group, or None."""
        if not rot_map or len(cand_a_) < 2:
            return None
        grp = [m for m, s_ in zip(lmoves_, leff_) if s_ == sig_]
        if len(grp) != len(cand_a_):
            _rot_why.append({"why": "group-size-mismatch"})
            return None                       # the groups do not correspond: say nothing
        cells = sorted(sig_)

        def lkey(m):
            c3 = Context(ctx); game.apply(c3, m)
            rot = ludii_rotations(c3, stype, nsites); wh = ludii_what(c3, stype, nsites)
            return tuple((wh[j], rot[j]) for j in cells if 0 <= j < nsites)

        def pkey_(a):
            s3, _, _ = g.step(st, a); f3 = _flatten_state(s3)
            if f3 is None or len(f3) < off + nsites:
                return None
            out = []
            for j in cells:
                i = Tinv.get(j)
                if i is None or not (0 <= i < nsites) or i >= len(rot_map):
                    continue
                idx = rot_map[i]
                v = f3[off + i]
                out.append(((_wmap_of(v) or v), f3[idx] if 0 <= idx < len(f3) else 0))
            return tuple(out)
        lk = [lkey(m) for m in grp]
        pk = [pkey_(a) for a in cand_a_]
        # With the correspondence known, match the KEYS directly. Order matching cannot survive a duplicate
        # -- trax step 14 offers the same resulting position twice (one per hand source), so the group is
        # {A, B, B, C} on both sides and position-in-order is undefined, which is exactly where the two
        # engines started holding different tiles. Key matching is indifferent to duplicates: the candidates
        # sharing a key are the same position, so any of them is right.
        if rpair and not any(x is None for x in pk):
            want = lkey(mv_)
            hits = [a for a, k in zip(cand_a_, pk)
                    if tuple((w, rpair.get(r)) for w, r in k) == want]
            if hits:
                return hits[0]
        if any(x is None for x in pk) or len(set(lk)) != len(lk) or len(set(pk)) != len(pk):
            _rot_why.append({"why": ("python-unreadable" if any(x is None for x in pk)
                                     else "ludii-tie" if len(set(lk)) != len(lk) else "python-tie"),
                             "ludii_keys": [str(x) for x in lk], "python_keys": [str(x) for x in pk],
                             "distinct_ludii": len(set(lk)), "distinct_python": len(set(pk)),
                             "cells": cells[:8]})
            return None                       # ties on either side: the order does not identify a move
        order_l = sorted(range(len(grp)), key=lambda i: lk[i])
        order_p = sorted(range(len(cand_a_)), key=lambda i: pk[i])
        try:
            pos = order_l.index(next(i for i, m in enumerate(grp) if m is mv_))
        except StopIteration:
            return None
        return cand_a_[order_p[pos]]
    _learn_wmap(st, ctx)
    _learn_rpair(st, ctx)
    for step in range(max_steps):
        if time.time() - t0 > time_limit:
            res.update(steps=step, l5="UNREACHED:timeout"); break
        if trial.over():
            res.update(steps=step, l5=f"FAIL:term@{step}(ludii-over-first)"); break
        legal = game.moves(ctx).moves()
        lmoves = [legal.get(i) for i in range(legal.size())]
        leff = _ludii_effects(game, ctx, stype, nsites, lmoves, geo, {v: k for k, v in (geo or {}).items()})
        pylegal = g.legal_actions(st)
        peff = _python_effects(g, st, pylegal, nsites, off)
        if peff is None:
            res.update(l3=f"FAIL@{step}", steps=step, note="python state not comparable"); break
        _note_board(step, st, ctx)
        pm = [_map_sig(sg, T) for sg in peff]
        lset, pset = set(leff), set(pm)
        if lset != pset:
            if not Tinv:
                for i in range(nsites):
                    j = T(i)
                    if j is not None and j >= 0:
                        Tinv.setdefault(j, i)
            back = lambda x: sorted(Tinv.get(j, j) for j in x)
            if trace is not None:
                who_, cnt_ = ludii_board(ctx, stype, nsites)
                what_ = ludii_what(ctx, stype, nsites)
                rot_ = ludii_rotations(ctx, stype, nsites)
                # the per-player visibility, where the game has any: without it a reader cannot tell why
                # Ludii offers a move onto an occupied site, since its rule is written on what the MOVER
                # can see (`(not (is Hidden What at:(to) to:Mover))`)
                vis_ = ludii_visibility(game, ctx, stype, nsites)
                fl_ = _flatten_state(st) or []
                cells = sorted({c for x in (lset ^ pset) for c in x})
                trace.update({
                    "step": step, "seed": seed, "mapping": name, "off": off,
                    "history": [repr(a) for a in history],
                    "divergent_cells_ludii": cells,
                    "ludii": {str(j): {"who": who_[j], "count": cnt_[j], "what": what_[j], "rotation": rot_[j],
                                       **({"hidden_what": vis_[j]} if vis_ is not None else {})}
                              for j in cells if 0 <= j < nsites},
                    "python": {str(j): {"cell": (fl_[off + Tinv[j]] if j in Tinv and off + Tinv[j] < len(fl_) else None),
                                        "py_index": Tinv.get(j)} for j in cells},
                    # the DIFFERENCE, not a prefix of each side. Truncating the two lists separately and
                    # reading them as disjoint is a trap on a high-branching game: quantum_leap's swap
                    # phase has hundreds of moves, and the first 24 of each side share nothing simply
                    # because they are sorted differently (2026-10-08).
                    "n_ludii_moves": len(lmoves), "n_python_moves": len(pylegal),
                    "ludii_moves_here": [{"effect": sorted(e)} for e in sorted(lset - pset)][:24],
                    "python_moves_here": [{"effect": sorted(sg), "action": repr(a)}
                                          for a, sg in zip(pylegal, pm) if sg in (pset - lset)][:24],
                    "n_ludii_only": len(lset - pset), "n_python_only": len(pset - lset),
                    "n_in_common": len(lset & pset),
                    "note": ("the position the CERTIFIER is in at the failing step, with its own mapping and its "
                             "own tie-breaks already applied. Cells are Ludii site ids; `python.py_index` is the "
                             "translation's own cell index for the same site."),
                    # Two verdicts the reader must not have to derive, because deriving them was skipped twice
                    # (2026-10-06, adidada and ex_nihilo) and both readings had to be withdrawn.
                    "first_board_divergence": (first_board_div or None),
                    **_agreement(st, ctx, cells, T, Tinv, off)})
            res.update(l3=f"FAIL@{step}", steps=step, note="effect-set diverge",
                       counterexample={"level": "L3", "step": step, "seed": seed,
                                       "history": [repr(a) for a in history],
                                       "moves_ludii_allows_but_python_lacks_as_changed_cells":
                                           [back(x) for x in list(lset - pset)[:8]],
                                       "moves_python_allows_but_ludii_forbids_as_changed_cells":
                                           [back(x) for x in list(pset - lset)[:8]],
                                       "python_legal_actions_now": [repr(a) for a in pylegal[:40]],
                                       "note": ("moves are identified by the SET OF BOARD CELLS (python "
                                                "indices into state[off:off+n]) they change; a move that "
                                                "changes different cells is a different move"),
                                       "encoding": name})
            break
        p_ = g.current_player(st); lm = int(ctx.state().mover())
        if p_ in pmap:
            pmap_ok = pmap_ok and pmap[p_] == lm
        elif lm in pmap.values():
            pmap_ok = False
        else:
            pmap[p_] = lm
        if script is not None:
            if step >= len(script):
                res.update(steps=step, l5="UNREACHED:script-end"); break
            a_s = script[step]
            if a_s not in pylegal:
                res.update(l3=f"FAIL@{step}", steps=step, note="script action not legal in python",
                           counterexample={"level": "L3", "step": step, "seed": seed, "history": [repr(a) for a in history],
                                           "script_action": repr(a_s), "encoding": name}); break
            sig = pm[pylegal.index(a_s)]
            idxs = [i for i, s_ in enumerate(leff) if s_ == sig]
            idx = idxs[0]
            if len(idxs) > 1:      # several Ludii moves with this effect: the one whose post-move board matches Python's
                s2, _, _ = g.step(st, a_s); f2 = _flatten_state(s2); w = f2[off:off + nsites] if f2 else []
                best = -1
                for i in idxs:
                    c2 = Context(ctx); game.apply(c2, lmoves[i])
                    who2, cnt2 = ludii_board(c2, stype, nsites); what2 = ludii_what(c2, stype, nsites)
                    sc = sum(1 for k in range(nsites) if T(k) is not None and 0 <= T(k) < nsites and k < len(w)
                             and (w[k] == cnt2[T(k)] or w[k] == who2[T(k)] or w[k] == what2[T(k)]))
                    sc += 5 if bool(lmoves[i].isPass()) == _looks_pass(a_s) else 0
                    if sc > best:
                        best, idx = sc, i
            mv, sig = lmoves[idx], leff[idx]
            cand_a = [a_s]; ambiguous = False
        else:
            idx = rng.randrange(len(lmoves))
            mv, sig = lmoves[idx], leff[idx]
            cand_a = [a for a, s_ in zip(pylegal, pm) if s_ == sig]
            ambiguous = len(cand_a) > 1
        if len(cand_a) > 1:
            # Two moves can change the same SET of sites (two 11+-seed sowings
            # both touch all pits). Disambiguate by the board AFTER the move:
            # apply the Ludii move to a copy and pick the Python action whose
            # resulting window matches it under the current L4 hypotheses.
            if not Tinv:
                for i_ in range(nsites):
                    j_ = T(i_)
                    if j_ is not None and j_ >= 0:
                        Tinv.setdefault(j_, i_)
            by_rot = _rot_rank_pick(sig, cand_a, lmoves, leff, mv)
            if trace is not None and rot_map:
                grp_ = [m for m, s_ in zip(lmoves, leff) if s_ == sig]
                trace.setdefault("tiebreak_log", []).append(
                    {"step": step, "candidates": len(cand_a), "ludii_group": len(grp_),
                     "by_rot_used": by_rot is not None,
                     "why_not": (_rot_why[-1] if by_rot is None and _rot_why else None)})
            c2 = Context(ctx); game.apply(c2, mv)
            who2, cnt2 = ludii_board(c2, stype, nsites)
            what2 = ludii_what(c2, stype, nsites)
            best_a, best_score = cand_a[0], -1
            cnt_only_b = count_only_board(game)
            verified = []
            for a in cand_a:
                s2, _, _ = g.step(st, a)
                f2 = _flatten_state(s2)
                if f2 is None or len(f2) < off + nsites:
                    continue
                w = f2[off:off + nsites]
                score = sum(1 for i in range(nsites) if T(i) is not None and 0 <= T(i) < nsites
                            and (w[i] == cnt2[T(i)] or w[i] == who2[T(i)]))
                # piece-type agreement on the changed sites outweighs who/count agreement; with no
                # learned evidence for a value, fall back to direct id equality (many translations
                # number piece types exactly as Ludii's component ids)
                score += 3 * sum(1 for i in range(nsites) if T(i) is not None and 0 <= T(i) < nsites and w[i]
                                 and _wmap_of(w[i]) is not None and _wmap_of(w[i]) == what2[T(i)])
                score += 2 * sum(1 for i in range(nsites) if T(i) is not None and 0 <= T(i) < nsites and w[i]
                                 and _wmap_of(w[i]) is None and w[i] == what2[T(i)])
                # two board-neutral moves (a Pass and a Skip/SetNextPlayer) have the same empty effect:
                # match Ludii's pass flag with a pass-looking Python action (-1 or 'pass')
                score += 5 if bool(mv.isPass()) == _looks_pass(a) else 0
                score += _hand_score(s2, c2)          # declared off-board fields after the move (hand vs pool source)
                score += _score_score(s2, c2)         # declared score fields (azteka: the round-1 count that survives the reset)
                score += 8 if (by_rot is not None and a is by_rot) else 0   # declared rotation, matched by order
                if score > best_score:
                    best_a, best_score = a, score
                # A score is not a verification. The pairing is only right if the two engines stand on
                # the SAME board afterwards, so check that, and prefer a candidate that passes it over a
                # candidate that merely scores well. Where the cell value is the count (a mancala pit),
                # the two numbers are the same quantity and can be compared directly; otherwise occupancy
                # is all that is comparable without knowing the translation's encoding.
                agree = True
                for i in range(nsites):
                    j = T(i)
                    if j is None or not (0 <= j < nsites):
                        continue
                    if cnt_only_b:
                        if int(w[i]) != int(cnt2[j]):
                            agree = False; break
                    elif bool(w[i]) != bool(who2[j] or cnt2[j]):
                        agree = False; break
                if agree:
                    verified.append(a)
            if verified:
                # one or more candidates land on Ludii's board; the score only breaks ties among those
                cand_a = [verified[0]] if len(verified) == 1 else [
                    best_a if best_a in verified else verified[0]]
            elif len(cand_a) > 1:
                # none of them does. Taking the best score here is what parted the boards in adidada and
                # ako_okwe, and the move-set difference that surfaced one or two plies later was read as a
                # rule difference three times. Say so instead (2026-10-16).
                if trace is not None:
                    trace.update({"step": step, "seed": seed, "mapping": name, "off": off,
                                  "history": [repr(a) for a in history],
                                  "ambiguous_candidates": [repr(a) for a in cand_a][:12],
                                  "note": "no python action reproduces the drawn Ludii move's board"})
                res.update(l3=f"AMBIGUOUS@{step}", steps=step,
                           note="tie-break unresolved: no candidate reproduces the drawn move's board",
                           counterexample={"level": "L3", "step": step, "seed": seed,
                                           "history": [repr(a) for a in history],
                                           "candidates": [repr(a) for a in cand_a][:12],
                                           "encoding": name})
                break
            else:
                cand_a = [best_a]
        game.apply(ctx, mv)
        history.append(cand_a[0])
        st, reward, done = g.step(st, cand_a[0])
        _learn_rpair(st, ctx)
        if trace is not None and callable(trace.get("probe")):
            # An OBSERVER called once per ply with the certifier's own Ludii context, for questions that
            # can only be answered on the position the ladder is actually in. It cannot steer anything --
            # it receives the context and returns a value that is appended to trace["probe_log"] -- so it
            # gives a game-specific measurement without reopening the hand-replay door that four withdrawn
            # diagnoses came through (see pipeline/oracle_position.py).
            try:
                trace.setdefault("probe_log", []).append(
                    {"step": step, "value": trace["probe"](game, ctx, st, step)})
            except Exception as e:
                trace.setdefault("probe_log", []).append({"step": step, "error": f"{type(e).__name__}: {e}"})
        if trace is not None and rot_map:
            # Per-ply rotation audit. The effect sets can agree ply after ply while the two engines put
            # differently-ORIENTED tiles down, because a matching set does not say which member was played;
            # the divergence then surfaces much later as a cascade of the wrong length (trax, ply 15). This
            # records the first ply at which no 1-to-1 correspondence between the two rotation numberings
            # explains the board, which is where to look -- not where the ladder reports the failure.
            if not Tinv:
                for i_ in range(nsites):
                    j_ = T(i_)
                    if j_ is not None and j_ >= 0:
                        Tinv.setdefault(j_, i_)
            f_ = _flatten_state(st) or []
            rot_ = ludii_rotations(ctx, stype, nsites)
            w_, c_ = ludii_board(ctx, stype, nsites)
            pairs = {}
            bad = []
            for i_ in range(min(nsites, len(rot_map))):
                j_ = T(i_); idx = rot_map[i_]
                if j_ is None or not (0 <= j_ < nsites) or not (0 <= idx < len(f_)):
                    continue
                if not (off + i_ < len(f_) and f_[off + i_]) or not w_[j_]:
                    continue                       # only cells both engines agree are occupied
                pv, lv = f_[idx], rot_[j_]
                if pv in pairs and pairs[pv] != lv:
                    bad.append({"cell": i_, "site": j_, "python_rot": pv,
                                "ludii_rot": lv, "already_paired_with": pairs[pv]})
                else:
                    pairs.setdefault(pv, lv)
            au = trace.setdefault("rotation_audit", [])
            if bad and not any(a.get("inconsistent") for a in au):
                au.append({"step": step, "inconsistent": True, "pairs_so_far": dict(pairs),
                           "conflicts": bad[:6],
                           "note": ("FIRST ply at which the two rotation numberings cannot both be right: a "
                                    "python rotation value is paired with two different Ludii values. The "
                                    "engines are holding differently-oriented tiles from here on.")})
            elif not au or au[-1].get("step", -1) < step - 3:
                au.append({"step": step, "inconsistent": False, "pairs_so_far": dict(pairs)})
        if l4_fail is None:
            if not ambiguous:
                _learn_wmap(st, ctx)      # never learn piece types from a guessed tie-break (circular)
            flat = _flatten_state(st)
            if flat is None or len(flat) < off + nsites:
                l4_fail = "UNTESTABLE:state-shape"
            else:
                who, cnt = ludii_board(ctx, stype, nsites)
                if l4c is None:
                    # the L4 board window need not be the window the EFFECTS use. Expanding tile
                    # placements moves the effect window to the block that records coverage, which carries
                    # no owner, so pinning L4 to it drops domineering from L5 to L3 (2026-09-30). Offer L4
                    # every plausible window and let it keep the ones that match.
                    # ADD windows, never replace: sher_bakr's L4 window is off1, which is neither 0 nor
                    # flat-nsites nor a multiple of nsites, and replacing the effect window with a fixed
                    # candidate set dropped it (L5 -> L3 in v8, 2026-10-01). The effect window stays first.
                    l4c = ([{"off": off, "mode": m_, "vmap": {}} for m_ in ("who", "count")]
                           + [dict(c_, vmap={}) for c_ in _l4_candidates(len(flat), nsites) if c_["off"] != off])
                l4c = _l4_check(l4c, flat, nsites, T, who, cnt)
                if not l4c:
                    l4_fail = f"FAIL@{step}"
        lo = bool(trial.over())
        if done and not lo:
            lo = settle_forced_passes(game, ctx, trial)
        if lo or done:
            res["steps"] = step + 1

            def _final_boards():
                """both boards in Python cell order, so the repairer can see what the end position was"""
                flat = _flatten_state(st)
                if flat is None:
                    return {}
                who_, cnt_ = ludii_board(ctx, stype, nsites)
                if nsites > 400:
                    # a big board used to return NOTHING, so an L5 counterexample on trax (1681 sites, ~30 of
                    # them occupied) reached the repair agent with no position at all. Send the occupied cells
                    # instead of the whole grid: sparse is the natural form for a boardless game.
                    what_ = ludii_what(ctx, stype, nsites)
                    rot_ = ludii_rotations(ctx, stype, nsites)
                    occ = sorted({i for i in range(nsites)
                                  if (off + i < len(flat) and flat[off + i])
                                  or (T(i) is not None and 0 <= T(i) < nsites and who_[T(i)])})
                    if not Tinv:
                        for i_ in range(nsites):
                            j_ = T(i_)
                            if j_ is not None and j_ >= 0:
                                Tinv.setdefault(j_, i_)
                    return {"occupied_cells": {
                                str(i): {"python": flat[off + i] if off + i < len(flat) else None,
                                         # the translation's OWN rotation at this cell, where it declares one:
                                         # without it a tile game's end position cannot be reconstructed at all
                                         # (trax, 2026-10-07 -- a reconstruction from the type block alone read
                                         # the wrong tiles and the diagnosis had to be withdrawn)
                                         "python_rotation": (flat[rot_map[i]]
                                                             if rot_map and i < len(rot_map) and 0 <= rot_map[i] < len(flat)
                                                             else None),
                                         "ludii_site": T(i),
                                         "ludii_who": who_[T(i)] if T(i) is not None and 0 <= T(i) < nsites else None,
                                         "ludii_what": what_[T(i)] if T(i) is not None and 0 <= T(i) < nsites else None,
                                         "ludii_rotation": rot_[T(i)] if T(i) is not None and 0 <= T(i) < nsites else None}
                                for i in occ[:400]},
                            "board_note": (f"{len(occ)} occupied cells of {nsites} (sparse: the full grid is too "
                                           "large to print). Keys are PYTHON cell indices."),
                            **_agreement(st, ctx, sorted(Tinv), T, Tinv, off)}
                lw_ = [who_[T(i)] if T(i) is not None and 0 <= T(i) < nsites else None for i in range(nsites)]
                if not Tinv:
                    for i_ in range(nsites):
                        j_ = T(i_)
                        if j_ is not None and j_ >= 0:
                            Tinv.setdefault(j_, i_)
                return {"python_board": list(flat[off:off + nsites]), "ludii_owner": lw_,
                        "board_note": "python_board = the state's board window; ludii_owner = 0 empty / 1 P1 / 2 P2 at the same Python cell index",
                        **_agreement(st, ctx, sorted(Tinv), T, Tinv, off)}

            def _ce(**payload):
                # the effect path never built one, so an L5 failure here left the rule agent with nothing to
                # work from -- azteka's recorded next method ('second run seeded with the failing seed's
                # counterexample') was unrunnable for that reason alone (found 2026-09-25, review #49)
                return {"level": "L5", "step": step, "seed": seed, "history": [repr(a) for a in history],
                        "mapping": name, **payload, **_final_boards()}
            if lo != done:
                res["l5"] = f"FAIL:term@{step}({'ludii' if lo else 'python'}-only)"
                res["counterexample"] = res["counterexample"] or _ce(
                    kind="termination", ended_in=("ludii" if lo else "python"), last_action=repr(history[-1]) if history else None,
                    python_reward=reward, ludii_status=str(trial.status()),
                    python_legal_after=[repr(a) for a in g.legal_actions(st)[:12]],
                    note="one engine ended and the other did not, at the same ply")
            elif not pmap_ok or p_ not in pmap:
                res["l5"] = "UNTESTABLE:player-map"
            else:
                res["l5"], pw, lw = _winner_verdict(game, trial, reward, pmap[p_], pmap)
                if str(res["l5"]).startswith("FAIL"):
                    res["counterexample"] = res["counterexample"] or _ce(
                        kind="winner", last_action=repr(history[-1]) if history else None, python_mover=p_,
                        python_reward=reward, python_winner_ludii_id=pw, ludii_winner=lw,
                        ludii_status=str(trial.status()), player_map=dict(pmap),
                        note="reward is from the mover's perspective (base.py); both engines ended at the same ply")
            break
    else:
        res.update(steps=max_steps, l5="UNREACHED:max-steps")
    if res["l3"].startswith("PASS"):
        res["l4"] = l4_fail or (f"PASS(off{l4c[0]['off']}/{l4c[0]['mode']})" if l4c else "UNTESTABLE:no-steps")
    return res


MAX_MAPPING_CANDIDATES = 40


def check_history(module, ludii_name, history, mapping_hint=None, max_steps=8000, time_limit=600.0, options=None):
    """Differentially check ONE given trajectory (Python actions) against Ludii under the game's
    certified mapping: L3 at every ply, L4 after every ply, L5 at the end (or UNREACHED:script-end).
    Used for logged games from stronger policies (RL self-play) that random play never reaches."""
    game = load_game(ludii_name, options)
    stype, nsites = ludii_site_info(game)
    geo = ludii_geometry(game, stype)
    mod = importlib.import_module(module)
    cls = find_boardgame_class(mod)
    cand = None
    if mapping_hint and mapping_hint.startswith("effect:"):
        for c in effect_candidates(game, stype, nsites, geo, cls):
            if c[0] == mapping_hint:
                cand = c; break
    if cand is None:
        trial = Trial(game); ctx = Context(game, trial); game.start(ctx)
        legal = game.moves(ctx).moves()
        lset0 = {_norm(int(legal.get(i).fromNonDecision()), int(legal.get(i).toNonDecision())) for i in range(legal.size())}
        for m in candidate_mappings(cls().legal_actions(cls().reset()), lset0, geo):
            if mapping_hint is None or _mapstr(m) == mapping_hint:
                # syntactic mapping: replay through the syntactic path (meta-moves such as swap are keyed, not effect-matched)
                r = _replay(game, stype, nsites, cls, m, max_steps, 0, time_limit, script=list(history))
                l3, l4, l5 = (r.get(k) or "" for k in ("l3", "l4", "l5"))
                r["level"] = "L2-" if not l3.startswith("PASS") else "L3" if not l4.startswith("PASS") else "L4" if not l5.startswith("PASS") else "L5"
                return r
    if cand is None:
        ec = effect_candidates(game, stype, nsites, geo, cls)
        if not ec:
            return {"l3": "SKIP:no-mapping", "l4": None, "l5": None, "steps": 0, "mapping": None, "level": "L2-"}
        cand = ec[0]
    r = _replay_effect(game, stype, nsites, cls, cand, max_steps, 0, time_limit, script=list(history))
    l3, l4, l5 = (r.get(k) or "" for k in ("l3", "l4", "l5"))
    r["level"] = "L2-" if not l3.startswith("PASS") else "L3" if not l4.startswith("PASS") else "L4" if not l5.startswith("PASS") else "L5"
    return r


def certify_run(module, ludii_name, max_steps=600, seed=1, time_limit=60.0, mapping_hint=None, options=None,
                trace=None):
    """One full-game replay under the best of the board mappings that are
    consistent at the initial state (placement games cannot disambiguate
    symmetries on an empty board; the mapping is existentially quantified).
    Returns {'l3','l4','l5','steps','mapping','note','counterexample'}.
    l3: PASS | FAIL@k | SKIP:<reason>
    l4: PASS(<hypothesis>) | FAIL@k | UNTESTABLE:<reason>   (only meaningful if l3 PASS)
    l5: PASS | FAIL:term@k | FAIL:winner | UNTESTABLE:<reason> | UNREACHED:<why>
    """
    # the ruleset the translation says it implements, unless the caller names one (a variant run does)
    if options is None:
        options = declared_options(module)
    res = {"l3": "PASS", "l4": None, "l5": None, "steps": 0, "mapping": None, "note": "",
           "counterexample": None, "options": options,
           "options_declared": options is not None and declared_options(module) == options}
    game = load_game(ludii_name, options)
    stype, nsites = ludii_site_info(game)
    geo = ludii_geometry(game, stype)
    mod = importlib.import_module(module)
    cls = find_boardgame_class(mod)
    if cls is None:
        return {**res, "l3": "SKIP:no-class"}
    trial = Trial(game); ctx = Context(game, trial); game.start(ctx)
    legal = game.moves(ctx).moves()
    lset0 = {_norm(int(legal.get(i).fromNonDecision()), int(legal.get(i).toNonDecision()))
             for i in range(legal.size())}
    pylegal0 = cls().legal_actions(cls().reset())
    cands = candidate_mappings(pylegal0, lset0, geo)
    # Candidates: syntactic (from,to)-key mappings first, then encoding-
    # independent effect mappings. Effect candidates are tried whenever the
    # syntactic ones are absent OR none of them survives L3 (a syntactic
    # mapping can agree at the empty board by coincidence and fail at ply 1).
    syn = [("syn", m) for m in cands[:MAX_MAPPING_CANDIDATES]]
    if mapping_hint is not None:
        h = [c for c in syn if _mapstr(c[1]) == mapping_hint]
        syn = h or syn
    best = None
    tried = 0
    for kind, m in syn:
        r = _replay(game, stype, nsites, cls, m, max_steps, seed, time_limit)
        r["_mapping_obj"] = _mapstr(m); tried += 1
        if best is None or (_RANK[_level(r)], r["steps"]) > (_RANK[_level(best)], best["steps"]):
            best = r
        if _level(r) == "L5":
            break
    if best is None or _RANK[_level(best)] < 3:
        ec = effect_candidates(game, stype, nsites, geo, cls)
        if mapping_hint is not None:
            h = [c for c in ec if c[0] == mapping_hint]
            ec = h or ec
        for c in ec[:MAX_MAPPING_CANDIDATES]:
            # carry the observer into the per-candidate trace: certify_run gives each mapping candidate a
            # FRESH dict (so a rejected candidate's dump cannot be mistaken for the adopted one), which
            # silently dropped trace["probe"] and made the hook a no-op on its first use
            _t = None if trace is None else ({"probe": trace["probe"]} if callable(trace.get("probe")) else {})
            r = _replay_effect(game, stype, nsites, cls, c, max_steps, seed, time_limit, trace=_t)
            r["_mapping_obj"] = c[0]; tried += 1
            if best is None or (_RANK[_level(r)], r["steps"]) > (_RANK[_level(best)], best["steps"]):
                best = r
                if trace is not None:
                    # only the ADOPTED candidate's position is evidence: a rejected mapping fails for its own
                    # reasons and handing that dump to a repair agent is the stale-counterexample trap again
                    _probe = trace.get("probe")
                    trace.clear(); trace.update(_t or {})
                    if _probe is not None:
                        trace["probe"] = _probe
            if _level(r) == "L5":
                break
    if best is None:
        return {**res, "l3": "FAIL@0", "note": "no consistent mapping (syntactic or effect)"}
    best["mappings_tried"] = tried
    return best


def _replay(game, stype, nsites, cls, mapping, max_steps, seed, time_limit, script=None):
    import time
    t0 = time.time()
    res = {"l3": "PASS", "l4": None, "l5": None, "steps": 0, "mapping": _mapstr(mapping),
           "note": "", "counterexample": None}
    trial = Trial(game); ctx = Context(game, trial); game.start(ctx)
    g = cls(); st = _start_state(g, ctx, stype, nsites); rng = random.Random(seed)
    ncells, s, W, T = mapping
    Tinv = {}
    for i in range(W * W):
        j = T(i)
        if j is not None and j >= 0:
            Tinv.setdefault(j, i)
    l4c = None            # surviving L4 hypotheses (None until first check)
    l4_fail = None
    pmap = {}             # python player id -> ludii player id
    pmap_ok = True
    history = []          # python actions applied so far (for counterexamples)

    def ce(level, **payload):
        """Counterexample in PYTHON coordinates."""
        return {"level": level, "step": len(history), "seed": seed,
                "history": [repr(a) for a in history], **payload}

    def to_py(keys):
        return sorted(((None if f is None else Tinv.get(f, f),
                        None if t is None else Tinv.get(t, t)) for f, t in keys),
                      key=lambda k: (k[0] is None, -1 if k[0] is None else k[0], k[1] is None, -1 if k[1] is None else k[1]))

    for step in range(max_steps):
        if time.time() - t0 > time_limit:
            res.update(steps=step, l5="UNREACHED:timeout")
            break
        if trial.over():
            # Ludii ended but Python did not (Python end returns inside loop)
            res.update(steps=step, l5=f"FAIL:term@{step}(ludii-over-first)")
            break
        legal = game.moves(ctx).moves()
        lmoves = [legal.get(i) for i in range(legal.size())]
        lset = {_norm(int(m.fromNonDecision()), int(m.toNonDecision())) for m in lmoves}
        pylegal = g.legal_actions(st)
        ps = _pset(pylegal, ncells, T)
        if ps != lset:
            res.update(l3=f"FAIL@{step}", steps=step, note="legal-set diverge",
                       counterexample=ce("L3", ludii_only=to_py(lset - ps)[:12],
                                         python_only=to_py(ps - lset)[:12],
                                         encoding=_mapstr(mapping)))
            break
        # player correspondence (needed for the winner comparison)
        pm = g.current_player(st)
        lm = int(ctx.state().mover())
        if pm in pmap:
            pmap_ok = pmap_ok and (pmap[pm] == lm)
        elif lm in pmap.values():
            pmap_ok = False
        else:
            pmap[pm] = lm
        # advance both engines with the same Ludii move (or the scripted Python action)
        if script is not None:
            if step >= len(script):
                res.update(steps=step, l5="UNREACHED:script-end"); break
            a_s = script[step]
            if a_s not in pylegal:
                res.update(l3=f"FAIL@{step}", steps=step, note="script action not legal in python",
                           counterexample=ce("L3", script_action=repr(a_s), encoding=_mapstr(mapping))); break
            k_ = pkey(a_s, ncells)
            key = (None if k_[0] is None else T(k_[0]), None if k_[1] is None else T(k_[1]))
            hits = [m for m in lmoves if _norm(int(m.fromNonDecision()), int(m.toNonDecision())) == key]
            hits = _by_state(hits, a_s)
            if not hits:
                res.update(l3=f"FAIL@{step}", steps=step, note="python move absent in ludii",
                           counterexample=ce("L3", python_only=to_py({key}), ludii_only=[], encoding=_mapstr(mapping))); break
            mv = hits[0]
            game.apply(ctx, mv)
            cand = [a_s]
        else:
            mv = lmoves[rng.randrange(len(lmoves))]
            key = _norm(int(mv.fromNonDecision()), int(mv.toNonDecision()))
            game.apply(ctx, mv)
            cand = [a for a in pylegal
                    if (lambda k: (None if k[0] is None else T(k[0]),
                                   None if k[1] is None else T(k[1])))(pkey(a, ncells)) == key]
            cand = _by_move_state(cand, mv)
        if not cand:
            res.update(l3=f"FAIL@{step}", steps=step, note="ludii move absent",
                       counterexample=ce("L3", ludii_only=to_py({key}), python_only=[],
                                         encoding=_mapstr(mapping)))
            break
        history.append(cand[0])
        st, reward, done = g.step(st, cand[0])
        # ---- L4: compare post-move board ----
        if l4_fail is None:
            flat = _flatten_state(st)
            if flat is None or len(flat) < nsites:           # stacked boards: who = top owner, count = stack size
                l4_fail = "UNTESTABLE:state-shape"
            else:
                who, cnt = ludii_board(ctx, stype, nsites)
                if l4c is None:
                    l4c = _l4_candidates(len(flat), nsites)
                l4c = _l4_check(l4c, flat, nsites, T, who, cnt)
                if not l4c:
                    l4_fail = f"FAIL@{step}"
                    # Ludii board expressed in Python cell order (off0 window)
                    lw_ = [who[T(i)] if T(i) is not None and 0 <= T(i) < nsites else None
                           for i in range(nsites)]
                    lc_ = [cnt[T(i)] if T(i) is not None and 0 <= T(i) < nsites else None
                           for i in range(nsites)]
                    res["counterexample"] = res["counterexample"] or ce(
                        "L4", last_action=repr(cand[0]),
                        python_board=flat[:nsites],
                        ludii_owner=lw_, ludii_count=lc_,
                        note=("python_board is state[0:n_sites]; ludii_owner is "
                              "0=empty/1=P1/2=P2 at the same cell index; "
                              "ludii_count is pieces per site (mancala/stacks)"))
        # ---- L5: termination sync + winner ----
        lo = bool(trial.over())
        if done and not lo:
            # Ludii convention: a "no legal moves" ending is reached only after
            # forced Pass moves by both sides (Tic-Tac-Toe full board: over
            # becomes true two plies later). Python conversions end at once.
            # Let Ludii settle through forced passes before calling it a
            # termination mismatch.
            lo = settle_forced_passes(game, ctx, trial)
        if lo or done:
            res["steps"] = step + 1

            def final_boards():
                """Both boards in Python cell order (helps the repairer see
                which stones connect / what was captured at the end)."""
                flat = _flatten_state(st)
                if flat is None or nsites > 400:
                    return {}
                who, cnt = ludii_board(ctx, stype, nsites)
                lw_ = [who[T(i)] if T(i) is not None and 0 <= T(i) < nsites else None
                       for i in range(nsites)]
                return {"python_board": flat[:nsites], "ludii_owner": lw_,
                        "board_note": ("python_board = state[0:n_sites]; ludii_owner = "
                                       "0 empty / 1 P1 / 2 P2 at the same Python cell index")}

            if lo != done:
                res["l5"] = f"FAIL:term@{step}({'ludii' if lo else 'python'}-only)"
                nxt = 0 if lo else int(game.moves(ctx).moves().size())
                res["counterexample"] = res["counterexample"] or ce(
                    "L5", last_action=repr(cand[0]),
                    ludii_over=lo, python_done=done, python_reward=reward,
                    ludii_status=str(trial.status()) if lo else None,
                    ludii_legal_moves_after=nxt,
                    python_legal_moves_after=len(g.legal_actions(st)) if not done else 0,
                    **final_boards())
            elif not pmap_ok or pm not in pmap:
                res["l5"] = "UNTESTABLE:player-map"
            else:
                res["l5"], pw, lw = _winner_verdict(game, trial, reward, pmap[pm], pmap)
                if pw != lw:
                    res["counterexample"] = res["counterexample"] or ce(
                        "L5", last_action=repr(cand[0]), python_mover=pm,
                        python_reward=reward, python_winner_ludii_id=pw,
                        ludii_status=str(trial.status()), ludii_winner=lw,
                        player_map=dict(pmap),
                        note="reward is from the mover's perspective (base.py)",
                        **final_boards())
            break
    else:
        res.update(steps=max_steps, l5="UNREACHED:max-steps")
    if res["l3"].startswith("PASS"):
        if l4_fail:
            res["l4"] = l4_fail
        elif l4c is None:
            res["l4"] = "UNTESTABLE:no-steps"
        else:
            h = l4c[0]
            res["l4"] = f"PASS(off{h['off']}/{h['mode']})"
    return res


def select_mapping(module, ludii_name, seeds=(1, 2, 3), max_steps=600, time_limit=60.0, options=None):
    """Return the mapstr of the candidate whose WORST result over `seeds` is best
    (rank, then steps survived); None if no candidate is consistent at ply 0."""
    game = load_game(ludii_name, options)
    stype, nsites = ludii_site_info(game)
    geo = ludii_geometry(game, stype)
    cls = find_boardgame_class(importlib.import_module(module))
    if cls is None:
        return None
    trial = Trial(game); ctx = Context(game, trial); game.start(ctx)
    legal = game.moves(ctx).moves()
    lset0 = {_norm(int(legal.get(i).fromNonDecision()), int(legal.get(i).toNonDecision()))
             for i in range(legal.size())}
    cands = candidate_mappings(cls().legal_actions(cls().reset()), lset0, geo)
    best, best_score = None, None

    def consider(name, run):
        nonlocal best, best_score
        worst = None
        for sd in seeds:
            r = run(sd)
            sc = (_RANK[_level(r)], r["steps"])
            worst = sc if worst is None or sc < worst else worst
            if sc[0] < 3:
                break
        if best_score is None or worst > best_score:
            best, best_score = name, worst

    for m in cands[:MAX_MAPPING_CANDIDATES]:
        consider(_mapstr(m), lambda sd, m=m: _replay(game, stype, nsites, cls, m, max_steps, sd, time_limit))
        if best_score and best_score[0] >= 5:
            break
    if best_score is None or best_score[0] < 3:
        for c in effect_candidates(game, stype, nsites, geo, cls)[:MAX_MAPPING_CANDIDATES]:
            consider(c[0], lambda sd, c=c: _replay_effect(game, stype, nsites, cls, c, max_steps, sd, time_limit))
            if best_score and best_score[0] >= 5:
                break
    return best


def certify(module, ludii_name, episodes=3, max_steps=600, time_limit=60.0, options=None, seeds=None, extend_to=8000,
            deadline=None):
    """Aggregate certify_run over `episodes` seeds; a level PASSes only if it
    passes in every episode. Returns the merged dict + 'level' (strict ladder:
    highest consecutive level passed, counting UNTESTABLE as a stop), plus
      first_fail_episode: {l3,l4,l5: episode index (1-based) of the first
                           non-PASS verdict, or None}  -> leak-rate-vs-K curves
      lattice: 'L5' | 'L5(L4 untestable)' | 'L4' | 'L3' | 'L3(L4 untestable)' | 'L2-'
      episodes_run: number of episodes actually replayed
    Certification is relative to this budget; report K with every level."""
    import time as _time
    t_start = _time.time()
    agg = None
    first_fail = {"l3": None, "l4": None, "l5": None}
    ran = 0
    budget_cut = False
    # ONE mapping for all episodes: score every initial-state-consistent
    # candidate on the first min(3, episodes) seeds (worst case over seeds),
    # hold the best. Per-episode re-selection would let different symmetries
    # pass different episodes (too lenient); holding the first lucky candidate
    # would fail later episodes (too strict).
    # `seeds`: explicit seed list (e.g. the known failing seed first, then 1, 2) for a repair loop;
    # default is 1..episodes. first_fail_episode is reported as the SEED, so it stays comparable.
    seed_list = list(seeds) if seeds else list(range(1, episodes + 1))
    if options is None:                      # the ruleset the translation declares it implements
        options = declared_options(module)
    hint = select_mapping(module, ludii_name, seeds=tuple(seed_list[:3]),
                          max_steps=max_steps, time_limit=time_limit, options=options)
    for ep, seed in enumerate(seed_list):
        # `deadline`: wall-clock cap for the whole certification (slow games); the verdict is then
        # relative to the episodes actually run, reported in episodes_run / budget_cut
        if deadline and ep and _time.time() - t_start > deadline:
            budget_cut = True; break
        r = certify_run(module, ludii_name, max_steps=max_steps,
                        seed=seed, time_limit=time_limit, mapping_hint=hint, options=options)
        # Long games (Ludii's own draw cap is 1250 turns per player): an episode that has not ended
        # within max_steps is replayed once with a larger budget so L5 can be observed instead of
        # being reported UNREACHED. Only the non-terminating episodes pay for this.
        if extend_to and extend_to > max_steps and str(r.get("l5", "")).startswith("UNREACHED:max-steps"):
            r = certify_run(module, ludii_name, max_steps=extend_to,
                            seed=seed, time_limit=time_limit * 4, mapping_hint=hint, options=options)
        r.pop("_mapping_obj", None)
        ran += 1
        if agg is None:
            agg = r
        else:
            for k in ("l3", "l4", "l5"):
                a, b = agg[k] or "", r[k] or ""
                if not b.startswith("PASS"):     # keep the worst verdict
                    if a.startswith("PASS") or not a:
                        agg[k] = r[k]
                        if r.get("counterexample") and (
                                agg.get("counterexample") is None
                                or r["counterexample"]["level"] < agg["counterexample"]["level"]):
                            agg["counterexample"] = r["counterexample"]
            agg["steps"] = max(agg["steps"], r["steps"])
        for k in ("l3", "l4", "l5"):
            v = r[k] or ""
            if first_fail[k] is None and v and not v.startswith("PASS") \
                    and not v.startswith("UNTESTABLE") and not v.startswith("UNREACHED"):
                first_fail[k] = seed
        if not r["l3"].startswith("PASS"):
            break                                # ladder is consecutive
    # the row must say which ruleset it was certified against: a level earned under a non-default option
    # is not comparable with one earned under the default, and the reader cannot tell from the level
    agg["options"] = options
    l3, l4, l5 = (agg[k] or "" for k in ("l3", "l4", "l5"))
    if not l3.startswith("PASS"):
        agg["level"] = "L2-"                     # below move-level fidelity
    elif not l4.startswith("PASS"):
        agg["level"] = "L3"
    elif not l5.startswith("PASS"):
        agg["level"] = "L4"
    else:
        agg["level"] = "L5"
    l4_untestable = l4.startswith("UNTESTABLE")
    if agg["level"] == "L3" and l4_untestable and l5.startswith("PASS"):
        agg["lattice"] = "L5(L4 untestable)"
    elif agg["level"] == "L3" and l4_untestable:
        agg["lattice"] = "L3(L4 untestable)"
    else:
        agg["lattice"] = agg["level"]
    agg["first_fail_episode"] = first_fail
    agg["episodes_run"] = ran
    agg["episodes_budget"] = episodes
    if budget_cut:
        agg["budget_cut"] = f"deadline {deadline}s reached after {ran} episodes"
    return agg


if __name__ == "__main__":
    # self-test on known cases
    import json, sys
    sys.path.insert(0, str(SRC))
    conv = {c["module"].split(".")[-1]: c.get("lud_path", "")
            for c in json.load(open(ROOT / "results" / "conversion_results.json"))
            if c.get("module", "").startswith("games.converted.") and c.get("lud_path")}
    tests = ["hex", "squava", "english_draughts", "jeson_mor", "tandems", "tic_tac_toe"]
    for key in tests:
        lp = conv.get(key)
        if not lp:
            print(f"{key:18s} (lud unknown)"); continue
        r = oracle_run(f"games.converted.{key}", Path(lp).name, max_steps=8)
        print(f"{key:18s} faithful={r['faithful']} step={r.get('step')} "
              f"map={r.get('mapping')} note={r.get('note')} "
              f"l_only={r.get('ludii_only')} p_only={r.get('python_only')}")
