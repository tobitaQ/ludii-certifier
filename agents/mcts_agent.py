"""
MCTS agent with UCB1 selection (2-player zero-sum games).

Backprop convention: each node stores wins from the perspective of the
player who *moved into* that node (i.e., the player at the parent state).
"""
import math
import random

from base import Agent, State, Action, BoardGame


class _Node:
    __slots__ = ("visits", "wins", "children", "untried", "player_who_moved")

    def __init__(self, untried: list[Action], player_who_moved: int):
        self.visits = 0
        self.wins = 0.0
        self.children: dict[Action, "_Node"] = {}
        self.untried = list(untried)
        self.player_who_moved = player_who_moved  # player whose action led here


class MCTSAgent(Agent):
    def __init__(self, game: BoardGame, iterations: int = 1000, c: float = 1.414, seed: int | None = None,
                 temperature: float = 0.0, winrate_cap: float | None = None,
                 rollout_cap: int | None = None):
        self.game = game
        self.iterations = iterations
        self.c = c
        self.rng = random.Random(seed)
        # rollout_cap (int) truncates each random playout to that many plies and
        # scores the leaf with the game's own `heuristic(state) -> P(player0 wins)`
        # instead of playing to the end. This bounds the cost of games with long
        # playouts (Othello/Checkers), whose full rollouts are ~1-2s/move. If the
        # game carries `mcts_rollout_cap`, that is the default. Games without a
        # heuristic ignore the cap (fall back to a full playout) so behaviour is
        # unchanged. rollout_cap=None + no game attr => original full playout.
        self._rollout_cap = rollout_cap if rollout_cap is not None else getattr(game, "mcts_rollout_cap", None)
        self._heuristic = getattr(game, "heuristic", None)
        # Difficulty knobs (defaults == original argmax behaviour, so the
        # pipeline/agents that don't pass them are unaffected):
        #   temperature  (A) 0 = always strongest (argmax visits). >0 samples
        #                among MCTS-preferred moves (visit^(1/T)); higher = weaker
        #                but still believable (no random blunders).
        #   winrate_cap  (C) None = play to win. e.g. 0.75 = once our estimated
        #                win-rate exceeds the cap, stop piling on (prefer moves
        #                that keep it near/under the cap) WITHOUT throwing the
        #                game (floor 0.5) -> avoids demoralising blow-outs, does
        #                NOT manufacture comebacks (that would be active DDA).
        self.temperature = temperature
        self.winrate_cap = winrate_cap
        # Set after each select_action: estimated win-rate (AI's perspective) of
        # the chosen and the strongest move + per-move map. Lets the GUI log the
        # advantage trajectory (the structural flow/comeback signal) for free.
        self.last_eval: dict | None = None

    def select_action(self, state: State, legal_actions: list[Action]) -> Action:
        dummy_player = self.game.current_player(state)
        root = _Node(legal_actions, player_who_moved=1 - dummy_player)

        for _ in range(self.iterations):
            node = root
            s = state
            path: list[_Node] = [root]
            done = False

            # --- Selection ---
            while not node.untried and node.children and not done:
                action = self._ucb_select(node, s)
                s, _, done = self.game.step(s, action)
                node = node.children[action]
                path.append(node)

            # --- Expansion ---
            if not done and node.untried:
                action = self.rng.choice(node.untried)
                node.untried.remove(action)
                mover = self.game.current_player(s)
                s, _, done = self.game.step(s, action)
                next_legal = [] if done else self.game.legal_actions(s)
                child = _Node(next_legal, player_who_moved=mover)
                node.children[action] = child
                node = child
                path.append(node)

            # --- Simulation (rollout, optionally depth-capped) ---
            # v0 = player 0's win share of this leaf, in [0, 1] (None => no signal).
            # Terminal leaves credit the last mover (the existing convention);
            # a depth-capped leaf is scored by the game heuristic instead.
            cap = self._rollout_cap if self._heuristic is not None else None
            if done:
                v0 = 1.0 if node.player_who_moved == 0 else 0.0
            else:
                sim_s = s
                last_mover = node.player_who_moved
                depth = 0
                capped = False
                while True:
                    acts = self.game.legal_actions(sim_s)
                    if not acts:
                        break                                   # terminal: last mover wins
                    if cap is not None and depth >= cap:
                        capped = True
                        break                                   # depth cap: heuristic below
                    a = self.rng.choice(acts)
                    last_mover = self.game.current_player(sim_s)
                    sim_s, _, sim_done = self.game.step(sim_s, a)
                    depth += 1
                    if sim_done:
                        break                                   # terminal: last mover wins
                if capped:
                    v0 = self._heuristic(sim_s)                 # P(player 0 wins), in [0, 1]
                else:
                    v0 = 1.0 if last_mover == 0 else 0.0

            # --- Backpropagation --- (fractional; exactly the binary rule when v0 in {0,1})
            for n in path:
                n.visits += 1
                if v0 is not None:
                    n.wins += v0 if n.player_who_moved == 0 else (1.0 - v0)

        return self._choose(root)

    def _choose(self, root: _Node) -> Action:
        children = root.children
        # Root children store wins for the AI (the player to move at the root),
        # so wins/visits is the AI's estimated win-rate after that move.
        wr = {a: (n.wins / n.visits if n.visits else 0.0) for a, n in children.items()}
        vis = {a: n.visits for a, n in children.items()}
        best = max(children, key=lambda a: vis[a])

        # (C) anti-blow-out: when already winning past the cap, restrict to moves
        # that keep us in [0.5, cap] (never throw below 0.5). Only bites when ahead.
        cand = list(children)
        if self.winrate_cap is not None and wr[best] > self.winrate_cap:
            capped = [a for a in cand if 0.5 <= wr[a] <= self.winrate_cap]
            if capped:
                cand = capped

        # (A) weakening: temperature-sample over visit counts among candidates.
        if self.temperature <= 0:
            chosen = max(cand, key=lambda a: vis[a])
        else:
            w = [vis[a] ** (1.0 / self.temperature) for a in cand]
            tot = sum(w)
            chosen = (self.rng.choices(cand, weights=w)[0] if tot > 0
                      else max(cand, key=lambda a: vis[a]))

        self.last_eval = {"chosen": chosen, "chosen_winrate": wr[chosen],
                          "best": best, "best_winrate": wr[best],
                          "per_move": {a: wr[a] for a in children}}
        return chosen

    def _ucb_select(self, node: _Node, state: State) -> Action:
        mover = self.game.current_player(state)
        log_n = math.log(node.visits + 1)

        def ucb(action: Action) -> float:
            child = node.children[action]
            exploit = child.wins / (child.visits + 1e-8)
            explore = self.c * math.sqrt(log_n / (child.visits + 1e-8))
            return exploit + explore

        return max(node.children, key=ucb)
