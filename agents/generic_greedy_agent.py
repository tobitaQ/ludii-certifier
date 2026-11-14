"""Generic 2-ply greedy for any BoardGame (2026-09-21): take an immediate win, avoid a move that hands the
opponent an immediate win, otherwise random. Replaces the RandomAgent(seed=99) fallback that
pipeline.make_greedy_agent returned for every translation (the 'greedy' seat of the game-property
certificate was a second random seed — review #13/#14)."""
import random
from base import Agent, State, Action


class GenericGreedyAgent(Agent):
    def __init__(self, game, seed: int | None = 99, max_branch: int = 400):
        self.g = game; self.rng = random.Random(seed); self.max_branch = max_branch

    def select_action(self, state: State, legal_actions: list[Action]) -> Action:
        g = self.g
        if len(legal_actions) > self.max_branch:            # huge action spaces (pentomino): sample
            legal_actions = self.rng.sample(legal_actions, self.max_branch)
        safe, me = [], g.current_player(state)
        for a in legal_actions:
            try:
                s2, r, done = g.step(state, a)
            except Exception:
                continue
            if done:
                if r > 0: return a                          # immediate win
                if r < 0: continue                          # immediate loss (suicide move)
                safe.append(a); continue                    # draw: acceptable
            # does the opponent have an immediate win from s2?
            try:
                opp = g.current_player(s2)
                if opp == me:                               # move-again: treat as safe
                    safe.append(a); continue
                la2 = g.legal_actions(s2)
                if len(la2) > self.max_branch: la2 = self.rng.sample(la2, self.max_branch)
                gives_win = False
                for b in la2:
                    s3, r3, d3 = g.step(s2, b)
                    if d3 and r3 > 0: gives_win = True; break
                if not gives_win: safe.append(a)
            except Exception:
                safe.append(a)
        pool = safe if safe else legal_actions
        return self.rng.choice(pool)
