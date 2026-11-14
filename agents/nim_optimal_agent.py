"""
Optimal Nim agent using the Sprague-Grundy theorem (Nim-sum / XOR strategy).

A position is a losing position (P-position) iff XOR of all heap sizes == 0.
On each turn, find a move that leaves XOR == 0 for the opponent.
If already in a P-position, falls back to taking 1 from the largest heap.
"""
import random
from functools import reduce

from base import Agent, State, Action


class NimOptimalAgent(Agent):
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select_action(self, state: State, legal_actions: list[Action]) -> Action:
        heaps = list(state[:-1])           # any heap count; last field is the player
        nim_sum = reduce(lambda a, b: a ^ b, heaps)

        if nim_sum == 0:
            # Already in P-position (losing); take 1 from largest heap as fallback
            largest = max(range(len(heaps)), key=lambda i: heaps[i])
            return (largest, 1)

        # Find a move that results in nim_sum == 0
        for i, size in enumerate(heaps):
            target = size ^ nim_sum  # what this heap should become
            if target < size:
                return (i, size - target)

        # Should never reach here
        return self.rng.choice(legal_actions)
