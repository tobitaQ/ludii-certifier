import random
from base import Agent, State, Action


class RandomAgent(Agent):
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select_action(self, state: State, legal_actions: list[Action]) -> Action:
        return self.rng.choice(legal_actions)
