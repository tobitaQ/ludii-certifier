"""
Greedy agent for Connect Four: win > block > center preference.
Delegates to game.greedy_action() which implements the same logic.
"""
from base import Agent, State


class ConnectFourGreedyAgent(Agent):
    def __init__(self, game):
        self.game = game

    def select_action(self, state: State, legal_actions: list) -> int:
        return self.game.greedy_action(state, legal_actions)
