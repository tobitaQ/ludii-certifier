"""
Greedy agent: takes as many objects as possible from the largest heap.
"""
from base import Agent, State, Action


class GreedyAgent(Agent):
    def select_action(self, state: State, legal_actions: list[Action]) -> Action:
        # Take the maximum amount from the largest heap
        return max(legal_actions, key=lambda a: (a[1], state[a[0]]))
