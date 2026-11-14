"""
Game runner: plays n games between two agents and collects turn-level logs.
"""
import math
import time
from dataclasses import dataclass, field

from base import BoardGame, Agent, State, Action


@dataclass
class TurnLog:
    turn: int
    player: int
    legal_actions: list[Action]
    chosen_action: Action
    state_before: State
    state_after: State
    reward: float
    time_ms: float


@dataclass
class GameLog:
    game_id: int
    winner: int          # 0 or 1; -1 if draw
    n_turns: int
    turns: list[TurnLog] = field(default_factory=list)


def play_game(game: BoardGame, agents: list[Agent], game_id: int = 0) -> GameLog:
    state = game.reset()
    log = GameLog(game_id=game_id, winner=-1, n_turns=0)
    turn = 0

    while True:
        player = game.current_player(state)
        legal = game.legal_actions(state)
        if not legal:
            break

        t0 = time.perf_counter()
        action = agents[player].select_action(state, legal)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        next_state, reward, done = game.step(state, action)

        log.turns.append(TurnLog(
            turn=turn,
            player=player,
            legal_actions=legal,
            chosen_action=action,
            state_before=state,
            state_after=next_state,
            reward=reward,
            time_ms=elapsed_ms,
        ))
        turn += 1
        state = next_state

        if done:
            if reward > 0:
                log.winner = player
            elif reward < 0:
                log.winner = 1 - player
            else:
                log.winner = -1  # draw
            break

    log.n_turns = turn
    return log


def run_matches(game: BoardGame, agents: list[Agent], n_games: int) -> list[GameLog]:
    return [play_game(game, agents, game_id=i) for i in range(n_games)]
