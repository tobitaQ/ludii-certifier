"""
Stage 7: Human CLI agent for user study.

Displays the board, prompts the participant for a move, validates input,
and records the wall-clock time spent thinking.

Each game implements `format_action_hint(legal_actions, state) -> str`
and `parse_action(user_input, legal_actions, state) -> int | None`.
Falls back to integer parsing if the game does not define them.
"""
import sys
import time
from base import BoardGame, Agent, State, Action


class HumanCLIAgent(Agent):
    def __init__(self, game: BoardGame, player_label: str = "あなた"):
        self.game = game
        self.player_label = player_label
        self.last_think_ms: float = 0.0

    def select_action(self, state: State, legal_actions: list[Action]) -> Action:
        # Display the board
        print()
        print("─" * 40)
        if hasattr(self.game, "format_state"):
            print(self.game.format_state(state))
        else:
            print(f"state = {state}")
        print("─" * 40)

        # Game-specific hint
        hint = self._format_hint(state, legal_actions)
        print(hint)

        # Prompt with timing
        t0 = time.perf_counter()
        while True:
            raw = input(f"{self.player_label} の手を入力: ").strip()
            action = self._parse(raw, legal_actions, state)
            if action is not None and action in legal_actions:
                break
            print("  ⚠️ 無効な入力です。もう一度お試しください。")

        self.last_think_ms = (time.perf_counter() - t0) * 1000
        return action

    # ---------- game-specific helpers ----------

    def _format_hint(self, state: State, legal_actions: list[Action]) -> str:
        name = (self.game.game_name() if hasattr(self.game, "game_name") else "")

        if name == "Othello":
            cells = [f"{'abcdefgh'[a % 8]}{a // 8 + 1}" for a in legal_actions]
            return f"合法手 ({len(cells)} 個): " + " ".join(cells) + "\n  入力例: e4"

        if name == "Kalah":
            return f"合法手: 穴 {legal_actions} のいずれか\n  入力例: 0（左端の穴）〜 5（右端の穴）"

        return f"合法手: {legal_actions}"

    def _parse(self, raw: str, legal_actions: list[Action], state: State) -> int | None:
        name = (self.game.game_name() if hasattr(self.game, "game_name") else "")

        if name == "Othello":
            raw = raw.lower().replace(" ", "")
            if len(raw) != 2 or raw[0] not in "abcdefgh" or not raw[1].isdigit():
                return None
            c = "abcdefgh".index(raw[0])
            r = int(raw[1]) - 1
            if not (0 <= r < 8):
                return None
            return r * 8 + c

        # Default: integer input
        try:
            return int(raw)
        except ValueError:
            return None
