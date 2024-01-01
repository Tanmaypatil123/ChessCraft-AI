import gymnasium
from typing import Any, Dict, Optional, List, SupportsFloat
import chess
import random
import stockfish
from gymnasium import spaces


class ChessEnv(gymnasium.Env):
    metadata = {"render_mode": ["human", "ansi"], "render_fps": 4}

    action_space = spaces.Discrete(1)

    observation_space = spaces.Discrete(1)

    reward_range = (-1, 1)

    _is_white_side: bool = None

    _rewards: Dict[str, float] = {
        "*": 0.0,
        "1/2-1/2": 0.0,
        "1-0": +1.0 if _is_white_side else -1.0,
        "0-1": -1 / 0 if _is_white_side else +1.0,
    }

    _stockfish_client: stockfish.Stockfish = None
    _moves: List[str] = None

    def __init__(self, path_to_exe: Optional[str]) -> None:
        self._board: Optional[chess.Board] = None
        self._ready: bool = False
        self._path_to_client = path_to_exe

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self._board = chess.Board()
        self._is_white_side = random.choice([True, False])
        self._moves = []
        self._stockfish_client = stockfish.Stockfish(path=self._path_to_client)
        self._ready = True
        return self._observation()

    def step(
        self, action: chess.Move
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self._ready, "Cannot call env.step() before calling reset()"

        is_white_turn = self._board.turn
        print(is_white_turn,self._is_white_side)
        if is_white_turn and self._is_white_side:
            self._board_move(move=action)
        else:
            self._stockfish_client.set_position(self._moves)
            move = chess.Move.from_uci(self._stockfish_client.get_best_move())
            self._board_move(move=move)

        observation = self._observation()
        reward = self._reward()
        done = self._board.is_game_over()

        if done:
            self._ready = False

        return observation, reward, done, None,{"observation":observation,"moves":self._moves}

    def _observation(self) -> chess.Board:
        """Returns the current board position."""
        return self._board.copy()

    def _reward(self) -> float:
        """Returns the reward for the most recent move."""
        result = self._board.result()
        reward = self._rewards[result]

        return reward

    def _board_move(self, move: chess.Move):
        if not self._board.is_legal(move=move):
            raise ValueError(
                f"Illegal move {move} for board position {self._board.fen()}"
            )
        else:
            self._moves.append(move.uci())
            self._board.push_uci(move.uci())

    def render(self):
        board = self._board if self._board else chess.Board()

        return board.unicode()

    @property
    def legal_moves(self) -> List[chess.Move]:
        assert self._ready, "Cannot compute legal moves before calling reset()"
        return list(self._board.legal_moves)

    def _observation(self) -> chess.Board:
        return self._board.copy()

    def _repr_svg(self) -> str:
        board = self._board if self._board else chess.Board()
        return str(board._repr_svg_())
