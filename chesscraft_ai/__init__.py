import gymnasium
from typing import Any, Dict ,Optional,List, SupportsFloat
import chess


class ChessEnv(gymnasium.Env):
    action_space = None

    observation_space = None

    reward_range = (-1,1)

    _is_white : bool = None

    _rewards = Dict[str,float] = {
        "*" : 0.0,
        "1/2-1/2" : 0.0,
        "1-0" : +1.0 if _is_white else -1.0,
        "0-1" : -1/0 if _is_white else +1.0
    }

    def __init__(self) -> None:
        self._board : Optional[chess.Board] = None
        self._ready : bool = False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self._board = chess.Board()
        ## TODO : add one opponent as stockfish between black or white.

        self._ready = True
        return self._observation()
    
    def step(self, action: chess.Move) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        
        ## TODO : add one opponent as stockfish and compute.
        return super().step(action)
    
    def render(self):
        board = self._board if self._board else chess.Board()

        ## TODO : render board with Renderframes

    @property
    def legal_moves(self) -> List[chess.Move]:
        assert self._ready,"Cannot compute legal moves before calling reset()"
        return list(self._board.legal_moves)
    
    def _observation(self) -> chess.Board :
        return self._board.copy()
    
    def _repr_svg(self) -> str:
        board = self._board if self._board else chess.Board()
        return str(board._repr_svg_())
        