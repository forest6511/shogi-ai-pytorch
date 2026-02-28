"""GameState implementation for 本将棋 (Full Shogi)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from shogi_ai.game.full_shogi.board import Board
from shogi_ai.game.full_shogi.moves import ACTION_SPACE
from shogi_ai.game.full_shogi.moves import apply_move as _apply_move
from shogi_ai.game.full_shogi.moves import legal_moves as _legal_moves
from shogi_ai.game.full_shogi.types import (
    COLS,
    HAND_PIECE_TYPES,
    ROWS,
    Player,
)


@dataclass(frozen=True)
class FullShogiState:
    """Immutable game state for 本将棋 (9x9).

    Terminal conditions:
    1. King capture: should not happen with legal move filtering (king in check).
    2. No legal moves (checkmate or stalemate): current player loses.
    """

    board: Board = field(default_factory=Board)
    _current_player: Player = Player.SENTE
    _move_count: int = 0

    @property
    def action_space_size(self) -> int:
        return ACTION_SPACE

    @property
    def current_player(self) -> int:
        return self._current_player.value

    @property
    def is_terminal(self) -> bool:
        # King missing = terminal (shouldn't happen with proper legal moves)
        for player in Player:
            if self.board.find_king(player) is None:
                return True
        return len(self.legal_moves()) == 0

    @property
    def winner(self) -> int | None:
        # King missing check
        for player in Player:
            if self.board.find_king(player) is None:
                return player.opponent.value

        # No legal moves = current player loses
        if len(self.legal_moves()) == 0:
            return self._current_player.opponent.value

        return None

    def legal_moves(self) -> list[int]:
        return _legal_moves(self.board, self._current_player)

    def apply_move(self, move: int) -> FullShogiState:
        new_board = _apply_move(self.board, self._current_player, move)
        return FullShogiState(
            board=new_board,
            _current_player=self._current_player.opponent,
            _move_count=self._move_count + 1,
        )

    def to_tensor_planes(self) -> torch.Tensor:
        """Convert to tensor planes for neural network input.

        Planes (43 total):
        - 14 planes: current player's pieces (one per PieceType)
        - 14 planes: opponent's pieces
        - 7 planes: current player's hand piece counts
        - 7 planes: opponent's hand piece counts
        - 1 plane: turn indicator
        """
        planes = torch.zeros(43, ROWS, COLS)
        cp = self._current_player

        # Board pieces
        for idx, piece in enumerate(self.board.squares):
            if piece is None:
                continue
            r, c = idx // COLS, idx % COLS
            if piece.owner == cp:
                planes[piece.piece_type.value, r, c] = 1.0
            else:
                planes[14 + piece.piece_type.value, r, c] = 1.0

        # Hand pieces
        for i, pt in enumerate(HAND_PIECE_TYPES):
            cp_count = self.board.hands[cp.value].count(pt)
            opp_count = self.board.hands[cp.opponent.value].count(pt)
            if cp_count > 0:
                planes[28 + i, :, :] = float(cp_count)
            if opp_count > 0:
                planes[35 + i, :, :] = float(opp_count)

        # Turn indicator
        if cp == Player.SENTE:
            planes[42, :, :] = 1.0

        return planes
