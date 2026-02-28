"""GameState implementation for どうぶつしょうぎ."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from shogi_ai.game.animal_shogi.board import Board
from shogi_ai.game.animal_shogi.moves import apply_move as _apply_move
from shogi_ai.game.animal_shogi.moves import legal_moves as _legal_moves
from shogi_ai.game.animal_shogi.types import (
    COLS,
    HAND_PIECE_TYPES,
    ROWS,
    Player,
)


@dataclass(frozen=True)
class AnimalShogiState:
    """Immutable game state for どうぶつしょうぎ.

    Terminal conditions:
    1. Lion capture: a player captures the opponent's lion → that player wins.
    2. Try rule: a player moves their lion to the opponent's back rank
       without being capturable → that player wins.
    3. No legal moves: the current player loses (stalemate = loss).
    """

    board: Board = field(default_factory=Board)
    _current_player: Player = Player.SENTE
    _move_count: int = 0

    @property
    def current_player(self) -> int:
        return self._current_player.value

    @property
    def is_terminal(self) -> bool:
        return self.winner is not None or len(self.legal_moves()) == 0

    @property
    def winner(self) -> int | None:
        # Lion capture: if a player's lion is missing, the opponent wins.
        for player in Player:
            if self.board.find_lion(player) is None:
                return player.opponent.value

        # Try rule: check if the previous player successfully moved their
        # lion to the opponent's back rank.
        prev_player = self._current_player.opponent
        lion_idx = self.board.find_lion(prev_player)
        if lion_idx is not None:
            lion_row = lion_idx // COLS
            target_row = 0 if prev_player == Player.SENTE else ROWS - 1
            if lion_row == target_row:
                # Check if the lion can be captured by the current player
                if not self._can_capture_lion(self._current_player, lion_idx):
                    return prev_player.value

        # No legal moves = current player loses
        if len(self.legal_moves()) == 0:
            return self._current_player.opponent.value

        return None

    def legal_moves(self) -> list[int]:
        return _legal_moves(self.board, self._current_player)

    def apply_move(self, move: int) -> AnimalShogiState:
        new_board = _apply_move(self.board, self._current_player, move)
        return AnimalShogiState(
            board=new_board,
            _current_player=self._current_player.opponent,
            _move_count=self._move_count + 1,
        )

    def to_tensor_planes(self) -> torch.Tensor:
        """Convert to tensor planes for neural network input.

        Planes (14 total):
        - 5 planes for current player's pieces (Chick, Giraffe, Elephant, Lion, Hen)
        - 5 planes for opponent's pieces
        - 3 planes for current player's hand (Chick, Giraffe, Elephant count)
        - 1 plane for turn indicator (all 1s if SENTE, all 0s if GOTE)
        """
        planes = torch.zeros(14, ROWS, COLS)
        cp = self._current_player

        for idx, piece in enumerate(self.board.squares):
            if piece is None:
                continue
            r, c = idx // COLS, idx % COLS
            if piece.owner == cp:
                planes[piece.piece_type.value, r, c] = 1.0
            else:
                planes[5 + piece.piece_type.value, r, c] = 1.0

        # Hand pieces
        for i, pt in enumerate(HAND_PIECE_TYPES):
            count = self.board.hands[cp.value].count(pt)
            if count > 0:
                planes[10 + i, :, :] = float(count)

        # Turn indicator
        if cp == Player.SENTE:
            planes[13, :, :] = 1.0

        return planes

    def _can_capture_lion(self, player: Player, lion_idx: int) -> bool:
        """Check if player can capture the piece at lion_idx."""
        moves = _legal_moves(self.board, player)
        for move in moves:
            if move < 144:  # Board moves only
                to_idx = move % 12
                if to_idx == lion_idx:
                    return True
        return False
