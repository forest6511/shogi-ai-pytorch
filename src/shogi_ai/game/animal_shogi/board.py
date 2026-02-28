"""Board representation for どうぶつしょうぎ."""

from __future__ import annotations

from dataclasses import dataclass, field

from shogi_ai.game.animal_shogi.types import (
    COLS,
    ROWS,
    PieceType,
    Player,
)


@dataclass(frozen=True)
class Piece:
    """A piece on the board."""

    piece_type: PieceType
    owner: Player


@dataclass(frozen=True)
class Board:
    """Immutable board state for 3x4 どうぶつしょうぎ.

    squares: tuple of 12 elements (row-major), each Piece | None.
    hands: tuple of 2 tuples, each containing pieces in hand.
    """

    squares: tuple[Piece | None, ...] = field(
        default_factory=lambda: Board._initial_squares()
    )
    hands: tuple[tuple[PieceType, ...], tuple[PieceType, ...]] = ((), ())

    @staticmethod
    def _initial_squares() -> tuple[Piece | None, ...]:
        """Return the standard starting position.

        Row 0 (top):    GOTE side — Elephant Lion Giraffe
        Row 1:          _ Chick(GOTE) _
        Row 2:          _ Chick(SENTE) _
        Row 3 (bottom): SENTE side — Giraffe Lion Elephant
        """
        squares: list[Piece | None] = [None] * (ROWS * COLS)

        # Row 0: Gote's back rank (from GOTE's perspective, this is their home)
        squares[0 * COLS + 0] = Piece(PieceType.GIRAFFE, Player.GOTE)
        squares[0 * COLS + 1] = Piece(PieceType.LION, Player.GOTE)
        squares[0 * COLS + 2] = Piece(PieceType.ELEPHANT, Player.GOTE)

        # Row 1: Gote's chick
        squares[1 * COLS + 1] = Piece(PieceType.CHICK, Player.GOTE)

        # Row 2: Sente's chick
        squares[2 * COLS + 1] = Piece(PieceType.CHICK, Player.SENTE)

        # Row 3: Sente's back rank
        squares[3 * COLS + 0] = Piece(PieceType.ELEPHANT, Player.SENTE)
        squares[3 * COLS + 1] = Piece(PieceType.LION, Player.SENTE)
        squares[3 * COLS + 2] = Piece(PieceType.GIRAFFE, Player.SENTE)

        return tuple(squares)

    def piece_at(self, row: int, col: int) -> Piece | None:
        """Return the piece at (row, col), or None."""
        return self.squares[row * COLS + col]

    def set_piece(self, row: int, col: int, piece: Piece | None) -> Board:
        """Return a new Board with the piece at (row, col) changed."""
        idx = row * COLS + col
        squares = list(self.squares)
        squares[idx] = piece
        return Board(squares=tuple(squares), hands=self.hands)

    def add_to_hand(self, player: Player, piece_type: PieceType) -> Board:
        """Return a new Board with piece_type added to player's hand."""
        hands = list(self.hands)
        hand = list(hands[player.value])
        # Captured pieces revert to unpromoted form
        if piece_type == PieceType.HEN:
            piece_type = PieceType.CHICK
        hand.append(piece_type)
        hand.sort()  # Keep hand sorted for consistency
        hands[player.value] = tuple(hand)
        return Board(squares=self.squares, hands=(hands[0], hands[1]))

    def remove_from_hand(self, player: Player, piece_type: PieceType) -> Board:
        """Return a new Board with one piece_type removed from player's hand."""
        hands = list(self.hands)
        hand = list(hands[player.value])
        hand.remove(piece_type)
        hands[player.value] = tuple(hand)
        return Board(squares=self.squares, hands=(hands[0], hands[1]))

    def find_lion(self, player: Player) -> int | None:
        """Return the index of player's lion, or None if captured."""
        for idx, piece in enumerate(self.squares):
            if piece is not None and piece.piece_type == PieceType.LION and piece.owner == player:
                return idx
        return None
