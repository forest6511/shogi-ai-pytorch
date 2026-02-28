"""Board representation for 本将棋 (9x9)."""

from __future__ import annotations

from dataclasses import dataclass, field

from shogi_ai.game.full_shogi.types import (
    COLS,
    NUM_SQUARES,
    ROWS,
    UNPROMOTION_MAP,
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
    """Immutable board state for 9x9 本将棋.

    squares: tuple of 81 elements (row-major), each Piece | None.
    hands: tuple of 2 tuples, each containing piece types in hand.
    """

    squares: tuple[Piece | None, ...] = field(
        default_factory=lambda: Board._initial_squares()
    )
    hands: tuple[tuple[PieceType, ...], tuple[PieceType, ...]] = ((), ())

    @staticmethod
    def _initial_squares() -> tuple[Piece | None, ...]:
        """Return the standard starting position (平手).

        Standard shogi initial arrangement.
        Row 0 = Gote's back rank (top), Row 8 = Sente's back rank (bottom).
        """
        squares: list[Piece | None] = [None] * NUM_SQUARES

        # Gote's pieces (rows 0-2)
        # Row 0: Lance Knight Silver Gold King Gold Silver Knight Lance
        gote_back = [
            PieceType.LANCE, PieceType.KNIGHT, PieceType.SILVER,
            PieceType.GOLD, PieceType.KING, PieceType.GOLD,
            PieceType.SILVER, PieceType.KNIGHT, PieceType.LANCE,
        ]
        for c, pt in enumerate(gote_back):
            squares[0 * COLS + c] = Piece(pt, Player.GOTE)

        # Row 1: _ Rook _ _ _ _ _ Bishop _
        squares[1 * COLS + 1] = Piece(PieceType.ROOK, Player.GOTE)
        squares[1 * COLS + 7] = Piece(PieceType.BISHOP, Player.GOTE)

        # Row 2: Pawns
        for c in range(COLS):
            squares[2 * COLS + c] = Piece(PieceType.PAWN, Player.GOTE)

        # Sente's pieces (rows 6-8, mirrored)
        # Row 6: Pawns
        for c in range(COLS):
            squares[6 * COLS + c] = Piece(PieceType.PAWN, Player.SENTE)

        # Row 7: _ Bishop _ _ _ _ _ Rook _
        squares[7 * COLS + 1] = Piece(PieceType.BISHOP, Player.SENTE)
        squares[7 * COLS + 7] = Piece(PieceType.ROOK, Player.SENTE)

        # Row 8: Lance Knight Silver Gold King Gold Silver Knight Lance
        sente_back = [
            PieceType.LANCE, PieceType.KNIGHT, PieceType.SILVER,
            PieceType.GOLD, PieceType.KING, PieceType.GOLD,
            PieceType.SILVER, PieceType.KNIGHT, PieceType.LANCE,
        ]
        for c, pt in enumerate(sente_back):
            squares[8 * COLS + c] = Piece(pt, Player.SENTE)

        return tuple(squares)

    def piece_at(self, row: int, col: int) -> Piece | None:
        return self.squares[row * COLS + col]

    def set_piece(self, row: int, col: int, piece: Piece | None) -> Board:
        idx = row * COLS + col
        squares = list(self.squares)
        squares[idx] = piece
        return Board(squares=tuple(squares), hands=self.hands)

    def add_to_hand(self, player: Player, piece_type: PieceType) -> Board:
        """Add piece to hand, reverting promoted pieces to base form."""
        hands = list(self.hands)
        hand = list(hands[player.value])
        # Revert promoted pieces
        base_type = UNPROMOTION_MAP.get(piece_type, piece_type)
        hand.append(base_type)
        hand.sort()
        hands[player.value] = tuple(hand)
        return Board(squares=self.squares, hands=(hands[0], hands[1]))

    def remove_from_hand(self, player: Player, piece_type: PieceType) -> Board:
        hands = list(self.hands)
        hand = list(hands[player.value])
        hand.remove(piece_type)
        hands[player.value] = tuple(hand)
        return Board(squares=self.squares, hands=(hands[0], hands[1]))

    def find_king(self, player: Player) -> int | None:
        for idx, piece in enumerate(self.squares):
            if (
                piece is not None
                and piece.piece_type == PieceType.KING
                and piece.owner == player
            ):
                return idx
        return None

    def count_pawns_in_column(self, player: Player, col: int) -> int:
        """Count unpromoted pawns of player in a column (for 二歩 check)."""
        count = 0
        for r in range(ROWS):
            p = self.piece_at(r, col)
            if p is not None and p.owner == player and p.piece_type == PieceType.PAWN:
                count += 1
        return count
