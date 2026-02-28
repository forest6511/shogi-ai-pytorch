"""Terminal display for どうぶつしょうぎ boards."""

from __future__ import annotations

from shogi_ai.game.animal_shogi.board import Board
from shogi_ai.game.animal_shogi.types import COLS, ROWS, PieceType, Player

# Display characters: uppercase = SENTE, lowercase = GOTE
PIECE_CHARS: dict[PieceType, str] = {
    PieceType.CHICK: "C",
    PieceType.GIRAFFE: "G",
    PieceType.ELEPHANT: "E",
    PieceType.LION: "L",
    PieceType.HEN: "H",
}

PIECE_NAMES_JA: dict[PieceType, str] = {
    PieceType.CHICK: "ひよこ",
    PieceType.GIRAFFE: "きりん",
    PieceType.ELEPHANT: "ぞう",
    PieceType.LION: "ライオン",
    PieceType.HEN: "にわとり",
}


def piece_to_char(piece_type: PieceType, owner: Player) -> str:
    """Convert a piece to its display character."""
    char = PIECE_CHARS[piece_type]
    if owner == Player.GOTE:
        return char.lower()
    return char


def hand_to_str(hand: tuple[PieceType, ...]) -> str:
    """Convert a hand to display string."""
    if not hand:
        return "-"
    return " ".join(PIECE_CHARS[pt] for pt in hand)


def board_to_str(board: Board) -> str:
    """Convert a board to a human-readable string.

    Example output:
        GOTE hand: C
          a b c
        1 g l e
        2 . c .
        3 . C .
        4 E L G
        SENTE hand: -
    """
    lines: list[str] = []

    # Gote's hand
    lines.append(f"GOTE hand: {hand_to_str(board.hands[Player.GOTE.value])}")

    # Column headers
    col_labels = " ".join(chr(ord("a") + c) for c in range(COLS))
    lines.append(f"  {col_labels}")

    # Board rows
    for r in range(ROWS):
        row_chars: list[str] = []
        for c in range(COLS):
            piece = board.piece_at(r, c)
            if piece is None:
                row_chars.append(".")
            else:
                row_chars.append(piece_to_char(piece.piece_type, piece.owner))
        lines.append(f"{r + 1} {' '.join(row_chars)}")

    # Sente's hand
    lines.append(f"SENTE hand: {hand_to_str(board.hands[Player.SENTE.value])}")

    return "\n".join(lines)
