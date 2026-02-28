"""Types and constants for 本将棋 (Full Shogi, 9x9)."""

from __future__ import annotations

from enum import IntEnum, unique

ROWS = 9
COLS = 9
NUM_SQUARES = ROWS * COLS  # 81


@unique
class Player(IntEnum):
    """Player identifiers."""

    SENTE = 0  # 先手 (first player, moves upward)
    GOTE = 1   # 後手 (second player, moves downward)

    @property
    def opponent(self) -> Player:
        return Player(1 - self.value)


@unique
class PieceType(IntEnum):
    """Piece types in 本将棋 (14 types)."""

    PAWN = 0       # 歩
    LANCE = 1      # 香
    KNIGHT = 2     # 桂
    SILVER = 3     # 銀
    GOLD = 4       # 金
    BISHOP = 5     # 角
    ROOK = 6       # 飛
    KING = 7       # 玉/王
    PRO_PAWN = 8   # と (promoted pawn)
    PRO_LANCE = 9  # 成香
    PRO_KNIGHT = 10  # 成桂
    PRO_SILVER = 11  # 成銀
    HORSE = 12     # 馬 (promoted bishop)
    DRAGON = 13    # 龍 (promoted rook)


# Which piece types can be promoted and what they promote to
PROMOTION_MAP: dict[PieceType, PieceType] = {
    PieceType.PAWN: PieceType.PRO_PAWN,
    PieceType.LANCE: PieceType.PRO_LANCE,
    PieceType.KNIGHT: PieceType.PRO_KNIGHT,
    PieceType.SILVER: PieceType.PRO_SILVER,
    PieceType.BISHOP: PieceType.HORSE,
    PieceType.ROOK: PieceType.DRAGON,
}

# Reverse: promoted → base type (for captured piece reversion)
UNPROMOTION_MAP: dict[PieceType, PieceType] = {v: k for k, v in PROMOTION_MAP.items()}

# Pieces that can be dropped from hand (unpromoted, non-king)
HAND_PIECE_TYPES = [
    PieceType.PAWN, PieceType.LANCE, PieceType.KNIGHT,
    PieceType.SILVER, PieceType.GOLD, PieceType.BISHOP, PieceType.ROOK,
]

# Step moves: (dr, dc) for SENTE perspective (negative row = forward)
# Sliding pieces use the same deltas but repeat until blocked.
STEP_MOVES: dict[PieceType, list[tuple[int, int]]] = {
    PieceType.PAWN: [(-1, 0)],
    PieceType.SILVER: [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 1)],
    PieceType.GOLD: [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)],
    PieceType.KING: [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ],
    # Promoted pieces move like Gold
    PieceType.PRO_PAWN: [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)],
    PieceType.PRO_LANCE: [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)],
    PieceType.PRO_KNIGHT: [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)],
    PieceType.PRO_SILVER: [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)],
}

# Knight jumps (not step moves — these are special)
KNIGHT_MOVES: list[tuple[int, int]] = [(-2, -1), (-2, 1)]

# Sliding directions for lance, bishop, rook, horse, dragon
SLIDE_MOVES: dict[PieceType, list[tuple[int, int]]] = {
    PieceType.LANCE: [(-1, 0)],  # Forward only
    PieceType.BISHOP: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    PieceType.ROOK: [(-1, 0), (1, 0), (0, -1), (0, 1)],
    PieceType.HORSE: [(-1, -1), (-1, 1), (1, -1), (1, 1)],   # Slide diagonal
    PieceType.DRAGON: [(-1, 0), (1, 0), (0, -1), (0, 1)],    # Slide orthogonal
}

# Extra step moves for promoted sliding pieces
HORSE_EXTRA_STEPS: list[tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DRAGON_EXTRA_STEPS: list[tuple[int, int]] = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
