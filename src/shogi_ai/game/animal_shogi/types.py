"""Types and constants for どうぶつしょうぎ (Animal Shogi)."""

from enum import IntEnum, unique

# Board dimensions: 3 columns x 4 rows
COLS = 3
ROWS = 4


@unique
class Player(IntEnum):
    """Player identifiers."""

    SENTE = 0  # 先手 (first player, moves upward)
    GOTE = 1  # 後手 (second player, moves downward)

    @property
    def opponent(self) -> Player:
        return Player(1 - self.value)


@unique
class PieceType(IntEnum):
    """Piece types in どうぶつしょうぎ."""

    CHICK = 0  # ひよこ — moves 1 forward
    GIRAFFE = 1  # きりん — moves in + directions (rook-like, 1 step)
    ELEPHANT = 2  # ぞう — moves in x directions (bishop-like, 1 step)
    LION = 3  # ライオン — moves 1 in any direction (king)
    HEN = 4  # にわとり — promoted chick (gold-general movement)


# Movement deltas: (row_delta, col_delta)
# For SENTE (moving upward = negative row), GOTE mirrors vertically.
PIECE_MOVES: dict[PieceType, list[tuple[int, int]]] = {
    PieceType.CHICK: [(-1, 0)],
    PieceType.GIRAFFE: [(-1, 0), (1, 0), (0, -1), (0, 1)],
    PieceType.ELEPHANT: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    PieceType.LION: [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ],
    PieceType.HEN: [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, 0),
    ],
}

# Pieces that can be in hand (captured pieces revert to unpromoted form)
HAND_PIECE_TYPES = [PieceType.CHICK, PieceType.GIRAFFE, PieceType.ELEPHANT]
