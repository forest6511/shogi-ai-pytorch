"""Legal move generation for どうぶつしょうぎ.

Move encoding (int):
  Board moves:  from_idx * 12 + to_idx  (range 0..143)
  Drop moves:   144 + piece_type * 12 + to_idx  (range 144..179)
    piece_type: 0=Chick, 1=Giraffe, 2=Elephant (only 3 droppable types)

  Total action space: 180
"""

from __future__ import annotations

from shogi_ai.game.animal_shogi.board import Board, Piece
from shogi_ai.game.animal_shogi.types import (
    COLS,
    HAND_PIECE_TYPES,
    PIECE_MOVES,
    ROWS,
    PieceType,
    Player,
)

ACTION_SPACE = 180


def encode_board_move(from_idx: int, to_idx: int) -> int:
    """Encode a board move as an integer."""
    return from_idx * 12 + to_idx


def encode_drop_move(piece_type: PieceType, to_idx: int) -> int:
    """Encode a drop move as an integer."""
    pt_index = HAND_PIECE_TYPES.index(piece_type)
    return 144 + pt_index * 12 + to_idx


def decode_move(move: int) -> dict:
    """Decode a move integer into a descriptive dict."""
    if move < 144:
        from_idx = move // 12
        to_idx = move % 12
        return {
            "type": "board",
            "from": (from_idx // COLS, from_idx % COLS),
            "to": (to_idx // COLS, to_idx % COLS),
        }
    else:
        remainder = move - 144
        pt_index = remainder // 12
        to_idx = remainder % 12
        return {
            "type": "drop",
            "piece_type": HAND_PIECE_TYPES[pt_index],
            "to": (to_idx // COLS, to_idx % COLS),
        }


def legal_moves(board: Board, player: Player) -> list[int]:
    """Generate all legal moves for the given player.

    Includes:
    - Board moves (piece movement + promotion)
    - Drop moves (placing captured pieces)

    Does NOT include moves that leave own lion in check
    (lion capture is allowed as a win condition).
    """
    moves: list[int] = []

    # Board moves
    for idx, piece in enumerate(board.squares):
        if piece is None or piece.owner != player:
            continue
        row, col = idx // COLS, idx % COLS
        deltas = PIECE_MOVES[piece.piece_type]

        for dr, dc in deltas:
            # GOTE moves are mirrored vertically
            if player == Player.GOTE:
                dr = -dr
            nr, nc = row + dr, col + dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            target = board.piece_at(nr, nc)
            if target is not None and target.owner == player:
                continue  # Can't capture own piece

            to_idx = nr * COLS + nc
            move = encode_board_move(idx, to_idx)

            # Check if this move is promotion
            if _should_promote(piece, player, nr):
                # Promotion is mandatory for chick reaching the far rank
                moves.append(move)
            else:
                moves.append(move)

    # Drop moves
    hand = board.hands[player.value]
    unique_in_hand = set(hand)
    for pt in unique_in_hand:
        for idx in range(ROWS * COLS):
            if board.squares[idx] is not None:
                continue  # Square occupied
            moves.append(encode_drop_move(pt, idx))

    return moves


def apply_move(board: Board, player: Player, move: int) -> Board:
    """Apply a move and return the new board state."""
    if move < 144:
        return _apply_board_move(board, player, move)
    else:
        return _apply_drop_move(board, player, move)


def _apply_board_move(board: Board, player: Player, move: int) -> Board:
    """Apply a board move (piece movement)."""
    from_idx = move // 12
    to_idx = move % 12
    from_row, from_col = from_idx // COLS, from_idx % COLS
    to_row = to_idx // COLS

    piece = board.piece_at(from_row, from_col)
    assert piece is not None, f"No piece at ({from_row}, {from_col})"

    # Capture
    target = board.squares[to_idx]
    new_board = board
    if target is not None:
        new_board = new_board.add_to_hand(player, target.piece_type)

    # Promotion check
    new_piece_type = piece.piece_type
    if _should_promote(piece, player, to_row):
        new_piece_type = PieceType.HEN

    # Move piece
    new_board = new_board.set_piece(from_row, from_col, None)
    new_board = new_board.set_piece(
        to_idx // COLS,
        to_idx % COLS,
        Piece(new_piece_type, player),
    )

    return new_board


def _apply_drop_move(board: Board, player: Player, move: int) -> Board:
    """Apply a drop move (placing a captured piece)."""
    remainder = move - 144
    pt_index = remainder // 12
    to_idx = remainder % 12

    piece_type = HAND_PIECE_TYPES[pt_index]
    to_row, to_col = to_idx // COLS, to_idx % COLS

    new_board = board.remove_from_hand(player, piece_type)
    new_board = new_board.set_piece(to_row, to_col, Piece(piece_type, player))

    return new_board


def _should_promote(piece: Piece, player: Player, dest_row: int) -> bool:
    """Check if a piece should promote on reaching dest_row."""
    if piece.piece_type != PieceType.CHICK:
        return False
    if player == Player.SENTE:
        return dest_row == 0  # Sente promotes on row 0 (top)
    else:
        return dest_row == ROWS - 1  # Gote promotes on row 3 (bottom)
