"""Legal move generation for 本将棋.

Move encoding (from×to approach):
  Board moves (no promotion):   from_idx * 81 + to_idx          (range 0..6560)
  Board moves (with promotion): 6561 + from_idx * 81 + to_idx   (range 6561..13121)
  Drop moves:                   13122 + piece_type_idx * 81 + to_idx (range 13122..13688)
    piece_type_idx: 0=Pawn..6=Rook (7 droppable types)

  Total action space: 13122 + 567 = 13689
"""

from __future__ import annotations

from shogi_ai.game.full_shogi.board import Board, Piece
from shogi_ai.game.full_shogi.types import (
    COLS,
    DRAGON_EXTRA_STEPS,
    HAND_PIECE_TYPES,
    HORSE_EXTRA_STEPS,
    KNIGHT_MOVES,
    NUM_SQUARES,
    PROMOTION_MAP,
    ROWS,
    SLIDE_MOVES,
    STEP_MOVES,
    PieceType,
    Player,
)

ACTION_SPACE = 13689

_BOARD_MOVE_BASE = 0
_PROMO_MOVE_BASE = NUM_SQUARES * NUM_SQUARES  # 6561
_DROP_MOVE_BASE = 2 * NUM_SQUARES * NUM_SQUARES  # 13122


def encode_board_move(from_idx: int, to_idx: int, promote: bool = False) -> int:
    if promote:
        return _PROMO_MOVE_BASE + from_idx * NUM_SQUARES + to_idx
    return from_idx * NUM_SQUARES + to_idx


def encode_drop_move(piece_type: PieceType, to_idx: int) -> int:
    pt_index = HAND_PIECE_TYPES.index(piece_type)
    return _DROP_MOVE_BASE + pt_index * NUM_SQUARES + to_idx


def decode_move(move: int) -> dict:
    if move < _PROMO_MOVE_BASE:
        from_idx = move // NUM_SQUARES
        to_idx = move % NUM_SQUARES
        return {
            "type": "board",
            "from": (from_idx // COLS, from_idx % COLS),
            "to": (to_idx // COLS, to_idx % COLS),
            "promote": False,
        }
    elif move < _DROP_MOVE_BASE:
        adjusted = move - _PROMO_MOVE_BASE
        from_idx = adjusted // NUM_SQUARES
        to_idx = adjusted % NUM_SQUARES
        return {
            "type": "board",
            "from": (from_idx // COLS, from_idx % COLS),
            "to": (to_idx // COLS, to_idx % COLS),
            "promote": True,
        }
    else:
        adjusted = move - _DROP_MOVE_BASE
        pt_index = adjusted // NUM_SQUARES
        to_idx = adjusted % NUM_SQUARES
        return {
            "type": "drop",
            "piece_type": HAND_PIECE_TYPES[pt_index],
            "to": (to_idx // COLS, to_idx % COLS),
        }


def legal_moves(board: Board, player: Player) -> list[int]:
    """Generate all legal moves (excluding moves that leave king in check)."""
    pseudo = _pseudo_legal_moves(board, player)
    legal: list[int] = []
    for move in pseudo:
        new_board = apply_move(board, player, move)
        if not _is_in_check(new_board, player):
            legal.append(move)
    return legal


def apply_move(board: Board, player: Player, move: int) -> Board:
    if move >= _DROP_MOVE_BASE:
        return _apply_drop(board, player, move)
    return _apply_board_move(board, player, move)


def _pseudo_legal_moves(board: Board, player: Player) -> list[int]:
    """Generate pseudo-legal moves (may leave own king in check)."""
    moves: list[int] = []
    _generate_board_moves(board, player, moves)
    _generate_drop_moves(board, player, moves)
    return moves


def _generate_board_moves(
    board: Board,
    player: Player,
    moves: list[int],
) -> None:
    """Generate board moves (step, slide, knight)."""
    for idx in range(NUM_SQUARES):
        piece = board.squares[idx]
        if piece is None or piece.owner != player:
            continue

        row, col = idx // COLS, idx % COLS
        pt = piece.piece_type

        # Step moves
        if pt in STEP_MOVES:
            for dr, dc in STEP_MOVES[pt]:
                if player == Player.GOTE:
                    dr = -dr
                nr, nc = row + dr, col + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    target = board.piece_at(nr, nc)
                    if target is None or target.owner != player:
                        to_idx = nr * COLS + nc
                        _add_move_with_promotion(
                            moves,
                            idx,
                            to_idx,
                            pt,
                            player,
                            row,
                            nr,
                        )

        # Knight moves
        if pt == PieceType.KNIGHT:
            for dr, dc in KNIGHT_MOVES:
                if player == Player.GOTE:
                    dr = -dr
                nr, nc = row + dr, col + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    target = board.piece_at(nr, nc)
                    if target is None or target.owner != player:
                        to_idx = nr * COLS + nc
                        _add_move_with_promotion(
                            moves,
                            idx,
                            to_idx,
                            pt,
                            player,
                            row,
                            nr,
                        )

        # Slide moves
        if pt in SLIDE_MOVES:
            for dr, dc in SLIDE_MOVES[pt]:
                if player == Player.GOTE:
                    dr, dc = -dr, -dc
                nr, nc = row + dr, col + dc
                while 0 <= nr < ROWS and 0 <= nc < COLS:
                    target = board.piece_at(nr, nc)
                    if target is not None and target.owner == player:
                        break
                    to_idx = nr * COLS + nc
                    _add_move_with_promotion(
                        moves,
                        idx,
                        to_idx,
                        pt,
                        player,
                        row,
                        nr,
                    )
                    if target is not None:
                        break  # Captured, stop sliding
                    nr, nc = nr + dr, nc + dc

        # Horse extra step moves (orthogonal 1-step)
        if pt == PieceType.HORSE:
            for dr, dc in HORSE_EXTRA_STEPS:
                if player == Player.GOTE:
                    dr = -dr
                nr, nc = row + dr, col + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    target = board.piece_at(nr, nc)
                    if target is None or target.owner != player:
                        to_idx = nr * COLS + nc
                        # Horse cannot promote further
                        moves.append(encode_board_move(idx, to_idx))

        # Dragon extra step moves (diagonal 1-step)
        if pt == PieceType.DRAGON:
            for dr, dc in DRAGON_EXTRA_STEPS:
                if player == Player.GOTE:
                    dr = -dr
                nr, nc = row + dr, col + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    target = board.piece_at(nr, nc)
                    if target is None or target.owner != player:
                        to_idx = nr * COLS + nc
                        moves.append(encode_board_move(idx, to_idx))


def _add_move_with_promotion(
    moves: list[int],
    from_idx: int,
    to_idx: int,
    piece_type: PieceType,
    player: Player,
    from_row: int,
    to_row: int,
) -> None:
    """Add a move, possibly with promotion variants."""
    can_promote = piece_type in PROMOTION_MAP
    in_promotion_zone = _in_promotion_zone(player, from_row) or _in_promotion_zone(player, to_row)
    must_promote = _must_promote(piece_type, player, to_row)

    if can_promote and in_promotion_zone:
        moves.append(encode_board_move(from_idx, to_idx, promote=True))
        if not must_promote:
            moves.append(encode_board_move(from_idx, to_idx, promote=False))
    elif not must_promote:
        moves.append(encode_board_move(from_idx, to_idx, promote=False))


def _in_promotion_zone(player: Player, row: int) -> bool:
    """Check if a row is in the promotion zone (enemy's 3 ranks)."""
    if player == Player.SENTE:
        return row <= 2
    return row >= 6


def _must_promote(piece_type: PieceType, player: Player, dest_row: int) -> bool:
    """Check if promotion is mandatory (piece has no further moves)."""
    if piece_type == PieceType.PAWN or piece_type == PieceType.LANCE:
        if player == Player.SENTE:
            return dest_row == 0
        return dest_row == 8
    if piece_type == PieceType.KNIGHT:
        if player == Player.SENTE:
            return dest_row <= 1
        return dest_row >= 7
    return False


def _generate_drop_moves(
    board: Board,
    player: Player,
    moves: list[int],
) -> None:
    """Generate drop moves with nifu (二歩) and dead-piece restrictions."""
    hand = board.hands[player.value]
    unique_in_hand = set(hand)

    for pt in unique_in_hand:
        for idx in range(NUM_SQUARES):
            if board.squares[idx] is not None:
                continue

            row = idx // ROWS
            col = idx % COLS

            # 二歩: Cannot drop pawn in column that already has own pawn
            if pt == PieceType.PAWN:
                if board.count_pawns_in_column(player, col) > 0:
                    continue

            # 行き所のない駒: Cannot drop where piece has no future moves
            if pt == PieceType.PAWN or pt == PieceType.LANCE:
                if player == Player.SENTE and row == 0:
                    continue
                if player == Player.GOTE and row == 8:
                    continue

            if pt == PieceType.KNIGHT:
                if player == Player.SENTE and row <= 1:
                    continue
                if player == Player.GOTE and row >= 7:
                    continue

            moves.append(encode_drop_move(pt, idx))


def _apply_board_move(board: Board, player: Player, move: int) -> Board:
    """Apply a board move (with or without promotion)."""
    promote = move >= _PROMO_MOVE_BASE
    if promote:
        adjusted = move - _PROMO_MOVE_BASE
    else:
        adjusted = move
    from_idx = adjusted // NUM_SQUARES
    to_idx = adjusted % NUM_SQUARES

    piece = board.squares[from_idx]
    assert piece is not None

    # Capture
    target = board.squares[to_idx]
    new_board = board
    if target is not None:
        new_board = new_board.add_to_hand(player, target.piece_type)

    # Determine piece type after move
    new_type = piece.piece_type
    if promote and piece.piece_type in PROMOTION_MAP:
        new_type = PROMOTION_MAP[piece.piece_type]

    # Move piece
    from_row, from_col = from_idx // COLS, from_idx % COLS
    to_row, to_col = to_idx // COLS, to_idx % COLS
    new_board = new_board.set_piece(from_row, from_col, None)
    new_board = new_board.set_piece(to_row, to_col, Piece(new_type, player))

    return new_board


def _apply_drop(board: Board, player: Player, move: int) -> Board:
    """Apply a drop move."""
    adjusted = move - _DROP_MOVE_BASE
    pt_index = adjusted // NUM_SQUARES
    to_idx = adjusted % NUM_SQUARES
    pt = HAND_PIECE_TYPES[pt_index]
    to_row, to_col = to_idx // COLS, to_idx % COLS

    new_board = board.remove_from_hand(player, pt)
    new_board = new_board.set_piece(to_row, to_col, Piece(pt, player))
    return new_board


def _is_in_check(board: Board, player: Player) -> bool:
    """Check if player's king is under attack."""
    king_idx = board.find_king(player)
    if king_idx is None:
        return True  # King captured = in check

    opponent = player.opponent
    king_row, king_col = king_idx // COLS, king_idx % COLS

    # Check all opponent pieces for attacks on king
    for idx in range(NUM_SQUARES):
        piece = board.squares[idx]
        if piece is None or piece.owner != opponent:
            continue
        if _attacks_square(board, piece, idx, king_row, king_col, opponent):
            return True

    return False


def _attacks_square(
    board: Board,
    piece: Piece,
    piece_idx: int,
    target_row: int,
    target_col: int,
    attacker: Player,
) -> bool:
    """Check if a piece at piece_idx attacks (target_row, target_col)."""
    row, col = piece_idx // COLS, piece_idx % COLS
    pt = piece.piece_type

    # Step attacks
    if pt in STEP_MOVES:
        for dr, dc in STEP_MOVES[pt]:
            if attacker == Player.GOTE:
                dr = -dr
            if row + dr == target_row and col + dc == target_col:
                return True

    # Knight attacks
    if pt == PieceType.KNIGHT:
        for dr, dc in KNIGHT_MOVES:
            if attacker == Player.GOTE:
                dr = -dr
            if row + dr == target_row and col + dc == target_col:
                return True

    # Slide attacks
    if pt in SLIDE_MOVES:
        for dr, dc in SLIDE_MOVES[pt]:
            if attacker == Player.GOTE:
                dr, dc = -dr, -dc
            nr, nc = row + dr, col + dc
            while 0 <= nr < ROWS and 0 <= nc < COLS:
                if nr == target_row and nc == target_col:
                    return True
                if board.piece_at(nr, nc) is not None:
                    break
                nr, nc = nr + dr, nc + dc

    # Horse extra steps
    if pt == PieceType.HORSE:
        for dr, dc in HORSE_EXTRA_STEPS:
            if attacker == Player.GOTE:
                dr = -dr
            if row + dr == target_row and col + dc == target_col:
                return True

    # Dragon extra steps
    if pt == PieceType.DRAGON:
        for dr, dc in DRAGON_EXTRA_STEPS:
            if attacker == Player.GOTE:
                dr = -dr
            if row + dr == target_row and col + dc == target_col:
                return True

    return False
