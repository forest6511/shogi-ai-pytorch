"""Minimax search with alpha-beta pruning for どうぶつしょうぎ."""

from __future__ import annotations

from shogi_ai.game.protocol import GameState

# Material values for evaluation
_PIECE_VALUES = {
    0: 1.0,   # CHICK
    1: 3.0,   # GIRAFFE
    2: 3.0,   # ELEPHANT
    3: 100.0, # LION
    4: 5.0,   # HEN (promoted chick)
}


def evaluate(state: GameState) -> float:
    """Evaluate a position from the current player's perspective.

    Scoring:
    - Material advantage (piece values)
    - Terminal bonus/penalty (±1000)

    Returns positive if current player is better off.
    """
    if state.is_terminal:
        if state.winner is None:
            return 0.0  # Draw
        if state.winner == state.current_player:
            return 1000.0
        return -1000.0

    # Material evaluation from current player's perspective
    score = 0.0
    board = state.board  # type: ignore[attr-defined]

    # Count material on board
    for piece in board.squares:
        if piece is None:
            continue
        value = _PIECE_VALUES.get(piece.piece_type.value, 0.0)
        if piece.owner.value == state.current_player:
            score += value
        else:
            score -= value

    # Count material in hand
    for i, hand in enumerate(board.hands):
        for pt in hand:
            value = _PIECE_VALUES.get(pt.value, 0.0)
            if i == state.current_player:
                score += value
            else:
                score -= value

    return score


def negamax(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
) -> tuple[int, float]:
    """Negamax search with alpha-beta pruning.

    Returns (best_move, score) from the current player's perspective.
    best_move is -1 when depth=0 or at terminal states.
    """
    if state.is_terminal:
        if state.winner is None:
            return -1, 0.0
        if state.winner == state.current_player:
            return -1, 1000.0 + depth  # Prefer faster wins
        return -1, -(1000.0 + depth)

    if depth == 0:
        return -1, evaluate(state)

    moves = state.legal_moves()
    best_move = moves[0]
    best_score = float("-inf")

    for move in moves:
        next_state = state.apply_move(move)
        # Negamax: negate the opponent's score
        _, score = negamax(next_state, depth - 1, -beta, -alpha)
        score = -score

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)
        if alpha >= beta:
            break  # Beta cutoff

    return best_move, best_score


def minimax_move(state: GameState, depth: int = 4) -> int:
    """Return the best move for the current player using minimax search."""
    move, _ = negamax(state, depth, float("-inf"), float("inf"))
    return move
