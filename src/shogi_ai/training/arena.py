"""Arena for evaluating player strength through head-to-head matches."""

from __future__ import annotations

from collections.abc import Callable

from shogi_ai.game.protocol import GameState


def pit(
    player1_fn: Callable[[GameState], int],
    player2_fn: Callable[[GameState], int],
    initial_state: GameState,
    num_games: int = 50,
    max_moves: int = 200,
) -> tuple[int, int, int]:
    """Play num_games between two players, alternating who goes first.

    Args:
        player1_fn: Function that takes a GameState and returns a move.
        player2_fn: Function that takes a GameState and returns a move.
        initial_state: Starting state for each game.
        num_games: Number of games to play.
        max_moves: Maximum moves per game before declaring draw.

    Returns:
        (player1_wins, player2_wins, draws)
    """
    p1_wins = 0
    p2_wins = 0
    draws = 0

    for game_idx in range(num_games):
        # Alternate who plays first
        if game_idx % 2 == 0:
            sente_fn, gote_fn = player1_fn, player2_fn
            p1_is_sente = True
        else:
            sente_fn, gote_fn = player2_fn, player1_fn
            p1_is_sente = False

        state = initial_state
        move_count = 0

        while not state.is_terminal and move_count < max_moves:
            if state.current_player == 0:
                move = sente_fn(state)
            else:
                move = gote_fn(state)
            state = state.apply_move(move)
            move_count += 1

        winner = state.winner
        if winner is None or move_count >= max_moves:
            draws += 1
        elif (winner == 0 and p1_is_sente) or (winner == 1 and not p1_is_sente):
            p1_wins += 1
        else:
            p2_wins += 1

    return p1_wins, p2_wins, draws
