"""Random player — selects a legal move uniformly at random."""

from __future__ import annotations

import random

from shogi_ai.game.protocol import GameState


def random_move(state: GameState) -> int:
    """Return a random legal move."""
    moves = state.legal_moves()
    if not moves:
        raise ValueError("No legal moves available")
    return random.choice(moves)
