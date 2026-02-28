"""GameState protocol — all games implement this interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class GameState(Protocol):
    """Common interface for all board game states.

    Immutable: apply_move() returns a new state.
    """

    @property
    def current_player(self) -> int:
        """Return the current player (0 or 1)."""
        ...

    @property
    def is_terminal(self) -> bool:
        """Return True if the game is over."""
        ...

    @property
    def winner(self) -> int | None:
        """Return the winner (0 or 1), or None for draw/ongoing."""
        ...

    def legal_moves(self) -> list[int]:
        """Return list of legal move indices."""
        ...

    def apply_move(self, move: int) -> GameState:
        """Return a new state after applying the move."""
        ...

    @property
    def action_space_size(self) -> int:
        """Return the total number of possible actions."""
        ...

    def to_tensor_planes(self) -> torch.Tensor:
        """Convert state to tensor planes for neural network input."""
        ...
