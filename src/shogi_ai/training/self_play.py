"""Self-play data generation for AlphaZero-style training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

from shogi_ai.engine.mcts import MCTS, MCTSConfig
from shogi_ai.game.protocol import GameState
from shogi_ai.model.network import DualHeadNetwork


class TrainingExample(NamedTuple):
    """A single training example from self-play."""

    state_tensor: Tensor   # (in_channels, board_h, board_w)
    policy_target: Tensor  # (action_space_size,)
    value_target: float    # +1 (win) / -1 (loss) / 0 (draw)


@dataclass(frozen=True)
class SelfPlayConfig:
    """Configuration for self-play data generation."""

    num_games: int = 20
    num_simulations: int = 50
    temperature_threshold: int = 10  # After this many moves, use temp=0


def play_game(
    network: DualHeadNetwork,
    state: GameState,
    config: SelfPlayConfig,
) -> list[TrainingExample]:
    """Play one game of self-play and return training examples.

    Temperature schedule:
    - First `temperature_threshold` moves: τ=1.0 (exploratory)
    - After that: τ→0 (deterministic, pick best)
    """
    examples: list[tuple[Tensor, Tensor, int]] = []
    mcts_config = MCTSConfig(num_simulations=config.num_simulations)
    mcts = MCTS(network, mcts_config)

    move_count = 0
    max_moves = 200

    while not state.is_terminal and move_count < max_moves:
        # Temperature schedule
        if move_count < config.temperature_threshold:
            mcts.config = MCTSConfig(
                num_simulations=config.num_simulations,
                temperature=1.0,
            )
        else:
            mcts.config = MCTSConfig(
                num_simulations=config.num_simulations,
                temperature=0.01,  # Near-deterministic
            )

        # MCTS search
        action_probs = mcts.search(state)
        tensor = state.to_tensor_planes()
        policy = torch.tensor(action_probs, dtype=torch.float32)

        # Store (state, policy, current_player) — value assigned after game ends
        examples.append((tensor, policy, state.current_player))

        # Select move
        move = _select_move(action_probs, state.legal_moves())
        state = state.apply_move(move)
        move_count += 1

    # Assign value targets based on game outcome
    winner = state.winner
    result: list[TrainingExample] = []
    for tensor, policy, player in examples:
        if winner is None:
            value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        result.append(TrainingExample(tensor, policy, value))

    return result


def generate_training_data(
    network: DualHeadNetwork,
    initial_state: GameState,
    config: SelfPlayConfig,
) -> list[TrainingExample]:
    """Generate training data from multiple self-play games."""
    all_examples: list[TrainingExample] = []
    for _ in range(config.num_games):
        examples = play_game(network, initial_state, config)
        all_examples.extend(examples)
    return all_examples


def _select_move(action_probs: list[float], legal_moves: list[int]) -> int:
    """Sample a move from the action probability distribution."""
    probs = torch.tensor([action_probs[m] for m in legal_moves])
    # Ensure valid distribution
    total = probs.sum()
    if total <= 0:
        # Fallback to uniform
        idx = torch.randint(len(legal_moves), (1,)).item()
    else:
        probs = probs / total
        idx = torch.multinomial(probs, 1).item()
    return legal_moves[idx]
