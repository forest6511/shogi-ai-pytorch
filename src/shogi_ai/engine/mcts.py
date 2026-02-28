"""Monte Carlo Tree Search (MCTS) with neural network guidance."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from shogi_ai.game.protocol import GameState
from shogi_ai.model.network import DualHeadNetwork


@dataclass
class MCTSNode:
    """A node in the MCTS tree."""

    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 0.0
    children: dict[int, MCTSNode] = field(default_factory=dict)

    @property
    def q_value(self) -> float:
        """Average value (W/N)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


@dataclass(frozen=True)
class MCTSConfig:
    """Configuration for MCTS search."""

    num_simulations: int = 50
    c_puct: float = 1.4
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25


class MCTS:
    """Monte Carlo Tree Search guided by a neural network."""

    def __init__(self, network: DualHeadNetwork, config: MCTSConfig) -> None:
        self.network = network
        self.config = config
        self.device = next(network.parameters()).device

    def search(self, state: GameState) -> list[float]:
        """Run MCTS and return action probabilities.

        Returns a list of length action_space_size with visit-count-based
        probabilities for each action.
        """
        root = MCTSNode()
        legal = state.legal_moves()

        if not legal:
            return [0.0] * state.action_space_size

        # Expand root with NN evaluation
        policy, _ = self._evaluate(state)
        for move in legal:
            root.children[move] = MCTSNode(prior=policy[move])

        # Add Dirichlet noise to root for exploration
        self._add_dirichlet_noise(root, legal)

        # Run simulations
        for _ in range(self.config.num_simulations):
            self._simulate(root, state)

        # Build action probabilities from visit counts
        action_probs = [0.0] * state.action_space_size
        if self.config.temperature == 0:
            # Deterministic: pick the most visited
            best_move = max(root.children, key=lambda m: root.children[m].visit_count)
            action_probs[best_move] = 1.0
        else:
            total = sum(
                child.visit_count ** (1.0 / self.config.temperature)
                for child in root.children.values()
            )
            if total > 0:
                for move, child in root.children.items():
                    action_probs[move] = (
                        child.visit_count ** (1.0 / self.config.temperature) / total
                    )

        return action_probs

    def _simulate(self, node: MCTSNode, state: GameState) -> float:
        """Run one simulation from the given node/state. Returns value."""
        if state.is_terminal:
            if state.winner is None:
                return 0.0
            if state.winner == state.current_player:
                return 1.0
            return -1.0

        # Leaf node: expand and return NN value
        if not node.children:
            policy, value = self._evaluate(state)
            legal = state.legal_moves()
            for move in legal:
                node.children[move] = MCTSNode(prior=policy[move])
            return value

        # Select child with highest PUCT score
        move = self._select_child(node)
        child = node.children[move]
        next_state = state.apply_move(move)

        # Recurse and negate value (opponent's perspective)
        value = -self._simulate(child, next_state)

        # Backup
        child.visit_count += 1
        child.total_value += value

        return value

    def _select_child(self, node: MCTSNode) -> int:
        """Select child with highest PUCT score."""
        parent_n = sum(c.visit_count for c in node.children.values())
        sqrt_parent = math.sqrt(parent_n + 1)

        best_move = -1
        best_score = float("-inf")

        for move, child in node.children.items():
            puct = (
                child.q_value
                + self.config.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            )
            if puct > best_score:
                best_score = puct
                best_move = move

        return best_move

    def _evaluate(self, state: GameState) -> tuple[list[float], float]:
        """Evaluate a state with the neural network.

        Returns (policy_probs, value) where policy_probs is masked to legal moves.
        """
        tensor = state.to_tensor_planes().unsqueeze(0).to(self.device)

        self.network.eval()
        with torch.no_grad():
            policy_logits, value_tensor = self.network(tensor)

        # Mask illegal moves and apply softmax
        legal = state.legal_moves()
        policy = policy_logits[0].cpu()

        # Set illegal moves to very negative
        mask = torch.full_like(policy, float("-inf"))
        for m in legal:
            mask[m] = policy[m]

        probs = torch.softmax(mask, dim=0).tolist()
        value = value_tensor.item()

        return probs, value

    def _add_dirichlet_noise(
        self, root: MCTSNode, legal_moves: list[int],
    ) -> None:
        """Add Dirichlet noise to root priors for exploration."""
        noise = torch.distributions.Dirichlet(
            torch.full((len(legal_moves),), self.config.dirichlet_alpha)
        ).sample().tolist()

        eps = self.config.dirichlet_epsilon
        for i, move in enumerate(legal_moves):
            child = root.children[move]
            child.prior = (1 - eps) * child.prior + eps * noise[i]
