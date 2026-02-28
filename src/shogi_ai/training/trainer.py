"""Training loop for AlphaZero-style learning."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn

from shogi_ai.model.network import DualHeadNetwork
from shogi_ai.training.self_play import TrainingExample


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for the training loop."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs_per_generation: int = 10
    buffer_size: int = 10000


class Trainer:
    """Train the dual-head network from self-play data."""

    def __init__(
        self,
        network: DualHeadNetwork,
        config: TrainerConfig,
        device: torch.device,
    ) -> None:
        self.network = network
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    def train(self, examples: list[TrainingExample]) -> dict[str, float]:
        """Train for one generation. Returns average losses."""
        if not examples:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        self.network.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_batches = 0

        for _ in range(self.config.epochs_per_generation):
            random.shuffle(examples)
            for i in range(0, len(examples), self.config.batch_size):
                batch = examples[i : i + self.config.batch_size]
                if not batch:
                    continue

                states = torch.stack([ex.state_tensor for ex in batch]).to(
                    self.device
                )
                target_policies = torch.stack(
                    [ex.policy_target for ex in batch]
                ).to(self.device)
                target_values = torch.tensor(
                    [ex.value_target for ex in batch],
                    dtype=torch.float32,
                ).unsqueeze(1).to(self.device)

                # Forward
                policy_logits, values = self.network(states)

                # Policy loss: cross-entropy with soft targets
                log_probs = nn.functional.log_softmax(policy_logits, dim=1)
                policy_loss = -(target_policies * log_probs).sum(dim=1).mean()

                # Value loss: MSE
                value_loss = nn.functional.mse_loss(values, target_values)

                # Combined loss
                loss = policy_loss + value_loss

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_batches += 1

        if total_batches == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        avg_policy = total_policy_loss / total_batches
        avg_value = total_value_loss / total_batches
        return {
            "policy_loss": avg_policy,
            "value_loss": avg_value,
            "total_loss": avg_policy + avg_value,
        }
