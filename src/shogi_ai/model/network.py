"""AlphaZero-style dual-head neural network."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from shogi_ai.model.config import NetworkConfig


class ResBlock(nn.Module):
    """Residual block: Conv → BN → ReLU → Conv → BN + skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = torch.relu(out + residual)
        return out


class DualHeadNetwork(nn.Module):
    """AlphaZero-style network with policy and value heads.

    Input:  (batch, in_channels, board_h, board_w)
    Output: (policy_logits, value)
        - policy_logits: (batch, action_size) — raw logits
        - value: (batch, 1) — tanh-bounded value in [-1, 1]
    """

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config

        # Input projection
        self.input_conv = nn.Conv2d(
            config.in_channels, config.num_channels, 3, padding=1, bias=False,
        )
        self.input_bn = nn.BatchNorm2d(config.num_channels)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(config.num_channels) for _ in range(config.num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(config.num_channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(
            2 * config.board_h * config.board_w, config.action_size,
        )

        # Value head
        self.value_conv = nn.Conv2d(config.num_channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(config.board_h * config.board_w, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Common body
        x = torch.relu(self.input_bn(self.input_conv(x)))
        x = self.res_blocks(x)

        # Policy head
        p = torch.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value head
        v = torch.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = torch.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v
