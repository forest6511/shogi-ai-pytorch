"""Network configuration for dual-head AlphaZero-style models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for DualHeadNetwork.

    Attributes:
        board_h: Board height (rows).
        board_w: Board width (columns).
        in_channels: Number of input feature planes.
        action_size: Total number of possible actions.
        num_res_blocks: Number of residual blocks.
        num_channels: Number of channels in conv layers.
    """

    board_h: int
    board_w: int
    in_channels: int
    action_size: int
    num_res_blocks: int = 3
    num_channels: int = 64


# Preset for どうぶつしょうぎ (3x4 board, 14 input planes, 180 actions)
ANIMAL_SHOGI_CONFIG = NetworkConfig(
    board_h=4,
    board_w=3,
    in_channels=14,
    action_size=180,
    num_res_blocks=3,
    num_channels=64,
)

# Preset for 本将棋 (9x9 board, 43 input planes, action size TBD in Phase 8)
FULL_SHOGI_CONFIG = NetworkConfig(
    board_h=9,
    board_w=9,
    in_channels=43,
    action_size=2187,
    num_res_blocks=5,
    num_channels=128,
)
