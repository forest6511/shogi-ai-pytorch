"""CLI entry point for shogi-ai — Human vs Random AI."""

from __future__ import annotations

from shogi_ai.engine.random_player import random_move
from shogi_ai.game.animal_shogi.display import (
    PIECE_NAMES_JA,
    board_to_str,
)
from shogi_ai.game.animal_shogi.moves import decode_move
from shogi_ai.game.animal_shogi.state import AnimalShogiState
from shogi_ai.game.animal_shogi.types import Player


def _format_move(move: int) -> str:
    """Format a move for display."""
    info = decode_move(move)
    if info["type"] == "board":
        fr, fc = info["from"]
        tr, tc = info["to"]
        return f"{chr(ord('a') + fc)}{fr + 1} -> {chr(ord('a') + tc)}{tr + 1}"
    else:
        pt = info["piece_type"]
        tr, tc = info["to"]
        return f"drop {PIECE_NAMES_JA[pt]} -> {chr(ord('a') + tc)}{tr + 1}"


def main() -> None:
    """Run a Human (SENTE) vs Random AI (GOTE) game."""
    print("=== どうぶつしょうぎ ===")
    print("You are SENTE (uppercase). AI is GOTE (lowercase).")
    print()

    state = AnimalShogiState()

    while not state.is_terminal:
        print(board_to_str(state.board))
        print()

        if state.current_player == Player.SENTE:
            # Human turn
            moves = state.legal_moves()
            print("Legal moves:")
            for i, m in enumerate(moves):
                print(f"  {i}: {_format_move(m)}")
            print()

            while True:
                try:
                    choice = input("Your move (number): ")
                    idx = int(choice)
                    if 0 <= idx < len(moves):
                        state = state.apply_move(moves[idx])
                        break
                    print(f"Invalid: choose 0-{len(moves) - 1}")
                except ValueError:
                    print("Enter a number.")
                except (EOFError, KeyboardInterrupt):
                    print("\nGame aborted.")
                    return
        else:
            # AI turn
            move = random_move(state)
            print(f"AI plays: {_format_move(move)}")
            state = state.apply_move(move)

        print()

    # Game over
    print(board_to_str(state.board))
    print()
    winner = state.winner
    if winner == Player.SENTE:
        print("You win!")
    elif winner == Player.GOTE:
        print("AI wins!")
    else:
        print("Draw!")


if __name__ == "__main__":
    main()
