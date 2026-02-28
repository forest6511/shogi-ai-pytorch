"""FastAPI web application for playing shogi against AI."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from shogi_ai.engine.mcts import MCTS, MCTSConfig
from shogi_ai.engine.minimax import minimax_move
from shogi_ai.engine.random_player import random_move
from shogi_ai.game.animal_shogi.display import board_to_str as animal_format
from shogi_ai.game.animal_shogi.moves import decode_move as animal_decode
from shogi_ai.game.animal_shogi.state import AnimalShogiState
from shogi_ai.game.full_shogi.display import format_board as full_format
from shogi_ai.game.full_shogi.moves import decode_move as full_decode
from shogi_ai.game.full_shogi.state import FullShogiState
from shogi_ai.game.protocol import GameState
from shogi_ai.model.config import ANIMAL_SHOGI_CONFIG
from shogi_ai.model.network import DualHeadNetwork

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Shogi AI")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory game storage
_games: dict[str, dict[str, Any]] = {}


class NewGameRequest(BaseModel):
    game_type: str = "animal"  # "animal" or "full"
    ai_type: str = "minimax"   # "minimax", "random", or "mcts"


class MoveRequest(BaseModel):
    game_id: str
    move: int


def _get_ai_fn(
    ai_type: str, game_type: str,
) -> Any:
    """Get the AI move function based on type."""
    if ai_type == "random":
        return lambda state: random_move(state)
    if ai_type == "minimax":
        depth = 4 if game_type == "animal" else 2
        return lambda state: minimax_move(state, depth=depth)
    if ai_type == "mcts":
        config = ANIMAL_SHOGI_CONFIG
        net = DualHeadNetwork(config)
        net.eval()
        mcts = MCTS(net, MCTSConfig(num_simulations=50))

        def mcts_move(state: GameState) -> int:
            probs = mcts.search(state)
            legal = state.legal_moves()
            best = max(legal, key=lambda m: probs[m])
            return best

        return mcts_move
    msg = f"Unknown AI type: {ai_type}"
    raise ValueError(msg)


def _state_to_dict(state: GameState, game_type: str) -> dict[str, Any]:
    """Convert game state to JSON-serializable dict."""
    board = state.board  # type: ignore[attr-defined]

    if game_type == "animal":
        board_display = animal_format(board)
        squares = []
        for piece in board.squares:
            if piece is None:
                squares.append(None)
            else:
                squares.append({
                    "type": piece.piece_type.value,
                    "owner": piece.owner.value,
                    "name": piece.piece_type.name,
                })
        hands = [
            [pt.name for pt in board.hands[0]],
            [pt.name for pt in board.hands[1]],
        ]
        rows, cols = 4, 3
    else:
        board_display = full_format(board)
        squares = []
        for piece in board.squares:
            if piece is None:
                squares.append(None)
            else:
                squares.append({
                    "type": piece.piece_type.value,
                    "owner": piece.owner.value,
                    "name": piece.piece_type.name,
                })
        hands = [
            [pt.name for pt in board.hands[0]],
            [pt.name for pt in board.hands[1]],
        ]
        rows, cols = 9, 9

    return {
        "current_player": state.current_player,
        "is_terminal": state.is_terminal,
        "winner": state.winner,
        "legal_moves": state.legal_moves(),
        "squares": squares,
        "hands": hands,
        "rows": rows,
        "cols": cols,
        "board_display": board_display,
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text())


@app.post("/api/new-game")
async def new_game(req: NewGameRequest) -> dict[str, Any]:
    game_id = str(uuid.uuid4())[:8]

    if req.game_type == "animal":
        state: GameState = AnimalShogiState()
    elif req.game_type == "full":
        state = FullShogiState()
    else:
        raise HTTPException(400, f"Unknown game type: {req.game_type}")

    ai_fn = _get_ai_fn(req.ai_type, req.game_type)

    _games[game_id] = {
        "state": state,
        "game_type": req.game_type,
        "ai_fn": ai_fn,
    }

    return {
        "game_id": game_id,
        "state": _state_to_dict(state, req.game_type),
    }


@app.post("/api/move")
async def make_move(req: MoveRequest) -> dict[str, Any]:
    game = _games.get(req.game_id)
    if game is None:
        raise HTTPException(404, "Game not found")

    state: GameState = game["state"]
    game_type: str = game["game_type"]

    if state.is_terminal:
        raise HTTPException(400, "Game is already over")

    if req.move not in state.legal_moves():
        raise HTTPException(400, f"Illegal move: {req.move}")

    # Apply player move
    state = state.apply_move(req.move)

    # AI response if game not over
    ai_move = None
    if not state.is_terminal:
        ai_fn = game["ai_fn"]
        ai_move = ai_fn(state)
        state = state.apply_move(ai_move)

    game["state"] = state

    decode_fn = animal_decode if game_type == "animal" else full_decode

    return {
        "state": _state_to_dict(state, game_type),
        "player_move": req.move,
        "ai_move": ai_move,
        "ai_move_decoded": decode_fn(ai_move) if ai_move is not None else None,
    }


@app.get("/api/state/{game_id}")
async def get_state(game_id: str) -> dict[str, Any]:
    game = _games.get(game_id)
    if game is None:
        raise HTTPException(404, "Game not found")
    return _state_to_dict(game["state"], game["game_type"])


def main() -> None:
    """Run the web server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
