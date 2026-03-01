"""FastAPI web application for playing shogi against AI.

FastAPI を使った将棋AI Web アプリケーション。
ブラウザから将棋AIと対戦できる REST API を提供する。

エンドポイント:
  GET  /              — フロントエンドの HTML を返す
  POST /api/new-game  — 新規対局を開始（ゲームIDを返す）
  POST /api/move      — プレイヤーが手を指す（AIが応答して次局面を返す）
  GET  /api/state/{id} — 現在の局面情報を取得
"""

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

# 静的ファイル（HTML, CSS, JS）のディレクトリ
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Shogi AI")
# /static/ 以下で静的ファイルを配信
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 対局情報のインメモリストレージ（サーバ再起動で消える簡易実装）
# 本番環境では Redis や DB に保存する
_games: dict[str, dict[str, Any]] = {}


class NewGameRequest(BaseModel):
    """新規対局リクエストのスキーマ。"""

    game_type: str = "animal"   # "animal"（どうぶつしょうぎ）or "full"（本将棋）
    ai_type: str = "minimax"    # "minimax", "random", "mcts"


class MoveRequest(BaseModel):
    """指し手リクエストのスキーマ。"""

    game_id: str  # 対局ID（/api/new-game で取得）
    move: int     # 手のエンコード値（整数）


def _get_ai_fn(
    ai_type: str, game_type: str,
) -> Any:
    """Get the AI move function based on type.

    AI種別に応じた手選択関数を返す。
    """
    if ai_type == "random":
        # ランダムAI（最弱、動作確認用）
        return lambda state: random_move(state)
    if ai_type == "minimax":
        # ミニマックスAI: どうぶつは深さ4、本将棋は深さ2（計算時間の兼ね合い）
        depth = 4 if game_type == "animal" else 2
        return lambda state: minimax_move(state, depth=depth)
    if ai_type == "mcts":
        # MCTSAIを初期化（未学習のランダムネットワークを使用）
        config = ANIMAL_SHOGI_CONFIG
        net = DualHeadNetwork(config)
        net.eval()  # 推論モード
        mcts = MCTS(net, MCTSConfig(num_simulations=50))

        def mcts_move(state: GameState) -> int:
            probs = mcts.search(state)
            legal = state.legal_moves()
            # 最も確率の高い合法手を選択
            best = max(legal, key=lambda m: probs[m])
            return best

        return mcts_move
    msg = f"Unknown AI type: {ai_type}"
    raise ValueError(msg)


def _state_to_dict(state: GameState, game_type: str) -> dict[str, Any]:
    """Convert game state to JSON-serializable dict.

    局面情報を JSON 形式（辞書）に変換する。
    フロントエンドの JavaScript がこの形式を受け取って盤面を描画する。
    """
    board = state.board  # type: ignore[attr-defined]

    if game_type == "animal":
        board_display = animal_format(board)
        squares = []
        for piece in board.squares:
            if piece is None:
                squares.append(None)
            else:
                squares.append({
                    "type": piece.piece_type.value,   # 駒種インデックス
                    "owner": piece.owner.value,        # 所有者（0=先手, 1=後手）
                    "name": piece.piece_type.name,     # 駒名（文字列）
                })
        hands = [
            [pt.name for pt in board.hands[0]],  # 先手の持ち駒
            [pt.name for pt in board.hands[1]],  # 後手の持ち駒
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
        "current_player": state.current_player,   # 手番（0=先手, 1=後手）
        "is_terminal": state.is_terminal,          # 終局フラグ
        "winner": state.winner,                    # 勝者（None=対局中）
        "legal_moves": state.legal_moves(),        # 合法手リスト
        "squares": squares,                        # 盤面の駒情報（81または12要素）
        "hands": hands,                            # 持ち駒情報
        "rows": rows,
        "cols": cols,
        "board_display": board_display,            # テキスト形式の盤面表示
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """フロントエンドの HTML を配信する。"""
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text())


@app.post("/api/new-game")
async def new_game(req: NewGameRequest) -> dict[str, Any]:
    """新規対局を開始する。

    対局IDと初期局面情報を返す。
    対局IDはその後の手番送信（/api/move）で使用する。
    """
    game_id = str(uuid.uuid4())[:8]  # 短いIDを生成

    # ゲーム種別に応じた初期局面を作成
    if req.game_type == "animal":
        state: GameState = AnimalShogiState()
    elif req.game_type == "full":
        state = FullShogiState()
    else:
        raise HTTPException(400, f"Unknown game type: {req.game_type}")

    # AI の手選択関数を初期化
    ai_fn = _get_ai_fn(req.ai_type, req.game_type)

    # 対局情報をメモリに保存
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
    """プレイヤーの手を受け取り、AIが応答して次の局面を返す。

    処理フロー:
    1. プレイヤーの手を検証して適用
    2. AI がミニマックス/MCTS で最善手を計算
    3. AI の手を適用して新局面を返す
    """
    game = _games.get(req.game_id)
    if game is None:
        raise HTTPException(404, "Game not found")

    state: GameState = game["state"]
    game_type: str = game["game_type"]

    if state.is_terminal:
        raise HTTPException(400, "Game is already over")

    if req.move not in state.legal_moves():
        raise HTTPException(400, f"Illegal move: {req.move}")

    # プレイヤーの手を適用
    state = state.apply_move(req.move)

    # ゲームが終わっていなければ AI が応答
    ai_move = None
    if not state.is_terminal:
        ai_fn = game["ai_fn"]
        ai_move = ai_fn(state)
        state = state.apply_move(ai_move)

    # 最新の局面を保存
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
    """現在の局面情報を取得する（ページ再読み込み時などに使用）。"""
    game = _games.get(game_id)
    if game is None:
        raise HTTPException(404, "Game not found")
    return _state_to_dict(game["state"], game["game_type"])


def main() -> None:
    """Run the web server.

    `uv run shogi-web` または `python -m shogi_ai.web.app` で起動する。
    ブラウザで http://localhost:8000 にアクセスして対局できる。
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
