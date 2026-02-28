# CLAUDE.md — shogi-ai-pytorch 公開リポジトリ

## プロジェクト概要

「Claude Codeで作る将棋AI」のコンパニオンリポジトリ。
読者がバイブコーディングで構築する将棋AIの完成コード。

## 技術スタック

| 技術 | バージョン | 用途 |
|------|-----------|------|
| Python | 3.13+ | メイン言語 |
| PyTorch | 2.10+ | ニューラルネットワーク |
| uv | 0.10+ | パッケージマネージャ |
| pytest | 9.0+ | テスト |
| ruff | 0.15+ | Linter |

## コード規約

- **src レイアウト**: `src/shogi_ai/` にパッケージ配置
- **不変 State**: `apply_move()` は新しい State を返す（MCTS 用）
- **Protocol**: `typing.Protocol`（PEP 544）で共通インターフェース定義
- **フルスクラッチ**: cshogi 等の外部将棋ライブラリ不使用
- **型ヒント**: 全関数に型アノテーション必須
- **ruff**: `E`, `F`, `I`, `W`, `UP` ルール、行長 99

## デバイス戦略

```python
import torch

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

- 書籍は Apple Silicon (MPS) 前提
- CI は CPU のみ（ubuntu-latest）
- CUDA 対応は明示的にスコープ外

## ディレクトリ構造

```
src/shogi_ai/
├── game/
│   ├── protocol.py          # GameState Protocol
│   └── animal_shogi/        # どうぶつしょうぎ
├── engine/                   # AI エンジン
├── model/                    # ニューラルネットワーク
├── training/                 # 訓練パイプライン
├── web/                      # Web UI
└── cli.py                    # CLI エントリポイント
```

## テスト

```bash
uv run pytest -v              # 全テスト
uv run pytest tests/game/ -v  # ゲームロジックのみ
uv run ruff check src/ tests/ # Lint
```

## Git タグ規約

- `v0.2.0` — Phase 2: どうぶつしょうぎエンジン
- `v0.3.0` — Phase 3: ニューラルネットワーク基礎
- `v0.4.0` — Phase 4: PolicyNetwork + 自己対局
- 以降、書籍フェーズに対応
