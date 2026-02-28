# shogi-ai-pytorch

「Claude Codeで作る将棋AI — PyTorch×強化学習×バイブコーディング実践入門」のコンパニオンリポジトリです。

## 必要環境

- Python 3.13+
- macOS (Apple Silicon: M1/M2/M3/M4)
- [uv](https://docs.astral.sh/uv/) パッケージマネージャ

## セットアップ

```bash
git clone https://github.com/forest6511/shogi-ai-pytorch.git
cd shogi-ai-pytorch
uv sync
```

## 実行

```bash
# どうぶつしょうぎ対局（Human vs Random AI）
uv run shogi-ai
```

## テスト

```bash
uv run pytest -v
```

## 章・コード対応表

| 章 | 内容 | Git タグ | 主要ファイル |
|----|------|---------|-------------|
| Ch.1 | LLMとバイブコーディング | — | — |
| Ch.2 | 環境構築 | — | — |
| Ch.3 | Python速習 | — | — |
| Ch.4 | どうぶつしょうぎルール | `v0.2.0` | `src/shogi_ai/game/animal_shogi/` |
| Ch.5 | テスト駆動開発 | `v0.2.0` | `tests/game/animal_shogi/` |
| Ch.6 | CUI対局 | `v0.2.0` | `src/shogi_ai/cli.py` |
| Ch.7 | ニューラルネットワーク基礎 | `v0.3.0` | — |
| Ch.8 | PolicyNetwork | `v0.4.0` | `src/shogi_ai/model/` |
| Ch.9 | 自己対局データ生成 | `v0.4.0` | `src/shogi_ai/training/` |
| Ch.10 | 訓練ループ | `v0.5.0` | `src/shogi_ai/training/` |
| Ch.11 | MCTS | `v0.6.0` | `src/shogi_ai/engine/mcts.py` |
| Ch.12 | AlphaZeroパイプライン | `v0.7.0` | `src/shogi_ai/training/` |
| Ch.13 | 本将棋への拡張 | `v0.8.0` | `src/shogi_ai/game/shogi/` |
| Ch.14 | 対局分析 | `v0.9.0` | — |
| Ch.15 | Web UI | `v1.0.0` | `src/shogi_ai/web/` |

## ライセンス

MIT License
