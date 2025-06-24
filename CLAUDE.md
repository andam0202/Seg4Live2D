---
title: CLAUDE.md
created_at: 2025-06-24
updated_at: 2025-06-24
# このプロパティは、Claude Codeが関連するドキュメントの更新を検知するために必要です。消去しないでください。
---

このファイルは、[Claude Code](https://www.anthropic.com/claude-code) がSeg4Live2Dプロジェクトのコードを扱う際のガイダンスを提供します。

## 🔨 最重要ルール - 新しいルールの追加プロセス

以降の出力は日本語でしてください。

ユーザーから今回限りではなく常に対応が必要だと思われる指示を受けた場合：

1. 「これを標準のルールにしますか？」と質問する
2. YESの回答を得た場合、CLAUDE.mdに追加ルールとして記載する
3. 以降は標準ルールとして常に適用する

このプロセスにより、プロジェクトのルールを継続的に改善していきます。

また、実装ログをdocs/logs内にYYYY-mm-dd-hh-[適切な名前].mdに形式で、適宜残していってください。

## プロジェクト概要

**Seg4Live2D** は、SAM2 (Segment Anything Model 2) セグメンテーション技術を活用してイラストを自動的にLive2D用素材として分割するシステムです。

### 🎯 プロジェクトの目的
- **創作効率化**: 手動でのパーツ分割作業を自動化
- **品質向上**: 一貫した分割基準による高品質な素材生成  
- **アクセシビリティ**: Live2D制作の技術的ハードルを下げる

### 🎨 主要機能
- **プロンプトベースセグメンテーション**: SAM2を使った高精度なパーツ分割
- **Live2D特化プリセット**: 髪・顔・体・アクセサリーの自動認識
- **インタラクティブ編集**: 点・ボックス・マスクプロンプトによる精密制御
- **品質管理**: リアルタイムプレビューと手動調整機能

## 技術スタック

### コア技術
- **機械学習**: SAM2 (Segment Anything Model 2)
- **画像処理**: OpenCV, PIL/Pillow, scikit-image  
- **深層学習**: PyTorch
- **UI**: Streamlit / Gradio (Webベース)
- **言語**: Python 3.9+

### 開発環境
- **OS**: Windows 11 + WSL2 (Ubuntu)
- **開発**: VS Code + Python Extension
- **バージョン管理**: Git
- **依存関係管理**: uv (高速Pythonパッケージマネージャー)
- **コンテナ**: Docker (オプション)

## プロジェクト構造

```
Seg4Live2D/                           # プロジェクトルート
├── src/                               # ソースコード
│   ├── core/                          # コア機能
│   │   ├── sam2/                      # SAM2セグメンテーション処理  
│   │   │   ├── sam2_model.py          # SAM2モデル管理
│   │   │   ├── prompt_handler.py      # プロンプト処理
│   │   │   └── post_processor.py      # 後処理
│   │   ├── live2d/                    # Live2D特化処理
│   │   │   ├── mesh_generator.py      # メッシュ生成
│   │   │   ├── transparency.py        # 透明度処理
│   │   │   └── layer_manager.py       # レイヤー管理
│   │   └── utils/                     # ユーティリティ
│   │       ├── file_handler.py        # ファイル操作
│   │       ├── config.py              # 設定管理
│   │       └── logger.py              # ログ管理
│   ├── ui/                            # ユーザーインターフェース
│   │   ├── streamlit_app.py           # メインアプリケーション
│   │   ├── components/                # UIコンポーネント
│   │   └── static/                    # 静的ファイル
│   ├── training/                      # モデル学習
│   │   ├── dataset/                   # データセット管理
│   │   ├── train.py                   # 学習スクリプト
│   │   └── evaluate.py                # 評価スクリプト
│   └── api/                           # API (オプション)
│       ├── main.py                    # FastAPI main
│       └── endpoints/                 # API エンドポイント
├── models/                            # 学習済みモデル
│   ├── yolo_live2d_v1.pt             # カスタム学習モデル
│   └── checkpoints/                   # チェックポイント
├── data/                              # データファイル
│   ├── training/                      # 学習用データ
│   │   ├── images/                    # 学習画像
│   │   ├── labels/                    # アノテーション
│   │   └── annotations/               # セグメンテーションマスク
│   ├── test/                          # テスト用データ
│   └── output/                        # 出力結果
├── config/                            # 設定ファイル
│   ├── model_config.yaml              # モデル設定
│   ├── processing_config.yaml         # 処理設定
│   └── live2d_config.yaml            # Live2D出力設定
├── tests/                             # テストコード
│   ├── unit/                          # 単体テスト
│   ├── integration/                   # 統合テスト
│   └── fixtures/                      # テスト用データ
├── docs/                              # ドキュメント
│   ├── api/                           # API仕様
│   ├── tutorials/                     # チュートリアル
│   ├── architecture/                  # システム設計
│   └── research/                      # 技術調査・研究
├── scripts/                           # スクリプト
│   ├── setup.py                       # 環境セットアップ
│   ├── download_models.py             # モデルダウンロード
│   └── batch_process.py               # バッチ処理
├── requirements/                      # 依存関係
│   ├── base.txt                       # 基本依存関係
│   ├── dev.txt                        # 開発用
│   └── training.txt                   # 学習用
├── .env.example                       # 環境変数テンプレート
├── .gitignore                         # Git管理除外
├── pyproject.toml                     # プロジェクト設定
├── Dockerfile                         # Docker設定
├── docker-compose.yml                 # Docker Compose
├── README.md                          # プロジェクト説明
└── CLAUDE.md                          # このファイル
```

## 現在の実装ステータス

### Phase A: SAM2 Core Implementation ✅
- [x] SAM2ライブラリ統合
- [x] モデル管理システム (SAM2ModelManager)
- [x] プロンプトハンドラー (SAM2PromptHandler) 
- [x] Live2Dプリセットプロンプト
- [x] 高精度セグメンテーション実装
- [x] マルチパート分割テスト完了

**テスト結果**: 
- 髪セグメンテーション: スコア 0.811
- 顔セグメンテーション: スコア 0.902  
- 体セグメンテーション: スコア 0.927

### Phase B: Web UI 📋
- [ ] Streamlit/Gradio インターフェース
- [ ] インタラクティブプロンプト編集
- [ ] リアルタイムプレビュー
- [ ] マスク編集機能

### Phase C: Live2D統合 📋
- [ ] Live2D形式エクスポート
- [ ] マスク後処理（平滑化、ノイズ除去）
- [ ] 部位別レイヤー管理
- [ ] アニメーション対応

### Phase D: 運用最適化 📋  
- [ ] モデル量子化・高速化
- [ ] バッチ処理システム
- [ ] API サーバー化
- [ ] 推論キャッシュシステム

## Pythonコーディング規約

### 基本スタイル
- **フォーマッター**: Black (line-length=88)
- **リンター**: Ruff
- **型チェック**: mypy
- **インポート整理**: isort
- **ドキュメント**: Google Style Docstrings

### 命名規則
```python
# クラス: PascalCase
class ImageProcessor:
    pass

# 関数・変数: snake_case  
def process_image(input_path: str) -> np.ndarray:
    image_data = load_image(input_path)
    return image_data

# 定数: UPPER_SNAKE_CASE
MAX_IMAGE_SIZE = 2048
DEFAULT_MODEL_PATH = "models/yolo_live2d_v1.pt"

# プライベート: アンダースコア前置
def _internal_process(data: np.ndarray) -> np.ndarray:
    return data
```

### 型ヒントの使用
```python
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from pathlib import Path

def segment_image(
    image_path: Union[str, Path],
    model_config: Dict[str, Any],
    output_dir: Optional[Path] = None
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    画像をセグメンテーションして分割します。
    
    Args:
        image_path: 入力画像のパス
        model_config: モデル設定辞書
        output_dir: 出力ディレクトリ（オプション）
        
    Returns:
        分割された画像のリストとメタデータ辞書
        
    Raises:
        FileNotFoundError: 画像ファイルが見つからない場合
        ValueError: 無効な設定値の場合
    """
    # 実装
    pass
```

## 開発ワークフロー

### 環境セットアップ
```bash
# uvのインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# プロジェクト初期化・依存関係インストール
uv sync

# 開発用セットアップスクリプト実行
uv run python scripts/setup.py
```

### 日常的な開発フロー
```bash
# コード品質チェック
uv run ruff check src/
uv run black --check src/
uv run mypy src/

# テスト実行
uv run pytest tests/ -v

# カバレッジ確認
uv run pytest --cov=src tests/

# セキュリティチェック
uv run bandit -r src/
```

### Git管理

#### コミット規約
```bash
# 機能追加
git commit -m "feat: YOLO セグメンテーション機能を追加"

# バグ修正  
git commit -m "fix: 透明度処理のエラーを修正"

# UI改善
git commit -m "ui: Live2D出力プレビュー機能を実装"

# ドキュメント
git commit -m "docs: API仕様書を更新"

# テスト
git commit -m "test: セグメンテーション処理の単体テストを追加"
```

#### .gitignore設定
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
.env

# 機械学習
*.pt
*.pth
*.onnx
*.h5
data/output/
models/checkpoints/
wandb/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# ログ・キャッシュ
*.log
.cache/
.pytest_cache/
```

## Live2D特化開発指針

### セグメンテーション対象
```python
# Live2D用パーツ分類
LIVE2D_PARTS = {
    "face": ["head", "face_base", "cheek"],
    "eyes": ["eye_left", "eye_right", "pupil_left", "pupil_right"],
    "eyebrows": ["eyebrow_left", "eyebrow_right"], 
    "mouth": ["mouth_base", "teeth", "tongue"],
    "hair": ["hair_front", "hair_back", "hair_side"],
    "body": ["neck", "shoulder", "torso"],
    "arms": ["arm_left", "arm_right", "hand_left", "hand_right"],
    "clothing": ["top", "bottom", "accessories"],
    "background": ["bg_elements", "effects"]
}
```

### 出力形式
```python
# Live2D用出力仕様
OUTPUT_SPECS = {
    "format": "PNG",  # 透明度対応
    "resolution": (2048, 2048),  # 高解像度
    "channels": "RGBA",  # アルファチャンネル必須
    "mesh_ready": True,  # メッシュ変形対応
    "layer_separation": True,  # レイヤー分離
}
```

## 機械学習・モデル管理

### YOLOカスタム学習
```bash
# データセット準備
uv run python src/training/dataset/prepare_dataset.py

# 学習実行
uv run python src/training/train.py --config config/model_config.yaml

# 評価
uv run python src/training/evaluate.py --model models/yolo_live2d_v1.pt
```

### モデルバージョン管理
- **v1.0**: 基本的なパーツ分割（顔・体・髪）- YOLOv11ベース
- **v1.1**: 表情パーツ対応（目・眉・口）
- **v1.2**: 衣服・アクセサリー対応
- **v2.0**: Live2D最適化版

## テスト戦略

### テスト分類
```python
# 単体テスト例
def test_image_segmentation():
    """セグメンテーション処理のテスト"""
    processor = ImageProcessor()
    result = processor.segment(test_image_path)
    assert len(result.masks) > 0
    assert result.confidence > 0.5

# 統合テスト例  
def test_end_to_end_pipeline():
    """エンドツーエンドパイプラインのテスト"""
    pipeline = Live2DPipeline()
    output = pipeline.process(input_image, output_dir)
    assert output.success
    assert len(output.parts) >= 5
```

## デプロイメント・運用

### Docker化
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements/ requirements/
RUN pip install -r requirements/base.txt

COPY src/ src/
COPY models/ models/
COPY config/ config/

EXPOSE 8501
CMD ["streamlit", "run", "src/ui/streamlit_app.py"]
```

### 監視・ログ管理
```python
import logging
from src.core.utils.logger import setup_logger

logger = setup_logger(__name__)

def process_image(image_path: str):
    logger.info(f"Processing image: {image_path}")
    try:
        # 処理実行
        result = segment_image(image_path)
        logger.info(f"Segmentation completed: {len(result.masks)} parts found")
        return result
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise
```

## パフォーマンス・最適化

### GPU活用
```python
import torch

# GPU使用可能性チェック
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# モデルをGPUに移動
model = YOLO("models/yolo_live2d_v1.pt")
model = model.to(device)
```

### バッチ処理最適化
```python
def batch_process_images(
    image_paths: List[Path], 
    batch_size: int = 8
) -> List[SegmentationResult]:
    """複数画像のバッチ処理"""
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        batch_results = model.predict(batch, device=device)
        results.extend(batch_results)
    return results
```

## セキュリティ・品質管理

### セキュリティチェックリスト
- [ ] 入力ファイルの検証（サイズ・形式）
- [ ] パストラバーサル攻撃の防止
- [ ] 機密情報のログ出力防止
- [ ] APIレート制限の実装

### コード品質管理
```bash
# pre-commit フックの設定
pre-commit install

# CI/CDでの品質チェック
- ruff check
- black --check  
- mypy
- pytest --cov=80
- bandit -r src/
```

## トラブルシューティング

### よくある問題

#### 1. CUDA/GPU関連エラー
```bash
# PyTorch GPU対応確認
uv run python -c "import torch; print(torch.cuda.is_available())"

# CUDA バージョン確認
nvidia-smi
```

#### 2. メモリ不足エラー
```python
# バッチサイズを小さくする
batch_size = 4  # デフォルト8から削減

# ガベージコレクション強制実行
import gc
gc.collect()
torch.cuda.empty_cache()
```

#### 3. モデル読み込みエラー
```bash
# モデルファイルの整合性確認
uv run python scripts/download_models.py --verify

# 権限確認
ls -la models/
```

## プロジェクト固有設定

### 対象イラストスタイル
- **アニメ・マンガ風イラスト**: 主要ターゲット
- **リアル系イラスト**: サポート予定
- **ちびキャラ**: 専用モデル検討

### Live2D連携仕様
- **Cubism Editor**: 4.0以上対応
- **出力形式**: PSD互換レイヤー構造
- **メッシュ**: 自動生成 + 手動調整可能
- **物理演算**: パラメータ推定機能

### パフォーマンス目標
- **処理時間**: 1枚あたり30秒以内
- **精度**: mIoU 0.85以上
- **バッチ処理**: 同時10枚まで対応

---

このCLAUDE.mdは、プロジェクトの進展に合わせて継続的に更新してください。