# Seg4Live2D

YOLOv11セグメンテーション技術を活用してイラストを自動的にLive2D用素材として分割するシステム

## 🎯 概要

Seg4Live2Dは、最新のYOLOv11セグメンテーション技術を使用して、アニメ・マンガ風イラストを自動的にLive2D用の素材として分割するWebアプリケーションです。

### 主要機能

- **🤖 自動セグメンテーション**: YOLOv11による高精度なパーツ分割
- **🎨 Live2D最適化**: メッシュ変形・透明度処理に対応した出力
- **⚡ バッチ処理**: 複数イラストの一括処理
- **✋ 品質管理**: 人手による確認・修正インターフェース
- **🌐 Webベース**: ブラウザで動作するユーザーフレンドリーなUI

### 対象パーツ

- **顔**: 頭部、頬、基本フェイス
- **目**: 左右の目、瞳孔、まつ毛
- **眉**: 左右の眉毛
- **口**: 口の基本形、歯、舌
- **髪**: 前髪、後髪、サイド
- **体**: 首、肩、胴体
- **腕**: 左右の腕、手
- **服装**: トップス、ボトムス、アクセサリー
- **背景**: 背景要素、エフェクト

## 🚀 クイックスタート

### 前提条件

- Python 3.9-3.12
- uv (パッケージマネージャー)
- NVIDIA GPU (推奨、CPUでも動作可能)

### インストール

```bash
# リポジトリをクローン
git clone https://github.com/username/seg4live2d.git
cd seg4live2d

# uvをインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係をインストール
uv sync

# セットアップスクリプトを実行
uv run python scripts/setup.py
```

### 使用方法

#### Webアプリの起動

```bash
# Streamlitアプリを起動
uv run streamlit run src/ui/streamlit_app.py
```

ブラウザで `http://localhost:8501` にアクセス

#### CLIの使用

```bash
# 単一画像の処理
uv run seg4live2d segment path/to/image.png

# バッチ処理
uv run seg4live2d batch path/to/images/

# 学習
uv run seg4live2d-train --config config/model_config.yaml
```

## 📋 技術仕様

### 技術スタック

- **機械学習**: YOLOv11 (Ultralytics)
- **画像処理**: OpenCV, PIL/Pillow, scikit-image
- **深層学習**: PyTorch
- **UI**: Streamlit
- **API**: FastAPI (オプション)
- **パッケージ管理**: uv

### パフォーマンス目標

- **処理時間**: 30秒以内/枚
- **精度**: mIoU 0.85以上
- **バッチ処理**: 同時10枚まで対応
- **対応解像度**: 512x512 ～ 4096x4096px

### 対応フォーマット

#### 入力
- PNG, JPEG, WebP, TIFF

#### 出力
- PNG (透明度対応)
- PSD (レイヤー構造)
- JSON (メタデータ)

## 🏗️ プロジェクト構造

```
Seg4Live2D/
├── src/                    # ソースコード
│   ├── core/              # コア機能
│   ├── ui/                # ユーザーインターフェース
│   ├── training/          # モデル学習
│   └── api/               # API (オプション)
├── models/                # 学習済みモデル
├── data/                  # データファイル
├── config/                # 設定ファイル
├── tests/                 # テストコード
├── docs/                  # ドキュメント
└── scripts/               # スクリプト
```

## 🧪 開発

### 開発環境セットアップ

```bash
# 開発用依存関係をインストール
uv sync --group dev

# pre-commitフックをセットアップ
uv run pre-commit install
```

### コード品質チェック

```bash
# リンター実行
uv run ruff check src/

# フォーマッター実行
uv run black src/

# 型チェック
uv run mypy src/
```

### テスト実行

```bash
# 全テスト実行
uv run pytest

# カバレッジ付きテスト
uv run pytest --cov=src

# 特定のテストマーク
uv run pytest -m "not slow"
```

### モデル学習

```bash
# データセット準備
uv run python src/training/dataset/prepare_dataset.py

# 学習実行
uv run python src/training/train.py --config config/model_config.yaml

# 評価
uv run python src/training/evaluate.py --model models/yolo_live2d_v1.pt
```

## 📚 ドキュメント

詳細なドキュメントは [`docs/`](docs/) ディレクトリに格納されています：

- [システム設計](docs/architecture/system_design.md)
- [技術要件](docs/specifications/technical_requirements.md)
- [実装計画](docs/planning/implementation_plan.md)
- [ディレクトリ構成](docs/specifications/directory_structure.md)

## 🤝 貢献

プロジェクトへの貢献を歓迎します！

1. フォークを作成
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

### 貢献ガイドライン

- [コーディング規約](docs/development/coding_standards.md)
- [テストガイドライン](docs/development/testing_guidelines.md)
- [貢献ガイド](docs/development/contributing.md)

## ⚖️ ライセンス

このプロジェクトはMITライセンスの下で公開されています。

### 注意事項

- YOLOv11はAGPL-3.0ライセンスです（商用利用には別途ライセンスが必要）
- 学習データの著作権にご注意ください
- Live2D Cubism Editorとの互換性確認を推奨します

## 🔗 関連リンク

- [YOLOv11 (Ultralytics)](https://github.com/ultralytics/ultralytics)
- [Live2D Cubism](https://www.live2d.com/)
- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/)

## 📞 サポート

- **バグレポート**: [GitHub Issues](https://github.com/username/seg4live2d/issues)
- **機能要求**: [GitHub Discussions](https://github.com/username/seg4live2d/discussions)
- **ドキュメント**: [docs/](docs/)

## 🙏 謝辞

- [Ultralytics](https://ultralytics.com/) - YOLOv11の開発
- [Live2D Inc.](https://www.live2d.com/) - Live2D技術
- オープンソースコミュニティの皆様

---

**⭐ このプロジェクトが役に立った場合は、スターをお願いします！**