# Seg4Live2D 使用ガイド

Live2D用パーツ自動分割システムの使用方法

## 🚀 クイックスタート

### 1. 統合実行（推奨）
```bash
# 初回セットアップ + セグメンテーション実行
uv run python scripts/setup_and_run.py --input data/samples/anime_woman1

# 特定パーツのみ処理
uv run python scripts/setup_and_run.py --input data/samples/anime_woman1 --parts face hair
```

### 2. 個別実行
```bash
# モデルダウンロード（初回のみ）
uv run python scripts/download_sam2_models.py

# セグメンテーション実行
uv run python scripts/sam2_segmentation.py --input data/samples/anime_woman1
```

## 📁 入力・出力

### 入力
- **対応形式**: PNG, JPG, JPEG
- **推奨サイズ**: 800×1000px程度
- **画像タイプ**: アニメキャラクター画像

### 出力
```
data/output/live2d_segmentation/
├── original_image.png          # 元画像
├── live2d_face_image.png      # 顔パーツ（透明PNG）
├── live2d_hair_image.png      # 髪パーツ（透明PNG）
├── live2d_body_image.png      # 体パーツ（透明PNG）
├── live2d_eyes_image.png      # 目パーツ（透明PNG）
├── mask_face_image.png        # 顔マスク（白黒）
└── viz_face_image.png         # 可視化画像
```

## 🎯 パーツ別分割

### 対象パーツ
- **face**: 顔の肌部分
- **hair**: 髪の毛
- **body**: 首から下の体
- **eyes**: 目の部分

### 精度データ
| パーツ | 精度 | カバー率 | 特徴 |
|--------|------|----------|------|
| Eyes | 高精度 | 0.87% | 効率性55.8（最高） |
| Face | 高精度 | 16.74% | 肌部分のみ抽出 |
| Hair | 中精度 | 42.28% | 髪の境界認識 |
| Body | 高精度 | 21.20% | 体のシルエット |

## ⚙️ オプション

### 基本オプション
```bash
--input, -i     # 入力画像フォルダ（必須）
--output, -o    # 出力フォルダ（デフォルト: data/output/live2d_segmentation）
--parts, -p     # 対象パーツ（デフォルト: 全パーツ）
--pattern       # 画像ファイルパターン（デフォルト: *.png）
--max-images    # 最大処理画像数
--verbose, -v   # 詳細ログ表示
```

### セットアップオプション
```bash
--check-only      # モデル存在確認のみ
--force-download  # 強制再ダウンロード
--skip-download   # ダウンロードスキップ
```

## 📋 使用例

### 1. 基本的な使用
```bash
# デフォルト設定で実行
uv run python scripts/sam2_segmentation.py --input data/samples/anime_woman1
```

### 2. カスタム設定
```bash
# 顔と髪のみ、JPEGファイル対象
uv run python scripts/sam2_segmentation.py \
  --input data/samples/photos \
  --parts face hair \
  --pattern "*.jpg" \
  --output my_results
```

### 3. 大量処理
```bash
# 最大10枚まで処理
uv run python scripts/sam2_segmentation.py \
  --input data/samples/anime_images \
  --max-images 10 \
  --verbose
```

### 4. 統合実行
```bash
# セットアップから実行まで一括
uv run python scripts/setup_and_run.py \
  --input data/samples/anime_woman1 \
  --parts face hair body
```

## 🔧 トラブルシューティング

### モデルダウンロードエラー
```bash
# 強制再ダウンロード
uv run python scripts/setup_and_run.py --force-download
```

### CUDA/GPU エラー
```bash
# CPU使用の場合、config/app/development.yamlで設定変更
device: "cpu"
```

### メモリ不足エラー
```bash
# 画像数を制限
uv run python scripts/sam2_segmentation.py --input data/samples/anime_woman1 --max-images 1
```

## 📈 性能向上のコツ

### 1. 入力画像の最適化
- **解像度**: 800×1000px程度が最適
- **形式**: PNG推奨（透明度対応）
- **品質**: 明瞭なキャラクター画像

### 2. パーツ選択の最適化
```bash
# 高精度パーツのみ処理
--parts eyes face body

# 髪が複雑な場合はスキップ
--parts face body eyes
```

### 3. バッチ処理の最適化
```bash
# 少数ずつ処理
--max-images 5
```

## 🎨 Live2D での使用

### 1. ファイル使用方法
- `live2d_*.png` ファイルをLive2D Cubism Editorにインポート
- 透明度情報がそのまま使用可能
- レイヤー分けが自動的に完了

### 2. 推奨ワークフロー
1. Seg4Live2Dでパーツ分割
2. `live2d_*.png` をCubism Editorにインポート
3. 必要に応じて手動微調整
4. メッシュ生成・アニメーション設定

### 3. 品質チェック
- `viz_*.png` で分割結果を確認
- 境界線が期待通りかチェック
- 必要に応じて再実行

## 💡 ベストプラクティス

### 1. プロジェクト構成
```
your_project/
├── input_images/           # 元画像
├── seg4live2d_output/     # セグメンテーション結果
└── live2d_project/        # Live2Dプロジェクト
```

### 2. ファイル命名
- 元画像名を分かりやすく
- 出力フォルダを用途別に分離

### 3. 品質管理
- まず1枚でテスト実行
- 結果確認後にバッチ処理
- 定期的な手動チェック

## 🔗 関連リンク

- [Live2D Cubism Editor](https://www.live2d.com/)
- [SAM2 公式リポジトリ](https://github.com/facebookresearch/segment-anything-2)
- [プロジェクトGitHub](https://github.com/andam0202/Seg4Live2D)