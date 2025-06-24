# Seg4Live2D ディレクトリ構成仕様書

## 1. 全体構成

```
Seg4Live2D/                           # プロジェクトルート
├── src/                               # ソースコード
├── models/                            # 学習済みモデル
├── data/                              # データファイル
├── config/                            # 設定ファイル
├── tests/                             # テストコード
├── docs/                              # ドキュメント
├── scripts/                           # スクリプト
├── requirements/                      # 依存関係
├── docker/                            # Docker関連ファイル
├── .github/                           # GitHub設定
├── logs/                              # ログファイル
├── .env.example                       # 環境変数テンプレート
├── .gitignore                         # Git管理除外
├── pyproject.toml                     # プロジェクト設定
├── README.md                          # プロジェクト説明
└── CLAUDE.md                          # Claude Code設定
```

## 2. ソースコード構成 (src/)

### 2.1 基本構造

```
src/
├── core/                              # コア機能
│   ├── __init__.py
│   ├── segmentation/                  # セグメンテーション処理
│   │   ├── __init__.py
│   │   ├── yolo_model.py              # YOLOモデル管理
│   │   ├── image_processor.py         # 画像前処理
│   │   ├── post_processor.py          # 後処理
│   │   ├── batch_processor.py         # バッチ処理
│   │   └── quality_evaluator.py       # 品質評価
│   ├── live2d/                        # Live2D特化処理
│   │   ├── __init__.py
│   │   ├── layer_manager.py           # レイヤー管理
│   │   ├── transparency.py            # 透明度処理
│   │   ├── mesh_generator.py          # メッシュ生成
│   │   ├── psd_exporter.py            # PSD出力
│   │   └── parts_classifier.py        # パーツ分類
│   └── utils/                         # ユーティリティ
│       ├── __init__.py
│       ├── config.py                  # 設定管理
│       ├── logger.py                  # ログ管理
│       ├── file_handler.py            # ファイル操作
│       ├── image_utils.py             # 画像ユーティリティ
│       ├── validation.py              # バリデーション
│       └── exceptions.py              # カスタム例外
├── ui/                                # ユーザーインターフェース
│   ├── __init__.py
│   ├── streamlit_app.py               # メインアプリケーション
│   ├── components/                    # UIコンポーネント
│   │   ├── __init__.py
│   │   ├── upload_component.py        # アップロード機能
│   │   ├── preview_component.py       # プレビュー機能
│   │   ├── settings_component.py      # 設定パネル
│   │   ├── results_component.py       # 結果表示
│   │   └── batch_component.py         # バッチ処理UI
│   ├── pages/                         # ページ
│   │   ├── __init__.py
│   │   ├── main_page.py               # メインページ
│   │   ├── batch_page.py              # バッチ処理ページ
│   │   ├── settings_page.py           # 設定ページ
│   │   └── help_page.py               # ヘルプページ
│   └── static/                        # 静的ファイル
│       ├── css/
│       │   └── custom.css
│       ├── js/
│       │   └── custom.js
│       └── images/
│           ├── logo.png
│           └── icons/
├── training/                          # モデル学習
│   ├── __init__.py
│   ├── dataset/                       # データセット管理
│   │   ├── __init__.py
│   │   ├── dataset_manager.py         # データセット管理
│   │   ├── annotation_parser.py       # アノテーション解析
│   │   ├── data_augmentation.py       # データ拡張
│   │   └── data_validator.py          # データ検証
│   ├── models/                        # モデル定義
│   │   ├── __init__.py
│   │   ├── yolo_custom.py             # カスタムYOLO
│   │   ├── model_factory.py           # モデル生成
│   │   └── model_registry.py          # モデル登録
│   ├── train.py                       # 学習スクリプト
│   ├── evaluate.py                    # 評価スクリプト
│   ├── export.py                      # モデルエクスポート
│   └── metrics.py                     # 評価指標
├── api/                               # API (オプション)
│   ├── __init__.py
│   ├── main.py                        # FastAPI main
│   ├── routers/                       # ルーター
│   │   ├── __init__.py
│   │   ├── upload.py                  # アップロードAPI
│   │   ├── segmentation.py            # セグメンテーションAPI
│   │   ├── export.py                  # エクスポートAPI
│   │   └── batch.py                   # バッチ処理API
│   ├── models/                        # APIモデル
│   │   ├── __init__.py
│   │   ├── request_models.py          # リクエストモデル
│   │   ├── response_models.py         # レスポンスモデル
│   │   └── error_models.py            # エラーモデル
│   ├── middleware/                    # ミドルウェア
│   │   ├── __init__.py
│   │   ├── auth.py                    # 認証
│   │   ├── cors.py                    # CORS設定
│   │   └── rate_limit.py              # レート制限
│   └── dependencies.py                # 依存関係
└── cli/                               # CLI (オプション)
    ├── __init__.py
    ├── main.py                        # CLIメイン
    ├── commands/                      # コマンド
    │   ├── __init__.py
    │   ├── segment.py                 # セグメンテーションコマンド
    │   ├── batch.py                   # バッチ処理コマンド
    │   ├── train.py                   # 学習コマンド
    │   └── export.py                  # エクスポートコマンド
    └── utils.py                       # CLIユーティリティ
```

### 2.2 データ構造

```python
# データクラス例
@dataclass
class ImageMetadata:
    """画像メタデータ"""
    id: str
    filename: str
    width: int
    height: int
    format: str
    size_bytes: int
    upload_time: datetime
    processing_status: ProcessingStatus

@dataclass
class SegmentationResult:
    """セグメンテーション結果"""
    image_id: str
    masks: List[np.ndarray]
    classes: List[str]
    confidence_scores: List[float]
    bounding_boxes: List[Tuple[int, int, int, int]]
    processing_time: float
    model_version: str

@dataclass
class Live2DOutput:
    """Live2D出力データ"""
    image_id: str
    layers: Dict[str, LayerData]
    mesh_data: Dict[str, MeshInfo]
    export_path: str
    metadata: Dict[str, Any]
```

## 3. モデルファイル構成 (models/)

```
models/
├── yolo/                              # YOLOモデル
│   ├── pretrained/                    # 事前学習モデル
│   │   ├── yolo11n-seg.pt            # YOLOv11 nano
│   │   ├── yolo11s-seg.pt            # YOLOv11 small
│   │   ├── yolo11m-seg.pt            # YOLOv11 medium
│   │   └── yolo11l-seg.pt            # YOLOv11 large
│   ├── custom/                        # カスタムモデル
│   │   ├── live2d_v1.0.pt            # Live2D特化v1.0
│   │   ├── live2d_v1.1.pt            # Live2D特化v1.1
│   │   └── live2d_latest.pt          # 最新モデル（シンボリックリンク）
│   └── experiments/                   # 実験用モデル
│       ├── exp_001_anime_focused.pt   # アニメ特化実験
│       ├── exp_002_high_resolution.pt # 高解像度実験
│       └── benchmarks/                # ベンチマーク結果
├── checkpoints/                       # チェックポイント
│   ├── training_runs/                 # 学習実行ログ
│   │   ├── run_001/
│   │   ├── run_002/
│   │   └── latest/
│   └── best_models/                   # ベストモデル保存
└── config/                            # モデル設定
    ├── model_configs.yaml             # モデル設定一覧
    ├── class_definitions.yaml         # クラス定義
    └── training_configs/              # 学習設定
        ├── base_config.yaml
        ├── anime_config.yaml
        └── high_res_config.yaml
```

## 4. データファイル構成 (data/)

```
data/
├── training/                          # 学習用データ
│   ├── images/                        # 学習画像
│   │   ├── anime/                     # アニメ系画像
│   │   ├── realistic/                 # リアル系画像
│   │   └── chibi/                     # ちび系画像
│   ├── labels/                        # YOLOラベル
│   │   ├── anime/
│   │   ├── realistic/
│   │   └── chibi/
│   ├── annotations/                   # セグメンテーションマスク
│   │   ├── anime/
│   │   ├── realistic/
│   │   └── chibi/
│   ├── splits/                        # データ分割
│   │   ├── train.txt                  # 学習データリスト
│   │   ├── val.txt                    # 検証データリスト
│   │   └── test.txt                   # テストデータリスト
│   └── metadata/                      # メタデータ
│       ├── dataset_info.yaml          # データセット情報
│       ├── class_distribution.json    # クラス分布
│       └── quality_metrics.json       # 品質指標
├── test/                              # テスト用データ
│   ├── unit_test/                     # 単体テスト用
│   │   ├── small_images/              # 小サイズ画像
│   │   ├── medium_images/             # 中サイズ画像
│   │   └── large_images/              # 大サイズ画像
│   ├── integration_test/              # 統合テスト用
│   │   ├── typical_cases/             # 典型的ケース
│   │   ├── edge_cases/                # エッジケース
│   │   └── error_cases/               # エラーケース
│   └── benchmark/                     # ベンチマーク用
│       ├── accuracy_test/             # 精度テスト
│       ├── performance_test/          # パフォーマンステスト
│       └── reference_results/         # 参照結果
├── samples/                           # サンプルデータ
│   ├── demo_images/                   # デモ用画像
│   ├── tutorial_data/                 # チュートリアル用
│   └── showcase/                      # ショーケース用
├── output/                            # 出力結果
│   ├── segmentation/                  # セグメンテーション結果
│   ├── live2d/                        # Live2D素材
│   ├── batch_results/                 # バッチ処理結果
│   └── exports/                       # エクスポート結果
└── temp/                              # 一時ファイル
    ├── uploads/                       # アップロード一時保存
    ├── processing/                    # 処理中ファイル
    └── cache/                         # キャッシュファイル
```

## 5. 設定ファイル構成 (config/)

```
config/
├── app/                               # アプリケーション設定
│   ├── default.yaml                   # デフォルト設定
│   ├── development.yaml               # 開発環境設定
│   ├── production.yaml                # 本番環境設定
│   └── testing.yaml                   # テスト環境設定
├── models/                            # モデル設定
│   ├── yolo_config.yaml               # YOLO設定
│   ├── segmentation_config.yaml       # セグメンテーション設定
│   └── live2d_config.yaml            # Live2D出力設定
├── processing/                        # 処理設定
│   ├── image_processing.yaml          # 画像処理設定
│   ├── batch_processing.yaml          # バッチ処理設定
│   └── quality_settings.yaml          # 品質設定
├── ui/                                # UI設定
│   ├── streamlit_config.toml          # Streamlit設定
│   ├── theme.yaml                     # テーマ設定
│   └── layout.yaml                    # レイアウト設定
├── logging/                           # ログ設定
│   ├── logging_config.yaml            # ログ設定
│   ├── development_logging.yaml       # 開発用ログ設定
│   └── production_logging.yaml        # 本番用ログ設定
└── deployment/                        # デプロイ設定
    ├── docker_config.yaml             # Docker設定
    ├── k8s_config.yaml                # Kubernetes設定
    └── monitoring_config.yaml         # 監視設定
```

## 6. テストファイル構成 (tests/)

```
tests/
├── unit/                              # 単体テスト
│   ├── core/                          # コア機能テスト
│   │   ├── test_segmentation/
│   │   │   ├── test_yolo_model.py
│   │   │   ├── test_image_processor.py
│   │   │   └── test_post_processor.py
│   │   ├── test_live2d/
│   │   │   ├── test_layer_manager.py
│   │   │   ├── test_transparency.py
│   │   │   └── test_mesh_generator.py
│   │   └── test_utils/
│   │       ├── test_config.py
│   │       ├── test_logger.py
│   │       └── test_file_handler.py
│   ├── ui/                            # UI機能テスト
│   │   ├── test_components/
│   │   └── test_pages/
│   ├── training/                      # 学習機能テスト
│   │   ├── test_dataset/
│   │   └── test_models/
│   └── api/                           # API機能テスト
│       ├── test_routers/
│       └── test_models/
├── integration/                       # 統合テスト
│   ├── test_segmentation_pipeline.py  # セグメンテーションパイプライン
│   ├── test_live2d_pipeline.py       # Live2Dパイプライン
│   ├── test_ui_integration.py         # UI統合テスト
│   └── test_api_integration.py        # API統合テスト
├── performance/                       # パフォーマンステスト
│   ├── test_processing_speed.py       # 処理速度テスト
│   ├── test_memory_usage.py           # メモリ使用量テスト
│   ├── test_gpu_utilization.py        # GPU使用率テスト
│   └── test_batch_performance.py      # バッチ処理性能テスト
├── e2e/                              # エンドツーエンドテスト
│   ├── test_user_scenarios.py         # ユーザーシナリオテスト
│   ├── test_batch_scenarios.py        # バッチ処理シナリオ
│   └── test_error_scenarios.py        # エラーシナリオテスト
├── fixtures/                          # テストデータ
│   ├── images/                        # テスト画像
│   ├── models/                        # テスト用モデル
│   ├── configs/                       # テスト設定
│   └── expected_results/              # 期待結果
├── utils/                             # テストユーティリティ
│   ├── test_helpers.py                # テストヘルパー
│   ├── mock_data.py                   # モックデータ
│   └── assertions.py                  # カスタムアサーション
├── conftest.py                        # pytest設定
└── pytest.ini                        # pytest設定ファイル
```

## 7. ドキュメント構成 (docs/)

```
docs/
├── architecture/                      # アーキテクチャ文書
│   ├── system_design.md               # システム設計書
│   ├── component_design.md            # コンポーネント設計
│   ├── database_design.md             # データベース設計
│   └── security_design.md             # セキュリティ設計
├── specifications/                    # 仕様書
│   ├── technical_requirements.md      # 技術要件仕様
│   ├── functional_requirements.md     # 機能要件仕様
│   ├── api_specification.md           # API仕様書
│   └── directory_structure.md         # このファイル
├── planning/                          # 計画文書
│   ├── implementation_plan.md         # 実装計画書
│   ├── risk_assessment.md             # リスク評価
│   ├── project_timeline.md            # プロジェクトタイムライン
│   └── resource_plan.md               # リソース計画
├── tutorials/                         # チュートリアル
│   ├── getting_started.md             # はじめに
│   ├── installation_guide.md          # インストールガイド
│   ├── user_guide.md                  # ユーザーガイド
│   ├── developer_guide.md             # 開発者ガイド
│   └── troubleshooting.md             # トラブルシューティング
├── research/                          # 技術調査
│   ├── yolo_comparison.md             # YOLO比較調査
│   ├── live2d_integration.md          # Live2D統合調査
│   ├── performance_analysis.md        # パフォーマンス分析
│   └── related_works.md               # 関連研究
├── api/                               # API文書
│   ├── openapi.yaml                   # OpenAPI仕様
│   ├── endpoints.md                   # エンドポイント一覧
│   ├── authentication.md              # 認証方法
│   └── examples.md                    # API使用例
├── deployment/                        # デプロイメント文書
│   ├── docker_deployment.md           # Docker導入
│   ├── kubernetes_deployment.md       # Kubernetes導入
│   ├── cloud_deployment.md            # クラウド導入
│   └── monitoring_setup.md            # 監視設定
├── development/                       # 開発文書
│   ├── coding_standards.md            # コーディング規約
│   ├── testing_guidelines.md          # テストガイドライン
│   ├── contributing.md                # 貢献ガイド
│   └── release_process.md             # リリースプロセス
└── assets/                            # 文書用アセット
    ├── images/                        # 図表画像
    ├── diagrams/                      # 図表ソース
    └── screenshots/                   # スクリーンショット
```

## 8. スクリプト構成 (scripts/)

```
scripts/
├── setup/                             # セットアップスクリプト
│   ├── install_dependencies.py        # 依存関係インストール
│   ├── setup_environment.py           # 環境セットアップ
│   ├── download_models.py             # モデルダウンロード
│   └── initialize_database.py         # データベース初期化
├── data/                              # データ処理スクリプト
│   ├── collect_dataset.py             # データセット収集
│   ├── preprocess_data.py             # データ前処理
│   ├── validate_annotations.py        # アノテーション検証
│   └── split_dataset.py               # データセット分割
├── training/                          # 学習スクリプト
│   ├── train_model.py                 # モデル学習
│   ├── evaluate_model.py              # モデル評価
│   ├── hyperparameter_tuning.py       # ハイパーパラメータ調整
│   └── export_model.py                # モデルエクスポート
├── deployment/                        # デプロイスクリプト
│   ├── build_docker.py                # Docker ビルド
│   ├── deploy_to_cloud.py             # クラウドデプロイ
│   ├── update_models.py               # モデル更新
│   └── backup_data.py                 # データバックアップ
├── maintenance/                       # 保守スクリプト
│   ├── cleanup_temp_files.py          # 一時ファイル削除
│   ├── check_system_health.py         # システムヘルスチェック
│   ├── update_dependencies.py         # 依存関係更新
│   └── generate_reports.py            # レポート生成
├── utilities/                         # ユーティリティスクリプト
│   ├── convert_formats.py             # フォーマット変換
│   ├── batch_process.py               # バッチ処理
│   ├── benchmark.py                   # ベンチマーク実行
│   └── quality_check.py               # 品質チェック
└── monitoring/                        # 監視スクリプト
    ├── system_monitor.py              # システム監視
    ├── model_performance_monitor.py   # モデル性能監視
    ├── log_analyzer.py                # ログ分析
    └── alert_handler.py               # アラート処理
```

## 9. 設定・メタファイル

### 9.1 依存関係管理 (requirements/)

```
requirements/
├── base.txt                           # 基本依存関係
├── dev.txt                            # 開発用依存関係
├── training.txt                       # 学習用依存関係
├── api.txt                            # API用依存関係
└── deployment.txt                     # デプロイ用依存関係
```

### 9.2 Docker設定 (docker/)

```
docker/
├── Dockerfile                         # メインDockerfile
├── Dockerfile.dev                     # 開発用Dockerfile
├── Dockerfile.training                # 学習用Dockerfile
├── docker-compose.yml                 # Docker Compose設定
├── docker-compose.dev.yml             # 開発用Compose
├── docker-compose.prod.yml            # 本番用Compose
└── scripts/                           # Docker用スクリプト
    ├── entrypoint.sh                  # エントリーポイント
    ├── health_check.sh                # ヘルスチェック
    └── init_container.sh              # コンテナ初期化
```

### 9.3 GitHub設定 (.github/)

```
.github/
├── workflows/                         # GitHub Actions
│   ├── ci.yml                         # 継続的インテグレーション
│   ├── cd.yml                         # 継続的デプロイメント
│   ├── test.yml                       # テスト実行
│   ├── code_quality.yml               # コード品質チェック
│   └── security.yml                   # セキュリティチェック
├── ISSUE_TEMPLATE/                    # Issue テンプレート
│   ├── bug_report.md                  # バグレポート
│   ├── feature_request.md             # 機能要求
│   └── custom.md                      # カスタムテンプレート
├── PULL_REQUEST_TEMPLATE.md           # プルリクエストテンプレート
├── CONTRIBUTING.md                    # 貢献ガイド
└── CODE_OF_CONDUCT.md                # 行動規範
```

## 10. ディレクトリ管理ルール

### 10.1 命名規則

#### ディレクトリ名
- **小文字＋アンダースコア**: `image_processor`, `batch_processing`
- **複数形**: 複数のファイルを含む場合 (`models/`, `tests/`, `docs/`)
- **単数形**: 単一機能の場合 (`config/`, `training/`)

#### ファイル名
- **Python**: `snake_case.py`
- **設定ファイル**: `kebab-case.yaml`, `snake_case.toml`
- **ドキュメント**: `snake_case.md`

### 10.2 ファイル配置原則

#### 責任の分離
- 各ディレクトリは明確な責任を持つ
- 関連する機能は同じディレクトリにまとめる
- 共通機能は`utils/`や`common/`に配置

#### 依存関係の管理
- 上位レイヤーは下位レイヤーに依存可能
- 下位レイヤーは上位レイヤーに依存不可
- 循環依存を避ける

#### 設定の外部化
- ハードコードを避け、設定ファイルを使用
- 環境別設定の分離
- デフォルト値の提供

### 10.3 権限・アクセス制御

#### ファイル権限
- **実行ファイル**: 755
- **設定ファイル**: 644
- **秘密情報**: 600

#### ディレクトリ権限
- **一般ディレクトリ**: 755
- **ログディレクトリ**: 755 (書き込み可能)
- **一時ディレクトリ**: 777 (必要に応じて)

### 10.4 クリーンアップルール

#### 自動削除対象
- `data/temp/`: 24時間後
- `logs/`: 30日後（圧縮保存）
- `data/output/`: 7日後（ユーザー指定除く）

#### 手動管理対象
- `models/`: 明示的削除のみ
- `data/training/`: 手動管理
- `docs/`: バージョン管理

## 11. 今後の拡張予定

### 11.1 追加予定ディレクトリ

```
# 将来的な拡張
├── plugins/                           # プラグインシステム
├── themes/                            # UIテーマ
├── localization/                      # 多言語対応
├── benchmarks/                        # ベンチマーク結果
└── examples/                          # 使用例・サンプル
```

### 11.2 外部統合予定

```
# 外部サービス統合
├── integrations/                      # 外部統合
│   ├── live2d_cubism/                # Cubism Editor連携
│   ├── cloud_storage/                # クラウドストレージ
│   └── analytics/                    # 分析サービス
```

---

このディレクトリ構成は、プロジェクトの成長と要件変化に応じて継続的に見直し・更新されます。