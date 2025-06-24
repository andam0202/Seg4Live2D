#!/usr/bin/env python3
"""
セグメンテーション機能の簡易テストスクリプト

ユーザー提供画像での動作確認用
"""

import sys
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils import setup_logging, get_logger

def main():
    """メイン実行"""
    
    # ログ設定
    setup_logging(level="INFO", console_output=True, structured=False)
    logger = get_logger(__name__)
    
    logger.info("=== Seg4Live2D セグメンテーション簡易テスト ===")
    
    try:
        # 統合テストを実行
        from tests.integration.test_segmentation_pipeline import TestSegmentationPipeline
        
        # テスト実行
        tester = TestSegmentationPipeline()
        tester.sample_images_path = project_root / "data" / "samples" / "demo_images"
        tester.output_dir = project_root / "data" / "output" / "test_results"
        tester.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定読み込み
        from src.core.utils import load_config
        tester.config = load_config()
        
        # エンジン初期化
        from src.core.segmentation import get_segmentation_engine
        tester.engine = get_segmentation_engine()
        
        logger.info("1. サンプル画像確認...")
        tester.test_sample_images_exist()
        
        logger.info("2. エンジン初期化...")
        tester.test_engine_initialization()
        
        logger.info("3. 単一画像テスト...")
        tester.test_single_image_segmentation()
        
        logger.info("4. 複数画像テスト...")
        tester.test_multiple_images_segmentation()
        
        logger.info("5. Live2D適性評価...")
        evaluation = tester.test_live2d_applicability_evaluation()
        
        # 最終評価
        logger.info("\n=== 最終評価 ===")
        if all(evaluation.values()):
            logger.info("🎉 優秀: 事前学習モデルでも十分なLive2D用途適性")
        elif evaluation["person_detected"]:
            logger.info("👍 良好: 基本的な人物検出は可能、カスタマイズで改善可能")
        else:
            logger.info("🔧 要改善: Live2D特化学習が必要")
        
        logger.info("=== テスト完了 ===")
        
    except KeyboardInterrupt:
        logger.info("ユーザーによって中断されました")
    except Exception as e:
        logger.error(f"テスト実行中にエラー: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # クリーンアップ
        try:
            tester.teardown_method()
        except:
            pass

if __name__ == "__main__":
    main()