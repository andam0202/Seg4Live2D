"""
セグメンテーションパイプライン統合テスト

ユーザー提供画像でYOLOv11セグメンテーションの統合テストを実行
"""

import pytest
import time
from pathlib import Path

from src.core.utils import get_logger, load_config
from src.core.segmentation import get_segmentation_engine
from src import OUTPUT_DIR

logger = get_logger(__name__)


class TestSegmentationPipeline:
    """セグメンテーションパイプライン統合テスト"""
    
    @pytest.fixture(autouse=True)
    def setup(self, sample_images_path):
        """テスト前セットアップ"""
        self.sample_images_path = sample_images_path
        self.output_dir = OUTPUT_DIR / "test_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定読み込み
        self.config = load_config()
        
        # エンジン初期化
        self.engine = get_segmentation_engine()
        
    def test_sample_images_exist(self):
        """サンプル画像の存在確認"""
        assert self.sample_images_path.exists(), f"サンプル画像ディレクトリが見つかりません: {self.sample_images_path}"
        
        image_files = list(self.sample_images_path.glob("*.png")) + list(self.sample_images_path.glob("*.jpg"))
        assert len(image_files) > 0, f"画像ファイルが見つかりません: {self.sample_images_path}"
        
        logger.info(f"テスト対象画像: {len(image_files)}枚")
        for img_file in image_files:
            logger.info(f"  - {img_file.name}")
    
    def test_engine_initialization(self):
        """エンジン初期化テスト"""
        logger.info("セグメンテーションエンジン初期化テスト")
        
        # 初期化前は未初期化
        assert not self.engine.is_initialized()
        
        # 初期化実行
        self.engine.initialize()
        
        # 初期化後は初期化済み
        assert self.engine.is_initialized()
        
        logger.info("✅ エンジン初期化成功")
    
    def test_single_image_segmentation(self):
        """単一画像セグメンテーションテスト"""
        logger.info("単一画像セグメンテーションテスト")
        
        # サンプル画像取得
        image_files = list(self.sample_images_path.glob("*.png"))
        if not image_files:
            pytest.skip("テスト用画像が見つかりません")
        
        test_image = image_files[0]
        logger.info(f"テスト画像: {test_image.name}")
        
        # エンジン初期化
        if not self.engine.is_initialized():
            self.engine.initialize()
        
        # セグメンテーション実行
        start_time = time.time()
        result = self.engine.process_image(
            image=test_image,
            save_masks=True,
            output_dir=self.output_dir
        )
        processing_time = time.time() - start_time
        
        # 結果検証
        assert result.success, f"セグメンテーション失敗: {result.error_message}"
        assert result.processing_time > 0
        assert len(result.original_size) == 2
        
        logger.info(f"✅ セグメンテーション成功")
        logger.info(f"   処理時間: {processing_time:.3f}s")
        logger.info(f"   検出マスク数: {len(result.masks)}")
        logger.info(f"   元画像サイズ: {result.original_size}")
        
        # 検出されたクラス情報
        for i, mask in enumerate(result.masks[:5]):  # 最初の5つを表示
            logger.info(f"   マスク{i}: {mask.class_name} (confidence={mask.confidence:.3f})")
        
        return result
    
    def test_multiple_images_segmentation(self):
        """複数画像セグメンテーションテスト"""
        logger.info("複数画像セグメンテーションテスト")
        
        # サンプル画像取得（最大3枚）
        image_files = list(self.sample_images_path.glob("*.png"))[:3]
        if len(image_files) == 0:
            pytest.skip("テスト用画像が見つかりません")
        
        logger.info(f"テスト対象: {len(image_files)}枚")
        
        # エンジン初期化
        if not self.engine.is_initialized():
            self.engine.initialize()
        
        results = []
        total_processing_time = 0.0
        successful_count = 0
        
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"--- 画像 {i}/{len(image_files)}: {image_file.name} ---")
            
            try:
                result = self.engine.process_image(
                    image=image_file,
                    save_masks=True,
                    output_dir=self.output_dir / f"test_{i}"
                )
                
                results.append(result)
                
                if result.success:
                    successful_count += 1
                    total_processing_time += result.processing_time
                    
                    logger.info(f"✅ 成功: {len(result.masks)}マスク, {result.processing_time:.3f}s")
                else:
                    logger.error(f"❌ 失敗: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"❌ 例外エラー: {e}")
        
        # 結果サマリー
        logger.info(f"\n=== テスト結果サマリー ===")
        logger.info(f"成功: {successful_count}/{len(image_files)}")
        
        if successful_count > 0:
            avg_time = total_processing_time / successful_count
            logger.info(f"平均処理時間: {avg_time:.3f}s")
        
        # 最低1枚は成功することを確認
        assert successful_count > 0, "すべての画像でセグメンテーションが失敗しました"
        
        return results
    
    def test_live2d_applicability_evaluation(self):
        """Live2D用途適性評価テスト"""
        logger.info("Live2D用途適性評価テスト")
        
        # 1枚目の画像でテスト
        image_files = list(self.sample_images_path.glob("*.png"))
        if not image_files:
            pytest.skip("テスト用画像が見つかりません")
        
        # エンジン初期化
        if not self.engine.is_initialized():
            self.engine.initialize()
        
        result = self.engine.process_image(image=image_files[0])
        
        if not result.success:
            pytest.skip(f"セグメンテーション失敗: {result.error_message}")
        
        # Live2D関連クラスの検出確認
        person_masks = [mask for mask in result.masks if mask.class_name == 'person']
        
        logger.info(f"=== Live2D用途評価 ===")
        logger.info(f"検出された'person'クラス: {len(person_masks)}個")
        
        for i, mask in enumerate(person_masks):
            logger.info(f"  Person {i+1}: confidence={mask.confidence:.3f}, area={mask.area}")
        
        # 評価基準
        evaluation = {
            "person_detected": len(person_masks) > 0,
            "high_confidence": any(mask.confidence > 0.7 for mask in person_masks),
            "sufficient_area": any(mask.area > 1000 for mask in person_masks),
        }
        
        logger.info(f"評価結果:")
        logger.info(f"  人物検出: {'✅' if evaluation['person_detected'] else '❌'}")
        logger.info(f"  高信頼度: {'✅' if evaluation['high_confidence'] else '❌'}")
        logger.info(f"  十分な面積: {'✅' if evaluation['sufficient_area'] else '❌'}")
        
        # 改善提案
        if not evaluation["person_detected"]:
            logger.info("💡 改善提案: Live2D特化学習が必要")
        elif not evaluation["high_confidence"]:
            logger.info("💡 改善提案: 信頼度向上のためのファインチューニング検討")
        elif not evaluation["sufficient_area"]:
            logger.info("💡 改善提案: より大きな人物画像での検証推奨")
        else:
            logger.info("✅ 事前学習モデルでも十分な品質")
        
        return evaluation
    
    def teardown_method(self):
        """テスト後クリーンアップ"""
        if hasattr(self, 'engine') and self.engine.is_initialized():
            self.engine.cleanup()


if __name__ == "__main__":
    # 直接実行時のテスト
    import sys
    from pathlib import Path
    
    # プロジェクトルートを追加
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # ログ設定
    from src.core.utils import setup_logging
    setup_logging(level="INFO", console_output=True, structured=False)
    
    # テスト実行
    tester = TestSegmentationPipeline()
    tester.setup(project_root / "data" / "samples" / "demo_images")
    
    try:
        logger.info("=== Seg4Live2D セグメンテーション統合テスト ===")
        tester.test_sample_images_exist()
        tester.test_engine_initialization()
        tester.test_single_image_segmentation()
        tester.test_multiple_images_segmentation()
        tester.test_live2d_applicability_evaluation()
        logger.info("=== 全テスト完了 ===")
    finally:
        tester.teardown_method()