#!/usr/bin/env python3
"""
高精度セグメンテーションテスト

より大きなYOLOモデルでの詳細分割テスト
yolo11n → yolo11l での性能比較
"""

import sys
import time
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils import setup_logging, get_logger

def compare_models_on_sample(image_path, output_dir):
    """複数モデルでの比較テスト"""
    from src.core.segmentation import get_segmentation_engine
    from src.core.utils import load_config
    
    logger = get_logger(__name__)
    output_dir = Path(output_dir)
    
    # テスト対象モデル（軽量→高精度順）
    models_to_test = [
        {"name": "yolo11n-seg.pt", "desc": "Nano (軽量・高速)"},
        {"name": "yolo11s-seg.pt", "desc": "Small (バランス)"},
        {"name": "yolo11m-seg.pt", "desc": "Medium (中精度)"},
        {"name": "yolo11l-seg.pt", "desc": "Large (高精度)"},
    ]
    
    results = {}
    
    for model_info in models_to_test:
        model_name = model_info["name"]
        model_desc = model_info["desc"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"モデルテスト: {model_name} - {model_desc}")
        logger.info(f"{'='*60}")
        
        try:
            # モデル名を直接指定してYOLOModelManagerを作成
            from src.core.segmentation.yolo_model import YOLOModelManager
            from src.core.segmentation.image_processor import ImageProcessor
            from src.core.utils import get_config
            
            # 設定をロード
            config = get_config()
            
            # モデル名を直接指定してモデルマネージャーを作成
            model_manager = YOLOModelManager(model_name=model_name)
            image_processor = ImageProcessor()
            
            # 簡易エンジンクラス
            class SimpleEngine:
                def __init__(self, model_manager, image_processor, config):
                    self.model_manager = model_manager
                    self.image_processor = image_processor
                    self.config = config
                    self.initialized = False
                
                def initialize(self):
                    self.model_manager.load_model()
                    self.initialized = True
                
                def process_image(self, image, save_masks=False, output_dir=None):
                    from src.core.segmentation.segmentation_engine import SegmentationResult, SegmentationMask
                    import time
                    
                    try:
                        start_time = time.time()
                        
                        # 画像読み込み
                        processing_result = self.image_processor.preprocess_for_yolo(image)
                        processed_image = processing_result.processed_image
                        original_size = processing_result.original_size
                        
                        # YOLO推論（低い閾値で試行）
                        yolo_results = self.model_manager.predict(
                            processed_image,
                            confidence=0.05,  # より低い閾値
                            iou_threshold=self.config.model.iou_threshold
                        )
                        
                        # 結果処理
                        masks = []
                        if yolo_results and len(yolo_results) > 0:
                            result = yolo_results[0]
                            if hasattr(result, 'masks') and result.masks is not None:
                                for i, (box, mask, conf, cls) in enumerate(zip(
                                    result.boxes.xyxy.cpu().numpy(),
                                    result.masks.data.cpu().numpy(),
                                    result.boxes.conf.cpu().numpy(),
                                    result.boxes.cls.cpu().numpy()
                                )):
                                    class_name = result.names[int(cls)]
                                    
                                    # マスクの後処理
                                    mask_resized = self.image_processor.postprocess_mask(
                                        mask, processing_result
                                    )
                                    
                                    mask_obj = SegmentationMask(
                                        mask=mask_resized,
                                        class_id=int(cls),
                                        class_name=class_name,
                                        confidence=float(conf),
                                        bbox=tuple(box.astype(int).tolist()),
                                        area=int(mask_resized.sum())
                                    )
                                    masks.append(mask_obj)
                        
                        processing_time = time.time() - start_time
                        
                        return SegmentationResult(
                            image_path=str(image),
                            original_size=original_size,
                            masks=masks,
                            processing_time=processing_time,
                            model_info={"name": self.model_manager.model_name, "device": self.model_manager.device},
                            success=True,
                            error_message=None
                        )
                    
                    except Exception as e:
                        return SegmentationResult(
                            image_path=str(image),
                            original_size=(0, 0),
                            masks=[],
                            processing_time=0,
                            model_info={"name": self.model_manager.model_name, "device": self.model_manager.device},
                            success=False,
                            error_message=str(e)
                        )
                
                def cleanup(self):
                    if hasattr(self.model_manager, 'unload_model'):
                        self.model_manager.unload_model()
            
            engine = SimpleEngine(model_manager, image_processor, config)
            
            logger.info(f"モデル初期化中: {model_name}")
            start_init = time.time()
            engine.initialize()
            init_time = time.time() - start_init
            
            # セグメンテーション実行
            logger.info(f"セグメンテーション実行中...")
            start_process = time.time()
            result = engine.process_image(
                image=image_path,
                save_masks=True,
                output_dir=output_dir / model_name.replace('.pt', '')
            )
            process_time = time.time() - start_process
            
            if result.success:
                # 結果分析
                mask_areas = [mask.area for mask in result.masks]
                high_conf_masks = [m for m in result.masks if m.confidence > 0.3]
                person_masks = [m for m in result.masks if m.class_name == 'person']
                
                results[model_name] = {
                    "success": True,
                    "init_time": init_time,
                    "process_time": process_time,
                    "total_masks": len(result.masks),
                    "high_confidence_masks": len(high_conf_masks),
                    "person_detections": len(person_masks),
                    "total_area": sum(mask_areas),
                    "avg_confidence": sum(m.confidence for m in result.masks) / len(result.masks) if result.masks else 0,
                    "max_confidence": max((m.confidence for m in result.masks), default=0),
                    "masks_detail": [
                        {
                            "class": m.class_name,
                            "conf": round(m.confidence, 3),
                            "area": m.area,
                            "bbox": m.bbox
                        } for m in result.masks
                    ]
                }
                
                logger.info(f"✅ 成功")
                logger.info(f"   初期化時間: {init_time:.2f}秒")
                logger.info(f"   処理時間: {process_time:.2f}秒")
                logger.info(f"   検出マスク数: {len(result.masks)}")
                logger.info(f"   高信頼度(>0.3): {len(high_conf_masks)}")
                logger.info(f"   person検出: {len(person_masks)}")
                logger.info(f"   平均信頼度: {results[model_name]['avg_confidence']:.3f}")
                logger.info(f"   最高信頼度: {results[model_name]['max_confidence']:.3f}")
                
                # 詳細マスク情報
                for i, mask in enumerate(result.masks[:5]):  # 上位5つ
                    logger.info(f"   マスク{i+1}: {mask.class_name} (conf={mask.confidence:.3f}, area={mask.area})")
                
            else:
                logger.error(f"❌ 失敗: {result.error_message}")
                results[model_name] = {
                    "success": False,
                    "error": result.error_message,
                    "init_time": init_time,
                    "process_time": process_time
                }
            
            # クリーンアップ
            engine.cleanup()
            
        except Exception as e:
            logger.error(f"❌ エラー: {e}")
            results[model_name] = {
                "success": False,
                "error": str(e),
                "init_time": 0,
                "process_time": 0
            }
    
    return results

def analyze_model_comparison(results):
    """モデル比較結果の分析"""
    logger = get_logger(__name__)
    
    logger.info(f"\n{'='*80}")
    logger.info("=== モデル比較分析 ===")
    logger.info(f"{'='*80}")
    
    successful_results = {k: v for k, v in results.items() if v.get("success", False)}
    
    if not successful_results:
        logger.error("すべてのモデルで失敗しました")
        return
    
    # 比較表作成
    logger.info(f"\n{'モデル':<20} {'マスク数':<8} {'高信頼度':<8} {'person':<6} {'平均信頼度':<10} {'処理時間':<8}")
    logger.info("-" * 80)
    
    best_model = None
    best_score = 0
    
    for model_name, result in successful_results.items():
        model_short = model_name.replace('yolo11', '').replace('-seg.pt', '').upper()
        
        # 総合スコア計算
        score = (
            result["total_masks"] * 1.0 +
            result["high_confidence_masks"] * 2.0 +
            result["person_detections"] * 3.0 +
            result["avg_confidence"] * 10.0
        )
        
        if score > best_score:
            best_score = score
            best_model = model_name
        
        logger.info(
            f"{model_short:<20} "
            f"{result['total_masks']:<8} "
            f"{result['high_confidence_masks']:<8} "
            f"{result['person_detections']:<6} "
            f"{result['avg_confidence']:<10.3f} "
            f"{result['process_time']:<8.2f}s"
        )
    
    # 推奨モデル
    logger.info(f"\n🏆 推奨モデル: {best_model}")
    logger.info(f"   総合スコア: {best_score:.1f}")
    
    if best_model in successful_results:
        best_result = successful_results[best_model]
        logger.info(f"   検出マスク数: {best_result['total_masks']}")
        logger.info(f"   高信頼度マスク: {best_result['high_confidence_masks']}")
        logger.info(f"   person検出: {best_result['person_detections']}")
        logger.info(f"   処理時間: {best_result['process_time']:.2f}秒")
    
    # Live2D用途での推奨事項
    logger.info(f"\n💡 Live2D用途での推奨:")
    
    best_result = successful_results[best_model]
    if best_result["person_detections"] > 0 and best_result["total_masks"] >= 3:
        logger.info("- ✅ Live2D用途に適用可能")
        logger.info("- より詳細な分割のため、以下を検討:")
        logger.info("  • confidence閾値をさらに下げる (0.1-0.05)")
        logger.info("  • SAM (Segment Anything Model) との併用")
        logger.info("  • 後処理での境界調整")
    elif best_result["total_masks"] >= 2:
        logger.info("- 🔧 基本的な分割は可能、改善余地あり")
        logger.info("- より高精度なモデル (yolo11x-seg.pt) を試行")
        logger.info("- 画像前処理（リサイズ、コントラスト調整）を検討")
    else:
        logger.info("- ❌ 現在の手法では分割困難")
        logger.info("- 別のアプローチ (SAM, 手動アノテーション) を検討")

def main():
    """メイン実行"""
    
    # ログ設定
    setup_logging(level="INFO", console_output=True, structured=False)
    logger = get_logger(__name__)
    
    logger.info("=== 高精度セグメンテーション比較テスト ===")
    
    try:
        # サンプル画像取得（最初の1枚で詳細テスト）
        sample_images_path = project_root / "data" / "samples" / "demo_images"
        image_files = list(sample_images_path.glob("*.png"))
        
        if not image_files:
            logger.error("サンプル画像が見つかりません")
            return
        
        test_image = image_files[0]  # 最初の画像で詳細テスト
        logger.info(f"テスト画像: {test_image.name}")
        
        output_dir = project_root / "data" / "output" / "model_comparison"
        
        # モデル比較実行
        results = compare_models_on_sample(test_image, output_dir)
        
        # 結果分析
        analyze_model_comparison(results)
        
        logger.info(f"\n詳細結果は以下に保存されました:")
        logger.info(f"{output_dir}")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()