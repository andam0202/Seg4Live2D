"""
セグメンテーション実行エンジン

YOLOモデルと画像処理を統合したセグメンテーション実行システム
"""

import time
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.core.utils import get_logger
from src.core.utils.exceptions import ProcessingError, ModelError
from .yolo_model import get_model_manager, YOLOModelManager
from .image_processor import get_image_processor, ImageProcessor, ProcessingResult

logger = get_logger(__name__)


@dataclass
class SegmentationMask:
    """セグメンテーションマスク情報"""
    mask: np.ndarray  # マスク配列（0-1 or 0-255）
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    area: int


@dataclass  
class SegmentationResult:
    """セグメンテーション結果"""
    image_path: Optional[str]
    original_size: Tuple[int, int]  # (width, height)
    masks: List[SegmentationMask]
    processing_time: float
    model_info: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class SegmentationEngine:
    """セグメンテーション実行エンジン"""
    
    # COCO-80クラス名（YOLOv11デフォルト）
    COCO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
        35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
        39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
        44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
        49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
        54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
        59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
        64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
        69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
        74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
        79: 'toothbrush'
    }
    
    def __init__(self) -> None:
        """セグメンテーションエンジンの初期化"""
        self.model_manager: Optional[YOLOModelManager] = None
        self.image_processor: Optional[ImageProcessor] = None
        self._initialized = False
        
        logger.info("セグメンテーションエンジン初期化")
    
    def initialize(self) -> None:
        """エンジンの初期化"""
        
        try:
            # モデル管理とプロセッサの初期化
            self.model_manager = get_model_manager()
            self.image_processor = get_image_processor()
            
            # YOLOモデルの読み込み
            if not self.model_manager.is_loaded():
                logger.info("YOLOモデルを読み込み中...")
                self.model_manager.load_model()
            
            self._initialized = True
            logger.info("セグメンテーションエンジン初期化完了")
            
        except Exception as e:
            raise ModelError(f"セグメンテーションエンジン初期化に失敗: {e}") from e
    
    def process_image(
        self,
        image: Union[str, Path, np.ndarray],
        confidence: Optional[float] = None,
        save_masks: bool = False,
        output_dir: Optional[Path] = None
    ) -> SegmentationResult:
        """
        単一画像のセグメンテーション処理
        
        Args:
            image: 入力画像
            confidence: 信頼度閾値
            save_masks: マスク保存フラグ
            output_dir: 出力ディレクトリ
            
        Returns:
            セグメンテーション結果
        """
        
        start_time = time.time()
        image_path_str = str(image) if isinstance(image, (str, Path)) else None
        
        try:
            # 初期化確認
            if not self._initialized:
                self.initialize()
            
            logger.info(f"セグメンテーション開始: {image_path_str}")
            
            # 画像バリデーション
            validation_result = self.image_processor.validate_image_for_processing(image)
            if not validation_result["valid"]:
                raise ProcessingError(
                    f"画像バリデーションエラー: {validation_result['errors']}",
                    stage="validation",
                    image_path=image_path_str
                )
            
            # 警告ログ
            for warning in validation_result["warnings"]:
                logger.warning(warning)
            
            # 画像前処理
            processing_result = self.image_processor.preprocess_for_yolo(image)
            logger.debug(f"前処理完了: {processing_result.processing_time:.3f}s")
            
            # YOLO推論実行
            yolo_results = self.model_manager.predict(
                processing_result.processed_image,
                confidence=confidence
            )
            
            # 結果の後処理
            masks = self._process_yolo_results(yolo_results, processing_result)
            
            # マスク保存（オプション）
            if save_masks and output_dir:
                self._save_masks(masks, output_dir, image_path_str)
            
            processing_time = time.time() - start_time
            
            # モデル情報取得
            model_info = {}
            if self.model_manager.model_info:
                model_info = {
                    "name": self.model_manager.model_info.name,
                    "device": self.model_manager.model_info.device,
                    "input_size": self.model_manager.model_info.input_size
                }
            
            result = SegmentationResult(
                image_path=image_path_str,
                original_size=processing_result.original_size,
                masks=masks,
                processing_time=processing_time,
                model_info=model_info,
                success=True
            )
            
            logger.info(f"セグメンテーション完了: {len(masks)}マスク, {processing_time:.3f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"セグメンテーション処理エラー: {e}"
            logger.error(error_msg)
            
            return SegmentationResult(
                image_path=image_path_str,
                original_size=(0, 0),
                masks=[],
                processing_time=processing_time,
                model_info={},
                success=False,
                error_message=error_msg
            )
    
    def _process_yolo_results(
        self,
        yolo_results: Any,
        processing_result: ProcessingResult
    ) -> List[SegmentationMask]:
        """YOLO結果の後処理"""
        
        masks = []
        
        try:
            # YOLOv11の結果処理
            for result in yolo_results:
                if hasattr(result, 'masks') and result.masks is not None:
                    # マスクデータの取得
                    masks_data = result.masks.data.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for i, (mask_data, box, class_id, conf) in enumerate(
                        zip(masks_data, boxes, classes, confidences)
                    ):
                        # マスクの後処理（元画像サイズに変換）
                        processed_mask = self.image_processor.postprocess_mask(
                            mask_data, processing_result
                        )
                        
                        # バウンディングボックスの後処理
                        bbox_coords = np.array([box])
                        processed_bbox = self.image_processor.postprocess_coordinates(
                            bbox_coords, processing_result
                        )[0]
                        
                        # クラス名取得
                        class_name = self.COCO_CLASSES.get(class_id, f"class_{class_id}")
                        
                        # マスク面積計算
                        mask_binary = (processed_mask > 0.5).astype(np.uint8)
                        area = np.sum(mask_binary)
                        
                        # セグメンテーションマスク作成
                        seg_mask = SegmentationMask(
                            mask=processed_mask,
                            class_id=class_id,
                            class_name=class_name,
                            confidence=float(conf),
                            bbox=tuple(processed_bbox.astype(int)),
                            area=int(area)
                        )
                        
                        masks.append(seg_mask)
                        
                        logger.debug(f"マスク処理完了: {class_name}, conf={conf:.3f}, area={area}")
                
            return masks
            
        except Exception as e:
            raise ProcessingError(
                f"YOLO結果後処理中にエラー: {e}",
                stage="yolo_postprocess"
            ) from e
    
    def _save_masks(
        self,
        masks: List[SegmentationMask],
        output_dir: Path,
        image_path: Optional[str]
    ) -> None:
        """マスクファイルの保存"""
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # ベースファイル名
            if image_path:
                base_name = Path(image_path).stem
            else:
                base_name = f"image_{int(time.time())}"
            
            for i, mask in enumerate(masks):
                # マスクファイル名
                mask_filename = f"{base_name}_mask_{i:03d}_{mask.class_name}.png"
                mask_path = output_dir / mask_filename
                
                # マスクを0-255に変換
                mask_uint8 = (mask.mask * 255).astype(np.uint8)
                
                # 保存
                import cv2
                cv2.imwrite(str(mask_path), mask_uint8)
                
                logger.debug(f"マスク保存: {mask_path}")
                
        except Exception as e:
            logger.warning(f"マスク保存中にエラー: {e}")
    
    def get_class_names(self) -> Dict[int, str]:
        """クラス名一覧取得"""
        return self.COCO_CLASSES.copy()
    
    def is_initialized(self) -> bool:
        """初期化状態確認"""
        return self._initialized
    
    def cleanup(self) -> None:
        """リソースクリーンアップ"""
        
        try:
            if self.model_manager:
                self.model_manager.unload_model()
            
            self._initialized = False
            logger.info("セグメンテーションエンジンクリーンアップ完了")
            
        except Exception as e:
            logger.warning(f"クリーンアップ中にエラー: {e}")


# グローバルエンジンインスタンス
_segmentation_engine: Optional[SegmentationEngine] = None


def get_segmentation_engine() -> SegmentationEngine:
    """グローバルセグメンテーションエンジン取得"""
    global _segmentation_engine
    
    if _segmentation_engine is None:
        _segmentation_engine = SegmentationEngine()
    
    return _segmentation_engine