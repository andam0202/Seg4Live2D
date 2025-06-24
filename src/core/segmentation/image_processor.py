"""
画像前処理システム

YOLOセグメンテーション用の画像前処理機能を提供します。
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image, ImageOps

from src.core.utils import get_logger, get_config, file_handler
from src.core.utils.exceptions import ProcessingError, ValidationError

logger = get_logger(__name__)


@dataclass
class ProcessingResult:
    """前処理結果クラス"""
    processed_image: np.ndarray
    original_size: Tuple[int, int]  # (width, height)
    processed_size: Tuple[int, int]  # (width, height)
    scale_factor: float
    padding: Tuple[int, int, int, int]  # (top, bottom, left, right)
    format: str
    processing_time: float


class ImageProcessor:
    """画像前処理クラス"""
    
    def __init__(self) -> None:
        """画像前処理クラスの初期化"""
        self.config = get_config()
        self.target_size = self.config.processing.input_size
        
        logger.info(f"画像前処理クラス初期化: target_size={self.target_size}")
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        画像ファイルの読み込み
        
        Args:
            image_path: 画像ファイルパス
            
        Returns:
            BGR形式の画像配列
        """
        
        image_path = Path(image_path)
        
        try:
            # ファイルバリデーション
            file_handler.validate_image_file(image_path)
            
            # PIL経由での安全な読み込み
            with Image.open(image_path) as pil_image:
                # RGBA -> RGB変換（必要に応じて）
                if pil_image.mode in ('RGBA', 'LA', 'P'):
                    # 透明度のある画像は白背景に合成
                    background = Image.new('RGB', pil_image.size, (255, 255, 255))
                    if pil_image.mode == 'P':
                        pil_image = pil_image.convert('RGBA')
                    background.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode in ('RGBA', 'LA') else None)
                    pil_image = background
                elif pil_image.mode == 'L':
                    # グレースケール -> RGB
                    pil_image = pil_image.convert('RGB')
                elif pil_image.mode != 'RGB':
                    # その他 -> RGB
                    pil_image = pil_image.convert('RGB')
                
                # PIL -> OpenCV (RGB -> BGR)
                image_array = np.array(pil_image)
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            logger.debug(f"画像読み込み完了: {image_path}, shape: {image_bgr.shape}")
            return image_bgr
            
        except Exception as e:
            raise ProcessingError(
                f"画像読み込みに失敗しました: {e}",
                stage="load_image",
                image_path=str(image_path)
            ) from e
    
    def preprocess_for_yolo(
        self,
        image: Union[str, Path, np.ndarray],
        target_size: Optional[int] = None
    ) -> ProcessingResult:
        """
        YOLO用画像前処理
        
        Args:
            image: 入力画像（パスまたは配列）
            target_size: ターゲットサイズ（None の場合は設定値使用）
            
        Returns:
            前処理結果
        """
        
        import time
        start_time = time.time()
        
        try:
            # 画像の読み込み
            if isinstance(image, (str, Path)):
                image_array = self.load_image(image)
            else:
                image_array = image.copy()
            
            original_height, original_width = image_array.shape[:2]
            original_size = (original_width, original_height)
            
            # ターゲットサイズの決定
            if target_size is None:
                target_size = self.target_size
            
            # リサイズ処理（アスペクト比保持）
            processed_image, scale_factor, padding = self._resize_with_padding(
                image_array, target_size
            )
            
            processed_height, processed_width = processed_image.shape[:2]
            processed_size = (processed_width, processed_height)
            
            # 正規化（YOLOv11は内部で実行されるためここでは不要）
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                processed_image=processed_image,
                original_size=original_size,
                processed_size=processed_size,
                scale_factor=scale_factor,
                padding=padding,
                format="BGR",
                processing_time=processing_time
            )
            
            logger.debug(f"YOLO前処理完了: {original_size} -> {processed_size}, time: {processing_time:.3f}s")
            return result
            
        except Exception as e:
            raise ProcessingError(
                f"YOLO前処理中にエラーが発生しました: {e}",
                stage="preprocess_for_yolo"
            ) from e
    
    def _resize_with_padding(
        self,
        image: np.ndarray,
        target_size: int
    ) -> Tuple[np.ndarray, float, Tuple[int, int, int, int]]:
        """
        アスペクト比を保持したリサイズ（パディング付き）
        
        Args:
            image: 入力画像
            target_size: ターゲットサイズ（正方形）
            
        Returns:
            (リサイズ画像, スケールファクター, パディング)
        """
        
        height, width = image.shape[:2]
        
        # スケールファクターの計算
        scale = min(target_size / width, target_size / height)
        
        # 新しいサイズ
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # リサイズ
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # パディングサイズの計算
        pad_x = target_size - new_width
        pad_y = target_size - new_height
        
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        
        # パディング適用（グレー背景）
        padded = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)  # YOLOデフォルトのパディング色
        )
        
        padding = (pad_top, pad_bottom, pad_left, pad_right)
        
        return padded, scale, padding
    
    def postprocess_coordinates(
        self,
        coordinates: np.ndarray,
        processing_result: ProcessingResult
    ) -> np.ndarray:
        """
        座標の後処理（YOLO出力 -> 元画像座標）
        
        Args:
            coordinates: YOLO出力座標 (N, 4) [x1, y1, x2, y2] or (N, 2) [x, y]
            processing_result: 前処理結果
            
        Returns:
            元画像座標系に変換された座標
        """
        
        try:
            coords = coordinates.copy()
            scale = processing_result.scale_factor
            pad_top, pad_bottom, pad_left, pad_right = processing_result.padding
            
            # パディング除去
            if coords.ndim == 2 and coords.shape[1] >= 2:
                coords[:, 0] -= pad_left  # x座標
                coords[:, 1] -= pad_top   # y座標
                
                if coords.shape[1] >= 4:  # バウンディングボックス
                    coords[:, 2] -= pad_left  # x2座標
                    coords[:, 3] -= pad_top   # y2座標
            
            # スケール逆変換
            coords = coords / scale
            
            # 元画像範囲にクリップ
            original_width, original_height = processing_result.original_size
            if coords.ndim == 2 and coords.shape[1] >= 2:
                coords[:, 0] = np.clip(coords[:, 0], 0, original_width)
                coords[:, 1] = np.clip(coords[:, 1], 0, original_height)
                
                if coords.shape[1] >= 4:
                    coords[:, 2] = np.clip(coords[:, 2], 0, original_width)
                    coords[:, 3] = np.clip(coords[:, 3], 0, original_height)
            
            return coords
            
        except Exception as e:
            raise ProcessingError(
                f"座標後処理中にエラーが発生しました: {e}",
                stage="postprocess_coordinates"
            ) from e
    
    def postprocess_mask(
        self,
        mask: np.ndarray,
        processing_result: ProcessingResult
    ) -> np.ndarray:
        """
        マスクの後処理（YOLO出力 -> 元画像サイズ）
        
        Args:
            mask: YOLO出力マスク
            processing_result: 前処理結果
            
        Returns:
            元画像サイズに変換されたマスク
        """
        
        try:
            # パディング除去
            pad_top, pad_bottom, pad_left, pad_right = processing_result.padding
            target_size = processing_result.processed_size[0]  # 正方形
            
            # パディング部分を除去
            unpadded_height = target_size - pad_top - pad_bottom
            unpadded_width = target_size - pad_left - pad_right
            
            if mask.shape[-2:] != (target_size, target_size):
                # マスクサイズがターゲットサイズと異なる場合はリサイズ
                mask_resized = cv2.resize(
                    mask.astype(np.float32),
                    (target_size, target_size),
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                mask_resized = mask
            
            # パディング除去
            unpadded_mask = mask_resized[pad_top:pad_top+unpadded_height,
                                       pad_left:pad_left+unpadded_width]
            
            # 元画像サイズにリサイズ
            original_width, original_height = processing_result.original_size
            final_mask = cv2.resize(
                unpadded_mask,
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR
            )
            
            return final_mask
            
        except Exception as e:
            raise ProcessingError(
                f"マスク後処理中にエラーが発生しました: {e}",
                stage="postprocess_mask"
            ) from e
    
    def validate_image_for_processing(self, image: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """
        処理用画像の詳細バリデーション
        
        Args:
            image: 入力画像
            
        Returns:
            バリデーション結果と画像情報
        """
        
        try:
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                file_info = file_handler.get_file_info(image_path)
                
                # ファイルバリデーション
                file_handler.validate_image_file(image_path)
                
                # 画像情報取得
                image_array = self.load_image(image_path)
                
            else:
                image_array = image
                file_info = {
                    "path": "numpy_array",
                    "size": image_array.nbytes,
                    "size_mb": round(image_array.nbytes / (1024 * 1024), 2)
                }
            
            height, width = image_array.shape[:2]
            channels = image_array.shape[2] if len(image_array.shape) == 3 else 1
            
            # サイズチェック
            min_size = 32
            max_size = 8192
            
            validation_result = {
                "valid": True,
                "warnings": [],
                "errors": [],
                "image_info": {
                    "width": width,
                    "height": height,
                    "channels": channels,
                    "dtype": str(image_array.dtype),
                    "file_info": file_info
                }
            }
            
            # サイズバリデーション
            if width < min_size or height < min_size:
                validation_result["errors"].append(
                    f"画像が小さすぎます: {width}x{height} < {min_size}x{min_size}"
                )
                validation_result["valid"] = False
            
            if width > max_size or height > max_size:
                validation_result["warnings"].append(
                    f"画像が大きいです: {width}x{height}, 処理時間が長くなる可能性があります"
                )
            
            # アスペクト比チェック
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 10:
                validation_result["warnings"].append(
                    f"極端なアスペクト比です: {aspect_ratio:.1f}:1"
                )
            
            return validation_result
            
        except Exception as e:
            raise ProcessingError(
                f"画像バリデーション中にエラーが発生しました: {e}",
                stage="validate_image"
            ) from e


# グローバル処理インスタンス
_image_processor: Optional[ImageProcessor] = None


def get_image_processor() -> ImageProcessor:
    """グローバル画像処理インスタンス取得"""
    global _image_processor
    
    if _image_processor is None:
        _image_processor = ImageProcessor()
    
    return _image_processor