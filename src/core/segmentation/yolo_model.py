"""
YOLOv11モデル管理システム

YOLOv11セグメンテーションモデルの読み込み、設定、推論を管理します。
"""

import torch
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

from src.core.utils import get_logger, get_config
from src.core.utils.exceptions import ModelError, ConfigurationError
from src import MODELS_DIR

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """モデル情報クラス"""
    name: str
    path: Path
    device: str
    input_size: int
    loaded: bool = False
    parameters: Optional[int] = None
    model_size_mb: Optional[float] = None


class YOLOModelManager:
    """YOLOv11モデル管理クラス"""
    
    # 利用可能なYOLOv11モデル
    AVAILABLE_MODELS = {
        "yolo11n-seg.pt": "YOLOv11 Nano Segmentation",
        "yolo11s-seg.pt": "YOLOv11 Small Segmentation", 
        "yolo11m-seg.pt": "YOLOv11 Medium Segmentation",
        "yolo11l-seg.pt": "YOLOv11 Large Segmentation",
        "yolo11x-seg.pt": "YOLOv11 Extra Large Segmentation",
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        force_reload: bool = False
    ) -> None:
        """
        YOLOモデル管理クラスの初期化
        
        Args:
            model_name: モデル名（設定ファイルから取得されない場合）
            device: デバイス指定（auto, cpu, cuda, mps）
            force_reload: 強制再読み込み
        """
        self.config = get_config()
        self.model_name = model_name or self.config.model.name
        self.device = self._determine_device(device)
        self.model = None
        self.model_info: Optional[ModelInfo] = None
        
        logger.info(f"YOLOモデル管理クラス初期化: {self.model_name}, device: {self.device}")
        
        if not force_reload:
            self._validate_model_name()
    
    def _determine_device(self, device_override: Optional[str] = None) -> str:
        """最適なデバイスを決定"""
        
        device = device_override or self.config.model.device
        
        if device == "auto":
            # 自動デバイス選択 (CPU版では常にCPU)
            try:
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    device = "cuda"
                    logger.info(f"CUDA利用可能: {torch.cuda.get_device_name()}")
                elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps"
                    logger.info("Apple Silicon MPS利用可能")
                else:
                    device = "cpu"
                    logger.info("CPU使用")
            except Exception:
                # CPU版torchの場合は常にCPU
                device = "cpu"
                logger.info("CPU版Torch使用")
        
        # デバイス使用可能性の確認
        try:
            if device == "cuda" and hasattr(torch, 'cuda') and not torch.cuda.is_available():
                logger.warning("CUDA要求されましたが利用不可能、CPUにフォールバック")
                device = "cpu"
            elif device == "mps" and hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and not torch.backends.mps.is_available():
                logger.warning("MPS要求されましたが利用不可能、CPUにフォールバック")
                device = "cpu"
        except Exception:
            # CPU版の場合は常にCPU
            device = "cpu"
        
        return device
    
    def _validate_model_name(self) -> None:
        """モデル名の妥当性確認"""
        
        if self.model_name not in self.AVAILABLE_MODELS:
            available = list(self.AVAILABLE_MODELS.keys())
            raise ModelError(
                f"サポートされていないモデルです: {self.model_name}",
                model_name=self.model_name,
                details={"available_models": available}
            )
    
    def load_model(self, force_download: bool = False) -> bool:
        """
        YOLOモデルの読み込み
        
        Args:
            force_download: 強制ダウンロード
            
        Returns:
            読み込み成功フラグ
        """
        
        try:
            # ultralyticsの遅延インポート
            from ultralytics import YOLO
            
            # モデルパスの決定
            model_path = self._get_model_path()
            
            logger.info(f"YOLOモデル読み込み開始: {model_path}")
            
            # モデル読み込み（自動ダウンロード対応）
            if force_download or not model_path.exists():
                logger.info("事前学習モデルをダウンロード中...")
                # ultralyticsが自動的にダウンロード
                self.model = YOLO(self.model_name)
            else:
                self.model = YOLO(str(model_path))
            
            # デバイス設定
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            
            # モデル情報の設定
            self._setup_model_info(model_path)
            
            logger.info(f"YOLOモデル読み込み完了: {self.model_info}")
            return True
            
        except ImportError as e:
            raise ModelError(
                f"ultralyticsライブラリが見つかりません: {e}",
                model_name=self.model_name
            ) from e
        except Exception as e:
            raise ModelError(
                f"YOLOモデル読み込みに失敗しました: {e}",
                model_name=self.model_name,
                model_path=str(model_path) if 'model_path' in locals() else None
            ) from e
    
    def _get_model_path(self) -> Path:
        """モデルファイルパスを取得"""
        
        # プロジェクト内のモデルディレクトリ
        models_dir = MODELS_DIR / "yolo" / "pretrained"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        return models_dir / self.model_name
    
    def _setup_model_info(self, model_path: Path) -> None:
        """モデル情報のセットアップ"""
        
        try:
            # ファイルサイズ
            size_mb = None
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
            
            # パラメータ数（可能であれば）
            parameters = None
            try:
                if hasattr(self.model.model, 'parameters'):
                    parameters = sum(p.numel() for p in self.model.model.parameters())
            except Exception:
                pass
            
            self.model_info = ModelInfo(
                name=self.model_name,
                path=model_path,
                device=self.device,
                input_size=self.config.processing.input_size,
                loaded=True,
                parameters=parameters,
                model_size_mb=size_mb
            )
            
        except Exception as e:
            logger.warning(f"モデル情報設定中にエラー: {e}")
    
    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        セグメンテーション推論実行
        
        Args:
            image: 入力画像（パスまたは配列）
            confidence: 信頼度閾値
            iou_threshold: IoU閾値
            **kwargs: その他のYOLO推論パラメータ
            
        Returns:
            YOLO推論結果
        """
        
        if not self.is_loaded():
            raise ModelError(
                "モデルが読み込まれていません",
                model_name=self.model_name
            )
        
        try:
            # パラメータの設定
            params = {
                "conf": confidence or self.config.model.confidence,
                "iou": iou_threshold or self.config.model.iou_threshold,
                "device": self.device,
                "half": self.config.model.half,
                "max_det": self.config.model.max_det,
                **kwargs
            }
            
            logger.debug(f"セグメンテーション実行: {params}")
            
            # 推論実行
            results = self.model.predict(image, **params)
            
            logger.debug(f"セグメンテーション完了: {len(results)} results")
            return results
            
        except Exception as e:
            raise ModelError(
                f"セグメンテーション実行中にエラーが発生しました: {e}",
                model_name=self.model_name
            ) from e
    
    def is_loaded(self) -> bool:
        """モデル読み込み状態確認"""
        return self.model is not None and self.model_info is not None
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """モデル情報取得"""
        return self.model_info
    
    def unload_model(self) -> None:
        """モデルのアンロード"""
        
        if self.model is not None:
            try:
                # GPU メモリクリア
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.model = None
                if self.model_info:
                    self.model_info.loaded = False
                
                logger.info("YOLOモデルをアンロードしました")
                
            except Exception as e:
                logger.warning(f"モデルアンロード中にエラー: {e}")
    
    def switch_device(self, new_device: str) -> bool:
        """デバイス切り替え"""
        
        try:
            new_device = self._determine_device(new_device)
            
            if new_device == self.device:
                logger.info(f"既に同じデバイスです: {new_device}")
                return True
            
            if self.is_loaded():
                # モデルを新しいデバイスに移動
                if hasattr(self.model, 'to'):
                    self.model.to(new_device)
                
                self.device = new_device
                if self.model_info:
                    self.model_info.device = new_device
                
                logger.info(f"デバイスを切り替えました: {new_device}")
                return True
            else:
                self.device = new_device
                logger.info(f"デバイス設定を変更しました: {new_device}")
                return True
                
        except Exception as e:
            logger.error(f"デバイス切り替えに失敗: {e}")
            return False
    
    def __del__(self) -> None:
        """デストラクタ"""
        try:
            self.unload_model()
        except Exception:
            pass


# グローバルモデル管理インスタンス
_model_manager: Optional[YOLOModelManager] = None


def get_model_manager() -> YOLOModelManager:
    """グローバルモデル管理インスタンス取得"""
    global _model_manager
    
    if _model_manager is None:
        _model_manager = YOLOModelManager()
    
    return _model_manager


def initialize_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    force_reload: bool = False
) -> YOLOModelManager:
    """モデル管理システムの初期化"""
    global _model_manager
    
    _model_manager = YOLOModelManager(
        model_name=model_name,
        device=device,
        force_reload=force_reload
    )
    
    return _model_manager