"""
SAM2モデル管理システム

SAM2モデルのロード、管理、デバイス制御を担当
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from src.core.utils import get_logger, get_config
from src.core.utils.exceptions import ModelError
from src import MODELS_DIR

logger = get_logger(__name__)

@dataclass
class SAM2ModelInfo:
    """SAM2モデル情報"""
    name: str
    path: Path
    device: str
    input_size: int
    loaded: bool
    parameters: Optional[int] = None
    model_size_mb: Optional[float] = None

class SAM2ModelManager:
    """SAM2モデル管理クラス"""
    
    # 利用可能なSAM2モデル
    AVAILABLE_MODELS = {
        "sam2_hiera_tiny.pt": "SAM2 Hiera Tiny",
        "sam2_hiera_small.pt": "SAM2 Hiera Small", 
        "sam2_hiera_base_plus.pt": "SAM2 Hiera Base+",
        "sam2_hiera_large.pt": "SAM2 Hiera Large",
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        SAM2モデル管理クラスの初期化
        
        Args:
            model_name: モデル名（設定ファイルから取得されない場合）
            device: デバイス指定（auto, cpu, cuda）
        """
        self.config = get_config()
        
        # SAM2用設定（YOLO設定を一時的に使用、後で専用設定作成）
        self.model_name = model_name or getattr(self.config.model, 'sam2_name', 'sam2_hiera_large.pt')
        self.device = self._determine_device(device)
        self.model = None
        self.predictor = None
        self.model_info: Optional[SAM2ModelInfo] = None
        
        logger.info(f"SAM2モデル管理クラス初期化: {self.model_name}, device: {self.device}")
        
        # モデル名検証
        self._validate_model_name()
    
    def _determine_device(self, device_override: Optional[str] = None) -> str:
        """最適なデバイスを決定"""
        
        device = device_override or getattr(self.config.model, 'device', 'auto')
        
        if device == "auto":
            # 自動デバイス選択
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
        """モデル名の妥当性を検証"""
        if self.model_name not in self.AVAILABLE_MODELS:
            available = list(self.AVAILABLE_MODELS.keys())
            raise ModelError(
                f"サポートされていないSAM2モデル: {self.model_name}. "
                f"利用可能: {available}"
            )
    
    def load_model(self, force_download: bool = False) -> bool:
        """
        SAM2モデルの読み込み
        
        Args:
            force_download: 強制ダウンロード
            
        Returns:
            読み込み成功フラグ
        """
        
        try:
            # SAM2のインポート（遅延インポート）
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError as e:
                raise ModelError(
                    f"SAM2ライブラリがインストールされていません: {e}. "
                    "pip install SAM-2 を実行してください"
                ) from e
            
            # モデルパスの決定
            model_path = self._get_model_path()
            config_path = self._get_config_path()
            
            logger.info(f"SAM2モデル読み込み開始: {model_path}")
            
            # モデルファイルの存在確認
            if not model_path.exists():
                if force_download:
                    logger.info("SAM2モデルをダウンロード中...")
                    self._download_model()
                else:
                    raise ModelError(
                        f"SAM2モデルファイルが見つかりません: {model_path}. "
                        "download_sam2_models.py を実行してください"
                    )
            
            # SAM2モデル構築
            device_str = self.device
            # config_file には相対パス名のみを指定（SAM2のパッケージ内設定を使用）
            config_name = config_path.name.replace('.yaml', '')
            sam2_model = build_sam2(
                config_file=config_name,
                ckpt_path=str(model_path),
                device=device_str
            )
            
            # Predictor作成
            self.predictor = SAM2ImagePredictor(sam2_model)
            self.model = sam2_model
            
            logger.info("SAM2モデル読み込み完了")
            
            # モデル情報設定
            self._setup_model_info(model_path)
            
            return True
            
        except Exception as e:
            logger.error(f"SAM2モデル読み込み失敗: {e}")
            raise ModelError(f"SAM2モデル読み込みに失敗しました: {e}") from e
    
    def _get_model_path(self) -> Path:
        """モデルファイルパスを取得"""
        
        # プロジェクト内のモデルディレクトリ
        models_dir = MODELS_DIR / "sam2" / "checkpoints"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        return models_dir / self.model_name
    
    def _get_config_path(self) -> Path:
        """設定ファイルパスを取得"""
        
        # モデル名から設定ファイル名を推定
        config_mapping = {
            "sam2_hiera_tiny.pt": "sam2_hiera_t.yaml",
            "sam2_hiera_small.pt": "sam2_hiera_s.yaml",
            "sam2_hiera_base_plus.pt": "sam2_hiera_b+.yaml",
            "sam2_hiera_large.pt": "sam2_hiera_l.yaml",
        }
        
        config_name = config_mapping.get(self.model_name, "sam2_hiera_l.yaml")
        
        # プロジェクト内の設定ディレクトリ
        config_dir = MODELS_DIR / "sam2" / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        return config_dir / config_name
    
    def _download_model(self) -> None:
        """モデルの自動ダウンロード（実装は後で）"""
        raise ModelError(
            "SAM2モデルの自動ダウンロードは未実装です. "
            "scripts/download_sam2_models.py を使用してください"
        )
    
    def _setup_model_info(self, model_path: Path) -> None:
        """モデル情報のセットアップ"""
        
        try:
            # ファイルサイズ
            size_mb = None
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
            
            # パラメータ数（推定）
            parameters = None
            if self.model is not None:
                try:
                    parameters = sum(p.numel() for p in self.model.parameters())
                except Exception:
                    pass
            
            self.model_info = SAM2ModelInfo(
                name=self.model_name,
                path=model_path,
                device=self.device,
                input_size=1024,  # SAM2の標準入力サイズ
                loaded=True,
                parameters=parameters,
                model_size_mb=size_mb
            )
            
        except Exception as e:
            logger.warning(f"モデル情報設定中にエラー: {e}")
    
    def predict(
        self,
        image,
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
    ):
        """
        SAM2セグメンテーション推論実行
        
        Args:
            image: 入力画像
            point_coords: 点座標 [(x, y), ...]
            point_labels: 点ラベル [1(正), 0(負), ...]
            box: バウンディングボックス [x1, y1, x2, y2]
            mask_input: 入力マスク
            multimask_output: 複数マスク出力
            return_logits: ロジット返却
            
        Returns:
            マスク、スコア、ロジット
        """
        
        if not self.is_loaded():
            raise ModelError("SAM2モデルが読み込まれていません")
        
        try:
            # 画像設定
            self.predictor.set_image(image)
            
            # 推論実行
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                mask_input=mask_input,
                multimask_output=multimask_output,
                return_logits=return_logits,
            )
            
            return masks, scores, logits
            
        except Exception as e:
            logger.error(f"SAM2推論中にエラー: {e}")
            raise ModelError(f"SAM2推論に失敗: {e}") from e
    
    def is_loaded(self) -> bool:
        """モデルが読み込まれているかチェック"""
        return self.model is not None and self.predictor is not None
    
    def get_model_info(self) -> Optional[SAM2ModelInfo]:
        """モデル情報取得"""
        return self.model_info
    
    def unload_model(self) -> None:
        """モデルのアンロード"""
        
        if self.model is not None or self.predictor is not None:
            try:
                # GPU メモリクリア
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.model = None
                self.predictor = None
                if self.model_info:
                    self.model_info.loaded = False
                
                logger.info("SAM2モデルをアンロードしました")
                
            except Exception as e:
                logger.warning(f"SAM2モデルアンロード中にエラー: {e}")

# シングルトンパターン用のグローバル変数
_sam2_model_manager: Optional[SAM2ModelManager] = None

def get_sam2_model_manager(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    force_new: bool = False
) -> SAM2ModelManager:
    """
    SAM2ModelManagerのシングルトンインスタンス取得
    
    Args:
        model_name: モデル名
        device: デバイス
        force_new: 新しいインスタンスを強制作成
        
    Returns:
        SAM2ModelManagerインスタンス
    """
    global _sam2_model_manager
    
    if _sam2_model_manager is None or force_new:
        _sam2_model_manager = SAM2ModelManager(
            model_name=model_name,
            device=device
        )
    
    return _sam2_model_manager