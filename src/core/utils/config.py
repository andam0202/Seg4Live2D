"""
設定管理システム

YAML設定ファイルの読み込み・管理を行います。
環境変数との統合、設定バリデーション機能を提供します。
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, Type, TypeVar
import yaml
from dataclasses import dataclass, field

from src import CONFIG_DIR, PROJECT_ROOT
from .exceptions import ConfigurationError
from .logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


@dataclass
class ModelConfig:
    """モデル設定"""
    name: str = "yolo11n-seg.pt"
    device: str = "auto"  # auto, cpu, cuda, mps
    confidence: float = 0.25
    iou_threshold: float = 0.7
    max_det: int = 300
    half: bool = False  # FP16推論
    
    
@dataclass 
class ProcessingConfig:
    """処理設定"""
    input_size: int = 640
    batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    

@dataclass
class Live2DConfig:
    """Live2D出力設定"""
    output_format: str = "png"  # png, psd
    layer_separation: bool = True
    transparency_processing: bool = True
    mesh_generation: bool = False
    max_layers: int = 50
    

@dataclass
class UIConfig:
    """UI設定"""
    host: str = "localhost"
    port: int = 8501
    debug: bool = False
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: list = field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".webp"])
    

@dataclass
class Config:
    """メイン設定クラス"""
    model: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    live2d: Live2DConfig = field(default_factory=Live2DConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # メタ情報
    version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    

class ConfigManager:
    """設定管理クラス"""
    
    def __init__(self) -> None:
        self._config: Optional[Config] = None
        self._config_file: Optional[Path] = None
        
    def load(
        self, 
        config_file: Optional[Union[str, Path]] = None,
        environment: Optional[str] = None
    ) -> Config:
        """設定ファイルの読み込み"""
        
        # 環境の決定
        if environment is None:
            environment = os.getenv("SEG4LIVE2D_ENV", "development")
            
        # 設定ファイルの決定
        if config_file is None:
            config_file = self._find_config_file(environment)
        else:
            config_file = Path(config_file)
            
        if not config_file.exists():
            logger.warning(f"設定ファイルが見つかりません: {config_file}")
            logger.info("デフォルト設定を使用します")
            self._config = Config()
            return self._config
            
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                
            # 環境変数での上書き
            config_data = self._apply_env_overrides(config_data)
            
            # 設定オブジェクトの構築
            self._config = self._build_config(config_data)
            self._config_file = config_file
            
            logger.info(f"設定ファイルを読み込みました: {config_file}")
            return self._config
            
        except Exception as e:
            raise ConfigurationError(
                f"設定ファイルの読み込みに失敗しました: {e}",
                config_file=str(config_file)
            ) from e
    
    def _find_config_file(self, environment: str) -> Path:
        """環境に応じた設定ファイルを探索"""
        
        # 優先順位順で探索
        candidates = [
            CONFIG_DIR / "app" / f"{environment}.yaml",
            CONFIG_DIR / "app" / "default.yaml",
            PROJECT_ROOT / "config.yaml",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
                
        # 見つからない場合はデフォルトパス
        return CONFIG_DIR / "app" / "default.yaml"
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """環境変数による設定上書き"""
        
        # 環境変数マッピング
        env_mappings = {
            "SEG4LIVE2D_MODEL_DEVICE": ["model", "device"],
            "SEG4LIVE2D_MODEL_CONFIDENCE": ["model", "confidence"],
            "SEG4LIVE2D_PROCESSING_BATCH_SIZE": ["processing", "batch_size"],
            "SEG4LIVE2D_UI_HOST": ["ui", "host"],
            "SEG4LIVE2D_UI_PORT": ["ui", "port"],
            "SEG4LIVE2D_DEBUG": ["debug"],
        }
        
        for env_var, path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # 型変換
                if path[-1] in ["confidence"]:
                    value = float(value)
                elif path[-1] in ["batch_size", "port"]:
                    value = int(value)
                elif path[-1] in ["debug"]:
                    value = value.lower() in ["true", "1", "yes", "on"]
                
                # 設定に反映
                target = config_data
                for key in path[:-1]:
                    if key not in target:
                        target[key] = {}
                    target = target[key]
                target[path[-1]] = value
                
        return config_data
    
    def _build_config(self, config_data: Dict[str, Any]) -> Config:
        """設定データからConfigオブジェクトを構築"""
        
        try:
            # 各セクションの構築
            model_config = ModelConfig(**config_data.get("model", {}))
            processing_config = ProcessingConfig(**config_data.get("processing", {}))
            live2d_config = Live2DConfig(**config_data.get("live2d", {}))
            ui_config = UIConfig(**config_data.get("ui", {}))
            
            # メイン設定の構築
            config = Config(
                model=model_config,
                processing=processing_config,
                live2d=live2d_config,
                ui=ui_config,
                version=config_data.get("version", "0.1.0"),
                environment=config_data.get("environment", "development"),
                debug=config_data.get("debug", False),
            )
            
            return config
            
        except Exception as e:
            raise ConfigurationError(f"設定の構築に失敗しました: {e}") from e
    
    def get(self) -> Config:
        """現在の設定を取得"""
        if self._config is None:
            return self.load()
        return self._config
    
    def reload(self) -> Config:
        """設定の再読み込み"""
        self._config = None
        return self.load(self._config_file)
    

# グローバル設定マネージャー
_config_manager = ConfigManager()


def load_config(
    config_file: Optional[Union[str, Path]] = None,
    environment: Optional[str] = None
) -> Config:
    """設定の読み込み（グローバル関数）"""
    return _config_manager.load(config_file, environment)


def get_config() -> Config:
    """現在の設定を取得（グローバル関数）"""
    return _config_manager.get()


def reload_config() -> Config:
    """設定の再読み込み（グローバル関数）"""
    return _config_manager.reload()