"""
カスタム例外クラス

Seg4Live2D固有の例外を定義し、適切なエラーハンドリングを提供します。
"""

from typing import Optional, Any, Dict


class Seg4Live2DError(Exception):
    """Seg4Live2Dの基底例外クラス"""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ValidationError(Seg4Live2DError):
    """バリデーション関連のエラー"""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ) -> None:
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field = field
        self.value = value


class ProcessingError(Seg4Live2DError):
    """画像処理関連のエラー"""
    
    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        image_path: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(message, error_code="PROCESSING_ERROR", **kwargs)
        self.stage = stage
        self.image_path = image_path


class ModelError(Seg4Live2DError):
    """モデル関連のエラー"""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(message, error_code="MODEL_ERROR", **kwargs)
        self.model_name = model_name
        self.model_path = model_path


class ConfigurationError(Seg4Live2DError):
    """設定関連のエラー"""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        self.config_key = config_key
        self.config_file = config_file


class FileOperationError(Seg4Live2DError):
    """ファイル操作関連のエラー"""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(message, error_code="FILE_ERROR", **kwargs)
        self.file_path = file_path
        self.operation = operation


class NetworkError(Seg4Live2DError):
    """ネットワーク関連のエラー"""
    
    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__(message, error_code="NETWORK_ERROR", **kwargs)
        self.url = url
        self.status_code = status_code