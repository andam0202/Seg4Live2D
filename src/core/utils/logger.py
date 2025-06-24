"""
ログ管理システム

構造化ログを提供し、開発・運用の両方で有用な情報を記録します。
"""

import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json

from src import LOGS_DIR
from .exceptions import ConfigurationError


class StructuredFormatter(logging.Formatter):
    """構造化ログ用のカスタムフォーマッター"""
    
    def format(self, record: logging.LogRecord) -> str:
        # 基本ログ情報
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # 追加情報があれば含める
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
            
        # 例外情報があれば含める
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data, ensure_ascii=False)


class Seg4Live2DLogger:
    """Seg4Live2D専用ロガークラス"""
    
    def __init__(self, name: str) -> None:
        self.logger = logging.getLogger(name)
        self._setup_done = False
        
    def info(self, message: str, **kwargs: Any) -> None:
        """情報ログ"""
        self._log(logging.INFO, message, **kwargs)
        
    def warning(self, message: str, **kwargs: Any) -> None:
        """警告ログ"""
        self._log(logging.WARNING, message, **kwargs)
        
    def error(self, message: str, **kwargs: Any) -> None:
        """エラーログ"""
        self._log(logging.ERROR, message, **kwargs)
        
    def debug(self, message: str, **kwargs: Any) -> None:
        """デバッグログ"""
        self._log(logging.DEBUG, message, **kwargs)
        
    def critical(self, message: str, **kwargs: Any) -> None:
        """クリティカルログ"""
        self._log(logging.CRITICAL, message, **kwargs)
        
    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """内部ログメソッド"""
        if not self._setup_done:
            setup_logging()
            self._setup_done = True
            
        extra = {"extra_data": kwargs} if kwargs else {}
        self.logger.log(level, message, extra=extra)


def get_logger(name: str) -> Seg4Live2DLogger:
    """ロガーインスタンスを取得"""
    return Seg4Live2DLogger(name)


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Path] = None,
    structured: bool = True,
    console_output: bool = True
) -> None:
    """ログシステムの初期化"""
    
    # ログディレクトリの作成
    if not LOGS_DIR.exists():
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # デフォルトのログファイル
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = LOGS_DIR / f"seg4live2d_{timestamp}.log"
    
    # ログレベルの正規化
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # ハンドラーの設定
    handlers = {}
    
    # ファイルハンドラー
    if structured:
        file_formatter = StructuredFormatter()
    else:
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    handlers["file"] = {
        "class": "logging.handlers.RotatingFileHandler",
        "filename": str(log_file),
        "maxBytes": 10 * 1024 * 1024,  # 10MB
        "backupCount": 5,
        "formatter": "structured" if structured else "standard",
        "level": level,
    }
    
    # コンソールハンドラー
    if console_output:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "console",
            "level": level,
        }
    
    # フォーマッター設定
    formatters = {
        "structured": {
            "()": StructuredFormatter,
        },
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "console": {
            "format": "%(asctime)s - %(levelname)s - %(message)s",
            "datefmt": "%H:%M:%S",
        },
    }
    
    # ログ設定
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "root": {
            "level": level,
            "handlers": list(handlers.keys()),
        },
        "loggers": {
            "seg4live2d": {
                "level": level,
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
            "ultralytics": {
                "level": "WARNING",  # ultralyticsのログを抑制
            },
        },
    }
    
    try:
        logging.config.dictConfig(config)
    except Exception as e:
        raise ConfigurationError(f"ログ設定の初期化に失敗しました: {e}") from e


# モジュール読み込み時にデフォルト設定を適用
if not logging.getLogger().handlers:
    setup_logging()