"""
Seg4Live2D - YOLOv11セグメンテーション技術を活用したLive2D用素材自動分割システム

このパッケージは、イラストを自動的にLive2D用素材として分割するための
包括的なツールセットを提供します。

主要機能:
- YOLOv11による高精度セグメンテーション
- Live2D特化の後処理
- Webベースユーザーインターフェース
- バッチ処理機能
"""

__version__ = "0.1.0"
__author__ = "Seg4Live2D Team"
__email__ = "dev@seg4live2d.example.com"

from pathlib import Path

# プロジェクトルートディレクトリ
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = PROJECT_ROOT / "logs"

# 各データディレクトリ
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUT_DIR = DATA_DIR / "output"
TEMP_DIR = DATA_DIR / "temp"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "CONFIG_DIR",
    "LOGS_DIR",
    "SAMPLES_DIR",
    "OUTPUT_DIR",
    "TEMP_DIR",
]