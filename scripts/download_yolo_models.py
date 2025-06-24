#!/usr/bin/env python3
"""
YOLOモデルダウンロードスクリプト

必要なYOLOv11セグメンテーションモデルを事前ダウンロード
"""

import sys
import os
from pathlib import Path
import urllib.request
import time

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils import setup_logging, get_logger

def download_file_with_progress(url, filepath):
    """プログレスバー付きファイルダウンロード"""
    logger = get_logger(__name__)
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  進行状況: {percent:3d}% ({downloaded_mb:.1f}MB / {total_mb:.1f}MB)", end='', flush=True)
        else:
            downloaded_mb = downloaded / (1024 * 1024)
            print(f"\r  ダウンロード済み: {downloaded_mb:.1f}MB", end='', flush=True)
    
    try:
        logger.info(f"ダウンロード開始: {filepath.name}")
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print()  # 改行
        logger.info(f"✅ ダウンロード完了: {filepath.name}")
        return True
    except Exception as e:
        print()  # 改行
        logger.error(f"❌ ダウンロード失敗: {e}")
        return False

def download_yolo_models():
    """YOLOv11セグメンテーションモデルをダウンロード"""
    logger = get_logger(__name__)
    
    # モデル情報（サイズ順）
    models = {
        "yolo11n-seg.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
            "size": "約7MB",
            "description": "Nano - 最軽量・最高速"
        },
        "yolo11s-seg.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt", 
            "size": "約24MB",
            "description": "Small - バランス型"
        },
        "yolo11m-seg.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt",
            "size": "約51MB", 
            "description": "Medium - 中精度"
        },
        "yolo11l-seg.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt",
            "size": "約133MB",
            "description": "Large - 高精度"
        },
        "yolo11x-seg.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt",
            "size": "約265MB",
            "description": "Extra Large - 最高精度"
        }
    }
    
    # ダウンロード先ディレクトリ
    models_dir = project_root / "models" / "yolo" / "pretrained"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"=== YOLOv11セグメンテーションモデル ダウンロード ===")
    logger.info(f"保存先: {models_dir}")
    
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    for model_name, model_info in models.items():
        filepath = models_dir / model_name
        
        logger.info(f"\n📦 {model_name}")
        logger.info(f"   {model_info['description']} ({model_info['size']})")
        
        if filepath.exists():
            logger.info(f"   ⏭️  既に存在しています (スキップ)")
            skipped_count += 1
            continue
        
        # ダウンロード実行
        success = download_file_with_progress(model_info["url"], filepath)
        
        if success:
            # ファイルサイズ確認
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"   📁 ファイルサイズ: {file_size_mb:.1f}MB")
            downloaded_count += 1
        else:
            # 失敗した場合は部分ファイルを削除
            if filepath.exists():
                filepath.unlink()
            failed_count += 1
    
    # 結果サマリー
    logger.info(f"\n{'='*50}")
    logger.info(f"=== ダウンロード結果 ===")
    logger.info(f"✅ ダウンロード完了: {downloaded_count}個")
    logger.info(f"⏭️  スキップ: {skipped_count}個")
    logger.info(f"❌ 失敗: {failed_count}個")
    logger.info(f"📁 保存先: {models_dir}")
    
    # 利用可能モデル一覧
    available_models = list(models_dir.glob("*.pt"))
    if available_models:
        logger.info(f"\n利用可能なモデル:")
        for model_path in sorted(available_models):
            size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"  • {model_path.name} ({size_mb:.1f}MB)")
    
    return downloaded_count > 0 or skipped_count > 0

def check_models_exist():
    """既存モデルの確認"""
    logger = get_logger(__name__)
    
    models_dir = project_root / "models" / "yolo" / "pretrained"
    
    if not models_dir.exists():
        logger.info("モデルディレクトリが存在しません")
        return []
    
    existing_models = list(models_dir.glob("yolo11*-seg.pt"))
    
    if existing_models:
        logger.info(f"既存モデル ({len(existing_models)}個):")
        for model_path in sorted(existing_models):
            size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"  • {model_path.name} ({size_mb:.1f}MB)")
    else:
        logger.info("YOLOv11セグメンテーションモデルが見つかりません")
    
    return existing_models

def main():
    """メイン実行"""
    
    # ログ設定
    setup_logging(level="INFO", console_output=True, structured=False)
    logger = get_logger(__name__)
    
    logger.info("=== YOLOv11モデル管理 ===")
    
    try:
        # 既存モデル確認
        existing_models = check_models_exist()
        
        if len(existing_models) >= 4:
            logger.info("\n✅ 十分なモデルが既に存在します")
            logger.info("強制的に再ダウンロードする場合は、modelsディレクトリを削除してください")
            return
        
        # 不足モデルをダウンロード
        logger.info(f"\n追加モデルをダウンロードします...")
        success = download_yolo_models()
        
        if success:
            logger.info(f"\n🎉 モデルダウンロード完了!")
            logger.info(f"高精度セグメンテーションテストが実行可能です:")
            logger.info(f"  uv run python scripts/test_high_precision_segmentation.py")
        else:
            logger.error(f"\n❌ ダウンロードに失敗しました")
            logger.info(f"手動でのダウンロードを検討してください")
    
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()