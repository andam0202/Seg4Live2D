#!/usr/bin/env python3
"""
SAM2モデルダウンロードスクリプト

Live2D用SAM2モデルと設定ファイルを自動ダウンロード
- 必要なモデルファイルの自動取得
- 設定ファイルの自動配置
- 進行状況表示付きダウンロード
"""

import sys
import os
import urllib.request
import shutil
from pathlib import Path

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

def download_sam2_assets():
    """SAM2モデルと設定ファイルをダウンロード"""
    logger = get_logger(__name__)
    
    # SAM2アセット情報
    assets = {
        # モデルファイル
        "models": {
            "sam2_hiera_tiny.pt": {
                "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
                "size": "約155MB",
                "description": "SAM2 Hiera Tiny - 最軽量"
            },
            "sam2_hiera_small.pt": {
                "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt", 
                "size": "約184MB",
                "description": "SAM2 Hiera Small - バランス型"
            },
            "sam2_hiera_base_plus.pt": {
                "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
                "size": "約320MB",
                "description": "SAM2 Hiera Base+ - 高性能"
            },
            "sam2_hiera_large.pt": {
                "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
                "size": "約896MB", 
                "description": "SAM2 Hiera Large - 最高性能"
            }
        },
        # 設定ファイル
        "configs": {
            "sam2_hiera_t.yaml": {
                "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2_hiera_t.yaml",
                "size": "約2KB",
                "description": "Tiny用設定"
            },
            "sam2_hiera_s.yaml": {
                "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2_hiera_s.yaml",
                "size": "約2KB", 
                "description": "Small用設定"
            },
            "sam2_hiera_b+.yaml": {
                "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2_hiera_b+.yaml",
                "size": "約2KB",
                "description": "Base+用設定"
            },
            "sam2_hiera_l.yaml": {
                "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2_hiera_l.yaml",
                "size": "約2KB",
                "description": "Large用設定"
            }
        }
    }
    
    logger.info("=== SAM2アセット ダウンロード ===")
    
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    # モデルファイルダウンロード
    models_dir = project_root / "models" / "sam2" / "checkpoints"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n📦 モデルファイル ダウンロード")
    logger.info(f"保存先: {models_dir}")
    
    for model_name, model_info in assets["models"].items():
        filepath = models_dir / model_name
        
        logger.info(f"\n🤖 {model_name}")
        logger.info(f"   {model_info['description']} ({model_info['size']})")
        
        if filepath.exists():
            logger.info(f"   ⏭️  既に存在しています (スキップ)")
            skipped_count += 1
            continue
        
        # ダウンロード実行
        success = download_file_with_progress(model_info["url"], filepath)
        
        if success:
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"   📁 ファイルサイズ: {file_size_mb:.1f}MB")
            downloaded_count += 1
        else:
            if filepath.exists():
                filepath.unlink()
            failed_count += 1
    
    # 設定ファイルダウンロード
    configs_dir = project_root / "models" / "sam2" / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n⚙️  設定ファイル ダウンロード")
    logger.info(f"保存先: {configs_dir}")
    
    for config_name, config_info in assets["configs"].items():
        filepath = configs_dir / config_name
        
        logger.info(f"\n📄 {config_name}")
        logger.info(f"   {config_info['description']} ({config_info['size']})")
        
        if filepath.exists():
            logger.info(f"   ⏭️  既に存在しています (スキップ)")
            skipped_count += 1
            continue
        
        # ダウンロード実行
        success = download_file_with_progress(config_info["url"], filepath)
        
        if success:
            file_size_kb = filepath.stat().st_size / 1024
            logger.info(f"   📁 ファイルサイズ: {file_size_kb:.1f}KB")
            downloaded_count += 1
        else:
            if filepath.exists():
                filepath.unlink()
            failed_count += 1
    
    # 結果サマリー
    logger.info(f"\n{'='*50}")
    logger.info(f"=== ダウンロード結果 ===")
    logger.info(f"✅ ダウンロード完了: {downloaded_count}個")
    logger.info(f"⏭️  スキップ: {skipped_count}個")
    logger.info(f"❌ 失敗: {failed_count}個")
    
    # 利用可能アセット一覧
    available_models = list(models_dir.glob("*.pt"))
    available_configs = list(configs_dir.glob("*.yaml"))
    
    if available_models or available_configs:
        logger.info(f"\n利用可能なアセット:")
        
        if available_models:
            logger.info(f"  モデル ({len(available_models)}個):")
            for model_path in sorted(available_models):
                size_mb = model_path.stat().st_size / (1024 * 1024)
                logger.info(f"    • {model_path.name} ({size_mb:.1f}MB)")
        
        if available_configs:
            logger.info(f"  設定 ({len(available_configs)}個):")
            for config_path in sorted(available_configs):
                logger.info(f"    • {config_path.name}")
    
    return downloaded_count > 0 or skipped_count > 0

def check_sam2_environment():
    """SAM2環境のチェック"""
    logger = get_logger(__name__)
    
    logger.info("=== SAM2環境チェック ===")
    
    # SAM2ライブラリチェック
    try:
        import sam2
        logger.info(f"✅ SAM2ライブラリ: インストール済み")
        logger.info(f"   バージョン: {getattr(sam2, '__version__', 'unknown')}")
    except ImportError:
        logger.error("❌ SAM2ライブラリ: 未インストール")
        logger.info("   インストール方法:")
        logger.info("   git clone https://github.com/facebookresearch/segment-anything-2.git")
        logger.info("   cd segment-anything-2 && pip install -e .")
        return False
    
    # PyTorchチェック
    try:
        import torch
        logger.info(f"✅ PyTorch: {torch.__version__}")
        logger.info(f"   CUDA利用可能: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"   GPU: {torch.cuda.get_device_name()}")
    except ImportError:
        logger.error("❌ PyTorch: 未インストール")
        return False
    
    return True

def main():
    """メイン実行"""
    
    # ログ設定
    setup_logging(level="INFO", console_output=True, structured=False)
    logger = get_logger(__name__)
    
    logger.info("=== SAM2セットアップ ===")
    
    try:
        # 環境チェック
        if not check_sam2_environment():
            logger.error("SAM2環境が正しくセットアップされていません")
            logger.info("まずSAM2ライブラリをインストールしてください")
            return
        
        # アセットダウンロード
        logger.info(f"\nSAM2アセットをダウンロードします...")
        success = download_sam2_assets()
        
        if success:
            logger.info(f"\n🎉 SAM2セットアップ完了!")
            logger.info(f"SAM2テストが実行可能です:")
            logger.info(f"  uv run python scripts/test_sam2_basic.py")
        else:
            logger.error(f"\n❌ セットアップに失敗しました")
    
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()