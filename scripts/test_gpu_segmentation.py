#!/usr/bin/env python3
"""
GPU使用状況確認付きセグメンテーションテスト

GPUが正しく使用されているか確認しながらセグメンテーションを実行
"""

import sys
import time
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils import setup_logging, get_logger

def check_gpu_status():
    """GPU使用状況を確認"""
    import torch
    
    logger = get_logger(__name__)
    
    logger.info("=== GPU状況確認 ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Device count: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # メモリ使用状況
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        logger.info(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    else:
        logger.warning("CUDA is not available! Running on CPU.")
    
    return torch.cuda.is_available()

def monitor_gpu_memory(label=""):
    """GPU メモリ使用量をモニター"""
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        return f"{label} - GPU Memory: Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB"
    return f"{label} - Running on CPU"

def main():
    """メイン実行"""
    
    # ログ設定
    setup_logging(level="INFO", console_output=True, structured=False)
    logger = get_logger(__name__)
    
    logger.info("=== GPU対応セグメンテーションテスト ===")
    
    # GPU状況確認
    has_gpu = check_gpu_status()
    
    try:
        # セグメンテーションエンジンのインポートと初期化
        from src.core.segmentation import get_segmentation_engine
        from src.core.utils import load_config
        
        # 設定読み込み（GPU使用を明示的に指定）
        config = load_config()
        if has_gpu:
            config.model.device = "cuda"
            logger.info("設定: GPU (CUDA) を使用")
        else:
            config.model.device = "cpu"
            logger.info("設定: CPU を使用")
        
        # エンジン初期化
        logger.info("\n=== エンジン初期化 ===")
        logger.info(monitor_gpu_memory("初期化前"))
        
        engine = get_segmentation_engine()
        engine.initialize()
        
        logger.info(monitor_gpu_memory("初期化後"))
        
        # テスト画像の準備
        sample_images_path = project_root / "data" / "samples" / "demo_images"
        image_files = list(sample_images_path.glob("*.png"))[:3]  # 最初の3枚をテスト
        
        if not image_files:
            logger.error("テスト画像が見つかりません")
            return
        
        logger.info(f"\nテスト画像: {len(image_files)}枚")
        
        # 各画像でセグメンテーション実行
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"\n=== 画像 {i}/{len(image_files)}: {image_file.name} ===")
            
            logger.info(monitor_gpu_memory("処理前"))
            
            start_time = time.time()
            result = engine.process_image(
                image=image_file,
                save_masks=True,
                output_dir=project_root / "data" / "output" / "gpu_test"
            )
            processing_time = time.time() - start_time
            
            logger.info(monitor_gpu_memory("処理後"))
            
            if result.success:
                logger.info(f"✅ 成功 - 処理時間: {processing_time:.3f}秒")
                logger.info(f"   検出マスク数: {len(result.masks)}")
                logger.info(f"   デバイス: {engine.model_manager.device}")
                
                # 最初の5つのマスクを表示
                for j, mask in enumerate(result.masks[:5]):
                    logger.info(f"   マスク{j+1}: {mask.class_name} (confidence={mask.confidence:.3f})")
            else:
                logger.error(f"❌ 失敗: {result.error_message}")
        
        # 最終メモリ状況
        logger.info(f"\n{monitor_gpu_memory('最終状態')}")
        
        # GPU使用の総評
        if has_gpu and engine.model_manager.device == "cuda":
            logger.info("\n🎉 GPU（CUDA）を使用してセグメンテーションが実行されました！")
        else:
            logger.info("\n⚠️ CPUモードで実行されました")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # クリーンアップ
        if 'engine' in locals() and hasattr(engine, 'cleanup'):
            engine.cleanup()
            logger.info("クリーンアップ完了")

if __name__ == "__main__":
    main()