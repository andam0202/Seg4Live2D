#!/usr/bin/env python3
"""
SAM2基本動作テスト

SAM2の基本的なセグメンテーション機能をテストして動作確認
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils import setup_logging, get_logger

def test_sam2_import():
    """SAM2インポートテスト"""
    logger = get_logger(__name__)
    
    try:
        # SAM2関連のインポート
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        logger.info("✅ SAM2インポート成功")
        return True
        
    except ImportError as e:
        logger.error(f"❌ SAM2インポート失敗: {e}")
        return False

def test_sam2_model_loading():
    """SAM2モデル読み込みテスト"""
    logger = get_logger(__name__)
    
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # 最軽量のTinyモデルでテスト
        model_cfg = "sam2_hiera_t.yaml"
        sam2_checkpoint = project_root / "models" / "sam2" / "checkpoints" / "sam2_hiera_tiny.pt"
        
        if not sam2_checkpoint.exists():
            logger.error(f"❌ SAM2モデルファイルが見つかりません: {sam2_checkpoint}")
            return False
        
        logger.info(f"SAM2モデル読み込み開始: {sam2_checkpoint}")
        
        # SAM2モデル構築
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
        
        # Predictor作成
        predictor = SAM2ImagePredictor(sam2_model)
        
        logger.info("✅ SAM2モデル読み込み成功")
        return True, predictor
        
    except Exception as e:
        logger.error(f"❌ SAM2モデル読み込み失敗: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_sam2_segmentation():
    """SAM2セグメンテーションテスト"""
    logger = get_logger(__name__)
    
    try:
        # モデル読み込み
        success, predictor = test_sam2_model_loading()
        if not success:
            return False
        
        # テスト画像の準備
        sample_images_path = project_root / "data" / "samples" / "demo_images"
        image_files = list(sample_images_path.glob("*.png"))
        
        if not image_files:
            logger.error("❌ テスト画像が見つかりません")
            return False
        
        test_image_path = image_files[0]
        logger.info(f"テスト画像: {test_image_path.name}")
        
        # 画像読み込み
        image = cv2.imread(str(test_image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        logger.info(f"画像サイズ: {image.shape[:2]}")
        
        # SAM2に画像を設定
        predictor.set_image(image)
        
        # 簡単なテスト：画像中央に点プロンプト
        height, width = image.shape[:2]
        input_point = np.array([[width//2, height//2]])
        input_label = np.array([1])  # 1 = foreground点
        
        logger.info(f"プロンプト点: {input_point[0]}")
        
        # セグメンテーション実行
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        logger.info(f"✅ セグメンテーション成功")
        logger.info(f"   生成マスク数: {len(masks)}")
        logger.info(f"   スコア: {scores}")
        logger.info(f"   マスクサイズ: {masks[0].shape}")
        
        # 結果保存
        output_dir = project_root / "data" / "output" / "sam2_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 最高スコアのマスクを保存
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx].astype(bool)  # boolean型に変換
        
        # マスク画像として保存
        mask_image = (best_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(output_dir / f"sam2_mask_{test_image_path.stem}.png"), mask_image)
        
        # オリジナル画像にマスクをオーバーレイ
        overlay = image.copy().astype(np.float32)
        overlay[best_mask] = overlay[best_mask] * 0.7 + np.array([255, 0, 0]) * 0.3
        overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"sam2_overlay_{test_image_path.stem}.png"), overlay_bgr)
        
        logger.info(f"結果保存: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SAM2セグメンテーション失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sam2_multiple_prompts():
    """SAM2複数プロンプトテスト"""
    logger = get_logger(__name__)
    
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # モデル読み込み
        model_cfg = "sam2_hiera_t.yaml"
        sam2_checkpoint = project_root / "models" / "sam2" / "checkpoints" / "sam2_hiera_tiny.pt"
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
        predictor = SAM2ImagePredictor(sam2_model)
        
        # テスト画像
        sample_images_path = project_root / "data" / "samples" / "demo_images"
        image_files = list(sample_images_path.glob("*.png"))
        test_image_path = image_files[0]
        
        image = cv2.imread(str(test_image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        
        height, width = image.shape[:2]
        
        # 複数点プロンプト（例：キャラクターの髪、顔、体の部分）
        input_points = np.array([
            [width//3, height//4],      # 髪の想定位置
            [width//2, height//2],      # 顔の想定位置  
            [width//2, height*3//4],    # 体の想定位置
        ])
        input_labels = np.array([1, 1, 1])  # すべて前景点
        
        logger.info(f"複数プロンプトテスト")
        logger.info(f"プロンプト点数: {len(input_points)}")
        
        # セグメンテーション実行
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        logger.info(f"✅ 複数プロンプトセグメンテーション成功")
        logger.info(f"   生成マスク数: {len(masks)}")
        logger.info(f"   スコア: {scores}")
        
        # 結果保存
        output_dir = project_root / "data" / "output" / "sam2_test"
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx].astype(bool)  # boolean型に変換
        
        # プロンプト点も表示した結果画像
        result_image = image.copy()
        result_image[best_mask] = result_image[best_mask] * 0.7 + np.array([0, 255, 0]) * 0.3
        
        # プロンプト点を描画
        for i, (x, y) in enumerate(input_points):
            cv2.circle(result_image, (int(x), int(y)), 5, (255, 255, 0), -1)
            cv2.putText(result_image, str(i+1), (int(x)+10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        result_bgr = cv2.cvtColor(result_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"sam2_multi_prompt_{test_image_path.stem}.png"), result_bgr)
        
        logger.info(f"複数プロンプト結果保存: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 複数プロンプトテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行"""
    
    # ログ設定
    setup_logging(level="INFO", console_output=True, structured=False)
    logger = get_logger(__name__)
    
    logger.info("=== SAM2基本動作テスト ===")
    
    try:
        # 1. インポートテスト
        logger.info("\n1. SAM2インポートテスト")
        if not test_sam2_import():
            return
        
        # 2. 基本セグメンテーションテスト
        logger.info("\n2. 基本セグメンテーションテスト")
        if not test_sam2_segmentation():
            return
        
        # 3. 複数プロンプトテスト
        logger.info("\n3. 複数プロンプトテスト")
        if not test_sam2_multiple_prompts():
            return
        
        logger.info("\n🎉 すべてのSAM2テストが成功しました！")
        logger.info("結果は data/output/sam2_test/ で確認できます")
        
    except Exception as e:
        logger.error(f"テスト実行中にエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()