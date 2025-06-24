#!/usr/bin/env python3
"""
SAM2プロンプトハンドラーテスト

Live2D用のプロンプトベースセグメンテーションをテスト
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils import setup_logging, get_logger
from src.core.sam2.sam2_model import SAM2ModelManager
from src.core.sam2.prompt_handler import SAM2PromptHandler, Live2DPromptPresets

def test_prompt_handler_basic():
    """プロンプトハンドラーの基本機能テスト"""
    logger = get_logger(__name__)
    
    try:
        handler = SAM2PromptHandler()
        
        # セッション開始
        session_id = handler.start_new_session()
        logger.info(f"✅ セッション開始: {session_id}")
        
        # 点プロンプト追加
        handler.add_point_prompt(100, 150, 1, description="髪の毛")
        handler.add_point_prompt(200, 250, 1, description="顔")
        handler.add_point_prompt(50, 50, 0, description="背景")
        
        # ボックスプロンプト追加
        handler.add_box_prompt(80, 80, 300, 400, description="キャラクター全体")
        
        # プロンプト数確認
        summary = handler.get_prompt_summary()
        logger.info(f"プロンプト数: {summary}")
        
        # SAM2形式に変換
        sam2_prompts = handler.get_sam2_prompts()
        logger.info(f"点座標: {sam2_prompts['point_coords']}")
        logger.info(f"点ラベル: {sam2_prompts['point_labels']}")
        logger.info(f"ボックス: {sam2_prompts['box']}")
        
        logger.info("✅ プロンプトハンドラー基本テスト成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ プロンプトハンドラー基本テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_live2d_presets():
    """Live2Dプリセットテスト"""
    logger = get_logger(__name__)
    
    try:
        # 仮想画像サイズ
        image_shape = (800, 600)  # height, width
        
        # 各部位のプリセット取得
        hair_prompts = Live2DPromptPresets.hair_segmentation(image_shape)
        face_prompts = Live2DPromptPresets.face_segmentation(image_shape)
        body_prompts = Live2DPromptPresets.body_segmentation(image_shape)
        accessory_prompts = Live2DPromptPresets.accessories_segmentation(image_shape)
        
        logger.info(f"髪の毛プロンプト数: {len(hair_prompts)}")
        logger.info(f"顔プロンプト数: {len(face_prompts)}")
        logger.info(f"体プロンプト数: {len(body_prompts)}")
        logger.info(f"アクセサリープロンプト数: {len(accessory_prompts)}")
        
        # プロンプト詳細確認
        for i, prompt in enumerate(hair_prompts):
            logger.info(f"  髪 {i+1}: ({prompt.x}, {prompt.y}) - {prompt.description}")
        
        logger.info("✅ Live2Dプリセットテスト成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ Live2Dプリセットテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sam2_with_prompts():
    """SAM2とプロンプトハンドラーの統合テスト"""
    logger = get_logger(__name__)
    
    try:
        # サンプル画像取得
        sample_images_path = project_root / "data" / "samples" / "demo_images"
        image_files = list(sample_images_path.glob("*.png"))
        
        if not image_files:
            logger.error("❌ テスト画像が見つかりません")
            return False
        
        test_image_path = image_files[0]
        logger.info(f"テスト画像: {test_image_path.name}")
        
        # 画像読み込み
        image = cv2.imread(str(test_image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        
        logger.info(f"画像サイズ: {width}x{height}")
        
        # SAM2モデル初期化
        sam2_manager = SAM2ModelManager()
        if not sam2_manager.load_model():
            logger.error("❌ SAM2モデル読み込み失敗")
            return False
        
        # プロンプトハンドラー初期化
        handler = SAM2PromptHandler()
        session_id = handler.start_new_session(image_path=test_image_path)
        
        # Live2D用プロンプト設定（髪の毛をターゲット）
        hair_prompts = Live2DPromptPresets.hair_segmentation((height, width))
        
        # プロンプト追加
        for prompt in hair_prompts:
            handler.add_point_prompt(
                prompt.x, prompt.y, prompt.label, 
                description=prompt.description
            )
        
        # SAM2プロンプト取得
        sam2_prompts = handler.get_sam2_prompts()
        
        logger.info(f"プロンプト設定完了: {handler.get_prompt_summary()}")
        
        # SAM2セグメンテーション実行
        masks, scores, logits = sam2_manager.predict(
            image=image_rgb,
            **sam2_prompts,
            multimask_output=True
        )
        
        logger.info(f"✅ セグメンテーション成功")
        logger.info(f"   生成マスク数: {len(masks)}")
        logger.info(f"   スコア: {scores}")
        
        # 結果保存
        output_dir = project_root / "data" / "output" / "sam2_prompts_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # プロンプト可視化
        prompt_vis = handler.visualize_prompts(
            image_rgb, 
            output_path=output_dir / f"prompts_{test_image_path.stem}.png"
        )
        
        # 最高スコアのマスクで結果画像作成
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx].astype(bool)
        
        # マスク保存
        mask_image = (best_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(output_dir / f"hair_mask_{test_image_path.stem}.png"), mask_image)
        
        # オーバーレイ画像作成
        overlay = image_rgb.copy().astype(np.float32)
        overlay[best_mask] = overlay[best_mask] * 0.6 + np.array([255, 100, 100]) * 0.4
        
        overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"hair_overlay_{test_image_path.stem}.png"), overlay_bgr)
        
        logger.info(f"結果保存: {output_dir}")
        logger.info("✅ SAM2プロンプト統合テスト成功")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SAM2プロンプト統合テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_part_segmentation():
    """複数部位セグメンテーションテスト"""
    logger = get_logger(__name__)
    
    try:
        # テスト画像取得
        sample_images_path = project_root / "data" / "samples" / "demo_images"
        image_files = list(sample_images_path.glob("*.png"))
        test_image_path = image_files[0]
        
        image = cv2.imread(str(test_image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        
        # SAM2モデル
        sam2_manager = SAM2ModelManager()
        sam2_manager.load_model()
        
        # 出力ディレクトリ
        output_dir = project_root / "data" / "output" / "sam2_multipart_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 各部位をセグメンテーション
        parts = {
            "hair": Live2DPromptPresets.hair_segmentation((height, width)),
            "face": Live2DPromptPresets.face_segmentation((height, width)),
            "body": Live2DPromptPresets.body_segmentation((height, width)),
        }
        
        part_masks = {}
        colors = {
            "hair": [255, 100, 100],  # 赤系
            "face": [100, 255, 100],  # 緑系
            "body": [100, 100, 255],  # 青系
        }
        
        for part_name, prompts in parts.items():
            logger.info(f"\\n{part_name}セグメンテーション開始")
            
            # プロンプトハンドラー設定
            handler = SAM2PromptHandler()
            handler.start_new_session()
            
            for prompt in prompts:
                handler.add_point_prompt(
                    prompt.x, prompt.y, prompt.label,
                    description=prompt.description
                )
            
            # セグメンテーション実行
            sam2_prompts = handler.get_sam2_prompts()
            masks, scores, logits = sam2_manager.predict(
                image=image_rgb,
                **sam2_prompts,
                multimask_output=True
            )
            
            # 最良マスク保存
            best_mask = masks[np.argmax(scores)].astype(bool)
            part_masks[part_name] = best_mask
            
            # 個別マスク保存
            mask_image = (best_mask.astype(np.uint8) * 255)
            cv2.imwrite(str(output_dir / f"{part_name}_mask.png"), mask_image)
            
            logger.info(f"✅ {part_name} セグメンテーション完了 (スコア: {np.max(scores):.3f})")
        
        # 全部位統合画像作成
        combined_overlay = image_rgb.copy().astype(np.float32)
        
        for part_name, mask in part_masks.items():
            color = np.array(colors[part_name])
            combined_overlay[mask] = combined_overlay[mask] * 0.7 + color * 0.3
        
        # 統合結果保存
        combined_bgr = cv2.cvtColor(combined_overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / "combined_segmentation.png"), combined_bgr)
        
        logger.info(f"\\n🎉 複数部位セグメンテーション完了!")
        logger.info(f"結果: {output_dir}")
        logger.info(f"セグメンテーション部位: {list(parts.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 複数部位セグメンテーションテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行"""
    
    # ログ設定
    setup_logging(level="INFO", console_output=True, structured=False)
    logger = get_logger(__name__)
    
    logger.info("=== SAM2プロンプトハンドラーテスト ===")
    
    try:
        # 1. プロンプトハンドラー基本機能テスト
        logger.info("\\n1. プロンプトハンドラー基本テスト")
        if not test_prompt_handler_basic():
            return
        
        # 2. Live2Dプリセットテスト
        logger.info("\\n2. Live2Dプリセットテスト")
        if not test_live2d_presets():
            return
        
        # 3. SAM2統合テスト
        logger.info("\\n3. SAM2プロンプト統合テスト")
        if not test_sam2_with_prompts():
            return
        
        # 4. 複数部位セグメンテーション
        logger.info("\\n4. 複数部位セグメンテーションテスト")
        if not test_multi_part_segmentation():
            return
        
        logger.info("\\n🎉 すべてのプロンプトテストが成功しました！")
        logger.info("結果は data/output/ で確認できます")
        
    except Exception as e:
        logger.error(f"テスト実行中にエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()