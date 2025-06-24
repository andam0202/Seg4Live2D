#!/usr/bin/env python3
"""
SAM2モデル視覚的比較テスト

各モデルのセグメンテーション結果を視覚的に比較表示
"""

import sys
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils import setup_logging, get_logger
from src.core.sam2.sam2_model import SAM2ModelManager
from src.core.sam2.prompt_handler import SAM2PromptHandler, Live2DPromptPresets

def add_text_with_background(
    image: np.ndarray, 
    text: str, 
    position: Tuple[int, int], 
    font_scale: float = 0.7,
    thickness: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    padding: int = 5
) -> None:
    """背景付きテキストを画像に追加"""
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # テキストサイズを計算
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    
    # 背景矩形を描画
    bg_x1 = x - padding
    bg_y1 = y - text_height - padding
    bg_x2 = x + text_width + padding
    bg_y2 = y + baseline + padding
    
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    
    # テキストを描画
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)

def create_segmentation_overlay(
    image_rgb: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    part_name: str
) -> np.ndarray:
    """セグメンテーション結果のオーバーレイ画像を作成"""
    
    # 最高スコアのマスクを使用
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx].astype(bool)
    best_score = scores[best_mask_idx]
    
    # オーバーレイ画像作成
    overlay = image_rgb.copy().astype(np.float32)
    
    # 部位別の色設定
    colors = {
        "hair": [255, 100, 100],  # 赤系
        "face": [100, 255, 100],  # 緑系
        "body": [100, 100, 255],  # 青系
    }
    
    color = np.array(colors.get(part_name, [255, 255, 100]))
    
    # マスク部分をハイライト
    overlay[best_mask] = overlay[best_mask] * 0.6 + color * 0.4
    
    return overlay.astype(np.uint8)

def test_single_model_with_visualization(
    model_name: str,
    image_path: Path,
    image_rgb: np.ndarray,
    test_prompts: Dict[str, List],
    output_dir: Path
) -> Dict[str, Any]:
    """単一モデルでのテスト + 可視化結果作成"""
    logger = get_logger(__name__)
    
    try:
        # SAM2モデル初期化
        start_time = time.time()
        sam2_manager = SAM2ModelManager(model_name=model_name)
        if not sam2_manager.load_model():
            return {"error": f"モデル読み込み失敗: {model_name}"}
        
        model_load_time = time.time() - start_time
        
        results = {
            "model_name": model_name,
            "image_name": image_path.stem,
            "model_load_time": model_load_time,
            "part_results": {},
            "visualizations": {}
        }
        
        height, width = image_rgb.shape[:2]
        
        # 各部位でテスト
        for part_name, prompts in test_prompts.items():
            logger.info(f"  {part_name} セグメンテーション - {model_name}")
            
            # プロンプトハンドラー設定
            handler = SAM2PromptHandler()
            handler.start_new_session()
            
            for prompt in prompts:
                handler.add_point_prompt(
                    prompt.x, prompt.y, prompt.label,
                    description=prompt.description
                )
            
            # セグメンテーション実行
            start_time = time.time()
            sam2_prompts = handler.get_sam2_prompts()
            masks, scores, logits = sam2_manager.predict(
                image=image_rgb,
                **sam2_prompts,
                multimask_output=True
            )
            inference_time = time.time() - start_time
            
            # 結果記録
            best_score = float(np.max(scores))
            best_mask = masks[np.argmax(scores)].astype(bool)
            mask_coverage = float(np.sum(best_mask) / best_mask.size)
            
            results["part_results"][part_name] = {
                "score": best_score,
                "inference_time": inference_time,
                "mask_coverage": mask_coverage,
                "num_masks": len(masks)
            }
            
            # セグメンテーション結果の可視化
            overlay_image = create_segmentation_overlay(image_rgb, masks, scores, part_name)
            
            # プロンプト点を描画
            prompt_coords = sam2_prompts["point_coords"]
            if prompt_coords is not None:
                for i, (x, y) in enumerate(prompt_coords):
                    cv2.circle(overlay_image, (int(x), int(y)), 8, (255, 255, 0), -1)
                    cv2.circle(overlay_image, (int(x), int(y)), 10, (0, 0, 0), 2)
                    
                    # プロンプト番号
                    add_text_with_background(
                        overlay_image, str(i+1), 
                        (int(x)+15, int(y)), 
                        font_scale=0.6, 
                        text_color=(255, 255, 0),
                        bg_color=(0, 0, 0)
                    )
            
            # スコア表示
            score_text = f"{part_name}: {best_score:.3f}"
            add_text_with_background(
                overlay_image, score_text,
                (10, 30 + list(test_prompts.keys()).index(part_name) * 35),
                font_scale=0.8,
                text_color=(255, 255, 255),
                bg_color=(0, 0, 0)
            )
            
            results["visualizations"][part_name] = overlay_image
            
            logger.info(f"    スコア: {best_score:.3f}, 時間: {inference_time:.2f}s")
        
        # メモリクリア
        sam2_manager.unload_model()
        
        return results
        
    except Exception as e:
        logger.error(f"モデルテスト失敗 {model_name}: {e}")
        return {"error": str(e)}

def create_model_comparison_grid(
    image_rgb: np.ndarray,
    all_results: List[Dict[str, Any]],
    part_name: str,
    output_path: Path
) -> None:
    """モデル比較グリッド画像を作成"""
    logger = get_logger(__name__)
    
    try:
        models = ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
        model_short_names = ["Tiny", "Small", "Base+", "Large"]
        
        # 2x2グリッドで結果表示
        height, width = image_rgb.shape[:2]
        grid_image = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
        
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        for idx, (model_name, short_name, pos) in enumerate(zip(models, model_short_names, positions)):
            row, col = pos
            y_start, y_end = row * height, (row + 1) * height
            x_start, x_end = col * width, (col + 1) * width
            
            # 対応する結果を検索
            model_result = None
            for result in all_results:
                if result.get("model_name") == model_name and "error" not in result:
                    model_result = result
                    break
            
            if model_result and part_name in model_result.get("visualizations", {}):
                # セグメンテーション結果画像を使用
                model_image = model_result["visualizations"][part_name].copy()
                
                # モデル名とスコアをタイトルバーに表示
                score = model_result["part_results"][part_name]["score"]
                inference_time = model_result["part_results"][part_name]["inference_time"]
                
                title_text = f"{short_name} - Score: {score:.3f}"
                time_text = f"Time: {inference_time:.2f}s"
                
                # タイトル背景
                title_bg_height = 80
                cv2.rectangle(model_image, (0, 0), (width, title_bg_height), (0, 0, 0), -1)
                
                # タイトルテキスト
                add_text_with_background(
                    model_image, title_text,
                    (10, 25),
                    font_scale=0.8,
                    text_color=(255, 255, 255),
                    bg_color=(0, 0, 0),
                    padding=0
                )
                
                add_text_with_background(
                    model_image, time_text,
                    (10, 55),
                    font_scale=0.6,
                    text_color=(200, 200, 200),
                    bg_color=(0, 0, 0),
                    padding=0
                )
                
            else:
                # エラーまたは結果なしの場合
                model_image = image_rgb.copy()
                
                # エラー表示
                error_text = f"{short_name} - ERROR"
                add_text_with_background(
                    model_image, error_text,
                    (width//2 - 100, height//2),
                    font_scale=1.0,
                    text_color=(255, 255, 255),
                    bg_color=(0, 0, 255)
                )
            
            # グリッドに配置
            grid_image[y_start:y_end, x_start:x_end] = model_image
        
        # 保存
        grid_bgr = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), grid_bgr)
        logger.info(f"比較グリッド保存: {output_path}")
        
    except Exception as e:
        logger.error(f"グリッド作成失敗: {e}")

def main():
    """メイン実行"""
    
    # ログ設定
    setup_logging(level="INFO", console_output=True, structured=False)
    logger = get_logger(__name__)
    
    logger.info("=== SAM2モデル視覚的比較テスト ===")
    
    try:
        # テスト画像取得
        sample_images_path = project_root / "data" / "samples" / "demo_images"
        image_files = list(sample_images_path.glob("*.png"))
        
        if not image_files:
            logger.error("❌ テスト画像が見つかりません")
            return
        
        # 出力ディレクトリ準備
        output_dir = project_root / "data" / "output" / "sam2_visual_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # テストするモデル（高速化のため2つに限定）
        models_to_test = [
            "sam2_hiera_tiny.pt",
            "sam2_hiera_small.pt", 
            "sam2_hiera_base_plus.pt",
            "sam2_hiera_large.pt"
        ]
        
        # 最初の2枚の画像でテスト
        for img_idx, image_path in enumerate(image_files[:2]):
            logger.info(f"\\n📸 画像 {img_idx + 1}/2: {image_path.name}")
            
            # 画像読み込み
            image = cv2.imread(str(image_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            
            logger.info(f"   画像サイズ: {width}x{height}")
            
            # Live2Dプロンプト準備
            test_prompts = {
                "hair": Live2DPromptPresets.hair_segmentation((height, width)),
                "face": Live2DPromptPresets.face_segmentation((height, width)),
                "body": Live2DPromptPresets.body_segmentation((height, width))
            }
            
            image_results = []
            
            # 各モデルでテスト
            for model_name in models_to_test:
                logger.info(f"  🤖 モデル: {model_name}")
                
                result = test_single_model_with_visualization(
                    model_name, image_path, image_rgb, test_prompts, output_dir
                )
                
                if "error" not in result:
                    image_results.append(result)
                    
                    # 各部位の結果表示
                    for part_name, part_result in result["part_results"].items():
                        logger.info(f"    {part_name}: スコア {part_result['score']:.3f}, "
                                  f"時間 {part_result['inference_time']:.2f}s")
                else:
                    logger.error(f"    ❌ {result['error']}")
            
            # 各部位別の比較グリッド作成
            if image_results:
                for part_name in ["hair", "face", "body"]:
                    grid_path = output_dir / f"{part_name}_comparison_{image_path.stem}.png"
                    create_model_comparison_grid(image_rgb, image_results, part_name, grid_path)
        
        # 結果サマリー表示
        logger.info(f"\\n🎉 視覚的比較テスト完了!")
        logger.info(f"結果: {output_dir}")
        logger.info(f"  - 部位別比較: [part]_comparison_[image].png")
        logger.info(f"  - hair: 髪の毛セグメンテーション比較")
        logger.info(f"  - face: 顔セグメンテーション比較") 
        logger.info(f"  - body: 体セグメンテーション比較")
        
    except Exception as e:
        logger.error(f"テスト実行中にエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()