#!/usr/bin/env python3
"""
SAM2詳細パーツセグメンテーションテスト

Live2D用の細かいパーツ（角・鎖・首輪・目・眉・口など）を個別にセグメンテーション
"""

import sys
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils import setup_logging, get_logger
from src.core.sam2.sam2_model import SAM2ModelManager
from src.core.sam2.prompt_handler import SAM2PromptHandler, PointPrompt

@dataclass
class MainPartPrompts:
    """主要パーツ用プロンプトセット"""
    
    @staticmethod
    def hair(image_shape: Tuple[int, int]) -> List[PointPrompt]:
        """髪の毛セグメンテーション用プロンプト"""
        height, width = image_shape[:2]
        
        return [
            PointPrompt(width//2, height//4, 1, description="髪_中央"),
            PointPrompt(width//4, height//3, 1, description="髪_左"),
            PointPrompt(width*3//4, height//3, 1, description="髪_右"),
            PointPrompt(width//2, height//6, 1, description="髪_上部"),
            PointPrompt(width//6, height//2, 1, description="髪_左サイド"),
            PointPrompt(width*5//6, height//2, 1, description="髪_右サイド"),
            # 背景除外プロンプト
            PointPrompt(width//2, height//2, 0, description="顔_背景"),
            PointPrompt(width//2, height*2//3, 0, description="体_背景"),
        ]
    
    @staticmethod
    def horns(image_shape: Tuple[int, int]) -> List[PointPrompt]:
        """角セグメンテーション用プロンプト"""
        height, width = image_shape[:2]
        
        return [
            PointPrompt(width//2 - 60, height//8, 1, description="左角"),
            PointPrompt(width//2 + 60, height//8, 1, description="右角"),
            PointPrompt(width//2, height//10, 1, description="角_中央"),
            PointPrompt(width//2 - 80, height//6, 1, description="左角_先端"),
            PointPrompt(width//2 + 80, height//6, 1, description="右角_先端"),
            # 背景除外プロンプト
            PointPrompt(width//2, height//4, 0, description="髪_背景"),
            PointPrompt(width//4, height//5, 0, description="左側_背景"),
            PointPrompt(width*3//4, height//5, 0, description="右側_背景"),
        ]
    
    @staticmethod
    def face(image_shape: Tuple[int, int]) -> List[PointPrompt]:
        """顔セグメンテーション用プロンプト"""
        height, width = image_shape[:2]
        
        return [
            PointPrompt(width//2, height//2, 1, description="顔_中央"),
            PointPrompt(width*2//5, height*2//5, 1, description="左目"),
            PointPrompt(width*3//5, height*2//5, 1, description="右目"),
            PointPrompt(width//2, height//2 + 40, 1, description="口"),
            PointPrompt(width//2, height//2 + 10, 1, description="鼻"),
            PointPrompt(width//3, height//2, 1, description="左頬"),
            PointPrompt(width*2//3, height//2, 1, description="右頬"),
            # 背景除外プロンプト
            PointPrompt(width//2, height//3, 0, description="髪_背景"),
            PointPrompt(width//2, height*3//5, 0, description="首輪_背景"),
        ]
    
    @staticmethod
    def body(image_shape: Tuple[int, int]) -> List[PointPrompt]:
        """身体セグメンテーション用プロンプト"""
        height, width = image_shape[:2]
        
        return [
            PointPrompt(width//2, height*2//3, 1, description="体_中央"),
            PointPrompt(width//3, height*3//4, 1, description="体_左"),
            PointPrompt(width*2//3, height*3//4, 1, description="体_右"),
            PointPrompt(width//2, height*3//4, 1, description="胸部"),
            PointPrompt(width//2, height*5//6, 1, description="下半身"),
            # 背景除外プロンプト
            PointPrompt(width//2, height//2, 0, description="顔_背景"),
            PointPrompt(width//4, height*2//3, 0, description="左腕_背景"),
            PointPrompt(width*3//4, height*2//3, 0, description="右腕_背景"),
        ]
    
    @staticmethod
    def collar(image_shape: Tuple[int, int]) -> List[PointPrompt]:
        """首輪セグメンテーション用プロンプト"""
        height, width = image_shape[:2]
        
        return [
            PointPrompt(width//2, height*11//24, 1, description="首輪_中央"),
            PointPrompt(width//2 - 50, height*11//24, 1, description="首輪_左"),
            PointPrompt(width//2 + 50, height*11//24, 1, description="首輪_右"),
            PointPrompt(width//2, height*11//24 - 8, 1, description="首輪_上"),
            PointPrompt(width//2, height*11//24 + 8, 1, description="首輪_下"),
            PointPrompt(width//2, height*2//5, 1, description="鎖"),
            # 背景除外プロンプト
            PointPrompt(width//2, height*2//5 - 20, 0, description="顔_背景"),
            PointPrompt(width//2, height//2, 0, description="体_背景"),
        ]

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

def test_detailed_part_segmentation(
    image_rgb: np.ndarray,
    part_name: str,
    prompts: List[PointPrompt],
    sam2_manager: SAM2ModelManager,
    output_dir: Path,
    image_name: str
) -> Dict[str, Any]:
    """詳細パーツの個別セグメンテーションテスト"""
    logger = get_logger(__name__)
    
    try:
        logger.info(f"  🎯 {part_name} セグメンテーション")
        
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
        
        # 結果分析
        best_mask_idx = np.argmax(scores)
        best_score = float(scores[best_mask_idx])
        best_mask = masks[best_mask_idx].astype(bool)
        mask_coverage = float(np.sum(best_mask) / best_mask.size)
        
        # 可視化画像作成
        overlay = image_rgb.copy().astype(np.float32)
        
        # パーツ別色分け
        part_colors = {
            "hair": [255, 150, 0],       # オレンジ
            "horns": [255, 200, 0],      # 金色
            "face": [255, 200, 150],     # 肌色
            "body": [100, 150, 255],     # 青系
            "collar": [150, 150, 150],   # シルバー
        }
        
        color = np.array(part_colors.get(part_name, [255, 255, 100]))
        
        # マスクオーバーレイ
        overlay[best_mask] = overlay[best_mask] * 0.5 + color * 0.5
        overlay_image = overlay.astype(np.uint8)
        
        # プロンプト点を描画
        prompt_coords = sam2_prompts["point_coords"]
        prompt_labels = sam2_prompts["point_labels"]
        
        if prompt_coords is not None:
            for i, ((x, y), label) in enumerate(zip(prompt_coords, prompt_labels)):
                # 前景点は緑、背景点は赤
                point_color = (0, 255, 0) if label == 1 else (255, 0, 0)
                cv2.circle(overlay_image, (int(x), int(y)), 6, point_color, -1)
                cv2.circle(overlay_image, (int(x), int(y)), 8, (255, 255, 255), 2)
                
                # プロンプト番号
                add_text_with_background(
                    overlay_image, str(i+1),
                    (int(x)+12, int(y)),
                    font_scale=0.5,
                    text_color=(255, 255, 255),
                    bg_color=(0, 0, 0)
                )
        
        # 結果情報を画像に表示
        info_lines = [
            f"Part: {part_name}",
            f"Score: {best_score:.3f}",
            f"Time: {inference_time:.2f}s",
            f"Coverage: {mask_coverage*100:.1f}%",
            f"Prompts: {len(prompts)}"
        ]
        
        for i, line in enumerate(info_lines):
            add_text_with_background(
                overlay_image, line,
                (10, 25 + i * 25),
                font_scale=0.6,
                text_color=(255, 255, 255),
                bg_color=(0, 0, 0)
            )
        
        # 画像保存
        output_path = output_dir / f"{part_name}_{image_name}.png"
        overlay_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), overlay_bgr)
        
        logger.info(f"    スコア: {best_score:.3f}, カバレッジ: {mask_coverage*100:.1f}%, 時間: {inference_time:.2f}s")
        
        return {
            "part_name": part_name,
            "score": best_score,
            "inference_time": inference_time,
            "mask_coverage": mask_coverage,
            "num_prompts": len(prompts),
            "output_path": str(output_path)
        }
        
    except Exception as e:
        logger.error(f"パーツセグメンテーション失敗 {part_name}: {e}")
        return {"error": str(e)}

def create_summary_grid(
    image_rgb: np.ndarray,
    all_results: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """全パーツの結果サマリーグリッドを作成"""
    logger = get_logger(__name__)
    
    try:
        height, width = image_rgb.shape[:2]
        
        # 2x3グリッド（5パーツ + 1元画像）
        grid_rows, grid_cols = 2, 3
        grid_image = np.zeros((height * grid_rows, width * grid_cols, 3), dtype=np.uint8)
        
        # 元画像を左上に配置
        grid_image[0:height, 0:width] = image_rgb
        
        # 元画像にタイトル追加
        title_image = image_rgb.copy()
        add_text_with_background(
            title_image, "Original Image",
            (10, 30),
            font_scale=1.0,
            text_color=(255, 255, 255),
            bg_color=(0, 0, 0)
        )
        grid_image[0:height, 0:width] = title_image
        
        # 各パーツ結果を配置
        positions = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                if row == 0 and col == 0:  # 元画像スキップ
                    continue
                positions.append((row, col))
        
        for idx, result in enumerate(all_results[:5]):  # 最大5パーツ
            if "error" in result:
                continue
                
            row, col = positions[idx]
            y_start, y_end = row * height, (row + 1) * height
            x_start, x_end = col * width, (col + 1) * width
            
            # パーツ画像読み込み
            if "output_path" in result:
                part_image_path = Path(result["output_path"])
                if part_image_path.exists():
                    part_image = cv2.imread(str(part_image_path))
                    part_image_rgb = cv2.cvtColor(part_image, cv2.COLOR_BGR2RGB)
                    
                    # グリッドに配置
                    grid_image[y_start:y_end, x_start:x_end] = part_image_rgb
        
        # 保存
        grid_bgr = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), grid_bgr)
        logger.info(f"サマリーグリッド保存: {output_path}")
        
    except Exception as e:
        logger.error(f"サマリーグリッド作成失敗: {e}")

def main():
    """メイン実行"""
    
    # ログ設定
    setup_logging(level="INFO", console_output=True, structured=False)
    logger = get_logger(__name__)
    
    logger.info("=== SAM2詳細パーツセグメンテーションテスト ===")
    
    try:
        # テスト画像取得
        sample_images_path = project_root / "data" / "samples" / "demo_images"
        image_files = list(sample_images_path.glob("*.png"))
        
        if not image_files:
            logger.error("❌ テスト画像が見つかりません")
            return
        
        # 出力ディレクトリ準備
        output_dir = project_root / "data" / "output" / "sam2_detailed_parts"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # SAM2モデル初期化（最高精度のlargeモデルを使用）
        sam2_manager = SAM2ModelManager(model_name="sam2_hiera_large.pt")
        if not sam2_manager.load_model():
            logger.error("❌ SAM2モデル読み込み失敗")
            return
        
        all_images_results = []
        
        # 全ての画像でテスト
        for img_idx, image_path in enumerate(image_files):
            logger.info(f"\\n📸 画像 {img_idx + 1}/{len(image_files)}: {image_path.name}")
            
            # 画像読み込み
            image = cv2.imread(str(image_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            
            logger.info(f"   画像サイズ: {width}x{height}")
            
            # 主要パーツ定義（5パーツ）
            main_parts = {
                "hair": MainPartPrompts.hair((height, width)),
                "horns": MainPartPrompts.horns((height, width)),
                "face": MainPartPrompts.face((height, width)),
                "body": MainPartPrompts.body((height, width)),
                "collar": MainPartPrompts.collar((height, width)),
            }
            
            image_results = []
            
            # 各パーツをテスト
            for part_name, prompts in main_parts.items():
                result = test_detailed_part_segmentation(
                    image_rgb, part_name, prompts, sam2_manager, 
                    output_dir, image_path.stem
                )
                
                if "error" not in result:
                    result["image_name"] = image_path.stem
                    image_results.append(result)
            
            all_images_results.extend(image_results)
            
            # 画像別サマリーグリッド作成
            summary_path = output_dir / f"detailed_parts_summary_{image_path.stem}.png"
            create_summary_grid(image_rgb, image_results, summary_path)
        
        # 全体結果サマリー表示
        logger.info(f"\\n📊 全画像での詳細パーツセグメンテーション結果:")
        
        # パーツ別平均スコア計算
        part_stats = {}
        for result in all_images_results:
            part_name = result["part_name"]
            if part_name not in part_stats:
                part_stats[part_name] = {"scores": [], "times": [], "coverages": []}
            
            part_stats[part_name]["scores"].append(result["score"])
            part_stats[part_name]["times"].append(result["inference_time"])
            part_stats[part_name]["coverages"].append(result["mask_coverage"])
        
        # パーツ別統計表示
        logger.info(f"\\n📈 パーツ別統計（{len(image_files)}画像平均）:")
        sorted_parts = sorted(part_stats.items(), 
                            key=lambda x: np.mean(x[1]["scores"]), reverse=True)
        
        for part_name, stats in sorted_parts:
            avg_score = np.mean(stats["scores"])
            std_score = np.std(stats["scores"])
            avg_time = np.mean(stats["times"])
            avg_coverage = np.mean(stats["coverages"]) * 100
            
            logger.info(f"  {part_name:15} - スコア: {avg_score:.3f}(±{std_score:.3f}), "
                      f"カバレッジ: {avg_coverage:4.1f}%, 時間: {avg_time:.2f}s")
        
        # 全体統計情報
        all_scores = [r["score"] for r in all_images_results]
        all_times = [r["inference_time"] for r in all_images_results]
        
        logger.info(f"\\n📊 全体統計情報:")
        logger.info(f"  総テスト数: {len(all_images_results)} ({len(image_files)}画像 x 5パーツ)")
        logger.info(f"  平均スコア: {np.mean(all_scores):.3f} (±{np.std(all_scores):.3f})")
        logger.info(f"  最高スコア: {np.max(all_scores):.3f}")
        logger.info(f"  最低スコア: {np.min(all_scores):.3f}")
        logger.info(f"  平均処理時間: {np.mean(all_times):.2f}s")
        logger.info(f"  総処理時間: {np.sum(all_times):.1f}s")
        
        # 高性能パーツ特定
        high_score_parts = [part for part, stats in part_stats.items() 
                           if np.mean(stats["scores"]) >= 0.9]
        
        if high_score_parts:
            logger.info(f"\\n🏆 高精度パーツ (スコア≥0.9): {', '.join(high_score_parts)}")
        
        logger.info(f"\\n🎉 全画像での主要パーツテスト完了!")
        logger.info(f"結果: {output_dir}")
        logger.info(f"  - 個別パーツ: [part_name]_[image_name].png")
        logger.info(f"  - 画像別サマリー: detailed_parts_summary_[image_name].png")
        logger.info(f"  - テスト対象: 髪の毛、角、顔、身体、首輪")
        
    except Exception as e:
        logger.error(f"テスト実行中にエラー: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # メモリクリア
        if 'sam2_manager' in locals():
            sam2_manager.unload_model()

if __name__ == "__main__":
    main()