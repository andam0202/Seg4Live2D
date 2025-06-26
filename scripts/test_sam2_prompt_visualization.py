#!/usr/bin/env python3
"""
SAM2プロンプト可視化テスト

プロンプト点を可視化して、各戦略の違いを明確に表示
"""

import sys
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils import setup_logging, get_logger
from src.core.sam2.sam2_model import SAM2ModelManager
from src.core.sam2.prompt_handler import SAM2PromptHandler, PointPrompt

# ハイブリッド手法をインポート
from scripts.test_sam2_hybrid_enhancement import SAM2HybridEnhancer
from scripts.test_sam2_comprehensive_prompts import ComprehensivePromptTester

class PromptVisualizer:
    """プロンプト可視化クラス"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
    def draw_prompts_on_image(self, image: np.ndarray, prompts: List[PointPrompt], 
                            title: str = "") -> np.ndarray:
        """プロンプト点を画像上に描画"""
        
        # 画像をコピー
        vis_image = image.copy()
        
        # プロンプト点を描画
        for i, prompt in enumerate(prompts):
            x, y = int(prompt.x), int(prompt.y)
            
            # 正例（緑）と負例（赤）で色分け
            if prompt.label == 1:  # 正例
                color = (0, 255, 0)  # 緑
                marker = "+"
            else:  # 負例
                color = (255, 0, 0)  # 赤  
                marker = "-"
            
            # 円を描画
            cv2.circle(vis_image, (x, y), 8, color, -1)  # 塗りつぶし円
            cv2.circle(vis_image, (x, y), 12, (255, 255, 255), 2)  # 白い縁
            
            # 番号と説明を描画
            font_scale = 0.4
            font_thickness = 1
            
            # 番号
            number_text = f"{i+1}"
            number_size = cv2.getTextSize(number_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            number_x = x - number_size[0] // 2
            number_y = y + number_size[1] // 2
            
            # 説明文
            description = prompt.description if prompt.description else f"Point_{i+1}"
            desc_size = cv2.getTextSize(description, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            
            # 説明文の位置（点の下側に配置）
            desc_x = x - desc_size[0] // 2
            desc_y = y + 25 + desc_size[1]
            
            # 番号の背景（円の中央）
            cv2.rectangle(vis_image, 
                         (number_x - 2, number_y - number_size[1] - 2), 
                         (number_x + number_size[0] + 2, number_y + 2), 
                         (0, 0, 0), -1)
            cv2.putText(vis_image, number_text, (number_x, number_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            
            # 説明文の背景
            padding = 3
            cv2.rectangle(vis_image, 
                         (desc_x - padding, desc_y - desc_size[1] - padding), 
                         (desc_x + desc_size[0] + padding, desc_y + padding), 
                         (0, 0, 0), -1)  # 黒い背景
            
            # 説明文のテキスト（正例は緑、負例は赤）
            text_color = (0, 255, 0) if prompt.label == 1 else (255, 0, 0)
            cv2.putText(vis_image, description, (desc_x, desc_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
        
        # タイトルを描画
        if title:
            title_lines = [
                f"Strategy: {title}",
                f"Prompts: {len(prompts)} points",
                f"Positive: {sum(1 for p in prompts if p.label == 1)}",
                f"Negative: {sum(1 for p in prompts if p.label == 0)}"
            ]
            
            for i, line in enumerate(title_lines):
                y_pos = 30 + i * 25
                # 黒い背景
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(vis_image, (10, y_pos - 20), (text_size[0] + 20, y_pos + 5), (0, 0, 0), -1)
                # 白い文字
                cv2.putText(vis_image, line, (15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image
    
    def create_comparison_image(self, image: np.ndarray, 
                              prompt_strategies: Dict[str, List[PointPrompt]],
                              part_name: str) -> np.ndarray:
        """複数戦略の比較画像を作成"""
        
        strategy_names = list(prompt_strategies.keys())
        num_strategies = len(strategy_names)
        
        if num_strategies == 0:
            return image
        
        # 2×3または3×2のグリッドレイアウト
        if num_strategies <= 3:
            rows, cols = 1, num_strategies
        else:
            rows, cols = 2, (num_strategies + 1) // 2
        
        # 各画像のサイズを調整
        img_height, img_width = image.shape[:2]
        cell_width = img_width // cols
        cell_height = img_height // rows
        
        # 比較画像作成
        comparison_image = np.zeros((rows * cell_height, cols * cell_width, 3), dtype=np.uint8)
        
        for i, strategy_name in enumerate(strategy_names):
            prompts = prompt_strategies[strategy_name]
            
            # 位置計算
            row = i // cols
            col = i % cols
            
            # 画像リサイズ
            resized_image = cv2.resize(image, (cell_width, cell_height))
            
            # プロンプト座標もリサイズに合わせて調整
            scale_x = cell_width / img_width
            scale_y = cell_height / img_height
            
            scaled_prompts = []
            for prompt in prompts:
                scaled_prompt = PointPrompt(
                    int(prompt.x * scale_x),
                    int(prompt.y * scale_y),
                    prompt.label,
                    prompt.description
                )
                scaled_prompts.append(scaled_prompt)
            
            # プロンプトを描画
            vis_image = self.draw_prompts_on_image(resized_image, scaled_prompts, strategy_name)
            
            # 比較画像に配置
            y_start = row * cell_height
            y_end = y_start + cell_height
            x_start = col * cell_width
            x_end = x_start + cell_width
            
            comparison_image[y_start:y_end, x_start:x_end] = vis_image
        
        # 全体タイトル
        title_height = 50
        final_image = np.zeros((comparison_image.shape[0] + title_height, comparison_image.shape[1], 3), dtype=np.uint8)
        final_image[title_height:, :] = comparison_image
        
        # タイトル描画
        title_text = f"Prompt Strategies Comparison - {part_name.upper()} Part"
        text_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (final_image.shape[1] - text_size[0]) // 2
        text_y = 35
        
        cv2.putText(final_image, title_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return final_image

def test_prompt_visualization(
    image_rgb: np.ndarray,
    part_name: str,
    tester: ComprehensivePromptTester,
    output_dir: Path,
    image_name: str
) -> Dict[str, Any]:
    """プロンプト可視化テスト"""
    logger = get_logger(__name__)
    visualizer = PromptVisualizer()
    
    logger.info(f"🎯 プロンプト可視化テスト開始 - {part_name}")
    
    height, width = image_rgb.shape[:2]
    
    # 各戦略のプロンプトを生成
    strategies_prompts = {
        "basic": tester.generate_basic_prompts((height, width), part_name),
        "dense_grid": tester.generate_dense_grid_prompts((height, width), part_name),
        "anatomical": tester.generate_anatomical_prompts((height, width), part_name),
        "semantic": tester.generate_semantic_prompts((height, width), part_name),
        "adaptive_sparse": tester.generate_adaptive_sparse_prompts((height, width), part_name),
    }
    
    # 各戦略の個別可視化
    individual_images = {}
    for strategy_name, prompts in strategies_prompts.items():
        logger.info(f"  🎨 {strategy_name}戦略可視化: {len(prompts)}点")
        
        vis_image = visualizer.draw_prompts_on_image(image_rgb, prompts, strategy_name)
        individual_images[strategy_name] = vis_image
        
        # 個別画像保存
        individual_path = output_dir / f"prompts_{strategy_name}_{part_name}_{image_name}.png"
        individual_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(individual_path), individual_bgr)
    
    # 比較画像作成
    logger.info(f"  📊 比較画像作成")
    comparison_image = visualizer.create_comparison_image(image_rgb, strategies_prompts, part_name)
    
    # 比較画像保存
    comparison_path = output_dir / f"prompts_comparison_{part_name}_{image_name}.png"
    comparison_bgr = cv2.cvtColor(comparison_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(comparison_path), comparison_bgr)
    
    # 詳細情報保存
    prompt_info = {}
    for strategy_name, prompts in strategies_prompts.items():
        prompt_info[strategy_name] = {
            "total_prompts": len(prompts),
            "positive_prompts": sum(1 for p in prompts if p.label == 1),
            "negative_prompts": sum(1 for p in prompts if p.label == 0),
            "points": [
                {
                    "x": int(p.x), "y": int(p.y), 
                    "label": p.label, "description": p.description
                } for p in prompts
            ]
        }
    
    info_path = output_dir / f"prompts_info_{part_name}_{image_name}.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"  ✅ 可視化完了")
    logger.info(f"    📁 比較画像: {comparison_path}")
    logger.info(f"    📋 詳細情報: {info_path}")
    
    return {
        "part_name": part_name,
        "strategies": list(strategies_prompts.keys()),
        "comparison_image_path": str(comparison_path),
        "individual_images": {k: str(output_dir / f"prompts_{k}_{part_name}_{image_name}.png") 
                            for k in strategies_prompts.keys()},
        "prompt_info_path": str(info_path),
        "total_strategies": len(strategies_prompts)
    }

def main():
    """メイン実行"""
    
    # ログ設定
    setup_logging(level="INFO", console_output=True, structured=False)
    logger = get_logger(__name__)
    
    logger.info("=== SAM2プロンプト可視化テスト ===")
    
    try:
        # 本番画像取得
        sample_images_path = project_root / "data" / "samples" / "demo_images2"
        image_files = list(sample_images_path.glob("*.png"))
        
        if not image_files:
            logger.error("❌ 本番画像が見つかりません (data/samples/demo_images2/)")
            return
        
        # 出力ディレクトリ準備
        output_dir = project_root / "data" / "output" / "sam2_prompt_visualization"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # SAM2モデル初期化（プロンプト生成のみなのでロードは不要）
        logger.info("🎨 プロンプト可視化システム初期化")
        sam2_manager = SAM2ModelManager(model_name="sam2_hiera_large.pt")
        tester = ComprehensivePromptTester(sam2_manager)
        
        # テスト対象パーツ
        test_parts = ["hair", "face", "body", "eyes"]
        
        # 本番画像でテスト
        image_path = image_files[0]
        logger.info(f"📸 本番画像: {image_path.name}")
        
        # 画像読み込み
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        all_results = {}
        
        # 各パーツでプロンプト可視化
        for part_name in test_parts:
            logger.info(f"\n🎯 {part_name.upper()}パーツ - プロンプト可視化")
            
            result = test_prompt_visualization(
                image_rgb, part_name, tester, output_dir, image_path.stem
            )
            
            all_results[part_name] = result
        
        # 全体サマリー
        logger.info(f"\n📊 可視化結果サマリー:")
        
        total_images = 0
        for part_name, result in all_results.items():
            strategies_count = result["total_strategies"]
            total_images += strategies_count + 1  # 個別画像 + 比較画像
            
            logger.info(f"  🎨 {part_name.upper()}: {strategies_count}戦略")
            logger.info(f"    📁 比較画像: {Path(result['comparison_image_path']).name}")
        
        logger.info(f"\n📁 結果保存先: {output_dir}")
        logger.info(f"  🖼️ 生成画像数: {total_images}枚")
        logger.info(f"  📋 戦略詳細: {len(test_parts)}パーツ分のJSON")
        
        # 可視化説明
        logger.info(f"\n🎨 プロンプト可視化の見方:")
        logger.info(f"  🟢 緑の円: 正例プロンプト（この領域を含める）")
        logger.info(f"  🔴 赤の円: 負例プロンプト（この領域を除外する）")
        logger.info(f"  ⚪ 白い縁: プロンプト境界")
        logger.info(f"  🔢 数字: プロンプト番号（配置順序）")
        
        logger.info("🎉 プロンプト可視化テスト完了!")
        
    except Exception as e:
        logger.error(f"❌ テスト実行中にエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()