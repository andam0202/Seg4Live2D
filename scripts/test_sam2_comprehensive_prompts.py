#!/usr/bin/env python3
"""
SAM2包括的プロンプト戦略テスト

様々なプロンプト戦略と複数パーツでハイブリッド手法を比較テスト
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

@dataclass
class PromptStrategy:
    """プロンプト戦略定義"""
    name: str
    description: str
    prompt_generator: callable
    expected_parts: List[str]

class ComprehensivePromptTester:
    """包括的プロンプトテスター"""
    
    def __init__(self, sam2_manager: SAM2ModelManager):
        self.sam2_manager = sam2_manager
        self.hybrid_enhancer = SAM2HybridEnhancer(sam2_manager)
        self.logger = get_logger(__name__)
    
    def generate_basic_prompts(self, image_shape: Tuple[int, int], part_name: str) -> List[PointPrompt]:
        """基本プロンプト生成（従来型）"""
        height, width = image_shape[:2]
        
        prompts = {
            "hair": [
                PointPrompt(width//2, height//4, 1, "髪_中央上"),
                PointPrompt(width//4, height//3, 1, "髪_左側"),
                PointPrompt(3*width//4, height//3, 1, "髪_右側"),
                PointPrompt(width//2, height//6, 1, "髪_頭頂"),
                PointPrompt(width//6, 5*height//8, 1, "髪_左下"),
                PointPrompt(5*width//6, 5*height//8, 1, "髪_右下"),
                PointPrompt(width//2, 5*height//8, 0, "背景_除外"),
                PointPrompt(width//2, 5*height//6, 0, "体_除外"),
            ],
            "face": [
                PointPrompt(width//2, 2*height//5, 1, "顔_中央"),
                PointPrompt(2*width//5, 2*height//5, 1, "左頬_肌"),
                PointPrompt(3*width//5, 2*height//5, 1, "右頬_肌"),
                PointPrompt(width//2, height//3, 1, "額_部分"),
                PointPrompt(width//2, height//2, 1, "鼻_部分"),
                PointPrompt(width//2, 3*height//5, 1, "顎_部分"),
                PointPrompt(width//4, height//5, 0, "背景_左"),
                PointPrompt(3*width//4, height//5, 0, "背景_右"),
            ],
            "body": [
                PointPrompt(width//2, 3*height//5, 1, "胴体_中央"),
                PointPrompt(2*width//5, 2*height//3, 1, "体_左側"),
                PointPrompt(3*width//5, 2*height//3, 1, "体_右側"),
                PointPrompt(width//2, height//2, 1, "胸_上部"),
                PointPrompt(width//3, 4*height//5, 1, "腰_左側"),
                PointPrompt(2*width//3, 4*height//5, 1, "腰_右側"),
                PointPrompt(width//2, height//4, 0, "顔_除外"),
                PointPrompt(width//2, 9*height//10, 0, "下端_除外"),
            ],
            "eyes": [
                PointPrompt(2*width//5, 2*height//5, 1, "左眼_瞳"),
                PointPrompt(3*width//5, 2*height//5, 1, "右眼_瞳"),
                PointPrompt(width//3, 2*height//5, 1, "左目_周辺"),
                PointPrompt(2*width//3, 2*height//5, 1, "右目_周辺"),
                PointPrompt(width//2, height//3, 0, "額_除外"),
                PointPrompt(width//2, height//2, 0, "鼻_除外"),
            ]
        }
        
        return prompts.get(part_name, prompts["hair"])
    
    def generate_dense_grid_prompts(self, image_shape: Tuple[int, int], part_name: str) -> List[PointPrompt]:
        """密集グリッドプロンプト生成"""
        height, width = image_shape[:2]
        prompts = []
        
        # パーツ別の関心領域定義
        regions = {
            "hair": {"x_range": (0.1, 0.9), "y_range": (0.05, 0.7)},
            "face": {"x_range": (0.25, 0.75), "y_range": (0.2, 0.65)},
            "body": {"x_range": (0.2, 0.8), "y_range": (0.4, 0.9)},
            "eyes": {"x_range": (0.3, 0.7), "y_range": (0.35, 0.45)},
        }
        
        region = regions.get(part_name, regions["hair"])
        
        # グリッド生成
        grid_size = 5  # 5x5グリッド
        for i in range(grid_size):
            for j in range(grid_size):
                x_ratio = region["x_range"][0] + (region["x_range"][1] - region["x_range"][0]) * j / (grid_size - 1)
                y_ratio = region["y_range"][0] + (region["y_range"][1] - region["y_range"][0]) * i / (grid_size - 1)
                
                x = int(width * x_ratio)
                y = int(height * y_ratio)
                
                # 中央付近は正例、端は負例
                if 1 <= i <= 3 and 1 <= j <= 3:
                    label = 1
                    desc = f"内側G{i}-{j}"
                else:
                    label = 0
                    desc = f"外側G{i}-{j}"
                
                prompts.append(PointPrompt(x, y, label, desc))
        
        self.logger.info(f"  🔲 密集グリッドプロンプト生成: {len(prompts)}点")
        return prompts
    
    def generate_anatomical_prompts(self, image_shape: Tuple[int, int], part_name: str) -> List[PointPrompt]:
        """解剖学的プロンプト生成（Live2D特化）"""
        height, width = image_shape[:2]
        
        anatomical_prompts = {
            "hair": [
                # 髪の成長パターンに基づく
                PointPrompt(width//2, height//10, 1, "つむじ_頂点"),
                PointPrompt(width//3, height//8, 1, "前髪_左端"),
                PointPrompt(2*width//3, height//8, 1, "前髪_右端"),
                PointPrompt(width//2, height//6, 1, "前髪_中央"),
                PointPrompt(width//8, height//3, 1, "左サイド_髪"),
                PointPrompt(7*width//8, height//3, 1, "右サイド_髪"),
                PointPrompt(width//5, 3*height//5, 1, "後髪_左"),
                PointPrompt(4*width//5, 3*height//5, 1, "後髪_右"),
                PointPrompt(width//2, 2*height//3, 1, "後髪_中央"),
                # 髪の流れる方向
                PointPrompt(width//4, height//2, 1, "髪流れ_左"),
                PointPrompt(3*width//4, height//2, 1, "髪流れ_右"),
                # 除外領域
                PointPrompt(width//2, 2*height//5, 0, "顔_除外"),
                PointPrompt(width//2, 4*height//5, 0, "体_除外"),
            ],
            "face": [
                # 顔の骨格構造に基づく
                PointPrompt(width//2, height//3, 1, "額_中央"),
                PointPrompt(width//3, 2*height//5, 1, "左頬骨"),
                PointPrompt(2*width//3, 2*height//5, 1, "右頬骨"),
                PointPrompt(width//2, 2*height//5, 1, "鼻梁"),
                PointPrompt(width//2, height//2, 1, "鼻先"),
                PointPrompt(width//2, 3*height//5, 1, "上唇"),
                PointPrompt(width//2, 2*height//3, 1, "下顎"),
                # 顔の輪郭
                PointPrompt(width//4, height//2, 1, "左輪郭"),
                PointPrompt(3*width//4, height//2, 1, "右輪郭"),
                # 除外領域
                PointPrompt(width//2, height//5, 0, "髪_除外"),
                PointPrompt(width//5, height//2, 0, "背景左_除外"),
                PointPrompt(4*width//5, height//2, 0, "背景右_除外"),
            ],
            "body": [
                # 人体解剖学に基づく
                PointPrompt(width//2, height//2, 1, "胸部_中央"),
                PointPrompt(width//3, 3*height//5, 1, "左肩"),
                PointPrompt(2*width//3, 3*height//5, 1, "右肩"),
                PointPrompt(width//2, 2*height//3, 1, "腹部"),
                PointPrompt(width//4, 2*height//3, 1, "左腰"),
                PointPrompt(3*width//4, 2*height//3, 1, "右腰"),
                PointPrompt(width//5, 4*height//5, 1, "左腕"),
                PointPrompt(4*width//5, 4*height//5, 1, "右腕"),
                # 除外領域
                PointPrompt(width//2, height//3, 0, "首上_除外"),
                PointPrompt(width//2, 9*height//10, 0, "下端_除外"),
            ],
            "eyes": [
                # 眼球解剖学に基づく
                PointPrompt(width//3, 2*height//5, 1, "左眼球"),
                PointPrompt(2*width//3, 2*height//5, 1, "右眼球"),
                PointPrompt(width//4, 2*height//5, 1, "左目頭"),
                PointPrompt(3*width//4, 2*height//5, 1, "右目頭"),
                PointPrompt(width//3, height//3, 1, "左眉下"),
                PointPrompt(2*width//3, height//3, 1, "右眉下"),
                # 除外領域
                PointPrompt(width//2, height//4, 0, "額_除外"),
                PointPrompt(width//2, height//2, 0, "鼻_除外"),
            ]
        }
        
        prompts = anatomical_prompts.get(part_name, anatomical_prompts["hair"])
        self.logger.info(f"  🧬 解剖学的プロンプト生成: {len(prompts)}点")
        return prompts
    
    def generate_semantic_prompts(self, image_shape: Tuple[int, int], part_name: str) -> List[PointPrompt]:
        """セマンティック（意味的）プロンプト生成"""
        height, width = image_shape[:2]
        
        semantic_prompts = {
            "hair": [
                # 髪の質感・スタイルに基づく
                PointPrompt(width//2, height//8, 1, "髪_ボリューム中心"),
                PointPrompt(width//4, height//4, 1, "髪_左サイド"),
                PointPrompt(3*width//4, height//4, 1, "髪_右サイド"),
                PointPrompt(width//6, height//2, 1, "髪_左外側"),
                PointPrompt(5*width//6, height//2, 1, "髪_右外側"),
                # 髪の動きを表現
                PointPrompt(width//3, 2*height//3, 1, "左流れ_動き"),
                PointPrompt(2*width//3, 2*height//3, 1, "右流れ_動き"),
                PointPrompt(width//2, 3*height//4, 1, "後流れ_動き"),
                # 髪の境界
                PointPrompt(width//5, height//3, 1, "左境界_髪"),
                PointPrompt(4*width//5, height//3, 1, "右境界_髪"),
                # セマンティック除外
                PointPrompt(width//2, 2*height//5, 0, "肌_除外"),
                PointPrompt(width//2, 4*height//5, 0, "服_除外"),
                PointPrompt(width//10, height//2, 0, "背景左_除外"),
                PointPrompt(9*width//10, height//2, 0, "背景右_除外"),
            ],
            "face": [
                # 表情・感情に基づく
                PointPrompt(width//2, height//3, 1, "表情_中心"),
                PointPrompt(width//3, 2*height//5, 1, "左頬_表情"),
                PointPrompt(2*width//3, 2*height//5, 1, "右頬_表情"),
                PointPrompt(width//2, height//2, 1, "鼻_立体感"),
                PointPrompt(width//2, 3*height//5, 1, "口元_表情"),
                # 肌の質感
                PointPrompt(width//4, height//3, 1, "肌_左側"),
                PointPrompt(3*width//4, height//3, 1, "肌_右側"),
                PointPrompt(width//2, 2*height//3, 1, "顎_輪郭"),
                # セマンティック除外
                PointPrompt(width//2, height//5, 0, "髪_境界"),
                PointPrompt(width//6, height//2, 0, "背景_左側"),
                PointPrompt(5*width//6, height//2, 0, "背景_右側"),
            ],
            "body": [
                # 体型・ポーズに基づく
                PointPrompt(width//2, height//2, 1, "体幹_中央"),
                PointPrompt(width//3, 3*height//5, 1, "左肩_ライン"),
                PointPrompt(2*width//3, 3*height//5, 1, "右肩_ライン"),
                PointPrompt(width//2, 2*height//3, 1, "ウエスト"),
                PointPrompt(width//4, 3*height//4, 1, "左腕_付け根"),
                PointPrompt(3*width//4, 3*height//4, 1, "右腕_付け根"),
                # 服装との境界
                PointPrompt(width//2, height//2, 1, "服_胸部"),
                PointPrompt(width//2, 4*height//5, 1, "服_下部"),
                # セマンティック除外
                PointPrompt(width//2, height//4, 0, "首_境界"),
                PointPrompt(width//2, 9*height//10, 0, "画像_下端"),
            ],
            "eyes": [
                # 視線・眼差しに基づく
                PointPrompt(width//3, 2*height//5, 1, "左瞳_中心"),
                PointPrompt(2*width//3, 2*height//5, 1, "右瞳_中心"),
                PointPrompt(width//4, 2*height//5, 1, "左白目"),
                PointPrompt(3*width//4, 2*height//5, 1, "右白目"),
                PointPrompt(width//3, height//3, 1, "左睫毛"),
                PointPrompt(2*width//3, height//3, 1, "右睫毛"),
                # セマンティック除外
                PointPrompt(width//2, height//4, 0, "眉毛_境界"),
                PointPrompt(width//2, height//2, 0, "頬_境界"),
            ]
        }
        
        prompts = semantic_prompts.get(part_name, semantic_prompts["hair"])
        self.logger.info(f"  🎨 セマンティックプロンプト生成: {len(prompts)}点")
        return prompts
    
    def generate_adaptive_sparse_prompts(self, image_shape: Tuple[int, int], part_name: str) -> List[PointPrompt]:
        """適応的スパースプロンプト生成（最小限で最大効果）"""
        height, width = image_shape[:2]
        
        sparse_prompts = {
            "hair": [
                # 戦略的最小点配置
                PointPrompt(width//2, height//6, 1, "頭頂部_髪"),
                PointPrompt(width//4, height//2, 1, "左サイド_髪"),
                PointPrompt(3*width//4, height//2, 1, "右サイド_髪"),
                PointPrompt(width//2, 2*height//3, 1, "後髪_流れ"),
                # 重要な除外点
                PointPrompt(width//2, 2*height//5, 0, "顔_除外"),
                PointPrompt(width//2, 4*height//5, 0, "体_除外"),
            ],
            "face": [
                PointPrompt(width//2, 2*height//5, 1, "顔_中心"),
                PointPrompt(width//3, height//2, 1, "左頬_肌"),
                PointPrompt(2*width//3, height//2, 1, "右頬_肌"),
                PointPrompt(width//2, height//6, 0, "髪_除外"),
                PointPrompt(width//6, height//2, 0, "背景_除外"),
            ],
            "body": [
                PointPrompt(width//2, 3*height//5, 1, "胴体_中心"),
                PointPrompt(width//3, 2*height//3, 1, "左肩_体"),
                PointPrompt(2*width//3, 2*height//3, 1, "右肩_体"),
                PointPrompt(width//2, height//3, 0, "首上_除外"),
            ],
            "eyes": [
                PointPrompt(width//3, 2*height//5, 1, "左眼_瞳"),
                PointPrompt(2*width//3, 2*height//5, 1, "右眼_瞳"),
                PointPrompt(width//2, height//3, 0, "額_除外"),
            ]
        }
        
        prompts = sparse_prompts.get(part_name, sparse_prompts["hair"])
        self.logger.info(f"  ⚡ 適応的スパースプロンプト生成: {len(prompts)}点")
        return prompts

def test_comprehensive_prompts(
    image_rgb: np.ndarray,
    part_name: str,
    tester: ComprehensivePromptTester,
    output_dir: Path,
    image_name: str
) -> Dict[str, Any]:
    """包括的プロンプト戦略テスト"""
    logger = get_logger(__name__)
    
    logger.info(f"🎯 包括的プロンプトテスト開始 - {part_name}")
    
    # プロンプト戦略定義
    strategies = [
        PromptStrategy(
            "basic", "基本プロンプト（従来型）",
            tester.generate_basic_prompts, ["hair", "face", "body", "eyes"]
        ),
        PromptStrategy(
            "dense_grid", "密集グリッドプロンプト",
            tester.generate_dense_grid_prompts, ["hair", "face", "body", "eyes"]
        ),
        PromptStrategy(
            "anatomical", "解剖学的プロンプト",
            tester.generate_anatomical_prompts, ["hair", "face", "body", "eyes"]
        ),
        PromptStrategy(
            "semantic", "セマンティックプロンプト",
            tester.generate_semantic_prompts, ["hair", "face", "body", "eyes"]
        ),
        PromptStrategy(
            "adaptive_sparse", "適応的スパースプロンプト",
            tester.generate_adaptive_sparse_prompts, ["hair", "face", "body", "eyes"]
        ),
    ]
    
    results = {}
    height, width = image_rgb.shape[:2]
    
    for strategy in strategies:
        if part_name not in strategy.expected_parts:
            continue
            
        logger.info(f"\n🧪 戦略テスト: {strategy.name}")
        logger.info(f"   {strategy.description}")
        
        try:
            start_time = time.time()
            
            # プロンプト生成
            prompts = strategy.prompt_generator((height, width), part_name)
            
            # ハイブリッド手法で実行
            best_mask, best_score, enhancement_log = tester.hybrid_enhancer.hybrid_enhancement(
                image_rgb, prompts, part_name
            )
            
            processing_time = time.time() - start_time
            
            # 結果可視化
            overlay = image_rgb.copy().astype(np.float32)
            colors = {
                "basic": [255, 100, 100],          # 赤
                "dense_grid": [100, 255, 100],     # 緑
                "anatomical": [100, 100, 255],     # 青
                "semantic": [255, 255, 100],       # 黄
                "adaptive_sparse": [255, 100, 255] # マゼンタ
            }
            color = np.array(colors.get(strategy.name, [255, 100, 100]))
            
            mask_bool = best_mask.astype(bool)
            overlay[mask_bool] = overlay[mask_bool] * 0.6 + color * 0.4
            overlay_image = overlay.astype(np.uint8)
            
            # 情報表示
            info_lines = [
                f"Strategy: {strategy.name}",
                f"Part: {part_name}",
                f"Score: {best_score:.3f}",
                f"Time: {processing_time:.2f}s",
                f"Prompts: {len(prompts)} -> {enhancement_log['final_prompts_count']}",
            ]
            
            for i, line in enumerate(info_lines):
                cv2.putText(overlay_image, line, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 保存
            output_path = output_dir / f"{strategy.name}_{part_name}_{image_name}.png"
            overlay_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), overlay_bgr)
            
            # 結果記録
            results[strategy.name] = {
                "score": best_score,
                "processing_time": processing_time,
                "initial_prompts": len(prompts),
                "final_prompts": enhancement_log['final_prompts_count'],
                "enhancement_log": enhancement_log,
                "output_path": str(output_path)
            }
            
            logger.info(f"   📊 結果: スコア {best_score:.3f}, 時間 {processing_time:.2f}s, プロンプト {len(prompts)}→{enhancement_log['final_prompts_count']}")
            
        except Exception as e:
            logger.error(f"   ❌ 戦略テスト失敗 {strategy.name}: {e}")
            results[strategy.name] = {"error": str(e)}
    
    return results

def main():
    """メイン実行"""
    
    # ログ設定
    setup_logging(level="INFO", console_output=True, structured=False)
    logger = get_logger(__name__)
    
    logger.info("=== SAM2包括的プロンプト戦略テスト ===")
    
    try:
        # 本番画像取得
        sample_images_path = project_root / "data" / "samples" / "demo_images2"
        image_files = list(sample_images_path.glob("*.png"))
        
        if not image_files:
            logger.error("❌ 本番画像が見つかりません (data/samples/demo_images2/)")
            return
        
        # 出力ディレクトリ準備
        output_dir = project_root / "data" / "output" / "sam2_comprehensive_prompts"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # SAM2モデル初期化
        logger.info("🤖 SAM2モデル初期化")
        sam2_manager = SAM2ModelManager(model_name="sam2_hiera_large.pt")
        if not sam2_manager.load_model():
            logger.error("❌ SAM2モデル読み込み失敗")
            return
        
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
        
        # 各パーツで包括的テスト
        for part_name in test_parts:
            logger.info(f"\n🎯 パーツテスト開始: {part_name.upper()}")
            
            part_results = test_comprehensive_prompts(
                image_rgb, part_name, tester, output_dir, image_path.stem
            )
            
            all_results[part_name] = part_results
        
        # 結果分析
        logger.info(f"\n📊 包括的結果分析:")
        
        # パーツ別最高性能
        for part_name, part_results in all_results.items():
            logger.info(f"\n🏆 {part_name.upper()}パーツ - 戦略別性能:")
            
            valid_results = {k: v for k, v in part_results.items() if "error" not in v}
            if not valid_results:
                logger.warning(f"  ⚠️ {part_name}に有効な結果がありません")
                continue
                
            sorted_strategies = sorted(valid_results.items(), key=lambda x: x[1]["score"], reverse=True)
            
            for i, (strategy, result) in enumerate(sorted_strategies):
                score = result["score"]
                time_taken = result["processing_time"]
                initial_prompts = result["initial_prompts"]
                final_prompts = result["final_prompts"]
                
                logger.info(f"  {i+1}. {strategy:15} - スコア: {score:.3f}, 時間: {time_taken:.2f}s, プロンプト: {initial_prompts}→{final_prompts}")
        
        # 全体最高性能
        logger.info(f"\n🥇 全体最高性能戦略:")
        all_scores = []
        for part_results in all_results.values():
            for strategy, result in part_results.items():
                if "error" not in result:
                    all_scores.append((strategy, result["score"], result["processing_time"]))
        
        if all_scores:
            all_scores.sort(key=lambda x: x[1], reverse=True)
            best_strategy, best_score, best_time = all_scores[0]
            logger.info(f"  🎉 最優秀戦略: {best_strategy}")
            logger.info(f"  📊 最高スコア: {best_score:.3f}")
            logger.info(f"  ⏱️ 処理時間: {best_time:.2f}秒")
        
        # 結果をJSONで保存
        results_json_path = output_dir / f"comprehensive_results_{image_path.stem}.json"
        with open(results_json_path, 'w', encoding='utf-8') as f:
            # enhancement_logは複雑すぎるので除外
            simplified_results = {}
            for part, strategies in all_results.items():
                simplified_results[part] = {}
                for strategy, result in strategies.items():
                    if "error" not in result:
                        simplified_results[part][strategy] = {
                            "score": result["score"],
                            "processing_time": result["processing_time"],
                            "initial_prompts": result["initial_prompts"],
                            "final_prompts": result["final_prompts"]
                        }
                    else:
                        simplified_results[part][strategy] = result
            
            json.dump(simplified_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📁 結果保存先: {output_dir}")
        logger.info(f"📋 詳細結果JSON: {results_json_path}")
        logger.info("🎉 包括的プロンプト戦略テスト完了!")
        
    except Exception as e:
        logger.error(f"❌ テスト実行中にエラー: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # メモリクリア
        if 'sam2_manager' in locals():
            sam2_manager.unload_model()

if __name__ == "__main__":
    main()