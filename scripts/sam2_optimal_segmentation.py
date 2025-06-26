#!/usr/bin/env python3
"""
SAM2最適セグメンテーション実行

最良の戦略（Adaptive Sparse + ハイブリッド手法）で各パーツを分割し、
結果をオリジナル画像と共に保存する
"""

import sys
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json
import shutil

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils import setup_logging, get_logger
from src.core.sam2.sam2_model import SAM2ModelManager
from src.core.sam2.prompt_handler import SAM2PromptHandler, PointPrompt

# ハイブリッド手法をインポート
from scripts.test_sam2_hybrid_enhancement import SAM2HybridEnhancer

@dataclass
class SegmentationResult:
    """セグメンテーション結果"""
    part_name: str
    mask: np.ndarray
    score: float
    processing_time: float
    prompts_used: int
    strategy: str
    enhancement_log: Dict[str, Any]

class OptimalSegmenter:
    """最適セグメンテーター"""
    
    def __init__(self, sam2_manager: SAM2ModelManager):
        self.sam2_manager = sam2_manager
        self.hybrid_enhancer = SAM2HybridEnhancer(sam2_manager)
        self.logger = get_logger(__name__)
        
    def generate_adaptive_sparse_prompts(self, image_shape: Tuple[int, int], part_name: str) -> List[PointPrompt]:
        """適応的スパースプロンプト生成（最優秀戦略）"""
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
    
    def segment_part(self, image_rgb: np.ndarray, part_name: str) -> SegmentationResult:
        """パーツセグメンテーション実行"""
        self.logger.info(f"🎯 {part_name.upper()}セグメンテーション開始")
        
        start_time = time.time()
        height, width = image_rgb.shape[:2]
        
        # 適応的スパースプロンプト生成
        prompts = self.generate_adaptive_sparse_prompts((height, width), part_name)
        
        # ハイブリッド手法で実行
        best_mask, best_score, enhancement_log = self.hybrid_enhancer.hybrid_enhancement(
            image_rgb, prompts, part_name
        )
        
        processing_time = time.time() - start_time
        
        result = SegmentationResult(
            part_name=part_name,
            mask=best_mask,
            score=best_score,
            processing_time=processing_time,
            prompts_used=len(prompts),
            strategy="adaptive_sparse_hybrid",
            enhancement_log=enhancement_log
        )
        
        self.logger.info(f"  ✅ 完了: スコア {best_score:.3f}, 時間 {processing_time:.2f}s")
        return result
    
    def create_visualization(self, image_rgb: np.ndarray, result: SegmentationResult) -> np.ndarray:
        """結果可視化"""
        # オーバーレイ作成
        overlay = image_rgb.copy().astype(np.float32)
        
        # パーツ別色分け
        colors = {
            "hair": [255, 100, 150],    # ピンク
            "face": [255, 200, 100],    # オレンジ
            "body": [100, 200, 255],    # 青
            "eyes": [150, 255, 100],    # 緑
        }
        color = np.array(colors.get(result.part_name, [255, 255, 255]))
        
        # マスク適用
        mask_bool = result.mask.astype(bool)
        overlay[mask_bool] = overlay[mask_bool] * 0.4 + color * 0.6
        
        # 情報表示
        info_lines = [
            f"Part: {result.part_name.upper()}",
            f"Score: {result.score:.3f}",
            f"Strategy: {result.strategy}",
            f"Time: {result.processing_time:.2f}s",
            f"Prompts: {result.prompts_used}",
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            # 黒い背景
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(overlay.astype(np.uint8), (10, y_pos - 20), 
                         (text_size[0] + 20, y_pos + 5), (0, 0, 0), -1)
            # 白い文字
            cv2.putText(overlay.astype(np.uint8), line, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay.astype(np.uint8)
    
    def save_results(self, image_rgb: np.ndarray, results: List[SegmentationResult], 
                    output_dir: Path, image_name: str) -> Dict[str, Any]:
        """結果保存"""
        
        # オリジナル画像保存
        original_path = output_dir / f"original_{image_name}.png"
        original_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(original_path), original_bgr)
        self.logger.info(f"  📸 オリジナル画像保存: {original_path.name}")
        
        saved_files = {"original": str(original_path)}
        
        # パーツ別結果保存
        for result in results:
            part_name = result.part_name
            
            # マスク画像保存（バイナリ）
            mask_path = output_dir / f"mask_{part_name}_{image_name}.png"
            cv2.imwrite(str(mask_path), (result.mask * 255).astype(np.uint8))
            saved_files[f"mask_{part_name}"] = str(mask_path)
            
            # 透明PNG保存（アルファチャンネル付き）
            transparent_path = output_dir / f"transparent_{part_name}_{image_name}.png"
            transparent_image = image_rgb.copy()
            alpha_channel = result.mask.astype(np.uint8) * 255
            transparent_rgba = np.dstack([transparent_image, alpha_channel])
            transparent_bgra = cv2.cvtColor(transparent_rgba, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(str(transparent_path), transparent_bgra)
            saved_files[f"transparent_{part_name}"] = str(transparent_path)
            
            # 可視化画像保存
            vis_image = self.create_visualization(image_rgb, result)
            vis_path = output_dir / f"visualization_{part_name}_{image_name}.png"
            vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(vis_path), vis_bgr)
            saved_files[f"visualization_{part_name}"] = str(vis_path)
        
        # 全パーツ統合画像作成
        combined_image = self.create_combined_visualization(image_rgb, results)
        combined_path = output_dir / f"combined_all_parts_{image_name}.png"
        combined_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(combined_path), combined_bgr)
        saved_files["combined"] = str(combined_path)
        
        self.logger.info(f"  🎨 統合画像保存: {combined_path.name}")
        
        return saved_files
    
    def create_combined_visualization(self, image_rgb: np.ndarray, 
                                    results: List[SegmentationResult]) -> np.ndarray:
        """全パーツ統合可視化"""
        overlay = image_rgb.copy().astype(np.float32)
        
        # パーツ別色分け
        colors = {
            "hair": [255, 100, 150],    # ピンク
            "face": [255, 200, 100],    # オレンジ  
            "body": [100, 200, 255],    # 青
            "eyes": [150, 255, 100],    # 緑
        }
        
        # 各パーツのマスクを重ね合わせ
        for result in results:
            color = np.array(colors.get(result.part_name, [255, 255, 255]))
            mask_bool = result.mask.astype(bool)
            overlay[mask_bool] = overlay[mask_bool] * 0.7 + color * 0.3
        
        # 統合情報表示
        total_score = np.mean([r.score for r in results])
        total_time = sum([r.processing_time for r in results])
        
        info_lines = [
            "All Parts Segmentation",
            f"Parts: {len(results)}",
            f"Avg Score: {total_score:.3f}",
            f"Total Time: {total_time:.2f}s",
            f"Strategy: adaptive_sparse_hybrid",
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(overlay.astype(np.uint8), (10, y_pos - 20), 
                         (text_size[0] + 20, y_pos + 5), (0, 0, 0), -1)
            cv2.putText(overlay.astype(np.uint8), line, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return overlay.astype(np.uint8)

def main():
    """メイン実行"""
    
    # ログ設定
    setup_logging(level="INFO", console_output=True, structured=False)
    logger = get_logger(__name__)
    
    logger.info("=== SAM2最適セグメンテーション実行 ===")
    
    try:
        # 本番画像取得
        sample_images_path = project_root / "data" / "samples" / "demo_images2"
        image_files = list(sample_images_path.glob("*.png"))
        
        if not image_files:
            logger.error("❌ 本番画像が見つかりません (data/samples/demo_images2/)")
            return
        
        # 出力ディレクトリ準備
        output_dir = project_root / "data" / "output" / "sam2_optimal_segmentation"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # SAM2モデル初期化
        logger.info("🤖 SAM2モデル初期化")
        sam2_manager = SAM2ModelManager(model_name="sam2_hiera_large.pt")
        if not sam2_manager.load_model():
            logger.error("❌ SAM2モデル読み込み失敗")
            return
        
        segmenter = OptimalSegmenter(sam2_manager)
        
        # テスト対象パーツ（最適順序）
        target_parts = ["face", "body", "hair", "eyes"]  # Adaptive Sparseが最も得意な順序
        
        # 各画像で実行
        for image_path in image_files:
            logger.info(f"\n📸 画像処理開始: {image_path.name}")
            
            # 画像読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"  ⚠️ 画像読み込み失敗: {image_path}")
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.info(f"  📐 画像サイズ: {image_rgb.shape[1]}×{image_rgb.shape[0]}")
            
            # 各パーツをセグメンテーション
            results = []
            total_start_time = time.time()
            
            for part_name in target_parts:
                logger.info(f"\n  🎯 {part_name.upper()}パーツ処理")
                
                try:
                    result = segmenter.segment_part(image_rgb, part_name)
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"    ❌ {part_name}セグメンテーション失敗: {e}")
                    continue
            
            total_processing_time = time.time() - total_start_time
            
            if not results:
                logger.warning(f"  ⚠️ {image_path.name}の処理結果がありません")
                continue
            
            # 結果保存
            logger.info(f"\n  💾 結果保存中...")
            saved_files = segmenter.save_results(image_rgb, results, output_dir, image_path.stem)
            
            # 詳細情報JSON保存
            results_json = {
                "image_name": image_path.name,
                "image_size": {"width": image_rgb.shape[1], "height": image_rgb.shape[0]},
                "total_processing_time": total_processing_time,
                "parts_processed": len(results),
                "strategy": "adaptive_sparse_hybrid",
                "results": {
                    result.part_name: {
                        "score": result.score,
                        "processing_time": result.processing_time,
                        "prompts_used": result.prompts_used,
                        "strategy": result.strategy
                    } for result in results
                },
                "saved_files": saved_files,
                "summary": {
                    "average_score": np.mean([r.score for r in results]),
                    "total_time": total_processing_time,
                    "best_part": max(results, key=lambda r: r.score).part_name,
                    "best_score": max(results, key=lambda r: r.score).score
                }
            }
            
            json_path = output_dir / f"results_{image_path.stem}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_json, f, indent=2, ensure_ascii=False)
            
            # 結果サマリー
            logger.info(f"\n  📊 {image_path.name} 処理完了:")
            logger.info(f"    🎯 処理パーツ: {len(results)}")
            logger.info(f"    📈 平均スコア: {np.mean([r.score for r in results]):.3f}")
            logger.info(f"    ⏱️ 総処理時間: {total_processing_time:.2f}秒")
            logger.info(f"    🏆 最高スコア: {max(results, key=lambda r: r.score).score:.3f} ({max(results, key=lambda r: r.score).part_name})")
            logger.info(f"    📁 保存ファイル数: {len(saved_files)}")
        
        # 全体サマリー
        logger.info(f"\n🎉 全画像処理完了!")
        logger.info(f"📁 結果保存先: {output_dir}")
        logger.info(f"🖼️ 処理画像数: {len(list(output_dir.glob('original_*.png')))}")
        
        # 出力ファイル説明
        logger.info(f"\n📋 出力ファイル説明:")
        logger.info(f"  📸 original_*.png: オリジナル画像")
        logger.info(f"  🎭 mask_*.png: バイナリマスク（白黒）")
        logger.info(f"  🌟 transparent_*.png: 透明PNG（アルファチャンネル付き）")
        logger.info(f"  🎨 visualization_*.png: 可視化画像（オーバーレイ）")
        logger.info(f"  🎯 combined_all_parts_*.png: 全パーツ統合画像")
        logger.info(f"  📋 results_*.json: 詳細結果データ")
        
    except Exception as e:
        logger.error(f"❌ 処理中にエラー: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # メモリクリア
        if 'sam2_manager' in locals():
            sam2_manager.unload_model()

if __name__ == "__main__":
    main()