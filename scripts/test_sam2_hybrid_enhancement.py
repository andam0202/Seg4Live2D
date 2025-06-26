#!/usr/bin/env python3
"""
SAM2ハイブリッド精度向上テスト

Edge-guided + Iterative + 適応的パラメータ最適化を組み合わせた最高精度手法
"""

import sys
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils import setup_logging, get_logger
from src.core.sam2.sam2_model import SAM2ModelManager
from src.core.sam2.prompt_handler import SAM2PromptHandler, PointPrompt

@dataclass
class ImageCharacteristics:
    """画像特性分析結果"""
    brightness: float
    contrast: float
    edge_density: float
    color_variance: float
    is_anime_style: bool

class SAM2HybridEnhancer:
    """SAM2ハイブリッド精度向上クラス"""
    
    def __init__(self, sam2_manager: SAM2ModelManager):
        self.sam2_manager = sam2_manager
        self.logger = get_logger(__name__)
        
    def analyze_image_characteristics(self, image: np.ndarray) -> ImageCharacteristics:
        """画像特性を詳細分析"""
        self.logger.info("📊 画像特性分析開始")
        
        # 明度分析
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        brightness = float(np.mean(gray))
        
        # コントラスト分析
        contrast = float(np.std(gray))
        
        # エッジ密度分析
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)
        
        # 色分散分析
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        color_variance = float(np.std(hsv[:, :, 1]))  # 彩度の分散
        
        # アニメスタイル判定（彩度が高く、エッジがはっきりしている）
        is_anime_style = color_variance > 30 and edge_density > 0.05
        
        characteristics = ImageCharacteristics(
            brightness=brightness,
            contrast=contrast,
            edge_density=edge_density,
            color_variance=color_variance,
            is_anime_style=is_anime_style
        )
        
        self.logger.info(f"  🔍 明度: {brightness:.1f}, コントラスト: {contrast:.1f}")
        self.logger.info(f"  🌐 エッジ密度: {edge_density:.3f}, 色分散: {color_variance:.1f}")
        self.logger.info(f"  🎨 アニメスタイル: {'Yes' if is_anime_style else 'No'}")
        
        return characteristics
    
    def adaptive_preprocess_image(self, image: np.ndarray, characteristics: ImageCharacteristics) -> np.ndarray:
        """画像特性に応じた適応的前処理"""
        self.logger.info("⚙️ 適応的前処理適用")
        
        processed = image.copy()
        
        # 明度に応じた調整
        if characteristics.brightness < 80:  # 暗い画像
            self.logger.info("  🌙 暗い画像検出 → CLAHE強化")
            lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))  # 強めのCLAHE
            l_channel = clahe.apply(l_channel)
            processed = cv2.merge((l_channel, a, b))
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2RGB)
            
        elif characteristics.brightness > 180:  # 明るい画像
            self.logger.info("  ☀️ 明るい画像検出 → コントラスト調整")
            processed = cv2.convertScaleAbs(processed, alpha=1.2, beta=-20)
        
        # コントラストに応じた調整
        if characteristics.contrast < 30:  # 低コントラスト
            self.logger.info("  📈 低コントラスト検出 → シャープ化")
            kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
            processed = cv2.filter2D(processed, -1, kernel)
        
        # アニメスタイルに応じた調整
        if characteristics.is_anime_style:
            self.logger.info("  🎨 アニメスタイル検出 → 彩度強化")
            hsv = cv2.cvtColor(processed, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.1)  # 彩度を10%向上
            processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return processed
    
    def generate_adaptive_edge_prompts(self, image: np.ndarray, base_prompts: List[PointPrompt], 
                                     characteristics: ImageCharacteristics) -> List[PointPrompt]:
        """画像特性に応じた適応的エッジ誘導プロンプト生成"""
        self.logger.info("🌐 適応的エッジ誘導プロンプト生成")
        
        # エッジ検出パラメータを画像特性に応じて調整
        if characteristics.edge_density > 0.1:  # エッジが多い画像
            low_threshold, high_threshold = 80, 160  # 厳しい閾値
            self.logger.info("  📊 高エッジ密度 → 厳しい閾値設定")
        else:  # エッジが少ない画像
            low_threshold, high_threshold = 30, 100  # 緩い閾値
            self.logger.info("  📊 低エッジ密度 → 緩い閾値設定")
        
        # Cannyエッジ検出
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # 形態学的処理でエッジを強化
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        edge_prompts = base_prompts.copy()
        
        # エッジ上の点を取得
        edge_points = np.column_stack(np.where(edges > 0))
        
        if len(edge_points) == 0:
            self.logger.warning("  ⚠️ エッジ点が検出されませんでした")
            return edge_prompts
        
        # サンプリング数を画像特性に応じて調整
        if characteristics.is_anime_style:
            sample_count = min(30, len(edge_points))  # アニメスタイルはより多くのプロンプト
        else:
            sample_count = min(20, len(edge_points))
        
        # エッジ点をサンプリング
        if len(edge_points) > sample_count:
            indices = np.random.choice(len(edge_points), sample_count, replace=False)
            sampled_edges = edge_points[indices]
        else:
            sampled_edges = edge_points
        
        # 既存プロンプトから遠い点を優先選択
        filtered_edges = []
        for y, x in sampled_edges:
            min_distance = float('inf')
            for base_prompt in base_prompts:
                distance = np.sqrt((x - base_prompt.x)**2 + (y - base_prompt.y)**2)
                min_distance = min(min_distance, distance)
            
            # 距離が30ピクセル以上離れている点のみ追加
            if min_distance > 30:
                filtered_edges.append((y, x))
        
        # エッジプロンプトを追加
        for y, x in filtered_edges:
            edge_prompts.append(
                PointPrompt(int(x), int(y), 1, description="適応的エッジ誘導点")
            )
        
        self.logger.info(f"  🎯 エッジプロンプト追加: +{len(filtered_edges)}点 (総計{len(edge_prompts)}点)")
        return edge_prompts
    
    def advanced_iterative_refinement(self, image: np.ndarray, initial_prompts: List[PointPrompt], 
                                    characteristics: ImageCharacteristics, 
                                    max_iterations: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[PointPrompt]]:
        """高度な反復的マスク改良"""
        self.logger.info(f"🔄 高度な反復的改良開始 (最大{max_iterations}回)")
        
        current_prompts = initial_prompts.copy()
        best_score = -1
        best_result = None
        convergence_threshold = 0.005  # 収束判定閾値
        
        for iteration in range(max_iterations):
            self.logger.info(f"  🔄 反復 {iteration + 1}/{max_iterations}")
            
            # プロンプトハンドラー設定
            handler = SAM2PromptHandler()
            handler.start_new_session()
            
            for prompt in current_prompts:
                handler.add_point_prompt(
                    prompt.x, prompt.y, prompt.label, prompt.description
                )
            
            # 推論実行
            sam2_prompts = handler.get_sam2_prompts()
            masks, scores, logits = self.sam2_manager.predict(
                image=image,
                **sam2_prompts,
                multimask_output=True
            )
            
            current_best_score = float(np.max(scores))
            current_best_mask = masks[np.argmax(scores)]
            
            self.logger.info(f"    📊 スコア: {current_best_score:.3f}")
            
            # 最高スコア更新
            score_improvement = current_best_score - best_score
            if score_improvement > 0:
                best_score = current_best_score
                best_result = (masks, scores, logits)
                self.logger.info(f"    ✨ 新最高スコア! 改善: +{score_improvement:.3f}")
            
            # 収束判定
            if iteration > 0 and score_improvement < convergence_threshold:
                self.logger.info(f"    🎯 収束検出 (改善: {score_improvement:.3f} < {convergence_threshold:.3f})")
                break
            
            # 次の反復のためのプロンプト改良
            if iteration < max_iterations - 1:
                self.logger.info("    🔧 次回反復用プロンプト改良")
                
                # マスクエッジの詳細分析
                mask_uint8 = (current_best_mask * 255).astype(np.uint8)
                
                # 外側エッジ（偽陽性を減らす負例プロンプト）
                dilated = cv2.dilate(mask_uint8, np.ones((5,5), np.uint8), iterations=1)
                outer_edge = dilated - mask_uint8
                outer_points = np.column_stack(np.where(outer_edge > 0))
                
                # 内側エッジ（偽陰性を減らす正例プロンプト）
                eroded = cv2.erode(mask_uint8, np.ones((3,3), np.uint8), iterations=1)
                inner_edge = mask_uint8 - eroded
                inner_points = np.column_stack(np.where(inner_edge > 0))
                
                # 負例プロンプト追加（外側エッジから）
                if len(outer_points) > 0:
                    n_negative = min(3, len(outer_points))  # 最大3点
                    neg_indices = np.random.choice(len(outer_points), n_negative, replace=False)
                    for idx in neg_indices:
                        y, x = outer_points[idx]
                        current_prompts.append(
                            PointPrompt(int(x), int(y), 0, f"反復改良_負例_{iteration}")
                        )
                    self.logger.info(f"      ➖ 負例プロンプト追加: {n_negative}点")
                
                # 正例プロンプト追加（内側エッジから）
                if len(inner_points) > 0:
                    n_positive = min(2, len(inner_points))  # 最大2点
                    pos_indices = np.random.choice(len(inner_points), n_positive, replace=False)
                    for idx in pos_indices:
                        y, x = inner_points[idx]
                        current_prompts.append(
                            PointPrompt(int(x), int(y), 1, f"反復改良_正例_{iteration}")
                        )
                    self.logger.info(f"      ➕ 正例プロンプト追加: {n_positive}点")
        
        self.logger.info(f"  🏁 反復改良完了: 最終スコア {best_score:.3f}")
        return best_result[0], best_result[1], best_result[2], current_prompts
    
    def hybrid_enhancement(self, image: np.ndarray, base_prompts: List[PointPrompt], 
                         part_name: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """ハイブリッド精度向上手法の統合実行"""
        self.logger.info(f"🚀 ハイブリッド精度向上開始 - パーツ: {part_name}")
        
        start_time = time.time()
        enhancement_log = {
            "part_name": part_name,
            "phases": [],
            "total_prompts": len(base_prompts)
        }
        
        # Phase 1: 画像特性分析
        self.logger.info("📋 Phase 1: 画像特性分析")
        characteristics = self.analyze_image_characteristics(image)
        enhancement_log["characteristics"] = characteristics.__dict__
        
        # Phase 2: 適応的前処理
        self.logger.info("📋 Phase 2: 適応的前処理")
        processed_image = self.adaptive_preprocess_image(image, characteristics)
        enhancement_log["phases"].append("adaptive_preprocessing")
        
        # Phase 3: 適応的エッジ誘導プロンプト生成
        self.logger.info("📋 Phase 3: 適応的エッジ誘導プロンプト生成")
        edge_enhanced_prompts = self.generate_adaptive_edge_prompts(
            processed_image, base_prompts, characteristics
        )
        enhancement_log["edge_prompts_added"] = len(edge_enhanced_prompts) - len(base_prompts)
        enhancement_log["phases"].append("adaptive_edge_guided")
        
        # Phase 4: 高度な反復的改良
        self.logger.info("📋 Phase 4: 高度な反復的改良")
        final_masks, final_scores, final_logits, final_prompts = self.advanced_iterative_refinement(
            processed_image, edge_enhanced_prompts, characteristics
        )
        enhancement_log["final_prompts_count"] = len(final_prompts)
        enhancement_log["phases"].append("advanced_iterative")
        
        # 最終結果
        best_mask = final_masks[np.argmax(final_scores)]
        best_score = float(np.max(final_scores))
        
        total_time = time.time() - start_time
        enhancement_log["processing_time"] = total_time
        enhancement_log["final_score"] = best_score
        
        self.logger.info(f"🎉 ハイブリッド精度向上完了!")
        self.logger.info(f"  📊 最終スコア: {best_score:.3f}")
        self.logger.info(f"  ⏱️ 処理時間: {total_time:.2f}秒")
        self.logger.info(f"  🎯 プロンプト数: {len(base_prompts)} → {len(final_prompts)}")
        
        return best_mask, best_score, enhancement_log

def test_hybrid_enhancement(
    image_rgb: np.ndarray,
    part_name: str, 
    base_prompts: List[PointPrompt],
    enhancer: SAM2HybridEnhancer,
    output_dir: Path,
    image_name: str
) -> Dict[str, Any]:
    """ハイブリッド精度向上手法のテスト"""
    logger = get_logger(__name__)
    
    try:
        logger.info(f"🧪 ハイブリッド手法テスト開始 - {part_name}")
        
        # ハイブリッド手法実行
        best_mask, best_score, enhancement_log = enhancer.hybrid_enhancement(
            image_rgb, base_prompts, part_name
        )
        
        # 結果可視化
        overlay = image_rgb.copy().astype(np.float32)
        color = np.array([255, 100, 100])  # 赤系
        mask_bool = best_mask.astype(bool)  # ブール型に変換
        overlay[mask_bool] = overlay[mask_bool] * 0.6 + color * 0.4
        overlay_image = overlay.astype(np.uint8)
        
        # 情報表示
        info_lines = [
            f"Hybrid Enhancement: {part_name}",
            f"Score: {best_score:.3f}",
            f"Time: {enhancement_log['processing_time']:.2f}s",
            f"Prompts: {enhancement_log['total_prompts']} -> {enhancement_log['final_prompts_count']}",
            f"Phases: {len(enhancement_log['phases'])}",
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(overlay_image, line, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 保存
        output_path = output_dir / f"hybrid_{part_name}_{image_name}.png"
        overlay_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), overlay_bgr)
        
        # 詳細ログ保存
        log_path = output_dir / f"hybrid_{part_name}_{image_name}_log.txt"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"=== ハイブリッド精度向上ログ ===\n")
            f.write(f"パーツ: {part_name}\n")
            f.write(f"最終スコア: {best_score:.3f}\n")
            f.write(f"処理時間: {enhancement_log['processing_time']:.2f}秒\n")
            f.write(f"実行フェーズ: {', '.join(enhancement_log['phases'])}\n")
            f.write(f"\n=== 画像特性 ===\n")
            for key, value in enhancement_log['characteristics'].items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"✅ テスト完了 - スコア: {best_score:.3f}")
        
        return {
            "technique": "hybrid_enhancement",
            "part_name": part_name,
            "score": best_score,
            "inference_time": enhancement_log['processing_time'],
            "enhancement_log": enhancement_log,
            "output_path": str(output_path),
            "log_path": str(log_path)
        }
        
    except Exception as e:
        logger.error(f"❌ ハイブリッド手法テスト失敗 {part_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def main():
    """メイン実行"""
    
    # ログ設定
    setup_logging(level="INFO", console_output=True, structured=False)
    logger = get_logger(__name__)
    
    logger.info("=== SAM2ハイブリッド精度向上テスト ===")
    
    try:
        # 本番画像取得
        sample_images_path = project_root / "data" / "samples" / "demo_images2"
        image_files = list(sample_images_path.glob("*.png"))
        
        if not image_files:
            logger.error("❌ 本番画像が見つかりません (data/samples/demo_images2/)")
            return
        
        # 出力ディレクトリ準備
        output_dir = project_root / "data" / "output" / "sam2_hybrid_enhancement"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # SAM2モデル初期化
        logger.info("🤖 SAM2モデル初期化")
        sam2_manager = SAM2ModelManager(model_name="sam2_hiera_large.pt")
        if not sam2_manager.load_model():
            logger.error("❌ SAM2モデル読み込み失敗")
            return
        
        enhancer = SAM2HybridEnhancer(sam2_manager)
        
        # 本番画像でテスト
        image_path = image_files[0]
        logger.info(f"📸 本番画像: {image_path.name}")
        
        # 画像読み込み
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        
        # テスト対象：髪の毛パーツ
        from scripts.test_sam2_detailed_parts import MainPartPrompts
        hair_prompts = MainPartPrompts.hair((height, width))
        
        logger.info(f"🎯 テスト対象: 髪の毛 ({len(hair_prompts)}個の基本プロンプト)")
        
        # ベースライン測定（比較用）
        logger.info("📊 ベースライン測定開始")
        baseline_handler = SAM2PromptHandler()
        baseline_handler.start_new_session()
        
        for prompt in hair_prompts:
            baseline_handler.add_point_prompt(
                prompt.x, prompt.y, prompt.label, prompt.description
            )
        
        baseline_start = time.time()
        sam2_prompts = baseline_handler.get_sam2_prompts()
        masks, scores, logits = sam2_manager.predict(
            image=image_rgb,
            **sam2_prompts,
            multimask_output=True
        )
        baseline_time = time.time() - baseline_start
        baseline_score = float(np.max(scores))
        
        logger.info(f"📊 ベースライン結果: スコア {baseline_score:.3f}, 時間 {baseline_time:.2f}秒")
        
        # ハイブリッド手法実行
        logger.info("🚀 ハイブリッド手法実行開始")
        result = test_hybrid_enhancement(
            image_rgb, "hair", hair_prompts, 
            enhancer, output_dir, image_path.stem
        )
        
        if "error" not in result:
            # 結果比較
            hybrid_score = result["score"]
            hybrid_time = result["inference_time"]
            improvement = ((hybrid_score - baseline_score) / baseline_score) * 100
            
            logger.info(f"\n🏆 最終結果比較:")
            logger.info(f"  📊 ベースライン:     スコア {baseline_score:.3f}, 時間 {baseline_time:.2f}秒")
            logger.info(f"  🚀 ハイブリッド:     スコア {hybrid_score:.3f}, 時間 {hybrid_time:.2f}秒")
            logger.info(f"  📈 精度向上:         +{improvement:.1f}%")
            logger.info(f"  ⏱️ 処理時間比:       {hybrid_time/baseline_time:.1f}倍")
            
            if hybrid_score > 0.90:
                logger.info("🎉 目標スコア0.90を達成!")
            elif improvement > 5.0:
                logger.info("✨ 5%以上の精度向上を達成!")
            
            logger.info(f"\n📁 結果保存先: {output_dir}")
            logger.info(f"  🖼️ 結果画像: {result['output_path']}")
            logger.info(f"  📋 詳細ログ: {result['log_path']}")
        
        logger.info("🎉 ハイブリッド精度向上テスト完了!")
        
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