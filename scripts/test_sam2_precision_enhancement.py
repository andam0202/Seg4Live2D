#!/usr/bin/env python3
"""
SAM2精度向上テスト

様々な手法でSAM2の精度を向上させるテスト
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
class EnhancementTechnique:
    """精度向上手法"""
    name: str
    description: str
    enabled: bool = True

class SAM2PrecisionEnhancer:
    """SAM2精度向上クラス"""
    
    def __init__(self, sam2_manager: SAM2ModelManager):
        self.sam2_manager = sam2_manager
        self.logger = get_logger(__name__)
    
    def preprocess_image_contrast(self, image: np.ndarray, alpha: float = 1.5, beta: int = 20) -> np.ndarray:
        """コントラスト・明度調整による前処理"""
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        self.logger.info(f"  📈 コントラスト調整: α={alpha}, β={beta}")
        return enhanced
    
    def preprocess_image_sharpen(self, image: np.ndarray) -> np.ndarray:
        """シャープ化による前処理"""
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        self.logger.info(f"  🔍 シャープ化適用")
        return sharpened
    
    def preprocess_image_denoise(self, image: np.ndarray) -> np.ndarray:
        """ノイズ除去による前処理"""
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        self.logger.info(f"  🧹 ノイズ除去適用")
        return denoised
    
    def preprocess_image_clahe(self, image: np.ndarray) -> np.ndarray:
        """CLAHE（適応的ヒストグラム均等化）による前処理"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_channel = clahe.apply(l_channel)
        
        enhanced = cv2.merge((l_channel, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        self.logger.info(f"  ⚡ CLAHE適用")
        return enhanced
    
    def generate_dense_prompts(self, part_name: str, image_shape: Tuple[int, int], 
                             base_prompts: List[PointPrompt], density: int = 3) -> List[PointPrompt]:
        """密集プロンプト生成"""
        height, width = image_shape[:2]
        
        dense_prompts = base_prompts.copy()
        
        # 既存プロンプト周辺に追加点を生成
        for base_prompt in base_prompts:
            if base_prompt.label == 1:  # 前景点のみ
                for i in range(density):
                    # 半径内にランダム点を追加
                    radius = 20 + i * 10
                    angle = (2 * np.pi * i) / density
                    
                    new_x = int(base_prompt.x + radius * np.cos(angle))
                    new_y = int(base_prompt.y + radius * np.sin(angle))
                    
                    # 画像境界内チェック
                    if 0 <= new_x < width and 0 <= new_y < height:
                        dense_prompts.append(
                            PointPrompt(new_x, new_y, 1, 
                                      description=f"{base_prompt.description}_密集{i+1}")
                        )
        
        self.logger.info(f"  🎯 密集プロンプト生成: {len(base_prompts)} → {len(dense_prompts)}点")
        return dense_prompts
    
    def generate_edge_guided_prompts(self, image: np.ndarray, 
                                   base_prompts: List[PointPrompt]) -> List[PointPrompt]:
        """エッジ誘導プロンプト生成"""
        # Cannyエッジ検出
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        edge_prompts = base_prompts.copy()
        
        # エッジ上の点を追加プロンプトとして使用
        edge_points = np.column_stack(np.where(edges > 0))
        
        # サンプリング（全エッジ点は多すぎるため）
        if len(edge_points) > 20:
            indices = np.random.choice(len(edge_points), 20, replace=False)
            sampled_edges = edge_points[indices]
        else:
            sampled_edges = edge_points
        
        for y, x in sampled_edges:
            edge_prompts.append(
                PointPrompt(int(x), int(y), 1, description="エッジ誘導点")
            )
        
        self.logger.info(f"  🌐 エッジ誘導プロンプト追加: +{len(sampled_edges)}点")
        return edge_prompts
    
    def multi_scale_inference(self, image: np.ndarray, prompts: List[PointPrompt], 
                            scales: List[float] = [0.8, 1.0, 1.2]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """マルチスケール推論"""
        height, width = image.shape[:2]
        best_score = -1
        best_masks = None
        best_scores = None
        best_logits = None
        
        for scale in scales:
            self.logger.info(f"  📏 スケール {scale} で推論実行")
            
            # 画像リサイズ
            new_width = int(width * scale)
            new_height = int(height * scale)
            scaled_image = cv2.resize(image, (new_width, new_height))
            
            # プロンプト座標もスケール
            scaled_prompts = []
            for prompt in prompts:
                scaled_x = int(prompt.x * scale)
                scaled_y = int(prompt.y * scale)
                scaled_prompts.append(
                    PointPrompt(scaled_x, scaled_y, prompt.label, prompt.description)
                )
            
            # プロンプトハンドラー設定
            handler = SAM2PromptHandler()
            handler.start_new_session()
            
            for prompt in scaled_prompts:
                handler.add_point_prompt(
                    prompt.x, prompt.y, prompt.label, prompt.description
                )
            
            # 推論実行
            sam2_prompts = handler.get_sam2_prompts()
            masks, scores, logits = self.sam2_manager.predict(
                image=scaled_image,
                **sam2_prompts,
                multimask_output=True
            )
            
            # 結果を元サイズにリサイズ
            resized_masks = []
            for mask in masks:
                resized_mask = cv2.resize(mask.astype(np.uint8), (width, height))
                resized_masks.append(resized_mask.astype(bool))
            
            # 最高スコアを更新
            current_best_score = np.max(scores)
            if current_best_score > best_score:
                best_score = current_best_score
                best_masks = np.array(resized_masks)
                best_scores = scores
                best_logits = logits
                self.logger.info(f"    ✨ 新最高スコア: {current_best_score:.3f}")
        
        return best_masks, best_scores, best_logits
    
    def iterative_refinement(self, image: np.ndarray, initial_prompts: List[PointPrompt], 
                           max_iterations: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[PointPrompt]]:
        """反復的マスク改良"""
        current_prompts = initial_prompts.copy()
        best_score = -1
        best_result = None
        
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
            
            current_best_score = np.max(scores)
            current_best_mask = masks[np.argmax(scores)]
            
            self.logger.info(f"    スコア: {current_best_score:.3f}")
            
            if current_best_score > best_score:
                best_score = current_best_score
                best_result = (masks, scores, logits)
            
            # 次の反復のためのプロンプト改良
            if iteration < max_iterations - 1:
                # マスクエッジ付近に負例プロンプトを追加
                edges = cv2.Canny((current_best_mask * 255).astype(np.uint8), 50, 150)
                edge_points = np.column_stack(np.where(edges > 0))
                
                if len(edge_points) > 5:
                    # ランダムに5点選択して負例プロンプトとして追加
                    indices = np.random.choice(len(edge_points), 5, replace=False)
                    for idx in indices:
                        y, x = edge_points[idx]
                        # マスク外側の点を負例として追加
                        if not current_best_mask[y, x]:
                            current_prompts.append(
                                PointPrompt(int(x), int(y), 0, f"反復改良_負例_{iteration}")
                            )
        
        return best_result[0], best_result[1], best_result[2], current_prompts
    
    def ensemble_prediction(self, image: np.ndarray, prompts: List[PointPrompt], 
                          techniques: List[str] = ['normal', 'contrast', 'sharpen']) -> Tuple[np.ndarray, float]:
        """アンサンブル予測"""
        self.logger.info(f"  🎭 アンサンブル予測: {techniques}")
        
        all_masks = []
        all_scores = []
        
        for technique in techniques:
            processed_image = image.copy()
            
            # 前処理適用
            if technique == 'contrast':
                processed_image = self.preprocess_image_contrast(processed_image)
            elif technique == 'sharpen':
                processed_image = self.preprocess_image_sharpen(processed_image)
            elif technique == 'denoise':
                processed_image = self.preprocess_image_denoise(processed_image)
            elif technique == 'clahe':
                processed_image = self.preprocess_image_clahe(processed_image)
            
            # プロンプトハンドラー設定
            handler = SAM2PromptHandler()
            handler.start_new_session()
            
            for prompt in prompts:
                handler.add_point_prompt(
                    prompt.x, prompt.y, prompt.label, prompt.description
                )
            
            # 推論実行
            sam2_prompts = handler.get_sam2_prompts()
            masks, scores, logits = self.sam2_manager.predict(
                image=processed_image,
                **sam2_prompts,
                multimask_output=True
            )
            
            best_idx = np.argmax(scores)
            all_masks.append(masks[best_idx])
            all_scores.append(scores[best_idx])
        
        # アンサンブル統合（多数決）
        ensemble_mask = np.mean(all_masks, axis=0) > 0.5
        ensemble_score = np.mean(all_scores)
        
        self.logger.info(f"    🏆 アンサンブルスコア: {ensemble_score:.3f}")
        
        return ensemble_mask, ensemble_score

def test_enhancement_technique(
    image_rgb: np.ndarray,
    part_name: str, 
    base_prompts: List[PointPrompt],
    technique: EnhancementTechnique,
    enhancer: SAM2PrecisionEnhancer,
    output_dir: Path,
    image_name: str
) -> Dict[str, Any]:
    """個別の精度向上手法をテスト"""
    logger = get_logger(__name__)
    
    try:
        logger.info(f"  🧪 手法テスト: {technique.name}")
        
        start_time = time.time()
        
        if technique.name == "baseline":
            # ベースライン（通常のSAM2）
            handler = SAM2PromptHandler()
            handler.start_new_session()
            
            for prompt in base_prompts:
                handler.add_point_prompt(
                    prompt.x, prompt.y, prompt.label, prompt.description
                )
            
            sam2_prompts = handler.get_sam2_prompts()
            masks, scores, logits = enhancer.sam2_manager.predict(
                image=image_rgb,
                **sam2_prompts,
                multimask_output=True
            )
            
            best_mask = masks[np.argmax(scores)].astype(bool)
            best_score = float(np.max(scores))
        
        elif technique.name == "dense_prompts":
            # 密集プロンプト
            dense_prompts = enhancer.generate_dense_prompts(
                part_name, image_rgb.shape[:2], base_prompts, density=4
            )
            
            handler = SAM2PromptHandler()
            handler.start_new_session()
            
            for prompt in dense_prompts:
                handler.add_point_prompt(
                    prompt.x, prompt.y, prompt.label, prompt.description
                )
            
            sam2_prompts = handler.get_sam2_prompts()
            masks, scores, logits = enhancer.sam2_manager.predict(
                image=image_rgb,
                **sam2_prompts,
                multimask_output=True
            )
            
            best_mask = masks[np.argmax(scores)].astype(bool)
            best_score = float(np.max(scores))
        
        elif technique.name == "edge_guided":
            # エッジ誘導プロンプト
            edge_prompts = enhancer.generate_edge_guided_prompts(image_rgb, base_prompts)
            
            handler = SAM2PromptHandler()
            handler.start_new_session()
            
            for prompt in edge_prompts:
                handler.add_point_prompt(
                    prompt.x, prompt.y, prompt.label, prompt.description
                )
            
            sam2_prompts = handler.get_sam2_prompts()
            masks, scores, logits = enhancer.sam2_manager.predict(
                image=image_rgb,
                **sam2_prompts,
                multimask_output=True
            )
            
            best_mask = masks[np.argmax(scores)].astype(bool)
            best_score = float(np.max(scores))
        
        elif technique.name == "multi_scale":
            # マルチスケール推論
            masks, scores, logits = enhancer.multi_scale_inference(
                image_rgb, base_prompts, scales=[0.8, 1.0, 1.2, 1.4]
            )
            
            best_mask = masks[np.argmax(scores)].astype(bool)
            best_score = float(np.max(scores))
        
        elif technique.name == "iterative":
            # 反復的改良
            masks, scores, logits, final_prompts = enhancer.iterative_refinement(
                image_rgb, base_prompts, max_iterations=3
            )
            
            best_mask = masks[np.argmax(scores)].astype(bool)
            best_score = float(np.max(scores))
        
        elif technique.name == "ensemble":
            # アンサンブル予測
            best_mask, best_score = enhancer.ensemble_prediction(
                image_rgb, base_prompts, 
                techniques=['normal', 'contrast', 'sharpen', 'clahe']
            )
        
        else:
            # 前処理ベース手法
            processed_image = image_rgb.copy()
            
            if technique.name == "contrast_enhanced":
                processed_image = enhancer.preprocess_image_contrast(processed_image)
            elif technique.name == "sharpened":
                processed_image = enhancer.preprocess_image_sharpen(processed_image)
            elif technique.name == "denoised":
                processed_image = enhancer.preprocess_image_denoise(processed_image)
            elif technique.name == "clahe":
                processed_image = enhancer.preprocess_image_clahe(processed_image)
            
            handler = SAM2PromptHandler()
            handler.start_new_session()
            
            for prompt in base_prompts:
                handler.add_point_prompt(
                    prompt.x, prompt.y, prompt.label, prompt.description
                )
            
            sam2_prompts = handler.get_sam2_prompts()
            masks, scores, logits = enhancer.sam2_manager.predict(
                image=processed_image,
                **sam2_prompts,
                multimask_output=True
            )
            
            best_mask = masks[np.argmax(scores)].astype(bool)
            best_score = float(np.max(scores))
        
        inference_time = time.time() - start_time
        mask_coverage = float(np.sum(best_mask) / best_mask.size)
        
        # 結果可視化
        overlay = image_rgb.copy().astype(np.float32)
        color = np.array([255, 100, 100])  # 赤系
        overlay[best_mask] = overlay[best_mask] * 0.6 + color * 0.4
        overlay_image = overlay.astype(np.uint8)
        
        # 情報表示
        info_lines = [
            f"Technique: {technique.name}",
            f"Score: {best_score:.3f}",
            f"Time: {inference_time:.2f}s",
            f"Coverage: {mask_coverage*100:.1f}%",
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(overlay_image, line, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 保存
        output_path = output_dir / f"{technique.name}_{part_name}_{image_name}.png"
        overlay_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), overlay_bgr)
        
        logger.info(f"    📊 結果: スコア {best_score:.3f}, 時間 {inference_time:.2f}s")
        
        return {
            "technique": technique.name,
            "part_name": part_name,
            "score": best_score,
            "inference_time": inference_time,
            "mask_coverage": mask_coverage,
            "output_path": str(output_path)
        }
        
    except Exception as e:
        logger.error(f"手法テスト失敗 {technique.name}: {e}")
        return {"error": str(e)}

def main():
    """メイン実行"""
    
    # ログ設定
    setup_logging(level="INFO", console_output=True, structured=False)
    logger = get_logger(__name__)
    
    logger.info("=== SAM2精度向上テスト ===")
    
    try:
        # 本番画像取得
        sample_images_path = project_root / "data" / "samples" / "demo_images2"
        image_files = list(sample_images_path.glob("*.png"))
        
        if not image_files:
            logger.error("❌ 本番画像が見つかりません (data/samples/demo_images2/)")
            return
        
        if len(image_files) > 1:
            logger.warning(f"複数画像が見つかりました。最初の画像を使用: {image_files[0].name}")
        
        # 出力ディレクトリ準備
        output_dir = project_root / "data" / "output" / "sam2_precision_enhancement"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # SAM2モデル初期化
        sam2_manager = SAM2ModelManager(model_name="sam2_hiera_large.pt")
        if not sam2_manager.load_model():
            logger.error("❌ SAM2モデル読み込み失敗")
            return
        
        enhancer = SAM2PrecisionEnhancer(sam2_manager)
        
        # 精度向上手法定義
        enhancement_techniques = [
            EnhancementTechnique("baseline", "ベースライン（通常のSAM2）"),
            EnhancementTechnique("contrast_enhanced", "コントラスト強化"),
            EnhancementTechnique("sharpened", "シャープ化"),
            EnhancementTechnique("denoised", "ノイズ除去"),
            EnhancementTechnique("clahe", "適応的ヒストグラム均等化"),
            EnhancementTechnique("dense_prompts", "密集プロンプト"),
            EnhancementTechnique("edge_guided", "エッジ誘導プロンプト"),
            EnhancementTechnique("multi_scale", "マルチスケール推論"),
            EnhancementTechnique("iterative", "反復的マスク改良"),
            EnhancementTechnique("ensemble", "アンサンブル予測"),
        ]
        
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
        
        all_results = []
        
        # 各手法をテスト
        for technique in enhancement_techniques:
            logger.info(f"\\n🔬 精度向上手法: {technique.name}")
            logger.info(f"   {technique.description}")
            
            result = test_enhancement_technique(
                image_rgb, "hair", hair_prompts, technique, 
                enhancer, output_dir, image_path.stem
            )
            
            if "error" not in result:
                all_results.append(result)
        
        # 結果分析
        logger.info(f"\\n📊 精度向上結果分析:")
        
        # スコア順にソート
        sorted_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
        
        baseline_score = None
        for result in sorted_results:
            if result["technique"] == "baseline":
                baseline_score = result["score"]
                break
        
        logger.info(f"\\n🏆 手法別性能ランキング:")
        for i, result in enumerate(sorted_results):
            technique = result["technique"]
            score = result["score"]
            time_taken = result["inference_time"]
            
            improvement = ""
            if baseline_score and technique != "baseline":
                improvement_pct = ((score - baseline_score) / baseline_score) * 100
                improvement = f" ({improvement_pct:+.1f}%)"
            
            logger.info(f"  {i+1:2d}. {technique:20} - スコア: {score:.3f}{improvement}, 時間: {time_taken:.2f}s")
        
        # 最高性能手法
        best_technique = sorted_results[0]
        logger.info(f"\\n🥇 最高性能手法: {best_technique['technique']}")
        logger.info(f"   スコア: {best_technique['score']:.3f}")
        logger.info(f"   処理時間: {best_technique['inference_time']:.2f}s")
        
        if baseline_score:
            improvement = ((best_technique['score'] - baseline_score) / baseline_score) * 100
            logger.info(f"   ベースライン比改善: +{improvement:.1f}%")
        
        logger.info(f"\\n🎉 精度向上テスト完了!")
        logger.info(f"結果: {output_dir}")
        
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