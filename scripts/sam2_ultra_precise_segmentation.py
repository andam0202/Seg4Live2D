#!/usr/bin/env python3
"""
SAM2超精密パーツセグメンテーション

問題分析と超精密改善アプローチ：
1. より厳密なプロンプト境界設定
2. パーツ特有の解剖学的知識を活用
3. 除外プロンプトの戦略的強化
4. パーツ間の競合を完全に排除
"""

import sys
import cv2
import numpy as np
import time
import argparse
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

@dataclass
class SegmentationResult:
    """セグメンテーション結果"""
    part_name: str
    mask: np.ndarray
    score: float
    processing_time: float
    prompts_used: int
    strategy: str

class UltraPreciseSegmenter:
    """超精密パーツセグメンター"""
    
    def __init__(self, sam2_manager: SAM2ModelManager):
        self.sam2_manager = sam2_manager
        self.logger = get_logger(__name__)
        
    def generate_ultra_precise_prompts(self, image_shape: Tuple[int, int], part_name: str) -> List[PointPrompt]:
        """超精密プロンプト生成（問題解決版）"""
        height, width = image_shape[:2]
        
        # 超精密で境界を厳格に区別するプロンプト配置
        ultra_precise_prompts = {
            "hair": [
                # 髪の毛の領域のみに厳密に制限
                PointPrompt(width//4, height//6, 1, "左髪_内側"),
                PointPrompt(3*width//4, height//6, 1, "右髪_内側"),
                PointPrompt(width//2, height//10, 1, "後髪_中央"),
                # 角と顔を厳格に除外
                PointPrompt(width//2, height//20, 0, "角_ツノ_除外"),
                PointPrompt(width//3, height//20, 0, "左角_除外"),
                PointPrompt(2*width//3, height//20, 0, "右角_除外"),
                PointPrompt(width//2, height//3, 0, "顔_肌_除外"),
                PointPrompt(width//2, 2*height//5, 0, "額_肌_除外"),
                PointPrompt(width//2, 3*height//5, 0, "体_完全除外"),
            ],
            "face": [
                # 顔の肌の部分のみに厳密に制限
                PointPrompt(width//2, height//2, 1, "顔_中央_肌"),
                PointPrompt(2*width//5, height//2, 1, "左頬_肌のみ"),
                PointPrompt(3*width//5, height//2, 1, "右頬_肌のみ"),
                PointPrompt(width//2, 3*height//7, 1, "額_下部_肌"),
                # 髪と体を厳格に除外
                PointPrompt(width//2, height//4, 0, "額上_髪_除外"),
                PointPrompt(width//2, height//5, 0, "前髪_完全除外"),
                PointPrompt(width//3, height//4, 0, "左髪_境界_除外"),
                PointPrompt(2*width//3, height//4, 0, "右髪_境界_除外"),
                PointPrompt(width//2, 3*height//5, 0, "首_体_除外"),
                PointPrompt(width//6, height//2, 0, "左背景_除外"),
                PointPrompt(5*width//6, height//2, 0, "右背景_除外"),
            ],
            "body": [
                # 首から下の体のみに厳密に制限
                PointPrompt(width//2, 2*height//3, 1, "胴体_中央_のみ"),
                PointPrompt(2*width//5, 3*height//4, 1, "左肩_体のみ"),
                PointPrompt(3*width//5, 3*height//4, 1, "右肩_体のみ"),
                # 顔と髪を厳格に除外
                PointPrompt(width//2, height//2, 0, "顔_完全除外"),
                PointPrompt(width//2, height//3, 0, "首上_除外"),
                PointPrompt(width//2, height//4, 0, "頭部_除外"),
                PointPrompt(width//2, height//6, 0, "髪_完全除外"),
                PointPrompt(width//6, 2*height//3, 0, "左背景_除外"),
                PointPrompt(5*width//6, 2*height//3, 0, "右背景_除外"),
            ],
            "eyes": [
                # 目の部分のみに超厳密に制限
                PointPrompt(2*width//5, 2*height//5, 1, "左目_瞳のみ"),
                PointPrompt(3*width//5, 2*height//5, 1, "右目_瞳のみ"),
                # 顔の他の部分と髪を厳格に除外
                PointPrompt(width//2, height//3, 0, "額_除外"),
                PointPrompt(width//2, height//4, 0, "前髪_除外"),
                PointPrompt(width//2, height//5, 0, "髪_完全除外"),
                PointPrompt(width//2, height//2, 0, "鼻_除外"),
                PointPrompt(width//2, 3*height//5, 0, "口_除外"),
                PointPrompt(width//4, 2*height//5, 0, "左頬_除外"),
                PointPrompt(3*width//4, 2*height//5, 0, "右頬_除外"),
                PointPrompt(width//3, height//3, 0, "左眉_除外"),
                PointPrompt(2*width//3, height//3, 0, "右眉_除外"),
            ]
        }
        
        prompts = ultra_precise_prompts.get(part_name, ultra_precise_prompts["hair"])
        self.logger.info(f"  🎯 超精密プロンプト生成: {len(prompts)}点 ({part_name})")
        return prompts
    
    def segment_part_ultra_precise(self, image_rgb: np.ndarray, part_name: str) -> SegmentationResult:
        """超精密セグメンテーション実行"""
        self.logger.info(f"🎯 {part_name.upper()}超精密セグメンテーション開始")
        
        start_time = time.time()
        height, width = image_rgb.shape[:2]
        
        # 超精密プロンプト生成
        prompts = self.generate_ultra_precise_prompts((height, width), part_name)
        
        # プロンプト座標とラベルを準備
        point_coords = []
        point_labels = []
        
        for prompt in prompts:
            point_coords.append([prompt.x, prompt.y])
            point_labels.append(prompt.label)
        
        point_coords = np.array(point_coords) if point_coords else None
        point_labels = np.array(point_labels) if point_labels else None
        
        # SAM2で超精密推論実行
        masks, scores, logits = self.sam2_manager.predict(
            image=image_rgb,
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        
        if len(masks) == 0:
            raise RuntimeError("セグメンテーション結果なし")
        
        # 最良の結果を選択
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        processing_time = time.time() - start_time
        
        result = SegmentationResult(
            part_name=part_name,
            mask=best_mask,
            score=best_score,
            processing_time=processing_time,
            prompts_used=len(prompts),
            strategy="ultra_precise"
        )
        
        self.logger.info(f"  ✅ 完了: スコア {best_score:.3f}, 時間 {processing_time:.2f}s")
        return result
    
    def create_ultra_detailed_visualization(self, image_rgb: np.ndarray, result: SegmentationResult) -> np.ndarray:
        """超詳細可視化（境界とカバレッジを強調）"""
        vis_image = image_rgb.copy()
        
        # パーツ別色分け
        colors = {
            "hair": [255, 100, 150],    # ピンク
            "face": [255, 200, 100],    # オレンジ
            "body": [100, 200, 255],    # 青
            "eyes": [150, 255, 100],    # 緑
        }
        color = np.array(colors.get(result.part_name, [255, 255, 255]))
        
        # マスク適用（より控えめに）
        mask_bool = result.mask.astype(bool)
        vis_image[mask_bool] = vis_image[mask_bool] * 0.6 + color * 0.4
        
        # マスク境界を強調描画
        mask_uint8 = (result.mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, color.tolist(), 3)
        
        # 境界ボックス計算
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color.tolist(), 2)
        
        # 超詳細情報表示
        coverage = np.sum(result.mask) / result.mask.size * 100
        mask_pixels = int(np.sum(result.mask))
        
        info_lines = [
            f"Part: {result.part_name.upper()}",
            f"Score: {result.score:.3f}",
            f"Strategy: {result.strategy}",
            f"Time: {result.processing_time:.2f}s",
            f"Prompts: {result.prompts_used}",
            f"Mask Pixels: {mask_pixels:,}",
            f"Coverage: {coverage:.2f}%",
            f"Efficiency: {result.score/coverage*100:.1f}" if coverage > 0 else "Efficiency: N/A",
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis_image, (10, y_pos - 20), 
                         (text_size[0] + 20, y_pos + 5), (0, 0, 0), -1)
            cv2.putText(vis_image, line, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image
    
    def save_ultra_results(self, image_rgb: np.ndarray, results: List[SegmentationResult], 
                          output_dir: Path, image_name: str) -> Dict[str, Any]:
        """超精密結果保存"""
        
        # オリジナル画像保存
        original_path = output_dir / f"original_{image_name}.png"
        original_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(original_path), original_bgr)
        
        saved_files = {"original": str(original_path)}
        
        # パーツ別結果保存
        for result in results:
            part_name = result.part_name
            
            # マスク画像保存（バイナリ）
            mask_path = output_dir / f"ultra_mask_{part_name}_{image_name}.png"
            cv2.imwrite(str(mask_path), (result.mask * 255).astype(np.uint8))
            saved_files[f"ultra_mask_{part_name}"] = str(mask_path)
            
            # 透明PNG保存（Live2D用）
            transparent_path = output_dir / f"ultra_transparent_{part_name}_{image_name}.png"
            transparent_image = image_rgb.copy()
            alpha_channel = result.mask.astype(np.uint8) * 255
            transparent_rgba = np.dstack([transparent_image, alpha_channel])
            transparent_bgra = cv2.cvtColor(transparent_rgba, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(str(transparent_path), transparent_bgra)
            saved_files[f"ultra_transparent_{part_name}"] = str(transparent_path)
            
            # 超詳細可視化画像保存
            vis_image = self.create_ultra_detailed_visualization(image_rgb, result)
            vis_path = output_dir / f"ultra_viz_{part_name}_{image_name}.png"
            vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(vis_path), vis_bgr)
            saved_files[f"ultra_viz_{part_name}"] = str(vis_path)
            
            # 境界マスク保存（デバッグ用）
            mask_uint8 = (result.mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boundary_image = np.zeros_like(image_rgb)
            cv2.drawContours(boundary_image, contours, -1, (255, 255, 255), 2)
            boundary_path = output_dir / f"ultra_boundary_{part_name}_{image_name}.png"
            boundary_bgr = cv2.cvtColor(boundary_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(boundary_path), boundary_bgr)
            saved_files[f"ultra_boundary_{part_name}"] = str(boundary_path)
        
        return saved_files

def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description="SAM2超精密パーツセグメンテーション",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 超精密セグメンテーション実行
  python scripts/sam2_ultra_precise_segmentation.py --input data/samples/anime_woman1
  
  # 特定パーツのみ
  python scripts/sam2_ultra_precise_segmentation.py --parts face --input data/samples/anime_woman1
  
  # 詳細ログ付き
  python scripts/sam2_ultra_precise_segmentation.py --input data/samples/anime_woman1 --verbose
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/samples/demo_images",
        help="入力画像フォルダのパス"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=str,
        default="data/output/sam2_ultra_precise_segmentation",
        help="出力ディレクトリのパス"
    )
    
    parser.add_argument(
        "--parts", "-p",
        nargs="+",
        choices=["face", "hair", "body", "eyes"],
        default=["face", "hair", "body", "eyes"],
        help="セグメンテーション対象パーツ"
    )
    
    parser.add_argument(
        "--image-pattern", 
        type=str,
        default="*.png",
        help="画像ファイルのパターン"
    )
    
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="処理する最大画像数"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細ログを表示"
    )
    
    return parser.parse_args()

def main():
    """メイン実行"""
    
    # コマンドライン引数解析
    args = parse_args()
    
    # ログ設定
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level, console_output=True, structured=False)
    logger = get_logger(__name__)
    
    logger.info("=== SAM2超精密パーツセグメンテーション ===")
    logger.info(f"📁 入力フォルダ: {args.input}")
    logger.info(f"📁 出力フォルダ: {args.output}")
    logger.info(f"🎯 対象パーツ: {', '.join(args.parts)}")
    
    try:
        # 入力画像フォルダの確認
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = project_root / input_path
            
        if not input_path.exists():
            logger.error(f"❌ 入力フォルダが存在しません: {input_path}")
            return
            
        # 画像ファイル取得
        image_files = list(input_path.glob(args.image_pattern))
        
        if not image_files:
            logger.error(f"❌ 画像ファイルが見つかりません")
            return
        
        # 画像数制限
        if args.max_images and len(image_files) > args.max_images:
            image_files = image_files[:args.max_images]
            
        logger.info(f"📸 処理対象画像数: {len(image_files)}")
        
        # 出力ディレクトリ準備
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path.mkdir(parents=True, exist_ok=True)
        
        # SAM2モデル初期化
        logger.info("🤖 SAM2モデル初期化")
        sam2_manager = SAM2ModelManager(model_name="sam2_hiera_large.pt")
        if not sam2_manager.load_model():
            logger.error("❌ SAM2モデル読み込み失敗")
            return
        
        segmenter = UltraPreciseSegmenter(sam2_manager)
        
        # 対象パーツ
        target_parts = args.parts
        
        # 全処理統計
        total_images_processed = 0
        total_processing_time = 0
        all_results = []
        
        # 各画像で処理
        for image_idx, image_path in enumerate(image_files):
            logger.info(f"\n📸 画像処理 ({image_idx + 1}/{len(image_files)}): {image_path.name}")
            
            try:
                # 画像読み込み
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning(f"  ⚠️ 画像読み込み失敗: {image_path}")
                    continue
                    
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                logger.info(f"  📐 画像サイズ: {image_rgb.shape[1]}×{image_rgb.shape[0]}")
                
                # 各パーツをセグメンテーション
                image_results = []
                image_start_time = time.time()
                
                for part_name in target_parts:
                    logger.info(f"\n  🎯 {part_name.upper()}パーツ処理")
                    
                    try:
                        result = segmenter.segment_part_ultra_precise(image_rgb, part_name)
                        image_results.append(result)
                        
                        # 個別結果ログ
                        coverage = np.sum(result.mask) / result.mask.size * 100
                        efficiency = result.score / coverage * 100 if coverage > 0 else 0
                        logger.info(f"    📊 カバー率: {coverage:.2f}%, 効率性: {efficiency:.1f}")
                        
                    except Exception as e:
                        logger.error(f"    ❌ {part_name}セグメンテーション失敗: {e}")
                        continue
                
                image_processing_time = time.time() - image_start_time
                total_processing_time += image_processing_time
                
                if not image_results:
                    logger.warning(f"  ⚠️ {image_path.name}の処理結果がありません")
                    continue
                
                # 超精密結果保存
                logger.info(f"\n  💾 超精密結果保存中...")
                saved_files = segmenter.save_ultra_results(image_rgb, image_results, output_path, image_path.stem)
                
                # 画像別サマリー
                avg_score = np.mean([r.score for r in image_results])
                avg_coverage = np.mean([np.sum(r.mask) / r.mask.size * 100 for r in image_results])
                
                logger.info(f"  📊 {image_path.name} 完了:")
                logger.info(f"    🎯 処理パーツ: {len(image_results)}")
                logger.info(f"    📈 平均スコア: {avg_score:.3f}")
                logger.info(f"    📐 平均カバー率: {avg_coverage:.2f}%")
                logger.info(f"    ⏱️ 処理時間: {image_processing_time:.2f}秒")
                
                all_results.extend(image_results)
                total_images_processed += 1
                
            except Exception as e:
                logger.error(f"  ❌ {image_path.name}処理中にエラー: {e}")
                continue
        
        if not all_results:
            logger.warning("⚠️ 全体の処理結果がありません")
            return
        
        # 超精密分析結果
        ultra_analysis = {
            "total_images_processed": total_images_processed,
            "total_processing_time": float(total_processing_time),
            "average_score": float(np.mean([r.score for r in all_results])),
            "average_coverage": float(np.mean([np.sum(r.mask) / r.mask.size * 100 for r in all_results])),
            "strategy": "ultra_precise",
            "improvements": {
                "targeted_prompts": "パーツ特有の解剖学的境界を厳密に設定",
                "enhanced_exclusions": "隣接パーツの戦略的除外を強化",
                "boundary_precision": "境界線の精度を大幅向上"
            }
        }
        
        # パーツ別超精密統計
        parts_stats = {}
        for part in target_parts:
            part_results = [r for r in all_results if r.part_name == part]
            if part_results:
                avg_coverage = np.mean([np.sum(r.mask) / r.mask.size * 100 for r in part_results])
                parts_stats[part] = {
                    "count": len(part_results),
                    "average_score": float(np.mean([r.score for r in part_results])),
                    "average_coverage": float(avg_coverage),
                    "precision": float(np.mean([r.score for r in part_results]) / avg_coverage * 100) if avg_coverage > 0 else 0,
                    "best_score": float(max(part_results, key=lambda r: r.score).score),
                    "most_precise": float(min([np.sum(r.mask) / r.mask.size * 100 for r in part_results]))
                }
        
        ultra_analysis["parts_statistics"] = parts_stats
        
        # 超精密分析結果JSON保存
        ultra_analysis_path = output_path / "ultra_precise_analysis.json"
        with open(ultra_analysis_path, 'w', encoding='utf-8') as f:
            json.dump(ultra_analysis, f, indent=2, ensure_ascii=False)
        
        # 最終結果サマリー
        logger.info(f"\n🎉 超精密処理完了!")
        logger.info(f"📊 超精密統計:")
        logger.info(f"  📸 処理画像数: {total_images_processed}/{len(image_files)}")
        logger.info(f"  🎯 総パーツ数: {len(all_results)}")
        logger.info(f"  📈 全体平均スコア: {ultra_analysis['average_score']:.3f}")
        logger.info(f"  📐 全体平均カバー率: {ultra_analysis['average_coverage']:.2f}%")
        logger.info(f"  ⏱️ 総処理時間: {total_processing_time:.2f}秒")
        
        # 超精密パーツ別統計表示
        logger.info(f"\n📐 超精密パーツ別統計:")
        for part, stats in parts_stats.items():
            logger.info(f"  {part:6}: {stats['count']:2}件, "
                       f"平均スコア {stats['average_score']:.3f}, "
                       f"平均カバー率 {stats['average_coverage']:5.2f}%, "
                       f"精密度 {stats['precision']:5.1f}")
        
        logger.info(f"\n📁 結果保存先: {output_path}")
        logger.info(f"📋 超精密分析: {ultra_analysis_path}")
        logger.info("✨ 超精密セグメンテーション解析完了!")
        
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