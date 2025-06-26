#!/usr/bin/env python3
"""
SAM2 Live2Dパーツセグメンテーション

Live2D制作用の高精度パーツ分割システム
- 髪、顔、体、目の自動分割
- 透明PNG出力でLive2D即利用可能
- argparse対応で柔軟な実行オプション
"""

import sys
import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import json

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils import setup_logging, get_logger
from src.core.sam2.sam2_model import SAM2ModelManager
from src.core.sam2.prompt_handler import PointPrompt

@dataclass
class SegmentationResult:
    """セグメンテーション結果"""
    part_name: str
    mask: np.ndarray
    score: float
    processing_time: float

class Live2DSegmenter:
    """Live2D用パーツセグメンター"""
    
    def __init__(self, sam2_manager: SAM2ModelManager):
        self.sam2_manager = sam2_manager
        self.logger = get_logger(__name__)
        
    def get_optimized_prompts(self, image_shape: Tuple[int, int], part_name: str) -> List[PointPrompt]:
        """最適化されたプロンプト生成"""
        height, width = image_shape[:2]
        
        # 実験で最適化されたプロンプト配置
        prompts = {
            "hair": [
                # 髪の中核部分（角を除外）
                PointPrompt(width//4, height//6, 1, "左髪_内側"),
                PointPrompt(3*width//4, height//6, 1, "右髪_内側"),
                PointPrompt(width//2, height//10, 1, "後髪_中央"),
                # 強力な除外
                PointPrompt(width//2, height//20, 0, "角_除外"),
                PointPrompt(width//2, height//3, 0, "顔_除外"),
                PointPrompt(width//2, 3*height//5, 0, "体_除外"),
            ],
            "face": [
                # 顔の肌部分のみ
                PointPrompt(width//2, height//2, 1, "顔_中央"),
                PointPrompt(2*width//5, height//2, 1, "左頬"),
                PointPrompt(3*width//5, height//2, 1, "右頬"),
                PointPrompt(width//2, 3*height//7, 1, "額_下部"),
                # 髪と体を除外
                PointPrompt(width//2, height//4, 0, "前髪_除外"),
                PointPrompt(width//2, 3*height//5, 0, "首_除外"),
                PointPrompt(width//6, height//2, 0, "背景_除外"),
            ],
            "body": [
                # 首から下の体
                PointPrompt(width//2, 2*height//3, 1, "胴体_中央"),
                PointPrompt(2*width//5, 3*height//4, 1, "左肩"),
                PointPrompt(3*width//5, 3*height//4, 1, "右肩"),
                # 顔を除外
                PointPrompt(width//2, height//2, 0, "顔_除外"),
                PointPrompt(width//2, height//3, 0, "首上_除外"),
                PointPrompt(width//6, 2*height//3, 0, "背景_除外"),
            ],
            "eyes": [
                # 目の部分のみ（最高効率性達成済み）
                PointPrompt(2*width//5, 2*height//5, 1, "左目"),
                PointPrompt(3*width//5, 2*height//5, 1, "右目"),
                # 厳格な除外
                PointPrompt(width//2, height//3, 0, "額_除外"),
                PointPrompt(width//2, height//4, 0, "前髪_除外"),
                PointPrompt(width//2, height//2, 0, "鼻_除外"),
                PointPrompt(width//4, 2*height//5, 0, "左頬_除外"),
                PointPrompt(3*width//4, 2*height//5, 0, "右頬_除外"),
            ]
        }
        
        return prompts.get(part_name, prompts["hair"])
    
    def segment_part(self, image_rgb: np.ndarray, part_name: str) -> SegmentationResult:
        """パーツセグメンテーション実行"""
        self.logger.info(f"🎯 {part_name.upper()}セグメンテーション開始")
        
        start_time = time.time()
        prompts = self.get_optimized_prompts(image_rgb.shape, part_name)
        
        # プロンプト準備
        point_coords = [[p.x, p.y] for p in prompts]
        point_labels = [p.label for p in prompts]
        
        # SAM2実行
        masks, scores, _ = self.sam2_manager.predict(
            image=image_rgb,
            point_coords=np.array(point_coords),
            point_labels=np.array(point_labels),
            multimask_output=True
        )
        
        # 最良結果選択
        best_idx = np.argmax(scores)
        processing_time = time.time() - start_time
        
        result = SegmentationResult(
            part_name=part_name,
            mask=masks[best_idx],
            score=scores[best_idx],
            processing_time=processing_time
        )
        
        coverage = np.sum(result.mask) / result.mask.size * 100
        self.logger.info(f"  ✅ 完了: スコア {result.score:.3f}, カバー率 {coverage:.1f}%")
        
        return result
    
    def create_visualization(self, image_rgb: np.ndarray, result: SegmentationResult) -> np.ndarray:
        """可視化画像作成"""
        vis_image = image_rgb.copy()
        
        # パーツ別色分け
        colors = {
            "hair": [255, 100, 150],    # ピンク
            "face": [255, 200, 100],    # オレンジ
            "body": [100, 200, 255],    # 青
            "eyes": [150, 255, 100],    # 緑
        }
        color = np.array(colors.get(result.part_name, [255, 255, 255]))
        
        # マスク適用とオーバーレイ
        mask_bool = result.mask.astype(bool)
        vis_image[mask_bool] = vis_image[mask_bool] * 0.6 + color * 0.4
        
        # 境界線描画
        mask_uint8 = (result.mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, color.tolist(), 2)
        
        # 情報表示
        coverage = np.sum(result.mask) / result.mask.size * 100
        info_lines = [
            f"Part: {result.part_name.upper()}",
            f"Score: {result.score:.3f}",
            f"Time: {result.processing_time:.2f}s",
            f"Coverage: {coverage:.1f}%",
        ]
        
        for i, line in enumerate(info_lines):
            y = 30 + i * 30
            cv2.putText(vis_image, line, (15, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image
    
    def save_results(self, image_rgb: np.ndarray, results: List[SegmentationResult], 
                    output_dir: Path, image_name: str) -> Dict[str, str]:
        """結果保存（Live2D用）"""
        saved_files = {}
        
        # オリジナル画像
        original_path = output_dir / f"original_{image_name}.png"
        cv2.imwrite(str(original_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        saved_files["original"] = str(original_path)
        
        for result in results:
            part = result.part_name
            
            # マスク（白黒）
            mask_path = output_dir / f"mask_{part}_{image_name}.png"
            cv2.imwrite(str(mask_path), (result.mask * 255).astype(np.uint8))
            saved_files[f"mask_{part}"] = str(mask_path)
            
            # 透明PNG（Live2D用）
            transparent_path = output_dir / f"live2d_{part}_{image_name}.png"
            alpha = result.mask.astype(np.uint8) * 255
            rgba = np.dstack([image_rgb, alpha])
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(str(transparent_path), bgra)
            saved_files[f"live2d_{part}"] = str(transparent_path)
            
            # 可視化
            viz_path = output_dir / f"viz_{part}_{image_name}.png"
            viz_image = self.create_visualization(image_rgb, result)
            cv2.imwrite(str(viz_path), cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))
            saved_files[f"viz_{part}"] = str(viz_path)
        
        return saved_files

def parse_args():
    """コマンドライン引数解析"""
    parser = argparse.ArgumentParser(
        description="SAM2 Live2Dパーツセグメンテーション",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本実行
  python scripts/sam2_segmentation.py --input data/samples/anime_woman1
  
  # 特定パーツのみ
  python scripts/sam2_segmentation.py --input data/samples/anime_woman1 --parts face hair
  
  # 出力先指定
  python scripts/sam2_segmentation.py --input data/samples/anime_woman1 --output my_output
  
  # JPEGファイルも処理
  python scripts/sam2_segmentation.py --input data/samples/photos --pattern "*.jpg"
        """
    )
    
    parser.add_argument("--input", "-i", required=True, help="入力画像フォルダ")
    parser.add_argument("--output", "-o", default="data/output/live2d_segmentation", help="出力フォルダ")
    parser.add_argument("--parts", "-p", nargs="+", choices=["face", "hair", "body", "eyes"], 
                       default=["face", "hair", "body", "eyes"], help="対象パーツ")
    parser.add_argument("--pattern", default="*.png", help="画像ファイルパターン")
    parser.add_argument("--max-images", type=int, help="最大処理画像数")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細ログ")
    
    return parser.parse_args()

def main():
    """メイン実行"""
    args = parse_args()
    
    # ログ設定
    setup_logging(level="DEBUG" if args.verbose else "INFO", console_output=True, structured=False)
    logger = get_logger(__name__)
    
    logger.info("🎨 SAM2 Live2Dパーツセグメンテーション開始")
    logger.info(f"📁 入力: {args.input}")
    logger.info(f"📁 出力: {args.output}")
    logger.info(f"🎯 パーツ: {', '.join(args.parts)}")
    
    try:
        # パス設定
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = project_root / input_path
            
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 画像取得
        image_files = list(input_path.glob(args.pattern))
        if args.max_images:
            image_files = image_files[:args.max_images]
            
        if not image_files:
            logger.error(f"❌ 画像が見つかりません: {input_path}/{args.pattern}")
            return
            
        logger.info(f"📸 処理画像数: {len(image_files)}")
        
        # SAM2初期化
        logger.info("🤖 SAM2モデル読み込み中...")
        sam2_manager = SAM2ModelManager(model_name="sam2_hiera_large.pt")
        if not sam2_manager.load_model():
            logger.error("❌ SAM2モデル読み込み失敗")
            return
            
        segmenter = Live2DSegmenter(sam2_manager)
        
        # 統計
        total_time = 0
        all_results = []
        processed_count = 0
        
        # 画像処理
        for i, image_path in enumerate(image_files):
            logger.info(f"\n📸 処理中 ({i+1}/{len(image_files)}): {image_path.name}")
            
            try:
                # 画像読み込み
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning(f"⚠️ 読み込み失敗: {image_path}")
                    continue
                    
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # パーツ処理
                image_results = []
                image_start = time.time()
                
                for part in args.parts:
                    try:
                        result = segmenter.segment_part(image_rgb, part)
                        image_results.append(result)
                        all_results.append(result)
                    except Exception as e:
                        logger.error(f"❌ {part}処理失敗: {e}")
                
                if not image_results:
                    continue
                    
                # 結果保存
                saved_files = segmenter.save_results(image_rgb, image_results, output_path, image_path.stem)
                
                image_time = time.time() - image_start
                total_time += image_time
                processed_count += 1
                
                # 画像サマリー
                avg_score = np.mean([r.score for r in image_results])
                logger.info(f"✅ 完了: 平均スコア {avg_score:.3f}, 時間 {image_time:.1f}s")
                
            except Exception as e:
                logger.error(f"❌ {image_path.name}処理エラー: {e}")
        
        # 全体サマリー
        if all_results:
            logger.info(f"\n🎉 処理完了!")
            logger.info(f"📊 統計:")
            logger.info(f"  処理画像: {processed_count}/{len(image_files)}")
            logger.info(f"  総パーツ: {len(all_results)}")
            logger.info(f"  平均スコア: {np.mean([r.score for r in all_results]):.3f}")
            logger.info(f"  総時間: {total_time:.1f}秒")
            
            # パーツ別統計
            for part in args.parts:
                part_results = [r for r in all_results if r.part_name == part]
                if part_results:
                    avg_score = np.mean([r.score for r in part_results])
                    logger.info(f"  {part}: {len(part_results)}個, 平均スコア {avg_score:.3f}")
            
            logger.info(f"\n📁 結果: {output_path}")
            logger.info("💡 live2d_*.png ファイルをLive2Dで使用してください")
        else:
            logger.warning("⚠️ 処理結果がありません")
            
    except Exception as e:
        logger.error(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'sam2_manager' in locals():
            sam2_manager.unload_model()

if __name__ == "__main__":
    main()