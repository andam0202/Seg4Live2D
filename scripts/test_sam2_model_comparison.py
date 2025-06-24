#!/usr/bin/env python3
"""
SAM2モデル比較テスト

全SAM2モデル（tiny/small/base+/large）での精度・速度比較を複数画像で実行
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

def test_single_model_on_image(
    model_name: str,
    image_path: Path,
    image_rgb: np.ndarray,
    test_prompts: Dict[str, List]
) -> Dict[str, Any]:
    """単一モデルでの単一画像テスト"""
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
            "part_results": {}
        }
        
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
            
            logger.info(f"    スコア: {best_score:.3f}, 時間: {inference_time:.2f}s")
        
        # メモリクリア
        sam2_manager.unload_model()
        
        return results
        
    except Exception as e:
        logger.error(f"モデルテスト失敗 {model_name}: {e}")
        return {"error": str(e)}

def create_comparison_visualization(
    image_rgb: np.ndarray,
    all_results: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """比較結果の可視化画像を作成"""
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
            
            # ベース画像をコピー
            model_image = image_rgb.copy()
            
            # 対応する結果を検索
            model_result = None
            for result in all_results:
                if result.get("model_name") == model_name and "error" not in result:
                    model_result = result
                    break
            
            if model_result:
                # スコア情報をオーバーレイ
                y_offset = 30
                cv2.putText(model_image, f"{short_name}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                y_offset += 40
                for part_name, part_result in model_result["part_results"].items():
                    score_text = f"{part_name}: {part_result['score']:.3f}"
                    cv2.putText(model_image, score_text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_offset += 25
                
                # 推論時間表示
                total_inference_time = sum(p["inference_time"] for p in model_result["part_results"].values())
                time_text = f"Time: {total_inference_time:.2f}s"
                cv2.putText(model_image, time_text, (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # エラーの場合
                cv2.putText(model_image, f"{short_name} - ERROR", (10, height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # グリッドに配置
            grid_image[y_start:y_end, x_start:x_end] = model_image
        
        # 保存
        grid_bgr = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), grid_bgr)
        logger.info(f"比較可視化保存: {output_path}")
        
    except Exception as e:
        logger.error(f"可視化作成失敗: {e}")

def create_detailed_comparison_report(
    all_results: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """詳細比較レポートをJSON形式で保存"""
    logger = get_logger(__name__)
    
    try:
        # 統計情報を計算
        summary = {
            "models": ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"],
            "parts": ["hair", "face", "body"],
            "detailed_results": all_results,
            "summary_stats": {}
        }
        
        # モデル別統計
        for model_name in summary["models"]:
            model_results = [r for r in all_results if r.get("model_name") == model_name and "error" not in r]
            
            if model_results:
                model_stats = {
                    "avg_scores": {},
                    "avg_inference_time": {},
                    "total_time": {}
                }
                
                for part_name in summary["parts"]:
                    scores = [r["part_results"][part_name]["score"] for r in model_results if part_name in r["part_results"]]
                    times = [r["part_results"][part_name]["inference_time"] for r in model_results if part_name in r["part_results"]]
                    
                    if scores:
                        model_stats["avg_scores"][part_name] = {
                            "mean": float(np.mean(scores)),
                            "std": float(np.std(scores)),
                            "min": float(np.min(scores)),
                            "max": float(np.max(scores))
                        }
                    
                    if times:
                        model_stats["avg_inference_time"][part_name] = {
                            "mean": float(np.mean(times)),
                            "std": float(np.std(times))
                        }
                
                # 全体時間統計
                total_times = []
                for result in model_results:
                    total_time = sum(p["inference_time"] for p in result["part_results"].values())
                    total_times.append(total_time)
                
                if total_times:
                    model_stats["total_time"] = {
                        "mean": float(np.mean(total_times)),
                        "std": float(np.std(total_times))
                    }
                
                summary["summary_stats"][model_name] = model_stats
        
        # JSON保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"詳細レポート保存: {output_path}")
        
    except Exception as e:
        logger.error(f"レポート作成失敗: {e}")

def main():
    """メイン実行"""
    
    # ログ設定
    setup_logging(level="INFO", console_output=True, structured=False)
    logger = get_logger(__name__)
    
    logger.info("=== SAM2モデル比較テスト ===")
    
    try:
        # テスト画像取得
        sample_images_path = project_root / "data" / "samples" / "demo_images"
        image_files = list(sample_images_path.glob("*.png"))
        
        if not image_files:
            logger.error("❌ テスト画像が見つかりません")
            return
        
        # 出力ディレクトリ準備
        output_dir = project_root / "data" / "output" / "sam2_model_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # テストするモデル
        models_to_test = [
            "sam2_hiera_tiny.pt",
            "sam2_hiera_small.pt", 
            "sam2_hiera_base_plus.pt",
            "sam2_hiera_large.pt"
        ]
        
        all_results = []
        
        # 各画像でテスト
        for img_idx, image_path in enumerate(image_files[:5]):  # 最大5枚でテスト
            logger.info(f"\\n📸 画像 {img_idx + 1}/{min(5, len(image_files))}: {image_path.name}")
            
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
                
                result = test_single_model_on_image(
                    model_name, image_path, image_rgb, test_prompts
                )
                
                if "error" not in result:
                    image_results.append(result)
                    all_results.append(result)
                    
                    # 各部位の結果表示
                    for part_name, part_result in result["part_results"].items():
                        logger.info(f"    {part_name}: スコア {part_result['score']:.3f}, "
                                  f"時間 {part_result['inference_time']:.2f}s")
                else:
                    logger.error(f"    ❌ {result['error']}")
            
            # 画像別比較可視化
            if image_results:
                comparison_image_path = output_dir / f"comparison_{image_path.stem}.png"
                create_comparison_visualization(image_rgb, image_results, comparison_image_path)
        
        # 全体結果レポート作成
        if all_results:
            logger.info(f"\\n📊 全体結果サマリー作成中...")
            
            # 詳細レポート
            report_path = output_dir / "detailed_comparison_report.json"
            create_detailed_comparison_report(all_results, report_path)
            
            # 簡易サマリー表示
            logger.info(f"\\n🏆 モデル別平均スコア:")
            
            models = ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
            parts = ["hair", "face", "body"]
            
            for model_name in models:
                model_results = [r for r in all_results if r.get("model_name") == model_name and "error" not in r]
                
                if model_results:
                    logger.info(f"  {model_name}:")
                    
                    for part_name in parts:
                        scores = [r["part_results"][part_name]["score"] for r in model_results if part_name in r["part_results"]]
                        if scores:
                            avg_score = np.mean(scores)
                            logger.info(f"    {part_name}: {avg_score:.3f} (±{np.std(scores):.3f})")
                    
                    # 平均推論時間
                    total_times = []
                    for result in model_results:
                        total_time = sum(p["inference_time"] for p in result["part_results"].values())
                        total_times.append(total_time)
                    
                    if total_times:
                        avg_time = np.mean(total_times)
                        logger.info(f"    平均推論時間: {avg_time:.2f}s")
        
        logger.info(f"\\n🎉 モデル比較テスト完了!")
        logger.info(f"結果: {output_dir}")
        logger.info(f"  - 比較画像: comparison_*.png")
        logger.info(f"  - 詳細レポート: detailed_comparison_report.json")
        
    except Exception as e:
        logger.error(f"テスト実行中にエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()