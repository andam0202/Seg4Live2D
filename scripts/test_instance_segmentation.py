#!/usr/bin/env python3
"""
インスタンス分割テスト

現在のYOLOv11でサンプル画像（ドット絵っぽい女の子）の
詳細なインスタンス分割を試行し、Live2D用途での実用性を評価
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils import setup_logging, get_logger

def analyze_masks_in_detail(result, image_path, output_dir):
    """マスクの詳細分析とLive2D適性評価"""
    logger = get_logger(__name__)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not result.success or len(result.masks) == 0:
        logger.warning("マスクが生成されませんでした")
        return {}
    
    # 元画像読み込み
    original_image = cv2.imread(str(image_path))
    height, width = original_image.shape[:2]
    
    logger.info(f"=== 詳細マスク分析: {image_path.name} ===")
    logger.info(f"元画像サイズ: {width}x{height}")
    logger.info(f"検出されたマスク数: {len(result.masks)}")
    
    analysis_results = {
        "image_name": image_path.name,
        "total_masks": len(result.masks),
        "masks_detail": [],
        "live2d_potential": {
            "face_area": 0,
            "body_area": 0,
            "hair_candidates": [],
            "accessory_candidates": []
        }
    }
    
    # 全マスクを統合した画像を作成
    combined_mask = np.zeros((height, width, 3), dtype=np.uint8)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
    ]
    
    for i, mask in enumerate(result.masks):
        color = colors[i % len(colors)]
        
        # マスク詳細情報
        mask_detail = {
            "id": i + 1,
            "class_name": mask.class_name,
            "confidence": mask.confidence,
            "area": mask.area,
            "area_ratio": mask.area / (width * height),
            "bbox": mask.bbox,
            "live2d_analysis": {}
        }
        
        # Live2D用途分析
        if mask.class_name == "person":
            # 人物として検出された場合の詳細分析
            bbox_width = mask.bbox[2] - mask.bbox[0]
            bbox_height = mask.bbox[3] - mask.bbox[1]
            aspect_ratio = bbox_height / bbox_width if bbox_width > 0 else 0
            
            mask_detail["live2d_analysis"] = {
                "type": "character_body",
                "aspect_ratio": aspect_ratio,
                "position": "center" if mask.bbox[0] < width * 0.3 and mask.bbox[2] > width * 0.7 else "partial",
                "size_category": "large" if mask.area > width * height * 0.1 else "small"
            }
            
            analysis_results["live2d_potential"]["body_area"] = mask.area
            
        elif mask.area > 1000:  # 十分大きい領域
            # その他の大きな領域の分析
            center_x = (mask.bbox[0] + mask.bbox[2]) / 2
            center_y = (mask.bbox[1] + mask.bbox[3]) / 2
            
            if center_y < height * 0.4:  # 上部領域
                analysis_results["live2d_potential"]["hair_candidates"].append({
                    "mask_id": i + 1,
                    "area": mask.area,
                    "position": "top"
                })
                mask_detail["live2d_analysis"]["potential_use"] = "hair_or_accessory"
            else:
                mask_detail["live2d_analysis"]["potential_use"] = "clothing_or_body_part"
        
        # マスクを画像に描画
        if hasattr(mask, 'mask_array') and mask.mask_array is not None:
            mask_colored = np.zeros_like(original_image)
            mask_colored[mask.mask_array > 0] = color
            combined_mask = cv2.addWeighted(combined_mask, 1.0, mask_colored, 0.7, 0)
        
        analysis_results["masks_detail"].append(mask_detail)
        
        logger.info(f"マスク{i+1}: {mask.class_name}")
        logger.info(f"  信頼度: {mask.confidence:.3f}")
        logger.info(f"  面積: {mask.area} px² ({mask.area/(width*height)*100:.1f}%)")
        logger.info(f"  境界: {mask.bbox}")
        if "live2d_analysis" in mask_detail and mask_detail["live2d_analysis"]:
            logger.info(f"  Live2D用途: {mask_detail['live2d_analysis']}")
    
    # 結果画像保存
    result_overlay = cv2.addWeighted(original_image, 0.6, combined_mask, 0.4, 0)
    
    output_files = {
        "original": output_dir / f"{image_path.stem}_original.png",
        "masks_only": output_dir / f"{image_path.stem}_masks.png", 
        "overlay": output_dir / f"{image_path.stem}_overlay.png"
    }
    
    cv2.imwrite(str(output_files["original"]), original_image)
    cv2.imwrite(str(output_files["masks_only"]), combined_mask)
    cv2.imwrite(str(output_files["overlay"]), result_overlay)
    
    logger.info(f"結果画像保存:")
    for name, path in output_files.items():
        logger.info(f"  {name}: {path}")
    
    return analysis_results

def evaluate_live2d_potential(analysis_results):
    """Live2D用途としての総合評価"""
    logger = get_logger(__name__)
    
    logger.info("\n=== Live2D用途適性評価 ===")
    
    total_masks = analysis_results["total_masks"]
    body_detected = analysis_results["live2d_potential"]["body_area"] > 0
    hair_candidates = len(analysis_results["live2d_potential"]["hair_candidates"])
    
    score = 0
    feedback = []
    
    # 評価基準
    if total_masks > 0:
        score += 2
        feedback.append(f"✅ セグメンテーション成功 ({total_masks}個の領域)")
    else:
        feedback.append("❌ セグメンテーション失敗")
        return score, feedback
    
    if body_detected:
        score += 3
        feedback.append("✅ キャラクター本体検出")
    else:
        feedback.append("⚠️ キャラクター本体未検出")
    
    if hair_candidates > 0:
        score += 2
        feedback.append(f"✅ 髪/アクセサリー候補検出 ({hair_candidates}個)")
    
    if total_masks >= 3:
        score += 2
        feedback.append("✅ 複数パーツ分割可能")
    elif total_masks >= 2:
        score += 1
        feedback.append("⚠️ 最小限の分割")
    
    # 詳細分析
    for mask in analysis_results["masks_detail"]:
        if mask["confidence"] > 0.5:
            score += 1
            break
    else:
        feedback.append("⚠️ 高信頼度検出なし")
    
    # 総合評価
    if score >= 7:
        overall = "🎉 Live2D用途に適用可能"
    elif score >= 5:
        overall = "👍 部分的に適用可能、要調整"
    elif score >= 3:
        overall = "🔧 基本機能のみ、大幅改善必要"
    else:
        overall = "❌ 現時点では実用困難"
    
    logger.info(f"スコア: {score}/10")
    logger.info(f"総合評価: {overall}")
    for fb in feedback:
        logger.info(f"  {fb}")
    
    return score, feedback, overall

def main():
    """メイン実行"""
    
    # ログ設定
    setup_logging(level="INFO", console_output=True, structured=False)
    logger = get_logger(__name__)
    
    logger.info("=== インスタンス分割詳細テスト ===")
    
    try:
        # セグメンテーションエンジン初期化
        from src.core.segmentation import get_segmentation_engine
        
        engine = get_segmentation_engine()
        engine.initialize()
        
        # サンプル画像取得
        sample_images_path = project_root / "data" / "samples" / "demo_images"
        image_files = list(sample_images_path.glob("*.png"))
        
        if not image_files:
            logger.error("サンプル画像が見つかりません")
            return
        
        logger.info(f"分析対象: {len(image_files)}枚の画像")
        
        output_base_dir = project_root / "data" / "output" / "instance_analysis"
        
        all_results = []
        
        # 各画像で詳細分析
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"画像 {i}/{len(image_files)}: {image_file.name}")
            logger.info(f"{'='*50}")
            
            # セグメンテーション実行
            result = engine.process_image(
                image=image_file,
                save_masks=True,
                output_dir=output_base_dir / f"image_{i:02d}"
            )
            
            if result.success:
                # 詳細分析
                analysis = analyze_masks_in_detail(
                    result, image_file, 
                    output_base_dir / f"image_{i:02d}" / "analysis"
                )
                
                # Live2D適性評価
                score, feedback, overall = evaluate_live2d_potential(analysis)
                analysis["live2d_score"] = score
                analysis["live2d_feedback"] = feedback
                analysis["live2d_overall"] = overall
                
                all_results.append(analysis)
            else:
                logger.error(f"セグメンテーション失敗: {result.error_message}")
        
        # 全体総括
        logger.info(f"\n{'='*60}")
        logger.info("=== 全画像分析総括 ===")
        logger.info(f"{'='*60}")
        
        if all_results:
            avg_score = sum(r["live2d_score"] for r in all_results) / len(all_results)
            total_masks = sum(r["total_masks"] for r in all_results)
            
            logger.info(f"平均Live2Dスコア: {avg_score:.1f}/10")
            logger.info(f"総検出マスク数: {total_masks}")
            logger.info(f"画像あたり平均マスク数: {total_masks/len(all_results):.1f}")
            
            # 最も良い結果を表示
            best_result = max(all_results, key=lambda x: x["live2d_score"])
            logger.info(f"\n最高スコア: {best_result['image_name']} ({best_result['live2d_score']}/10)")
            logger.info(f"評価: {best_result['live2d_overall']}")
            
            # 推奨事項
            logger.info(f"\n💡 推奨事項:")
            if avg_score >= 6:
                logger.info("- 現在の手法でLive2D素材分割を試行可能")
                logger.info("- 後処理での境界調整・レイヤー分離を実装")
            elif avg_score >= 4:
                logger.info("- より大きなYOLOモデル (yolo11m-seg, yolo11l-seg) を試行")
                logger.info("- 画像の前処理（コントラスト調整等）を検討")
            else:
                logger.info("- SAM (Segment Anything Model) の併用検討")
                logger.info("- 手動アノテーション + 半自動化アプローチ")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # クリーンアップ
        if 'engine' in locals() and hasattr(engine, 'cleanup'):
            engine.cleanup()

if __name__ == "__main__":
    main()