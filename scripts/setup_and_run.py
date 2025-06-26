#!/usr/bin/env python3
"""
Seg4Live2D セットアップ・実行統合スクリプト

初回セットアップからセグメンテーション実行まで一括対応
- モデル自動ダウンロード
- 環境確認
- セグメンテーション実行
"""

import sys
import subprocess
import argparse
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils import setup_logging, get_logger

def check_models():
    """モデルファイル存在確認"""
    sam2_model_path = project_root / "models" / "sam2" / "checkpoints" / "sam2_hiera_large.pt"
    return sam2_model_path.exists()

def download_models():
    """モデルダウンロード実行"""
    logger = get_logger(__name__)
    logger.info("📥 SAM2モデルをダウンロード中...")
    
    download_script = project_root / "scripts" / "download_sam2_models.py"
    if not download_script.exists():
        logger.error(f"❌ ダウンロードスクリプトが見つかりません: {download_script}")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(download_script)], 
                              capture_output=True, text=True, cwd=project_root)
        if result.returncode == 0:
            logger.info("✅ モデルダウンロード完了")
            return True
        else:
            logger.error(f"❌ ダウンロード失敗: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ ダウンロード実行エラー: {e}")
        return False

def run_segmentation(args):
    """セグメンテーション実行"""
    logger = get_logger(__name__)
    logger.info("🎨 セグメンテーション開始...")
    
    segmentation_script = project_root / "scripts" / "sam2_segmentation.py"
    if not segmentation_script.exists():
        logger.error(f"❌ セグメンテーションスクリプトが見つかりません: {segmentation_script}")
        return False
    
    # コマンドライン引数構築
    cmd = [sys.executable, str(segmentation_script)]
    
    if args.input:
        cmd.extend(["--input", args.input])
    if args.output:
        cmd.extend(["--output", args.output])
    if args.parts:
        cmd.extend(["--parts"] + args.parts)
    if args.pattern:
        cmd.extend(["--pattern", args.pattern])
    if args.max_images:
        cmd.extend(["--max-images", str(args.max_images)])
    if args.verbose:
        cmd.append("--verbose")
    
    try:
        # リアルタイム出力で実行
        process = subprocess.Popen(cmd, cwd=project_root, stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        
        for line in process.stdout:
            print(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            logger.info("✅ セグメンテーション完了")
            return True
        else:
            logger.error(f"❌ セグメンテーション失敗 (終了コード: {process.returncode})")
            return False
            
    except Exception as e:
        logger.error(f"❌ セグメンテーション実行エラー: {e}")
        return False

def parse_args():
    """コマンドライン引数解析"""
    parser = argparse.ArgumentParser(
        description="Seg4Live2D セットアップ・実行統合スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 初回セットアップ + セグメンテーション実行
  python scripts/setup_and_run.py --input data/samples/anime_woman1
  
  # モデルチェックのみ
  python scripts/setup_and_run.py --check-only
  
  # 強制再ダウンロード
  python scripts/setup_and_run.py --force-download --input data/samples/anime_woman1
  
  # 特定パーツのみ処理
  python scripts/setup_and_run.py --input data/samples/anime_woman1 --parts face hair
        """
    )
    
    # セットアップオプション
    parser.add_argument("--check-only", action="store_true", help="モデル存在確認のみ")
    parser.add_argument("--force-download", action="store_true", help="強制再ダウンロード")
    parser.add_argument("--skip-download", action="store_true", help="ダウンロードをスキップ")
    
    # セグメンテーションオプション（sam2_segmentation.pyと同じ）
    parser.add_argument("--input", "-i", help="入力画像フォルダ")
    parser.add_argument("--output", "-o", help="出力フォルダ")
    parser.add_argument("--parts", "-p", nargs="+", choices=["face", "hair", "body", "eyes"], help="対象パーツ")
    parser.add_argument("--pattern", help="画像ファイルパターン")
    parser.add_argument("--max-images", type=int, help="最大処理画像数")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細ログ")
    
    return parser.parse_args()

def main():
    """メイン実行"""
    args = parse_args()
    
    # ログ設定
    setup_logging(level="DEBUG" if args.verbose else "INFO", console_output=True, structured=False)
    logger = get_logger(__name__)
    
    logger.info("🚀 Seg4Live2D セットアップ・実行開始")
    
    # モデル存在確認
    models_exist = check_models()
    logger.info(f"📦 モデル状況: {'✅ 存在' if models_exist else '❌ 不足'}")
    
    if args.check_only:
        if models_exist:
            logger.info("✅ モデルファイルは正常に配置されています")
            return
        else:
            logger.warning("⚠️ モデルファイルが不足しています")
            logger.info("💡 --force-download オプションでダウンロードしてください")
            return
    
    # モデルダウンロード
    if args.force_download or (not models_exist and not args.skip_download):
        if not download_models():
            logger.error("❌ モデルダウンロードに失敗しました")
            return
    elif not models_exist:
        logger.error("❌ モデルファイルが不足しており、ダウンロードもスキップされました")
        logger.info("💡 --force-download オプションを使用してください")
        return
    
    # セグメンテーション実行
    if args.input:
        if not run_segmentation(args):
            logger.error("❌ セグメンテーション実行に失敗しました")
            return
    else:
        logger.info("✅ セットアップ完了")
        logger.info("💡 --input オプションでセグメンテーションを実行してください")
        logger.info("   例: python scripts/setup_and_run.py --input data/samples/anime_woman1")

if __name__ == "__main__":
    main()