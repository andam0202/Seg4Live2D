"""
ファイル操作ユーティリティ

安全で効率的なファイル操作機能を提供します。
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Optional, Union, Tuple, Generator, Any, Dict
import mimetypes
from datetime import datetime

from .exceptions import FileOperationError, ValidationError
from .logger import get_logger

logger = get_logger(__name__)


class FileHandler:
    """ファイル操作ハンドラークラス"""
    
    # 許可される画像拡張子
    ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}
    
    # 最大ファイルサイズ（デフォルト50MB）
    DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024
    
    def __init__(self, max_file_size: Optional[int] = None) -> None:
        self.max_file_size = max_file_size or self.DEFAULT_MAX_FILE_SIZE
    
    def validate_image_file(
        self, 
        file_path: Union[str, Path],
        check_size: bool = True,
        check_extension: bool = True,
        check_existence: bool = True
    ) -> bool:
        """画像ファイルのバリデーション"""
        
        file_path = Path(file_path)
        
        try:
            # 存在チェック
            if check_existence and not file_path.exists():
                raise ValidationError(
                    f"ファイルが存在しません: {file_path}",
                    field="file_path",
                    value=str(file_path)
                )
            
            # 拡張子チェック
            if check_extension:
                extension = file_path.suffix.lower()
                if extension not in self.ALLOWED_IMAGE_EXTENSIONS:
                    raise ValidationError(
                        f"サポートされていない拡張子です: {extension}",
                        field="extension",
                        value=extension,
                        details={"allowed_extensions": list(self.ALLOWED_IMAGE_EXTENSIONS)}
                    )
            
            # ファイルサイズチェック
            if check_size and file_path.exists():
                file_size = file_path.stat().st_size
                if file_size > self.max_file_size:
                    raise ValidationError(
                        f"ファイルサイズが制限を超えています: {file_size} bytes > {self.max_file_size} bytes",
                        field="file_size",
                        value=file_size,
                        details={"max_size": self.max_file_size}
                    )
            
            logger.debug(f"ファイルバリデーション成功: {file_path}")
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise FileOperationError(
                f"ファイルバリデーション中にエラーが発生しました: {e}",
                file_path=str(file_path),
                operation="validation"
            ) from e
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """ファイル情報の取得"""
        
        file_path = Path(file_path)
        
        try:
            if not file_path.exists():
                raise FileOperationError(
                    f"ファイルが存在しません: {file_path}",
                    file_path=str(file_path),
                    operation="get_info"
                )
            
            stat = file_path.stat()
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            info = {
                "path": str(file_path.absolute()),
                "name": file_path.name,
                "stem": file_path.stem,
                "suffix": file_path.suffix,
                "size": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "mime_type": mime_type,
                "is_image": file_path.suffix.lower() in self.ALLOWED_IMAGE_EXTENSIONS,
            }
            
            # 画像ファイルの場合、追加情報を取得
            if info["is_image"]:
                try:
                    from PIL import Image
                    with Image.open(file_path) as img:
                        info.update({
                            "width": img.width,
                            "height": img.height,
                            "mode": img.mode,
                            "format": img.format,
                        })
                except Exception as e:
                    logger.warning(f"画像情報の取得に失敗: {e}")
            
            return info
            
        except Exception as e:
            raise FileOperationError(
                f"ファイル情報取得中にエラーが発生しました: {e}",
                file_path=str(file_path),
                operation="get_info"
            ) from e
    
    def calculate_file_hash(
        self, 
        file_path: Union[str, Path], 
        algorithm: str = "md5"
    ) -> str:
        """ファイルハッシュの計算"""
        
        file_path = Path(file_path)
        
        try:
            hash_obj = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            raise FileOperationError(
                f"ファイルハッシュ計算中にエラーが発生しました: {e}",
                file_path=str(file_path),
                operation="calculate_hash"
            ) from e
    
    def ensure_directory(self, directory: Union[str, Path]) -> Path:
        """ディレクトリの確実な作成"""
        
        directory = Path(directory)
        
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"ディレクトリを作成しました: {directory}")
            return directory
            
        except Exception as e:
            raise FileOperationError(
                f"ディレクトリ作成中にエラーが発生しました: {e}",
                file_path=str(directory),
                operation="create_directory"
            ) from e
    
    def safe_copy(
        self, 
        src: Union[str, Path], 
        dst: Union[str, Path],
        overwrite: bool = False
    ) -> Path:
        """安全なファイルコピー"""
        
        src = Path(src)
        dst = Path(dst)
        
        try:
            # 元ファイルの存在確認
            if not src.exists():
                raise FileOperationError(
                    f"コピー元ファイルが存在しません: {src}",
                    file_path=str(src),
                    operation="copy"
                )
            
            # 宛先ディレクトリの作成
            self.ensure_directory(dst.parent)
            
            # 上書き確認
            if dst.exists() and not overwrite:
                raise FileOperationError(
                    f"宛先ファイルが既に存在します: {dst}",
                    file_path=str(dst),
                    operation="copy"
                )
            
            # コピー実行
            shutil.copy2(src, dst)
            logger.info(f"ファイルをコピーしました: {src} -> {dst}")
            
            return dst
            
        except Exception as e:
            raise FileOperationError(
                f"ファイルコピー中にエラーが発生しました: {e}",
                file_path=str(src),
                operation="copy"
            ) from e
    
    def safe_delete(self, file_path: Union[str, Path]) -> bool:
        """安全なファイル削除"""
        
        file_path = Path(file_path)
        
        try:
            if file_path.exists():
                if file_path.is_file():
                    file_path.unlink()
                    logger.info(f"ファイルを削除しました: {file_path}")
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    logger.info(f"ディレクトリを削除しました: {file_path}")
                return True
            else:
                logger.warning(f"削除対象が存在しません: {file_path}")
                return False
                
        except Exception as e:
            raise FileOperationError(
                f"ファイル削除中にエラーが発生しました: {e}",
                file_path=str(file_path),
                operation="delete"
            ) from e
    
    def find_images(
        self, 
        directory: Union[str, Path], 
        recursive: bool = True
    ) -> List[Path]:
        """ディレクトリ内の画像ファイルを検索"""
        
        directory = Path(directory)
        
        try:
            if not directory.exists():
                raise FileOperationError(
                    f"ディレクトリが存在しません: {directory}",
                    file_path=str(directory),
                    operation="find_images"
                )
            
            pattern = "**/*" if recursive else "*"
            images = []
            
            for file_path in directory.glob(pattern):
                if (file_path.is_file() and 
                    file_path.suffix.lower() in self.ALLOWED_IMAGE_EXTENSIONS):
                    images.append(file_path)
            
            logger.debug(f"画像ファイル検索完了: {len(images)}件 in {directory}")
            return sorted(images)
            
        except Exception as e:
            raise FileOperationError(
                f"画像ファイル検索中にエラーが発生しました: {e}",
                file_path=str(directory),
                operation="find_images"
            ) from e
    
    def cleanup_temp_files(
        self, 
        temp_dir: Union[str, Path], 
        max_age_hours: float = 24.0
    ) -> int:
        """一時ファイルのクリーンアップ"""
        
        temp_dir = Path(temp_dir)
        
        try:
            if not temp_dir.exists():
                return 0
            
            current_time = datetime.now().timestamp()
            max_age_seconds = max_age_hours * 3600
            deleted_count = 0
            
            for file_path in temp_dir.rglob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        try:
                            file_path.unlink()
                            deleted_count += 1
                        except Exception as e:
                            logger.warning(f"一時ファイル削除失敗: {file_path} - {e}")
            
            logger.info(f"一時ファイルクリーンアップ完了: {deleted_count}件削除")
            return deleted_count
            
        except Exception as e:
            raise FileOperationError(
                f"一時ファイルクリーンアップ中にエラーが発生しました: {e}",
                file_path=str(temp_dir),
                operation="cleanup"
            ) from e


# グローバルインスタンス
file_handler = FileHandler()