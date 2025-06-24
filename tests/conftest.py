"""
pytest設定・共通フィクスチャ
"""

import pytest
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils import setup_logging


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """テスト用ログ設定"""
    setup_logging(level="DEBUG", console_output=True, structured=False)


@pytest.fixture
def project_root_path():
    """プロジェクトルートパス"""
    return project_root


@pytest.fixture
def sample_images_path():
    """サンプル画像パス"""
    return project_root / "data" / "samples" / "demo_images"