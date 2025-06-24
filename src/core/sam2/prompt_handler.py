"""
SAM2プロンプトハンドラー

点、ボックス、テキストプロンプトを統一的に管理し、
Live2D用の高精度セグメンテーションを実現
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

from src.core.utils import get_logger
from src.core.utils.exceptions import ValidationError

logger = get_logger(__name__)

@dataclass
class PointPrompt:
    """点プロンプト"""
    x: int
    y: int
    label: int  # 1=foreground, 0=background
    confidence: float = 1.0
    description: str = ""

@dataclass 
class BoxPrompt:
    """ボックスプロンプト"""
    x1: int
    y1: int
    x2: int
    y2: int
    label: int = 1  # 1=foreground, 0=background
    confidence: float = 1.0
    description: str = ""
    
    @property
    def width(self) -> int:
        return abs(self.x2 - self.x1)
    
    @property
    def height(self) -> int:
        return abs(self.y2 - self.y1)
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

@dataclass
class MaskPrompt:
    """マスクプロンプト"""
    mask: np.ndarray
    confidence: float = 1.0
    description: str = ""

@dataclass
class PromptSession:
    """プロンプトセッション - 一つの画像に対する全プロンプト"""
    image_path: Optional[Path] = None
    points: List[PointPrompt] = field(default_factory=list)
    boxes: List[BoxPrompt] = field(default_factory=list)
    masks: List[MaskPrompt] = field(default_factory=list)
    session_id: Optional[str] = None
    created_at: Optional[str] = None

class SAM2PromptHandler:
    """SAM2プロンプト管理クラス"""
    
    def __init__(self):
        """初期化"""
        self.current_session: Optional[PromptSession] = None
        self.sessions: Dict[str, PromptSession] = {}
        
        logger.info("SAM2プロンプトハンドラー初期化完了")
    
    def start_new_session(self, image_path: Optional[Path] = None, session_id: Optional[str] = None) -> str:
        """
        新しいプロンプトセッションを開始
        
        Args:
            image_path: 対象画像パス
            session_id: セッションID（指定しない場合は自動生成）
            
        Returns:
            セッションID
        """
        
        if session_id is None:
            session_id = f"session_{len(self.sessions) + 1:04d}"
        
        from datetime import datetime
        
        self.current_session = PromptSession(
            image_path=image_path,
            session_id=session_id,
            created_at=datetime.now().isoformat()
        )
        
        self.sessions[session_id] = self.current_session
        
        logger.info(f"新しいプロンプトセッション開始: {session_id}")
        if image_path:
            logger.info(f"対象画像: {image_path}")
        
        return session_id
    
    def add_point_prompt(
        self, 
        x: int, 
        y: int, 
        label: int = 1, 
        confidence: float = 1.0,
        description: str = ""
    ) -> None:
        """
        点プロンプトを追加
        
        Args:
            x, y: 座標
            label: ラベル (1=foreground, 0=background)
            confidence: 信頼度
            description: 説明
        """
        
        if not self.current_session:
            raise ValidationError("アクティブなセッションがありません")
        
        point = PointPrompt(
            x=x, y=y, label=label, 
            confidence=confidence, description=description
        )
        
        self.current_session.points.append(point)
        
        logger.info(f"点プロンプト追加: ({x}, {y}), label={label}, desc='{description}'")
    
    def add_box_prompt(
        self,
        x1: int, y1: int, x2: int, y2: int,
        label: int = 1,
        confidence: float = 1.0,
        description: str = ""
    ) -> None:
        """
        ボックスプロンプトを追加
        
        Args:
            x1, y1, x2, y2: ボックス座標
            label: ラベル
            confidence: 信頼度
            description: 説明
        """
        
        if not self.current_session:
            raise ValidationError("アクティブなセッションがありません")
        
        box = BoxPrompt(
            x1=x1, y1=y1, x2=x2, y2=y2,
            label=label, confidence=confidence, description=description
        )
        
        self.current_session.boxes.append(box)
        
        logger.info(f"ボックスプロンプト追加: ({x1},{y1})-({x2},{y2}), desc='{description}'")
    
    def add_mask_prompt(
        self,
        mask: np.ndarray,
        confidence: float = 1.0,
        description: str = ""
    ) -> None:
        """
        マスクプロンプトを追加
        
        Args:
            mask: マスク配列
            confidence: 信頼度
            description: 説明
        """
        
        if not self.current_session:
            raise ValidationError("アクティブなセッションがありません")
        
        mask_prompt = MaskPrompt(
            mask=mask.copy(),
            confidence=confidence,
            description=description
        )
        
        self.current_session.masks.append(mask_prompt)
        
        logger.info(f"マスクプロンプト追加: shape={mask.shape}, desc='{description}'")
    
    def get_sam2_prompts(self) -> Dict[str, Optional[np.ndarray]]:
        """
        SAM2推論用のプロンプト形式に変換
        
        Returns:
            SAM2推論用プロンプト辞書
        """
        
        if not self.current_session:
            return {"point_coords": None, "point_labels": None, "box": None, "mask_input": None}
        
        result = {
            "point_coords": None,
            "point_labels": None,
            "box": None,
            "mask_input": None
        }
        
        # 点プロンプト処理
        if self.current_session.points:
            points = [(p.x, p.y) for p in self.current_session.points]
            labels = [p.label for p in self.current_session.points]
            
            result["point_coords"] = np.array(points, dtype=np.float32)
            result["point_labels"] = np.array(labels, dtype=np.int32)
        
        # ボックスプロンプト処理（最初のボックスのみ使用）
        if self.current_session.boxes:
            box = self.current_session.boxes[0]
            result["box"] = np.array([box.x1, box.y1, box.x2, box.y2], dtype=np.float32)
        
        # マスクプロンプト処理（最初のマスクのみ使用）
        if self.current_session.masks:
            result["mask_input"] = self.current_session.masks[0].mask
        
        return result
    
    def clear_prompts(self, prompt_type: Optional[str] = None) -> None:
        """
        プロンプトをクリア
        
        Args:
            prompt_type: クリアするタイプ ('points', 'boxes', 'masks', None=全て)
        """
        
        if not self.current_session:
            return
        
        if prompt_type is None or prompt_type == 'points':
            self.current_session.points.clear()
        if prompt_type is None or prompt_type == 'boxes':
            self.current_session.boxes.clear()
        if prompt_type is None or prompt_type == 'masks':
            self.current_session.masks.clear()
        
        logger.info(f"プロンプトクリア: {prompt_type or '全て'}")
    
    def get_prompt_summary(self) -> Dict[str, int]:
        """
        現在のプロンプト数を取得
        
        Returns:
            プロンプトタイプ別の数
        """
        
        if not self.current_session:
            return {"points": 0, "boxes": 0, "masks": 0}
        
        return {
            "points": len(self.current_session.points),
            "boxes": len(self.current_session.boxes),
            "masks": len(self.current_session.masks)
        }
    
    def visualize_prompts(self, image: np.ndarray, output_path: Optional[Path] = None) -> np.ndarray:
        """
        プロンプトを画像上に可視化
        
        Args:
            image: ベース画像
            output_path: 保存パス（オプション）
            
        Returns:
            プロンプト描画済み画像
        """
        
        if not self.current_session:
            return image.copy()
        
        vis_image = image.copy()
        
        # 点プロンプト描画
        for i, point in enumerate(self.current_session.points):
            color = (0, 255, 0) if point.label == 1 else (0, 0, 255)  # 緑=前景, 赤=背景
            cv2.circle(vis_image, (point.x, point.y), 8, color, -1)
            cv2.circle(vis_image, (point.x, point.y), 10, (255, 255, 255), 2)
            
            # 番号表示
            cv2.putText(vis_image, str(i+1), (point.x+15, point.y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # ボックスプロンプト描画
        for i, box in enumerate(self.current_session.boxes):
            color = (255, 255, 0)  # 黄色
            cv2.rectangle(vis_image, (box.x1, box.y1), (box.x2, box.y2), color, 3)
            
            # ボックス番号
            cv2.putText(vis_image, f"B{i+1}", (box.x1, box.y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # マスクプロンプト描画（輪郭のみ）
        for i, mask_prompt in enumerate(self.current_session.masks):
            mask = mask_prompt.mask
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, (255, 0, 255), 2)  # マゼンタ
        
        # 保存
        if output_path:
            cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            logger.info(f"プロンプト可視化保存: {output_path}")
        
        return vis_image

class Live2DPromptPresets:
    """Live2D用プリセットプロンプト"""
    
    @staticmethod
    def hair_segmentation(image_shape: Tuple[int, int]) -> List[PointPrompt]:
        """髪の毛セグメンテーション用プロンプト"""
        height, width = image_shape[:2]
        
        # 一般的な髪の位置にプロンプト
        return [
            PointPrompt(width//2, height//4, 1, description="髪の毛_中央"),
            PointPrompt(width//4, height//3, 1, description="髪の毛_左"),
            PointPrompt(width*3//4, height//3, 1, description="髪の毛_右"),
        ]
    
    @staticmethod
    def face_segmentation(image_shape: Tuple[int, int]) -> List[PointPrompt]:
        """顔セグメンテーション用プロンプト"""
        height, width = image_shape[:2]
        
        return [
            PointPrompt(width//2, height//2, 1, description="顔_中央"),
            PointPrompt(width*2//5, height*2//5, 1, description="顔_左目"),
            PointPrompt(width*3//5, height*2//5, 1, description="顔_右目"),
            PointPrompt(width//2, height*3//5, 1, description="顔_口"),
        ]
    
    @staticmethod
    def body_segmentation(image_shape: Tuple[int, int]) -> List[PointPrompt]:
        """体セグメンテーション用プロンプト"""
        height, width = image_shape[:2]
        
        return [
            PointPrompt(width//2, height*2//3, 1, description="体_中央"),
            PointPrompt(width//3, height*3//4, 1, description="体_左"),
            PointPrompt(width*2//3, height*3//4, 1, description="体_右"),
        ]
    
    @staticmethod
    def accessories_segmentation(image_shape: Tuple[int, int]) -> List[PointPrompt]:
        """アクセサリーセグメンテーション用プロンプト"""
        height, width = image_shape[:2]
        
        return [
            PointPrompt(width//2, height//6, 1, description="頭部アクセサリー"),
            PointPrompt(width//2, height//3, 1, description="首元アクセサリー"),
        ]