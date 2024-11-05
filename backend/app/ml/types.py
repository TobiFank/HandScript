# app/ml/types.py
from dataclasses import dataclass
from typing import Optional

from PIL import Image


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates and confidence score"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    def area(self) -> int:
        """Calculate area of bounding box"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def merge(self, other: 'BoundingBox') -> 'BoundingBox':
        """Merge two bounding boxes"""
        return BoundingBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2),
            confidence=(self.confidence + other.confidence) / 2
        )

    def overlaps_vertically(self, other: 'BoundingBox', threshold: float = 0.5) -> bool:
        """Check if two boxes overlap vertically"""
        intersection = min(self.y2, other.y2) - max(self.y1, other.y1)
        if intersection <= 0:
            return False

        min_height = min(self.y2 - self.y1, other.y2 - other.y1)
        return intersection / min_height >= threshold


@dataclass
class LineSegment:
    """Represents a single line segment from a document"""
    image: Image.Image  # Image within the line bounding box
    bbox: BoundingBox
    text: Optional[str] = None
