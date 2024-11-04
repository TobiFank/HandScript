# backend/app/schemas/training_sample.py
from datetime import datetime
from typing import List, Dict, Optional

from pydantic import BaseModel

from .base import BaseSchema, TimestampMixin


class LineInfo(BaseModel):
    bbox: tuple[int, int, int, int]
    text: str
    confidence: float


class TrainingSampleBase(BaseSchema):
    text: str
    needs_review: bool = True


class TrainingSampleCreate(TrainingSampleBase):
    pass


class TrainingSampleUpdate(TrainingSampleBase):
    pass


class TrainingSample(TrainingSampleBase, TimestampMixin):
    id: int
    writer_id: int
    image_path: str

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class TrainingSampleWithLines(TrainingSample):
    line_count: Optional[int] = 0
    lines: Optional[List[Dict]] = []  # List of line segments with their info
    needs_review: bool


class LineSegmentBase(BaseSchema):
    text: str
    order: int
    bbox: List[float]
    confidence: float


class LineSegment(LineSegmentBase):
    id: int
    image_path: str
    training_sample_id: int


class LineSegmentCreate(LineSegmentBase):
    image_path: str


class TrainingSample(TrainingSampleBase, TimestampMixin):
    id: int
    writer_id: int
    image_path: str
    line_segments: List[LineSegment] = []
    needs_review: bool

    class Config:
        from_attributes = True


class ExportResponse(BaseSchema):
    success: bool
    sample_count: int
    export_path: str
