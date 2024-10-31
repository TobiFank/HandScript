# backend/app/schemas/page.py
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel
from pydantic import Field

from .base import BaseSchema, TimestampMixin
from ..models.page import ProcessingStatus


class WriterBase(BaseSchema):
    name: str


class WriterCreate(WriterBase):
    language: Optional[str] = None


class WriterUpdate(WriterBase):
    pass


class Writer(WriterBase, TimestampMixin):
    id: int
    model_path: Optional[str] = None
    status: str  # 'untrained', 'training', 'ready', 'error'
    accuracy: Optional[float] = None
    pages_processed: int = 0
    last_trained: Optional[datetime] = None
    is_trained: bool = False

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class PageBase(BaseSchema):
    page_number: int
    updated_at: datetime = Field(default_factory=datetime.now)  # Add default


class PageCreate(PageBase):
    document_id: int
    writer_id: Optional[int] = None


class LineSegment(BaseModel):
    image_path: str
    text: str
    bbox: List[float]
    confidence: Optional[float] = None


class LineSegmentUpdate(BaseModel):
    text: str
    image_path: Optional[str]
    bbox: Optional[List[float]]

class PageUpdate(BaseModel):
    extracted_text: Optional[str] = None
    formatted_text: Optional[str] = None
    writer_id: Optional[int] = None
    lines: Optional[List[LineSegmentUpdate]] = None


class Page(PageBase, TimestampMixin):
    id: int
    document_id: int
    writer_id: Optional[int]
    image_path: str
    extracted_text: Optional[str] = None
    formatted_text: Optional[str] = None
    processing_status: ProcessingStatus
    updated_at: Optional[datetime] = None  # Make it optional
    writer: Optional[Writer] = None
    lines: Optional[List[LineSegment]] = None
