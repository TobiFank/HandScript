# backend/app/schemas/document.py
from typing import Optional, List
from .base import BaseSchema, TimestampMixin
from .page import Page

class DocumentBase(BaseSchema):
    name: str
    description: Optional[str] = None

class DocumentCreate(DocumentBase):
    project_id: int

class DocumentUpdate(DocumentBase):
    pass

class Document(DocumentBase, TimestampMixin):
    id: int
    project_id: int

class DocumentDetail(Document):
    pages: List[Page] = []
    page_count: int = 0