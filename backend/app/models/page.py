# backend/app/models/page.py
import enum

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.types import JSON

from ..database import Base


class ProcessingStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class Page(Base):
    __tablename__ = "pages"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    writer_id = Column(Integer, ForeignKey("writers.id"), nullable=True)
    image_path = Column(String(255), nullable=False)
    extracted_text = Column(Text, nullable=True)
    formatted_text = Column(Text, nullable=True)
    page_number = Column(Integer, nullable=False)
    processing_status = Column(
        Enum(ProcessingStatus),
        nullable=False,
        default=ProcessingStatus.PENDING
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    lines = Column(JSON)

    document = relationship("Document", back_populates="pages")
    writer = relationship("Writer", back_populates="pages")
