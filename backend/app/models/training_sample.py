# backend/app/models/training_sample.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Float, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class LineSegment(Base):
    __tablename__ = "line_segments"

    id = Column(Integer, primary_key=True, index=True)
    training_sample_id = Column(Integer, ForeignKey("training_samples.id", ondelete="CASCADE"))
    image_path = Column(String(255), nullable=False)
    text = Column(Text, nullable=False)
    order = Column(Integer, nullable=False)
    bbox = Column(JSON, nullable=False)  # Stores [x1, y1, x2, y2]
    confidence = Column(Float, nullable=False)

    training_sample = relationship("TrainingSample", back_populates="line_segments")


class TrainingSample(Base):
    __tablename__ = "training_samples"

    id = Column(Integer, primary_key=True, index=True)
    writer_id = Column(Integer, ForeignKey("writers.id", ondelete="CASCADE"), nullable=False)
    image_path = Column(String(255), nullable=False)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    needs_review = Column(Boolean, nullable=False, default=True)

    writer = relationship("Writer", back_populates="training_samples")
    line_segments = relationship("LineSegment", back_populates="training_sample",
                                 cascade="all, delete-orphan", order_by="LineSegment.order")
