# backend/app/models/writer.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class Writer(Base):
    __tablename__ = "writers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    language = Column(String(50), nullable=True, server_default='english')  # Added language field
    model_path = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String(50), nullable=False, server_default='untrained')
    accuracy = Column(Float, nullable=True)
    pages_processed = Column(Integer, nullable=False, server_default='0')
    last_trained = Column(DateTime(timezone=True), nullable=True)
    is_trained = Column(Boolean, nullable=False, server_default='0')
    evaluation_metrics = Column(JSON, nullable=True)  # Store CER, WER, etc.

    # Relationships
    pages = relationship("Page", back_populates="writer")
    training_samples = relationship("TrainingSample", back_populates="writer", cascade="all, delete-orphan")
