# backend/app/schemas/__init__.py
from .ocr import ocr_service
from .training import training_service

__all__ = ["ocr_service", "training_service"]