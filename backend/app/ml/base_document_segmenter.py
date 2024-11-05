# app/ml/base_document_segmenter.py
import abc
import gc
from typing import List

import torch
from PIL import Image

from .types import LineSegment


class BaseDocumentSegmenter(abc.ABC):
    """Abstract base class for document segmentation"""

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._is_loaded = False

    @abc.abstractmethod
    def load(self):
        """Load segmentation model"""
        pass

    @abc.abstractmethod
    def unload(self):
        """Unload segmentation model"""
        pass

    @abc.abstractmethod
    def segment_page(self, image: Image.Image) -> List[LineSegment]:
        """Segment a page into line regions"""
        pass

    def _cleanup_gpu_memory(self):
        """Common GPU memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
