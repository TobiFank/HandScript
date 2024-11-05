# backend/app/ml/segmentation.py

from . import BaseDocumentSegmenter


def create_segmenter(segmenter_type: str = "layout", device: str = None) -> 'BaseDocumentSegmenter':
    """Factory function to create document segmenters"""
    if segmenter_type == "doctr":
        from .doctr_segmenter import DoctrSegmenter
        return DoctrSegmenter(device=device)
    elif segmenter_type == "layout":
        from .layout_segmenter import LayoutSegmenter
        return LayoutSegmenter(device=device)
    else:
        raise ValueError(f"Unknown segmenter type: {segmenter_type}")
