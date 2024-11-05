# app/ml/doctr_segmenter.py
from typing import List

import numpy as np
from PIL import Image
from doctr.models import detection_predictor

from .base_document_segmenter import BaseDocumentSegmenter
from .types import LineSegment, BoundingBox
from .utils import save_debug_image
from ..utils.logging import ml_logger


class DoctrSegmenter(BaseDocumentSegmenter):
    """Document segmentation using doctr's db_resnet50"""

    def __init__(self, device: str = None):
        super().__init__(device)
        self.model = None

        # Minimum height (in pixels) for a line to be considered valid
        # For handwriting, set lower as lines can be thinner
        self.min_line_height = 2

        # Threshold for considering two boxes as vertically overlapping
        # Lower threshold allows for more uneven handwriting
        self.vertical_overlap_threshold = 0.3

        # Maximum horizontal distance (in pixels) between words to be merged
        # Increased for handwriting as spacing is often irregular
        self.horizontal_merge_threshold = 70

        # Minimum confidence score for word detection
        # Lowered because handwriting detection typically has lower confidence
        self.min_word_confidence = 0.0

    def load(self):
        """Load doctr detection model"""
        if self._is_loaded:
            return

        try:
            ml_logger.info("Loading doctr detection model")
            self.model = detection_predictor(
                'db_resnet50',
                pretrained=True,
                assume_straight_pages=True
            )
            self._is_loaded = True
            ml_logger.info("Detection model loaded successfully")

        except Exception as e:
            ml_logger.error(f"Failed to load detection model: {str(e)}")
            raise

    def unload(self):
        """Unload model and clean memory"""
        if self._is_loaded:
            if self.model is not None:
                self.model.cpu()
                del self.model
                self.model = None

            self._cleanup_gpu_memory()
            self._is_loaded = False

    def _convert_detections_to_boxes(self, detections: List, image_width: int, image_height: int) -> List[BoundingBox]:
        """Convert raw detections to BoundingBox objects"""
        boxes = []

        for detection in detections:
            try:
                coords = np.array(detection[:-1])
                confidence = float(detection[-1])

                if confidence < self.min_word_confidence:
                    continue

                if coords.ndim == 1:
                    coords = coords.reshape(-1, 2)

                x_coords = coords[:, 0] * image_width
                y_coords = coords[:, 1] * image_height

                box = BoundingBox(
                    x1=int(min(x_coords)),
                    y1=int(min(y_coords)),
                    x2=int(max(x_coords)),
                    y2=int(max(y_coords)),
                    confidence=confidence
                )

                if box.y2 - box.y1 >= self.min_line_height:
                    boxes.append(box)

            except Exception as e:
                ml_logger.warning(f"Error processing detection: {str(e)}")
                continue

        return boxes

    def _merge_words_into_lines(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """Group word boxes into line boxes"""
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: b.y1)
        line_boxes = []
        current_line = [boxes[0]]

        for box in boxes[1:]:
            overlaps = any(prev_box.overlaps_vertically(box, self.vertical_overlap_threshold)
                           for prev_box in current_line)

            if overlaps:
                current_line.append(box)
            else:
                line_box = self._create_line_bbox(current_line)
                line_boxes.append(line_box)
                current_line = [box]

        if current_line:
            line_box = self._create_line_bbox(current_line)
            line_boxes.append(line_box)

        return line_boxes

    def _create_line_bbox(self, word_boxes: List[BoundingBox]) -> BoundingBox:
        """Create a single bounding box for a line from word boxes"""
        if not word_boxes:
            raise ValueError("No word boxes provided for line")

        line_box = word_boxes[0]
        for box in word_boxes[1:]:
            line_box = line_box.merge(box)

        padding = 5
        return BoundingBox(
            x1=max(0, line_box.x1 - padding),
            y1=max(0, line_box.y1 - padding),
            x2=line_box.x2 + padding,
            y2=line_box.y2 + padding,
            confidence=sum(b.confidence for b in word_boxes) / len(word_boxes)
        )

    def segment_page(self, image: Image.Image) -> List[LineSegment]:
        """Segment a page into line regions using doctr"""
        if not self._is_loaded:
            self.load()

        image_np = np.array(image)
        height, width = image_np.shape[:2]

        result = self.model([image_np])

        if len(result[0]['words']) == 0:
            line_boxes = [BoundingBox(x1=0, y1=0, x2=width, y2=height, confidence=1.0)]
        else:
            word_boxes = self._convert_detections_to_boxes(result[0]['words'], width, height)
            line_boxes = self._merge_words_into_lines(word_boxes)

        line_segments = []
        for line_bbox in line_boxes:
            line_image = image.crop((line_bbox.x1, line_bbox.y1, line_bbox.x2, line_bbox.y2))
            line_segments.append(LineSegment(image=line_image, bbox=line_bbox))

        ml_logger.info(f"Segmented {len(line_segments)} lines from page.")
        save_debug_image(image, line_segments, 'storage/debug_images/debug.png')
        return line_segments
