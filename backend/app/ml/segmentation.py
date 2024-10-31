# backend/app/ml/segmentation.py
import gc
from typing import List, Tuple, Optional
import numpy as np
import torch
from PIL import Image
from dataclasses import dataclass
from doctr.models import detection_predictor
from ..utils.logging import ml_logger

@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates and confidence score"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    def area(self) -> int:
        """Calculate area of bounding box"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def merge(self, other: 'BoundingBox') -> 'BoundingBox':
        """Merge two bounding boxes"""
        return BoundingBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2),
            confidence=(self.confidence + other.confidence) / 2
        )

    def overlaps_vertically(self, other: 'BoundingBox', threshold: float = 0.5) -> bool:
        """Check if two boxes overlap vertically"""
        intersection = min(self.y2, other.y2) - max(self.y1, other.y1)
        if intersection <= 0:
            return False

        min_height = min(self.y2 - self.y1, other.y2 - other.y1)
        return intersection / min_height >= threshold

@dataclass
class LineSegment:
    """Represents a single line segment from a document"""
    image: Image.Image  # Image within the line bounding box
    bbox: BoundingBox
    text: Optional[str] = None


class DocumentSegmenter:
    """Handles document layout analysis and line segmentation using doctr"""
    def __init__(self, device: str = None):
        self.device = device
        self.model = None
        self._is_loaded = False

        # Configuration parameters
        self.min_line_height = 10  # Minimum height for a valid line
        self.vertical_overlap_threshold = 0.5  # Threshold for merging boxes vertically
        self.horizontal_merge_threshold = 50  # Maximum horizontal gap between words to merge
        self.min_word_confidence = 0.0  # Minimum confidence for word detection

    def load(self):
        """Load segmentation model"""
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
        """Enhanced unload method with memory cleanup"""
        if self._is_loaded:
            if self.model is not None:
                self.model.cpu()
                del self.model
                self.model = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

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

                # Convert relative coordinates to absolute pixels
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

                # Filter out boxes that are too small
                if box.y2 - box.y1 >= self.min_line_height:
                    boxes.append(box)

            except Exception as e:
                ml_logger.warning(f"Error processing detection: {str(e)}")
                continue

        return boxes

    def _merge_words_into_lines(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """Group word boxes into line boxes without storing individual words."""
        if not boxes:
            return []

        # Sort boxes by y-coordinate (top to bottom) to process lines in order
        boxes = sorted(boxes, key=lambda b: b.y1)

        line_boxes = []
        current_line = [boxes[0]]

        for box in boxes[1:]:
            overlaps = any(prev_box.overlaps_vertically(box, self.vertical_overlap_threshold) for prev_box in current_line)

            if overlaps:
                current_line.append(box)
            else:
                # Create line bounding box from current line
                line_box = self._create_line_bbox(current_line)
                line_boxes.append(line_box)
                current_line = [box]

        # Process the last line
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

        # Add padding
        padding = 5
        return BoundingBox(
            x1=max(0, line_box.x1 - padding),
            y1=max(0, line_box.y1 - padding),
            x2=line_box.x2 + padding,
            y2=line_box.y2 + padding,
            confidence=sum(b.confidence for b in word_boxes) / len(word_boxes)
        )

    def segment_page(self, image: Image.Image) -> List[LineSegment]:
        """Segment a page into line regions, directly outputting line boxes."""
        if not self._is_loaded:
            self.load()

        # Convert PIL Image to numpy array
        image_np = np.array(image)
        height, width = image_np.shape[:2]

        result = self.model([image_np])

        if len(result[0]['words']) == 0:
            line_boxes = [BoundingBox(x1=0, y1=0, x2=width, y2=height, confidence=1.0)]
        else:
            # Convert detections to BoundingBox objects (using entire lines)
            word_boxes = self._convert_detections_to_boxes(result[0]['words'], width, height)
            line_boxes = self._merge_words_into_lines(word_boxes)

        line_segments = []
        for line_bbox in line_boxes:
            line_image = image.crop((line_bbox.x1, line_bbox.y1, line_bbox.x2, line_bbox.y2))
            line_segments.append(LineSegment(image=line_image, bbox=line_bbox))

        ml_logger.info(f"Segmented {len(line_segments)} lines from page.")

        # Save debug image
        save_debug_image(image, line_segments, 'storage/debug_images/debug.png')
        return line_segments


def save_debug_image(image: Image.Image, line_segments: List[LineSegment], output_path: str):
    """Save a debug image showing detected lines and words"""
    import cv2
    import numpy as np

    # Convert PIL to CV2
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Draw line boxes in green
    for i, segment in enumerate(line_segments):
        cv2.rectangle(
            img,
            (segment.bbox.x1, segment.bbox.y1),
            (segment.bbox.x2, segment.bbox.y2),
            (0, 255, 0),
            2
        )

        # Add line number
        cv2.putText(
            img,
            f"Line {i+1}",
            (segment.bbox.x1, segment.bbox.y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    cv2.imwrite(output_path, img)