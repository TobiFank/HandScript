# backend/app/ml/segmentation.py
import abc
import gc
import os
from dataclasses import dataclass
from typing import List, Tuple
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from doctr.models import detection_predictor
from transformers import LayoutLMv3FeatureExtractor

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


class DoctrSegmenter(BaseDocumentSegmenter):
    """Document segmentation using doctr's db_resnet50"""

    def __init__(self, device: str = None):
        super().__init__(device)
        self.model = None

        # Configuration parameters
        self.min_line_height = 5
        self.vertical_overlap_threshold = 0.5
        self.horizontal_merge_threshold = 50
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


class LayoutSegmenter(BaseDocumentSegmenter):
    """Document segmentation using hybrid approach for line detection"""

    def __init__(self, device: str = None):
        super().__init__(device)
        self.feature_extractor = None
        self.min_line_height = 5
        self.min_segment_width = 20  # Minimum width for a meaningful text segment
        self.vertical_merge_threshold = 5  # Maximum vertical gap to merge
        self.horizontal_density_threshold = 0.1

    def load(self):
        """Load required models and initialize resources"""
        if self._is_loaded:
            return

        try:
            self.feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained(
                "microsoft/layoutlmv3-large",
                apply_ocr=False
            )
            self._is_loaded = True
            ml_logger.info("LayoutSegmenter loaded successfully")

        except Exception as e:
            ml_logger.error(f"Failed to load LayoutSegmenter: {str(e)}")
            raise

    def unload(self):
        """Unload models and clean up resources"""
        if self._is_loaded:
            try:
                if self.feature_extractor is not None:
                    del self.feature_extractor
                    self.feature_extractor = None

                self._cleanup_gpu_memory()
                self._is_loaded = False
                ml_logger.info("LayoutSegmenter unloaded successfully")

            except Exception as e:
                ml_logger.error(f"Error during LayoutSegmenter unloading: {str(e)}")
                raise

    def _analyze_horizontal_density(self, binary_image: np.ndarray, y1: int, y2: int) -> List[Tuple[int, int]]:
        """Analyze horizontal text density to find real text segments"""
        line_region = binary_image[y1:y2, :]
        horizontal_profile = np.sum(line_region, axis=0) > 0

        # Find continuous text segments
        segments = []
        start = None

        for x, val in enumerate(horizontal_profile):
            if val and start is None:
                start = x
            elif (not val or x == len(horizontal_profile) - 1) and start is not None:
                end = x if not val else x + 1
                if end - start >= self.min_segment_width:
                    segments.append((start, end))
                start = None

        return segments

    def _should_merge_lines(self, line1: Tuple[int, int], line2: Tuple[int, int],
                            binary_image: np.ndarray) -> bool:
        """Determine if two line segments should be merged"""
        y1_end = line1[1]
        y2_start = line2[0]

        if y2_start - y1_end > self.vertical_merge_threshold:
            return False

        # Check if there's significant text content in both lines
        region1 = binary_image[line1[0]:line1[1], :]
        region2 = binary_image[line2[0]:line2[1], :]

        density1 = np.sum(region1) / region1.size
        density2 = np.sum(region2) / region2.size

        # If one segment has very low density compared to the other,
        # they might be part of the same line
        avg_density = (density1 + density2) / 2
        density_ratio = min(density1, density2) / max(density1, density2)

        # Check for horizontal overlap
        horizontal_profile1 = np.sum(region1, axis=0) > 0
        horizontal_profile2 = np.sum(region2, axis=0) > 0

        overlap = np.sum(horizontal_profile1 & horizontal_profile2)
        max_width = max(np.sum(horizontal_profile1), np.sum(horizontal_profile2))

        overlap_ratio = overlap / max_width if max_width > 0 else 0

        # Merge if:
        # 1. Lines are close vertically
        # 2. One line has much lower density (might be part of the same character)
        # 3. There's significant horizontal overlap
        return (density_ratio < 0.3 or overlap_ratio > 0.3) and avg_density > self.horizontal_density_threshold

    def _merge_line_segments(self, lines: List[Tuple[int, int]],
                             binary_image: np.ndarray) -> List[Tuple[int, int]]:
        """Merge line segments that are likely part of the same line"""
        if not lines:
            return lines

        merged_lines = []
        current_line = lines[0]

        for next_line in lines[1:]:
            if self._should_merge_lines(current_line, next_line, binary_image):
                # Merge lines
                current_line = (current_line[0], next_line[1])
            else:
                merged_lines.append(current_line)
                current_line = next_line

        merged_lines.append(current_line)
        return merged_lines

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing with better text preservation"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Use Otsu's thresholding for better text separation
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Remove small noise while preserving text
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Enhance text connectivity
        kernel_vertical = np.ones((2, 1), np.uint8)
        binary = cv2.dilate(binary, kernel_vertical, iterations=1)

        return binary

    def segment_page(self, image: Image.Image) -> List[LineSegment]:
        """Segment page with improved line detection and merging"""
        if not self._is_loaded:
            self.load()

        try:
            image_np = np.array(image)
            binary = self._preprocess_image(image_np)

            # Get vertical projection profile
            vertical_profile = np.sum(binary, axis=1)

            # Smooth profile
            kernel_size = 3
            vertical_profile = np.convolve(
                vertical_profile,
                np.ones(kernel_size) / kernel_size,
                mode='same'
            )

            # Find initial line boundaries
            min_value = np.mean(vertical_profile[vertical_profile > 0]) * 0.2
            lines = []
            start = None

            for i, val in enumerate(vertical_profile):
                if val > min_value and start is None:
                    start = i
                elif (val <= min_value or i == len(vertical_profile) - 1) and start is not None:
                    if i - start >= self.min_line_height:
                        lines.append((start, i))
                    start = None

            # Merge lines that were incorrectly split
            merged_lines = self._merge_line_segments(lines, binary)

            # Create line segments
            line_segments = []
            for y1, y2 in merged_lines:
                # Analyze horizontal density to refine segment boundaries
                text_segments = self._analyze_horizontal_density(binary, y1, y2)

                if text_segments:
                    # Use the full width of all text segments
                    x1 = max(0, min(seg[0] for seg in text_segments) - 5)
                    x2 = min(image_np.shape[1], max(seg[1] for seg in text_segments) + 5)

                    bbox = BoundingBox(
                        x1=int(x1),
                        y1=int(max(0, y1 - 2)),
                        x2=int(x2),
                        y2=int(min(image_np.shape[0], y2 + 2)),
                        confidence=1.0
                    )

                    line_image = image.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
                    line_segments.append(LineSegment(image=line_image, bbox=bbox))

            ml_logger.info(f"Segmented {len(line_segments)} lines from page")
            save_debug_image(image, line_segments, 'storage/debug_images/debug.png')

            return line_segments

        except Exception as e:
            ml_logger.error(f"Error during page segmentation: {str(e)}")
            raise


def save_debug_image(image: Image.Image, line_segments: List[LineSegment], output_path: str):
    """Save a debug image showing detected lines"""

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
            f"Line {i + 1}",
            (segment.bbox.x1, segment.bbox.y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    cv2.imwrite(output_path, img)


def create_segmenter(segmenter_type: str = "layout", device: str = None) -> BaseDocumentSegmenter:
    """Factory function to create document segmenters"""
    segmenters = {
        "doctr": DoctrSegmenter,
        "layout": LayoutSegmenter
    }

    if segmenter_type not in segmenters:
        raise ValueError(f"Unknown segmenter type: {segmenter_type}. Valid types are: {list(segmenters.keys())}")

    return segmenters[segmenter_type](device=device)
