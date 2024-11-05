# app/ml/layout_segmenter.py
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from transformers import LayoutLMv3FeatureExtractor

from .base_document_segmenter import BaseDocumentSegmenter
from .types import LineSegment, BoundingBox
from ..utils.logging import ml_logger


class LayoutSegmenter(BaseDocumentSegmenter):
    """Document segmentation using hybrid approach for line detection"""

    def __init__(self, device: str = None):
        super().__init__(device)
        self.feature_extractor = None
        # Minimum height (in pixels) for a text line
        # Set low for thin handwriting strokes
        self.min_line_height = 2

        # Minimum width (in pixels) for a text segment
        # Reduced to catch shorter handwritten words
        self.min_segment_width = 10

        # Maximum vertical distance (in pixels) to merge nearby lines
        # Increased to handle uneven handwriting
        self.vertical_merge_threshold = 1

        # Minimum text density threshold for line detection
        # Lowered because handwriting is typically less dense than printed text
        self.horizontal_density_threshold = 0.05

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
        """Enhanced preprocessing optimized for handwritten text"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Enhance contrast to better separate text from background
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)

        # Use adaptive thresholding instead of Otsu's
        # Better handles varying stroke intensities in handwriting
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=25,  # Adjust based on typical text size
            C=10
        )

        # Minimal noise removal to preserve handwriting details
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Light dilation to connect broken strokes
        kernel_vertical = np.ones((2, 1), np.uint8)
        binary = cv2.dilate(binary, kernel_vertical, iterations=1)

        return binary

    def _should_merge_lines(self, line1: Tuple[int, int], line2: Tuple[int, int],
                            binary_image: np.ndarray) -> bool:
        """Determine if two line segments should be merged, optimized for handwriting"""
        y1_end = line1[1]
        y2_start = line2[0]

        # Increased threshold for vertical gaps in handwriting
        if y2_start - y1_end > self.vertical_merge_threshold:
            return False

        region1 = binary_image[line1[0]:line1[1], :]
        region2 = binary_image[line2[0]:line2[1], :]

        density1 = np.sum(region1) / region1.size
        density2 = np.sum(region2) / region2.size

        avg_density = (density1 + density2) / 2
        density_ratio = min(density1, density2) / max(density1, density2)

        horizontal_profile1 = np.sum(region1, axis=0) > 0
        horizontal_profile2 = np.sum(region2, axis=0) > 0

        overlap = np.sum(horizontal_profile1 & horizontal_profile2)
        max_width = max(np.sum(horizontal_profile1), np.sum(horizontal_profile2))

        overlap_ratio = overlap / max_width if max_width > 0 else 0

        # More lenient merging criteria for handwriting
        # Density ratio threshold increased to handle varying stroke weights
        # Overlap ratio threshold decreased to handle uneven baselines
        return (density_ratio < 0.4 or overlap_ratio > 0.2) and avg_density > self.horizontal_density_threshold

    def segment_page(self, image: Image.Image) -> List[LineSegment]:
        """Segment page with parameters optimized for handwritten text"""
        if not self._is_loaded:
            self.load()

        try:
            image_np = np.array(image)
            binary = self._preprocess_image(image_np)

            # Vertical profile with gentler smoothing
            vertical_profile = np.sum(binary, axis=1)
            kernel_size = 5  # Increased for more stable line detection
            vertical_profile = np.convolve(
                vertical_profile,
                np.ones(kernel_size) / kernel_size,
                mode='same'
            )

            # Lower threshold for detecting handwritten lines
            min_value = np.mean(vertical_profile[vertical_profile > 0]) * 0.15
            lines = []
            start = None

            for i, val in enumerate(vertical_profile):
                if val > min_value and start is None:
                    start = i
                elif (val <= min_value or i == len(vertical_profile) - 1) and start is not None:
                    if i - start >= self.min_line_height:
                        lines.append((start, i))
                    start = None

            merged_lines = self._merge_line_segments(lines, binary)

            # Create line segments with increased padding for handwriting
            line_segments = []
            for y1, y2 in merged_lines:
                text_segments = self._analyze_horizontal_density(binary, y1, y2)

                if text_segments:
                    # Increased padding to catch descenders and ascenders
                    x1 = max(0, min(seg[0] for seg in text_segments) - 8)
                    x2 = min(image_np.shape[1], max(seg[1] for seg in text_segments) + 8)

                    bbox = BoundingBox(
                        x1=int(x1),
                        y1=int(max(0, y1 - 4)),  # Increased vertical padding
                        x2=int(x2),
                        y2=int(min(image_np.shape[0], y2 + 4)),  # Increased vertical padding
                        confidence=1.0
                    )

                    line_image = image.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
                    line_segments.append(LineSegment(image=line_image, bbox=bbox))

            return line_segments

        except Exception as e:
            ml_logger.error(f"Error during page segmentation: {str(e)}")
            raise
