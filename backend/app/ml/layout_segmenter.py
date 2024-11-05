# app/ml/layout_segmenter.py
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import find_peaks, argrelmin
from transformers import LayoutLMv3FeatureExtractor

from .base_document_segmenter import BaseDocumentSegmenter
from .types import LineSegment, BoundingBox
from ..utils.logging import ml_logger


@dataclass
class LineDetectionParams:
    """Parameters for fine-tuning line detection"""
    # Baseline detection
    baseline_variation_tolerance: int = 2  # Keep this small
    max_baseline_skew: float = 0.05  # Keep this for horizontal lines

    # Text height analysis
    min_text_height: int = 10  # Slightly increased
    max_text_height: int = 40  # Keep this
    min_line_spacing: int = 2  # Slightly increased
    height_estimation_window: int = 20  # Keep this

    # Horizontal text analysis
    min_word_spacing: int = 5  # Keep this
    horizontal_density_threshold: float = 0.015  # Slightly increased to avoid fragments

    # Profile analysis
    smoothing_window: int = 7  # Increased to reduce local variations
    peak_prominence: float = 0.08  # Increased to be less sensitive
    valley_depth: float = 0.04  # Increased to be less sensitive

    # Valley detection
    min_valley_width: int = 2  # Slightly increased
    max_valley_width: int = 10  # Keep this

    # Peak detection
    peak_min_distance: int = 15  # Increased to match typical line spacing


class LayoutSegmenter(BaseDocumentSegmenter):
    """Enhanced document segmentation optimized for handwritten text lines"""

    def __init__(self, device: str = None):
        super().__init__(device)
        self.feature_extractor = None
        self.params = LineDetectionParams()

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

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Improved preprocessing optimized for handwritten text"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Enhanced contrast with more aggressive CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Binary thresholding with Otsu's method
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Save debug image
        cv2.imwrite('debug_preprocessed.png', binary)

        return binary

    def _compute_vertical_profile(self, binary_image: np.ndarray) -> np.ndarray:
        """Improved vertical profile computation for handwritten text"""
        height = binary_image.shape[0]

        # Calculate raw profile
        profile = np.sum(binary_image, axis=1).astype(np.float32)

        # Normalize
        if profile.max() > 0:
            profile = profile / profile.max()

        # Use smaller window Gaussian smoothing
        kernel_size = self.params.smoothing_window
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        profile = cv2.GaussianBlur(profile, (1, kernel_size), 0)

        # Debug visualization
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 5))
            plt.plot(profile)
            plt.title("Vertical Profile")
            # Add horizontal lines for thresholds
            plt.axhline(y=self.params.valley_depth, color='r', linestyle='--', label='Valley Threshold')
            plt.axhline(y=self.params.peak_prominence, color='g', linestyle='--', label='Peak Threshold')
            plt.legend()
            plt.savefig('debug_profile.png')
            plt.close()
        except Exception as e:
            ml_logger.warning(f"Failed to save debug profile: {str(e)}")

        return profile

    def _detect_text_lines(self, binary_image: np.ndarray,
                           vertical_profile: np.ndarray) -> List[Tuple[int, int]]:
        """Enhanced text line detection using valley detection"""
        profile = vertical_profile.flatten()
        height = len(profile)

        # Find valleys (local minima)
        valleys = argrelmin(profile, order=self.params.smoothing_window)[0]

        # Find peaks (local maxima)
        peaks, _ = find_peaks(
            profile,
            distance=self.params.peak_min_distance,
            prominence=self.params.peak_prominence
        )

        # Combine peaks and valleys to determine line regions
        line_regions = []

        # Add image top if needed
        if peaks[0] > self.params.min_text_height:
            peaks = np.insert(peaks, 0, 0)

        # Add image bottom if needed
        if peaks[-1] < height - self.params.min_text_height:
            peaks = np.append(peaks, height - 1)

        # Create line regions between valleys
        for i in range(len(valleys) - 1):
            start = valleys[i]
            end = valleys[i + 1]

            # Check if there's a peak between these valleys
            peaks_between = peaks[(peaks > start) & (peaks < end)]

            if len(peaks_between) > 0:
                # Expand region slightly
                region_start = max(0, start - 2)
                region_end = min(height - 1, end + 2)

                # Verify region
                region_height = region_end - region_start
                if region_height >= self.params.min_text_height:
                    # Check density
                    region = binary_image[region_start:region_end, :]
                    density = np.sum(region) / region.size

                    if density > self.params.horizontal_density_threshold:
                        line_regions.append((region_start, region_end))

        # Debug visualization
        try:
            debug_img = binary_image.copy()
            if len(debug_img.shape) == 2:
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)

            # Draw detected lines
            for start, end in line_regions:
                cv2.line(debug_img, (0, start), (binary_image.shape[1], start), (0, 255, 0), 1)
                cv2.line(debug_img, (0, end), (binary_image.shape[1], end), (0, 0, 255), 1)

            cv2.imwrite('debug_lines.png', debug_img)

            # Save profile with detected regions
            plt.figure(figsize=(15, 5))
            plt.plot(profile)
            plt.vlines(valleys, 0, 1, colors='r', linestyles='dashed', label='Valleys')
            plt.vlines(peaks, 0, 1, colors='g', linestyles='dashed', label='Peaks')
            plt.title("Profile with Detected Regions")
            plt.legend()
            plt.savefig('debug_regions.png')
            plt.close()

        except Exception as e:
            ml_logger.warning(f"Failed to save debug images: {str(e)}")

        return line_regions

    def _analyze_line_components(self, binary_image: np.ndarray,
                                 y1: int, y2: int) -> Optional[Tuple[int, int]]:
        """Analyze components within a line region"""
        line_region = binary_image[y1:y2, :]

        # Simple horizontal bounds
        col_profile = np.sum(line_region, axis=0) > 0
        cols = np.where(col_profile)[0]

        if len(cols) == 0:
            return None

        x1, x2 = cols[0], cols[-1]
        return (x1, x2)

    def segment_page(self, image: Image.Image) -> List[LineSegment]:
        """Segment page into text lines with improved handling of handwritten text"""
        if not self._is_loaded:
            self.load()

        try:
            # Convert and preprocess
            image_np = np.array(image)
            binary = self._preprocess_image(image_np)

            # Compute vertical profile
            vertical_profile = self._compute_vertical_profile(binary)

            # Detect line regions
            line_regions = self._detect_text_lines(binary, vertical_profile)

            ml_logger.info(f"Detected {len(line_regions)} initial line regions")

            # Process each line region
            line_segments = []
            for y1, y2 in line_regions:
                x_bounds = self._analyze_line_components(binary, y1, y2)

                if x_bounds is None:
                    continue

                x1, x2 = x_bounds

                # Increased padding for bounding boxes
                bbox = BoundingBox(
                    x1=max(0, int(x1) - 5),  # Increased horizontal padding
                    y1=max(0, int(y1) - 3),  # Increased top padding
                    x2=min(image_np.shape[1], int(x2) + 5),  # Increased horizontal padding
                    y2=min(image_np.shape[0], int(y2) + 4),  # Increased bottom padding slightly more
                    confidence=1.0
                )

                # Extract line image
                line_image = image.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
                line_segments.append(LineSegment(image=line_image, bbox=bbox))

            ml_logger.info(f"Final number of line segments: {len(line_segments)}")
            return line_segments

        except Exception as e:
            ml_logger.error(f"Error during page segmentation: {str(e)}")
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
