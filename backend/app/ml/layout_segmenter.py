# app/ml/layout_segmenter.py
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import LayoutLMv3FeatureExtractor
from scipy.signal import find_peaks, argrelmin, savgol_filter

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
        """Improved vertical profile computation with better smoothing"""
        # Calculate raw profile
        profile = np.sum(binary_image, axis=1).astype(np.float32)

        # Normalize
        if profile.max() > 0:
            profile = profile / profile.max()

        # Apply Savitzky-Golay filter for better smoothing
        window = 21  # Increased window size
        profile_smooth = savgol_filter(profile, window, 3)

        # Debug visualization
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 5))
            plt.plot(profile, 'b-', alpha=0.5, label='Original')
            plt.plot(profile_smooth, 'r-', label='Smoothed')
            plt.title("Vertical Profile")
            plt.legend()
            plt.savefig('debug_profile.png')
            plt.close()
        except Exception as e:
            ml_logger.warning(f"Failed to save debug profile: {str(e)}")

        return profile_smooth

    def _detect_text_lines(self, binary_image: np.ndarray,
                           vertical_profile: np.ndarray) -> List[Tuple[int, int]]:
        """Enhanced text line detection with better valley identification"""
        profile = vertical_profile.flatten()
        height = len(profile)

        # Find the peaks (text line centers)
        peaks, peak_properties = find_peaks(
            profile,
            distance=self.params.peak_min_distance,
            prominence=self.params.peak_prominence,
            width=5  # Added width parameter
        )

        # Calculate the typical line height from peak widths
        peak_widths = peak_properties["widths"]
        typical_height = int(np.median(peak_widths) * 2.5)  # Use 2.5x the peak width

        # Create line regions based on peaks
        line_regions = []
        for i, peak in enumerate(peaks):
            # Calculate region bounds using typical height
            half_height = typical_height // 2
            start = max(0, peak - half_height)
            end = min(height - 1, peak + half_height)

            # Extend to local minima
            # Look for local minimum above
            local_profile = profile[max(0, start-5):peak]
            if len(local_profile) > 0:
                local_min = max(0, start-5 + np.argmin(local_profile))
                start = local_min

            # Look for local minimum below
            local_profile = profile[peak:min(height, end+5)]
            if len(local_profile) > 0:
                local_min = peak + np.argmin(local_profile)
                end = min(height-1, local_min)

            # Add the region if it doesn't overlap significantly with previous regions
            if not line_regions or start > line_regions[-1][1] - typical_height * 0.1:
                line_regions.append((start, end))

        return line_regions

    def _analyze_line_components(self, binary_image: np.ndarray,
                                 y1: int, y2: int) -> Optional[Tuple[int, int]]:
        """Improved component analysis to avoid horizontal cuts"""
        line_region = binary_image[y1:y2, :]

        # Use horizontal projection to find text bounds
        horiz_proj = np.sum(line_region, axis=0) > 0

        # Find continuous text regions
        transitions = np.diff(horiz_proj.astype(int))
        starts = np.where(transitions == 1)[0] + 1
        ends = np.where(transitions == -1)[0] + 1

        if len(starts) == 0 or len(ends) == 0:
            # Use full width if no clear transitions
            cols = np.where(horiz_proj)[0]
            if len(cols) == 0:
                return None
            return (cols[0], cols[-1])

        # Take the leftmost start and rightmost end
        x1 = starts[0]
        x2 = ends[-1]

        # Add extra padding for safety
        x1 = max(0, x1 - 10)
        x2 = min(binary_image.shape[1], x2 + 10)

        return (x1, x2)

    def segment_page(self, image: Image.Image) -> List[LineSegment]:
        """Segment page into text lines with improved handling"""
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

                # Create bounding box with generous padding
                bbox = BoundingBox(
                    x1=max(0, int(x1) - 15),  # Increased horizontal padding
                    y1=max(0, int(y1) - 15),   # Increased top padding
                    x2=min(image_np.shape[1], int(x2) + 15),  # Increased horizontal padding
                    y2=min(image_np.shape[0], int(y2) + 15),   # Slightly more bottom padding
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
