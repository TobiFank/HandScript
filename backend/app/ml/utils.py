# app/ml/utils.py
import os
from typing import List

import cv2
import numpy as np
from PIL import Image

from .types import LineSegment


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
