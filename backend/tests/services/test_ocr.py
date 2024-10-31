# tests/services/test_ocr.py
import pytest
from PIL import Image, ImageDraw, ImageFont
from app.ml.segmentation import DocumentSegmenter
from app.services.ocr import OCRService


class MockOCRModel:
    """Enhanced mock OCR model for testing"""

    def __init__(self, *args, **kwargs):
        self.segmenter = DocumentSegmenter()
        self.loaded = False

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def extract_text(self, image):
        return "Test extracted text"


def create_test_page_image(lines: list[str]) -> Image.Image:
    """Create a test page with multiple lines of text"""
    # Create A4 sized image (2480 x 3508 pixels at 300 DPI)
    image = Image.new('RGB', (2480, 3508), 'white')
    draw = ImageDraw.Draw(image)

    # Try to use a system font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    # Write lines with proper spacing
    y_position = 100
    for line in lines:
        draw.text((100, y_position), line, fill='black', font=font)
        y_position += 60

    return image


@pytest.fixture
def mock_ocr_model(monkeypatch):
    """Mock OCR model fixture"""
    monkeypatch.setattr("app.services.ocr.OCRModel", MockOCRModel)
    return MockOCRModel


@pytest.mark.asyncio
async def test_ocr_service_process_image(mock_ocr_model, temp_storage_dir):
    """Test OCR service image processing"""
    # Create test image
    test_lines = [
        "This is the first line of text.",
        "Here is a second line to test.",
        "And finally a third line."
    ]

    test_image = create_test_page_image(test_lines)
    image_path = temp_storage_dir / "test.png"
    test_image.save(image_path)

    # Process image
    service = OCRService()
    result = await service.process_image(image_path)

    assert result is not None
    assert len(result) > 0
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_ocr_service_line_segmentation(mock_ocr_model, temp_storage_dir):
    """Test OCR service's line segmentation capabilities"""
    # Create test page with known line spacing
    test_lines = [
        "Line 1: Test text with consistent spacing",
        "Line 2: Another line to verify segmentation",
        "Line 3: Final line with specific content"
    ]

    test_image = create_test_page_image(test_lines)
    image_path = temp_storage_dir / "test_lines.png"
    test_image.save(image_path)

    service = OCRService()
    result = await service.process_image(image_path)

    # Basic verification
    assert result is not None
    assert len(result) > 0

    # Split into lines and verify line count
    result_lines = [line.strip() for line in result.split('\n') if line.strip()]
    assert len(result_lines) > 0
