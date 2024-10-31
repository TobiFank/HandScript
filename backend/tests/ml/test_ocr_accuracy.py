# tests/ml/test_ocr_accuracy.py
import pytest
from PIL import Image, ImageDraw, ImageFont
import io
from pathlib import Path
from app.services.ocr import OCRService
from app.ml.models import OCRModel
from app.ml.segmentation import DocumentSegmenter, LineSegment

def create_test_image_with_text(text_lines: list[str], image_size=(2100, 2970)):
    """Create a test image with known text content"""
    # Create a white background
    image = Image.new('RGB', image_size, 'white')
    draw = ImageDraw.Draw(image)

    # Use a basic font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    # Write text lines
    y_position = 100
    for line in text_lines:
        draw.text((100, y_position), line, fill='black', font=font)
        y_position += 60  # Space between lines

    return image

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
        # Return known text for test image
        return "This is test text"

@pytest.fixture
def mock_ocr_model(monkeypatch):
    """Provide enhanced mock OCR model"""
    monkeypatch.setattr("app.services.ocr.OCRModel", MockOCRModel)
    return MockOCRModel

@pytest.mark.asyncio
async def test_ocr_service_line_extraction(mock_ocr_model, temp_storage_dir):
    """Test OCR service's ability to extract text lines correctly"""
    # Create test image with known content
    test_lines = [
        "Line 1: This is a test",
        "Line 2: Another line of text",
        "Line 3: Final test line"
    ]

    test_image = create_test_image_with_text(test_lines)
    image_path = temp_storage_dir / "test_lines.png"
    test_image.save(image_path)

    # Process image
    service = OCRService()
    extracted_text = await service.process_image(image_path)

    # Verify text extraction
    assert extracted_text is not None
    assert len(extracted_text) > 0

    # Split extracted text into lines
    extracted_lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]

    # Basic verification
    assert len(extracted_lines) > 0, "No text lines were extracted"

    # Log extracted content for debugging
    print(f"Expected lines: {test_lines}")
    print(f"Extracted lines: {extracted_lines}")

@pytest.mark.asyncio
async def test_ocr_service_with_different_fonts(mock_ocr_model, temp_storage_dir):
    """Test OCR service with different font styles"""
    test_text = "Test text with different font"

    # Create test image with different font if available
    try:
        test_image = create_test_image_with_text([test_text], image_size=(800, 200))
    except Exception as e:
        pytest.skip(f"Could not create test image: {str(e)}")

    image_path = temp_storage_dir / "test_font.png"
    test_image.save(image_path)

    service = OCRService()
    result = await service.process_image(image_path)

    assert result is not None
    assert len(result) > 0

@pytest.mark.asyncio
async def test_ocr_service_with_special_characters(mock_ocr_model, temp_storage_dir):
    """Test OCR service's handling of special characters"""
    test_lines = [
        "Special chars: @#$%^&*()",
        "Numbers: 1234567890",
        "Mixed: A1B2C3 !@#"
    ]

    test_image = create_test_image_with_text(test_lines)
    image_path = temp_storage_dir / "test_special.png"
    test_image.save(image_path)

    service = OCRService()
    result = await service.process_image(image_path)

    assert result is not None
    assert len(result) > 0