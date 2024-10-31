# tests/services/test_training.py
import pytest
from PIL import Image, ImageDraw, ImageFont
from app.models.training_sample import TrainingSample
from app.services.training import TrainingService


def create_test_training_image(text: str, size=(1240, 1754)) -> Image.Image:
    """Create a test image with clearly visible text for line detection"""
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
    except IOError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 40)
        except IOError:
            font = ImageFont.load_default()

    y_position = size[1] // 4
    for line in text.split('\n'):
        draw.text((100, y_position), line, fill=(0, 0, 0), font=font)
        y_position += 80

    return image


@pytest.fixture
def mock_ocr_service(monkeypatch):
    """Mock OCR service for testing"""

    class MockOCRService:
        async def process_training_sample(self, image_path):
            from app.ml.segmentation import LineSegment
            # Create mock line segments
            segments = [
                LineSegment(
                    image=Image.new('RGB', (100, 30), 'white'),
                    bbox=(0, 0, 100, 30),
                    text="Test line 1",
                    confidence=0.9
                ),
                LineSegment(
                    image=Image.new('RGB', (100, 30), 'white'),
                    bbox=(0, 40, 100, 70),
                    text="Test line 2",
                    confidence=0.9
                )
            ]
            return segments, "Test line 1\nTest line 2"

    monkeypatch.setattr("app.services.training.ocr_service", MockOCRService())
    return MockOCRService


@pytest.mark.asyncio
async def test_training_service_train_writer(
        mock_lora_trainer,
        mock_ocr_service,
        db_session,
        sample_writer,
        temp_storage_dir
):
    """Test writer model training"""
    # Create training sample with multiple lines of text
    test_text = "First line of test text\nSecond line of text\nThird line here"
    test_image = create_test_training_image(test_text)
    image_path = temp_storage_dir / "training_samples" / str(sample_writer.id) / "sample.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    test_image.save(image_path, format='PNG')

    # Add training sample to database
    sample = TrainingSample(
        writer_id=sample_writer.id,
        image_path=str(image_path.relative_to(temp_storage_dir)),
        text=test_text
    )
    db_session.add(sample)
    db_session.commit()

    # Train writer
    service = TrainingService()
    result = await service.train_writer_model(
        db_session,
        sample_writer.id,
        [(image_path, test_text)]
    )

    assert result["success"] is True
    assert "model_path" in result

    # Verify writer status was updated
    db_session.refresh(sample_writer)
    assert sample_writer.status == "ready"
    assert sample_writer.is_trained is True


@pytest.mark.asyncio
async def test_training_service_error_handling(mock_lora_trainer, db_session, sample_writer):
    """Test training service error handling"""
    service = TrainingService()
    with pytest.raises(ValueError, match="No training samples found"):
        await service.train_writer_model(db_session, sample_writer.id, [])
