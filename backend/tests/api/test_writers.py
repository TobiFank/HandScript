# tests/api/test_writers.py
import io

import pytest
from PIL import Image
from app.models.training_sample import TrainingSample
from fastapi import status


def create_test_image_with_text(text: str, size=(100, 100)):
    """Create a test image with text"""
    image = Image.new('RGB', size, 'white')
    return image


@pytest.fixture
def mock_training_service(monkeypatch):
    """Mock training service for testing"""

    class MockTrainingService:
        async def train_writer_model(self, db, writer_id, sample_pages):
            return {"success": True, "model_path": "models/test_model.pt"}

    monkeypatch.setattr("app.api.writers.training_service", MockTrainingService())
    return MockTrainingService()


def test_create_writer_with_full_data(client):
    """Test writer creation with all fields"""
    response = client.post(
        "/api/writers",
        json={
            "name": "New Writer",
            "language": "english"
        }
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["name"] == "New Writer"
    assert data["status"] == "untrained"  # Default value
    assert data["pages_processed"] == 0


def test_upload_writer_model_invalid_writer(client):
    """Test uploading model for non-existent writer"""
    model_file = io.BytesIO(b"fake model data")
    response = client.post(
        "/api/writers/99999/upload_model",
        files={"file": ("model.pt", model_file, "application/octet-stream")}
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_train_writer_with_mock(client, sample_writer, mock_training_service, temp_storage_dir, db_session):
    """Test writer training with mocked training service"""
    # Create training sample
    training_dir = temp_storage_dir / "training_samples" / str(sample_writer.id)
    training_dir.mkdir(parents=True, exist_ok=True)

    # Create test image
    test_image = create_test_image_with_text("Sample training text")
    image_path = training_dir / "sample1.png"
    test_image.save(image_path)

    # Create training sample in database
    sample = TrainingSample(
        writer_id=sample_writer.id,
        image_path=str(image_path.relative_to(temp_storage_dir)),
        text="Sample training text"
    )
    db_session.add(sample)
    db_session.commit()

    response = client.post(f"/api/writers/{sample_writer.id}/train")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert "message" in data
    assert "sample_count" in data


def test_train_writer_invalid_writer(client):
    """Test training non-existent writer"""
    response = client.post("/api/writers/99999/train")
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_training_status_error_handling(client, sample_writer, db_session):
    """Test training status with error state"""
    sample_writer.status = "error"
    db_session.commit()

    response = client.get(f"/api/writers/{sample_writer.id}/training_status")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "error"
