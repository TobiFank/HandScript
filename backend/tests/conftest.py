# tests/conftest.py
from datetime import datetime
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from pathlib import Path
import shutil
import tempfile
import os

from app.main import app
from app.database import Base, get_db
from app.models import Project, Document, Page, Writer
from app.config import settings

# Create test database
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture(scope="session")
def engine():
    """Create test database engine"""
    engine = create_engine(
        SQLALCHEMY_TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool  # Needed for SQLite in-memory database
    )
    return engine

@pytest.fixture(scope="session")
def tables(engine):
    """Create all tables in the test database"""
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)

@pytest.fixture
def db_session(engine, tables):
    """Creates a new database session for a test"""
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection)()

    yield session

    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(scope="session")
def temp_storage_dir():
    """Create temporary storage directory for test files"""
    temp_dir = tempfile.mkdtemp()
    # Create subdirectories
    for subdir in ["images", "models", "exports", "training_samples"]:
        Path(temp_dir, subdir).mkdir(parents=True, exist_ok=True)
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture(autouse=True)
def override_settings(temp_storage_dir):
    """Override settings for testing"""
    original_storage = settings.STORAGE_PATH
    original_images = settings.IMAGES_PATH
    original_models = settings.MODELS_PATH
    original_exports = settings.EXPORTS_PATH

    # Override settings
    settings.STORAGE_PATH = temp_storage_dir
    settings.IMAGES_PATH = temp_storage_dir / "images"
    settings.MODELS_PATH = temp_storage_dir / "models"
    settings.EXPORTS_PATH = temp_storage_dir / "exports"

    yield

    # Restore settings
    settings.STORAGE_PATH = original_storage
    settings.IMAGES_PATH = original_images
    settings.MODELS_PATH = original_models
    settings.EXPORTS_PATH = original_exports

@pytest.fixture
def client(db_session):
    """Test client using the test database"""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def sample_project(db_session):
    """Create a sample project"""
    project = Project(
        name="Test Project",
        description="Test Description"
    )
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)
    return project

@pytest.fixture
def sample_document(db_session, sample_project):
    """Create a sample document"""
    document = Document(
        name="Test Document",
        description="Test Description",
        project_id=sample_project.id
    )
    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)
    return document

@pytest.fixture
def sample_writer(db_session):
    """Create a sample writer"""
    writer = Writer(
        name="Test Writer",
        status="ready",
        pages_processed=0
    )
    db_session.add(writer)
    db_session.commit()
    db_session.refresh(writer)
    return writer

@pytest.fixture
def sample_page(db_session, sample_document, sample_writer, temp_storage_dir):
    """Create a sample page"""
    # Create test image file
    image_path = temp_storage_dir / "images" / "test_image.png"
    with open(image_path, "wb") as f:
        f.write(b"fake image content")

    page = Page(
        document_id=sample_document.id,
        writer_id=sample_writer.id,
        image_path=str(image_path.relative_to(temp_storage_dir)),
        page_number=1,
        processing_status="completed",
        updated_at=datetime.utcnow()  # Add this line
    )
    db_session.add(page)
    db_session.commit()
    db_session.refresh(page)
    return page

@pytest.fixture
def mock_ocr_model(monkeypatch):
    """Mock OCR model for testing"""
    class MockOCRModel:
        def __init__(self, *args, **kwargs):
            pass

        def extract_text(self, image):
            return "Sample extracted text"

    monkeypatch.setattr("app.services.ocr.OCRModel", MockOCRModel)
    return MockOCRModel

@pytest.fixture
def mock_lora_trainer(monkeypatch, temp_storage_dir):
    """Mock LORA trainer for testing"""
    class MockLoraTrainer:
        def setup_model(self):
            return True

        def train(self, train_images, train_texts, *args, **kwargs):
            # Use a path within temp_storage_dir
            return temp_storage_dir / "models" / "test_model"

    monkeypatch.setattr("app.services.training.LoraTrainer", MockLoraTrainer)
    return MockLoraTrainer

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_files():
    """Clean up test files after all tests are done"""
    yield
    # Cleanup any test files that might have been created
    test_files = [
        "test.db",
        "handscript.db",
        "test-handscript.db"
    ]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)