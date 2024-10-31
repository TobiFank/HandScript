# tests/utils/test_files.py
import pytest
import io
import shutil
from pathlib import Path
from app.utils.files import save_upload_file, delete_file, get_relative_path
from fastapi import UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile

@pytest.fixture
def mock_upload_file():
    async def _create_upload_file(filename: str, content: bytes):
        spooled_file = io.BytesIO(content)
        return UploadFile(
            filename=filename,
            file=spooled_file
        )
    return _create_upload_file

@pytest.mark.asyncio
async def test_save_upload_file(mock_upload_file, temp_storage_dir):
    """Test saving an uploaded file"""
    test_content = b"test file content"
    upload_file = await mock_upload_file("test.txt", test_content)

    # Ensure directory exists
    temp_storage_dir.mkdir(parents=True, exist_ok=True)

    saved_path = await save_upload_file(upload_file, temp_storage_dir)

    assert saved_path.exists()
    assert saved_path.read_bytes() == test_content
    assert saved_path.suffix == ".txt"

@pytest.mark.asyncio
async def test_save_upload_file_creates_directory(mock_upload_file, temp_storage_dir):
    """Test saving file creates directory if it doesn't exist"""
    new_dir = temp_storage_dir / "new_directory"
    if new_dir.exists():
        shutil.rmtree(new_dir)

    new_dir.mkdir(parents=True, exist_ok=True)

    upload_file = await mock_upload_file("test.txt", b"content")
    saved_path = await save_upload_file(upload_file, new_dir)

    assert new_dir.exists()
    assert saved_path.exists()
    assert saved_path.read_bytes() == b"content"