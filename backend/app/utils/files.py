# backend/app/utils/files.py
import shutil
from pathlib import Path
from uuid import uuid4
from fastapi import UploadFile
from ..config import settings

async def save_upload_file(upload_file: UploadFile, directory: Path) -> Path:
    """Save an uploaded file with a unique name and return the path"""
    # Create unique filename
    file_extension = Path(upload_file.filename).suffix
    unique_filename = f"{uuid4()}{file_extension}"
    file_path = directory / unique_filename

    # Save file
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return file_path

async def delete_file(file_path: Path):
    """Safely delete a file if it exists"""
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

def get_relative_path(absolute_path: Path, base_path: Path) -> str:
    """Convert absolute path to relative path for database storage"""
    # Ensure both paths are absolute
    absolute_path = Path(absolute_path).absolute()
    base_path = Path(base_path).absolute()

    try:
        return str(absolute_path.relative_to(base_path))
    except ValueError:
        # If paths are not in expected format, try to resolve relative paths
        if not absolute_path.is_absolute():
            # If absolute_path is relative, make it relative to base_path
            return str(absolute_path)
        raise

async def load_file_as_uploadfile(file_path: Path) -> UploadFile:
    """Convert a file path to an UploadFile object"""
    # Open the file in binary read mode
    file = open(file_path, "rb")

    # Create an UploadFile instance
    upload_file = UploadFile(
        filename=file_path.name,
        file=file,
        content_type="image/jpeg"  # You might want to detect this dynamically
    )

    return upload_file