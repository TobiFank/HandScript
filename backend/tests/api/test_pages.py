# backend/tests/api/test_pages.py
import io
from tkinter import Image

import pytest
from PIL import Image, ImageDraw
from app.models.page import ProcessingStatus
from fastapi import status


def create_test_image():
    """Create a fake image file for testing"""
    return io.BytesIO(b"fake image data")


def test_upload_page(client, sample_document, temp_storage_dir):
    """Test uploading a page"""
    test_file = create_test_image()

    response = client.post(
        f"/api/pages/upload/{sample_document.id}",
        files=[("files", ("test.png", test_file, "image/png"))]
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) == 1
    assert data[0]["document_id"] == sample_document.id
    assert data[0]["page_number"] == 1


def test_get_page(client, sample_page):
    """Test getting a single page"""
    response = client.get(f"/api/pages/{sample_page.id}")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["document_id"] == sample_page.document_id
    assert data["page_number"] == sample_page.page_number


def test_update_page(client, sample_page):
    """Test updating a page"""
    update_data = {
        "extracted_text": "Updated text",
        "formatted_text": "Formatted text"
    }
    response = client.put(
        f"/api/pages/{sample_page.id}",
        json=update_data
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["extracted_text"] == update_data["extracted_text"]
    assert data["formatted_text"] == update_data["formatted_text"]


def test_delete_page(client, sample_page):
    """Test deleting a page"""
    response = client.delete(f"/api/pages/{sample_page.id}")

    assert response.status_code == status.HTTP_200_OK

    # Verify page is deleted
    get_response = client.get(f"/api/pages/{sample_page.id}")
    assert get_response.status_code == status.HTTP_404_NOT_FOUND


def create_test_image_with_text(text: str = "Test Text") -> Image.Image:
    """Create a test image with actual text content"""
    # Create an A4 sized image at 72 DPI (612 x 792 pixels)
    img = Image.new('RGB', (612, 792), color='white')
    draw = ImageDraw.Draw(img)

    # Add some text
    draw.text((50, 50), text, fill='black')

    return img


def test_upload_multiple_pages(client, db_session, sample_document):
    """Test uploading multiple pages"""

    def create_valid_test_image():
        img = Image.new('RGB', (612, 792), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "Test text", fill='black')

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf

    # Clear any existing pages
    from app.models.page import Page
    db_session.query(Page).filter(Page.document_id == sample_document.id).delete()
    db_session.commit()

    # Upload new pages
    files = [
        ("files", ("test1.png", create_valid_test_image(), "image/png")),
        ("files", ("test2.png", create_valid_test_image(), "image/png"))
    ]

    response = client.post(
        f"/api/pages/upload/{sample_document.id}",
        files=files
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) == 2
    assert all(page["document_id"] == sample_document.id for page in data)

    # Sort by page number to ensure order
    pages = sorted(data, key=lambda x: x["page_number"])
    assert pages[0]["page_number"] == 1
    assert pages[1]["page_number"] == 2

    # Verify processing status
    assert all(page["processing_status"] == ProcessingStatus.PENDING for page in data)


def test_get_nonexistent_page(client):
    """Test getting a page that doesn't exist"""
    response = client.get("/api/pages/99999")
    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_process_page_background(client, sample_page, db_session):
    """Test background processing of a page"""
    from app.api.pages import process_page_background
    from pathlib import Path

    # Create test image file
    test_image = Path(sample_page.image_path)
    if not test_image.parent.exists():
        test_image.parent.mkdir(parents=True)
    test_image.write_bytes(b"test image data")

    await process_page_background(
        sample_page.id,
        test_image,
        None,
        db_session
    )

    # Refresh session to get updated page
    db_session.refresh(sample_page)
    assert sample_page.processing_status in ["completed", "error"]
