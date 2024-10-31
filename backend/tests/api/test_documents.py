# backend/tests/api/test_documents.py
import pytest
from fastapi import status

def test_create_document(client, sample_project):
    """Test document creation"""
    response = client.post(
        "/api/documents",
        json={
            "name": "New Document",
            "description": "Document Description",
            "project_id": sample_project.id
        }
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["name"] == "New Document"
    assert data["project_id"] == sample_project.id

def test_get_document(client, sample_document):
    """Test getting a single document"""
    response = client.get(f"/api/documents/{sample_document.id}")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["name"] == sample_document.name
    assert data["project_id"] == sample_document.project_id
    assert "page_count" in data

def test_list_project_documents(client, sample_project, sample_document):
    """Test listing documents in a project"""
    response = client.get(f"/api/documents/project/{sample_project.id}")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) >= 1
    assert any(d["id"] == sample_document.id for d in data)

def test_update_document(client, sample_document):
    """Test updating a document"""
    update_data = {
        "name": "Updated Document",
        "description": "Updated Description"
    }
    response = client.put(
        f"/api/documents/{sample_document.id}",
        json=update_data
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["name"] == update_data["name"]
    assert data["description"] == update_data["description"]

def test_delete_document(client, sample_document):
    """Test deleting a document"""
    response = client.delete(f"/api/documents/{sample_document.id}")

    assert response.status_code == status.HTTP_200_OK

    # Verify document is deleted
    get_response = client.get(f"/api/documents/{sample_document.id}")
    assert get_response.status_code == status.HTTP_404_NOT_FOUND

def test_get_nonexistent_document(client):
    """Test getting a document that doesn't exist"""
    response = client.get("/api/documents/99999")
    assert response.status_code == status.HTTP_404_NOT_FOUND

@pytest.mark.skip("Export functionality needs to be mocked")
def test_export_document(client, sample_document):
    """Test document export"""
    response = client.get(f"/api/documents/{sample_document.id}/export?format=pdf")

    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-type"] == "application/pdf"