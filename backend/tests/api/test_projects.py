# backend/tests/api/test_projects.py
import pytest
from fastapi import status

def test_create_project(client):
    """Test project creation"""
    response = client.post(
        "/api/projects",
        json={"name": "New Project", "description": "Project Description"}
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["name"] == "New Project"
    assert data["description"] == "Project Description"
    assert "id" in data
    assert "created_at" in data

def test_get_project(client, sample_project):
    """Test getting a single project"""
    response = client.get(f"/api/projects/{sample_project.id}")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["name"] == sample_project.name
    assert data["description"] == sample_project.description
    assert "document_count" in data

def test_list_projects(client, sample_project):
    """Test listing all projects"""
    response = client.get("/api/projects")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) >= 1
    assert any(p["id"] == sample_project.id for p in data)

def test_update_project(client, sample_project):
    """Test updating a project"""
    update_data = {
        "name": "Updated Project",
        "description": "Updated Description"
    }
    response = client.put(
        f"/api/projects/{sample_project.id}",
        json=update_data
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["name"] == update_data["name"]
    assert data["description"] == update_data["description"]

def test_delete_project(client, sample_project):
    """Test deleting a project"""
    response = client.delete(f"/api/projects/{sample_project.id}")

    assert response.status_code == status.HTTP_200_OK

    # Verify project is deleted
    get_response = client.get(f"/api/projects/{sample_project.id}")
    assert get_response.status_code == status.HTTP_404_NOT_FOUND

def test_get_nonexistent_project(client):
    """Test getting a project that doesn't exist"""
    response = client.get("/api/projects/99999")
    assert response.status_code == status.HTTP_404_NOT_FOUND

def test_invalid_project_data(client):
    """Test creating a project with invalid data"""
    response = client.post(
        "/api/projects",
        json={"description": "Missing name field"}
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY