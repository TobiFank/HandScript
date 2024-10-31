# backend/app/api/projects.py
import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from starlette.responses import Response
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from ..database import get_db
from ..models.project import Project
from ..models.document import Document
from ..schemas.project import ProjectCreate, ProjectUpdate, Project as ProjectSchema, ProjectDetail
from ..services.cleanup import cleanup_service
from ..utils.logging import api_logger

router = APIRouter(prefix="/api/projects", tags=["projects"])

@router.get("")
async def list_projects(db: Session = Depends(get_db)):
    """List all projects"""
    api_logger.info("Starting projects list operation", extra={
        "endpoint": "/api/projects",
        "method": "GET"
    })

    try:
        # Query projects
        projects = db.query(Project).all()
        api_logger.info(f"Found {len(projects)} projects")

        # Create response data with a default empty list
        result = []
        for project in projects:
            api_logger.debug("Processing project", extra={
                "project_id": project.id,
                "project_name": project.name
            })
            result.append({
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "created_at": project.created_at.isoformat() if project.created_at else None,
                "document_count": 0
            })

        # Use JSONResponse with explicit headers
        response = JSONResponse(
            content=result,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Content-Type, Content-Length"
            }
        )

        api_logger.info("Sending response", extra={"project_count": len(result)})
        return response
    except Exception as e:
        api_logger.error("Failed to list projects", extra={"error": str(e)})
        raise

@router.get("/test-endpoint")
async def test_projects():
    """Test endpoint that always returns a valid JSON response"""
    api_logger.info("Accessing test endpoint")
    return JSONResponse(
        content=[{
            "id": 0,
            "name": "Test Project",
            "description": "This is a test project",
            "created_at": "2024-10-23T00:00:00",
            "document_count": 0
        }],
        headers={
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "Content-Type, Content-Length"
        }
    )

@router.get("/{project_id}", response_model=ProjectDetail)
async def get_project(project_id: int, db: Session = Depends(get_db)):
    api_logger.info("Fetching project", extra={"project_id": project_id})

    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            api_logger.warning("Project not found", extra={"project_id": project_id})
            raise HTTPException(status_code=404, detail="Project not found")

        doc_count = db.query(func.count(Document.id)) \
            .filter(Document.project_id == project_id) \
            .scalar()

        project_dict = {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "created_at": project.created_at,
            "document_count": doc_count or 0
        }

        api_logger.info("Project retrieved successfully", extra={
            "project_id": project_id,
            "document_count": doc_count
        })
        return project_dict
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Failed to get project", extra={
            "project_id": project_id,
            "error": str(e)
        })
        raise

@router.post("", response_model=ProjectSchema)
async def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    api_logger.info("Creating new project", extra={"project_name": project.name})

    try:
        db_project = Project(**project.model_dump())
        db.add(db_project)
        db.commit()
        db.refresh(db_project)

        api_logger.info("Project created successfully", extra={
            "project_id": db_project.id,
            "project_name": db_project.name
        })
        return db_project
    except Exception as e:
        api_logger.error("Failed to create project", extra={
            "project_name": project.name,
            "error": str(e)
        })
        db.rollback()
        raise

@router.put("/{project_id}", response_model=ProjectSchema)
async def update_project(project_id: int, project: ProjectUpdate, db: Session = Depends(get_db)):
    api_logger.info("Updating project", extra={"project_id": project_id})

    try:
        db_project = db.query(Project).filter(Project.id == project_id).first()
        if not db_project:
            api_logger.warning("Project not found for update", extra={"project_id": project_id})
            raise HTTPException(status_code=404, detail="Project not found")

        for field, value in project.model_dump(exclude_unset=True).items():
            setattr(db_project, field, value)

        db.commit()
        db.refresh(db_project)

        api_logger.info("Project updated successfully", extra={"project_id": project_id})
        return db_project
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Failed to update project", extra={
            "project_id": project_id,
            "error": str(e)
        })
        db.rollback()
        raise

@router.delete("/{project_id}")
async def delete_project(project_id: int, db: Session = Depends(get_db)):
    api_logger.info("Deleting project", extra={"project_id": project_id})

    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            api_logger.warning("Project not found for deletion")
            raise HTTPException(status_code=404, detail="Project not found")

        # Delete project artifacts (includes documents and pages)
        await cleanup_service.delete_project_artifacts(project, db)

        # Delete the project (will cascade delete documents and pages in database)
        db.delete(project)
        db.commit()

        api_logger.info(f"Successfully deleted project {project_id}")
        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        api_logger.error(f"Failed to delete project: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))