# backend/app/api/writers.py
import json
import shutil
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import Page, TrainingSample
from ..models.writer import Writer
from ..schemas.page import Writer as WriterSchema, WriterCreate, WriterUpdate
from ..schemas.training_sample import ExportResponse
from ..services.training import training_service
from ..utils.files import save_upload_file, get_relative_path
from ..utils.logging import api_logger

router = APIRouter(prefix="/api/writers", tags=["writers"])


@router.get("", response_model=List[WriterSchema])
async def list_writers(db: Session = Depends(get_db)):
    """List all writers in the system"""
    try:
        api_logger.info("Fetching all writers")
        writers = db.query(Writer).all()
        api_logger.info(f"Successfully retrieved {len(writers)} writers")
        return writers
    except Exception as e:
        api_logger.error("Failed to fetch writers", extra={
            "error": str(e)
        }, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch writers: {str(e)}"
        )


@router.post("", response_model=WriterSchema)
async def create_writer(writer: WriterCreate, db: Session = Depends(get_db)):
    """Create a new writer"""
    api_logger.info("Creating new writer", extra={
        "writer_name": writer.name
    })
    try:
        db_writer = Writer(**writer.model_dump())
        db.add(db_writer)
        db.commit()
        db.refresh(db_writer)
        api_logger.info("Successfully created writer", extra={
            "writer_id": db_writer.id,
            "writer_name": db_writer.name
        })
        return db_writer
    except Exception as e:
        api_logger.error("Failed to create writer", extra={
            "writer_name": writer.name,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail="Failed to create writer")


@router.get("/{writer_id}", response_model=WriterSchema)
async def get_writer(writer_id: int, db: Session = Depends(get_db)):
    """Get a specific writer by ID"""
    api_logger.info("Fetching writer", extra={"writer_id": writer_id})
    writer = db.query(Writer).filter(Writer.id == writer_id).first()
    if not writer:
        api_logger.warning("Writer not found", extra={"writer_id": writer_id})
        raise HTTPException(status_code=404, detail="Writer not found")
    api_logger.info("Successfully retrieved writer", extra={
        "writer_id": writer.id,
        "writer_name": writer.name,
        "status": writer.status
    })
    return writer


@router.put("/{writer_id}", response_model=WriterSchema)
async def update_writer(writer_id: int, writer: WriterUpdate, db: Session = Depends(get_db)):
    """Update a specific writer"""
    api_logger.info("Updating writer", extra={
        "writer_id": writer_id,
        "update_fields": writer.model_dump(exclude_unset=True)
    })

    try:
        db_writer = db.query(Writer).filter(Writer.id == writer_id).first()
        if not db_writer:
            api_logger.warning("Writer not found for update", extra={
                "writer_id": writer_id
            })
            raise HTTPException(status_code=404, detail="Writer not found")

        # Update fields
        for field, value in writer.model_dump(exclude_unset=True).items():
            setattr(db_writer, field, value)

        db.commit()
        db.refresh(db_writer)

        api_logger.info("Writer updated successfully", extra={
            "writer_id": writer_id,
            "writer_name": db_writer.name
        })
        return db_writer

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Failed to update writer", extra={
            "writer_id": writer_id,
            "error": str(e)
        })
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update writer: {str(e)}"
        )


@router.post("/{writer_id}/upload_model")
async def upload_writer_model(
        writer_id: int,
        file: UploadFile,
        db: Session = Depends(get_db)
):
    """Upload a model file for a specific writer"""
    api_logger.info("Uploading model for writer", extra={
        "writer_id": writer_id,
        "filename": file.filename
    })

    writer = db.query(Writer).filter(Writer.id == writer_id).first()
    if not writer:
        api_logger.warning("Writer not found during model upload", extra={
            "writer_id": writer_id
        })
        raise HTTPException(status_code=404, detail="Writer not found")

    try:
        # Save model file
        saved_path = await save_upload_file(file, settings.MODELS_PATH)
        relative_path = get_relative_path(saved_path, settings.STORAGE_PATH)

        api_logger.info("Model file saved successfully", extra={
            "writer_id": writer_id,
            "file_path": str(saved_path)
        })

        # Update writer
        writer.model_path = relative_path
        writer.status = "ready"
        db.commit()

        api_logger.info("Writer model updated successfully", extra={
            "writer_id": writer_id,
            "model_path": relative_path,
            "status": "ready"
        })

        return {"success": True, "model_path": relative_path}

    except Exception as e:
        api_logger.error("Failed to upload writer model", extra={
            "writer_id": writer_id,
            "filename": file.filename,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail="Failed to upload model")


@router.delete("/{writer_id}")
async def delete_writer(writer_id: int, db: Session = Depends(get_db)):
    """Delete a specific writer and all associated artifacts"""
    api_logger.info("Attempting to delete writer", extra={"writer_id": writer_id})

    # Start transaction
    try:
        # Get writer with associated training samples
        writer = db.query(Writer).filter(Writer.id == writer_id).first()
        if not writer:
            api_logger.warning("Writer not found for deletion", extra={
                "writer_id": writer_id
            })
            raise HTTPException(status_code=404, detail="Writer not found")

        # 1. Clean up model files
        writer_model_dir = settings.MODELS_PATH / str(writer_id)
        if writer_model_dir.exists():
            shutil.rmtree(writer_model_dir)
            api_logger.info("Deleted model files", extra={
                "writer_id": writer_id,
                "model_dir": str(writer_model_dir)
            })

        # 2. Clean up training samples files
        training_samples_dir = settings.STORAGE_PATH / "training_samples" / str(writer_id)
        if training_samples_dir.exists():
            shutil.rmtree(training_samples_dir)
            api_logger.info("Deleted training samples", extra={
                "writer_id": writer_id,
                "samples_dir": str(training_samples_dir)
            })

        # 3. Update associated pages (set writer_id to NULL)
        pages = db.query(Page).filter(Page.writer_id == writer_id).all()
        for page in pages:
            page.writer_id = None
        api_logger.info("Updated associated pages", extra={
            "writer_id": writer_id,
            "pages_updated": len(pages)
        })

        # 4. Delete writer from database
        # This will automatically cascade delete training samples due to the relationship configuration
        db.delete(writer)
        db.commit()

        api_logger.info("Writer deleted successfully", extra={
            "writer_id": writer_id,
            "writer_name": writer.name
        })
        return {"success": True}

    except Exception as e:
        api_logger.error("Failed to delete writer", extra={
            "writer_id": writer_id,
            "error": str(e)
        }, exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete writer: {str(e)}"
        )


@router.post("/{writer_id}/train")
async def train_writer(
        writer_id: int,
        db: Session = Depends(get_db)
):
    """Start training using stored training samples"""
    api_logger.info("Starting writer training", extra={
        "writer_id": writer_id
    })

    # Verify writer exists
    writer = db.query(Writer).filter(Writer.id == writer_id).first()
    if not writer:
        raise HTTPException(status_code=404, detail="Writer not found")

    # Get training samples from database
    training_samples = db.query(TrainingSample).filter(
        TrainingSample.writer_id == writer_id
    ).all()

    if not training_samples:
        raise HTTPException(
            status_code=400,
            detail="No training samples found for this writer"
        )

    # Convert samples to the format expected by training service
    sample_pages = [
        (
            settings.STORAGE_PATH / sample.image_path,
            sample.text
        )
        for sample in training_samples
    ]

    try:
        # Start training process
        result = await training_service.train_writer_model(
            db,
            writer_id,
            sample_pages
        )

        api_logger.info("Training initiated successfully", extra={
            "writer_id": writer_id,
            "sample_count": len(training_samples)
        })

        return {
            "success": True,
            "message": "Training started successfully",
            "sample_count": len(training_samples)
        }

    except Exception as e:
        api_logger.error("Failed to start training", extra={
            "writer_id": writer_id,
            "error": str(e)
        })
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/{writer_id}/training_status")
async def get_training_status(writer_id: int, db: Session = Depends(get_db)):
    """Get the current training status for a writer"""
    api_logger.info("Fetching training status", extra={"writer_id": writer_id})

    writer = db.query(Writer).filter(Writer.id == writer_id).first()
    if not writer:
        api_logger.warning("Writer not found for status check", extra={
            "writer_id": writer_id
        })
        raise HTTPException(status_code=404, detail="Writer not found")

    status_data = {
        "status": writer.status,
        "last_trained": writer.last_trained,
        "model_path": writer.model_path
    }

    api_logger.info("Retrieved training status", extra={
        "writer_id": writer_id,
        "status": status_data
    })

    return status_data


@router.get("/{writer_id}/stats")
async def get_writer_stats(writer_id: int, db: Session = Depends(get_db)):
    """Get writer statistics and performance metrics"""
    writer = db.query(Writer).filter(Writer.id == writer_id).first()
    if not writer:
        raise HTTPException(status_code=404, detail="Writer not found")

    if not writer.evaluation_metrics:
        raise HTTPException(status_code=404, detail="No evaluation metrics available")

    # Format metrics for frontend
    metrics = writer.evaluation_metrics
    return {
        "accuracy_trend": [{
            "date": metrics["evaluation_date"],
            "accuracy": (1 - metrics["character_error_rate"]) * 100
        }],
        "avg_processing_time": round(metrics["avg_processing_time"] * 1000, 2),  # Convert to ms
        "char_accuracy": round((1 - metrics["character_error_rate"]) * 100, 2),
        "word_accuracy": round((1 - metrics["word_error_rate"]) * 100, 2),
        "total_pages": writer.pages_processed,
        "error_types": []  # Simplified error types for now
    }
