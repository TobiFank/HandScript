# backend/app/api/training_samples.py
import json
import shutil
from typing import List
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import Page, Writer
from ..models.training_sample import TrainingSample
from ..schemas.training_sample import (
    TrainingSample as TrainingSampleSchema,
    TrainingSampleUpdate,
    TrainingSampleWithLines, ExportResponse
)
from ..services.ocr import ocr_service
from ..utils.files import save_upload_file, get_relative_path, delete_file, load_file_as_uploadfile
from ..utils.logging import api_logger

router = APIRouter(prefix="/api/training-samples", tags=["training-samples"])


# backend/app/api/training_samples.py
@router.get("/writer/{writer_id}", response_model=List[TrainingSampleWithLines])
async def list_training_samples(writer_id: int, db: Session = Depends(get_db)):
    """List all training samples for a writer with their line segments"""
    api_logger.info(f"Fetching training samples for writer {writer_id}")

    # Get samples with their line segments in a single query
    samples = (
        db.query(TrainingSample)
        .filter(TrainingSample.writer_id == writer_id)
        .all()
    )

    enhanced_samples = []
    for sample in samples:
        try:
            # Create sample data with stored line segments
            enhanced_sample = {
                'id': sample.id,
                'writer_id': sample.writer_id,
                'image_path': sample.image_path,
                'text': sample.text,
                'created_at': sample.created_at,
                'needs_review': sample.needs_review,
                'line_count': len(sample.line_segments),
                'lines': [{
                    'bbox': segment.bbox,
                    'text': segment.text,
                    'confidence': segment.confidence
                } for segment in sorted(sample.line_segments, key=lambda x: x.order)]
            }

            enhanced_samples.append(TrainingSampleWithLines(**enhanced_sample))

        except Exception as e:
            api_logger.error(f"Error processing sample {sample.id}: {str(e)}")
            # Include sample without line information as fallback
            enhanced_samples.append(TrainingSampleWithLines.from_orm(sample))

    api_logger.info(f"Successfully retrieved {len(enhanced_samples)} training samples")
    return enhanced_samples


@router.post("/writer/{writer_id}")
async def add_training_sample(
        writer_id: int,
        text: str = Form(...),
        image: UploadFile = File(...),
        needs_review: bool = Form(True),  # Add this parameter
        db: Session = Depends(get_db)
):
    """Add a training sample"""
    try:
        api_logger.info(f"Adding training sample for writer {writer_id}")

        # Save the image
        save_dir = settings.STORAGE_PATH / "training_samples" / str(writer_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        saved_path = await save_upload_file(image, save_dir)
        relative_path = get_relative_path(saved_path, settings.STORAGE_PATH)

        api_logger.debug(f"Saved image to {relative_path}")

        # Create sample
        sample = TrainingSample(
            writer_id=writer_id,
            image_path=relative_path,
            text=text,
            needs_review=needs_review  # Set the flag
        )

        db.add(sample)
        db.commit()
        db.refresh(sample)

        api_logger.info(f"Created training sample {sample.id}")
        return sample

    except Exception as e:
        api_logger.error(f"Failed to create training sample: {str(e)}")
        if 'sample' in locals():
            db.delete(sample)
            db.commit()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/writer/{writer_id}/process")
async def process_training_sample(
        writer_id: int,
        image: UploadFile = File(...),
        needs_review: bool = Form(True),  # Add this parameter
        db: Session = Depends(get_db)
):
    """Process multiline image into individual line training samples"""
    try:
        api_logger.info(f"Processing training sample", extra={
            "writer_id": writer_id,
            "filename": image.filename
        })

        # Save the uploaded image
        save_dir = settings.STORAGE_PATH / "training_samples" / str(writer_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        saved_path = await save_upload_file(image, save_dir)

        # Process image using existing OCR functionality
        result = await ocr_service.process_image(saved_path, writer_id, db)

        # Create training samples for each line
        created_samples = []
        lines_dir = save_dir / f"lines_{uuid4().hex[:8]}"
        lines_dir.mkdir(exist_ok=True)

        for idx, line in enumerate(result['lines']):
            try:
                if line.get('image'):
                    line_path = lines_dir / f"line_{idx}.png"
                    line['image'].save(line_path)
                    relative_path = get_relative_path(line_path, settings.STORAGE_PATH)

                    sample = TrainingSample(
                        writer_id=writer_id,
                        image_path=relative_path,
                        text=line['text'],
                        needs_review=needs_review  # Set the flag
                    )
                    db.add(sample)
                    db.flush()

                    created_samples.append({
                        "id": sample.id,
                        "image_path": relative_path,
                        "text": line['text'],
                        "needs_review": needs_review,
                        "bbox": line.get('bbox'),
                        "confidence": line.get('confidence', 0.0)
                    })

            except Exception as e:
                api_logger.error(f"Error processing line {idx}", extra={
                    "error": str(e)
                }, exc_info=True)
                continue

        db.commit()
        return created_samples

    except Exception as e:
        api_logger.error("Failed to process training samples", extra={
            "error": str(e)
        }, exc_info=True)
        if 'db' in locals():
            db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{sample_id}")
async def delete_training_sample(sample_id: int, db: Session = Depends(get_db)):
    """Delete a training sample and its associated files"""
    sample = db.query(TrainingSample).filter(TrainingSample.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    try:
        # Delete main image file
        main_file_path = settings.STORAGE_PATH / sample.image_path
        if main_file_path.exists():
            await delete_file(main_file_path)

        # Delete line segment files if they exist
        line_segments_dir = main_file_path.parent / f"lines_{sample_id}"
        if line_segments_dir.exists():
            for line_file in line_segments_dir.glob("*.png"):
                await delete_file(line_file)
            line_segments_dir.rmdir()

        db.delete(sample)
        db.commit()
        return {"success": True}
    except Exception as e:
        api_logger.error(f"Error deleting training sample: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{sample_id}", response_model=TrainingSampleSchema)
async def update_training_sample(
        sample_id: int,
        update_data: TrainingSampleUpdate,
        db: Session = Depends(get_db)
):
    """Update a training sample's text and mark it as reviewed"""
    api_logger.info(f"Updating training sample {sample_id}")

    sample = db.query(TrainingSample).filter(TrainingSample.id == sample_id).first()
    if not sample:
        api_logger.warning(f"Training sample {sample_id} not found")
        raise HTTPException(status_code=404, detail="Training sample not found")

    try:
        # Update the text
        sample.text = update_data.text
        # Mark as reviewed
        sample.needs_review = False

        db.commit()
        db.refresh(sample)
        api_logger.info(f"Successfully updated training sample {sample_id}")
        return sample
    except Exception as e:
        api_logger.error(f"Error updating training sample {sample_id}: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update training sample")


@router.post("/writer/{writer_id}/from-page/{page_id}")
async def convert_page_to_sample(
        writer_id: int,
        page_id: int,
        db: Session = Depends(get_db)
):
    """Convert a page to a training sample - no review needed as text is already verified"""
    page = db.query(Page).filter(Page.id == page_id).first()
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    page_image_path = settings.STORAGE_PATH / page.image_path
    if not page_image_path.exists():
        raise HTTPException(status_code=404, detail="Page image not found")

    # Create upload file from page image
    upload_file = await load_file_as_uploadfile(page_image_path)

    # Use the formatted_text if available, otherwise use extracted_text
    text = page.formatted_text or page.extracted_text or ""

    # Create training sample using add_training_sample but set needs_review=False
    # since this comes from an already verified page
    return await add_training_sample(
        writer_id=writer_id,
        text=text,
        image=upload_file,
        needs_review=False,  # Don't need review for pages converted to samples
        db=db
    )

@router.get("/export", response_model=ExportResponse)
async def export_training_samples(db: Session = Depends(get_db)):
    """Export all reviewed training samples"""
    try:
        api_logger.info("Starting training samples export")

        # Create export directory
        export_base_dir = settings.STORAGE_PATH / "exported_training_samples"
        if export_base_dir.exists():
            shutil.rmtree(export_base_dir)
        export_base_dir.mkdir(parents=True)

        api_logger.info(f"Created export directory: {export_base_dir}")

        # Get all reviewed training samples with their writers
        samples = (
            db.query(TrainingSample, Writer)
            .join(Writer)
            .filter(TrainingSample.needs_review == False)
            .all()
        )

        api_logger.info(f"Found {len(samples)} samples to export")

        metadata = []
        exported_count = 0

        for sample, writer in samples:
            try:
                # Create writer-specific directory
                writer_dir = export_base_dir / f"writer_{writer.id}"
                writer_dir.mkdir(exist_ok=True)

                # Copy image file
                source_path = settings.STORAGE_PATH / sample.image_path
                if not source_path.exists():
                    api_logger.warning(f"Source file not found: {source_path}")
                    continue

                # Create new filename based on sample ID
                new_filename = f"sample_{sample.id}{source_path.suffix}"
                dest_path = writer_dir / new_filename

                shutil.copy2(source_path, dest_path)
                exported_count += 1

                # Add metadata
                metadata.append({
                    "image_path": str(dest_path.relative_to(export_base_dir)),
                    "text": sample.text,
                    "writer_id": writer.id,
                    "writer_name": writer.name,
                    "language": writer.language or "english",
                    "sample_id": sample.id
                })

            except Exception as e:
                api_logger.error(f"Error processing sample {sample.id}: {str(e)}")
                continue

        api_logger.info(f"Successfully exported {exported_count} samples")

        # Write metadata file
        metadata_path = export_base_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Create zip file
        zip_path = str(export_base_dir) + ".zip"
        shutil.make_archive(
            str(export_base_dir),
            'zip',
            export_base_dir
        )

        # Clean up the unzipped directory
        shutil.rmtree(export_base_dir)

        return ExportResponse(
            success=True,
            sample_count=exported_count,
            export_path=zip_path
        )

    except Exception as e:
        api_logger.error(f"Export failed: {str(e)}")
        return ExportResponse(
            success=False,
            sample_count=0,
            export_path=""
        )
