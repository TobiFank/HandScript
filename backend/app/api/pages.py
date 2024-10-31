# backend/app/api/pages.py
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, BackgroundTasks, Body
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.sql import func

from ..config import settings
from ..database import get_db
from ..models.page import Page, ProcessingStatus
from ..models.writer import Writer
from ..schemas.page import PageUpdate, Page as PageSchema
from ..services.cleanup import cleanup_service
from ..services.ocr import ocr_service
from ..utils.files import save_upload_file, get_relative_path
from ..utils.logging import api_logger, service_logger

router = APIRouter(prefix="/api/pages", tags=["pages"])


async def process_page_background(
        page_id: int,
        image_path: Path,
        writer_id: int,
        db: Session
):
    """Background task for processing page OCR with line segmentation"""
    service_logger.info(
        f"Starting background processing for page {page_id}",
        extra={
            "page_id": page_id,
            "image_path": str(image_path),
            "writer_id": writer_id
        }
    )

    try:
        # Get page from database
        page = db.query(Page).filter(Page.id == page_id).first()
        if not page:
            service_logger.error(
                f"Page {page_id} not found for processing",
                extra={"page_id": page_id}
            )
            return

        # Update status to processing
        page.processing_status = ProcessingStatus.PROCESSING
        db.commit()

        # Process OCR with line segmentation
        result = await ocr_service.process_image(
            image_path,
            writer_id,
            db
        )

        # Create lines directory if it doesn't exist
        lines_dir = settings.STORAGE_PATH / "lines" / str(page_id)
        lines_dir.mkdir(parents=True, exist_ok=True)

        line_segments = []
        for idx, line_info in enumerate(result['lines']):
            try:
                # Save line image
                line_image_path = lines_dir / f"line_{idx}.png"
                line_info['image'].save(line_image_path)

                relative_path = get_relative_path(line_image_path, settings.STORAGE_PATH)

                line_segments.append({
                    'image_path': relative_path,
                    'text': line_info['text'],
                    'bbox': line_info['bbox']
                })
            except Exception as e:
                service_logger.error(f"Error saving line {idx}: {str(e)}")
                continue

        # Update page with extracted text and line information
        page.extracted_text = result['full_text']
        page.formatted_text = result['full_text']
        page.lines = line_segments
        page.processing_status = ProcessingStatus.COMPLETED

        # Update writer stats if applicable
        if page.writer_id:
            writer = db.query(Writer).filter(Writer.id == page.writer_id).first()
            if writer:
                writer.pages_processed += 1

        db.commit()

    except Exception as e:
        service_logger.error(f"Error processing page {page_id}: {str(e)}", exc_info=True)
        page.processing_status = ProcessingStatus.ERROR
        db.commit()

@router.post("/upload/{document_id}", response_model=List[PageSchema])
async def upload_pages(
        document_id: int,
        files: List[UploadFile],
        writer_id: int | None = None,
        background_tasks: BackgroundTasks = None,
        db: Session = Depends(get_db)
):
    api_logger.info(
        f"Starting upload of {len(files)} pages to document {document_id}",
        extra={
            "document_id": document_id,
            "writer_id": writer_id,
            "file_count": len(files),
            "file_names": [f.filename for f in files]
        }
    )

    created_pages = []

    # Get the current highest page number for this document once before the loop
    current_max_page = db.query(func.max(Page.page_number)) \
                           .filter(Page.document_id == document_id) \
                           .scalar() or 0

    for idx, file in enumerate(files):
        api_logger.debug(
            f"Processing file {idx + 1}/{len(files)}",
            extra={
                "file_name": file.filename,
                "file_size": file.size,
                "content_type": file.content_type
            }
        )

        try:
            # Save file and get its unique path
            saved_path = await save_upload_file(file, settings.IMAGES_PATH)
            relative_path = get_relative_path(saved_path, settings.STORAGE_PATH)

            api_logger.debug(
                f"File saved successfully",
                extra={
                    "original_name": file.filename,
                    "saved_path": str(saved_path),
                    "relative_path": relative_path
                }
            )

            # Create page with correct sequential page number
            page = Page(
                document_id=document_id,
                writer_id=writer_id,
                image_path=relative_path,
                page_number=current_max_page + idx + 1
            )

            db.add(page)
            db.commit()
            db.refresh(page)

            page_with_writer = db.query(Page) \
                .options(joinedload(Page.writer)) \
                .filter(Page.id == page.id) \
                .first()
            created_pages.append(page_with_writer)

            api_logger.info(
                f"Page {page.id} created successfully",
                extra={
                    "page_id": page.id,
                    "document_id": document_id,
                    "page_number": page.page_number
                }
            )

            # Schedule background processing
            background_tasks.add_task(
                process_page_background,
                page.id,
                saved_path,
                writer_id,
                db
            )

        except Exception as e:
            api_logger.error(
                f"Error processing upload for file {file.filename}",
                extra={
                    "file_name": file.filename,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file {file.filename}"
            )

    api_logger.info(
        f"Successfully created {len(created_pages)} pages",
        extra={
            "document_id": document_id,
            "created_page_ids": [p.id for p in created_pages]
        }
    )
    return created_pages


@router.get("/{page_id}", response_model=PageSchema)
async def get_page(page_id: int, db: Session = Depends(get_db)):
    api_logger.debug(f"Fetching page {page_id}", extra={"page_id": page_id})

    page = db.query(Page).filter(Page.id == page_id).first()
    if not page:
        api_logger.warning(
            f"Page {page_id} not found",
            extra={"page_id": page_id}
        )
        raise HTTPException(status_code=404, detail="Page not found")

    api_logger.debug(
        f"Successfully retrieved page {page_id}",
        extra={
            "page_id": page_id,
            "document_id": page.document_id,
            "processing_status": page.processing_status
        }
    )
    return page


@router.put("/{page_id}", response_model=PageSchema)
async def update_page(page_id: int, page_update: PageUpdate, db: Session = Depends(get_db)):
    api_logger.info(
        f"Updating page {page_id}",
        extra={
            "page_id": page_id,
            "update_fields": page_update.model_dump(exclude_unset=True)
        }
    )

    page = db.query(Page).filter(Page.id == page_id).first()
    if not page:
        api_logger.warning(
            f"Page {page_id} not found for update",
            extra={"page_id": page_id}
        )
        raise HTTPException(status_code=404, detail="Page not found")

    # Update fields
    for field, value in page_update.model_dump(exclude_unset=True).items():
        api_logger.debug(
            f"Updating field {field} for page {page_id}",
            extra={
                "page_id": page_id,
                "field": field,
                "old_value": getattr(page, field),
                "new_value": value
            }
        )
        setattr(page, field, value)

    db.commit()
    db.refresh(page)

    api_logger.info(
        f"Successfully updated page {page_id}",
        extra={
            "page_id": page_id,
            "updated_fields": page_update.model_dump(exclude_unset=True)
        }
    )
    return page


@router.delete("/{page_id}")
async def delete_page(page_id: int, db: Session = Depends(get_db)):
    api_logger.info(f"Deleting page {page_id}", extra={"page_id": page_id})

    page = db.query(Page).filter(Page.id == page_id).first()
    if not page:
        api_logger.warning(f"Page {page_id} not found for deletion")
        raise HTTPException(status_code=404, detail="Page not found")

    document_id = page.document_id
    deleted_page_number = page.page_number

    try:
        # Delete page artifacts
        await cleanup_service.delete_page_artifacts(page)

        # Delete the page from database
        db.delete(page)

        # Update page numbers for remaining pages
        remaining_pages = (
            db.query(Page)
            .filter(
                Page.document_id == document_id,
                Page.page_number > deleted_page_number
            )
            .order_by(Page.page_number)
            .all()
        )

        for p in remaining_pages:
            p.page_number -= 1

        db.commit()

        api_logger.info(f"Successfully deleted page {page_id}")
        return {"success": True}

    except Exception as e:
        db.rollback()
        api_logger.error(f"Failed to delete page: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{page_id}/reorder")
async def reorder_page(
        page_id: int,
        new_position: int = Body(..., embed=True),
        db: Session = Depends(get_db)
):
    """Update the order of pages within a document"""
    page = db.query(Page).filter(Page.id == page_id).first()
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    old_position = page.page_number
    document_id = page.document_id

    if old_position == new_position:
        return page

    # Get all pages in the document
    document_pages = (
        db.query(Page)
        .filter(Page.document_id == document_id)
        .order_by(Page.page_number)
        .all()
    )

    # Validate new position
    if new_position < 1 or new_position > len(document_pages):
        raise HTTPException(status_code=400, detail="Invalid page position")

    # Update page numbers
    if new_position > old_position:
        # Moving page forward
        for p in document_pages:
            if old_position < p.page_number <= new_position:
                p.page_number -= 1
    else:
        # Moving page backward
        for p in document_pages:
            if new_position <= p.page_number < old_position:
                p.page_number += 1

    page.page_number = new_position
    db.commit()
    db.refresh(page)

    return page

@router.put("/{page_id}/writer")
async def assign_writer(
        page_id: int,
        writer_id: int = Body(..., embed=True),
        db: Session = Depends(get_db)
):
    page = db.query(Page).filter(Page.id == page_id).first()
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    page.writer_id = writer_id
    db.commit()
    db.refresh(page)

    return page
