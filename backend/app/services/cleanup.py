# backend/app/services/cleanup.py
import shutil

from sqlalchemy.orm import Session

from ..config import settings
from ..models import Project, Document, Page
from ..utils.logging import service_logger


class CleanupService:
    """Service to handle cascading deletion of artifacts"""

    @staticmethod
    async def delete_page_artifacts(page: Page) -> None:
        """Delete all files associated with a page"""
        try:
            # Delete main image file
            if page.image_path:
                image_path = settings.STORAGE_PATH / page.image_path
                if image_path.exists():
                    image_path.unlink()
                    service_logger.info(f"Deleted page image: {image_path}")

            # Delete line images directory if it exists
            lines_dir = settings.STORAGE_PATH / "lines" / str(page.id)
            if lines_dir.exists():
                shutil.rmtree(lines_dir)
                service_logger.info(f"Deleted line images directory: {lines_dir}")

        except Exception as e:
            service_logger.error(f"Error deleting page artifacts: {str(e)}", extra={
                "page_id": page.id,
                "image_path": page.image_path
            })
            raise

    @staticmethod
    async def delete_document_artifacts(document: Document, db: Session) -> None:
        """Delete all pages and their artifacts for a document"""
        try:
            # Get all pages for the document
            pages = db.query(Page).filter(Page.document_id == document.id).all()

            # Delete each page's artifacts
            for page in pages:
                await CleanupService.delete_page_artifacts(page)

            service_logger.info(f"Deleted all artifacts for document {document.id}")

        except Exception as e:
            service_logger.error(f"Error deleting document artifacts: {str(e)}", extra={
                "document_id": document.id
            })
            raise

    @staticmethod
    async def delete_project_artifacts(project: Project, db: Session) -> None:
        """Delete all documents and their artifacts for a project"""
        try:
            # Get all documents for the project
            documents = db.query(Document).filter(Document.project_id == project.id).all()

            # Delete each document's artifacts
            for document in documents:
                await CleanupService.delete_document_artifacts(document, db)

            service_logger.info(f"Deleted all artifacts for project {project.id}")

        except Exception as e:
            service_logger.error(f"Error deleting project artifacts: {str(e)}", extra={
                "project_id": project.id
            })
            raise

    @staticmethod
    async def cleanup_training_checkpoints(writer_id: int) -> None:
        """Delete all checkpoint files after training is complete, keeping only the final model."""
        try:
            # Get the writer's model directory
            writer_dir = settings.MODELS_PATH / str(writer_id)
            if not writer_dir.exists():
                return

            # List all items in the directory
            items_to_remove = []
            for item in writer_dir.glob('*'):
                # Keep the 'model' directory which contains the final weights
                if item.name == 'model':
                    continue

                # Remove checkpoint directories and files
                if item.name.startswith('checkpoint-') or \
                        item.name == 'optimizer.pt' or \
                        item.name == 'scheduler.pt' or \
                        item.name == 'trainer_state.json' or \
                        item.name == 'training_args.bin' or \
                        item.name.endswith('.safetensors'):
                    items_to_remove.append(item)

            # Delete identified items
            for item in items_to_remove:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
                service_logger.info(f"Deleted training artifact: {item}")

            service_logger.info(f"Cleaned up training checkpoints for writer {writer_id}")

        except Exception as e:
            service_logger.error(f"Error cleaning up training checkpoints: {str(e)}", extra={
                "writer_id": writer_id
            })
            raise


cleanup_service = CleanupService()
