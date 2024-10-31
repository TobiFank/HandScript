# backend/app/api/__init__.py
from .projects import router as projects_router
from .documents import router as documents_router
from .pages import router as pages_router
from .writers import router as writers_router
from .training_samples import router as training_samples_router

__all__ = ["projects_router", "documents_router", "pages_router", "writers_router", "training_samples_router"]