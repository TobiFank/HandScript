# backend/app/schemas/__init__.py
from .project import Project, ProjectCreate, ProjectUpdate, ProjectDetail
from .document import Document, DocumentCreate, DocumentUpdate, DocumentDetail
from .page import Page, PageCreate, PageUpdate
from .page import Writer, WriterCreate, WriterUpdate

__all__ = [
    "Project", "ProjectCreate", "ProjectUpdate", "ProjectDetail",
    "Document", "DocumentCreate", "DocumentUpdate", "DocumentDetail",
    "Page", "PageCreate", "PageUpdate",
    "Writer", "WriterCreate", "WriterUpdate"
]