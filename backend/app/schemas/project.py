# backend/app/schemas/project.py - Let's check the schema
from typing import Optional
from .base import BaseSchema, TimestampMixin

class ProjectBase(BaseSchema):
    name: str
    description: Optional[str] = None

class ProjectCreate(ProjectBase):
    pass

class ProjectUpdate(ProjectBase):
    pass

class Project(ProjectBase, TimestampMixin):
    id: int

    class Config:
        from_attributes = True

class ProjectDetail(Project):
    document_count: int

    class Config:
        from_attributes = True