# backend/app/__init__.py
from .config import settings
from .database import Base, engine, get_db
from . import models
from . import schemas
from . import api
from . import services
from . import ml

__version__ = "0.1.0"