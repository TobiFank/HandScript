# backend/app/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .database import engine
from . import models
from .api import projects, documents, pages, writers, training_samples
from .utils.logging import api_logger
from fastapi.staticfiles import StaticFiles
from .config import settings

# Create all tables on startup
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="HandScript API")

# Configure CORS with explicit headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your actual frontend URL
    allow_credentials=False,  # Changed from True since we're not using credentials
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/storage", StaticFiles(directory=str(settings.STORAGE_PATH)), name="storage")

# Include routers
app.include_router(projects.router)
app.include_router(documents.router)
app.include_router(pages.router)
app.include_router(writers.router)
app.include_router(training_samples.router)

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "Content-Type, Content-Length"
    return response

@app.get("/")
async def root():
    return {"message": "HandScript API is running"}