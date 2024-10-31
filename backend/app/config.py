# backend/app/config.py
import os
from typing import Literal
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "sqlite:///./handscript.db"  # Default if not in .env

    # Storage Paths
    STORAGE_PATH: Path = Path("storage")
    IMAGES_PATH: Path | None = None  # Will be set based on STORAGE_PATH
    MODELS_PATH: Path | None = None  # Will be set based on STORAGE_PATH
    EXPORTS_PATH: Path | None = None  # Will be set based on STORAGE_PATH

    # ML Settings
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL_ENGLISH", "microsoft/trocr-large-handwritten")
    DEFAULT_MODEL_GERMAN: str = os.getenv("DEFAULT_MODEL_GERMAN", "fhswf/TrOCR_german_handwritten")
    DEVICE: Literal["cuda", "cpu"] = "cuda"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def model_post_init(self, __context) -> None:
        """Post initialization hook to set derived paths"""
        # Convert STORAGE_PATH to Path if it's a string
        if isinstance(self.STORAGE_PATH, str):
            self.STORAGE_PATH = Path(self.STORAGE_PATH)

        # Set derived paths if not explicitly provided
        self.IMAGES_PATH = Path(self.IMAGES_PATH) if self.IMAGES_PATH else self.STORAGE_PATH / "images"
        self.MODELS_PATH = Path(self.MODELS_PATH) if self.MODELS_PATH else self.STORAGE_PATH / "models"
        self.EXPORTS_PATH = Path(self.EXPORTS_PATH) if self.EXPORTS_PATH else self.STORAGE_PATH / "exports"

        # Create directories
        self.create_storage_dirs()

    def create_storage_dirs(self) -> None:
        """Create necessary storage directories if they don't exist"""
        for path in [self.STORAGE_PATH, self.IMAGES_PATH, self.MODELS_PATH, self.EXPORTS_PATH]:
            path.mkdir(parents=True, exist_ok=True)

settings = Settings()