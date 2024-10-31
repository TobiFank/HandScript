# backend/app/ml/factory.py
from typing import Dict
from ..config import settings

class ModelNameFactory:
    """Simple factory for getting the correct model name based on language"""

    LANGUAGE_MODELS: Dict[str, str] = {
        'english': settings.DEFAULT_MODEL,
        'german': settings.DEFAULT_MODEL_GERMAN
    }

    @classmethod
    def get_model_name(cls, writer_language: str = 'english') -> str:
        """Get the appropriate model name for the writer's language"""
        language = writer_language.lower()
        return cls.LANGUAGE_MODELS.get(language, cls.LANGUAGE_MODELS['english'])
