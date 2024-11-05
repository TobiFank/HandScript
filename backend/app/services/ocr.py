# backend/app/services/ocr.py
import asyncio
import gc
import time
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

from ..config import settings
from ..ml.factory import ModelNameFactory
from ..ml.models import OCRModel
from ..ml.types import LineSegment
from ..models import Writer
from ..utils.logging import service_logger

from sqlalchemy.orm import Session


class OCRService:
    def __init__(self):
        self.model = None
        self._model_lock = asyncio.Lock()
        service_logger.info("OCR Service initialized", extra={
            "default_model": settings.DEFAULT_MODEL,
            "device": settings.DEVICE
        })

    async def get_model(self, writer_id: int, db: Session) -> OCRModel:
        """Get or initialize the OCR model based on writer settings"""
        async with self._model_lock:
            try:
                # Get writer and their settings
                writer = db.query(Writer).filter(Writer.id == writer_id).first()
                if not writer:
                    raise ValueError(f"Writer {writer_id} not found")

                writer_language = writer.language or 'english'
                writer_model_path = (settings.MODELS_PATH / writer.model_path) if writer.model_path else None

                model_name = ModelNameFactory.get_model_name(writer_language)

                if self.model is None or (writer_model_path and writer_model_path.exists()):
                    service_logger.info(
                        "Initializing OCR model",
                        extra={
                            "writer_id": writer_id,
                            "language": writer_language,
                            "model_name": model_name,
                            "has_custom_model": writer_model_path is not None
                        }
                    )
                    self.model = OCRModel(
                        model_name=model_name,
                        lora_path=writer_model_path
                    )

                self.model.load()
                return self.model

            except Exception as e:
                service_logger.error(
                    "Failed to initialize OCR model",
                    extra={
                        "writer_id": writer_id,
                        "error": str(e)
                    }
                )
                raise

    async def process_image(self, image_path: Path, writer_id: int, db: Session) -> str:
        """Process a full page image and return extracted text with proper line breaks"""
        start_time = time.perf_counter()

        service_logger.info("Starting image processing", extra={
            "image_path": str(image_path),
            "writer_id": writer_id
        })

        try:
            model = await self.get_model(writer_id, db)

            # Load and preprocess image
            service_logger.debug("Loading image file", extra={
                "image_path": str(image_path)
            })

            try:
                with Image.open(image_path) as img:
                    original_mode = img.mode
                    if img.mode != 'RGB':
                        service_logger.debug("Converting image mode", extra={
                            "original_mode": original_mode,
                            "target_mode": "RGB"
                        })
                        img = img.convert('RGB')

                    # Extract text using the new line-aware processing
                    service_logger.debug("Extracting text from image", extra={
                        "image_size": img.size
                    })
                    extracted_text = model.extract_text(img)

                    elapsed_time = time.perf_counter() - start_time
                    service_logger.info("Image processing completed successfully", extra={
                        "text_length": len(extracted_text),
                        "processing_time_ms": round(elapsed_time * 1000, 2),
                        "image_path": str(image_path)
                    })

                    return extracted_text

            except IOError as e:
                service_logger.error("Failed to open or process image file", extra={
                    "image_path": str(image_path),
                    "error_type": "IOError",
                    "error_details": str(e)
                })
                raise

        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            service_logger.error("Image processing failed", extra={
                "error_type": type(e).__name__,
                "image_path": str(image_path),
                "writer_id": writer_id,
                "processing_time_ms": round(elapsed_time * 1000, 2)
            })
            raise

        finally:
            # Always unload the model after processing
            if self.model:
                self.model.unload()
                self.model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

    async def process_training_sample(
            self,
            image_path: Path
    ) -> Tuple[List[LineSegment], str]:
        """Process a training sample image into line segments with text"""
        try:
            model = await self.get_model()

            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Get line segments and extract text for each line
                line_segments = model.segmenter.segment_page(img)

                # Collect text from each line segment
                texts = [model.extract_text(segment.image) for segment in line_segments]

                # Assign text back to line segments for consistency
                for segment, text in zip(line_segments, texts):
                    segment.text = text

                # Return line segments and combined text for training sample
                full_text = "\n".join(texts)
                return line_segments, full_text

        finally:
            if self.model:
                self.model.unload()

    @torch.no_grad()
    async def process_lines_for_training(
            self,
            image_path: Path
    ) -> List[LineSegment]:
        """Process a multiline image and return individual line segments with OCR text"""
        try:
            service_logger.info("Starting line processing", extra={
                "image_path": str(image_path)
            })

            # Get or initialize model using existing mechanism
            model = await self.get_model()

            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    service_logger.debug("Converting image mode", extra={
                        "original_mode": img.mode,
                        "new_mode": "RGB"
                    })
                    img = img.convert('RGB')

                # Get line segments using existing segmenter
                service_logger.debug("Starting line segmentation")
                line_segments = self.segmenter.segment_page(img)
                service_logger.info("Line segmentation complete", extra={
                    "num_lines": len(line_segments)
                })

                # Process each line for text
                for idx, segment in enumerate(line_segments):
                    try:
                        service_logger.debug(f"Processing line {idx}")

                        # Process line image using existing processor
                        pixel_values = self.processor(segment.image, return_tensors="pt").pixel_values
                        pixel_values = pixel_values.to(self.device)

                        generated_ids = self.model.generate(
                            pixel_values,
                            max_length=128,
                            num_beams=4,
                            early_stopping=True
                        )

                        text = self.processor.batch_decode(
                            generated_ids,
                            skip_special_tokens=True
                        )[0].strip()

                        # Update segment with extracted text
                        segment.text = text

                        service_logger.debug(f"Processed line {idx}", extra={
                            "text_length": len(text)
                        })

                    except Exception as e:
                        service_logger.error(f"Error processing line {idx}", extra={
                            "error": str(e)
                        }, exc_info=True)
                        # Continue with next line instead of failing completely
                        segment.text = ""  # Set empty text for failed OCR
                        continue

                return line_segments

        except Exception as e:
            service_logger.error("Error in line processing", extra={
                "error": str(e)
            }, exc_info=True)
            raise
        finally:
            # Use existing unload mechanism
            if self.model:
                self.unload()


ocr_service = OCRService()
