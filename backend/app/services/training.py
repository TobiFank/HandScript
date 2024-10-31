# backend/app/services/training.py
import asyncio
import gc
import time
from pathlib import Path
from typing import List, Tuple

import torch
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from .cleanup import cleanup_service
from ..config import settings
from ..ml.factory import ModelNameFactory
from ..ml.training import LoraTrainer
from ..models.training_sample import TrainingSample
from ..models.writer import Writer
from ..utils.logging import service_logger

from jiwer import wer, cer



class TrainingService:
    def __init__(self):
        self.trainer = None
        self._training_lock = asyncio.Lock()
        self._active_trainings = {}
        service_logger.info(
            f"TrainingService initialized; active_trainings={len(self._active_trainings)}; trainer_status='uninitialized'"
        )

    async def initialize_trainer(self, writer_id: int, db: Session):
        """Initialize the trainer if needed"""
        start_time = time.time()
        service_logger.info(
            f"Initializing trainer; trainer_exists={self.trainer is not None}"
        )

        try:
            if self.trainer is None:
                writer = db.query(Writer).filter(Writer.id == writer_id).first()
                if not writer:
                    service_logger.error(f"Writer not found; writer_id={writer_id}")
                    raise ValueError("Writer not found")

                writer_language = writer.language or 'english'
                service_logger.info(f"Writer language: {writer_language}")
                model_name = ModelNameFactory.get_model_name(writer_language)

                self.trainer = LoraTrainer(model_name=model_name)
                await asyncio.to_thread(self.trainer.setup_model)
                setup_time = time.time() - start_time
                service_logger.info(
                    f"Trainer initialized successfully; setup_time_seconds={round(setup_time, 2)}; model_type='{model_name}'"
                )
            else:
                service_logger.debug("Trainer already initialized, skipping initialization")
        except Exception as e:
            service_logger.error(
                f"Failed to initialize trainer; error_type='{type(e).__name__}'; setup_time_seconds={round(time.time() - start_time, 2)}",
                exc_info=True
            )
            raise

    async def prepare_training_data(
            self,
            db: Session,
            writer_id: int,
            original_samples: List[TrainingSample]
    ) -> Tuple[List[Path], List[str]]:
        """Prepare line-level training data from samples"""
        images = []
        texts = []

        service_logger.info(f"Preparing training data for writer {writer_id}")

        for sample in original_samples:
            service_logger.debug(f"Processing sample {sample.id}: {sample.image_path}")

            image_path = settings.STORAGE_PATH / sample.image_path
            if not image_path.exists():
                service_logger.warning(f"Image not found: {image_path}")
                continue

            # Add the image and text directly
            images.append(image_path)
            texts.append(sample.text)

        service_logger.info(f"Prepared {len(images)} training samples")

        if not images:
            service_logger.error("No valid training samples found")
            raise ValueError("No valid training samples found")

        return images, texts

    async def train_writer_model(
            self,
            db: Session,
            writer_id: int,
            sample_pages: List[Tuple[Path, str]]
    ):
        """Train a writer-specific model using line-level training"""
        start_time = time.time()
        service_logger.info(
            f"Starting writer model training; writer_id={writer_id}; total_samples={len(sample_pages)}"
        )

        writer = db.query(Writer).filter(Writer.id == writer_id).first()
        if not writer:
            service_logger.error(f"Writer not found; writer_id={writer_id}")
            raise ValueError("Writer not found")

        if writer_id in self._active_trainings:
            service_logger.warning(
                f"Training already in progress; writer_id={writer_id}; writer_status='{writer.status}'"
            )
            raise ValueError("Training already in progress for this writer")

        try:
            self._active_trainings[writer_id] = True

            # Update writer status
            writer.status = "training"
            db.commit()

            # Initialize trainer
            await self.initialize_trainer(writer_id, db)

            # Get training samples from database
            training_samples = db.query(TrainingSample).filter(
                TrainingSample.writer_id == writer_id
            ).all()

            if not training_samples:
                raise ValueError("No training samples found")

            # Prepare line-level training data
            train_images, train_texts = await self.prepare_training_data(
                db, writer_id, training_samples
            )

            if not train_images:
                raise ValueError("No valid training samples provided")

            # Create output directory for this writer
            output_dir = settings.MODELS_PATH / str(writer_id)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Train LORA adapter
            service_logger.info(
                f"Initiating LORA adapter training; writer_id={writer_id}; valid_samples={len(train_images)}; output_dir='{output_dir}'"
            )

            # Run training synchronously in a thread pool
            output_path = await asyncio.to_thread(
                self.trainer.train,
                train_images,
                train_texts,
                output_dir
            )

            try:
                # Evaluate model performance
                evaluation_results = await asyncio.to_thread(
                    self.trainer.evaluate_samples,
                    train_images,
                    train_texts
                )

                if evaluation_results:
                    # Update writer with evaluation metrics
                    writer.evaluation_metrics = evaluation_results
                    writer.accuracy = (1 - evaluation_results["character_error_rate"]) * 100

                    # Update writer status and other fields
                    writer.model_path = str(output_path.relative_to(settings.MODELS_PATH))
                    writer.status = "ready"
                    writer.last_trained = func.now()
                    writer.is_trained = True
                    db.commit()

                    service_logger.info(
                        f"Model evaluation completed; writer_id={writer_id}; cer={evaluation_results['character_error_rate']}; wer={evaluation_results['word_error_rate']}"
                    )

            except Exception as e:
                service_logger.error(
                    f"Evaluation failed; writer_id={writer_id}; error={str(e)}",
                    exc_info=True
                )

            # Continue with writer update even if evaluation fails
            writer.model_path = str(output_path.relative_to(settings.MODELS_PATH))
            writer.status = "ready"
            writer.last_trained = func.now()
            writer.is_trained = True
            db.commit()

            await cleanup_service.cleanup_training_checkpoints(writer_id)

            training_time = time.time() - start_time
            service_logger.info(
                f"LORA adapter training completed; writer_id={writer_id}; training_time_seconds={round(training_time, 2)}; adapter_path='{writer.model_path}'; samples_processed={len(train_images)}"
            )

            return {"success": True, "model_path": writer.model_path}

        except Exception as e:
            error_time = time.time() - start_time
            service_logger.error(
                f"Training failed; writer_id={writer_id}; error_type='{type(e).__name__}'; error_message='{str(e)}'; time_before_error_seconds={round(error_time, 2)}",
                exc_info=True
            )
            writer.status = "error"
            db.commit()
            raise

        finally:
            self._active_trainings.pop(writer_id, None)
            # Unload trainer to free memory
            if self.trainer:
                if hasattr(self.trainer, 'model'):
                    self.trainer.model.cpu()
                    del self.trainer.model
                if hasattr(self.trainer, 'processor'):
                    del self.trainer.processor
                del self.trainer
                self.trainer = None

                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()


training_service = TrainingService()
