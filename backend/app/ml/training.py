# backend/app/ml/training.py
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback
)

from jiwer import wer, cer  # For word/character error rate calculation
from datetime import datetime


@dataclass
class TrainingConfig:
    """Enhanced training configuration with quality-focused parameters"""
    # Basic training params
    num_epochs: int = 15
    batch_size: int = 2  # Reduced from 4 to be safe with memory
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    max_length: int = 256

    # Learning rate schedule
    warmup_ratio: float = 0.2
    weight_decay: float = 0.0005
    max_grad_norm: float = 1.0

    # LoRA specific
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Quality focused parameters
    label_smoothing: float = 0.1
    validation_split: float = 0.2
    early_stopping_patience: int = 3
    min_samples_for_training: int = 3

    # New parameters for better training
    max_grad_norm: float = 1.0
    warmup_steps: Optional[int] = None  # Will be calculated based on data size
    eval_steps: int = 50
    save_steps: int = 50
    logging_steps: int = 10

    # Quality metrics
    eval_accumulation_steps: int = 4
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


def get_sample_adjusted_config(num_samples: int) -> TrainingConfig:
    """Comprehensive config adjustment based on handwriting dataset size"""
    base_config = TrainingConfig()

    if num_samples < base_config.min_samples_for_training:
        raise ValueError(f"Need at least {base_config.min_samples_for_training} samples")

    # Very Few Samples (<10)
    if num_samples < 10:
        base_config.learning_rate = 2e-6
        base_config.lora_r = 4  # Minimal LORA rank
        base_config.lora_alpha = 8
        base_config.early_stopping_patience = 8
        base_config.num_epochs = 25  # More epochs for thorough learning
        base_config.weight_decay = 0.0001  # Reduce regularization
        base_config.warmup_ratio = 0.3  # Longer warmup
        base_config.label_smoothing = 0.2  # More smoothing for generalization
        # High augmentation probability would be set in dataset

    # Initial Learning (10-30)
    elif num_samples < 30:
        base_config.learning_rate = 5e-6
        base_config.lora_r = 8
        base_config.lora_alpha = 16
        base_config.early_stopping_patience = 6
        base_config.num_epochs = 20
        base_config.weight_decay = 0.0002
        base_config.warmup_ratio = 0.25
        base_config.label_smoothing = 0.15

    # Basic Pattern Recognition (30-100)
    elif num_samples < 100:
        base_config.learning_rate = 1e-5
        base_config.lora_r = 12
        base_config.lora_alpha = 24
        base_config.early_stopping_patience = 5
        base_config.num_epochs = 15
        base_config.weight_decay = 0.0003
        base_config.warmup_ratio = 0.2
        base_config.label_smoothing = 0.1

    # Comprehensive Learning (100-500)
    elif num_samples < 500:
        base_config.learning_rate = 1.5e-5
        base_config.lora_r = 16
        base_config.lora_alpha = 32
        base_config.early_stopping_patience = 4
        base_config.num_epochs = 12
        base_config.weight_decay = 0.0004
        base_config.warmup_ratio = 0.15
        base_config.label_smoothing = 0.08

    # Rich Dataset (500+)
    else:
        base_config.learning_rate = 2e-5
        base_config.lora_r = 16
        base_config.lora_alpha = 32
        base_config.early_stopping_patience = 3
        base_config.num_epochs = 10
        base_config.weight_decay = 0.0005
        base_config.warmup_ratio = 0.1
        base_config.label_smoothing = 0.05

    # Dynamic batch sizing based on GPU memory and dataset size
    base_config.batch_size = min(2, max(1, num_samples // 100))
    base_config.gradient_accumulation_steps = max(4, min(16, 32 // base_config.batch_size))

    # Adjust validation split (smaller for small datasets)
    base_config.validation_split = min(0.2, max(0.1, 10 / num_samples))

    # Calculate steps based on dataset size
    total_steps = math.ceil(num_samples * base_config.num_epochs / base_config.batch_size)
    base_config.eval_steps = max(1, min(50, num_samples // (4 * base_config.batch_size)))
    base_config.save_steps = base_config.eval_steps
    base_config.logging_steps = max(1, base_config.eval_steps // 5)
    base_config.warmup_steps = int(total_steps * base_config.warmup_ratio)

    return base_config


class HandwritingAugmenter:
    """Enhanced augmentation with quality focus"""

    @staticmethod
    def add_handwriting_variation(image: Image.Image) -> Image.Image:
        """Simulate handwriting style variations with improved quality."""
        img_array = np.array(image)

        # Enhanced slant variation
        slant_factor = np.random.uniform(-0.2, 0.2)
        height, width = img_array.shape[:2]
        slant_matrix = np.float32([[1, slant_factor, 0], [0, 1, 0]])
        img_array = cv2.warpAffine(img_array, slant_matrix, (width, height),
                                   borderMode=cv2.BORDER_REPLICATE)

        # Improved thickness variation
        kernel_size = np.random.choice([2, 3])
        if np.random.random() > 0.5:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            img_array = cv2.dilate(img_array, kernel)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            img_array = cv2.erode(img_array, kernel)

        return Image.fromarray(img_array)

    @staticmethod
    def add_paper_texture(image: Image.Image) -> Image.Image:
        """Add realistic paper texture with improved quality."""
        img_array = np.array(image)

        # Higher quality noise
        noise = np.random.normal(0, 1.5, img_array.shape).astype(np.uint8)
        img_array = cv2.addWeighted(img_array, 1.0, noise, 0.1, 0)

        # Improved paper texture
        texture = np.random.normal(250, 3, img_array.shape).astype(np.uint8)
        img_array = cv2.addWeighted(img_array, 0.95, texture, 0.05, 0)

        return Image.fromarray(img_array)


class HandwritingDataset(Dataset):
    """Enhanced dataset with improved augmentation pipeline"""

    def __init__(
            self,
            image_paths: List[Path],
            texts: List[str],
            processor: TrOCRProcessor,
            max_length: int = 256,
            augment: bool = False
    ):
        if not image_paths or not texts or len(image_paths) != len(texts):
            raise ValueError("Invalid or mismatched image_paths and texts")

        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor
        self.max_length = max_length
        self.augment = augment
        self.augmenter = HandwritingAugmenter()

        # Enhanced augmentation pipeline
        self.augmentation = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomRotation((-3, 3)),  # More subtle rotation
                transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),  # More subtle translation
                    scale=(0.98, 1.02),  # More subtle scaling
                    shear=(-5, 5)  # More subtle shear
                )
            ], p=0.7),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.05
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(3, sigma=(0.1, 0.5))
            ], p=0.2),
            transforms.RandomApply([
                transforms.Lambda(lambda x: self.augmenter.add_handwriting_variation(x))
            ], p=0.3),
            transforms.RandomApply([
                transforms.Lambda(lambda x: self.augmenter.add_paper_texture(x))
            ], p=0.2)
        ]) if augment else None

    def __len__(self) -> int:
        return len(self.image_paths)

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        if self.augment and self.augmentation:
            image = self.augmentation(image)
        return image

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            image = self.preprocess_image(image)

            encoder_inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).pixel_values.squeeze(0)

            tokenizer_output = self.processor.tokenizer(
                self.texts[idx],
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )

            decoder_input_ids = tokenizer_output.input_ids.clone().squeeze(0)
            labels = tokenizer_output.input_ids.squeeze(0)
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

            return {
                "pixel_values": encoder_inputs,
                "decoder_input_ids": decoder_input_ids,
                "labels": labels,
            }

        except Exception as e:
            raise RuntimeError(f"Error processing item {idx}: {str(e)}")


class LoraTrainer:
    """Enhanced trainer with improved quality focus"""

    def __init__(
            self,
            model_name: str,
            device: Optional[str] = None,
            config: Optional[TrainingConfig] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.config = config or TrainingConfig()

    def _initialize_missing_weights(self, base_model: VisionEncoderDecoderModel):
        """Initialize missing weights from base model."""
        if not hasattr(base_model.decoder, "output_projection"):
            return

        # Copy output projection weights if they exist in base model
        if hasattr(self.model.decoder, "output_projection"):
            self.model.decoder.output_projection = base_model.decoder.output_projection

    def setup_model(self) -> bool:
        try:
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            base_model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

            # Create our model instance
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

            # Initialize missing weights
            self._initialize_missing_weights(base_model)

            # Model configuration
            self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
            self.model.config.vocab_size = self.processor.tokenizer.vocab_size
            self.model.config.use_cache = False

            # Set requires_grad before LoRA
            self.model.requires_grad_(True)

            # Enhanced LoRA configurations with comprehensive target modules
            encoder_lora = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=["query", "key", "value", "output.dense"],
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="VISION"
            )

            decoder_lora = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )

            # Apply LoRA with gradient tracking
            self.model.encoder = get_peft_model(self.model.encoder, encoder_lora)
            self.model.decoder = get_peft_model(self.model.decoder, decoder_lora)

            # Verify gradients
            for name, param in self.model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad_(True)

            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            if total_params == 0:
                raise ValueError("No parameters have requires_grad=True")

            self.model.to(self.device)

            # Clean up base model
            del base_model
            torch.cuda.empty_cache()

            return True

        except Exception as e:
            print(f"Model setup failed: {str(e)}")
            return False

    def train(self, train_images: List[Path], train_texts: List[str], output_dir: Path) -> Path:
        if not self.model or not self.processor:
            if not self.setup_model():
                raise RuntimeError("Failed to initialize model")

        # Get sample-adjusted config
        if not self.config:
            self.config = get_sample_adjusted_config(len(train_images))

        # Create dataset
        full_dataset = HandwritingDataset(
            train_images, train_texts,
            self.processor,
            max_length=self.config.max_length,
            augment=True
        )

        # Split dataset
        val_size = int(len(full_dataset) * self.config.validation_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        class CacheClearCallback(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                torch.cuda.empty_cache()

        # Calculate proper warmup steps
        num_training_steps = len(train_dataset) * self.config.num_epochs // (
                self.config.batch_size * self.config.gradient_accumulation_steps)
        warmup_steps = max(1, int(num_training_steps * self.config.warmup_ratio))

        # Training arguments with built-in scheduler configuration
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            save_strategy="steps",
            evaluation_strategy="steps",
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            fp16=False,
            label_smoothing_factor=self.config.label_smoothing,
            gradient_checkpointing=False,
            group_by_length=False,
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=True,
            torch_compile=False,
            seed=42,
            load_best_model_at_end=True,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            eval_accumulation_steps=self.config.eval_accumulation_steps,
            # Add scheduler configuration here
            lr_scheduler_type="cosine",  # Use cosine scheduler
            warmup_ratio=self.config.warmup_ratio  # Use ratio instead of steps for more flexibility
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collate_fn,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=0.001
                ),
                CacheClearCallback()
            ]
        )

        # Remove custom scheduler creation
        # Train with enhanced error handling
        try:
            trainer.train()
        except Exception as e:
            # Log error and try to save partial progress
            print(f"Training interrupted: {str(e)}")
            try:
                self.model.encoder.save_pretrained(output_dir / "model" / "encoder")
                self.model.decoder.save_pretrained(output_dir / "model" / "decoder")
                self.processor.save_pretrained(output_dir / "model")
                print("Partial model saved despite training interruption")
            except:
                print("Could not save partial model")
            raise

        # Save the best model
        output_path = output_dir / "model"
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            self.model.encoder.save_pretrained(output_path / "encoder")
            self.model.decoder.save_pretrained(output_path / "decoder")
            self.processor.save_pretrained(output_path)
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise

        return output_path

    @torch.no_grad()
    def evaluate_samples(self, validation_images: List[Path], validation_texts: List[str]) -> Dict:
        """Evaluate model performance on provided samples"""
        self.model.eval()
        total_chars = 0
        total_words = 0
        predictions = []
        start_time = time.perf_counter()

        try:
            for image_path, ground_truth in zip(validation_images, validation_texts):
                # Process image
                image = Image.open(image_path).convert('RGB')
                pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)

                # Generate prediction
                generated_ids = self.model.generate(pixel_values)
                pred_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                predictions.append(pred_text)
                total_chars += len(ground_truth)
                total_words += len(ground_truth.split())

            # Calculate metrics
            character_error = cer(validation_texts, predictions)
            word_error = wer(validation_texts, predictions)
            avg_time = (time.perf_counter() - start_time) / len(validation_images)

            return {
                "character_error_rate": float(character_error),
                "word_error_rate": float(word_error),
                "samples_evaluated": len(validation_images),
                "avg_processing_time": avg_time,
                "evaluation_date": datetime.now().isoformat(),
                "predictions": predictions  # Include for potential detailed analysis
            }

        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            return None


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Enhanced collate function with error checking"""
    try:
        pixel_values = torch.stack([example["pixel_values"] for example in batch])
        labels = torch.stack([example["labels"] for example in batch])
        decoder_input_ids = torch.stack([example["decoder_input_ids"] for example in batch])

        # Verify tensor shapes
        assert all(p.shape == pixel_values[0].shape for p in pixel_values), "Mismatched pixel value shapes"
        assert all(l.shape == labels[0].shape for l in labels), "Mismatched label shapes"
        assert all(d.shape == decoder_input_ids[0].shape for d in decoder_input_ids), "Mismatched decoder input shapes"

        return {
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
    except Exception as e:
        raise RuntimeError(f"Collation error: {str(e)}")
