# backend/app/ml/models.py
import gc
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

from .segmentation import DocumentSegmenter
from ..utils.logging import ml_logger


class OCRModel:
    def __init__(
            self,
            model_name: str,
            lora_path: Optional[Path] = None,
            device: Optional[str] = None
    ):
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.lora_path = lora_path
        self.processor = None
        self.model = None
        self.segmenter = DocumentSegmenter(device=self.device)
        self._is_loaded = False

    def load(self):
        """Load OCR models"""
        if self._is_loaded:
            return

        try:
            ml_logger.debug("Loading TrOCR Processor")
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            ml_logger.info("Processor loaded successfully")

            ml_logger.debug("Loading Vision2Seq model")
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
            )

            if self.lora_path and self.lora_path.exists():
                self.load_lora(self.lora_path)

            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            ml_logger.info(f"{self.model_name} loaded successfully")

        except Exception as e:
            ml_logger.error(f"Failed to initialize OCR Model: {str(e)}")
            raise

    def unload(self):
        """Unload models and force garbage collection"""
        if self._is_loaded:
            ml_logger.info("Unloading OCR models")
            try:
                if self.model is not None:
                    # Move model to CPU first
                    self.model.cpu()
                    # Delete model and clear CUDA cache
                    del self.model
                    self.model = None

                if self.processor is not None:
                    del self.processor
                    self.processor = None

                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Additional GPU memory cleanup
                    torch.cuda.synchronize()

                self._is_loaded = False
                ml_logger.info("OCR models unloaded and memory cleared successfully")

            except Exception as e:
                ml_logger.error(f"Error during model unloading: {str(e)}")
                raise
            finally:
                # Ensure segmenter is unloaded even if there's an error
                if self.segmenter:
                    self.segmenter.unload()

    @torch.no_grad()
    def load_lora(self, lora_path: Path):
        """Load LORA weights for fine-tuned recognition"""
        try:
            ml_logger.info(f"Loading LoRA weights from {lora_path}")
            if not lora_path.exists():
                raise FileNotFoundError(f"LoRA weights not found at {lora_path}")

            # Load encoder LoRA
            encoder_path = lora_path / "encoder"
            if encoder_path.exists():
                ml_logger.debug("Loading encoder LoRA weights")
                self.model.encoder.load_adapter(encoder_path)

            # Load decoder LoRA
            decoder_path = lora_path / "decoder"
            if decoder_path.exists():
                ml_logger.debug("Loading decoder LoRA weights")
                self.model.decoder.load_adapter(decoder_path)

            if not encoder_path.exists() and not decoder_path.exists():
                raise FileNotFoundError("Neither encoder nor decoder LoRA weights found")

            self.model.to(self.device)
            self.model.eval()
            ml_logger.info("LoRA weights loaded successfully")

        except Exception as e:
            ml_logger.error(f"Failed to load LoRA weights: {str(e)}")
            raise

    @torch.no_grad()
    def extract_text(self, image: Image.Image) -> dict:
        """Extract text from each line image provided by line segmenter."""
        if not self._is_loaded:
            self.load()

        line_segments = self.segmenter.segment_page(image)

        lines = []
        for segment in line_segments:
            try:
                # Process line image directly
                pixel_values = self.processor(segment.image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                generated_ids = self.model.generate(pixel_values, max_length=128, num_beams=4, early_stopping=True)
                line_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Include the line regardless of whether text was recognized
                lines.append({
                    'bbox': [
                        segment.bbox.x1, segment.bbox.y1,
                        segment.bbox.x2, segment.bbox.y2
                    ],
                    'text': line_text.strip(),  # Store empty string if no text recognized
                    'confidence': segment.bbox.confidence,
                    'image': segment.image,
                    'text_recognized': bool(line_text.strip())  # New flag indicating if text was recognized
                })

            except Exception as e:
                ml_logger.warning(f"Error processing line segment: {str(e)}")
                # Include the line even if processing failed
                lines.append({
                    'bbox': [
                        segment.bbox.x1, segment.bbox.y1,
                        segment.bbox.x2, segment.bbox.y2
                    ],
                    'text': '',
                    'confidence': segment.bbox.confidence,
                    'image': segment.image,
                    'text_recognized': False
                })

        return {
            'lines': lines,
            'full_text': '\n'.join(line['text'] for line in lines if line['text']),
            'total_lines': len(lines),
            'lines_with_text': sum(1 for line in lines if line['text']),
            'lines_without_text': sum(1 for line in lines if not line['text'])
        }

