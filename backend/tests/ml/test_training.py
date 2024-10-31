# tests/ml/test_training.py
from typing import List, Dict
import pytest
import torch
from pathlib import Path
from PIL import Image
from app.ml.training import HandwritingDataset, LoraTrainer
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForVision2Seq
from torch.utils.data import Dataset

@pytest.fixture
def mock_processor(monkeypatch):
    class MockProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, images=None, text=None, return_tensors="pt", **kwargs):
            output = {}

            if images is not None:
                output["pixel_values"] = torch.randn(1, 3, 224, 224)

            if text is not None:
                output["input_ids"] = torch.randint(0, 1000, (1, 10))
                output["attention_mask"] = torch.ones(1, 10)

            return type('ProcessorOutput', (dict,), {
                '__getitem__': lambda self, key: output[key],
                'pixel_values': output.get('pixel_values'),
                'input_ids': output.get('input_ids'),
                'attention_mask': output.get('attention_mask')
            })()

        def batch_decode(self, *args, **kwargs):
            return ["test output"]

    monkeypatch.setattr("transformers.AutoProcessor.from_pretrained", lambda *args, **kwargs: MockProcessor())
    return MockProcessor()

class MockDataset(Dataset):
    def __init__(self, size=2):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "attention_mask": torch.ones(10),
            "input_ids": torch.randint(0, 1000, (10,)),
            "pixel_values": torch.randn(3, 224, 224),
            "labels": torch.randint(0, 1000, (10,))
        }

class ModelOutput(Dict):
    """Mock ModelOutput that behaves like transformers.modeling_outputs.ModelOutput"""
    def __init__(self, loss, logits=None, **kwargs):
        super().__init__()
        self["loss"] = loss if isinstance(loss, torch.Tensor) else torch.tensor(loss, requires_grad=True)
        self["logits"] = logits if logits is not None else torch.randn(1, 10, 1000, requires_grad=True)
        for k, v in kwargs.items():
            self[k] = v

    def __getitem__(self, k):
        return super().__getitem__(k)

    def keys(self):
        return super().keys()

class MockConfig:
    def __init__(self):
        self.task_specific_params = {}
        self.architectures = ['MockModel']
        self.tie_word_embeddings = False
        self.model_type = 'vision2seq'

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self):
        return {
            'task_specific_params': self.task_specific_params,
            'architectures': self.architectures,
            'tie_word_embeddings': self.tie_word_embeddings,
            'model_type': self.model_type
        }

@pytest.fixture
def mock_trainer_obj(monkeypatch):
    """Create and configure mock trainer for testing"""
    class MockTrainer:
        def __init__(self, *args, **kwargs):
            self.args = kwargs.get('args', None)
            self.model = kwargs.get('model', None)
            self.train_dataset = kwargs.get('train_dataset', None)

        def train(self, *args, **kwargs):
            # Mock successful training
            return None

        def compute_loss(self, *args, **kwargs):
            return torch.tensor(0.1, requires_grad=True, device='cuda:0')

    # We need to patch the actual Trainer import location in our app code
    monkeypatch.setattr("app.ml.training.Trainer", MockTrainer)
    return MockTrainer

class MockModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.query = nn.Linear(10, 10)
        self.value = nn.Linear(10, 10)
        self.config = MockConfig()

    def to(self, device):
        # Ensure internal tensors move to the right device
        self.query = self.query.to(device)
        self.value = self.value.to(device)
        return self

    def train(self, mode=True):
        return self

    def eval(self):
        return self

    def forward(self, pixel_values=None, labels=None, attention_mask=None, **kwargs):
        # Make sure we use the same device for all tensors
        device = pixel_values.device if pixel_values is not None else 'cuda:0'

        # Use the linear layers to create a computation graph on the correct device
        dummy_input = torch.randn(1, 10, requires_grad=True, device=device)
        query_output = self.query(dummy_input)
        value_output = self.value(query_output)
        loss = value_output.mean()

        return ModelOutput(
            loss=loss,  # This will be on the correct device because it comes from the model
            logits=value_output
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {}

    def generate(self, *args, **kwargs):
        return torch.tensor([[1, 2, 3]])

    def get_encoder(self):
        return self

    def get_decoder(self):
        return self

    def get_input_embeddings(self):
        return None

    def get_output_embeddings(self):
        return None

    def parameters(self):
        return [p for p in super().parameters()]

    def state_dict(self):
        return {}

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

# Here's the critical fix - we need to use @pytest.fixture to properly register our mock_model
@pytest.fixture
def mock_model(monkeypatch):
    """Create and configure mock model for testing"""
    def mock_create_model(*args, **kwargs):
        return MockModel()
    monkeypatch.setattr("transformers.AutoModelForVision2Seq.from_pretrained", mock_create_model)
    return MockModel()

@pytest.mark.asyncio
async def test_lora_trainer_train(mock_processor, mock_model, mock_trainer_obj, temp_storage_dir, monkeypatch):
    """Test LoraTrainer training"""
    # Mock the dataset creation with max_length parameter
    def mock_dataset_init(self, image_paths, texts, processor, max_length=512):
        # Store the original arguments as required by HandwritingDataset
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor
        self.max_length = max_length
        # Add the mock dataset functionality
        self.dataset = MockDataset()
        self.__getitem__ = self.dataset.__getitem__

    monkeypatch.setattr("app.ml.training.HandwritingDataset.__init__", mock_dataset_init)

    # Create test data paths
    image_paths = [temp_storage_dir / f"train_image_{i}.png" for i in range(2)]
    for path in image_paths:
        Image.new('RGB', (100, 100), color='white').save(path)

    texts = ["Training text 1", "Training text 2"]

    # Create training output directory
    train_output_dir = temp_storage_dir / "training_output"
    train_output_dir.mkdir(parents=True, exist_ok=True)

    trainer = LoraTrainer()
    trainer.setup_model()

    try:
        output_path = trainer.train(
            train_images=image_paths,
            train_texts=texts,
            output_dir=train_output_dir,
            num_epochs=1,
            batch_size=1,
            max_length=128  # Test with smaller max_length
        )

        assert isinstance(output_path, Path)
        assert str(temp_storage_dir) in str(output_path)
    except Exception as e:
        pytest.fail(f"Training failed: {str(e)}")

@pytest.mark.asyncio
async def test_lora_trainer_error_handling(mock_processor, mock_model, mock_trainer_obj, temp_storage_dir):
    """Test LoraTrainer error handling with empty input"""
    trainer = LoraTrainer()
    trainer.setup_model()

    with pytest.raises(ValueError, match="No training samples provided"):
        await trainer.train(
            train_images=[],
            train_texts=[],
            output_dir=temp_storage_dir / "training_output"
        )