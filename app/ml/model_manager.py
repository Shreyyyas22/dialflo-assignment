import logging
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoProcessor, Wav2Vec2Model

from app.core.config import settings

logger = logging.getLogger(__name__)


class Wav2Vec2AgeGenderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = nn.ModuleDict(
            {
                "dense": nn.Linear(config.hidden_size, config.hidden_size),
                "out_proj": nn.Linear(config.hidden_size, 1),
            }
        )
        self.gender = nn.ModuleDict(
            {
                "dense": nn.Linear(config.hidden_size, config.hidden_size),
                "out_proj": nn.Linear(config.hidden_size, 3),
            }
        )

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state.mean(dim=1)

        age_hidden = torch.relu(self.age["dense"](hidden_states))
        age_logits = self.age["out_proj"](age_hidden)

        gender_hidden = torch.relu(self.gender["dense"](hidden_states))
        gender_logits = self.gender["out_proj"](gender_hidden)

        return age_logits, gender_logits


_model_loaded: bool = False
_model: Wav2Vec2AgeGenderModel | None = None
_processor: AutoProcessor | None = None


def load_model() -> None:
    global _model_loaded, _model, _processor
    if _model_loaded:
        return

    model_name = settings.MODEL_NAME
    logger.info(f"Loading processor and model '{model_name}'...")
    try:
        config = AutoConfig.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        model = Wav2Vec2AgeGenderModel(config)

        weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        # Set PyTorch thread count to optimize CPU execution
        torch.set_num_threads(max(1, os.cpu_count() or 1))

        _processor = processor
        _model = model
        _model_loaded = True
        logger.info("Model and processor successfully loaded and ready.")
    except Exception:
        _model_loaded = False
        logger.exception(f"Failed to load model '{model_name}'")
        raise


def is_model_loaded() -> bool:
    return _model_loaded


def set_model_loaded(status: bool) -> None:
    global _model_loaded
    _model_loaded = status


def get_model() -> Wav2Vec2AgeGenderModel | None:
    return _model


def get_processor() -> AutoProcessor | None:
    return _processor
