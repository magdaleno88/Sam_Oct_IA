"""Checkpoint loading and research-only OCT inference."""

from pathlib import Path

import torch
from PIL import Image

from sam_ml.oct.constants import CLASS_NAMES, DISPLAY_NAMES, RESEARCH_DISCLAIMER
from sam_ml.oct.dataset import build_oct_transform
from sam_ml.oct.ensemble import sequential_ensemble_predict
from sam_ml.oct.models import create_oct_model
from sam_ml.oct.config import OCTPreprocessingConfig
from sam_ml.oct.preprocessing import load_oct_image, preprocess_oct_image


def load_checkpoint_model(checkpoint: str | Path, model_name: str, model_kwargs: dict | None = None):
    model = create_oct_model(model_name, **(model_kwargs or {}))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    state = {key.removeprefix("model."): value for key, value in state.items()}
    model.load_state_dict(state)
    return model.eval()


def prepare_image(
    path: str | Path, image_size: int = 224,
    preprocessing: OCTPreprocessingConfig | None = None,
) -> torch.Tensor:
    try:
        if preprocessing is not None and preprocessing.enabled:
            cleaned = preprocess_oct_image(load_oct_image(path), preprocessing).image
            image = Image.fromarray(cleaned, mode="L")
        else:
            with Image.open(path) as source:
                image = source.convert("L")
        return build_oct_transform(False, image_size)(image).unsqueeze(0)
    except Exception as exc:
        raise ValueError(f"Invalid OCT image {path}: {exc}") from exc


def predict_image(
    models, image: str | Path, device: str = "cpu", image_size: int = 224,
    preprocessing: OCTPreprocessingConfig | None = None,
):
    result = sequential_ensemble_predict(
        models, prepare_image(image, image_size, preprocessing), device
    )
    probabilities = result["probabilities"][0]
    predicted = int(probabilities.argmax())
    return {
        "prediction": CLASS_NAMES[predicted],
        "display_name": DISPLAY_NAMES[CLASS_NAMES[predicted]],
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)},
        "confidence": float(probabilities[predicted]),
        "uncertainty": float(result["std"][0, predicted]),
        "entropy": float(result["entropy"][0]),
        "model_type": "ensemble" if len(models) > 1 else "single",
        "ensemble_size": len(models),
        "disclaimer": RESEARCH_DISCLAIMER,
    }
