"""Synthetic-only tests for the OCT infrastructure; no clinical data is used."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

from sam_ml.oct.config import OCTConfig
from sam_ml.oct.constants import CLASS_NAMES, CLASS_TO_INDEX
from sam_ml.oct.data import audit_dataset, create_manifests, infer_patient_id, validate_no_leakage
from sam_ml.oct.dataset import OCTManifestDataset, build_oct_transform
from sam_ml.oct.ensemble import average_probabilities, sequential_ensemble_predict
from sam_ml.oct.explain import occlusion_sensitivity
from sam_ml.oct.inference import predict_image
from sam_ml.oct.metrics import evaluate_predictions, specificity_per_class
from sam_ml.oct.models import create_oct_model


def _write_image(path: Path, value: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((64, 96), value, dtype=np.uint8)).save(path)


def _official_dataset(root: Path) -> None:
    for class_index, label in enumerate(CLASS_NAMES):
        for patient in range(4):
            _write_image(root / "train" / label / f"{label}-{class_index}{patient:02d}-1.jpeg", 20 + class_index * 30 + patient)
        _write_image(root / "test" / label / f"{label}-9{class_index}9-1.jpeg", 180 + class_index)


class TinyModel(nn.Module):
    def __init__(self, bias: int = 0):
        super().__init__()
        self.bias = bias

    def forward(self, x):
        logits = torch.zeros((len(x), 4), device=x.device)
        logits[:, self.bias] = 2
        return logits


def test_fixed_class_mapping():
    assert CLASS_TO_INDEX == {"CNV": 0, "DME": 1, "DRUSEN": 2, "NORMAL": 3}


def test_patient_id_is_conservative():
    assert infer_patient_id("CNV-1016042-1.jpeg") == "CNV-1016042"
    assert infer_patient_id("unknown.jpeg") is None


def test_manifests_preserve_official_test_and_prevent_leakage(tmp_path):
    root = tmp_path / "raw"
    _official_dataset(root)
    config = OCTConfig()
    config.data.root = root
    config.data.manifest_dir = tmp_path / "manifests"
    config.data.exclusions_file = config.data.manifest_dir / "excluded_images.csv"
    manifests = create_manifests(config)
    assert set(manifests) == {"train", "val", "test"}
    assert len(manifests["test"]) == 4
    validate_no_leakage(manifests)


def test_leakage_validator_rejects_shared_patient(tmp_path):
    path_a, path_b = tmp_path / "a.jpeg", tmp_path / "b.jpeg"
    _write_image(path_a, 1); _write_image(path_b, 2)
    base = {"label": "CNV", "class_index": 0, "patient_id": "CNV-1", "source": "synthetic"}
    train = pd.DataFrame([{**base, "image_path": str(path_a), "split": "train"}])
    val = pd.DataFrame([{**base, "image_path": str(path_b), "split": "val"}])
    with pytest.raises(ValueError, match="Patient leakage"):
        validate_no_leakage({"train": train, "val": val})


def test_audit_reports_corrupt_image(tmp_path):
    root = tmp_path / "raw"
    _write_image(root / "CNV" / "CNV-1-1.jpeg")
    broken = root / "DME" / "DME-2-1.jpeg"
    broken.parent.mkdir(parents=True); broken.write_bytes(b"not-an-image")
    config = OCTConfig()
    config.data.root = root; config.data.manifest_dir = tmp_path / "manifests"
    config.data.exclusions_file = config.data.manifest_dir / "excluded_images.csv"
    summary = audit_dataset(config)
    assert summary["corrupt_images"] == 1
    assert (config.data.manifest_dir / "quality_report.csv").exists()


def test_dataset_tensor_is_three_by_224(tmp_path):
    path = tmp_path / "CNV-1-1.jpeg"; _write_image(path)
    frame = pd.DataFrame([{"image_path": str(path), "label": "CNV", "class_index": 0, "split": "test"}])
    tensor, label = OCTManifestDataset(frame)[0]
    assert tensor.shape == (3, 224, 224) and label == 0


def test_resnet_models_output_four_logits():
    for name in ("baseline_resnet50", "improved_resnet50"):
        model = create_oct_model(name, pretrained=False, num_classes=4, dropout=0.0)
        model.eval()
        with torch.inference_mode():
            assert model(torch.randn(1, 3, 64, 64)).shape == (1, 4)


def test_ensemble_is_arithmetic_probability_mean():
    logits = [torch.tensor([[2.0, 0, 0, 0]]), torch.tensor([[0.0, 2, 0, 0]])]
    mean, std = average_probabilities(logits)
    expected = torch.stack([torch.softmax(item, 1) for item in logits]).mean(0)
    assert torch.allclose(mean, expected) and torch.allclose(mean.sum(1), torch.ones(1))
    assert std.shape == mean.shape


def test_metrics_specificity_and_auc():
    truth = np.array([0, 1, 2, 3] * 3)
    probabilities = np.eye(4)[truth] * 0.9 + 0.025
    probabilities /= probabilities.sum(1, keepdims=True)
    result = evaluate_predictions(truth, probabilities)
    assert result["accuracy"] == 1.0
    assert result["per_class"]["CNV"]["specificity"] == 1.0
    assert result["macro_auc"] == 1.0


def test_cpu_inference_and_occlusion(tmp_path):
    path = tmp_path / "oct.jpeg"; _write_image(path)
    result = predict_image([TinyModel(1), TinyModel(1)], path, "cpu")
    assert result["prediction"] == "DME"
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    image = torch.zeros(1, 3, 28, 28)
    heat = occlusion_sensitivity(TinyModel(0), image, window=14, stride=14)
    assert heat.shape == (28, 28)
