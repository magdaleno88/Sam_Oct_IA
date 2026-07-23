#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sam_ml.oct.config import load_config
from sam_ml.oct.dataset import OCTManifestDataset, build_oct_transform
from sam_ml.oct.ensemble import sequential_ensemble_predict
from sam_ml.oct.inference import load_checkpoint_model
from sam_ml.oct.metrics import (
    bootstrap_confidence_intervals,
    evaluate_predictions,
    save_evaluation_artifacts,
    save_metrics,
)

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--split", default="test", choices=["val", "test"])
parser.add_argument("--config", default="configs/oct.yaml")
parser.add_argument("--bootstrap", type=int, default=1000)
args = parser.parse_args()
config = load_config(args.config)
run = Path(args.run)
ensemble_file = run / "ensemble.json"
if ensemble_file.exists():
    checkpoints = json.loads(ensemble_file.read_text(encoding="utf-8"))["checkpoints"]
else:
    candidates = sorted((run / "checkpoints").glob("*.ckpt"))
    checkpoints = [str(next((item for item in candidates if "best" in item.name), candidates[-1]))]
kwargs = {"num_classes": 4, "pretrained": False, "dropout": config.model.dropout, "freeze_backbone": False}
if config.model.name == "improved_resnet50":
    kwargs["replace_stride_with_dilation"] = config.model.replace_stride_with_dilation
models = [load_checkpoint_model(item, config.model.name, kwargs) for item in checkpoints]
manifest = pd.read_csv(config.data.manifest_dir / f"{args.split}.csv")
dataset = OCTManifestDataset(
    manifest, build_oct_transform(False, config.data.image_size),
    preprocessing=config.preprocessing,
)
loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False)
all_probabilities, all_std, labels = [], [], []
for images, targets in loader:
    output = sequential_ensemble_predict(models, images, "cuda" if torch.cuda.is_available() else "cpu")
    all_probabilities.append(output["probabilities"].numpy())
    all_std.append(output["std"].numpy())
    labels.extend(targets.numpy().tolist())
probabilities = np.concatenate(all_probabilities)
std = np.concatenate(all_std)
metrics = evaluate_predictions(labels, probabilities)
metrics["bootstrap_95_ci"] = bootstrap_confidence_intervals(labels, probabilities, args.bootstrap)
evaluation_dir = run / "evaluation" / args.split
save_metrics(metrics, evaluation_dir / "metrics.json")
predictions = manifest.copy()
predictions["predicted_index"] = probabilities.argmax(1)
predictions["confidence"] = probabilities.max(1)
predictions["prediction_margin"] = np.sort(probabilities, axis=1)[:, -1] - np.sort(probabilities, axis=1)[:, -2]
predictions["ensemble_uncertainty"] = std.mean(1)
for index, name in enumerate(("CNV", "DME", "DRUSEN", "NORMAL")):
    predictions[f"probability_{name}"] = probabilities[:, index]
predictions["correct"] = predictions["class_index"] == predictions["predicted_index"]
predictions["checkpoint"] = ";".join(checkpoints)
(run / "predictions").mkdir(parents=True, exist_ok=True)
predictions.to_csv(run / "predictions" / f"{args.split}.csv", index=False)
predictions.to_json(
    run / "predictions" / f"{args.split}.json", orient="records", indent=2,
)
artifacts = save_evaluation_artifacts(
    labels, probabilities, metrics, evaluation_dir,
)
metrics["artifacts"] = artifacts
save_metrics(metrics, evaluation_dir / "metrics.json")
if args.split == "test":
    save_metrics(metrics, run / "metrics.json")
Path("reports").mkdir(exist_ok=True)
predictions.to_csv("reports/error_analysis.csv", index=False)
print(json.dumps(metrics, indent=2))
