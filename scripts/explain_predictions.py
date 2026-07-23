#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from sam_ml.oct.config import load_config
from sam_ml.oct.explain import occlusion_sensitivity, save_heatmap_triplet
from sam_ml.oct.inference import load_checkpoint_model, prepare_image

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--image", required=True)
parser.add_argument("--config", default="configs/oct.yaml")
args = parser.parse_args()
config = load_config(args.config)
run = Path(args.run)
ensemble_file = run / "ensemble.json"
checkpoints = json.loads(ensemble_file.read_text(encoding="utf-8"))["checkpoints"] if ensemble_file.exists() else [str(next((run / "checkpoints").glob("*best*.ckpt")))]
kwargs = {"num_classes": 4, "pretrained": False, "dropout": config.model.dropout, "freeze_backbone": False}
if config.model.name == "improved_resnet50": kwargs["replace_stride_with_dilation"] = config.model.replace_stride_with_dilation
model = load_checkpoint_model(checkpoints[0], config.model.name, kwargs)
image = prepare_image(args.image, config.data.image_size, config.preprocessing)
heat = occlusion_sensitivity(model, image, window=config.explainability.occlusion_window,
                             stride=config.explainability.occlusion_stride,
                             value=config.explainability.occlusion_value)
(run / "figures").mkdir(parents=True, exist_ok=True)
save_heatmap_triplet(image, heat, str(run / "figures" / Path(args.image).stem))
