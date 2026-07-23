#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from sam_ml.oct.config import load_config
from sam_ml.oct.training import train_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/oct.yaml")
parser.add_argument("--ensemble-size", type=int)
parser.add_argument("--experiment", required=True)
args = parser.parse_args()
config = load_config(args.config)
size = args.ensemble_size or config.training.ensemble_size
checkpoints = []
for member in range(size):
    run, checkpoint = train_experiment(
        config, f"{args.experiment}/member_{member + 1}", seed=config.training.seed + member
    )
    checkpoints.append(str(checkpoint))
ensemble_dir = Path("runs") / args.experiment
(ensemble_dir / "ensemble.json").write_text(json.dumps({"checkpoints": checkpoints}, indent=2), encoding="utf-8")
print(f"Ensemble manifest: {ensemble_dir / 'ensemble.json'}")
