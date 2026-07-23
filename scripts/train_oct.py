#!/usr/bin/env python3
import argparse

from sam_ml.oct.config import load_config
from sam_ml.oct.training import train_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/oct.yaml")
parser.add_argument("--model", choices=["baseline_resnet50", "improved_resnet50"])
parser.add_argument("--experiment", required=True)
parser.add_argument("--resume")
args = parser.parse_args()
config = load_config(args.config)
if args.model:
    config.model.name = args.model
run, checkpoint = train_experiment(config, args.experiment, args.resume)
print(f"Run: {run}\nBest checkpoint: {checkpoint}")
