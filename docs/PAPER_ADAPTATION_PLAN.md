# OCT paper adaptation plan

## Scope

This project will gain a new, self-contained OCT classification workflow for CNV, DME,
DRUSEN, and NORMAL based on Li et al. (2019), while the existing DDR2019 fundus workflow
remains available for backward compatibility. OCT commands, datasets, models, metrics, and
run artifacts must not depend on the five-class diabetic-retinopathy code path.

The public UCSD/Kermany dataset is not present in this checkout. Infrastructure will therefore
be verified with explicitly synthetic fixtures; no clinical performance will be claimed.

## Current architecture

- Python 3.12 package built with `uv_build` and managed by `uv`.
- Pydantic Settings supplies defaults for preprocessing, training, and models.
- Registry-driven image preprocessors and PyTorch Lightning models.
- DDR2019 loaders read CSV manifests and produce single-image or dual-filter tensors.
- `train-model` constructs datasets, callbacks, CSV logging, and a Lightning trainer.
- FiftyOne optionally stores visualization metadata in a remote MongoDB database.
- There is no web application or inference API; `sam_ml/modeling/predict.py` is empty.
- Baseline before adaptation: 137 tests passed on Python 3.12.12.

## Paper versus this repository

| Topic | Existing project | Li et al. methodology / adaptation |
|---|---|---|
| Modality | Color fundus photographs | Macular OCT B-scans |
| Labels | Five DDR grades | CNV, DME, DRUSEN, NORMAL |
| Input | Usually 512 x 512 | 224 x 224 |
| Model | Simple CNN and dual Inception/VGG | ImageNet ResNet50 and dilated ResNet50 |
| Validation | Image-level 80/20 split | Participant-level holdout and grouped CV |
| Ensemble | None | Mean probability from four independent models |
| Explainability | None | 28 x 28 occlusion; Grad-CAM as an addition |
| Evaluation | Loss and accuracy | Multiclass, binary, ROC/PR, calibration-ready metrics and CIs |
| Reproducibility | Checkpoints and CSV logs | Immutable manifests plus complete run metadata |

## Explicit paper facts

The paper specifies ImageNet initialization, 224 x 224 inputs, horizontal mirroring, random
cropping, a four-model probability ensemble, participant-level splitting, learning rate 1e-5,
batch size 200, 10,000 steps, and occlusion analysis. It describes dilated convolution but does
not provide enough implementation detail to reconstruct every stride/dilation choice uniquely.
It also does not clearly specify the optimizer or loss implementation.

## Reproduction assumptions

- `torchvision.models.resnet50(replace_stride_with_dilation=...)` is the compatibility-preserving
  interpretation of the improved model. Configuration records the exact dilation tuple.
- AdamW and cross-entropy are implementation decisions, not claims about the paper.
- Conservative `RandomResizedCrop` substitutes for manually reviewed crops.
- Kermany filenames are parsed only when a defensible patient token is present. Image-level
  splitting requires an explicit opt-in and is labelled as a leakage risk.
- An official test directory is immutable; validation is derived only from official training.
- Test results never select checkpoints or hyperparameters.
- Ensemble members run sequentially and accumulate probabilities on CPU.

## Files to add

- `configs/oct.yaml`
- `sam_ml/oct/`: configuration, discovery/audit, manifests, dataset/transforms, models,
  training utilities, ensemble, metrics, inference, and explainability.
- `scripts/audit_dataset.py`, `scripts/create_splits.py`, `scripts/train_oct.py`,
  `scripts/train_ensemble.py`, `scripts/evaluate.py`, `scripts/predict.py`, and
  `scripts/explain_predictions.py`.
- OCT-focused unit and integration tests using synthetic images.
- `docs/PAPER_REPRODUCTION_NOTES.md`.

## Files to modify

- `pyproject.toml`: direct dependencies and OCT command entry points.
- `.gitignore`: manifests containing local paths, runs, reports, and model artifacts.
- `README.md`: OCT setup, commands, methodology, limitations, and research disclaimer.
- `sam_ml/modeling/models/__init__.py`: only if shared model discovery is useful; OCT models
  otherwise remain isolated under `sam_ml.oct`.

## Implementation stages

1. Add typed YAML configuration and fixed class mapping.
2. Discover official or flat directory layouts; audit corruption, dimensions, blur, and SHA-256.
3. Generate manifests with official-test preservation and strict group-leakage validation.
4. Add deterministic evaluation transforms and conservative training augmentation.
5. Add baseline and improved ResNet50 implementations without downloading weights in tests.
6. Add reproducible Lightning training, resumable checkpoints, run provenance, and balancing modes.
7. Add sequential ensemble prediction and complete metrics/error analysis.
8. Add occlusion sensitivity, Grad-CAM, CLI inference, and the medical disclaimer.
9. Add synthetic tests, run the complete suite, and document actual verification results.

## Technical risks

- Public Kermany releases often encode case information inconsistently; patient-level extraction
  must not silently manufacture groups.
- Official train/test layouts may contain duplicate patients or exact image duplicates.
- Dilation increases feature-map memory and may make the paper batch size infeasible.
- ImageNet weights require network/cache access; commands must permit `pretrained: false`.
- A 10-fold grouped split is impossible when any class has fewer than ten patient groups.
- Occlusion over every test image is computationally expensive.
- Dataset-specific prevalence means paper metrics cannot be expected on Kermany.
- Bootstrap confidence intervals are invalid if images from one participant are treated as
  independent; grouped bootstrap is preferred when reliable patient identifiers exist.

## Acceptance and evidence

Completion means infrastructure and tests pass, not that clinical accuracy has been reproduced.
Real training/evaluation remains pending until the dataset is supplied. Every generated result
must identify whether its split is patient-level or the explicit image-level fallback.
