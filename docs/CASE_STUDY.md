# SAM AI | Diabetic retinopathy and OCT

[Español](#español) · [English](#english)

## Español

### El problema

Las imágenes de fondo de ojo y los cortes OCT requieren preparación, modelos y evaluaciones distintos. El proyecto conserva el flujo original de retinopatía diabética en DDR2019 y añade un flujo separado de clasificación OCT en el mismo repositorio.

### Mi trabajo

El código incluye carga y preprocesamiento de imágenes, preparación de conjuntos de datos, entrenamiento, evaluación y herramientas de explicabilidad. El flujo OCT dispone de configuración propia en [`configs/oct.yaml`](../configs/oct.yaml), comandos en [`scripts/`](../scripts/) y pruebas en [`tests/`](../tests/). El trabajo original de DDR2019 permanece en [`sam_ml/datasets/`](../sam_ml/datasets/) y [`sam_ml/modeling/`](../sam_ml/modeling/).

### Cómo revisarlo

1. Consulta el [README](../README.md) para instalar dependencias y ejecutar los flujos.
2. Revisa la [preparación de OCT](../sam_ml/oct/preparation.py) y la [evaluación](../scripts/evaluate.py).
3. Ejecuta `uv sync --extra test` y `uv run pytest -m "not slow and not gpu" tests` para comprobar el código sin un conjunto clínico.

El repositorio contiene el **código para calcular** métricas, matriz de confusión y ROC-AUC, pero este caso de estudio todavía no publica un resultado numérico verificable ni pesos entrenados. No se presenta como herramienta de diagnóstico clínico.

## English

### The problem

Fundus photographs and OCT B-scans require different preparation, models and evaluation workflows. This repository preserves the original DDR2019 diabetic retinopathy work and adds a separate OCT classification pipeline.

### What I built

The code covers image loading and preprocessing, dataset preparation, training, evaluation and explainability. OCT has its own [`configs/oct.yaml`](../configs/oct.yaml), command scripts in [`scripts/`](../scripts/) and tests in [`tests/`](../tests/). The earlier DDR2019 work remains under [`sam_ml/datasets/`](../sam_ml/datasets/) and [`sam_ml/modeling/`](../sam_ml/modeling/).

### How to inspect it

1. Use the [README](../README.md) for installation and workflow commands.
2. Inspect [OCT preparation](../sam_ml/oct/preparation.py) and [evaluation](../scripts/evaluate.py).
3. Run `uv sync --extra test` and `uv run pytest -m "not slow and not gpu" tests` to exercise the code without clinical data.

The repository includes code for classification metrics, a confusion matrix and ROC-AUC, but this case study does not yet publish verified numeric results or trained weights. It is not presented as a clinical diagnostic tool.
