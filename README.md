# SAM-AI: clasificación de OCT macular

Proyecto de investigación para clasificar imágenes B-scan de OCT macular en cuatro clases:

| Índice | Clase | Descripción |
|---:|---|---|
| 0 | CNV | Neovascularización coroidea |
| 1 | DME | Edema macular diabético |
| 2 | DRUSEN | Drusas |
| 3 | NORMAL | Retina sin las patologías anteriores |

El flujo OCT adapta la metodología de Li et al. (2019) al dataset público UCSD/Kermany. Incluye:

- Auditoría del dataset y control de calidad.
- Prevención de fuga de pacientes y duplicados exactos.
- Muestreo porcentual reproducible.
- Balanceo moderado físico o lógico.
- Limpieza conservadora de artefactos blancos de borde.
- ResNet50 baseline y ResNet50 con dilatación configurable.
- Entrenamiento reproducible y ensemble de cuatro modelos.
- Evaluación multiclase y binaria.
- Occlusion sensitivity y Grad-CAM.
- Inferencia por línea de comandos.
- Registro de configuración, entorno, checkpoints, predicciones y métricas.

El flujo histórico DDR2019 de fotografías de fondo de ojo permanece disponible para compatibilidad.

> **Advertencia médica:** resultado generado por un modelo experimental de investigación. No
> sustituye la evaluación de un oftalmólogo.

## Contenido

- [Instalación](#instalación)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Configuración](#configuración)
- [Dataset OCT](#dataset-oct)
- [Flujo recomendado](#flujo-recomendado)
- [Auditoría](#1-auditoría)
- [Distribución de clases](#2-distribución-de-clases)
- [Muestreo porcentual](#3-muestreo-porcentual)
- [Balanceo moderado](#4-balanceo-moderado)
- [Creación de splits](#5-creación-de-splits)
- [Preprocesamiento OCT](#6-preprocesamiento-oct)
- [Visualización](#7-visualización)
- [Modelos y entrenamiento](#8-modelos-y-entrenamiento)
- [Ensemble](#9-ensemble)
- [Evaluación](#10-evaluación)
- [Explicabilidad](#11-explicabilidad)
- [Inferencia](#12-inferencia)
- [Artefactos de ejecución](#artefactos-de-ejecución)
- [Pruebas](#pruebas)
- [Compatibilidad DDR2019](#compatibilidad-ddr2019)
- [Limitaciones](#limitaciones)

## Instalación

Requisitos:

- Python 3.12 o superior.
- `uv`.
- CUDA opcional para entrenamiento.

Desde la raíz del repositorio:

```powershell
uv sync
```

Dependencias de pruebas:

```powershell
uv sync --extra test
```

Dependencias opcionales:

```powershell
# Notebooks
uv sync --extra notebook

# FiftyOne
uv sync --extra fiftyone
```

Verificación básica:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Estructura del proyecto

```text
sam-ai/
├── configs/
│   └── oct.yaml
├── data/
│   ├── raw/
│   ├── sampled/
│   ├── balanced/
│   ├── processed/
│   └── manifests/
├── docs/
│   ├── PAPER_ADAPTATION_PLAN.md
│   └── PAPER_REPRODUCTION_NOTES.md
├── sam_ml/
│   ├── oct/
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── data.py
│   │   ├── dataset.py
│   │   ├── dataset_management.py
│   │   ├── preprocessing.py
│   │   ├── models.py
│   │   ├── training.py
│   │   ├── ensemble.py
│   │   ├── metrics.py
│   │   ├── inference.py
│   │   ├── explain.py
│   │   └── cli.py
│   ├── preprocessing/        # Flujo histórico DDR2019
│   ├── datasets/             # Datasets históricos
│   └── modeling/             # Modelos históricos
├── scripts/
├── tests/
├── runs/
└── reports/
```

## Configuración

La configuración OCT está centralizada en [`configs/oct.yaml`](configs/oct.yaml) y validada con
Pydantic.

Secciones principales:

```yaml
data:
  root: data/raw
  manifest_dir: data/manifests
  image_size: 224
  classes: [CNV, DME, DRUSEN, NORMAL]
  patient_level_split: true
  allow_image_level_split: false
  seed: 42

preprocessing:
  enabled: true
  target_size: 224
  white_threshold: 250
  max_aspect_ratio: 2.0
  extreme_aspect_mode: letterbox
  percentile_normalization: false
  clahe: false
  light_denoise: none
  seed: 42

dataset_management:
  seed: 42
  sampling:
    percentage: 10.0
    unit: auto
    mode: copy
  balancing:
    enabled: false
    mode: class-weights
    splits: [train]
    max_ratio: 2.0

model:
  name: improved_resnet50
  pretrained: true
  num_classes: 4
  dropout: 0.2
  replace_stride_with_dilation: [false, true, true]

training:
  seed: 42
  learning_rate: 0.00001
  batch_size: 16
  effective_batch_size: 128
  max_steps: 10000
  mixed_precision: true
  ensemble_size: 4
```

Los argumentos CLI sobrescriben los valores correspondientes cuando el comando ofrece esa opción.

## Dataset OCT

Se acepta la estructura oficial:

```text
data/raw/
├── train/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
├── val/                     # Opcional
└── test/
```

También se acepta una estructura plana:

```text
data/raw/
├── CNV/
├── DME/
├── DRUSEN/
└── NORMAL/
```

El proyecto no descarga el dataset automáticamente. Las imágenes originales nunca se sobrescriben.

### Identificación de pacientes

El parser reconoce nombres Kermany con forma:

```text
CNV-1016042-1.jpeg
```

y obtiene:

```text
patient_id = CNV-1016042
```

No se inventan IDs si el patrón no es fiable. Si no puede conservarse la separación por paciente,
el modo por imagen requiere autorización explícita y queda etiquetado como riesgo de leakage.

## Flujo recomendado

```text
1. Auditar el dataset
2. Contar la distribución
3. Crear una muestra porcentual, si se necesita
4. Volver a contar la distribución
5. Evaluar si train requiere balanceo
6. Crear o validar manifests
7. Preprocesar y revisar controles de calidad
8. Visualizar transformaciones
9. Entrenar baseline e improved ResNet50
10. Entrenar ensemble
11. Evaluar test una sola vez
12. Generar explicaciones e inferencias
```

Todos los comandos largos relacionados con el dataset muestran barras `tqdm` con porcentaje,
velocidad, tiempo transcurrido y ETA. Para CI o logs redirigidos:

```powershell
--no-progress
```

## 1. Auditoría

```powershell
uv run python scripts/audit_dataset.py `
  --config configs/oct.yaml
```

La auditoría reporta:

- Imágenes totales y por clase.
- Pacientes identificables por clase.
- Archivos corruptos.
- Dimensiones y formatos.
- Estimación opcional de desenfoque.
- Duplicados exactos mediante SHA-256.
- Imágenes sospechosas.

Archivos:

```text
data/manifests/quality_report.csv
data/manifests/excluded_images.csv
```

Un umbral de desenfoque es solo una señal de revisión; no elimina imágenes automáticamente.

## 2. Distribución de clases

```powershell
uv run python scripts/dataset_distribution.py `
  --config configs/oct.yaml `
  --input-root data/raw `
  --output-dir reports/dataset_management `
  --report-prefix original
```

El conteo usa rutas y extensiones; no decodifica imágenes. Calcula:

- Conteos por split y clase.
- Porcentajes globales.
- Proporción por split.
- Clases mayoritaria y minoritaria.
- Relación mayoría/minoría.
- Clases vacías o subrepresentadas.
- Advertencias de posible desbalance.

## 3. Muestreo porcentual

Ejemplo para seleccionar aproximadamente 10%:

```powershell
uv run python scripts/sample_dataset.py `
  --config configs/oct.yaml `
  --input-root data/raw `
  --output-root data/sampled/oct_10 `
  --percentage 10 `
  --sampling-unit auto `
  --mode copy `
  --seed 42
```

El muestreo:

- Se realiza dentro de cada combinación split/clase.
- Conserva train, val y test existentes.
- Selecciona pacientes completos cuando sus IDs son fiables.
- Ordena las rutas antes de aplicar la semilla.
- Registra el porcentaje solicitado y el real.
- Rechaza pacientes presentes en varios splits durante muestreo por paciente.
- Conserva al menos una unidad por clase no vacía.

Un porcentaje por paciente puede diferir del solicitado porque todos sus B-scans se mantienen juntos.

### Modos de salida

| Modo | Comportamiento |
|---|---|
| `copy` | Copia física; opción segura en Windows |
| `hardlink` | Enlace duro; origen y destino deben compartir filesystem |
| `symlink` | Enlace simbólico; puede requerir permisos en Windows |
| `manifest` | Solo CSV/JSON, sin copiar imágenes |

No existe fallback silencioso. Las colisiones fallan salvo que se especifique `--overwrite`.

## 4. Balanceo moderado

El modo recomendado genera pesos sin copiar imágenes:

```powershell
uv run python scripts/balance_dataset.py `
  --config configs/oct.yaml `
  --input-root data/sampled/oct_10 `
  --strategy moderate `
  --balance-mode class-weights `
  --max-ratio 2.0 `
  --seed 42
```

Modos:

| Modo | Resultado |
|---|---|
| `class-weights` | `class_weights.json` con pesos moderados |
| `sampler` | Manifest con pesos por muestra |
| `physical` | Nuevo árbol con submuestreo/sobremuestreo controlado |

Los pesos se calculan con la raíz cuadrada de la frecuencia inversa relativa a la mediana, se
recortan y se normalizan cerca de media uno.

Por defecto:

- Solo se balancea `train`.
- No se igualan agresivamente todas las clases.
- No se elimina más de 30% de una clase mayoritaria.
- No se aumenta una clase minoritaria por encima de 1.5 veces.
- Solicitar balanceo de val/test muestra una advertencia.
- No se usa SMOTE sobre píxeles ni embeddings.

Ejemplo físico:

```powershell
uv run python scripts/balance_dataset.py `
  --config configs/oct.yaml `
  --input-root data/sampled/oct_10 `
  --output-root data/balanced/oct_10 `
  --balance-mode physical `
  --seed 42
```

## 5. Creación de splits

```powershell
uv run python scripts/create_splits.py `
  --config configs/oct.yaml
```

Genera:

```text
data/manifests/train.csv
data/manifests/val.csv
data/manifests/test.csv
```

Columnas:

```text
image_path, label, class_index, patient_id, source, split
```

Reglas:

- Si existe un test oficial, se conserva.
- Validation se obtiene únicamente de train.
- Ningún paciente puede aparecer en varios splits.
- Los hashes exactos tampoco pueden cruzar splits.
- Test no participa en selección de hiperparámetros.

## 6. Preprocesamiento OCT

```powershell
uv run python scripts/preprocess_oct.py `
  --config configs/oct.yaml `
  --input-root data/raw `
  --output-root data/processed/oct
```

Pipeline:

```text
Lectura en gris 0–255
→ detección de componentes casi blancos conectados al borde
→ estimación del fondo oscuro y granular
→ relleno texturizado determinista
→ recorte conservador de márgenes vacíos
→ control de panorámicas
→ padding cuadrado con textura
→ resize a 224×224
```

La detección no elimina todos los píxeles brillantes. Solo acepta componentes casi blancos:

- Conectados a un borde.
- De tamaño suficiente.
- Concentrados en esquinas o márgenes.
- Con alta proporción de blanco casi puro.

Las estructuras hiperreflectivas internas no se seleccionan solo por intensidad.

El relleno combina inpainting local con textura obtenida de la misma imagen. La semilla se deriva
de la configuración y del contenido, por lo que es reproducible.

Las panorámicas superiores a `max_aspect_ratio` se conservan completas mediante letterboxing
texturizado y se marcan para revisión.

Opciones desactivadas por defecto:

- Normalización robusta por percentiles.
- CLAHE suave.
- Filtro mediano o bilateral ligero.

Los resultados conservan la estructura split/clase y nunca reemplazan los originales.

### Control de calidad

```text
reports/oct_preprocessing/
├── preprocessing_report.csv
├── preprocessing_report.json
├── original/
├── mask/
├── corrected/
├── padded/
└── final/
```

El reporte registra dimensiones, porcentaje corregido, recorte, padding, relación de aspecto,
advertencias, errores y estadísticas del fondo.

## 7. Visualización

```powershell
uv run python scripts/visualize_transforms.py `
  --config configs/oct.yaml `
  --count 8 `
  --output reports/transform_samples.png
```

Permite revisar muestras antes y después de los transforms de entrenamiento.

## 8. Modelos y entrenamiento

Modelos OCT:

- `baseline_resnet50`: ResNet50 estándar preentrenada en ImageNet.
- `improved_resnet50`: ResNet50 con `replace_stride_with_dilation` configurable.

Ambos producen cuatro logits. Softmax se aplica únicamente durante inferencia o ensemble, nunca
antes de `CrossEntropyLoss`.

### Baseline

```powershell
uv run python scripts/train_oct.py `
  --config configs/oct.yaml `
  --model baseline_resnet50 `
  --experiment baseline
```

### Modelo mejorado

```powershell
uv run python scripts/train_oct.py `
  --config configs/oct.yaml `
  --model improved_resnet50 `
  --experiment improved
```

Entrenamiento:

- AdamW.
- Learning rate inicial `1e-5`.
- Cross-entropy o focal loss configurable.
- Mixed precision cuando CUDA está disponible.
- Gradient accumulation para batch efectivo.
- Early stopping.
- ReduceLROnPlateau.
- Checkpoint del mejor `val_loss`.
- Semilla global.
- Reanudación desde checkpoint.

```powershell
uv run python scripts/train_oct.py `
  --config configs/oct.yaml `
  --model improved_resnet50 `
  --experiment improved `
  --resume runs/improved/checkpoints/last.ckpt
```

El optimizador y la loss son decisiones documentadas porque el paper no las especifica con
suficiente precisión.

## 9. Ensemble

```powershell
uv run python scripts/train_ensemble.py `
  --config configs/oct.yaml `
  --ensemble-size 4 `
  --experiment paper_ensemble
```

Cada miembro:

- Usa una semilla distinta.
- Conserva el mismo pipeline científico.
- Guarda su mejor checkpoint.

La predicción final es:

```text
P_final = (P_1 + P_2 + P_3 + P_4) / 4
```

La inferencia puede ejecutar modelos secuencialmente, acumulando probabilidades en CPU para reducir
memoria GPU. También calcula desviación estándar y entropía como indicadores de incertidumbre.

## 10. Evaluación

```powershell
uv run python scripts/evaluate.py `
  --run runs/paper_ensemble `
  --split test `
  --config configs/oct.yaml
```

Métricas:

- Accuracy y balanced accuracy.
- Precision, recall/sensitivity, specificity y F1 por clase.
- Macro precision, macro recall, macro F1 y weighted F1.
- Cohen's kappa.
- Log loss.
- Matriz de confusión y normalizada.
- ROC-AUC one-vs-rest, macro y micro.
- Average precision.
- Intervalos bootstrap al 95%.

Análisis binarios:

- ABNORMAL vs NORMAL.
- CNV vs NORMAL.
- DME vs NORMAL.
- DRUSEN vs NORMAL.

Las predicciones se guardan para análisis de errores, incluyendo probabilidades, confianza, margen
e incertidumbre del ensemble.

## 11. Explicabilidad

```powershell
uv run python scripts/explain_predictions.py `
  --run runs/paper_ensemble `
  --image "ruta/a/oct.jpeg" `
  --config configs/oct.yaml
```

Métodos:

- Occlusion sensitivity con ventana predeterminada de 28×28.
- Grad-CAM como método complementario, separado de la reproducción del paper.

Se guardan imagen original, mapa de importancia y superposición.

## 12. Inferencia

Un modelo:

```powershell
uv run python scripts/predict.py `
  --image "ruta/a/oct.jpeg" `
  --checkpoint "runs/improved/checkpoints/best.ckpt" `
  --config configs/oct.yaml
```

Ensemble:

```powershell
uv run python scripts/predict.py `
  --image "ruta/a/oct.jpeg" `
  --checkpoint "runs/paper_ensemble/member_1/checkpoints/best.ckpt" `
  --checkpoint "runs/paper_ensemble/member_2/checkpoints/best.ckpt" `
  --checkpoint "runs/paper_ensemble/member_3/checkpoints/best.ckpt" `
  --checkpoint "runs/paper_ensemble/member_4/checkpoints/best.ckpt" `
  --config configs/oct.yaml
```

Respuesta:

```json
{
  "prediction": "DME",
  "display_name": "Edema macular diabético",
  "probabilities": {
    "CNV": 0.01,
    "DME": 0.94,
    "DRUSEN": 0.02,
    "NORMAL": 0.03
  },
  "confidence": 0.94,
  "uncertainty": 0.02,
  "model_type": "ensemble",
  "ensemble_size": 4,
  "disclaimer": "Resultado generado por un modelo experimental de investigación."
}
```

## Artefactos de ejecución

Cada entrenamiento crea:

```text
runs/<experiment_name>/
├── config.yaml
├── environment.json
├── training_history.csv
├── dataset_summary.json
├── checkpoints/
├── predictions/
├── figures/
├── metrics.json
└── logs/
```

Se registran:

- Semilla.
- Versiones de Python y librerías.
- Plataforma y GPU.
- Parámetros.
- Fecha.
- Commit Git, cuando está disponible.
- Hashes de manifests.

Reportes de gestión:

```text
reports/dataset_management/
├── original_distribution.csv
├── original_distribution.json
├── sampled_distribution.csv
├── sampled_distribution.json
├── balanced_distribution.csv
├── balanced_distribution.json
├── sample_manifest.csv
├── sample_manifest.json
├── balanced_manifest.csv
└── class_weights.json
```

## Pruebas

Suite completa:

```powershell
uv run --extra test pytest
```

Cobertura:

```powershell
uv run --extra test pytest --cov=sam_ml --cov-report=html
```

Estado verificado más reciente:

```text
171 passed
```

Las pruebas incluyen:

- Configuración y registros históricos.
- Preprocesamiento DDR2019.
- Datasets y modelos históricos.
- Mapeo de clases OCT.
- Splits y prevención de leakage.
- ResNet50 y ensemble.
- Métricas e inferencia en CPU.
- Artefactos blancos, panorámicas y archivos corruptos.
- Barras de progreso y modo silencioso.
- Distribución, muestreo reproducible y pacientes completos.
- Copy/manifest y seguridad de rutas.
- Balanceo moderado y preservación de val/test.

Las pruebas OCT usan imágenes sintéticas; no representan rendimiento clínico.

## Compatibilidad DDR2019

El flujo histórico continúa disponible:

```powershell
uv run preprocess-dataset ddr2019
uv run preprocess-dataset ddr2019_dualfilters
uv run train-model --model simple_cnn --dataset ddr2019
uv run train-model --model dual_channel --dataset ddr2019_dualfilters
```

Documentación histórica:

- [`docs/preprocessing.md`](docs/preprocessing.md)
- [`docs/modeling.md`](docs/modeling.md)
- [`docs/creating-models.md`](docs/creating-models.md)
- [`docs/configuration.md`](docs/configuration.md)
- [`docs/testing.md`](docs/testing.md)

## Limitaciones

- El dataset hospitalario privado del paper no está disponible.
- UCSD/Kermany no reproduce exactamente población, equipos ni prevalencias del estudio.
- La topología exacta de dilatación y el optimizador original no están completamente especificados.
- Los crops automáticos conservadores no equivalen a la revisión manual del paper.
- El parser de pacientes depende del patrón de nombres; debe auditarse en cada distribución.
- SHA-256 detecta duplicados exactos, no duplicados perceptuales o recodificados.
- El modo image-level implica riesgo de fuga si existen varios B-scans por paciente.
- El balanceo lógico genera recomendaciones/artefactos; no cambia experimentos existentes por defecto.
- El modo `macular_center` no dispone de localización macular clínicamente validada.
- Las panorámicas se conservan mediante letterboxing y deben analizarse como posible sesgo.
- Las máscaras de artefactos requieren revisión sobre imágenes reales de cada equipo.
- No se han generado ni deben inventarse métricas clínicas sin entrenamiento y evaluación reales.

## Reproducibilidad del paper

- [`docs/PAPER_ADAPTATION_PLAN.md`](docs/PAPER_ADAPTATION_PLAN.md)
- [`docs/PAPER_REPRODUCTION_NOTES.md`](docs/PAPER_REPRODUCTION_NOTES.md)

Referencia principal:

F. Li et al., “Deep learning-based automated detection of retinal diseases using optical coherence
tomography images,” *Biomedical Optics Express*, 10(12), 6204–6226, 2019.
DOI: `10.1364/BOE.10.006204`.

