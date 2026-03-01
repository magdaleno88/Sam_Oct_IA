import json
from pathlib import Path


NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "Avance4_Equipo21_ConvNeXt-Tiny.ipynb"


def to_source(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def find_cell(cells: list[dict], marker: str) -> dict:
    for cell in cells:
        if marker in "".join(cell.get("source", [])):
            return cell
    raise RuntimeError(f"Cell not found for marker: {marker}")


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Expected snippet not found:\n{old}")
    return text.replace(old, new, 1)


def clear_code_cell(cell: dict) -> None:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]

    import_cell = find_cell(cells, "from torch.utils.data import DataLoader, Dataset")
    import_src = "".join(import_cell["source"])
    import_src = replace_once(
        import_src,
        "from torch.utils.data import DataLoader, Dataset\n",
        "from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler\n",
    )
    import_cell["source"] = to_source(import_src)
    clear_code_cell(import_cell)

    dataset_cell = find_cell(cells, "class BaselineSingleFeatureDataset(Dataset):")
    dataset_src = "".join(dataset_cell["source"])
    dataset_src = replace_once(
        dataset_src,
        "        hist_bins: int = 16,\n        train_mode: bool = False,\n    ) -> None:\n",
        (
            "        hist_bins: int = 16,\n"
            "        train_mode: bool = False,\n"
            "        train_aug_modes: list[int] | None = None,\n"
            "    ) -> None:\n"
        ),
    )
    dataset_src = replace_once(
        dataset_src,
        "        self.train_mode = bool(train_mode)\n\n        self.mean, self.std = _imagenet_stats_from_convnext_tiny()\n",
        (
            "        self.train_mode = bool(train_mode)\n"
            "        self.train_aug_modes = [int(x) for x in (train_aug_modes if train_aug_modes is not None else [0])]\n"
            "        self._rng = random.Random(self.seed)\n\n"
            "        self.mean, self.std = _imagenet_stats_from_convnext_tiny()\n"
        ),
    )
    dataset_src = replace_once(
        dataset_src,
        (
            "        if self.train_mode:\n"
            "            image_aug = _apply_geometry(\n"
            "                image,\n"
            "                aug_id=int(aug_id),\n"
            "                idx=int(base_index),\n"
            "                seed=self.seed,\n"
            "            )\n"
            "        else:\n"
            "            image_aug = image\n"
        ),
        (
            "        if self.train_mode:\n"
            "            aug_id = int(aug_id)\n"
            "            if aug_id < 0:\n"
            "                aug_id = int(self._rng.choice(self.train_aug_modes))\n"
            "            image_aug = _apply_geometry(\n"
            "                image,\n"
            "                aug_id=aug_id,\n"
            "                idx=int(base_index),\n"
            "                seed=self.seed,\n"
            "            )\n"
            "        else:\n"
            "            image_aug = image\n"
        ),
    )
    dataset_cell["source"] = to_source(dataset_src)
    clear_code_cell(dataset_cell)

    cell25 = find_cell(cells, "# 2) Smart class balancing for training (after 5->3 merge).")
    cell25["source"] = to_source(
        """# 2) Sampling + augmentation strategy for training (after 5->3 merge).
# Cambios implementados:
# - Sin duplicacion fija de filas en train.
# - WeightedRandomSampler para balancear por epoca.
# - Augmentacion estocastica sobre las muestras originales.
train_rows_baseline = baseline_train_ds._rows.reset_index(drop=True).copy()
class_counts_baseline = train_rows_baseline["label"].value_counts().sort_index()

IMAGE_SIZE = 320
BATCH_SIZE = 16
HIST_BINS = 16
AUGMENTATION_MODES = [0, 1, 2, 3, 4, 5, 6, 7]
SAMPLER_WEIGHT_EXPONENT = 0.5

majority_class_baseline = int(class_counts_baseline.idxmax())

train_samples = [(int(i), -1) for i in range(len(baseline_train_ds))]
train_eval_samples = [(int(i), 0) for i in range(len(baseline_train_ds))]
val_samples = [(int(i), 0) for i in range(len(baseline_val_ds))]

sampler_class_weights = (
    (class_counts_baseline.max() / class_counts_baseline.astype(np.float64)) ** SAMPLER_WEIGHT_EXPONENT
).sort_index()
sampler_weight_map = {int(k): float(v) for k, v in sampler_class_weights.items()}
train_sample_weights = train_rows_baseline["label"].map(sampler_weight_map).to_numpy(dtype=np.float64)

expected_sampler_mass = (
    pd.Series(train_sample_weights, index=train_rows_baseline.index)
    .groupby(train_rows_baseline["label"])
    .sum()
    .sort_index()
)
expected_sampler_proportions = expected_sampler_mass / expected_sampler_mass.sum()

train_sampler = WeightedRandomSampler(
    weights=torch.as_tensor(train_sample_weights, dtype=torch.double),
    num_samples=len(train_sample_weights),
    replacement=True,
    generator=torch.Generator().manual_seed(SEED),
)

print("Training-set sampling summary (after merge, before model)")
print(f"  merged class_counts (raw train split): {class_counts_baseline.to_dict()}")
print(f"  majority_class: {majority_class_baseline}")
print(f"  image_size: {IMAGE_SIZE}")
print(f"  batch_size: {BATCH_SIZE}")
print(f"  augmentation_modes: {AUGMENTATION_MODES}")
print(f"  sampler_weight_exponent: {SAMPLER_WEIGHT_EXPONENT}")
print(f"  sampler_class_weights: {sampler_weight_map}")
print(
    "  expected sampler proportions (%): "
    + str({int(k): round(float(v) * 100.0, 2) for k, v in expected_sampler_proportions.items()})
)

train_fe_ds = BaselineSingleFeatureDataset(
    base_dataset=baseline_train_ds,
    samples=train_samples,
    seed=SEED,
    image_size=IMAGE_SIZE,
    hist_bins=HIST_BINS,
    train_mode=True,
    train_aug_modes=AUGMENTATION_MODES,
)
train_eval_ds = BaselineSingleFeatureDataset(
    base_dataset=baseline_train_ds,
    samples=train_eval_samples,
    seed=SEED,
    image_size=IMAGE_SIZE,
    hist_bins=HIST_BINS,
    train_mode=False,
)
val_fe_ds = BaselineSingleFeatureDataset(
    base_dataset=baseline_val_ds,
    samples=val_samples,
    seed=SEED,
    image_size=IMAGE_SIZE,
    hist_bins=HIST_BINS,
    train_mode=False,
)

num_workers = 0
train_loader = DataLoader(
    train_fe_ds,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)
train_eval_loader = DataLoader(
    train_eval_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_fe_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)
"""
    )
    clear_code_cell(cell25)

    cell26 = find_cell(cells, "Proporción final (merge + augmentación) que entra al modelo")
    cell26["source"] = to_source(
        """# Distribucion esperada del muestreo por clase
fig, ax = plt.subplots(figsize=(8, 4))

bars = ax.bar(
    expected_sampler_proportions.index.astype(str),
    expected_sampler_proportions.values,
    color=["#4C78A8", "#F58518", "#54A24B"],
    edgecolor="black",
    alpha=0.85,
)

for bar, prop in zip(bars, expected_sampler_proportions.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{prop * 100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
    )

ax.set_title("Proporcion esperada por clase con WeightedRandomSampler")
ax.set_xlabel("Clase fusionada")
ax.set_ylabel("Proporcion")
ax.set_ylim(0, min(1.0, float(expected_sampler_proportions.max()) + 0.08))
ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout()
plt.show()
"""
    )
    clear_code_cell(cell26)

    cell27 = find_cell(cells, "# 3) Configuración de entrenamiento (ConvNeXt-Tiny + cabeza lineal).")
    cell27["source"] = to_source(
        """
# 3) Configuracion de entrenamiento (ConvNeXt-Tiny + cabeza lineal).
TRAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo para entrenamiento (baseline): {TRAIN_DEVICE}")

if TRAIN_DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")

num_classes = int(len(sorted(baseline_train_ds._rows["label"].unique().tolist())))
extra_dim = int(train_fe_ds[0]["extra"].shape[0])

# Pesos de clase calculados sobre el split real de entrenamiento.
train_labels_for_weights = train_rows_baseline["label"].to_numpy(dtype=np.int64)
counts = np.bincount(train_labels_for_weights, minlength=num_classes).astype(np.float32)
class_weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=TRAIN_DEVICE)

print("Class counts (raw train split):", counts.astype(int).tolist())
print("Class weights:", class_weights.tolist())
print("Num classes:", num_classes)
print("Extra feature dim:", extra_dim)
"""
    )
    clear_code_cell(cell27)

    cell29 = find_cell(cells, "Baseline ingenuo (clase mayoritaria)")
    cell29["source"] = to_source(
        """# Baseline ingenuo (clase mayoritaria) usando la distribucion real de train.
majority_class = majority_class_baseline
y_val_true = baseline_val_ds._rows["label"].to_numpy(dtype=int)

y_val_pred_naive = np.full(shape=len(y_val_true), fill_value=majority_class, dtype=int)

naive_f1_macro = f1_score(y_val_true, y_val_pred_naive, average="macro", zero_division=0)
naive_acc = accuracy_score(y_val_true, y_val_pred_naive)

print("Baseline ingenuo (clase mayoritaria)")
print(f"Clase mayoritaria: {majority_class}")
print(f"Validacion macro-F1: {naive_f1_macro:.4f}")
print(f"Validacion accuracy: {naive_acc:.4f}")
"""
    )
    clear_code_cell(cell29)

    cell30 = find_cell(cells, "### Entrenamiento del clasificador (cabeza lineal + fine-tuning parcial)")
    cell30["source"] = to_source(
        """### Entrenamiento del clasificador (cabeza lineal + fine-tuning parcial)

En esta etapa entrenamos una **cabeza lineal** sobre embeddings de **ConvNeXt-Tiny** con **fine-tuning parcial** del backbone (ultimos bloques).

#### Estrategia de entrenamiento actualizada
- Se conserva el split original de entrenamiento.
- Se usa **`WeightedRandomSampler`** para balancear por epoca sin duplicar filas de forma permanente.
- La **augmentacion geometrica/fotometrica** se aplica de forma estocastica sobre las muestras originales.
- La evaluacion de entrenamiento se hace sobre un loader limpio, sin sampler ni augmentacion.

#### Funcion de perdida (ajuste sistematico)
Se prueban de forma sistematica varias opciones:
- **`weighted_ce`** con pesos por frecuencia inversa o por effective number.
- **`focal`** con `gamma` y pesos por clase.

Se selecciona automaticamente la configuracion con mejor **macro-F1 de validacion** antes de entrenar el ensemble final.
"""
    )

    cell31 = find_cell(cells, "### Hiperparámetros de entrenamiento")
    cell31["source"] = to_source(
        """### Hiperparametros de entrenamiento

- **`IMAGE_SIZE`**: resolucion de entrada aumentada para conservar lesion pequena.
- **`BATCH_SIZE`**: reducido para compensar el mayor costo de memoria.
- **`LOSS_CANDIDATES`**: busqueda sistematica de perdida para desbalance.
- **`LOSS_SEARCH_EPOCHS`**: busqueda mas larga para elegir mejor perdida.
- **`FINAL_EPOCHS`**: epocas finales por seed en el ensemble.
- **`UNFREEZE_BACKBONE_BLOCKS`**: numero de bloques finales de ConvNeXt-Tiny a entrenar.
- **`HEAD_LR` / `BACKBONE_LR`**: LR diferencial para cabeza y backbone.
- **`SAMPLER_WEIGHT_EXPONENT`**: intensidad del rebalanceo del sampler.
- **`TTA_MODES`**: se deja en `0` mientras TTA no mejore macro-F1.
- **`ENSEMBLE_SEEDS`**: semillas usadas para el ensamble final.

Se selecciona el mejor checkpoint por **macro-F1 de validacion** en cada seed.
"""
    )

    cell32 = find_cell(cells, "LOSS_SEARCH_EPOCHS = 4")
    cell32["source"] = to_source(
        """LOSS_SEARCH_EPOCHS = 8
FINAL_EPOCHS = 12

HEAD_LR = 5e-4
BACKBONE_LR = 3e-5
WEIGHT_DECAY = 5e-4
HEAD_DROPOUT = 0.40
LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 3

UNFREEZE_BACKBONE_BLOCKS = 3  # fine-tuning parcial mas profundo

# Inferencia sin TTA mientras macro-F1 no mejore con flips.
TTA_MODES = [0]

# Ensemble por semillas
ENSEMBLE_SEEDS = [SEED, SEED + 11, SEED + 29]

# Ajuste sistematico de perdida para desbalance
LOSS_CANDIDATES = [
    {"name": "weighted_ce_inv", "type": "weighted_ce", "weight_scheme": "inverse_freq"},
    {"name": "weighted_ce_eff", "type": "weighted_ce", "weight_scheme": "effective_num", "beta": 0.999},
    {"name": "focal_inv_g1.5", "type": "focal", "weight_scheme": "inverse_freq", "gamma": 1.5},
    {"name": "focal_eff_g2.0", "type": "focal", "weight_scheme": "effective_num", "beta": 0.999, "gamma": 2.0},
    {"name": "focal_eff_g2.5", "type": "focal", "weight_scheme": "effective_num", "beta": 0.999, "gamma": 2.5},
]
"""
    )
    clear_code_cell(cell32)

    cell36 = find_cell(cells, "### Evaluación del modelo ajustado")
    cell36["source"] = to_source(
        """### Evaluacion del modelo ajustado

La interpretacion se hace sobre este contexto:

- clases fusionadas en 3 niveles de severidad,
- muestreo balanceado por epoca con `WeightedRandomSampler`,
- augmentacion estocastica sin duplicacion fija de train,
- fine-tuning parcial mas profundo de ConvNeXt-Tiny,
- ajuste sistematico de la perdida para desbalance,
- inferencia sin TTA por defecto y ensemble de seeds como referencia secundaria.

Se usa **macro-F1** como metrica principal y accuracy como complemento.

La lectura de resultados se hace en este orden:
1. comparacion `single` vs `ensemble`,
2. desempeno por clase (`classification_report`),
3. patron de error (`confusion_matrix`).
"""
    )

    cell37 = find_cell(cells, "# 5) Validation evaluation: single model vs TTA vs ensemble+TTA.")
    cell37["source"] = to_source(
        """# 5) Validation evaluation: single model vs ensemble, with current inference setup.
inference_modes = TTA_MODES if TTA_MODES else [0]

y_val_single, probs_val_single = predict_proba(final_model, val_loader, TRAIN_DEVICE, tta_modes=inference_modes)
y_val_ens, probs_val_ens = ensemble_predict_proba(ensemble_models, val_loader, TRAIN_DEVICE, tta_modes=inference_modes)

y_val_pred_single = np.argmax(probs_val_single, axis=1)
y_val_pred_ens = np.argmax(probs_val_ens, axis=1)

val_f1_single = f1_score(y_val_single, y_val_pred_single, average="macro", zero_division=0)
val_f1_ens = f1_score(y_val_ens, y_val_pred_ens, average="macro", zero_division=0)

val_ce_single = cross_entropy_from_probs(y_val_single, probs_val_single)
val_ce_ens = cross_entropy_from_probs(y_val_ens, probs_val_ens)

print("Validation macro-F1 (ConvNeXt-Tiny + linear head)")
print(f"Inference modes used: {inference_modes}")
print(f"Single model:         {val_f1_single:.4f} | CE={val_ce_single:.4f}")
print(f"Ensemble seeds:       {val_f1_ens:.4f} | CE={val_ce_ens:.4f}")

print()
print("Modelo final reportado (single model, inferencia actual) - classification report")
print(classification_report(y_val_single, y_val_pred_single, digits=4, zero_division=0))

y_val = y_val_single
y_pred = y_val_pred_single
val_ce = val_ce_single
val_f1_macro = val_f1_single
"""
    )
    clear_code_cell(cell37)

    cell38 = find_cell(cells, "### Señal de generalización")
    cell38["source"] = to_source(
        """### Senal de generalizacion

Ademas de validacion, se compara entrenamiento vs validacion usando:
- un `train_eval_loader` limpio, sin sampler y sin augmentacion,
- el mismo esquema de inferencia configurado arriba (sin TTA por defecto).

Esto evita sobreestimar el F1 de entrenamiento por medir sobre muestras reamostradas o aumentadas.
"""
    )

    cell39 = find_cell(cells, 'print("Comparación de generalización (single model sin TTA, macro-F1)")')
    cell39["source"] = to_source(
        """y_train, probs_train_single = predict_proba(
    final_model,
    train_eval_loader,
    TRAIN_DEVICE,
    tta_modes=TTA_MODES,
)
y_train_pred = np.argmax(probs_train_single, axis=1)
train_ce = cross_entropy_from_probs(y_train, probs_train_single)

train_f1_macro = f1_score(y_train, y_train_pred, average="macro", zero_division=0)
val_f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)

print("Comparacion de generalizacion (single model, train limpio)")
print(f"Entrenamiento macro-F1: {train_f1_macro:.4f}")
print(f"Validacion   macro-F1: {val_f1_macro:.4f}")
print(f"Brecha (train - val): {(train_f1_macro - val_f1_macro):.4f}")
print(f"Train CE (referencia): {train_ce:.4f}")
"""
    )
    clear_code_cell(cell39)

    cell44 = find_cell(cells, "## 11) Conclusiones")
    cell44["source"] = to_source(
        """## 11) Conclusiones

### Que arquitectura usamos como baseline?
El baseline documentado en esta version del notebook es:
- **Backbone CNN**: `ConvNeXt-Tiny` con fine-tuning parcial de mas bloques.
- **Clasificador**: cabeza lineal entrenable con mayor regularizacion.
- **Balanceo**: `WeightedRandomSampler` por epoca sobre el split real de train.
- **Entrada**: resolucion mayor (`IMAGE_SIZE = 320`) para preservar detalle fino.
- **Funcion de perdida**: busqueda entre `Weighted CE` y `Focal`.

### Que cambio respecto a la version anterior?
1. Se elimino el upsampling fijo de filas en train y se reemplazo por muestreo balanceado por epoca.
2. Se separo un loader limpio para medir generalizacion sin augmentacion de entrenamiento.
3. Se aumento la resolucion de entrada y se ajustaron LR, dropout, weight decay y profundidad de fine-tuning.
4. Se desactivo TTA por defecto mientras no aporte mejora en macro-F1.

### Como cerrar esta seccion despues de rerun?
Tras volver a ejecutar las celdas de entrenamiento y evaluacion:
- actualiza aqui el **macro-F1 / accuracy / CE** del baseline ingenuo y del mejor modelo,
- reporta la metrica principal del modelo individual y del ensemble,
- revisa si la **clase 2** mejora en recall,
- usa la brecha train-vs-val del loader limpio como senal principal de sobreajuste.

### Conclusion operativa
Esta version del baseline esta preparada para una comparacion mas justa y mas robusta:
- menos riesgo de memorizar muestras duplicadas,
- mejor sensibilidad potencial a lesiones pequenas por mayor resolucion,
- medicion de generalizacion mas limpia,
- hiperparametros mas consistentes con el nivel de sobreajuste observado.
"""
    )

    for marker in [
        "Buscando mejor pérdida con seed=",
        "Mejor val macro-F1 (seed individual):",
        'plt.title("Matriz de confusión (single model sin TTA - VAL)")',
        'print("Top características por magnitud absoluta de pesos:")',
    ]:
        clear_code_cell(find_cell(cells, marker))

    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Patched notebook: {NOTEBOOK}")


if __name__ == "__main__":
    main()
