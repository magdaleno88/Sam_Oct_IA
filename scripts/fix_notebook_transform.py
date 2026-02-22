"""
Clean rewrite of the DDR2019_PCNN_ELM.ipynb patch.
Removes any malformed setup cell, inserts a clean one, and ensures
the DataLoader cell has the safe_transform correctly injected.
"""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "DDR2019_PCNN_ELM.ipynb"

SETUP_MARKER = "SETUP: pin numpy<2.0"

SETUP_CELL_SOURCE = [
    "# SETUP: pin numpy<2.0 for torchvision 0.17.2 compatibility\n",
    "# torchvision==0.17.2 was compiled against NumPy 1.x; NumPy 2.x breaks the ABI.\n",
    "# This cell installs numpy<2.0 and restarts the kernel once if needed.\n",
    "import subprocess, sys\n",
    "\n",
    "def _numpy_ok():\n",
    "    try:\n",
    "        import numpy as np\n",
    "        major = int(np.__version__.split('.')[0])\n",
    "        return major < 2\n",
    "    except Exception:\n",
    "        return False\n",
    "\n",
    "if not _numpy_ok():\n",
    "    print('Installing numpy<2.0 ...')\n",
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'numpy<2.0', '--upgrade'])\n",
    "    print('Done. Restarting kernel ...')\n",
    "    import IPython\n",
    "    IPython.get_ipython().kernel.do_shutdown(restart=True)\n",
    "else:\n",
    "    import numpy as np\n",
    "    print('numpy', np.__version__, '-- OK (< 2.0)')\n",
]

DATALOADER_CELL_SOURCE = [
    "from torch.utils.data import DataLoader\n",
    "from sam_ml.datasets import DDR2019Dataset\n",
    "from sam_ml.config import get_model_config, get_training_config\n",
    "import torchvision.transforms.v2 as T2\n",
    "\n",
    "# FIX: Use transforms.v2 to avoid torch.from_numpy() crash with NumPy 2.x\n",
    "safe_transform = T2.Compose([\n",
    "    T2.ToImage(),                           # PIL -> tv_tensors.Image (no numpy)\n",
    "    T2.ToDtype(torch.float32, scale=True),  # uint8 -> float32 [0, 1]\n",
    "])\n",
    "\n",
    "model_cfg = get_model_config()\n",
    "train_cfg = get_training_config()\n",
    "\n",
    "TRAIN_RATIO = 0.8\n",
    "VAL_RATIO = 0.2\n",
    "RANDOM_STATE = 42\n",
    "\n",
    "train_ds = DDR2019Dataset(data_dir, split=\"train\", train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, random_state=RANDOM_STATE, transform=safe_transform)\n",
    "val_ds   = DDR2019Dataset(data_dir, split=\"val\",   train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, random_state=RANDOM_STATE, transform=safe_transform)\n",
    "\n",
    "train_loader = DataLoader(\n",
    "    train_ds,\n",
    "    batch_size=train_cfg.batch_size,\n",
    "    shuffle=True,\n",
    "    num_workers=0,\n",
    ")\n",
    "\n",
    "val_loader = DataLoader(\n",
    "    val_ds,\n",
    "    batch_size=train_cfg.batch_size,\n",
    "    shuffle=False,\n",
    "    num_workers=0,\n",
    ")\n",
    "print(\"train:\", len(train_ds), \"val:\", len(val_ds), \"num_classes:\", model_cfg.num_classes)\n",
]


def make_code_cell(source, cell_id="patched-cell"):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def patch_notebook():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # ── 1) Remove any existing (possibly malformed) setup cells ──────────────
    cells[:] = [
        c for c in cells
        if not (
            c.get("cell_type") == "code"
            and SETUP_MARKER in "".join(c.get("source", []))
        )
    ]
    print("Removed old setup cell(s) if any.")

    # ── 2) Insert fresh setup cell at position 0 ─────────────────────────────
    cells.insert(0, make_code_cell(SETUP_CELL_SOURCE, "00-numpy-setup"))
    print("Inserted clean setup cell at position 0.")

    # ── 3) Replace DataLoader cell ────────────────────────────────────────────
    patched = False
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "DDR2019Dataset" in src and "get_model_config" in src:
            cell["source"] = DATALOADER_CELL_SOURCE
            cell["outputs"] = []
            cell["execution_count"] = None
            patched = True
            print("Replaced DataLoader cell.")
            break

    if not patched:
        print("WARNING: DataLoader cell not found!")

    # ── 4) Save ───────────────────────────────────────────────────────────────
    nb["cells"] = cells
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {NOTEBOOK}")

    # ── 5) Verify: print cell sources for inspection ──────────────────────────
    print("\n--- SETUP CELL ---")
    print("".join(cells[0]["source"]))

    for c in cells:
        src_str = "".join(c.get("source", []))
        if "DDR2019Dataset" in src_str and "get_model_config" in src_str:
            print("--- DATALOADER CELL ---")
            print(src_str)
            break


if __name__ == "__main__":
    patch_notebook()
