"""
Patches DDR2019_PCNN_ELM.ipynb to fix the NumPy 2.x / torchvision 0.17.2 incompatibility.

Two changes are made (notebook only):
1. Prepends a setup cell that pins numpy<2.0 and restarts the kernel.
2. Injects a numpy-free transform (torchvision.transforms.v2) into the DataLoader cell.
"""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "DDR2019_PCNN_ELM.ipynb"

# ── Cell 0: pin numpy<2.0 ────────────────────────────────────────────────────

SETUP_CELL_MARKER = "# SETUP: pin numpy<2.0 for torchvision 0.17.2 compatibility"

SETUP_CELL_SOURCE = [
    "# SETUP: pin numpy<2.0 for torchvision 0.17.2 compatibility\n",
    "# torchvision==0.17.2 was compiled against NumPy 1.x; NumPy 2.x breaks the ABI.\n",
    "# This installs numpy<2.0 silently if needed, then restarts the kernel once.\n",
    "import importlib, subprocess, sys\n",
    "\n",
    "def _numpy_ok() -> bool:\n",
    "    try:\n",
    "        import numpy as np\n",
    "        return tuple(int(x) for x in np.__version__.split('.')[:2]) < (2, 0)\n",
    "    except Exception:\n",
    "        return False\n",
    "\n",
    "if not _numpy_ok():\n",
    "    print('Installing numpy<2.0 …')\n",
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'numpy<2.0', '--upgrade'])\n",
    "    print('numpy<2.0 installed. Restarting kernel …')\n",
    "    import IPython\n",
    "    IPython.get_ipython().kernel.do_shutdown(restart=True)\n",
    "else:\n",
    "    import numpy as np\n",
    "    print(f'numpy {np.__version__} — OK (< 2.0)')\n",
]

# ── DataLoader cell: numpy-free transform ────────────────────────────────────

OLD_DATALOADER_PREFIX = [
    "from torch.utils.data import DataLoader\n",
    "from sam_ml.datasets import DDR2019Dataset\n",
    "from sam_ml.config import get_model_config, get_training_config\n",
    "\n",
    "model_cfg = get_model_config()\n",
]

NEW_DATALOADER_HEADER = [
    "from torch.utils.data import DataLoader\n",
    "from sam_ml.datasets import DDR2019Dataset\n",
    "from sam_ml.config import get_model_config, get_training_config\n",
    "import torchvision.transforms.v2 as T2\n",
    "\n",
    "# FIX: torchvision 0.17.2 was compiled against NumPy 1.x, but NumPy 2.x is installed.\n",
    "# transforms.ToTensor() crashes via torch.from_numpy(). Use transforms.v2 instead,\n",
    "# which converts PIL images to tensors without going through numpy.\n",
    "safe_transform = T2.Compose([\n",
    "    T2.ToImage(),                            # PIL -> tv_tensors.Image (no numpy)\n",
    "    T2.ToDtype(torch.float32, scale=True),  # uint8 -> float32 [0, 1]\n",
    "])\n",
    "\n",
    "model_cfg = get_model_config()\n",
]


def _already_has_setup_cell(nb: dict) -> bool:
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            if any(SETUP_CELL_MARKER in line for line in cell.get("source", [])):
                return True
    return False


def _patch_dataloader_cell(nb: dict) -> bool:
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = cell["source"]
        joined = "".join(src)
        if (
            "from sam_ml.datasets import DDR2019Dataset\n" in joined
            and "get_model_config" in joined
            and "import torchvision.transforms.v2 as T2\n" not in joined  # not yet patched
        ):
            # Build new source: replace the header, keep the rest (DataLoaders + print)
            old_header_len = len(OLD_DATALOADER_PREFIX)
            # Find the index where old header ends in src
            if src[:old_header_len] == OLD_DATALOADER_PREFIX:
                cell["source"] = NEW_DATALOADER_HEADER + src[old_header_len:]
            else:
                # Fallback: rebuild from full source string
                rest = joined.replace(
                    "".join(OLD_DATALOADER_PREFIX),
                    "".join(NEW_DATALOADER_HEADER),
                    1,
                )
                cell["source"] = rest.splitlines(keepends=True)

            # Replace dataset constructor calls to pass transform=safe_transform
            new_src = []
            for line in cell["source"]:
                if "DDR2019Dataset(data_dir," in line and "transform=" not in line:
                    line = line.rstrip()
                    if line.endswith(")"):
                        line = line[:-1] + ", transform=safe_transform)\n"
                    else:
                        line = line + "  # transform injected below\n"
                new_src.append(line)
            cell["source"] = new_src

            cell["outputs"] = []
            cell["execution_count"] = None
            return True
    return False


def patch_notebook() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    # 1) Add setup cell if missing
    if _already_has_setup_cell(nb):
        print("ℹ️  Setup cell already present.")
    else:
        setup_cell = {
            "cell_type": "code",
            "execution_count": None,
            "id": "00000000-setup-numpy-pin",
            "metadata": {},
            "outputs": [],
            "source": SETUP_CELL_SOURCE,
        }
        nb["cells"].insert(0, setup_cell)
        print("✅ Inserted numpy<2.0 setup cell at position 0.")

    # 2) Patch DataLoader cell
    if _patch_dataloader_cell(nb):
        print("✅ Patched DataLoader cell with safe_transform.")
    else:
        print("ℹ️  DataLoader cell already patched or not found.")

    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Saved {NOTEBOOK}")


if __name__ == "__main__":
    patch_notebook()
