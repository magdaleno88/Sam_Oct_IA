"""Tests for DDR2019 dual-filter preprocessing export."""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

from sam_ml.preprocessing import main


def _make_temp_dir() -> str:
    return tempfile.mkdtemp()


def test_dualfilters_export_creates_synchronized_folders_and_csv() -> None:
    temp_dir = _make_temp_dir()
    try:
        raw_dir = Path(temp_dir) / "raw_images"
        raw_dir.mkdir(parents=True)
        csv_path = Path(temp_dir) / "DR_grading.csv"
        processed_dir = Path(temp_dir) / "processed_dual"

        filenames = [f"img_{i:03d}.jpg" for i in range(4)]
        for i, name in enumerate(filenames):
            img = Image.new("RGB", (700, 640), color=(40 * i, 70, 120))
            img.save(raw_dir / name, "JPEG")

        pd.DataFrame(
            {
                "id_code": filenames,
                "diagnosis": [0, 1, 2, 3],
            }
        ).to_csv(csv_path, index=False)

        exit_code = main(
            [
                "ddr2019_dualfilters",
                "--raw-img-dir",
                str(raw_dir),
                "--raw-csv-path",
                str(csv_path),
                "--processed-dir",
                str(processed_dir),
            ]
        )
        assert exit_code == 0

        clahe_dir = processed_dir / "images_clahe"
        ceced_dir = processed_dir / "images_ceced"
        labels_csv = processed_dir / "labels_dual.csv"

        assert clahe_dir.exists()
        assert ceced_dir.exists()
        assert labels_csv.exists()

        clahe_files = sorted(f.name for f in clahe_dir.glob("*.jpg"))
        ceced_files = sorted(f.name for f in ceced_dir.glob("*.jpg"))
        assert clahe_files == ceced_files == sorted(filenames)

        for name in clahe_files:
            assert Image.open(clahe_dir / name).size == (299, 299)
            assert Image.open(ceced_dir / name).size == (224, 224)

        df = pd.read_csv(labels_csv)
        assert list(df.columns) == ["clahe_path", "ceced_path", "label"]
        assert len(df) == len(filenames)
        assert set(df["clahe_path"]) == {f"images_clahe/{f}" for f in filenames}
        assert set(df["ceced_path"]) == {f"images_ceced/{f}" for f in filenames}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_dualfilters_export_skips_small_images_consistently() -> None:
    temp_dir = _make_temp_dir()
    try:
        raw_dir = Path(temp_dir) / "raw_images"
        raw_dir.mkdir(parents=True)
        csv_path = Path(temp_dir) / "DR_grading.csv"
        processed_dir = Path(temp_dir) / "processed_dual"

        keep_name = "keep.jpg"
        drop_name = "drop.jpg"
        Image.new("RGB", (640, 640), color=(100, 100, 100)).save(raw_dir / keep_name, "JPEG")
        Image.new("RGB", (200, 200), color=(50, 50, 50)).save(raw_dir / drop_name, "JPEG")

        pd.DataFrame(
            {
                "id_code": [keep_name, drop_name],
                "diagnosis": [1, 4],
            }
        ).to_csv(csv_path, index=False)

        exit_code = main(
            [
                "ddr2019_dualfilters",
                "--raw-img-dir",
                str(raw_dir),
                "--raw-csv-path",
                str(csv_path),
                "--processed-dir",
                str(processed_dir),
                "--min-size",
                "299",
            ]
        )
        assert exit_code == 0

        clahe_files = sorted(f.name for f in (processed_dir / "images_clahe").glob("*.jpg"))
        ceced_files = sorted(f.name for f in (processed_dir / "images_ceced").glob("*.jpg"))
        assert clahe_files == [keep_name]
        assert ceced_files == [keep_name]

        df = pd.read_csv(processed_dir / "labels_dual.csv")
        assert len(df) == 1
        assert df.iloc[0]["clahe_path"] == f"images_clahe/{keep_name}"
        assert df.iloc[0]["ceced_path"] == f"images_ceced/{keep_name}"
        assert int(df.iloc[0]["label"]) == 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
