#!/usr/bin/env python3
"""Generate encounter-level CT DICOM embeddings with MedImageInsight.

This script is a publication-oriented extraction of the optional CT image
embedding branch from the study notebook. It intentionally preserves the
original analysis choices:

- MedImageInsight wrapper repository: ``lion-ai/MedImageInsights``
- Model release directory: ``2024.09.27``
- Vision weights: ``medimageinsigt-v1.0.0.pt``
- Language weights: ``language_model.pth``
- One single-frame 2D DICOM image per encounter
- Encounter ID inferred from the DICOM filename stem
- Lung and mediastinal CT windows
- Mean aggregation across the two window-specific embeddings
- L2 normalization of the resulting encounter-level image embedding
- NumPy ``float32`` embedding matrix plus CSV row index

The MedImageInsight repository is expected to have been cloned locally with
Git LFS before running this script. The model package itself is not installed;
``medimageinsightmodel.py`` is imported directly from the cloned repository,
matching the Python 3.11-compatible approach used in the notebook.

Example
-------
python feature-extraction/generate_ct_embeddings_medimageinsight.py \
    --dicom-dir data/dicom_ume \
    --medimageinsight-repo-dir models/MedImageInsights \
    --output-embeddings data/processed/ume_ct_medimageinsight_embeddings.npy \
    --output-index data/processed/ume_ct_medimageinsight_index.csv
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pydicom
from PIL import Image


ID_COL = "encounter_id"

MEDIMAGEINSIGHT_MODEL_RELEASE = "2024.09.27"
MEDIMAGEINSIGHT_VISION_MODEL_NAME = "medimageinsigt-v1.0.0.pt"
MEDIMAGEINSIGHT_LANGUAGE_MODEL_NAME = "language_model.pth"

CT_WINDOW_PRESETS: Dict[str, Dict[str, float]] = {
    "lung": {"window_center": -600, "window_width": 1500},
    "mediastinal": {"window_center": 40, "window_width": 400},
}
CT_WINDOWS_TO_USE = ["lung", "mediastinal"]
CT_WINDOW_AGGREGATION = "mean"
CT_EMBED_BATCH_SIZE = 16


def normalize_encounter_reference(x) -> str:
    """Normalize encounter identifiers to ``Encounter/<id>``."""
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    if x.lower().endswith(".txt"):
        x = x[:-4]

    x = x.replace("\\", "/").strip()

    if x.lower().startswith("encounter_"):
        suffix = x.split("_", 1)[1]
        return f"Encounter/{suffix}"

    if x.lower().startswith("encounter/"):
        suffix = x.split("/", 1)[1]
        return f"Encounter/{suffix}"

    return f"Encounter/{x}"


def strip_known_file_suffixes(x) -> str:
    """Strip filename suffixes recognized in the source notebook."""
    x = str(x).strip()
    for suffix in [".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".txt"]:
        if x.lower().endswith(suffix):
            x = x[: -len(suffix)]
    return x


def encounter_reference_from_dicom_path(path: Path) -> str:
    """Derive the encounter ID from a DICOM filename stem."""
    return normalize_encounter_reference(strip_known_file_suffixes(path.stem))


def find_dicom_files(dicom_dir: Path) -> pd.DataFrame:
    """Recursively find DICOM files and retain one file per encounter."""
    dicom_dir = Path(dicom_dir)
    paths = sorted(list(dicom_dir.rglob("*.dcm")) + list(dicom_dir.rglob("*.dicom")))

    rows = []
    for path in paths:
        rows.append(
            {
                ID_COL: encounter_reference_from_dicom_path(path),
                "dicom_path": str(path),
                "filename": path.name,
            }
        )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise FileNotFoundError(f"No .dcm/.dicom files found in {dicom_dir}")

    duplicate_ids = int(df[ID_COL].duplicated().sum())
    if duplicate_ids > 0:
        print(
            f"WARNING: {duplicate_ids} duplicated encounter IDs found in {dicom_dir}. "
            "Keeping the first file per encounter."
        )
        df = df.drop_duplicates(subset=[ID_COL], keep="first").reset_index(drop=True)

    return df


def inspect_dicom_file(path: Path) -> dict:
    """Return basic DICOM metadata without reading pixel data."""
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    return {
        "modality": getattr(ds, "Modality", None),
        "rows": getattr(ds, "Rows", None),
        "columns": getattr(ds, "Columns", None),
        "number_of_frames": int(getattr(ds, "NumberOfFrames", 1)),
        "series_description": getattr(ds, "SeriesDescription", None),
        "photometric_interpretation": getattr(ds, "PhotometricInterpretation", None),
    }


def dicom_to_hu(ds) -> np.ndarray:
    """Convert DICOM pixel values to Hounsfield units."""
    image = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return image * slope + intercept


def window_hu_to_uint8(
    hu: np.ndarray,
    window_center: float,
    window_width: float,
) -> np.ndarray:
    """Apply a CT window and rescale to 8-bit grayscale."""
    lower = float(window_center) - float(window_width) / 2.0
    upper = float(window_center) + float(window_width) / 2.0
    x = np.clip(hu, lower, upper)
    x = (x - lower) / max(upper - lower, 1e-6)
    return (x * 255.0).astype(np.uint8)


def read_dicom_as_rgb_image(
    path: Path,
    window_center: float = -600,
    window_width: float = 1500,
) -> Image.Image:
    """Read one single-frame DICOM, apply CT windowing, and return RGB PIL image."""
    ds = pydicom.dcmread(str(path), force=True)

    if int(getattr(ds, "NumberOfFrames", 1)) > 1:
        raise ValueError(
            f"{path} appears to be multi-frame DICOM. "
            "This pipeline expects one 2D image per encounter."
        )

    hu = dicom_to_hu(ds)
    image_uint8 = window_hu_to_uint8(
        hu,
        window_center=window_center,
        window_width=window_width,
    )

    if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
        image_uint8 = 255 - image_uint8

    return Image.fromarray(image_uint8).convert("RGB")


def pil_image_to_base64_png(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string for MedImageInsight."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def dicom_window_to_base64_png(path: Path, window_name: str) -> str:
    """Apply one named CT window and return a base64 PNG string."""
    if window_name not in CT_WINDOW_PRESETS:
        raise ValueError(f"Unknown CT window: {window_name}")

    params = CT_WINDOW_PRESETS[window_name]
    image = read_dicom_as_rgb_image(path, **params)
    return pil_image_to_base64_png(image)


def load_medimageinsight_encoder(repo_dir: Path):
    """Load MedImageInsight directly from the locally cloned wrapper repository."""
    repo_dir = Path(repo_dir)
    if not repo_dir.exists():
        raise FileNotFoundError(
            f"MedImageInsight repository not found at {repo_dir}. "
            "Clone https://huggingface.co/lion-ai/MedImageInsights with Git LFS first."
        )

    medimage_file = repo_dir / "medimageinsightmodel.py"
    if not medimage_file.exists():
        raise FileNotFoundError(
            f"Could not find medimageinsightmodel.py at {medimage_file}"
        )

    repo_resolved = str(repo_dir.resolve())
    if repo_resolved not in sys.path:
        sys.path.insert(0, repo_resolved)

    from medimageinsightmodel import MedImageInsight

    encoder = MedImageInsight(
        model_dir=str(repo_dir / MEDIMAGEINSIGHT_MODEL_RELEASE),
        vision_model_name=MEDIMAGEINSIGHT_VISION_MODEL_NAME,
        language_model_name=MEDIMAGEINSIGHT_LANGUAGE_MODEL_NAME,
    )
    encoder.load_model()
    return encoder


def create_medimageinsight_dicom_embeddings(
    dicom_dir: Path,
    medimageinsight_repo_dir: Path,
    output_embedding_path: Path,
    output_index_path: Path,
    batch_size: int = CT_EMBED_BATCH_SIZE,
    force_recompute: bool = False,
):
    """Create one normalized MedImageInsight embedding per encounter."""
    output_embedding_path = Path(output_embedding_path)
    output_index_path = Path(output_index_path)

    if (
        output_embedding_path.exists()
        and output_index_path.exists()
        and not force_recompute
    ):
        print("Existing MedImageInsight CT embeddings found. Loading existing files.")
        embeddings = np.load(output_embedding_path)
        index_df = pd.read_csv(output_index_path)
        index_df[ID_COL] = index_df[ID_COL].apply(normalize_encounter_reference)
        return embeddings, index_df

    dicom_df = find_dicom_files(dicom_dir)

    print(f"Found {len(dicom_df)} DICOM images in {dicom_dir}")
    print("First DICOM metadata:")
    print(inspect_dicom_file(Path(dicom_df.iloc[0]["dicom_path"])))

    encoder = load_medimageinsight_encoder(medimageinsight_repo_dir)
    all_embeddings = []

    for start in range(0, len(dicom_df), batch_size):
        end = min(start + batch_size, len(dicom_df))
        batch_df = dicom_df.iloc[start:end].copy()
        batch_paths = [Path(p) for p in batch_df["dicom_path"]]

        window_embeddings = []

        for window_name in CT_WINDOWS_TO_USE:
            images_base64 = [
                dicom_window_to_base64_png(path, window_name=window_name)
                for path in batch_paths
            ]

            encoded = encoder.encode(images=images_base64)
            emb = np.asarray(encoded["image_embeddings"], dtype=np.float32)
            window_embeddings.append(emb)

        if CT_WINDOW_AGGREGATION == "mean":
            batch_embeddings = np.mean(np.stack(window_embeddings, axis=0), axis=0)
        elif CT_WINDOW_AGGREGATION == "concat":
            batch_embeddings = np.concatenate(window_embeddings, axis=1)
        else:
            raise ValueError("CT_WINDOW_AGGREGATION must be 'mean' or 'concat'.")

        norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
        batch_embeddings = batch_embeddings / np.clip(norms, 1e-8, None)

        all_embeddings.append(batch_embeddings.astype(np.float32))
        print(f"Embedded {end}/{len(dicom_df)} DICOM images", end="\r")

    print()

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    index_df = pd.DataFrame(
        {
            ID_COL: dicom_df[ID_COL].values,
            "embedding_row": np.arange(len(dicom_df)),
            "dicom_path": dicom_df["dicom_path"].values,
            "ct_windows": "+".join(CT_WINDOWS_TO_USE),
            "ct_window_aggregation": CT_WINDOW_AGGREGATION,
        }
    )

    output_embedding_path.parent.mkdir(parents=True, exist_ok=True)
    output_index_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_embedding_path, embeddings)
    index_df.to_csv(output_index_path, index=False)

    print("Saved CT embeddings:", output_embedding_path)
    print("Saved CT index:", output_index_path)
    print("CT embedding shape:", embeddings.shape)

    return embeddings, index_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate encounter-level CT DICOM embeddings with MedImageInsight."
    )
    parser.add_argument(
        "--dicom-dir",
        type=Path,
        required=True,
        help="Directory containing .dcm/.dicom files. Files are searched recursively.",
    )
    parser.add_argument(
        "--medimageinsight-repo-dir",
        type=Path,
        default=Path("./models/MedImageInsights"),
        help="Local clone of lion-ai/MedImageInsights.",
    )
    parser.add_argument(
        "--output-embeddings",
        type=Path,
        required=True,
        help="Output .npy file for the float32 embedding matrix.",
    )
    parser.add_argument(
        "--output-index",
        type=Path,
        required=True,
        help="Output CSV mapping encounter IDs to embedding rows.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CT_EMBED_BATCH_SIZE,
        help=f"Embedding batch size (default: {CT_EMBED_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute embeddings even if both output files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_medimageinsight_dicom_embeddings(
        dicom_dir=args.dicom_dir,
        medimageinsight_repo_dir=args.medimageinsight_repo_dir,
        output_embedding_path=args.output_embeddings,
        output_index_path=args.output_index,
        batch_size=args.batch_size,
        force_recompute=args.force_recompute,
    )


if __name__ == "__main__":
    main()
