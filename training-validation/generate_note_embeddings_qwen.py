#!/usr/bin/env python3
"""Generate encounter-level Qwen3 embeddings from clinical-note text files.

This script is a publication-oriented extraction of the note-embedding stage used
in ``mm_NC_revisions_ume_nested_cv_early_fusion_c_grid-2.ipynb``.

Study implementation preserved
------------------------------
- Model: ``Qwen/Qwen3-Embedding-0.6B`` loaded with ``SentenceTransformer``.
- No explicit task prompt/instruction is supplied to ``model.encode``.
- One ``*.txt`` file represents one encounter.
- Encounter IDs are derived from file stems and normalized to ``Encounter/<id>``.
- Notes are whitespace-normalized and split by characters into 12,000-character
  chunks with 500-character overlap.
- Chunk embeddings are L2-normalized by ``SentenceTransformer``.
- Multiple chunks for one encounter are mean-pooled and L2-normalized again.
- Embeddings are saved as a float32 ``.npy`` matrix.
- The accompanying CSV contains ``encounter_id`` and ``embedding_row``.

Example
-------
python generate_note_embeddings_qwen.py \
    --notes-dir data/notes_ume/forms_txt_timestamps_polished \
    --output-embeddings data/processed/ume_note_embeddings.npy \
    --output-index data/processed/ume_note_embedding_index.csv

Raw clinical data and model weights are not distributed with this repository.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

LOGGER = logging.getLogger("generate_note_embeddings_qwen")

ID_COL = "encounter_id"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_CHARS_PER_CHUNK = 12_000
DEFAULT_CHUNK_OVERLAP = 500


def normalize_encounter_reference(value) -> str | float:
    """Normalize an encounter identifier to ``Encounter/<id>``.

    This reproduces the normalization logic from the source notebook and handles
    values such as ``Encounter_12345``, ``Encounter/12345``, and ``12345``.
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value.lower().endswith(".txt"):
        value = value[:-4]

    value = value.replace("\\", "/").strip()

    if value.lower().startswith("encounter_"):
        suffix = value.split("_", 1)[1]
        return f"Encounter/{suffix}"

    if value.lower().startswith("encounter/"):
        suffix = value.split("/", 1)[1]
        return f"Encounter/{suffix}"

    return f"Encounter/{value}"


def encounter_reference_from_txt(path: Path) -> str:
    """Derive the canonical encounter identifier from a note filename."""
    return normalize_encounter_reference(path.stem)


def read_text_file(path: Path) -> str:
    """Read UTF-8 text, falling back to Latin-1 as in the source notebook."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def split_long_text(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Whitespace-normalize and character-chunk a clinical note.

    Long notes are embedded chunk-wise and later mean-pooled. The source study
    used character-based rather than token-based chunking.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")

    text = " ".join(str(text).split())

    if len(text) == 0:
        text = "[EMPTY NOTE]"

    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def build_note_dataframe(notes_dir: Path) -> pd.DataFrame:
    """Load one text file per encounter in deterministic filename order."""
    notes_dir = Path(notes_dir)
    txt_files = sorted(notes_dir.glob("*.txt"))

    if len(txt_files) == 0:
        raise FileNotFoundError(f"No .txt files found in {notes_dir}")

    rows = []
    for path in txt_files:
        rows.append(
            {
                ID_COL: encounter_reference_from_txt(path),
                "note_path": str(path),
                "note_text": read_text_file(path),
            }
        )

    df = pd.DataFrame(rows)
    df[ID_COL] = df[ID_COL].apply(normalize_encounter_reference)

    # The study data contain one note file per encounter. Detect accidental
    # collisions after ID normalization rather than silently writing an ambiguous
    # index file.
    duplicated = df[ID_COL].duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = sorted(df.loc[duplicated, ID_COL].astype(str).unique())
        raise ValueError(
            "Multiple note files map to the same normalized encounter ID: "
            + ", ".join(duplicate_ids[:10])
            + (" ..." if len(duplicate_ids) > 10 else "")
        )

    return df


def create_note_embeddings(
    notes_dir: Path,
    output_embedding_path: Path,
    output_index_path: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    force_recompute: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create and save one frozen Qwen embedding per encounter."""
    notes_dir = Path(notes_dir)
    output_embedding_path = Path(output_embedding_path)
    output_index_path = Path(output_index_path)

    if (
        output_embedding_path.exists()
        and output_index_path.exists()
        and not force_recompute
    ):
        LOGGER.info("Existing note embeddings found; loading existing files.")
        embeddings = np.load(output_embedding_path)
        index_df = pd.read_csv(output_index_path)
        return embeddings, index_df

    notes_df = build_note_dataframe(notes_dir)
    LOGGER.info("Found %d note files.", len(notes_df))

    # Imported lazily so helper functions can be inspected/tested without loading
    # the heavyweight embedding stack.
    from sentence_transformers import SentenceTransformer

    st_device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Loading %s on %s.", model_name, st_device)

    try:
        model = SentenceTransformer(
            model_name,
            device=st_device,
            trust_remote_code=True,
        )
    except TypeError:
        # Compatibility fallback retained from the source notebook.
        model = SentenceTransformer(
            model_name,
            device=st_device,
        )

    all_chunks: list[str] = []
    chunk_to_encounter: list[str] = []

    for _, row in notes_df.iterrows():
        encounter_reference = row[ID_COL]
        chunks = split_long_text(
            row["note_text"],
            max_chars=max_chars_per_chunk,
            overlap=chunk_overlap,
        )

        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_to_encounter.append(encounter_reference)

    LOGGER.info("Created %d text chunks.", len(all_chunks))

    # IMPORTANT: no explicit prompt/prompt_name is passed. This exactly reflects
    # the embedding call used in the study notebook.
    chunk_embeddings = model.encode(
        all_chunks,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    grouped_embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
    for encounter_reference, embedding in zip(
        chunk_to_encounter, chunk_embeddings, strict=True
    ):
        grouped_embeddings[encounter_reference].append(embedding)

    encounter_references: list[str] = []
    note_embeddings: list[np.ndarray] = []

    # Preserve the deterministic order of ``notes_df`` / sorted filenames.
    for encounter_reference in notes_df[ID_COL]:
        encounter_chunk_embeddings = np.stack(
            grouped_embeddings[encounter_reference], axis=0
        )

        pooled = encounter_chunk_embeddings.mean(axis=0)
        pooled = pooled / max(np.linalg.norm(pooled), 1e-12)

        encounter_references.append(encounter_reference)
        note_embeddings.append(pooled.astype(np.float32))

    note_embedding_matrix = np.stack(note_embeddings, axis=0).astype(np.float32)

    index_df = pd.DataFrame(
        {
            ID_COL: encounter_references,
            "embedding_row": np.arange(len(encounter_references)),
        }
    )

    output_embedding_path.parent.mkdir(parents=True, exist_ok=True)
    output_index_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_embedding_path, note_embedding_matrix)
    index_df.to_csv(output_index_path, index=False)

    LOGGER.info("Saved embeddings to: %s", output_embedding_path)
    LOGGER.info("Saved index to: %s", output_index_path)
    LOGGER.info("Embedding shape: %s", note_embedding_matrix.shape)

    return note_embedding_matrix, index_df


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate encounter-level Qwen3 note embeddings using the exact "
            "chunking and pooling procedure from the study notebook."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--notes-dir", type=Path, required=True)
    parser.add_argument("--output-embeddings", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-chars-per-chunk",
        type=int,
        default=DEFAULT_MAX_CHARS_PER_CHUNK,
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute embeddings even if both output files already exist.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    create_note_embeddings(
        notes_dir=args.notes_dir,
        output_embedding_path=args.output_embeddings,
        output_index_path=args.output_index,
        model_name=args.model,
        batch_size=args.batch_size,
        max_chars_per_chunk=args.max_chars_per_chunk,
        chunk_overlap=args.chunk_overlap,
        force_recompute=args.force_recompute,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
