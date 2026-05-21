"""Shared file-loading pipeline: discover, extract text, and embed all files in a directory."""

import logging
import os

from sentence_transformers import SentenceTransformer

from extraction import extract_text
from records import FileRecord

logger = logging.getLogger(__name__)

TEXT_TRUNCATE_CHARS = 2000


def load_and_embed_files(
    source_dir: str,
    supported_extensions: list[str],
    embedder: SentenceTransformer,
) -> list[FileRecord]:
    """Discover supported files in source_dir, extract their text, and embed each one."""
    files = [
        os.path.join(source_dir, fname)
        for fname in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, fname))
        and os.path.splitext(fname)[1].lower() in supported_extensions
    ]

    if not files:
        logger.warning("No supported files found in %s", source_dir)
        return []

    records: list[FileRecord] = []
    for file_path in files:
        text = extract_text(file_path)
        if not text.strip():
            logger.debug("No text extracted from %s, skipping", file_path)
            continue
        truncated = text[:TEXT_TRUNCATE_CHARS]
        embed = embedder.encode(truncated)
        records.append({"path": file_path, "text": truncated, "embed": embed})

    logger.info("Loaded %d files with extractable text from %s", len(records), source_dir)
    return records
