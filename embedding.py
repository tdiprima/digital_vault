"""Embedding model creation and semantic search over indexed file records."""

import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from records import FileRecord

logger = logging.getLogger(__name__)


def create_embedder(model_name: str) -> SentenceTransformer:
    """Load the SentenceTransformer model. Deferred until first call — not at import time."""
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


def semantic_search(
    query: str,
    records: list[FileRecord],
    embedder: SentenceTransformer,
    top_k: int,
) -> list[FileRecord]:
    """Return the top_k FileRecords most semantically similar to query."""
    if not records:
        logger.warning("semantic_search called with empty records list")
        return []
    query_embed = embedder.encode(query)
    embeds = torch.tensor(np.array([r["embed"] for r in records]))
    sims = cos_sim(query_embed, embeds)[0]
    indices = torch.topk(sims, min(top_k, len(sims))).indices
    return [records[int(i)] for i in indices]
