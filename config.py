"""Configuration loading from CLI args with environment variable fallbacks."""

import argparse
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".pdf", ".md", ".txt", ".docx", ".ipynb", ".py", ".jpg", ".jpeg", ".png"]


@dataclass
class VaultConfig:
    source_dir: str
    vault_dir: str
    n_clusters: int
    embedding_model: str
    llm_model: str
    top_k: int
    supported_extensions: list[str] = field(default_factory=lambda: list(SUPPORTED_EXTENSIONS))


def load_config() -> VaultConfig:
    """Parse CLI args with env var fallbacks. Fails fast if required values are missing or invalid."""
    parser = argparse.ArgumentParser(
        description="Digital Vault: organize and query personal files semantically."
    )
    parser.add_argument(
        "--source-dir",
        default=os.getenv("VAULT_SOURCE_DIR"),
        help="Directory of files to process (or set VAULT_SOURCE_DIR)",
    )
    parser.add_argument(
        "--vault-dir",
        default=os.getenv("VAULT_DIR", "./vault"),
        help="Output vault directory for organize mode",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=int(os.getenv("VAULT_N_CLUSTERS", "7")),
        help="Number of semantic clusters for organize mode",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("VAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("VAULT_LLM_MODEL", "gpt-5.2"),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("VAULT_TOP_K", "5")),
        help="Number of top results returned by semantic search",
    )
    args = parser.parse_args()

    if not args.source_dir:
        raise ValueError(
            "source_dir is required. Use --source-dir or set the VAULT_SOURCE_DIR env var."
        )
    if not os.path.isdir(args.source_dir):
        raise ValueError(f"source_dir does not exist or is not a directory: {args.source_dir}")
    if args.n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {args.n_clusters}")
    if args.top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {args.top_k}")

    return VaultConfig(
        source_dir=args.source_dir,
        vault_dir=args.vault_dir,
        n_clusters=args.n_clusters,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        top_k=args.top_k,
    )
