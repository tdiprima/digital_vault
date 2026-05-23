"""
Digital Vault — RAG-powered semantic search and organization for personal files.
Inspired by: https://medium.com/codrift/the-python-archive-of-my-life-i-built-a-digital-vault-that-organizes-everything-ive-done-81c2094f66f2

Usage:
  python digital_vault.py --source-dir /path/to/files
  python digital_vault.py --source-dir /path/to/files --n-clusters 5 --vault-dir ./vault

Environment variables:
  VAULT_SOURCE_DIR   — fallback for --source-dir
  VAULT_DIR          — fallback for --vault-dir (default: ./vault)
  VAULT_N_CLUSTERS   — fallback for --n-clusters (default: 7)
  VAULT_TOP_K        — fallback for --top-k (default: 5)
  VAULT_LLM_MODEL    — fallback for --llm-model (default: gemma3)
  VAULT_OLLAMA_URL   — Ollama base URL (default: http://localhost:11434/v1)
  LOG_LEVEL          — logging verbosity (default: INFO)

Modes (selected interactively at startup):
  1. Index and query: embed files for search without moving them
  2. Organize and query: cluster files, name folders via LLM, move, then search
"""

import logging
import os
import sys
import warnings

from openai import OpenAI

from chat import create_chat_fn
from clustering import build_cluster_samples, cluster_records, name_clusters
from config import load_config
from embedding import create_embedder
from pipeline import load_and_embed_files
from vault import organize_vault

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )


def prompt_mode() -> int:
    print("Digital Vault Options:")
    print("1. Index and query files (files stay in place)")
    print("2. Reorganize files into vault and query")
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ("1", "2"):
            return int(choice)
        print("Please enter 1 or 2.")


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        config = load_config()
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)

    embedder = create_embedder(config.embedding_model)
    ollama_url = os.getenv("VAULT_OLLAMA_URL", "http://localhost:11434/v1")
    client = OpenAI(base_url=ollama_url, api_key="ollama")
    mode = prompt_mode()

    logger.info("Loading and embedding files from %s", config.source_dir)
    records = load_and_embed_files(config.source_dir, config.supported_extensions, embedder)

    if not records:
        logger.error("No files with extractable text found in %s. Exiting.", config.source_dir)
        sys.exit(1)

    if mode == 2:
        logger.info("Clustering %d files into %d groups", len(records), config.n_clusters)
        records = cluster_records(records, config.n_clusters)
        samples = build_cluster_samples(records)
        logger.info("Naming %d clusters via LLM (single batch call)", len(samples))
        cluster_names = name_clusters(samples, client, config.llm_model)
        records = organize_vault(records, config.vault_dir, cluster_names)
        mode_description = (
            "Query your organized digital vault "
            "(e.g., 'Show me resumes' or 'Find invoices from 2024')."
        )
    else:
        mode_description = (
            "Query your indexed files "
            "(e.g., 'Show me resumes' or 'Find invoices from 2024')."
        )

    chat_fn = create_chat_fn(records, embedder, client, config.llm_model, config.top_k)

    print(f"\n{mode_description}")
    print("Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Bye.")
            break
        answer = chat_fn(question, [])
        print(f"\nVault: {answer}\n")


if __name__ == "__main__":
    main()
