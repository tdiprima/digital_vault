# Digital Vault — Domain Glossary

## Core Concepts

**FileRecord** — a dict representing one processed file. Contains `path`, `text` (truncated), `embed` (numpy vector), and optionally `cluster` (int label). Flows through every stage of the pipeline. Defined in `records.py`.

**load-and-embed pipeline** — the shared, read-only phase: discover supported files in `source_dir`, extract text, embed each file's text into a vector. Returns a list of `FileRecord`s. Both modes (index and organize) start here. Implemented in `pipeline.py`.

**organize mode** — the destructive mode: runs the pipeline, clusters records by embedding similarity, names each cluster via LLM, then physically moves files into named folders under `vault_dir`. Distinct from index mode, which is read-only.

**index mode** — read-only mode: runs the pipeline, launches the RAG chatbot. Files stay in `source_dir`.

**VaultConfig** — the single config object. Loaded once at startup from CLI args with env var fallbacks. Passed explicitly to functions that need it. Defined in `config.py`.

**cluster** — a group of semantically similar files, assigned by KMeans on embeddings. Identified by an integer label. Named by LLM into a human-readable folder name.

**vault** — the organized output directory (`vault_dir`). Contains one subfolder per cluster, named by LLM.

**semantic search** — cosine similarity between a query embedding and all `FileRecord` embeddings. Returns the top-k most similar `FileRecord`s (not just paths). Implemented in `embedding.py`.

**RAG chat** — Retrieval-Augmented Generation: the chatbot retrieves the top-k relevant `FileRecord`s for a user query, formats their text as context, and sends it to the LLM for an answer with citations.

## Module Map

| Module | Responsibility |
|---|---|
| `records.py` | `FileRecord` TypedDict — shared data type |
| `config.py` | `VaultConfig` dataclass, CLI + env var loading |
| `extraction.py` | Text extraction from PDF, DOCX, TXT, IPYNB, images |
| `embedding.py` | Embedder creation, semantic search over FileRecords |
| `pipeline.py` | load-and-embed pipeline (shared by both modes) |
| `clustering.py` | KMeans clustering, batch LLM cluster naming |
| `vault.py` | File movement, vault organization |
| `chat.py` | RAG chat callback factory |
| `digital_vault.py` | Orchestration entry point |
