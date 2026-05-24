# Digital Vault

A RAG-powered tool that turns a chaotic pile of personal files into a searchable, conversational archive.

## Every File You've Ever Saved Is a Mess

Years of PDFs, Word docs, Jupyter notebooks, code files, screenshots, and notes scattered across one directory with no coherent structure. You know something is in there somewhere, but finding it means manually opening files one by one. Traditional search only matches filenames and exact keywords -- it can't understand what your documents are actually about.

## Semantic Search Meets Automatic Organization

Digital Vault reads the content of your files -- including OCR on images -- generates semantic embeddings, and lets you ask questions in plain English through a command-line chat loop. It supports two modes:

- **Index mode** leaves your files in place and builds a searchable semantic index over them.
- **Organize mode** clusters files by topic using KMeans, asks the LLM to name each cluster, and physically sorts everything into labeled folders.

Both modes drop you into an interactive CLI prompt that uses retrieval-augmented generation (RAG) to answer questions grounded in your actual documents, citing which files the answer came from. Uses Ollama with gemma3 locally -- no API key required.

### Supported file types

PDF, DOCX, Markdown, plain text, Python scripts, Jupyter notebooks, and images (JPG/PNG via Tesseract OCR).

## Example

```
$ python digital_vault.py --source-dir ~/Documents/my-files
Digital Vault Options:
1. Index and query files (files stay in place)
2. Reorganize files into vault and query

Enter your choice (1 or 2): 1
2026-05-21 07:45:00 INFO pipeline — Loaded 42 files with extractable text

Query your indexed files (e.g., 'Show me resumes' or 'Find invoices from 2024').
Type 'quit' or 'exit' to stop.

You: Find anything related to machine learning training pipelines

Vault: Based on the documents, I found references to training pipelines in two files.
From `ml_notes.md`: ... From `experiment_v3.ipynb`: ... These files discuss data
preprocessing steps and hyperparameter tuning for a convolutional model.

You: quit
Bye.
```

## Usage

### Prerequisites

- Python 3.11+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed system-wide
- [Ollama](https://ollama.com) installed and running with gemma3 pulled (`ollama pull gemma3`)

### Install

```bash
pip install -e .
```

### Configure

Make sure Ollama is running:

```bash
ollama serve
```

### Run

```bash
python digital_vault.py --source-dir /path/to/your/files
```

Choose mode 1 to index and search without moving files, or mode 2 to cluster, rename, reorganize, and search.

#### All options

```
--source-dir PATH     Directory of files to process (required, or set VAULT_SOURCE_DIR)
--vault-dir PATH      Output directory for organize mode (default: ./vault)
--n-clusters N        Number of semantic clusters (default: 7)
--embedding-model M   SentenceTransformer model name (default: all-MiniLM-L6-v2)
--llm-model M         Ollama model for naming and chat (default: gemma3)
--top-k N             Number of search results per query (default: 5)
```

All options can also be set via environment variables:

```bash
export VAULT_SOURCE_DIR=/path/to/your/files
export VAULT_DIR=./vault
export VAULT_N_CLUSTERS=7
export VAULT_TOP_K=5
export VAULT_OLLAMA_URL=http://localhost:11434/v1  # default
export LOG_LEVEL=INFO   # DEBUG for verbose output
```

## 📜 License

[MIT](LICENSE) 

<br>
