# Digital Vault

A RAG-powered tool that turns a chaotic pile of personal files into a searchable, conversational archive.

## Every File You've Ever Saved Is a Mess

Years of PDFs, Word docs, Jupyter notebooks, code files, screenshots, and notes scattered across one directory with no coherent structure. You know something is in there somewhere, but finding it means manually opening files one by one. Traditional search only matches filenames and exact keywords -- it can't understand what your documents are actually about.

## Semantic Search Meets Automatic Organization

Digital Vault reads the content of your files -- including OCR on images -- generates semantic embeddings, and lets you ask questions in plain English through a chat interface. It supports two modes:

- **Index mode** leaves your files in place and builds a searchable semantic index over them.
- **Organize mode** clusters files by topic using KMeans, asks GPT to name each cluster, and physically sorts everything into labeled folders.

Both modes launch a Gradio chatbot that uses retrieval-augmented generation (RAG) to answer questions grounded in your actual documents, citing which files the answer came from.

### Supported file types

PDF, DOCX, Markdown, plain text, Python scripts, Jupyter notebooks, and images (JPG/PNG via Tesseract OCR).

## Example

```
$ python digital_vault.py
Digital Vault Options:
1. Index and query files (files stay in place)
2. Reorganize files into vault and query

Enter your choice (1 or 2): 1
Indexed 42 files for search.
Running on local URL: http://127.0.0.1:7860
```

Then in the browser:

> **You:** Find anything related to machine learning training pipelines
>
> **Vault:** Based on the documents, I found references to training pipelines in two files. From `ml_notes.md`: ... From `experiment_v3.ipynb`: ... These files discuss data preprocessing steps and hyperparameter tuning for a convolutional model.

## Usage

### Prerequisites

- Python 3.11+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed system-wide
- An OpenAI API key

### Install

```bash
pip install -e .
```

### Configure

Export your OpenAI API key:

```bash
export OPENAI_API_KEY='your-key'
```

Edit the constants at the top of `digital_vault.py` to point at your file directory:

```python
SOURCE_DIR = "/path/to/your/files"
VAULT_DIR = "./vault"
N_CLUSTERS = 7
```

### 🚀 Run

```bash
python digital_vault.py
```

Choose mode 1 to index and search without moving files, or mode 2 to cluster, rename, reorganize, and search.

## 📜 License

[MIT](LICENSE) 

<br>
