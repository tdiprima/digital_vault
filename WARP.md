# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Digital Vault is a Python script that organizes files into a semantic digital vault using AI embeddings, clustering, and natural language queries. The system supports two operational modes: indexing files in place for search, or reorganizing them into AI-generated semantic clusters.

## Environment Setup

### Python Environment
The project uses Python 3.11 with a virtual environment:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Or create a new virtual environment if needed
python3 -m venv .venv
source .venv/bin/activate
```

### Dependencies Installation
```bash
# Install all required dependencies
pip install -r requirements.txt

# Key dependencies include:
# - sentence-transformers (semantic embeddings)
# - scikit-learn (clustering)
# - openai (GPT-4o-mini for folder naming)
# - gradio (web chat interface)
# - pytesseract (OCR for images)
```

### System Dependencies
```bash
# Install Tesseract OCR (required for image text extraction)
brew install tesseract  # macOS
```

### Environment Variables
```bash
# Required: Set OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
```

## Common Commands

### Running the Application
```bash
# Run the main script
python digital_vault.py

# The script will prompt for mode selection:
# 1. Index & Query (files stay in original location)  
# 2. Reorganize & Query (move files to semantic clusters)
```

### Development and Testing
```bash
# Check Python version and virtual environment
python --version
which python

# Verify dependencies are installed
pip list | grep -E "(sentence-transformers|openai|gradio)"

# Test Tesseract OCR installation
tesseract --version

# Quick test of import functionality
python -c "from digital_vault import embed_text, extract_text; print('✓ All imports working')"
```

### Troubleshooting

#### NumPy Compatibility Issues
If you encounter NumPy 2.x compatibility errors:

```bash
# Downgrade NumPy to 1.x series
source .venv/bin/activate
pip install "numpy<2"
```

#### Common Warnings
The application suppresses common warnings by default:

- `TOKENIZERS_PARALLELISM` warnings from HuggingFace
- `threadpoolctl` RuntimeWarnings from scikit-learn
- Gradio deprecation warnings (fixed with `type="messages"`)

### File Operations

```bash
# Check source directory contents (default: ./Misc)
ls -la Misc/

# Check vault directory (organized files destination)
ls -la vault/

# View supported file types in source
find Misc/ -type f \( -name "*.pdf" -o -name "*.md" -o -name "*.txt" -o -name "*.docx" -o -name "*.ipynb" -o -name "*.py" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \)
```

## Architecture Overview

### Core Components

#### 1. Text Extraction Pipeline (`extract_text`)
- Multi-format text extraction supporting PDF, DOCX, Markdown, Python files, Jupyter notebooks, and images (with OCR)
- Handles encoding issues and extraction failures gracefully
- Truncates content to 2000 characters for embedding model limits

#### 2. Semantic Embedding System
- Uses SentenceTransformer model 'all-MiniLM-L6-v2' for creating embeddings
- Generates dense vector representations of text content for semantic similarity
- Embedding dimension: 384 (model-specific)

#### 3. File Organization Modes

**Index Mode (`index_files`)**:

- Scans source directory without moving files
- Creates embeddings for all supported file types
- Preserves original file locations
- Returns structured data for search functionality

**Reorganize Mode (`organize_files`)**:

- Performs K-means clustering on file embeddings (default: 7 clusters)
- Uses GPT-4o-mini to generate descriptive folder names from sample texts
- Physically moves files to organized vault directory structure
- Updates file paths in data structure

#### 4. Semantic Search Engine (`semantic_search`)
- Computes cosine similarity between query embeddings and file embeddings
- Returns top-K most relevant files (default: 5)
- Uses PyTorch tensors for efficient similarity computation

#### 5. Web Interface Integration
- Gradio-based chat interface for natural language queries
- Real-time semantic search through conversational interface
- Launches local web server for user interaction

### Data Flow Architecture

1. **File Discovery**: Scans source directory for supported file types
2. **Text Extraction**: Extracts content from various file formats
3. **Embedding Generation**: Creates semantic embeddings using SentenceTransformer
4. **Mode-Specific Processing**:
   - Index Mode: Preserves file locations, builds search index
   - Reorganize Mode: Clusters files, generates folder names, moves files
5. **Search Interface**: Launches Gradio web interface for queries
6. **Query Processing**: Converts natural language queries to embeddings, finds similar files

### Configuration Parameters

Key constants defined in `digital_vault.py`:

- `SOURCE_DIR`: Input directory path (default: './Misc')
- `VAULT_DIR`: Organized vault directory (default: './vault')  
- `N_CLUSTERS`: Number of semantic clusters (default: 7)
- `MODEL_NAME`: SentenceTransformer model ('all-MiniLM-L6-v2')
- `TOP_K`: Number of search results returned (default: 5)
- `SUPPORTED_EXTENSIONS`: File types processed by the system

### External Dependencies

- **OpenAI API**: GPT-4o-mini for intelligent folder naming
- **SentenceTransformers**: Semantic text embeddings
- **Tesseract OCR**: Image text extraction
- **Gradio**: Web-based chat interface
- **scikit-learn**: K-means clustering algorithm

## Directory Structure

```
digital_vault/
├── digital_vault.py      # Main application script
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── Misc/               # Default source directory (chaotic files)
├── vault/              # Organized vault directory (created on reorganize)
├── .venv/              # Python virtual environment
└── .ropeproject/       # Python IDE project files
```

<br>
