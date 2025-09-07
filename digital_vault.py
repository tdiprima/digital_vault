"""
A Python script to organize files into a semantic digital vault using embeddings, clustering, AI naming, and a chatbot interface.
Inspired by: https://medium.com/codrift/the-python-archive-of-my-life-i-built-a-digital-vault-that-organizes-everything-ive-done-81c2094f66f2
Requirements: Install sentence-transformers, scikit-learn, openai, gradio, pytesseract, Pillow, PyPDF2, python-docx
Also, install Tesseract OCR system-wide.
Set your OpenAI API key as an environment variable: export OPENAI_API_KEY='your-key'
"""

import json
import os
import shutil
import warnings
from collections import defaultdict
from pathlib import Path

import gradio as gr
import numpy as np
import PyPDF2
import pytesseract
import torch
from docx import Document
from openai import OpenAI
from PIL import Image
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.cluster import KMeans

# Suppress common warnings for cleaner output
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

# Constants
SOURCE_DIR = "./inbox"  # Directory with chaotic files
VAULT_DIR = "./vault"  # Organized vault directory
N_CLUSTERS = 7  # Number of clusters (adjust as needed)
MODEL_NAME = "all-MiniLM-L6-v2"
SUPPORTED_EXTENSIONS = [
    ".pdf",
    ".md",
    ".txt",
    ".docx",
    ".ipynb",
    ".py",
    ".jpg",
    ".jpeg",
    ".png",
]
TOP_K = 5  # Top results for semantic search

# Load embedding model
model = SentenceTransformer(MODEL_NAME)

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)


def embed_text(text):
    """Embed text using SentenceTransformer."""
    return model.encode(text)


def extract_text(file_path):
    """Extract text from various file types, including OCR for images."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in (".txt", ".md", ".py"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text
        elif ext == ".docx":
            doc = Document(file_path)
            return " ".join([p.text for p in doc.paragraphs])
        elif ext == ".ipynb":
            with open(file_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            text = ""
            for cell in nb.get("cells", []):
                if cell["cell_type"] in ("code", "markdown"):
                    text += "".join(cell["source"]) + "\n"
            return text
        elif ext in (".jpg", ".jpeg", ".png"):
            return pytesseract.image_to_string(Image.open(file_path))
        else:
            return ""
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""


def name_cluster(sample_texts):
    """Use GPT-4o-mini to name a cluster based on sample texts."""
    prompt = f"""
    Provide a short, descriptive folder name for a group of files based on these sample contents (summarize the theme): 
    {'; '.join([text[:200] for text in sample_texts])}.
    But do NOT prefix the name with 'folder_name_'.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
    )
    folder_name = response.choices[0].message.content.strip()
    # Clean for filesystem (remove invalid chars)
    folder_name = "".join(
        c for c in folder_name if c.isalnum() or c in [" ", "_", "-"]
    ).replace(" ", "_")
    return folder_name


def move_file(file_path, destination_folder):
    """Move file to destination folder."""
    Path(destination_folder).mkdir(parents=True)
    shutil.move(
        file_path, os.path.join(destination_folder, os.path.basename(file_path))
    )


def index_files(source_dir):
    """Index files without moving them - just create embeddings for search."""
    # Collect files
    files = [
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f))
        and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    data = []
    for fp in files:
        text = extract_text(fp)
        if text.strip():
            # Truncate text for embedding if too long (model limit ~256 tokens, but we truncate to 2000 chars for safety)
            truncated_text = text[:2000]
            embed = embed_text(truncated_text)
            data.append({"path": fp, "text": truncated_text, "embed": embed})

    if not data:
        print("No files with extractable text found.")
        return []

    print(f"Indexed {len(data)} files for search.")
    return data


def organize_files():
    """Main function to process, cluster, name, and move files."""
    # Collect files
    files = [
        os.path.join(SOURCE_DIR, f)
        for f in os.listdir(SOURCE_DIR)
        if os.path.isfile(os.path.join(SOURCE_DIR, f))
        and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    data = []
    for fp in files:
        text = extract_text(fp)
        if text.strip():
            # Truncate text for embedding if too long (model limit ~256 tokens, but we truncate to 2000 chars for safety)
            truncated_text = text[:2000]
            embed = embed_text(truncated_text)
            data.append({"path": fp, "text": truncated_text, "embed": embed})

    if not data:
        print("No files with extractable text found.")
        return []

    # Cluster
    all_embeddings = np.array([d["embed"] for d in data])
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit(all_embeddings)

    for i, d in enumerate(data):
        d["cluster"] = kmeans.labels_[i]

    # Group by cluster
    clusters = defaultdict(list)
    for d in data:
        clusters[d["cluster"]].append(d)

    # Name and move
    for cluster_id, items in clusters.items():
        sample_texts = [item["text"] for item in items[:3]]
        folder_name = name_cluster(sample_texts)
        dest_folder = os.path.join(VAULT_DIR, folder_name)
        for item in items:
            move_file(item["path"], dest_folder)
            # Update path in data
            item["path"] = os.path.join(dest_folder, os.path.basename(item["path"]))

    print("Files organized into vault.")
    return data


def semantic_search(query, data, top_k=TOP_K):
    """Semantic search: find top_k similar files based on query embedding."""
    query_embed = embed_text(query)
    embeds = torch.tensor(np.array([d["embed"] for d in data]))
    sims = cos_sim(query_embed, embeds)[0]
    top_indices = torch.topk(sims, min(top_k, len(sims))).indices
    return [data[i]["path"] for i in top_indices]


def chat_query(message, history):
    """Chat function for Gradio: process query and return results."""
    results = semantic_search(message, data)
    response = "Here are the top relevant files:\n" + "\n".join(results)
    return response


if __name__ == "__main__":
    print("Digital Vault Options:")
    print("1. Index and query files (files stay in place)")
    print("2. Reorganize files into vault and query")

    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ("1", "2"):
            break
        print("Please enter 1 or 2.")

    if choice == "1":
        # Index files without moving them
        data = index_files(SOURCE_DIR)
        mode_description = "Query your indexed files (e.g., 'Show me resumes' or 'Find invoices from 2024')."
    else:
        # Organize files and get data for search
        data = organize_files()
        mode_description = "Query your organized digital vault (e.g., 'Show me resumes' or 'Find invoices from 2024')."

    # Launch Gradio chatbot
    if data:
        interface = gr.ChatInterface(
            fn=chat_query,
            title="Ask My Archive",
            description=mode_description,
            type="messages",
        )
        interface.launch()
    else:
        print("No data to query. Exiting.")
