"""Text extraction from supported file types, including OCR for images."""

import json
import logging
import os

import PyPDF2
import pytesseract
from docx import Document
from PIL import Image

logger = logging.getLogger(__name__)


def extract_text(file_path: str) -> str:
    """Extract text content from a file. Returns empty string on failure or unsupported type."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in (".txt", ".md", ".py"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "".join(page.extract_text() or "" for page in reader.pages)
        elif ext == ".docx":
            doc = Document(file_path)
            return " ".join(p.text for p in doc.paragraphs)
        elif ext == ".ipynb":
            with open(file_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            return "\n".join(
                "".join(cell["source"])
                for cell in nb.get("cells", [])
                if cell["cell_type"] in ("code", "markdown")
            )
        elif ext in (".jpg", ".jpeg", ".png"):
            return pytesseract.image_to_string(Image.open(file_path))
        else:
            return ""
    except Exception as e:
        logger.error("Failed to extract text from %s: %s", file_path, e)
        return ""
