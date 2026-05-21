"""RAG chat function factory. Returns a Gradio-compatible callback with explicit dependencies."""

import logging
from pathlib import Path
from typing import Callable

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from embedding import semantic_search
from records import FileRecord

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Answer the user's question based on the provided document excerpts. "
    "If the answer isn't in the documents, say so. "
    "Cite which file(s) the information came from."
)


def create_chat_fn(
    records: list[FileRecord],
    embedder: SentenceTransformer,
    client: OpenAI,
    llm_model: str,
    top_k: int,
) -> Callable[[str, list], str]:
    """Return a Gradio chat callback closed over the indexed records and clients."""

    def chat_query(message: str, history: list) -> str:
        results = semantic_search(message, records, embedder, top_k)
        if not results:
            return "No relevant files found for your query."

        context_texts = [
            f"From {Path(r['path']).name}:\n{r['text']}"
            for r in results
        ]
        context = "\n\n---\n\n".join(context_texts)

        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Documents:\n{context}\n\nQuestion: {message}",
                    },
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("LLM call failed during chat_query: %s", e)
            return f"Error retrieving response: {e}"

    return chat_query
