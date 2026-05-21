"""Shared data types for the digital vault pipeline."""

from typing import NotRequired, TypedDict

import numpy as np


class FileRecord(TypedDict):
    """A single file's extracted text, embedding, and optional cluster assignment."""
    path: str
    text: str
    embed: np.ndarray
    cluster: NotRequired[int]
