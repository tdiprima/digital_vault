"""KMeans clustering and batch LLM-based cluster naming for file records."""

import json
import logging
import re
from collections import defaultdict

import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans

from records import FileRecord

logger = logging.getLogger(__name__)

_CLUSTER_SAMPLE_SIZE = 3


def cluster_records(records: list[FileRecord], n_clusters: int) -> list[FileRecord]:
    """Assign a KMeans cluster label to each record. Returns new records with 'cluster' set."""
    effective_n = min(n_clusters, len(records))
    if effective_n < n_clusters:
        logger.warning(
            "Fewer records (%d) than requested clusters (%d); using %d clusters",
            len(records),
            n_clusters,
            effective_n,
        )
    embeddings = np.array([r["embed"] for r in records])
    labels = KMeans(n_clusters=effective_n, random_state=42).fit(embeddings).labels_
    return [{**r, "cluster": int(label)} for r, label in zip(records, labels)]


def build_cluster_samples(records: list[FileRecord]) -> dict[int, list[str]]:
    """Collect up to _CLUSTER_SAMPLE_SIZE text samples per cluster for LLM naming."""
    groups: dict[int, list[str]] = defaultdict(list)
    for r in records:
        cluster_id = r.get("cluster")
        if cluster_id is not None and len(groups[cluster_id]) < _CLUSTER_SAMPLE_SIZE:
            groups[cluster_id].append(r["text"])
    return dict(groups)


def name_clusters(
    cluster_samples: dict[int, list[str]],
    client: OpenAI,
    llm_model: str,
) -> dict[int, str]:
    """Name all clusters in a single LLM call. Returns mapping of cluster_id to sanitized folder name."""
    clusters_text = "\n\n".join(
        f'Cluster {cid}:\n{"; ".join(texts)}'
        for cid, texts in cluster_samples.items()
    )
    prompt = (
        "For each cluster below, provide a short descriptive folder name.\n"
        "Return ONLY a JSON object mapping cluster ID (as a string) to folder name.\n"
        'Do NOT prefix names with "folder_name_".\n'
        "Use only alphanumeric characters, spaces, underscores, or hyphens.\n\n"
        f"{clusters_text}"
    )

    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("LLM call for cluster naming failed: %s", e)
        return {cid: f"cluster_{cid}" for cid in cluster_samples}

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        logger.error("LLM response contained no JSON object. Raw: %.200s", raw)
        return {cid: f"cluster_{cid}" for cid in cluster_samples}

    try:
        names = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.error("Failed to parse cluster names JSON: %s. Raw: %.200s", e, raw)
        return {cid: f"cluster_{cid}" for cid in cluster_samples}

    result: dict[int, str] = {}
    for cid_str, name in names.items():
        try:
            cid = int(cid_str)
        except ValueError:
            logger.warning("Skipping unexpected cluster ID in LLM response: %r", cid_str)
            continue
        sanitized = (
            "".join(c for c in name if c.isalnum() or c in (" ", "_", "-"))
            .replace(" ", "_")
            .strip("_")
        )
        result[cid] = sanitized or f"cluster_{cid}"

    for cid in cluster_samples:
        if cid not in result:
            logger.warning("Cluster %d missing from LLM response, using fallback name", cid)
            result[cid] = f"cluster_{cid}"

    return result
