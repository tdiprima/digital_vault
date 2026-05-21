"""File movement and vault organization for the organize pipeline mode."""

import logging
import os
import shutil
from pathlib import Path

from records import FileRecord

logger = logging.getLogger(__name__)


def move_file(source_path: str, destination_folder: str) -> str:
    """Move source_path into destination_folder, creating it if needed. Returns the new file path."""
    dest_dir = Path(destination_folder)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(source_path).name
    shutil.move(source_path, dest_path)
    return str(dest_path)


def organize_vault(
    records: list[FileRecord],
    vault_dir: str,
    cluster_names: dict[int, str],
) -> list[FileRecord]:
    """Move each file into its named cluster folder inside vault_dir. Returns records with updated paths."""
    updated: list[FileRecord] = []
    for record in records:
        cluster_id = record.get("cluster")
        if cluster_id is None:
            logger.warning("Record has no cluster label, skipping move: %s", record["path"])
            updated.append(record)
            continue
        folder_name = cluster_names.get(cluster_id, f"cluster_{cluster_id}")
        dest_folder = os.path.join(vault_dir, folder_name)
        try:
            new_path = move_file(record["path"], dest_folder)
            updated.append({**record, "path": new_path})
        except OSError as e:
            logger.error(
                "Failed to move %s to %s: %s", record["path"], dest_folder, e
            )
            updated.append(record)
    logger.info("Files organized into %s", vault_dir)
    return updated
