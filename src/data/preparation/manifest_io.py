"""Small CSV helpers shared by current data tools."""
import csv
import os
import re
from pathlib import Path


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def safe_id(value):
    value = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Invalid ID: {value!r}")
    return value


def unique_rows(rows, field="clip_id"):
    index = {}
    for row in rows:
        key = safe_id(row.get(field))
        if key in index:
            raise ValueError(f"Duplicate {field}: {key}")
        index[key] = row
    return index


def write_rows(path, rows):
    """Publish a new manifest; never overwrite an existing dataset version."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise ValueError("Refusing to publish an empty manifest")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if path.exists():
        raise FileExistsError(path)
    os.replace(partial, path)
