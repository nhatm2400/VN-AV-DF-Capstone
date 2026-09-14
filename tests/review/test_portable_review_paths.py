import csv
from pathlib import Path

from src.tools.review.clip_review import default_out, load_clips


def test_assignment_paths_resolve_relative_to_csv(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    clip = media / "clip_1.mp4"
    clip.write_bytes(b"video")
    assignment = tmp_path / "assignment_linh.csv"
    with assignment.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_id", "file_path"])
        writer.writeheader()
        writer.writerow({"clip_id": "clip_1", "file_path": "media/clip_1.mp4"})

    rows = load_clips(str(assignment), "file_path")

    assert Path(rows[0]["_abspath"]) == clip
    assert Path(default_out(str(assignment), "linh")) == tmp_path / "review_linh.csv"
