import csv
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MERGE = ROOT / "src/pipeline/02_curate/02_scoring/02_active_speaker/02_merge_shards.py"
PYTHON = sys.executable


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ActiveSpeakerMergeTest(unittest.TestCase):
    def test_exact_coverage_merges(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            manifest = temp / "manifest.csv"
            rows = [{"clip_id": f"c{i}"} for i in range(4)]
            write_csv(manifest, rows, ["clip_id"])
            manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
            run_dir = temp / "run"
            fields = [
                "clip_id", "voiced_ms", "visible_active_speech_ratio",
                "unexplained_speech_ratio", "longest_unexplained_speech_ms",
                "static_speech_ratio", "asd_disagreement_ratio", "temporal_decision",
                "temporal_reason", "model_versions", "config_hash",
            ]
            for shard_index, (start, end) in enumerate(((0, 2), (2, 4))):
                shard = run_dir / "shards" / f"s{shard_index}"
                shard.mkdir(parents=True)
                shard_rows = []
                for i in range(start, end):
                    shard_rows.append({
                        "clip_id": f"c{i}", "voiced_ms": 1000,
                        "visible_active_speech_ratio": 1, "unexplained_speech_ratio": 0,
                        "longest_unexplained_speech_ms": 0, "static_speech_ratio": 0,
                        "asd_disagreement_ratio": 0, "temporal_decision": "pass",
                        "temporal_reason": "", "model_versions": "{}",
                        "config_hash": "same",
                    })
                write_csv(shard / "asd_clip_scores.csv", shard_rows, fields)
                write_csv(shard / "failures.csv", [],
                          ["clip_id", "file_path", "error_type", "error_message"])
                with gzip.open(shard / "asd_timeline.jsonl.gz", "wt", encoding="utf-8") as handle:
                    for i in range(start, end):
                        handle.write(json.dumps({"clip_id": f"c{i}", "bin_index": 0}) + "\n")
                (shard / "run_config.json").write_text(json.dumps({
                    "config_hash": "same", "manifest_sha256": manifest_hash,
                    "batch_start": start, "batch_end": end, "output_rows": end-start,
                    "model_versions": {}, "policy": {}, "run_id": "run",
                    "coverage_passed": True,
                }), encoding="utf-8")

            second_config_path = run_dir / "shards" / "s1" / "run_config.json"
            second_config = json.loads(second_config_path.read_text("utf-8"))
            second_config["config_hash"] = "different"
            second_config_path.write_text(json.dumps(second_config), encoding="utf-8")
            failed = subprocess.run([PYTHON, str(MERGE), "--manifest", str(manifest),
                                     "--run_dir", str(run_dir)], cwd=ROOT,
                                    capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((run_dir / "asd_clip_scores.csv").exists())
            second_config["config_hash"] = "same"
            second_config_path.write_text(json.dumps(second_config), encoding="utf-8")

            subprocess.run([PYTHON, str(MERGE), "--manifest", str(manifest),
                            "--run_dir", str(run_dir)], cwd=ROOT, check=True,
                           capture_output=True, text=True)
            self.assertTrue((run_dir / "asd_clip_scores.csv").is_file())
            config = json.loads((run_dir / "run_config.json").read_text("utf-8"))
            self.assertTrue(config["coverage_passed"])
            self.assertEqual(config["score_rows"], 4)


if __name__ == "__main__":
    unittest.main()
