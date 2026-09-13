"""Split reviewed REAL clips by connected speaker/source groups before synthesis.

Retains union-find from 05_build_labels, without four-fake coverage requirements.
"""
import argparse
import hashlib
import math
import random
from collections import defaultdict
from src.data.preparation.manifest_io import read_rows, unique_rows, write_rows


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, node):
        self.parent.setdefault(node, node)
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:
            self.parent[node], node = root, self.parent[node]
        return root

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def identity_nodes(row):
    speaker = str(row.get("speaker_id", "")).strip()
    source = str(row.get("source_video", "")).strip()
    if speaker.lower() in ("", "-1", "unknown", "nan") or not source:
        raise ValueError(f"Review speaker_id and source_video first: {row['clip_id']}")
    speakers = {speaker} | {s.strip() for s in row.get("speaker_ids", "").split(";") if s.strip()}
    nodes = [("speaker", s) for s in sorted(speakers)] + [("video", source)]
    if row.get("canonical_source_id"):
        nodes.append(("video", row["canonical_source_id"].strip()))
    if row.get("program_id") and row.get("episode_id"):
        nodes.append(("episode", row["program_id"], row["episode_id"]))
    return nodes


def verify_splits(rows):
    seen = {}
    unique_rows(rows)
    for row in rows:
        if row.get("split") not in ("train", "val", "test"):
            raise ValueError("Invalid split")
        for node in identity_nodes(row):
            if node in seen and seen[node] != row["split"]:
                raise ValueError(f"Leakage: {node}")
            seen[node] = row["split"]


def split_real_rows(rows, ratios=(0.7, 0.15, 0.15), seed=42):
    if len(ratios) != 3 or any(not math.isfinite(r) or r < 0 for r in ratios) or abs(sum(ratios)-1) > 1e-6:
        raise ValueError("Ratios must be three nonnegative numbers summing to 1")
    index = unique_rows(rows)
    uf = UnionFind()
    for cid, row in index.items():
        if str(row.get("label", "0")) != "0" or row.get("parent_clip_id") or row.get("source_clip"):
            raise ValueError("Split real master clips before generating variants")
        if row.get("split"):
            raise ValueError("Input already has a split; use the locked manifest")
        if row.get("decision") != "keep":
            raise ValueError(f"Manual review decision=keep required: {cid}")
        for node in identity_nodes(row):
            uf.union(("clip", cid), node)
    groups = defaultdict(list)
    for cid, row in sorted(index.items()):
        groups[uf.find(("clip", cid))].append(row)
    positive = [i for i, r in enumerate(ratios) if r > 0]
    if len(groups) < len(positive):
        raise ValueError("Too few independent speaker/source groups; add sources instead of splitting a host group")
    buckets = defaultdict(list)
    for members in groups.values():
        buckets[len(members)].append(members)
    rng = random.Random(seed)
    order = []
    for size in sorted(buckets, reverse=True):
        bucket = sorted(buckets[size], key=lambda group: group[0]["clip_id"])
        rng.shuffle(bucket)
        order.extend(bucket)
    filled = [0, 0, 0]
    targets = [r * len(rows) for r in ratios]
    names = ("train", "val", "test")
    output = []
    for position, members in enumerate(order):
        empty = [i for i in positive if filled[i] == 0]
        candidates = empty if len(order)-position == len(empty) else positive
        choice = max(candidates, key=lambda i: (targets[i]-filled[i])/targets[i])
        filled[choice] += len(members)
        group_key = "group_" + hashlib.sha256("\n".join(sorted(r["clip_id"] for r in members)).encode()).hexdigest()[:16]
        output.extend(dict(r, split=names[choice], group_id=group_key, label=0) for r in members)
    output.sort(key=lambda row: row["clip_id"])
    verify_splits(output)
    return output


def inherit_variant_split(variant, real_rows, held_out_generators=()):
    """Validate root provenance, including replacement audio, for generator adapters."""
    index = unique_rows(real_rows)
    parent = variant.get("source_clip")
    if parent not in index:
        raise ValueError("Unknown real source_clip")
    source = index[parent]
    audio_parent = variant.get("audio_source_clip_id") or parent
    if audio_parent not in index or index[audio_parent]["split"] != source["split"]:
        raise ValueError("Replacement audio crosses split or has no source")
    if variant.get("generator", "") in held_out_generators and source["split"] != "test":
        raise ValueError("Held-out generator is only allowed in test")
    if variant.get("split") and variant["split"] != source["split"]:
        raise ValueError("Variant split differs from real source")
    return dict(variant, split=source["split"], group_id=source["group_id"],
                speaker_id=source["speaker_id"], source_video=source["source_video"],
                audio_source_clip_id=audio_parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ratios", default="0.7,0.15,0.15")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = split_real_rows(read_rows(args.input), tuple(map(float, args.ratios.split(','))), args.seed)
    write_rows(args.out, output)
    print({name: sum(r['split'] == name for r in output) for name in ('train','val','test')})
    print(f"Independent groups: {len({r['group_id'] for r in output})}; output: {args.out}")


if __name__ == "__main__":
    main()
