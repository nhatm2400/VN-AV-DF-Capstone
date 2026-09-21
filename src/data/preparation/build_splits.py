"""Split reviewed REAL clips by connected speaker/source groups before synthesis.

Retains union-find from 05_build_labels, without four-fake coverage requirements.
"""
import argparse
import hashlib
import math
import random
from collections import defaultdict
from src.data.preparation.manifest_io import read_rows, unique_rows, write_rows

STRICT_PROTOCOL = "speaker_source_disjoint"
SINGLE_SPEAKER_PROTOCOL = "source_disjoint_single_speaker"


def validate_protocol(rows, protocol):
    if protocol not in (STRICT_PROTOCOL, SINGLE_SPEAKER_PROTOCOL):
        raise ValueError(f"Unknown split protocol: {protocol}")
    if protocol == SINGLE_SPEAKER_PROTOCOL:
        speakers = {node[1] for row in rows for node in identity_nodes(row)
                    if node[0] == "speaker"}
        if len(speakers) != 1:
            raise ValueError("Single-speaker protocol requires exactly one confirmed speaker")


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


def identity_nodes(row, protocol=STRICT_PROTOCOL):
    speaker = str(row.get("speaker_id", "")).strip()
    source = str(row.get("source_video", "")).strip()
    if speaker.lower() in ("", "-1", "unknown", "nan") or not source:
        raise ValueError(f"Review speaker_id and source_video first: {row['clip_id']}")
    speakers = {speaker} | {s.strip() for s in row.get("speaker_ids", "").split(";") if s.strip()}
    nodes = [("video", source)]
    if protocol == STRICT_PROTOCOL:
        nodes += [("speaker", s) for s in sorted(speakers)]
    if row.get("canonical_source_id"):
        nodes.append(("video", row["canonical_source_id"].strip()))
    if row.get("program_id") and row.get("episode_id"):
        nodes.append(("episode", row["program_id"], row["episode_id"]))
    return nodes


def verify_splits(rows):
    protocols = {r.get("split_protocol", STRICT_PROTOCOL) for r in rows}
    if len(protocols) != 1:
        raise ValueError("Expected one split protocol")
    protocol = next(iter(protocols))
    validate_protocol(rows, protocol)
    seen = {}
    unique_rows(rows)
    for row in rows:
        if row.get("split") not in ("train", "val", "test"):
            raise ValueError("Invalid split")
        for node in identity_nodes(row, protocol):
            if node in seen and seen[node] != row["split"]:
                raise ValueError(f"Leakage: {node}")
            seen[node] = row["split"]


def split_real_rows(rows, ratios=(0.7, 0.15, 0.15), seed=42, protocol=STRICT_PROTOCOL):
    if len(ratios) != 3 or any(not math.isfinite(r) or r < 0 for r in ratios) or abs(sum(ratios)-1) > 1e-6:
        raise ValueError("Ratios must be three nonnegative numbers summing to 1")
    index = unique_rows(rows)
    validate_protocol(rows, protocol)
    uf = UnionFind()
    for cid, row in index.items():
        if str(row.get("label", "0")) != "0" or row.get("parent_clip_id") or row.get("source_clip"):
            raise ValueError("Split real master clips before generating variants")
        if row.get("split"):
            raise ValueError("Input already has a split; use the locked manifest")
        if row.get("decision") != "keep":
            raise ValueError(f"Manual review decision=keep required: {cid}")
        for node in identity_nodes(row, protocol):
            uf.union(("clip", cid), node)
    groups = defaultdict(list)
    for cid, row in sorted(index.items()):
        groups[uf.find(("clip", cid))].append(row)
    positive = [i for i, r in enumerate(ratios) if r > 0]
    if len(groups) < len(positive):
        raise ValueError(f"Too few independent groups for {protocol}; add sources instead of breaking groups")
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
        output.extend(dict(r, split=names[choice], group_id=group_key, label=0,
                           split_protocol=protocol) for r in members)
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
                split_protocol=source.get("split_protocol", STRICT_PROTOCOL),
                speaker_id=source["speaker_id"], source_video=source["source_video"],
                audio_source_clip_id=audio_parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ratios", default="0.7,0.15,0.15")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocol", choices=(STRICT_PROTOCOL, SINGLE_SPEAKER_PROTOCOL),
                        default=STRICT_PROTOCOL)
    parser.add_argument("--single-speaker-id", help="Confirmed shared speaker ID; only for the single-speaker protocol")
    args = parser.parse_args()
    rows = read_rows(args.input)
    if args.single_speaker_id is not None:
        speaker = args.single_speaker_id.strip()
        if args.protocol != SINGLE_SPEAKER_PROTOCOL or speaker.lower() in ("", "-1", "unknown", "nan"):
            parser.error("--single-speaker-id requires a valid ID and the single-speaker protocol")
        for row in rows:
            if row.get("speaker_id", "").strip() not in ("", speaker):
                parser.error("Existing speaker_id conflicts with --single-speaker-id")
            row["speaker_id"] = speaker
    output = split_real_rows(rows, tuple(map(float, args.ratios.split(','))), args.seed, args.protocol)
    write_rows(args.out, output)
    print({name: sum(r['split'] == name for r in output) for name in ('train','val','test')})
    print(f"Independent groups: {len({r['group_id'] for r in output})}; output: {args.out}")
    print(f"Split protocol: {args.protocol}")
    if args.protocol == SINGLE_SPEAKER_PROTOCOL:
        print("Preliminary same-speaker evaluation only; not evidence of generalization to unseen speakers.")


if __name__ == "__main__":
    main()
