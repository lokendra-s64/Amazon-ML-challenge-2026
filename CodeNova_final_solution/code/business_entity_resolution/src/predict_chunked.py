#!/usr/bin/env python3
"""
Memory-efficient chunked prediction for large test sets.
Uses only token-based blocking (no TF-IDF) to avoid OOM.
Processes S1 in chunks, S2/S3 loaded once.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from blocking import _prep, _index_bundle, _candidates_for_row
from features import build_features
from io_utils import read_source, write_candidate_pairs, write_matching_results
from model import load_model, score_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="../../../dataset/test")
    ap.add_argument("--model", default="../models/model.pkl")
    ap.add_argument("--output-dir", default="../../../output")
    ap.add_argument("--s1-chunk", type=int, default=20000, help="S1 entities per chunk")
    ap.add_argument("--s2-limit", type=int, default=200000, help="Max S2 records to load")
    ap.add_argument("--s3-limit", type=int, default=200000, help="Max S3 records to load")
    ap.add_argument("--max-cand", type=int, default=10, help="Max candidates per source per S1")
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    print(f"[predict] loading test data from {test_dir}")

    # Load S2/S3 with limits for memory
    print("Loading S2...")
    s2 = read_source(test_dir / "test_source2.tsv", nrows=args.s2_limit)
    print("Loading S3...")
    s3 = read_source(test_dir / "test_source3.tsv", nrows=args.s3_limit)
    print(f"S2={len(s2)} S3={len(s3)} (limited)")

    # Load model
    clf, threshold = load_model(args.model)
    print(f"[predict] threshold={threshold:.2f}")

    # Prep S2/S3 once
    print("Preparing S2/S3 indices...")
    s2_prep = _prep(s2)
    s3_prep = _prep(s3)
    s2_idx = _index_bundle(s2_prep)
    s3_idx = _index_bundle(s3_prep)

    # Load all S1 IDs first
    print("Loading S1 IDs...")
    s1_ids = pd.read_csv(test_dir / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=["entity_id"])["entity_id"].tolist()
    print(f"Total S1 entities: {len(s1_ids)}")

    all_matches = {}
    all_candidates = {}

    # Process S1 in chunks
    for chunk_start in range(0, len(s1_ids), args.s1_chunk):
        chunk_end = min(chunk_start + args.s1_chunk, len(s1_ids))
        print(f"\nChunk {chunk_start//args.s1_chunk + 1}: S1[{chunk_start}:{chunk_end}] ({chunk_end-chunk_start} entities)")

        # Load this chunk of S1
        s1_chunk = read_source(test_dir / "test_source1.tsv", nrows=chunk_end)[chunk_start:chunk_end].reset_index(drop=True)
        s1_prep = _prep(s1_chunk)

        # Generate candidates for this chunk
        batch_rows = []
        for row in s1_prep.itertuples(index=False):
            for source_label, source_idx in (("S2", s2_idx), ("S3", s3_idx)):
                scores = _candidates_for_row(row, *source_idx)
                if not scores:
                    continue
                ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                ranked = ranked[:args.max_cand]
                for cid, sc in ranked:
                    batch_rows.append({
                        "source1_entity_id": row.entity_id,
                        "candidate_entity_id": cid,
                        "candidate_source": source_label,
                        "block_score": sc,
                    })

        if not batch_rows:
            for sid in s1_chunk["entity_id"]:
                all_candidates[sid] = []
                all_matches[sid] = []
            continue

        batch_cands = pd.DataFrame(batch_rows)
        batch_cands = batch_cands.sort_values("block_score", ascending=False)
        batch_cands = batch_cands.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"], keep="first")

        # Build features and score
        print(f"  Candidates: {len(batch_cands)}, building features...")
        feat = build_features(batch_cands, s1_chunk, s2, s3)
        feat["prob"] = score_pairs(clf, feat)

        # Collect candidates
        for sid, group in batch_cands.groupby("source1_entity_id")["candidate_entity_id"]:
            all_candidates[sid] = list(dict.fromkeys(group))

        # Collect matches
        kept = feat[feat["prob"] >= threshold]
        for sid, group in kept.groupby("source1_entity_id")["candidate_entity_id"]:
            all_matches[sid] = list(dict.fromkeys(group))

        # Ensure all S1 in chunk have entries
        for sid in s1_chunk["entity_id"]:
            if sid not in all_candidates:
                all_candidates[sid] = []
            if sid not in all_matches:
                all_matches[sid] = []

        matched_count = sum(1 for v in all_matches.values() if v)
        print(f"  Matches so far: {matched_count}")

    # Write outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_matching_results(output_dir / "matching_results.tsv", all_matches, s1_ids)
    write_candidate_pairs(output_dir / "candidate_pairs.tsv", all_candidates, s1_ids)
    print(f"\n[predict] wrote {output_dir / 'matching_results.tsv'} and {output_dir / 'candidate_pairs.tsv'}")

    n_matched = sum(1 for sid in s1_ids if all_matches.get(sid))
    print(f"[predict] {n_matched}/{len(s1_ids)} S1 entities matched ({len(s1_ids) - n_matched} singletons)")


if __name__ == "__main__":
    main()