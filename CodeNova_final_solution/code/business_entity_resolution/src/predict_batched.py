#!/usr/bin/env python3
"""
Batched prediction for large test sets - processes S1 entities in chunks.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from blocking import generate_candidates
from features import build_features
from io_utils import read_source, write_candidate_pairs, write_matching_results
from model import load_model, score_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="../../../dataset/test")
    ap.add_argument("--model", default="../models/model.pkl")
    ap.add_argument("--output-dir", default="../../../output")
    ap.add_argument("--batch-size", type=int, default=50000, help="S1 entities per batch")
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    print(f"[predict] loading test data from {test_dir}")
    s1 = read_source(test_dir / "test_source1.tsv")
    s2 = read_source(test_dir / "test_source2.tsv")
    s3 = read_source(test_dir / "test_source3.tsv")
    print(f"[predict] S1={len(s1)} S2={len(s2)} S3={len(s3)}")

    # Load model
    clf, threshold = load_model(args.model)
    print(f"[predict] using decision threshold={threshold:.2f}")

    # Prep S2/S3 once (they're the same for all batches)
    from blocking import _prep, _index_bundle, _candidates_for_row
    
    print("[predict] preparing S2/S3 indices...")
    s2_prep = _prep(s2)
    s3_prep = _prep(s3)
    s2_idx = _index_bundle(s2_prep)
    s3_idx = _index_bundle(s3_prep)

    s1_ids = s1["entity_id"].tolist()
    all_matches = {}
    all_candidates = {}

    # Process S1 in batches
    for batch_start in range(0, len(s1), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(s1))
        s1_batch = s1.iloc[batch_start:batch_end].reset_index(drop=True)
        print(f"[predict] processing batch {batch_start//args.batch_size + 1}: S1[{batch_start}:{batch_end}] ({len(s1_batch)} entities)")

        # Prep this S1 batch
        s1_batch_prep = _prep(s1_batch)

        # Generate candidates for this batch
        batch_rows = []
        for row in s1_batch_prep.itertuples(index=False):
            for source_label, source_idx in (("S2", s2_idx), ("S3", s3_idx)):
                scores = _candidates_for_row(row, *source_idx)
                if not scores:
                    continue
                ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                ranked = ranked[:20]  # MAX_CANDIDATES_PER_SOURCE
                for cid, sc in ranked:
                    batch_rows.append({
                        "source1_entity_id": row.entity_id,
                        "candidate_entity_id": cid,
                        "candidate_source": source_label,
                        "block_score": sc,
                    })

        if not batch_rows:
            for sid in s1_batch["entity_id"]:
                all_candidates[sid] = []
                all_matches[sid] = []
            continue

        batch_cands = pd.DataFrame(batch_rows)
        batch_cands = batch_cands.sort_values("block_score", ascending=False)
        batch_cands = batch_cands.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"], keep="first")

        # Build features and score
        print(f"  candidates: {len(batch_cands)}, building features...")
        feat = build_features(batch_cands, s1_batch, s2, s3)
        feat["prob"] = score_pairs(clf, feat)

        # Collect candidates
        for sid, group in batch_cands.groupby("source1_entity_id")["candidate_entity_id"]:
            all_candidates[sid] = list(dict.fromkeys(group))

        # Collect matches (above threshold)
        kept = feat[feat["prob"] >= threshold]
        for sid, group in kept.groupby("source1_entity_id")["candidate_entity_id"]:
            all_matches[sid] = list(dict.fromkeys(group))

        # Ensure all S1 in batch have entries
        for sid in s1_batch["entity_id"]:
            if sid not in all_candidates:
                all_candidates[sid] = []
            if sid not in all_matches:
                all_matches[sid] = []

        print(f"  matches in batch: {sum(1 for v in all_matches.values() if v)}")

    # Write outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_matching_results(output_dir / "matching_results.tsv", all_matches, s1_ids)
    write_candidate_pairs(output_dir / "candidate_pairs.tsv", all_candidates, s1_ids)
    print(f"[predict] wrote {output_dir / 'matching_results.tsv'} and {output_dir / 'candidate_pairs.tsv'}")

    n_matched = sum(1 for sid in s1_ids if all_matches.get(sid))
    print(f"[predict] {n_matched}/{len(s1_ids)} S1 entities received at least one match ({len(s1_ids) - n_matched} predicted as singletons)")


if __name__ == "__main__":
    main()