#!/usr/bin/env python3
"""
Run the full pipeline on the test set and write the two required outputs:
    output/matching_results.tsv   (scored on the leaderboard)
    output/candidate_pairs.tsv    (blocking stage, for pipeline auditing)

Usage:
    python3 predict.py --test-dir /path/to/dataset/test \
                        --model /path/to/models/model.pkl \
                        --output-dir /path/to/output
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from blocking import generate_candidates
from features import build_features
from io_utils import read_source, write_candidate_pairs, write_matching_results
from model import load_model, score_pairs


def basic_format_checks(matches, candidates, s1_ids, s2_ids, s3_ids):
    """A lightweight local stand-in for the official
    utils/validate_submission.py (which is not available in this
    environment). Run the OFFICIAL validator before submitting -- this
    only catches the same rules described in the problem statement."""
    issues = []
    valid_ids = s2_ids | s3_ids
    s1_set = set(s1_ids)

    if set(matches.keys()) != s1_set:
        issues.append("matching_results: entity set does not exactly equal test Source-1 ids")
    if set(candidates.keys()) != s1_set:
        issues.append("candidate_pairs: entity set does not exactly equal test Source-1 ids")

    for sid, ids in matches.items():
        if len(ids) != len(set(ids)):
            issues.append(f"matching_results: duplicate ids for {sid}")
        bad = [i for i in ids if i not in valid_ids or i.startswith("S1-")]
        if bad:
            issues.append(f"matching_results: {sid} references invalid ids {bad}")
        cand_set = set(candidates.get(sid, []))
        not_in_cand = [i for i in ids if i not in cand_set]
        if not_in_cand:
            issues.append(f"matching_results: {sid} has matches not present in its own candidates: {not_in_cand}")

    for sid, ids in candidates.items():
        if len(ids) != len(set(ids)):
            issues.append(f"candidate_pairs: duplicate ids for {sid}")
        bad = [i for i in ids if i not in valid_ids or i.startswith("S1-")]
        if bad:
            issues.append(f"candidate_pairs: {sid} references invalid ids {bad}")

    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="../../dataset/test")
    ap.add_argument("--model", default="../models/model.pkl")
    ap.add_argument("--output-dir", default="../../output")
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    print(f"[predict] loading test data from {test_dir}")
    s1 = read_source(test_dir / "test_source1.tsv")
    s2 = read_source(test_dir / "test_source2.tsv")
    s3 = read_source(test_dir / "test_source3.tsv")
    print(f"[predict] S1={len(s1)} S2={len(s2)} S3={len(s3)}")

    print("[predict] running blocking...")
    candidates = generate_candidates(s1, s2, s3)
    print(f"[predict] blocking produced {len(candidates)} candidate pairs "
          f"covering {candidates['source1_entity_id'].nunique() if len(candidates) else 0} / {len(s1)} S1 entities")

    print("[predict] building features + scoring with trained model...")
    feat = build_features(candidates, s1, s2, s3)
    clf, threshold = load_model(args.model)
    feat["prob"] = score_pairs(clf, feat) if len(feat) else []
    print(f"[predict] using decision threshold={threshold:.2f}")

    s1_ids = s1["entity_id"].tolist()

    # candidate_pairs.tsv: everything blocking produced (the exact set fed to the model)
    cand_map = {
        sid: list(dict.fromkeys(g)) for sid, g in candidates.groupby("source1_entity_id")["candidate_entity_id"]
    } if len(candidates) else {}

    # matching_results.tsv: only pairs the model scored above threshold
    kept = feat[feat["prob"] >= threshold] if len(feat) else feat
    match_map = {
        sid: list(dict.fromkeys(g)) for sid, g in kept.groupby("source1_entity_id")["candidate_entity_id"]
    } if len(kept) else {}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_matching_results(output_dir / "matching_results.tsv", match_map, s1_ids)
    write_candidate_pairs(output_dir / "candidate_pairs.tsv", cand_map, s1_ids)
    print(f"[predict] wrote {output_dir / 'matching_results.tsv'} and {output_dir / 'candidate_pairs.tsv'}")

    n_matched_entities = sum(1 for sid in s1_ids if match_map.get(sid))
    print(f"[predict] {n_matched_entities}/{len(s1_ids)} S1 entities received at least one match "
          f"({len(s1_ids) - n_matched_entities} predicted as singletons)")

    # Expand to include every S1 id (with an empty list for singletons) before
    # running local checks -- matching_results.tsv / candidate_pairs.tsv on
    # disk already do this via write_id_list_tsv's use of s1_ids.
    full_match_map = {sid: match_map.get(sid, []) for sid in s1_ids}
    full_cand_map = {sid: cand_map.get(sid, []) for sid in s1_ids}
    issues = basic_format_checks(full_match_map, full_cand_map, s1_ids, set(s2["entity_id"]), set(s3["entity_id"]))
    if issues:
        print("[predict] FORMAT WARNINGS (fix before submitting; run the official validator too):")
        for issue in issues:
            print("  -", issue)
    else:
        print("[predict] basic local format checks passed. Still run utils/validate_submission.py before submitting.")


if __name__ == "__main__":
    main()
