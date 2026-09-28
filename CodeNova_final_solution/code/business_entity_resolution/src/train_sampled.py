#!/usr/bin/env python3
"""
Train on a sampled subset of the training data for practical runtime.
Samples S1, S2, S3 proportionally to maintain match relationships.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from blocking import generate_candidates
from evaluate import macro_f_half
from features import FEATURE_COLUMNS, build_features
from io_utils import read_ground_truth, read_source
from model import save_model, train_model


def blocking_recall_ceiling(candidates: pd.DataFrame, ground_truth: dict[str, set[str]]) -> float:
    cand_by_s1: dict[str, set[str]] = {}
    for sid, group in candidates.groupby("source1_entity_id")["candidate_entity_id"]:
        cand_by_s1[sid] = set(group)

    total_true, total_found = 0, 0
    for sid, truth_ids in ground_truth.items():
        if not truth_ids:
            continue
        found = truth_ids & cand_by_s1.get(sid, set())
        total_true += len(truth_ids)
        total_found += len(found)
    return total_found / total_true if total_true else 1.0


def sweep_threshold(val_feat: pd.DataFrame, val_probs: np.ndarray, val_truth: dict[str, set[str]], val_s1_ids: list[str]):
    best_thr, best_score = 0.5, -1.0
    thresholds = np.arange(0.05, 0.96, 0.01)
    df = val_feat[["source1_entity_id", "candidate_entity_id"]].copy()
    df["prob"] = val_probs

    for thr in thresholds:
        kept = df[df["prob"] >= thr]
        pred_by_s1: dict[str, set[str]] = {
            sid: set(g) for sid, g in kept.groupby("source1_entity_id")["candidate_entity_id"]
        }
        truth_subset = {sid: val_truth.get(sid, set()) for sid in val_s1_ids}
        score = macro_f_half(pred_by_s1, truth_subset)
        if score > best_score:
            best_score, best_thr = score, float(thr)
    return best_thr, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="../../../dataset/train")
    ap.add_argument("--model-out", default="../models/model.pkl")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-s1", type=int, default=10000, help="Number of S1 entities to sample")
    ap.add_argument("--sample-s23", type=int, default=100000, help="Number of S2+S3 entities to sample")
    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    print(f"[train] loading data from {train_dir}")
    
    # Load full ground truth first to know which S1 have matches
    print("[train] loading ground truth...")
    gt_full = read_ground_truth(train_dir / "train_ground_truth.tsv")
    print(f"[train] full GT: {len(gt_full)} S1 entities with matches")
    
    # Load S1 with nrows to sample
    print("[train] loading and sampling S1...")
    s1_full = read_source(train_dir / "train_source1.tsv")
    
    # Prioritize S1 entities that have ground truth matches
    s1_with_matches = [sid for sid in s1_full["entity_id"] if sid in gt_full and gt_full[sid]]
    s1_without_matches = [sid for sid in s1_full["entity_id"] if sid not in gt_full or not gt_full[sid]]
    
    rng = np.random.default_rng(args.seed)
    
    # Sample S1: prefer entities with matches
    n_with = min(len(s1_with_matches), args.sample_s1 // 2)
    n_without = min(len(s1_without_matches), args.sample_s1 - n_with)
    
    sampled_with = rng.choice(s1_with_matches, size=n_with, replace=False) if n_with > 0 else []
    sampled_without = rng.choice(s1_without_matches, size=n_without, replace=False) if n_without > 0 else []
    sampled_s1_ids = set(sampled_with) | set(sampled_without)
    
    s1 = s1_full[s1_full["entity_id"].isin(sampled_s1_ids)].reset_index(drop=True)
    print(f"[train] sampled S1: {len(s1)} ({n_with} with matches, {n_without} without)")
    
    # Filter GT to sampled S1
    ground_truth = {k: v for k, v in gt_full.items() if k in sampled_s1_ids}
    
    # Load and sample S2, S3
    print("[train] loading and sampling S2, S3...")
    s2_full = read_source(train_dir / "train_source2.tsv")
    s3_full = read_source(train_dir / "train_source3.tsv")
    
    # IMPORTANT: Include all true match entities from ground truth for sampled S1
    true_match_ids = set()
    for v in ground_truth.values():
        true_match_ids.update(v)
    
    # Split true matches by source
    true_s2_ids = {eid for eid in true_match_ids if eid.startswith("S2-")}
    true_s3_ids = {eid for eid in true_match_ids if eid.startswith("S3-")}
    
    print(f"[train] true matches to preserve: S2={len(true_s2_ids)}, S3={len(true_s3_ids)}")
    
    # Get available IDs in full sources
    available_s2 = set(s2_full["entity_id"])
    available_s3 = set(s3_full["entity_id"])
    
    # Find which true matches exist in the source files
    true_s2_in_file = true_s2_ids & available_s2
    true_s3_in_file = true_s3_ids & available_s3
    print(f"[train] true matches present in files: S2={len(true_s2_in_file)}, S3={len(true_s3_in_file)}")
    
    # Sample additional entities to reach target
    remaining_s2 = args.sample_s23 - len(true_s2_in_file)
    remaining_s3 = args.sample_s23 - len(true_s3_in_file)
    
    if remaining_s2 > 0:
        other_s2 = list(available_s2 - true_s2_in_file)
        rng = np.random.default_rng(args.seed)
        extra_s2 = set(rng.choice(other_s2, size=min(remaining_s2, len(other_s2)), replace=False))
    else:
        extra_s2 = set()
    
    if remaining_s3 > 0:
        other_s3 = list(available_s3 - true_s3_in_file)
        rng = np.random.default_rng(args.seed + 1)
        extra_s3 = set(rng.choice(other_s3, size=min(remaining_s3, len(other_s3)), replace=False))
    else:
        extra_s3 = set()
    
    # Combine
    sampled_s2_ids = true_s2_in_file | extra_s2
    sampled_s3_ids = true_s3_in_file | extra_s3
    
    s2 = s2_full[s2_full["entity_id"].isin(sampled_s2_ids)].reset_index(drop=True)
    s3 = s3_full[s3_full["entity_id"].isin(sampled_s3_ids)].reset_index(drop=True)
    print(f"[train] sampled S2={len(s2)} (incl {len(true_s2_in_file)} true matches), S3={len(s3)} (incl {len(true_s3_in_file)} true matches)")
    
    print("[train] running blocking on sampled sources...")
    candidates = generate_candidates(s1, s2, s3)
    print(f"[train] blocking produced {len(candidates)} candidate pairs")

    ceiling = blocking_recall_ceiling(candidates, ground_truth)
    print(f"[train] BLOCKING RECALL CEILING = {ceiling:.4f}")

    print("[train] building pairwise features...")
    feat = build_features(candidates, s1, s2, s3)
    feat["label"] = [
        1 if cid in ground_truth.get(sid, set()) else 0
        for sid, cid in zip(feat["source1_entity_id"], feat["candidate_entity_id"])
    ]
    print(f"[train] labeled pairs: {len(feat)} total, {feat['label'].sum()} positive")

    s1_ids = s1["entity_id"].tolist()
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    train_idx, val_idx = next(gss.split(s1_ids, groups=s1_ids))
    train_s1_ids = {s1_ids[i] for i in train_idx}
    val_s1_ids = [s1_ids[i] for i in val_idx]

    train_feat = feat[feat["source1_entity_id"].isin(train_s1_ids)]
    val_feat = feat[feat["source1_entity_id"].isin(val_s1_ids)]
    print(f"[train] split: {len(train_s1_ids)} train S1 entities / {len(val_s1_ids)} val S1 entities")

    X_train = train_feat[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_feat["label"].to_numpy(dtype=int)
    clf = train_model(X_train, y_train, random_state=args.seed)

    X_val = val_feat[FEATURE_COLUMNS].to_numpy(dtype=float)
    val_probs = clf.predict_proba(X_val)[:, 1] if len(X_val) else np.array([])

    val_truth = {sid: ground_truth.get(sid, set()) for sid in val_s1_ids}
    best_thr, best_score = sweep_threshold(val_feat, val_probs, val_truth, val_s1_ids)
    print(f"[train] BEST VALIDATION macro F_0.5 = {best_score:.4f} at threshold={best_thr:.2f}")

    empty_pred = {sid: set() for sid in val_s1_ids}
    baseline = macro_f_half(empty_pred, val_truth)
    print(f"[train] (baseline: predicting all-empty scores {baseline:.4f} macro F_0.5)")

    print("[train] retraining on full sampled training data with chosen threshold...")
    X_all = feat[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_all = feat["label"].to_numpy(dtype=int)
    final_clf = train_model(X_all, y_all, random_state=args.seed)

    save_model(final_clf, best_thr, args.model_out)
    print(f"[train] saved model to {args.model_out} (+ .meta.json with threshold={best_thr:.2f})")


if __name__ == "__main__":
    main()