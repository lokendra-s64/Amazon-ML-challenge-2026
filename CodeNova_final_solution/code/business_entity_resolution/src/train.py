#!/usr/bin/env python3
"""
Train the entity-resolution matching model on the challenge's training data.

Usage (from anywhere):
    python3 train.py --train-dir /path/to/dataset/train --model-out ../models/model.pkl

What it does:
    1. Loads train_source{1,2,3}.tsv and train_ground_truth.tsv
    2. Runs blocking on the TRAIN sources and reports the blocking recall
       ceiling (what fraction of true matches even survive blocking --
       this upper-bounds everything downstream)
    3. Builds pairwise features for every (S1, candidate) pair
    4. Splits Source-1 entities (not raw pairs, to avoid leakage) into a
       train/validation group split
    5. Trains a HistGradientBoostingClassifier
    6. Sweeps the decision threshold on the validation split to maximize
       macro F_0.5 (the actual competition metric), not accuracy/AUC
    7. Retrains on all training data at the chosen threshold and saves
       the model + threshold to disk
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


def blocking_recall_ceiling(candidates, ground_truth):
    cand_by_s1 = {}
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


def sweep_threshold(val_feat, val_probs, val_truth, val_s1_ids):
    best_thr, best_score = 0.5, -1.0
    thresholds = np.arange(0.05, 0.96, 0.01)
    df = val_feat[["source1_entity_id", "candidate_entity_id"]].copy()
    df["prob"] = val_probs

    for thr in thresholds:
        kept = df[df["prob"] >= thr]
        pred_by_s1 = {
            sid: set(g) for sid, g in kept.groupby("source1_entity_id")["candidate_entity_id"]
        }
        truth_subset = {sid: val_truth.get(sid, set()) for sid in val_s1_ids}
        score = macro_f_half(pred_by_s1, truth_subset)
        if score > best_score:
            best_score, best_thr = score, float(thr)
    return best_thr, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="../../dataset/train")
    ap.add_argument("--model-out", default="../models/model.pkl")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-s1", type=int, default=0, help="Randomly sample N S1 entities for training (0=use all)")
    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    print(f"[train] loading data from {train_dir}")
    s1 = read_source(train_dir / "train_source1.tsv")
    s2 = read_source(train_dir / "train_source2.tsv")
    s3 = read_source(train_dir / "train_source3.tsv")
    ground_truth = read_ground_truth(train_dir / "train_ground_truth.tsv")
    print(f"[train] S1={len(s1)} S2={len(s2)} S3={len(s3)} ground-truth entities={len(ground_truth)}")

    # Optionally sample S1 entities for faster training
    if args.sample_s1 and args.sample_s1 < len(s1):
        rng = np.random.default_rng(args.seed)
        sampled_idx = rng.choice(len(s1), size=args.sample_s1, replace=False)
        s1 = s1.iloc[sampled_idx].reset_index(drop=True)
        # Filter ground truth to only sampled S1 entities
        s1_ids_set = set(s1["entity_id"])
        ground_truth = {k: v for k, v in ground_truth.items() if k in s1_ids_set}
        print(f"[train] sampled {len(s1)} S1 entities for training")

    print("[train] running blocking on train sources...")
    candidates = generate_candidates(s1, s2, s3)
    print(f"[train] blocking produced {len(candidates)} candidate pairs")

    ceiling = blocking_recall_ceiling(candidates, ground_truth)
    print(f"[train] BLOCKING RECALL CEILING = {ceiling:.4f}  "
          f"(fraction of true non-singleton matches that survive blocking; "
          f"raise MAX_CANDIDATES_PER_SOURCE / add blocking keys in blocking.py if this is low)")

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

    # Baseline sanity check: what if we predicted every entity as a singleton?
    empty_pred = {sid: set() for sid in val_s1_ids}
    baseline = macro_f_half(empty_pred, val_truth)
    print(f"[train] (baseline: predicting all-empty scores {baseline:.4f} macro F_0.5 -- your model should beat this)")

    print("[train] retraining on full training data with chosen threshold...")
    X_all = feat[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_all = feat["label"].to_numpy(dtype=int)
    final_clf = train_model(X_all, y_all, random_state=args.seed)

    save_model(final_clf, best_thr, args.model_out)
    print(f"[train] saved model to {args.model_out} (+ .meta.json with threshold={best_thr:.2f})")


if __name__ == "__main__":
    main()
