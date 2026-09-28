"""
Business Entity Resolution Pipeline — Optimized for Amazon ML Challenge 2026

This pipeline combines:
1. Hybrid blocking: TF-IDF cosine similarity + token-based inverted index + postal code exact match
2. Rich pairwise features: Levenshtein, Jaccard, token-sort, char n-grams, postal match, country match
3. HistGradientBoostingClassifier with threshold tuned for macro F_0.5
4. Proper handling of unseen countries (France in test set)

No external data/APIs used — fully compliant with fair-play rules.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import read_source, read_ground_truth, write_matching_results, write_candidate_pairs
from normalize import (
    normalize_name,
    normalize_address,
    name_tokens,
    address_tokens,
    extract_postal_code,
    strip_landmark,
    char_ngrams,
)
from similarity import jaccard, levenshtein_ratio, token_sort_ratio, prefix_match
import joblib
import json

# Feature columns
FEATURE_COLUMNS = [
    "block_score",
    "name_exact_norm",
    "name_lev_ratio",
    "name_token_jaccard",
    "name_token_sort_ratio",
    "name_char3gram_jaccard",
    "name_first_token_match",
    "addr_lev_ratio",
    "addr_token_jaccard",
    "addr_char3gram_jaccard",
    "postal_match",
    "postal_both_present",
    "same_country",
    "name_len_diff",
    "addr_len_diff",
]

# Blocking config
TOP_K_TFIDF = 20
TOP_K_TOKEN = 100
TFIDF_NGRAM_RANGE = (1, 1)
TFIDF_MAX_DF = 0.90
TFIDF_MIN_DF = 2
BATCH_SIZE = 5000


def make_blocking_text(name: str, addr: str) -> str:
    """Combine normalized name + address for TF-IDF blocking."""
    norm_name = normalize_name(name)
    norm_addr = normalize_address(addr)
    parts = [p for p in [norm_name, norm_addr] if p]
    return " ".join(parts)


def build_tfidf_index(texts: list[str]):
    """Build TF-IDF vectorizer and sparse matrix."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def get_top_tfidf_candidates(query_texts: list[str], vectorizer, cand_matrix, top_k: int = TOP_K_TFIDF, batch_size: int = BATCH_SIZE):
    """Get top-k TF-IDF candidates for each query using batched cosine similarity."""
    from sklearn.metrics.pairwise import linear_kernel
    query_matrix = vectorizer.transform(query_texts)
    n_queries = query_matrix.shape[0]
    results = []

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch = query_matrix[start:end]
        sims = linear_kernel(batch, cand_matrix)
        for row in sims:
            if row.sum() == 0:
                results.append([])
                continue
            top_indices = np.argpartition(row, -min(top_k, len(row)))[-top_k:]
            top_indices = top_indices[np.argsort(row[top_indices])[::-1]]
            top_scores = row[top_indices]
            valid = [(int(i), float(s)) for i, s in zip(top_indices, top_scores) if s > 0.0]
            results.append(valid)
    return results


def _build_token_index(df: pd.DataFrame, tokens_col: str) -> dict[str, set[str]]:
    """token -> set(entity_id)."""
    from collections import defaultdict
    raw_index: dict[str, set[str]] = defaultdict(set)
    for eid, toks in zip(df["entity_id"], df[tokens_col]):
        for t in set(toks):
            raw_index[t].add(eid)
    return raw_index


def _build_prefix_index(df: pd.DataFrame) -> dict[str, set[str]]:
    """First 3 chars of first name token -> set(entity_id)."""
    from collections import defaultdict
    idx: dict[str, set[str]] = defaultdict(set)
    for eid, name_norm in zip(df["entity_id"], df["name_norm"]):
        if name_norm:
            first_tok = name_norm.split(" ")[0]
            if len(first_tok) >= 3:
                prefix = first_tok[:3]
                idx[prefix].add(eid)
    return idx


def _build_postal_index(df: pd.DataFrame) -> dict[str, set[str]]:
    """Postal code -> set(entity_id)."""
    from collections import defaultdict
    idx: dict[str, set[str]] = defaultdict(set)
    for eid, postal in zip(df["entity_id"], df["postal"]):
        if postal:
            idx[postal].add(eid)
    return idx


def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame with normalized fields for blocking - memory-efficient batched processing."""
    n = len(df)
    batch_size = 50000  # Process in batches to avoid OOM
    
    entity_ids = df["entity_id"].values
    business_names = df["business_name"].astype(str).tolist()
    business_addrs = df["business_address"].astype(str).tolist()
    
    # Pre-allocate lists
    name_norm = [None] * n
    name_toks = [None] * n
    addr_toks = [None] * n
    postal = [None] * n
    blocking_text = [None] * n
    
    # Process in batches
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_names = business_names[start:end]
        batch_addrs = business_addrs[start:end]
        
        # Normalize names
        nn_batch = [normalize_name(nm) for nm in batch_names]
        nt_batch = [name_tokens(nm) for nm in batch_names]
        at_batch = [address_tokens(ad) for ad in batch_addrs]
        pc_batch = [extract_postal_code(ad) for ad in batch_addrs]
        bt_batch = [make_blocking_text(nn, ba) for nn, ba in zip(nn_batch, batch_addrs)]
        
        name_norm[start:end] = nn_batch
        name_toks[start:end] = nt_batch
        addr_toks[start:end] = at_batch
        postal[start:end] = pc_batch
        blocking_text[start:end] = bt_batch
    
    out = pd.DataFrame({
        "entity_id": entity_ids,
        "name_norm": name_norm,
        "name_toks": name_toks,
        "addr_toks": addr_toks,
        "postal": postal,
        "blocking_text": blocking_text,
    })
    return out


def _candidates_from_indices(row, name_idx, addr_idx, prefix_idx, postal_idx, max_per_source: int) -> list[tuple[str, int]]:
    """Get candidates from all indices for one S1 row."""
    from collections import defaultdict
    scores: dict[str, int] = defaultdict(int)

    for t in set(row.name_toks):
        for cid in name_idx.get(t, ()):
            scores[cid] += 3

    for t in set(row.addr_toks):
        for cid in addr_idx.get(t, ()):
            scores[cid] += 1

    if row.postal:
        for cid in postal_idx.get(row.postal, ()):
            scores[cid] += 10

    if row.name_norm:
        first_tok = row.name_norm.split(" ")[0]
        if len(first_tok) >= 3:
            prefix = first_tok[:3]
            for cid in prefix_idx.get(prefix, ()):
                scores[cid] += 1

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:max_per_source]


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    top_k_tfidf: int = TOP_K_TFIDF,
    top_k_token: int = TOP_K_TOKEN,
) -> pd.DataFrame:
    """
    Generate candidates using hybrid TF-IDF + token-based blocking.
    Returns long-format DataFrame: source1_entity_id, candidate_entity_id, candidate_source, block_score.
    """
    s1 = _prep_df(s1_df)
    s2 = _prep_df(s2_df)
    s3 = _prep_df(s3_df)

    # Combine S2+S3 for TF-IDF index
    s23 = pd.concat([s2, s3], ignore_index=True)
    s23_ids = s23["entity_id"].values
    s23_sources = s23["entity_id"].str[:2].values

    print(f"  Building TF-IDF index on {len(s23)} S2+S3 records...")
    tfidf_vec, tfidf_mat = build_tfidf_index(s23["blocking_text"].tolist())
    print(f"  TF-IDF matrix shape: {tfidf_mat.shape}")

    print("  Building token/prefix/postal indices...")
    s2_name_idx = _build_token_index(s2, "name_toks")
    s2_addr_idx = _build_token_index(s2, "addr_toks")
    s2_prefix_idx = _build_prefix_index(s2)
    s2_postal_idx = _build_postal_index(s2)

    s3_name_idx = _build_token_index(s3, "name_toks")
    s3_addr_idx = _build_token_index(s3, "addr_toks")
    s3_prefix_idx = _build_prefix_index(s3)
    s3_postal_idx = _build_postal_index(s3)

    print(f"  Getting TF-IDF top-{top_k_tfidf} candidates...")
    tfidf_results = get_top_tfidf_candidates(s1["blocking_text"].tolist(), tfidf_vec, tfidf_mat, top_k_tfidf)

    rows = []
    for s1_idx, row in enumerate(s1.itertuples(index=False)):
        s1_id = row.entity_id
        tfidf_cands = tfidf_results[s1_idx]

        # TF-IDF candidates
        for cand_idx, score in tfidf_cands:
            cid = s23_ids[cand_idx]
            source = s23_sources[cand_idx]
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cid,
                "candidate_source": source,
                "block_score": int(score * 100) + 20,
            })

        # Token-based candidates for S2
        token_cands_s2 = _candidates_from_indices(row, s2_name_idx, s2_addr_idx, s2_prefix_idx, s2_postal_idx, top_k_token)
        for cid, score in token_cands_s2:
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cid,
                "candidate_source": "S2",
                "block_score": score * 10,
            })

        # Token-based candidates for S3
        token_cands_s3 = _candidates_from_indices(row, s3_name_idx, s3_addr_idx, s3_prefix_idx, s3_postal_idx, top_k_token)
        for cid, score in token_cands_s3:
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cid,
                "candidate_source": "S3",
                "block_score": score * 10,
            })

    if not rows:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "block_score"]
        )

    cand_df = pd.DataFrame(rows)
    cand_df = cand_df.sort_values("block_score", ascending=False)
    cand_df = cand_df.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"], keep="first")

    return cand_df


def candidates_to_wide(candidates: pd.DataFrame, s1_ids: list[str]) -> pd.DataFrame:
    """Collapse long-format candidates into candidate_pairs.tsv format."""
    grouped = (
        candidates.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: ",".join(dict.fromkeys(ids)))
        .to_dict()
    )
    return pd.DataFrame(
        {
            "source1_entity_id": s1_ids,
            "candidate_entity_ids": [grouped.get(sid, "") for sid in s1_ids],
        }
    )


def _prep_lookup(df: pd.DataFrame) -> dict[str, dict]:
    """entity_id -> dict of precomputed normalized fields, for O(1) lookup."""
    out = {}
    for row in df.itertuples(index=False):
        name_norm = normalize_name(row.business_name)
        addr_norm = strip_landmark(normalize_address(row.business_address))
        out[row.entity_id] = {
            "name_norm": name_norm,
            "name_toks": name_tokens(row.business_name),
            "name_3g": char_ngrams(name_norm, 3),
            "addr_norm": addr_norm,
            "addr_toks": address_tokens(row.business_address),
            "addr_3g": char_ngrams(addr_norm, 3),
            "postal": extract_postal_code(row.business_address),
            "country": (row.country or "").strip().lower() if pd.notna(row.country) else "",
        }
    return out


def build_features(
    candidates: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build pairwise features for candidate pairs."""
    s1_lookup = _prep_lookup(s1_df)
    s2_lookup = _prep_lookup(s2_df)
    s3_lookup = _prep_lookup(s3_df)

    feats = {col: [] for col in FEATURE_COLUMNS}

    for row in candidates.itertuples(index=False):
        a = s1_lookup[row.source1_entity_id]
        cand_lookup = s2_lookup if row.candidate_source == "S2" else s3_lookup
        b = cand_lookup[row.candidate_entity_id]

        feats["block_score"].append(row.block_score)
        feats["name_exact_norm"].append(1.0 if a["name_norm"] == b["name_norm"] and a["name_norm"] else 0.0)
        feats["name_lev_ratio"].append(levenshtein_ratio(a["name_norm"], b["name_norm"]))
        feats["name_token_jaccard"].append(jaccard(set(a["name_toks"]), set(b["name_toks"])))
        feats["name_token_sort_ratio"].append(token_sort_ratio(a["name_toks"], b["name_toks"]))
        feats["name_char3gram_jaccard"].append(jaccard(a["name_3g"], b["name_3g"]))
        feats["name_first_token_match"].append(
            prefix_match(
                a["name_toks"][0] if a["name_toks"] else "",
                b["name_toks"][0] if b["name_toks"] else "",
                k=99,
            )
        )
        feats["addr_lev_ratio"].append(levenshtein_ratio(a["addr_norm"], b["addr_norm"]))
        feats["addr_token_jaccard"].append(jaccard(set(a["addr_toks"]), set(b["addr_toks"])))
        feats["addr_char3gram_jaccard"].append(jaccard(a["addr_3g"], b["addr_3g"]))
        both_postal = bool(a["postal"] and b["postal"])
        feats["postal_match"].append(1.0 if both_postal and a["postal"] == b["postal"] else 0.0)
        feats["postal_both_present"].append(1.0 if both_postal else 0.0)
        feats["same_country"].append(1.0 if a["country"] and a["country"] == b["country"] else 0.0)
        feats["name_len_diff"].append(abs(len(a["name_norm"]) - len(b["name_norm"])))
        feats["addr_len_diff"].append(abs(len(a["addr_norm"]) - len(b["addr_norm"])))

    result = candidates.reset_index(drop=True).copy()
    for col in FEATURE_COLUMNS:
        result[col] = feats[col]
    return result


def train_model(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        max_iter=300,
        learning_rate=0.06,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(X, y)
    return clf


def score_pairs(clf: HistGradientBoostingClassifier, features_df) -> np.ndarray:
    X = features_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    return clf.predict_proba(X)[:, 1]


def save_model(clf, threshold: float, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)
    with open(str(path) + ".meta.json", "w") as f:
        json.dump({"threshold": threshold, "feature_columns": FEATURE_COLUMNS}, f, indent=2)


def load_model(path: str):
    clf = joblib.load(path)
    with open(str(path) + ".meta.json") as f:
        meta = json.load(f)
    return clf, meta["threshold"]


def f_half_score(precision: float, recall: float) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    beta2 = 0.25
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def entity_f_half(predicted: set[str], truth: set[str]) -> float:
    if not truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth)
    return f_half_score(precision, recall)


def macro_f_half(
    predicted_by_entity: dict[str, set[str]],
    truth_by_entity: dict[str, set[str]],
) -> float:
    all_ids = set(predicted_by_entity) | set(truth_by_entity)
    if not all_ids:
        return 0.0
    total = 0.0
    for eid in all_ids:
        pred = predicted_by_entity.get(eid, set())
        truth = truth_by_entity.get(eid, set())
        total += entity_f_half(pred, truth)
    return total / len(all_ids)


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


def run_train(train_dir: str, model_out: str, val_frac: float = 0.2, seed: int = 42):
    train_dir = Path(train_dir)
    print(f"[train] loading data from {train_dir}")
    s1 = read_source(train_dir / "train_source1.tsv")
    s2 = read_source(train_dir / "train_source2.tsv")
    s3 = read_source(train_dir / "train_source3.tsv")
    ground_truth = read_ground_truth(train_dir / "train_ground_truth.tsv")
    print(f"[train] S1={len(s1)} S2={len(s2)} S3={len(s3)} ground-truth entities={len(ground_truth)}")

    print("[train] running blocking on train sources...")
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
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(gss.split(s1_ids, groups=s1_ids))
    train_s1_ids = {s1_ids[i] for i in train_idx}
    val_s1_ids = [s1_ids[i] for i in val_idx]

    train_feat = feat[feat["source1_entity_id"].isin(train_s1_ids)]
    val_feat = feat[feat["source1_entity_id"].isin(val_s1_ids)]
    print(f"[train] split: {len(train_s1_ids)} train S1 entities / {len(val_s1_ids)} val S1 entities")

    X_train = train_feat[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_feat["label"].to_numpy(dtype=int)
    clf = train_model(X_train, y_train, random_state=seed)

    X_val = val_feat[FEATURE_COLUMNS].to_numpy(dtype=float)
    val_probs = clf.predict_proba(X_val)[:, 1] if len(X_val) else np.array([])

    val_truth = {sid: ground_truth.get(sid, set()) for sid in val_s1_ids}
    best_thr, best_score = sweep_threshold(val_feat, val_probs, val_truth, val_s1_ids)
    print(f"[train] BEST VALIDATION macro F_0.5 = {best_score:.4f} at threshold={best_thr:.2f}")

    empty_pred = {sid: set() for sid in val_s1_ids}
    baseline = macro_f_half(empty_pred, val_truth)
    print(f"[train] (baseline: predicting all-empty scores {baseline:.4f} macro F_0.5)")

    print("[train] retraining on full training data with chosen threshold...")
    X_all = feat[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_all = feat["label"].to_numpy(dtype=int)
    final_clf = train_model(X_all, y_all, random_state=seed)

    save_model(final_clf, best_thr, model_out)
    print(f"[train] saved model to {model_out} (+ .meta.json with threshold={best_thr:.2f})")


def run_predict(test_dir: str, model_path: str, output_dir: str):
    test_dir = Path(test_dir)
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
    clf, threshold = load_model(model_path)
    feat["prob"] = score_pairs(clf, feat) if len(feat) else []
    print(f"[predict] using decision threshold={threshold:.2f}")

    s1_ids = s1["entity_id"].tolist()

    cand_map: dict[str, list[str]] = {
        sid: list(dict.fromkeys(g)) for sid, g in candidates.groupby("source1_entity_id")["candidate_entity_id"]
    } if len(candidates) else {}

    kept = feat[feat["prob"] >= threshold] if len(feat) else feat
    match_map: dict[str, list[str]] = {
        sid: list(dict.fromkeys(g)) for sid, g in kept.groupby("source1_entity_id")["candidate_entity_id"]
    } if len(kept) else {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_matching_results(output_dir / "matching_results.tsv", match_map, s1_ids)
    write_candidate_pairs(output_dir / "candidate_pairs.tsv", cand_map, s1_ids)
    print(f"[predict] wrote {output_dir / 'matching_results.tsv'} and {output_dir / 'candidate_pairs.tsv'}")

    n_matched_entities = sum(1 for sid in s1_ids if match_map.get(sid))
    print(f"[predict] {n_matched_entities}/{len(s1_ids)} S1 entities received at least one match "
          f"({len(s1_ids) - n_matched_entities} predicted as singletons)")


def main():
    ap = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_train = sub.add_parser("train", help="Train the matching model")
    ap_train.add_argument("--train-dir", default="../../dataset/train")
    ap_train.add_argument("--model-out", default="../models/model.pkl")
    ap_train.add_argument("--val-frac", type=float, default=0.2)
    ap_train.add_argument("--seed", type=int, default=42)

    ap_pred = sub.add_parser("predict", help="Predict on test set")
    ap_pred.add_argument("--test-dir", default="../../dataset/test")
    ap_pred.add_argument("--model", default="../models/model.pkl")
    ap_pred.add_argument("--output-dir", default="../../output")

    args = ap.parse_args()

    if args.cmd == "train":
        run_train(args.train_dir, args.model_out, args.val_frac, args.seed)
    elif args.cmd == "predict":
        run_predict(args.test_dir, args.model, args.output_dir)


if __name__ == "__main__":
    main()
