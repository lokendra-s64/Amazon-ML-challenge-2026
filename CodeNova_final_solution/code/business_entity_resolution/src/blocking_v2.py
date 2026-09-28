"""
Optimized TF-IDF + Token-based Blocking for Entity Resolution.

Combines:
- TF-IDF cosine similarity (vectorized, fast for millions of records)
- Shared token inverted index (high recall for exact token matches)
- Postal code exact match (strong signal)
- First-3-char prefix index (catches typos)

Uses the improved normalization from normalize.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from normalize import (
    normalize_name,
    normalize_address,
    extract_postal_code,
    name_tokens,
    address_tokens,
)

# Configuration
TOP_K_TFIDF = 50          # TF-IDF candidates per S1 entity per source
TOP_K_TOKEN = 200          # Token-based candidates per S1 entity per source
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_DF = 0.90
TFIDF_MIN_DF = 2
BATCH_SIZE = 5000


def make_blocking_text(name, addr):
    """Combine normalized name + address for TF-IDF blocking."""
    norm_name = normalize_name(name)
    norm_addr = normalize_address(addr)
    parts = [p for p in [norm_name, norm_addr] if p]
    return " ".join(parts)


def build_tfidf_index(texts):
    """Build TF-IDF vectorizer and sparse matrix."""
    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def get_top_tfidf_candidates(query_texts, vectorizer, cand_matrix, top_k=TOP_K_TFIDF, batch_size=BATCH_SIZE):
    """Get top-k TF-IDF candidates for each query using batched cosine similarity."""
    query_matrix = vectorizer.transform(query_texts)
    n_queries = query_matrix.shape[0]
    results = []

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch = query_matrix[start:end]
        sims = linear_kernel(batch, cand_matrix)  # (batch_size, n_cands) - dense array
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


def _build_token_index(df, tokens_col):
    """token -> set(entity_id), keeping all tokens (no frequency cap for recall)."""
    from collections import defaultdict
    raw_index = defaultdict(set)
    for eid, toks in zip(df["entity_id"], df[tokens_col]):
        for t in set(toks):
            raw_index[t].add(eid)
    return raw_index


def _build_prefix_index(df):
    """First 3 chars of first name token -> set(entity_id)."""
    from collections import defaultdict
    idx = defaultdict(set)
    for eid, name_norm in zip(df["entity_id"], df["name_norm"]):
        if name_norm:
            first_tok = name_norm.split(" ")[0]
            if len(first_tok) >= 3:
                prefix = first_tok[:3]
                idx[prefix].add(eid)
    return idx


def _build_postal_index(df):
    """Postal code -> set(entity_id)."""
    from collections import defaultdict
    idx = defaultdict(set)
    for eid, postal in zip(df["entity_id"], df["postal"]):
        if postal:
            idx[postal].add(eid)
    return idx


def _prep_df(df):
    """Prepare DataFrame with normalized fields for blocking - optimized with list comprehensions."""
    # Use list comprehensions instead of .map() for speed
    business_names = df["business_name"].astype(str).tolist()
    business_addrs = df["business_address"].astype(str).tolist()

    name_norm = [normalize_name(n) for n in business_names]
    name_toks = [name_tokens(n) for n in business_names]
    addr_toks = [address_tokens(a) for a in business_addrs]
    postal = [extract_postal_code(a) for a in business_addrs]
    blocking_text = [make_blocking_text(n, a) for n, a in zip(name_norm, business_addrs)]

    out = df.copy()
    out["name_norm"] = name_norm
    out["name_toks"] = name_toks
    out["addr_toks"] = addr_toks
    out["postal"] = postal
    out["blocking_text"] = blocking_text
    return out


def _candidates_from_indices(row, name_idx, addr_idx, prefix_idx, postal_idx, max_per_source):
    """Get candidates from all indices for one S1 row."""
    from collections import defaultdict
    scores = defaultdict(int)

    # Name tokens (weight 3)
    for t in set(row.name_toks):
        for cid in name_idx.get(t, ()):
            scores[cid] += 3

    # Address tokens (weight 1)
    for t in set(row.addr_toks):
        for cid in addr_idx.get(t, ()):
            scores[cid] += 1

    # Postal code exact match (weight 10 - very strong signal)
    if row.postal:
        for cid in postal_idx.get(row.postal, ()):
            scores[cid] += 10

    # Name prefix (weight 1)
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


def generate_candidates(s1_df, s2_df, s3_df, top_k_tfidf=TOP_K_TFIDF, top_k_token=TOP_K_TOKEN):
    """
    Generate candidates using hybrid TF-IDF + token-based blocking.

    Returns long-format DataFrame: source1_entity_id, candidate_entity_id, candidate_source, block_score.
    """
    # Prepare all dataframes
    s1 = _prep_df(s1_df)
    s2 = _prep_df(s2_df)
    s3 = _prep_df(s3_df)

    # Combine S2+S3 for TF-IDF index (per country would be better but we avoid hard country filtering)
    s23 = pd.concat([s2, s3], ignore_index=True)
    s23_ids = s23["entity_id"].values
    s23_sources = s23["entity_id"].str[:2].values  # 'S2' or 'S3'

    print(f"  Building TF-IDF index on {len(s23)} S2+S3 records...")
    tfidf_vec, tfidf_mat = build_tfidf_index(s23["blocking_text"].tolist())
    print(f"  TF-IDF matrix shape: {tfidf_mat.shape}")

    # Build token/prefix/postal indices for S2 and S3 separately
    print("  Building token/prefix/postal indices...")
    s2_name_idx = _build_token_index(s2, "name_toks")
    s2_addr_idx = _build_token_index(s2, "addr_toks")
    s2_prefix_idx = _build_prefix_index(s2)
    s2_postal_idx = _build_postal_index(s2)

    s3_name_idx = _build_token_index(s3, "name_toks")
    s3_addr_idx = _build_token_index(s3, "addr_toks")
    s3_prefix_idx = _build_prefix_index(s3)
    s3_postal_idx = _build_postal_index(s3)

    # Get TF-IDF candidates for all S1 entities at once
    print(f"  Getting TF-IDF top-{top_k_tfidf} candidates...")
    tfidf_results = get_top_tfidf_candidates(s1["blocking_text"].tolist(), tfidf_vec, tfidf_mat, top_k_tfidf)

    # Combine TF-IDF + token-based candidates
    rows = []
    for s1_idx, row in enumerate(s1.itertuples(index=False)):
        s1_id = row.entity_id
        tfidf_cands = tfidf_results[s1_idx]

        # TF-IDF candidates (weight by score * 2)
        for cand_idx, score in tfidf_cands:
            cid = s23_ids[cand_idx]
            source = s23_sources[cand_idx]
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cid,
                "candidate_source": source,
                "block_score": int(score * 100) + 20,  # TF-IDF score scaled
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

    # Deduplicate and keep highest score per (s1, candidate)
    cand_df = pd.DataFrame(rows)
    cand_df = cand_df.sort_values("block_score", ascending=False)
    cand_df = cand_df.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"], keep="first")

    return cand_df


def candidates_to_wide(candidates, s1_ids):
    """Collapse long-format candidates into candidate_pairs.tsv format."""
    grouped = (
        candidates.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: ",".join(dict.fromkeys(ids)))  # de-dup, keep order
        .to_dict()
    )
    return pd.DataFrame(
        {
            "source1_entity_id": s1_ids,
            "candidate_entity_ids": [grouped.get(sid, "") for sid in s1_ids],
        }
    )
