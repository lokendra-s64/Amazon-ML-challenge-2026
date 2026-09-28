"""
Pairwise feature engineering.

Takes the long-format candidate pairs from blocking.py plus the three
source tables and produces one feature row per (source1_entity_id,
candidate_entity_id) pair, ready for model.py to score.
"""
from __future__ import annotations

import pandas as pd

from normalize import (
    address_tokens,
    char_ngrams,
    extract_postal_code,
    name_tokens,
    normalize_address,
    normalize_name,
    strip_landmark,
)
from similarity import jaccard, levenshtein_ratio, prefix_match, token_sort_ratio

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


def _prep_lookup(df):
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


def build_features(candidates, s1_df, s2_df, s3_df):
    """
    candidates: long-format DataFrame with source1_entity_id,
    candidate_entity_id, candidate_source, block_score (from blocking.py).

    Returns candidates with FEATURE_COLUMNS appended.
    """
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
