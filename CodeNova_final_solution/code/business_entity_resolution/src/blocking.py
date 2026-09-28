"""
Blocking / candidate generation.

Goal: for every Source-1 entity, cheaply produce a short list of Source-2 /
Source-3 records that *might* match, with recall as high as we can afford,
before the (expensive, precise) feature+model stage narrows it down. Blocking
sets the recall ceiling for the whole pipeline, so we deliberately use several
independent, cheap signals and union them:

  1. Shared normalized name tokens (inverted index), excluding tokens that
     are too common within a source (they'd create useless giant blocks).
  2. Shared normalized address tokens (inverted index), same df-capping.
  3. Exact postal/ZIP code match, when present on both sides -- a strong,
     source/country-agnostic signal.
  4. A same-first-3-letters-of-first-name-token index, to catch cases
     where the only shared token got typo'd (so signal (1) misses it).

We never hard-filter on `country` (per the challenge rules -- country is
an open set of labels, unseen labels like "France" must still work) but
`country` IS used later as a plain feature for the matching model.

Output: a long-format DataFrame with one row per (source1_entity_id,
candidate_entity_id) pair and a cheap `block_score` used only to rank /
cap candidates when a block is too large -- it is NOT the final match
score, that comes from the trained model in model.py.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from normalize import (
    address_tokens,
    extract_postal_code,
    name_tokens,
    normalize_name,
)

# Remove frequency cap for recall - keep all tokens (filtering only extremely common ones)
MAX_TOKEN_DOC_FREQ_RATIO = 0.95  # only skip tokens seen in >95% of a source
MAX_CANDIDATES_PER_SOURCE = 20   # per S1 entity, per S2/S3 source, after ranking


def _build_token_index(df, id_col, tokens_col):
    """token -> set(entity_id), dropping only extremely frequent tokens."""
    raw_index = defaultdict(set)
    for eid, toks in zip(df[id_col], df[tokens_col]):
        for t in set(toks):
            raw_index[t].add(eid)

    n = max(len(df), 1)
    cap = max(int(n * MAX_TOKEN_DOC_FREQ_RATIO), 5)
    return {tok: ids for tok, ids in raw_index.items() if len(ids) <= cap}


def _prep(df):
    out = df.copy()
    out["name_norm"] = out["business_name"].map(normalize_name)
    out["name_toks"] = out["business_name"].map(name_tokens)
    out["addr_toks"] = out["business_address"].map(address_tokens)
    out["postal"] = out["business_address"].map(extract_postal_code)
    out["name_prefix3"] = out["name_norm"].str.replace(" ", "").str[:3]
    return out


def _candidates_for_row(row, name_idx, addr_idx, postal_idx, prefix_idx):
    """Return {candidate_id: cheap_score} from one source's indices."""
    scores = defaultdict(int)

    for t in set(row.name_toks):
        for cid in name_idx.get(t, ()):
            scores[cid] += 3  # name-token hits weigh more than address ones

    for t in set(row.addr_toks):
        for cid in addr_idx.get(t, ()):
            scores[cid] += 1

    if row.postal:
        for cid in postal_idx.get(row.postal, ()):
            scores[cid] += 5  # exact postal code match is a strong signal

    if row.name_prefix3:
        for cid in prefix_idx.get(row.name_prefix3, ()):
            scores[cid] += 1

    # Fallback: a record with only very common tokens (all filtered out of
    # the indices) would otherwise get zero candidates. Give it at least
    # the union with any exact-postal or prefix hits already computed
    # above; if still empty there is nothing cheap left to try, and the
    # (rare) miss is an accepted recall/speed trade-off documented in the
    # methodology write-up.
    return scores


def _index_bundle(df):
    return (
        _build_token_index(df, "entity_id", "name_toks"),
        _build_token_index(df, "entity_id", "addr_toks"),
        _group_index(df, "postal"),
        _group_index(df, "name_prefix3"),
    )


def _group_index(df, col):
    idx = defaultdict(set)
    for eid, val in zip(df["entity_id"], df[col]):
        if val:
            idx[val].add(eid)
    return idx


def generate_candidates(s1_df, s2_df, s3_df, max_candidates_per_source=MAX_CANDIDATES_PER_SOURCE):
    """
    Returns a long DataFrame: source1_entity_id, candidate_entity_id,
    candidate_source ('S2' or 'S3'), block_score.
    """
    s1 = _prep(s1_df)
    s2 = _prep(s2_df)
    s3 = _prep(s3_df)

    s2_idx = _index_bundle(s2)
    s3_idx = _index_bundle(s3)

    rows = []
    for row in s1.itertuples(index=False):
        for source_label, source_df_idx in (("S2", s2_idx), ("S3", s3_idx)):
            scores = _candidates_for_row(row, *source_df_idx)
            if not scores:
                continue
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            ranked = ranked[:max_candidates_per_source]
            for cid, sc in ranked:
                rows.append(
                    {
                        "source1_entity_id": row.entity_id,
                        "candidate_entity_id": cid,
                        "candidate_source": source_label,
                        "block_score": sc,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "block_score"]
        )
    return pd.DataFrame(rows)


def candidates_to_wide(candidates, s1_ids):
    """Collapse long-format candidates into the candidate_pairs.tsv shape:
    one row per Source-1 entity, comma-joined candidate_entity_ids."""
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
